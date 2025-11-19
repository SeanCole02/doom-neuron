"""
CL1 Neural Interface Server

This program runs on the CL1 device and acts as a neural hardware interface.
It receives stimulation commands via UDP, applies them to the neural hardware,
collects spike responses, and sends them back to the training system.

The CL1 device performs NO computation - it's purely a hardware interface.
All PyTorch models and game logic run on the remote training system.
"""

import argparse
import os
import socket
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

import cl
from cl.data_stream import DataStream
import udp_protocol

# LRU Cache for stimulation designs (copied from ppo_doom.py)
from collections import OrderedDict

class LRUCache(OrderedDict):
    def __init__(self, maxsize=2048):
        super().__init__()
        self.maxsize = maxsize

    def get_or_set(self, key, factory):
        if key in self:
            value = self[key]
            self.move_to_end(key)
            return value
        value = factory()
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)
        return value

    def clear(self):
        super().clear()


class CL1Config:
    """Minimal configuration for CL1 neural interface."""

    def __init__(self):
        # Channel configuration (must match training system)
        self.encoding_channels      = [8, 9, 10, 11, 12, 13, 14, 15]
        self.move_forward_channels  = [16, 17, 18, 19, 20, 21, 22, 23]
        self.move_backward_channels = [24, 25, 26, 27, 28, 29, 30, 31]
        self.move_left_channels     = [32, 33, 34, 35, 36, 37, 38, 39]
        self.move_right_channels    = [40, 41, 42, 43, 44, 45, 46, 47]
        self.turn_left_channels     = [48, 49, 50, 51, 52, 53, 54, 55]
        self.turn_right_channels    = [57, 58, 59, 60, 61, 62]
        self.attack_channels        = [1, 2, 3, 5, 6]

        # Stimulation design parameters
        self.phase1_duration = 120  # μs
        self.phase2_duration = 120  # μs
        self.burst_count = 1

        # All channels except forbidden ones
        self.all_channels = [i for i in range(64) if i not in {0, 4, 7, 56, 63}]

    def create_channel_sets(self):
        """Create CL SDK ChannelSet objects."""
        self.all_channels_set           = cl.ChannelSet(*self.all_channels)
        self.encoding_channels_set      = cl.ChannelSet(*self.encoding_channels)
        self.move_forward_channels_set  = cl.ChannelSet(*self.move_forward_channels)
        self.move_backward_channels_set = cl.ChannelSet(*self.move_backward_channels)
        self.move_left_channels_set     = cl.ChannelSet(*self.move_left_channels)
        self.move_right_channels_set    = cl.ChannelSet(*self.move_right_channels)
        self.turn_left_channels_set     = cl.ChannelSet(*self.turn_left_channels)
        self.turn_right_channels_set    = cl.ChannelSet(*self.turn_right_channels)
        self.attack_channels_set        = cl.ChannelSet(*self.attack_channels)


class CL1NeuralInterface:
    """
    Minimal neural interface that runs on CL1 device.

    Responsibilities:
    - Receive stimulation commands via UDP
    - Apply stimulation to neural hardware
    - Collect spike responses
    - Send spike counts back via UDP
    """

    def __init__(
        self,
        training_host: str,
        stim_port: int,
        spike_port: int,
        event_port: int,
        feedback_port: int,
        tick_frequency_hz: int = 240,
        recording_path: str = "/data/recordings/seandoom",
    ):
        self.training_host = training_host
        self.stim_port = stim_port
        self.spike_port = spike_port
        self.event_port = event_port
        self.feedback_port = feedback_port
        self.tick_frequency_hz = tick_frequency_hz
        self.recording_path = recording_path

        # Create config and channel sets
        self.config = CL1Config()
        self.config.create_channel_sets()

        # Build channel groups for stimulation application
        self.channel_groups: List[Tuple[str, List[int], cl.ChannelSet]] = [
            ('encoding', self.config.encoding_channels, self.config.encoding_channels_set),
            ('move_forward', self.config.move_forward_channels, self.config.move_forward_channels_set),
            ('move_backward', self.config.move_backward_channels, self.config.move_backward_channels_set),
            ('move_left', self.config.move_left_channels, self.config.move_left_channels_set),
            ('move_right', self.config.move_right_channels, self.config.move_right_channels_set),
            ('turn_left', self.config.turn_left_channels, self.config.turn_left_channels_set),
            ('turn_right', self.config.turn_right_channels, self.config.turn_right_channels_set),
            ('attack', self.config.attack_channels, self.config.attack_channels_set),
        ]

        # Build channel lookup for spike counting
        self.channel_lookup: Dict[int, int] = {}
        for idx, (_, channel_list, _) in enumerate(self.channel_groups):
            for ch in channel_list:
                self.channel_lookup[ch] = idx

        # Stimulation design cache
        self._stim_cache = LRUCache(maxsize=2048)

        # Statistics
        self.packets_received = 0
        self.packets_sent = 0
        self.total_spikes = 0
        self.stim_commands_applied = 0
        self.events_received = 0
        self.feedback_commands_received = 0

    def setup_sockets(self):
        """Create UDP sockets for communication."""
        # Socket for receiving stimulation commands from training system
        self.stim_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.stim_socket.bind(("0.0.0.0", self.stim_port))
        self.stim_socket.setblocking(False)  # Non-blocking to prevent loop stalling

        print(f"Listening for stimulation commands on port {self.stim_port}")

        # Socket for sending spike data to training system
        self.spike_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print(f"Will send spike data to {self.training_host}:{self.spike_port}")

        # Socket for receiving event metadata from training system
        self.event_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.event_socket.bind(("0.0.0.0", self.event_port))
        self.event_socket.setblocking(False)  # Non-blocking for event receiving

        print(f"Listening for event metadata on port {self.event_port}")

        # Socket for receiving feedback commands from training system
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.feedback_socket.bind(("0.0.0.0", self.feedback_port))
        self.feedback_socket.setblocking(False)  # Non-blocking for feedback receiving

        print(f"Listening for feedback commands on port {self.feedback_port}")

    def apply_stimulation(
        self,
        neurons: cl.Neurons,
        frequencies: np.ndarray,
        amplitudes: np.ndarray
    ):
        """
        Apply stimulation to neural hardware based on received commands.

        Args:
            neurons: CL SDK neurons interface
            frequencies: (8,) array of Hz values
            amplitudes: (8,) array of μA values
        """
        # Interrupt ongoing stimulation
        neurons.interrupt(self.config.all_channels_set)

        # Apply stimulation for each channel set
        for i, (_, _, channel_set) in enumerate(self.channel_groups):
            if channel_set is None:
                continue

            amplitude_value = float(amplitudes[i])
            freq_value = int(frequencies[i])
            cache_key = (i, freq_value, round(amplitude_value, 4))

            def _factory():
                stim_design = cl.StimDesign(
                    self.config.phase1_duration, -amplitude_value,
                    self.config.phase2_duration, amplitude_value
                )
                burst_design = cl.BurstDesign(self.config.burst_count, freq_value)
                return (stim_design, burst_design)

            stim_design, burst_design = self._stim_cache.get_or_set(cache_key, _factory)

            # Apply stimulation
            neurons.stim(channel_set, stim_design, burst_design)

        self.stim_commands_applied += 1

    def collect_spikes(self, tick) -> np.ndarray:
        """
        Collect and count spikes from CL SDK tick.

        Args:
            tick

        Returns:
            spike_counts: (8,) array of spike counts per channel set
        """
        spike_counts = np.zeros(8, dtype=np.float32)
        for spike in tick.analysis.spikes:
            idx = self.channel_lookup.get(spike.channel)
            if idx is not None:
                spike_counts[idx] += 1
                self.total_spikes += 1

        return spike_counts

    def apply_feedback_command(
        self,
        neurons: cl.Neurons,
        feedback_type: str,
        channels: list,
        frequency: int,
        amplitude: float,
        pulses: int,
        unpredictable: bool,
        event_name: str
    ):
        """
        Apply feedback stimulation command to neural hardware.

        Args:
            neurons: CL SDK neurons interface
            feedback_type: Type of feedback ("interrupt", "event", or "reward")
            channels: List of channel numbers to stimulate
            frequency: Stimulation frequency in Hz
            amplitude: Stimulation amplitude in μA
            pulses: Number of pulses/bursts
            unpredictable: Whether this is unpredictable stimulation
            event_name: Name of the event (for logging)
        """
        if feedback_type == "interrupt":
            # Interrupt ongoing stimulation on specified channels
            if channels:
                channel_set = cl.ChannelSet(*channels)
                neurons.interrupt(channel_set)
            return

        # Apply feedback stimulation
        if not channels or frequency <= 0 or amplitude <= 0:
            return

        channel_set = cl.ChannelSet(*channels)

        # Create stimulation design (same as encoder stimulation)
        cache_key = (feedback_type, tuple(channels), frequency, round(amplitude, 4))

        def _factory():
            stim_design = cl.StimDesign(
                self.config.phase1_duration, -amplitude,
                self.config.phase2_duration, amplitude
            )
            burst_design = cl.BurstDesign(pulses, frequency)
            return (stim_design, burst_design)

        stim_design, burst_design = self._stim_cache.get_or_set(cache_key, _factory)

        # Apply stimulation
        neurons.stim(channel_set, stim_design, burst_design)
        self.feedback_commands_received += 1

    def run(self):
        """Main loop: receive stim commands, apply to hardware, send spikes back."""
        print("\n" + "="*70)
        print("CL1 Neural Interface Server")
        print("="*70)
        print(f"Tick frequency: {self.tick_frequency_hz} Hz")
        print(f"Expected packet rate: {self.tick_frequency_hz} packets/sec")
        print(f"Channel groups: {len(self.channel_groups)}")
        print("="*70 + "\n")

        self.setup_sockets()

        with cl.open() as neurons:
            print("[SUCCESS] Connected to CL1 hardware")

            # Get all used channels for recording
            used_channels = []
            for _, channel_list, _ in self.channel_groups:
                used_channels.extend(channel_list)
            used_channels = sorted(set(used_channels))
            print(f"Using {len(used_channels)} channels: {used_channels[:10]}...")

            # Create recording directory if it doesn't exist
            try:
                os.makedirs(self.recording_path, exist_ok=True)
                print(f"Recording directory: {os.path.abspath(self.recording_path)}")
            except Exception as e:
                print(f"Failed to create recording directory: {e}")
                raise

            # Create data stream for events
            event_datastream = neurons.create_data_stream(
                name="cl1_neural_interface",
                attributes={"used_channels": used_channels}
            )

            # NOTE: (2025-11-19, jz/al) start recording
            # TODO (al): Add labman plugin
            try:
                recording = neurons.record(
                    file_suffix=f"cl1_interface_{self.tick_frequency_hz}_hz",
                    file_location=self.recording_path,
                    attributes={"tick_frequency": self.tick_frequency_hz},
                )
                print(f"Recording started successfully")
            except Exception as e:
                print(f"Failed to start recording: {e}")
                raise

            print("\nWaiting for stimulation commands...\n")

            last_stats_time = time.time()
            tick_count = 0
            recording_stopped = False  # Track if we've already stopped

            try:
                # Main hardware loop
                for tick in neurons.loop(ticks_per_second=self.tick_frequency_hz):
                    tick_count += 1

                    # Try to receive stimulation command (non-blocking)
                    try:
                        packet, addr = self.stim_socket.recvfrom(
                            udp_protocol.STIM_PACKET_SIZE
                        )

                        # Unpack stimulation command
                        timestamp, frequencies, amplitudes = \
                            udp_protocol.unpack_stimulation_command(packet)

                        self.packets_received += 1

                        # Apply stimulation to hardware
                        self.apply_stimulation(neurons, frequencies, amplitudes)

                        # Log latency occasionally
                        if self.packets_received % 1000 == 0:
                            latency = udp_protocol.get_latency_ms(timestamp)
                            print(f"Packet latency: {latency:.2f} ms")

                    except BlockingIOError:
                        # No packet available - this is expected sometimes
                        # Use default stimulation or skip
                        frequencies = np.zeros(8, dtype=np.float32)
                        amplitudes = np.zeros(8, dtype=np.float32)

                    except Exception as e:
                        print(f"Error receiving/applying stimulation: {e}")
                        frequencies = np.zeros(8, dtype=np.float32)
                        amplitudes = np.zeros(8, dtype=np.float32)

                    # Collect spikes from this tick
                    spike_counts = self.collect_spikes(tick)

                    # Send spike data back to training system
                    try:
                        spike_packet = udp_protocol.pack_spike_data(spike_counts)
                        self.spike_socket.sendto(
                            spike_packet,
                            (self.training_host, self.spike_port)
                        )
                        self.packets_sent += 1

                    except Exception as e:
                        print(f"Error sending spike data: {e}")

                    # Check for event metadata (non-blocking)
                    try:
                        event_packet, _ = self.event_socket.recvfrom(4096)  # Larger buffer for JSON
                        timestamp, event_type, data = udp_protocol.unpack_event_metadata(event_packet)

                        self.events_received += 1

                        if event_type == "episode_end":
                            # Write to datastream
                            event_datastream.append(tick.timestamp, data)
                            if self.events_received <= 5:  # Log first few
                                print(f"  [EVENT] Episode {data['episode']} logged to datastream")

                        elif event_type == "training_complete":
                            # Training finished - stop recording gracefully
                            print(f"\n{'='*70}")
                            print(f"TRAINING COMPLETE")
                            print(f"  Total Episodes: {data.get('total_episodes', 'unknown')}")
                            print(f"  Total Steps: {data.get('total_steps', 'unknown')}")
                            print(f"  Reason: {data.get('reason', 'unknown')}")
                            print(f"{'='*70}\n")
                            print("Stopping recording and shutting down...")
                            try:
                                recording.stop()
                                recording_stopped = True
                                print(f"[SUCCESS] Recording saved to {os.path.abspath(self.recording_path)}")
                            except Exception as e:
                                print(f"Error stopping recording: {e}")
                                recording_stopped = True  # Prevent double-stop attempt
                            return  # Exit run() method gracefully

                    except BlockingIOError:
                        pass  # No event available, continue
                    except Exception as e:
                        if self.events_received == 0:  # Only warn once
                            print(f"  [WARNING] Error receiving event: {e}")

                    # Check for feedback commands (non-blocking)
                    try:
                        feedback_packet, _ = self.feedback_socket.recvfrom(udp_protocol.FEEDBACK_PACKET_SIZE)
                        timestamp, feedback_type, channels, frequency, amplitude, pulses, unpredictable, event_name = \
                            udp_protocol.unpack_feedback_command(feedback_packet)

                        # Apply feedback command to hardware
                        self.apply_feedback_command(
                            neurons,
                            feedback_type,
                            channels,
                            frequency,
                            amplitude,
                            pulses,
                            unpredictable,
                            event_name
                        )

                        # Log first few feedback commands
                        if self.feedback_commands_received <= 5:
                            print(f"  [FEEDBACK] {feedback_type} on {len(channels)} channels: {frequency}Hz, {amplitude}μA, {pulses} pulses ({event_name})")

                    except BlockingIOError:
                        pass  # No feedback available, continue
                    except Exception as e:
                        if self.feedback_commands_received == 0:  # Only warn once
                            print(f"  [WARNING] Error receiving feedback: {e}")

                    # Print statistics every 10 seconds
                    if time.time() - last_stats_time >= 10.0:
                        elapsed = time.time() - last_stats_time
                        recv_rate = self.packets_received / elapsed if elapsed > 0 else 0
                        send_rate = self.packets_sent / elapsed if elapsed > 0 else 0
                        avg_spikes = self.total_spikes / tick_count if tick_count > 0 else 0

                        print(f"Stats: {tick_count} ticks | "
                              f"Recv: {recv_rate:.1f} pkt/s | "
                              f"Send: {send_rate:.1f} pkt/s | "
                              f"Events: {self.events_received} | "
                              f"Feedback: {self.feedback_commands_received} | "
                              f"Avg spikes: {avg_spikes:.2f}/tick")

                        last_stats_time = time.time()
                        self.packets_received = 0
                        self.packets_sent = 0
                        self.events_received = 0
                        self.feedback_commands_received = 0

            except KeyboardInterrupt:
                print("\n\nShutting down...")

            finally:
                if not recording_stopped:
                    try:
                        recording.stop()
                        print(f"[SUCCESS] Recording saved to {os.path.abspath(self.recording_path)}")
                    except Exception as e:
                        print(f"[ERROR] Error stopping recording: {e}")
                print("\n" + "="*70)
                print(f"Total ticks processed: {tick_count}")
                print(f"Total stimulation commands applied: {self.stim_commands_applied}")
                print(f"Total feedback commands applied: {self.feedback_commands_received}")
                print(f"Total spikes collected: {self.total_spikes}")
                print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CL1 Neural Interface Server - Hardware interface for remote training'
    )

    parser.add_argument(
        '--training-host',
        type=str,
        required=True,
        help='IP address of training system'
    )
    parser.add_argument(
        '--stim-port',
        type=int,
        default=12345,
        help='UDP port for receiving stimulation commands (default: 12345)'
    )
    parser.add_argument(
        '--spike-port',
        type=int,
        default=12346,
        help='UDP port for sending spike data (default: 12346)'
    )
    parser.add_argument(
        '--event-port',
        type=int,
        default=12347,
        help='UDP port for receiving event metadata (default: 12347)'
    )
    parser.add_argument(
        '--feedback-port',
        type=int,
        default=12348,
        help='UDP port for receiving feedback commands (default: 12348)'
    )
    parser.add_argument(
        '--tick-frequency',
        type=int,
        default=240,
        help='Frequency to run neurons.loop() in Hz (default: 240)'
    )
    parser.add_argument(
        '--recording-path',
        type=str,
        default='./recordings',
        help='Path for saving CL1 recordings (default: ./recordings)'
    )

    args = parser.parse_args()

    interface = CL1NeuralInterface(
        training_host=args.training_host,
        stim_port=args.stim_port,
        spike_port=args.spike_port,
        event_port=args.event_port,
        feedback_port=args.feedback_port,
        tick_frequency_hz=args.tick_frequency,
        recording_path=args.recording_path,
    )

    interface.run()


if __name__ == '__main__':
    main()
