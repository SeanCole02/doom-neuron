"""
UDP Protocol for CL1 Neural Interface Communication

This module defines the binary packet formats for real-time communication
between the CL1 neural interface device and the training system.

Packet Formats:
    Stimulation Command (Training → CL1):
        [8 bytes timestamp][32 bytes frequencies][32 bytes amplitudes]
        Total: 72 bytes

    Spike Data (CL1 → Training):
        [8 bytes timestamp][32 bytes spike_counts]
        Total: 40 bytes

    Feedback Command (Training → CL1):
        [8 bytes timestamp][1 byte type][1 byte num_channels][64 bytes channels]
        [4 bytes frequency][4 bytes amplitude][4 bytes pulses][1 byte unpredictable]
        [32 bytes event_name][1 byte padding]
        Total: 120 bytes
"""

import struct
import time
import numpy as np
from typing import Tuple, List


# Packet size constants
TIMESTAMP_SIZE_BYTES = 8
NUM_CHANNEL_SETS = 8  # Updated from 9 - removed speed channels
FLOAT_SIZE_BYTES = 4
MAX_CHANNELS_PER_FEEDBACK = 64
FEEDBACK_NAME_SIZE = 32

STIM_PACKET_SIZE = TIMESTAMP_SIZE_BYTES + (NUM_CHANNEL_SETS * 2 * FLOAT_SIZE_BYTES)  # 72 bytes
SPIKE_PACKET_SIZE = TIMESTAMP_SIZE_BYTES + (NUM_CHANNEL_SETS * FLOAT_SIZE_BYTES)      # 40 bytes
FEEDBACK_PACKET_SIZE = 120  # See packet format in docstring

# Struct format strings (little-endian)
# '<' = little-endian, 'Q' = unsigned long long (8 bytes), 'f' = float (4 bytes)
STIM_FORMAT = '<Q' + ('f' * NUM_CHANNEL_SETS * 2)  # timestamp + freqs + amps
SPIKE_FORMAT = '<Q' + ('f' * NUM_CHANNEL_SETS)      # timestamp + spike counts

# Feedback command types
FEEDBACK_TYPE_INTERRUPT = 0
FEEDBACK_TYPE_EVENT = 1
FEEDBACK_TYPE_REWARD = 2


def pack_stimulation_command(frequencies: np.ndarray, amplitudes: np.ndarray) -> bytes:
    """
    Pack stimulation parameters into a binary UDP packet.

    Args:
        frequencies: Array of shape (num_channel_sets,) with frequency values in Hz
        amplitudes: Array of shape (num_channel_sets,) with amplitude values in μA

    Returns:
        72-byte binary packet ready to send via UDP

    Raises:
        ValueError: If arrays have incorrect shape
    """
    if frequencies.shape != (NUM_CHANNEL_SETS,):
        raise ValueError(f"frequencies must have shape ({NUM_CHANNEL_SETS},), got {frequencies.shape}")
    if amplitudes.shape != (NUM_CHANNEL_SETS,):
        raise ValueError(f"amplitudes must have shape ({NUM_CHANNEL_SETS},), got {amplitudes.shape}")

    # Get current timestamp in microseconds
    timestamp = int(time.time() * 1_000_000)

    # Convert to float32 to ensure 4-byte floats
    freq_floats = frequencies.astype(np.float32).tolist()
    amp_floats = amplitudes.astype(np.float32).tolist()

    # Pack: timestamp + all frequencies + all amplitudes
    packet = struct.pack(STIM_FORMAT, timestamp, *freq_floats, *amp_floats)

    assert len(packet) == STIM_PACKET_SIZE, f"Packet size mismatch: {len(packet)} != {STIM_PACKET_SIZE}"
    return packet


def unpack_stimulation_command(packet: bytes) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Unpack a stimulation command packet.

    Args:
        packet: 72-byte binary packet from UDP

    Returns:
        Tuple of (timestamp, frequencies, amplitudes)
        - timestamp: int (microseconds since epoch)
        - frequencies: np.ndarray of shape (num_channel_sets,)
        - amplitudes: np.ndarray of shape (num_channel_sets,)

    Raises:
        ValueError: If packet has incorrect size
    """
    if len(packet) != STIM_PACKET_SIZE:
        raise ValueError(f"Expected {STIM_PACKET_SIZE} bytes, got {len(packet)}")

    # Unpack all values
    values = struct.unpack(STIM_FORMAT, packet)

    timestamp = values[0]
    frequencies = np.array(values[1:NUM_CHANNEL_SETS+1], dtype=np.float32)
    amplitudes = np.array(values[NUM_CHANNEL_SETS+1:], dtype=np.float32)

    return timestamp, frequencies, amplitudes


def pack_spike_data(spike_counts: np.ndarray) -> bytes:
    """
    Pack spike count data into a binary UDP packet.

    Args:
        spike_counts: Array of shape (num_channel_sets,) with spike counts per channel set

    Returns:
        40-byte binary packet ready to send via UDP

    Raises:
        ValueError: If array has incorrect shape
    """
    if spike_counts.shape != (NUM_CHANNEL_SETS,):
        raise ValueError(f"spike_counts must have shape ({NUM_CHANNEL_SETS},), got {spike_counts.shape}")

    # Get current timestamp in microseconds
    timestamp = int(time.time() * 1_000_000)

    # Convert to float32
    spike_floats = spike_counts.astype(np.float32).tolist()

    # Pack: timestamp + all spike counts
    packet = struct.pack(SPIKE_FORMAT, timestamp, *spike_floats)

    assert len(packet) == SPIKE_PACKET_SIZE, f"Packet size mismatch: {len(packet)} != {SPIKE_PACKET_SIZE}"
    return packet


def unpack_spike_data(packet: bytes) -> Tuple[int, np.ndarray]:
    """
    Unpack a spike data packet.

    Args:
        packet: 40-byte binary packet from UDP

    Returns:
        Tuple of (timestamp, spike_counts)
        - timestamp: int (microseconds since epoch)
        - spike_counts: np.ndarray of shape (num_channel_sets,)

    Raises:
        ValueError: If packet has incorrect size
    """
    if len(packet) != SPIKE_PACKET_SIZE:
        raise ValueError(f"Expected {SPIKE_PACKET_SIZE} bytes, got {len(packet)}")

    # Unpack all values
    values = struct.unpack(SPIKE_FORMAT, packet)

    timestamp = values[0]
    spike_counts = np.array(values[1:], dtype=np.float32)

    return timestamp, spike_counts


def get_latency_ms(packet_timestamp: int) -> float:
    """
    Calculate network latency from packet timestamp to now.

    Args:
        packet_timestamp: Timestamp from packet (microseconds since epoch)

    Returns:
        Latency in milliseconds
    """
    now = int(time.time() * 1_000_000)
    latency_us = now - packet_timestamp
    return latency_us / 1000.0


# Event metadata packet format
def pack_event_metadata(event_type: str, data: dict) -> bytes:
    """
    Pack event metadata into a UDP packet.

    Args:
        event_type: Type of event ('episode_end', 'checkpoint', etc.)
        data: Dictionary of event data

    Returns:
        Binary packet with JSON payload
    """
    import json
    timestamp = int(time.time() * 1_000_000)

    payload = {
        'timestamp': timestamp,
        'event_type': event_type,
        'data': data
    }

    json_bytes = json.dumps(payload).encode('utf-8')

    # Packet format: [8 bytes timestamp][4 bytes length][JSON data]
    header = struct.pack('<QI', timestamp, len(json_bytes))
    return header + json_bytes


def unpack_event_metadata(packet: bytes) -> tuple:
    """
    Unpack event metadata packet.

    Args:
        packet: Binary packet from UDP

    Returns:
        Tuple of (timestamp, event_type, data)
    """
    import json

    if len(packet) < 12:
        raise ValueError(f"Packet too small: {len(packet)} bytes")

    timestamp, json_length = struct.unpack('<QI', packet[:12])
    json_bytes = packet[12:12+json_length]

    payload = json.loads(json_bytes.decode('utf-8'))

    return payload['timestamp'], payload['event_type'], payload['data']


def pack_feedback_command(
    feedback_type: str,
    channels: List[int],
    frequency: int,
    amplitude: float,
    pulses: int,
    unpredictable: bool = False,
    event_name: str = ""
) -> bytes:
    """
    Pack feedback stimulation command into a binary UDP packet.

    Args:
        feedback_type: Type of feedback ("interrupt", "event", or "reward")
        channels: List of channel numbers to stimulate
        frequency: Stimulation frequency in Hz
        amplitude: Stimulation amplitude in μA
        pulses: Number of pulses/bursts
        unpredictable: Whether this is unpredictable stimulation
        event_name: Name of the event (for event-based feedback)

    Returns:
        120-byte binary packet ready to send via UDP

    Raises:
        ValueError: If parameters are invalid
    """
    if len(channels) > MAX_CHANNELS_PER_FEEDBACK:
        raise ValueError(f"Too many channels: {len(channels)} > {MAX_CHANNELS_PER_FEEDBACK}")

    # Map feedback type string to integer
    type_map = {
        "interrupt": FEEDBACK_TYPE_INTERRUPT,
        "event": FEEDBACK_TYPE_EVENT,
        "reward": FEEDBACK_TYPE_REWARD
    }

    if feedback_type not in type_map:
        raise ValueError(f"Invalid feedback_type: {feedback_type}")

    type_byte = type_map[feedback_type]

    # Get current timestamp in microseconds
    timestamp = int(time.time() * 1_000_000)

    # Prepare channels array (pad with 0xFF for unused slots)
    channels_array = [0xFF] * MAX_CHANNELS_PER_FEEDBACK
    for i, ch in enumerate(channels):
        if ch < 0 or ch > 63:
            raise ValueError(f"Invalid channel number: {ch}")
        channels_array[i] = ch

    # Truncate/pad event name
    event_name_bytes = event_name.encode('utf-8')[:FEEDBACK_NAME_SIZE]
    event_name_bytes = event_name_bytes.ljust(FEEDBACK_NAME_SIZE, b'\x00')

    # Pack the packet
    # Format: Q (timestamp) + B (type) + B (num_channels) + 64B (channels) +
    #         I (freq) + f (amp) + I (pulses) + B (unpredictable) + 32s (name) + x (padding)
    packet = struct.pack(
        '<QBB64BIfIB32sx',
        timestamp,
        type_byte,
        len(channels),
        *channels_array,
        int(frequency),
        float(amplitude),
        int(pulses),
        1 if unpredictable else 0,
        event_name_bytes
    )

    assert len(packet) == FEEDBACK_PACKET_SIZE, f"Packet size mismatch: {len(packet)} != {FEEDBACK_PACKET_SIZE}"
    return packet


def unpack_feedback_command(packet: bytes) -> Tuple[int, str, List[int], int, float, int, bool, str]:
    """
    Unpack a feedback command packet.

    Args:
        packet: 120-byte binary packet from UDP

    Returns:
        Tuple of (timestamp, feedback_type, channels, frequency, amplitude, pulses, unpredictable, event_name)
        - timestamp: int (microseconds since epoch)
        - feedback_type: str ("interrupt", "event", or "reward")
        - channels: List[int] of channel numbers
        - frequency: int (Hz)
        - amplitude: float (μA)
        - pulses: int
        - unpredictable: bool
        - event_name: str

    Raises:
        ValueError: If packet has incorrect size
    """
    if len(packet) != FEEDBACK_PACKET_SIZE:
        raise ValueError(f"Expected {FEEDBACK_PACKET_SIZE} bytes, got {len(packet)}")

    # Unpack the packet
    unpacked = struct.unpack('<QBB64BIfIB32sx', packet)

    timestamp = unpacked[0]
    type_byte = unpacked[1]
    num_channels = unpacked[2]
    channels_array = unpacked[3:67]
    frequency = unpacked[67]
    amplitude = unpacked[68]
    pulses = unpacked[69]
    unpredictable_byte = unpacked[70]
    event_name_bytes = unpacked[71]

    # Map type byte to string
    type_map = {
        FEEDBACK_TYPE_INTERRUPT: "interrupt",
        FEEDBACK_TYPE_EVENT: "event",
        FEEDBACK_TYPE_REWARD: "reward"
    }

    feedback_type = type_map.get(type_byte, "unknown")

    # Extract actual channels (ignore 0xFF padding)
    channels = [ch for ch in channels_array[:num_channels] if ch != 0xFF]

    # Decode event name (strip null padding)
    event_name = event_name_bytes.rstrip(b'\x00').decode('utf-8')

    unpredictable = unpredictable_byte != 0

    return timestamp, feedback_type, channels, frequency, amplitude, pulses, unpredictable, event_name


if __name__ == "__main__":
    # Test the protocol
    print("Testing UDP protocol...")

    # Test stimulation command packing/unpacking
    test_freq = np.array(
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 8.0, 12.0, 18.0],
        dtype=np.float32
    )
    test_amp = np.array(
        [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
        dtype=np.float32
    )

    stim_packet = pack_stimulation_command(test_freq, test_amp)
    print(f"Stimulation packet size: {len(stim_packet)} bytes")

    timestamp, freq, amp = unpack_stimulation_command(stim_packet)
    print(f"Unpacked timestamp: {timestamp}")
    print(f"Frequencies match: {np.allclose(freq, test_freq)}")
    print(f"Amplitudes match: {np.allclose(amp, test_amp)}")

    # Test spike data packing/unpacking
    test_spikes = np.array([0, 2, 5, 1, 3, 0, 4, 2, 1, 0], dtype=np.float32)

    spike_packet = pack_spike_data(test_spikes)
    print(f"\nSpike packet size: {len(spike_packet)} bytes")

    timestamp, spikes = unpack_spike_data(spike_packet)
    print(f"Unpacked timestamp: {timestamp}")
    print(f"Spike counts match: {np.allclose(spikes, test_spikes)}")

    # Test latency calculation
    latency = get_latency_ms(timestamp)
    print(f"\nPacket latency: {latency:.3f} ms")

    print("\nAll protocol tests passed!")
