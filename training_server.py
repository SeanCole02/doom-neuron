"""
CHANGELOG:
(2025-12-04) Made survival.wad the default scenario. Added kill count to logs. Disabled movement reward.
"""
import gc
import math
import warnings
import multiprocessing as mp
import os
import random
import time
import traceback
from collections import deque
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal, Beta
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.connection import Connection

import vizdoom
from vizdoom import DoomGame, GameState, Mode, Button, GameVariable

# UDP communication for remote CL1 hardware
import socket
import udp_protocol

# CL SDK not needed on training system - hardware is remote
CL_AVAILABLE = False

# NOTE: (2025-11-18, jz) Use mjpeg for simple visualisation
from mjpeg_server import MJPEGServer

from collections import OrderedDict

class LRUCache(OrderedDict):
    def __init__(self, maxsize=2048):
        super().__init__()
        self.maxsize = maxsize
    def get_or_set(self, key, factory):
        if key in self:
            self.move_to_end(key)
            return self[key]
        value = factory()
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)
        return value


# Configuration

@dataclass
class EventFeedbackConfig:
    channels: List[int]
    base_frequency: float
    base_amplitude: float
    base_pulses: int
    info_key: str
    td_sign: str = 'positive'  # 'positive', 'negative', or 'absolute'
    freq_gain: float = 0.9
    freq_max_scale: float = 2.0
    amp_gain: float = 0.35
    amp_max_scale: float = 1.5
    pulse_gain: float = 0.5
    pulse_max_scale: float = 2.0
    ema_beta: float = 0.99
    unpredictable: bool = True
    unpredictable_frequency: float = 5.0
    unpredictable_duration_sec: float = 1.0
    unpredictable_rest_sec: float = 1.0
    unpredictable_channels: Optional[List[int]] = None
    unpredictable_amplitude: Optional[float] = None


@dataclass
class PPOConfig:
    """Configuration for PPO training and neural interface."""

    # Environment
    doom_config: str = "progressive_deathmatch.cfg"
    screen_resolution: str = "RES_320X240"
    use_screen_buffer: bool = True
    max_turn_delta: float = 360.0           # Maximum absolute degrees for TURN_LEFT_RIGHT_DELTA
    turn_step_degrees: float = 30.0         # Discrete turn step when using turn buttons
    camera_std_init: float = 3.0           # Initial std (degrees) for camera delta distribution
    use_discrete_action_set: bool = True   # Legacy flag; combinatorial action space is now default

    # Neural Interface - Channel Configuration
    num_channels: int = 64
    # Reduced from 9 to 8 channels to match NUM_CHANNEL_SETS (removed speed action)
    encoding_channels: List[int] = field(default_factory=lambda: [8, 9, 10, 17, 18, 25, 27, 28])
    move_forward_channels: List[int] = field(default_factory=lambda: [41, 42, 49])
    move_backward_channels: List[int] = field(default_factory=lambda: [50, 51, 58])
    move_left_channels: List[int] = field(default_factory=lambda: [13, 14, 21])
    move_right_channels: List[int] = field(default_factory=lambda: [45, 46, 53])
    turn_left_channels: List[int] = field(default_factory=lambda: [29, 30, 31, 37])
    turn_right_channels: List[int] = field(default_factory=lambda: [59, 60, 61, 62])
    attack_channels: List[int] = field(default_factory=lambda: [32, 33, 34])
    speed_channels: List[int] = field(default_factory=lambda: [5, 6])

    # Stimulation Design Parameters (for cl.StimDesign)
    # Biphasic pulse: phase1_duration, phase1_amplitude, phase2_duration, phase2_amplitude
    phase1_duration: float = 160.0  # μs
    phase2_duration: float = 160.0  # μs
    min_amplitude: float = 1.0      # μA (will be negative for phase1)
    max_amplitude: float = 2.5      # μA (will be positive for phase2)

    # Burst Design Parameters (for cl.BurstDesign)
    min_frequency: float = 4.0      # Hz - minimum burst frequency
    max_frequency: float = 40.0     # Hz - maximum burst frequency
    burst_count: int = 500          # Number of pulses per burst (long enough for inter-tick)

    # PPO Hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.997 # Increased from 0.99 to 9.997 as per Doom Initial Report
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.3
    entropy_coef: float = 0.02
    max_grad_norm: float = 3 # This can probably be reduced to 1, 0.5 was clipping policy movements too much
    normalize_returns: bool = False # Turned this off as part of Doom Initial Report

    # Training
    num_envs: int = 1
    steps_per_update: int = 2048  # Steps per env
    batch_size: int = 256
    num_epochs: int = 4
    max_episodes: int = 2000
    use_hardware: bool = True

    # UDP Configuration for remote CL1 hardware
    cl1_host: str = "localhost"  # IP address of CL1 device
    cl1_stim_port: int = 12345   # Port for sending stimulation commands to CL1
    cl1_spike_port: int = 12346  # Port for receiving spike data from CL1
    cl1_event_port: int = 12347  # Port for sending event metadata to CL1
    cl1_feedback_port: int = 12348  # Port for sending feedback stimulation to CL1

    # Visualisation
    vis_host: str = "0.0.0.0"      # IP for hosting visualisations
    vis_port: int = 12349          # Port for hosting visualisations
    vis_path: str = "/doom.mjpeg"  # Path for hosting visualisations

    # Network architecture
    hidden_size: int = 128

    # Logging and checkpointing
    log_dir: str = "checkpoints/l5_2048_rand/logs"
    checkpoint_dir: str = "checkpoints/l5_2048_rand"
    save_interval: int = 100  # episodes
    eval_interval: int = 50   # episodes
    feedback_positive_threshold: float = 1 # Needs tuning
    feedback_negative_threshold: float = -1 # Needs tuning
    aim_alignment_gain: float = 2.5
    aim_alignment_max_distance: float = 250.0
    aim_alignment_bonus: float = 2.5
    aim_alignment_bonus_deg: float = 4.0
    movement_velocity_reward_scale: float = 0.01
    simplified_reward: bool = True # Set to True as part of Doom Initial Report
    debug_joint_actions: bool = True
    debug_joint_actions_limit: int = 500
    feedback_positive_amplitude: float = 2.0
    feedback_positive_frequency: float = 20.0
    feedback_positive_pulses: int = 30
    feedback_negative_amplitude: float = 2.0
    feedback_negative_frequency: float = 60.0
    feedback_negative_pulses: int = 90
    feedback_episode_positive_pulses: int = 80
    feedback_episode_positive_frequency: float = 40.0
    feedback_episode_negative_pulses: int = 160
    feedback_episode_negative_frequency: float = 120.0
    episode_only_feedback: bool = False
    use_episode_feedback: bool = True  # Enable episode-level feedback stimulation
    episode_feedback_surprise_scaling: bool = True  # Scale episode feedback by surprise magnitude
    use_reward_feedback: bool = True
    reward_feedback_positive_channels: List[int] = field(default_factory=lambda: [19, 20, 22])
    reward_feedback_negative_channels: List[int] = field(default_factory=lambda: [23, 24, 26])
    event_movement_distance_threshold: float = 10.0
    event_feedback_settings: Dict[str, EventFeedbackConfig] = field( # MARK: feedback config
        default_factory=lambda: {
            'enemy_kill': EventFeedbackConfig(
                channels=[35, 36, 38],
                base_frequency=20.0,
                base_amplitude=2.5,
                base_pulses=40,
                info_key='event_enemy_kill',
                td_sign='positive',
                freq_gain=0.20,
                freq_max_scale=2.5,
                amp_gain=0.20,
                amp_max_scale=1.6,
                pulse_gain=0.20,
                pulse_max_scale=2.5
            ),
            'armor_pickup': EventFeedbackConfig(
                channels=[39, 40, 43],
                base_frequency=20.0,
                base_amplitude=2.0,
                base_pulses=35,
                info_key='event_armor_pickup',
                td_sign='positive',
                freq_gain=0.30,
                freq_max_scale=2.0,
                amp_gain=0.30,
                amp_max_scale=1.4,
                pulse_gain=0.30,
                pulse_max_scale=2.0
            ),
            'took_damage': EventFeedbackConfig(
                channels=[44, 47, 48],
                base_frequency=90.0,
                base_amplitude=2.2,
                base_pulses=50,
                info_key='event_took_damage',
                td_sign='negative',
                freq_gain=0.20,
                freq_max_scale=2.5,
                amp_gain=0.18,
                amp_max_scale=1.7,
                pulse_gain=0.20,
                pulse_max_scale=2.5,
                unpredictable=True,
                unpredictable_frequency=5.0,
                unpredictable_duration_sec=4.0,
                unpredictable_rest_sec=4.0,
                unpredictable_channels=[44, 47, 48],
                unpredictable_amplitude=2.2
            ),
            'ammo_waste': EventFeedbackConfig(
                channels=[52, 54, 55],
                base_frequency=60.0,
                base_amplitude=1.8,
                base_pulses=25,
                info_key='event_ammo_waste',
                td_sign='negative',
                freq_gain=0.15,
                freq_max_scale=1.8,
                amp_gain=0.15,
                amp_max_scale=1.3,
                pulse_gain=0.15,
                pulse_max_scale=1.8
            )
        }
    )
    decoder_enforce_nonnegative: bool = False # Can be changed, needs testing
    decoder_freeze_weights: bool = False # Can be changed, needs testing
    decoder_zero_bias: bool = True # Prefer to be true, needs testing, bias tends to cause the decoder to generate its own predictions for movement
    decoder_use_mlp: bool = False # Prefer to be false, causes decoder to learn how to play the game but was tested on random spikes, could be different in prod
    decoder_mlp_hidden: Optional[int] = 256 # Value felt ok, needs testing if you use decoder_use_mlp: True
    decoder_weight_l2_coef: float = 0.0 # Untuned
    decoder_bias_l2_coef: float = 0.0 # Untuned
    wall_ray_count: int = 0 # Disabled - not using wall rays
    wall_ray_max_range: int = 64 # Keep as is
    wall_depth_max_distance: float = 18.0 # Already calibrated, keep as is
    encoder_trainable: bool = True # Can try turning it False but I would say True is needed for reasonable PPO policy gradients especially if decoder_use_mlp: False
    encoder_entropy_coef: float = -0.10 # Entropy penalty for the encoder since we use beta sampling
    decoder_ablation_mode: str = 'none' # Ablation to test if decoder is learning on its own, "random" and "zero" are valid inputs
    encoder_use_cnn: bool = True # With my testing it seems like the CNN does not overfit/learn on its own, seems useful to keep True
    encoder_cnn_channels: int = 64 # Increased to 64 per Doom Iniital Report
    encoder_cnn_downsample: int = 4 # Arbitrary value, can be changed
    episode_positive_feedback_event: Optional[str] = None  # None defaults to overall reward
    episode_negative_feedback_event: Optional[str] = None  # None defaults to overall reward
    feedback_surprise_gain: float = 0.25 # Tune as needed, will depend on neurons
    feedback_surprise_max_scale: float = 2.0 # Tune as needed, will depend on neurons
    feedback_surprise_freq_gain: Optional[float] = 0.65 # Tune as needed, will depend on neurons
    feedback_surprise_amp_gain: Optional[float] = 0.35 # Tune as needed, will depend on neurons
    feedback_surprise_freq_max_scale: Optional[float] = 2.0 # Tune as needed, will depend on neurons
    feedback_surprise_amp_max_scale: Optional[float] = 1.5 # Tune as needed, will depend on neurons
    enemy_distance_normalization: float = 1024.0 # Already calibrated, do not change

    def __post_init__(self):
        """Initialize CL SDK channel sets (required)."""
        # Create ChannelSet objects for CL SDK when running on hardware
        self.all_channels = [i for i in range(64) if i not in {0, 4, 7, 56, 63}]

        # Training system doesn't need ChannelSet objects - those are on CL1 device
        # CL SDK not available/needed on training system
        self.all_channels_set = None
        self.encoding_channels_set = None
        self.move_forward_channels_set = None
        self.move_backward_channels_set = None
        self.move_left_channels_set = None
        self.move_right_channels_set = None
        self.turn_left_channels_set = None
        self.turn_right_channels_set = None
        self.attack_channels_set = None
        self.speed_channels_set = None
        self.event_feedback_channel_sets = {
            name: None for name in self.event_feedback_settings.keys()
        }
        self.reward_positive_channel_set = None
        self.reward_negative_channel_set = None

        forbidden_channels = {0, 4, 7, 56, 63}
        channel_usage: Dict[int, str] = {}
        channel_groups = [
            ('encoding', self.encoding_channels),
            ('move_forward', self.move_forward_channels),
            ('move_backward', self.move_backward_channels),
            ('move_left', self.move_left_channels),
            ('move_right', self.move_right_channels),
            ('turn_left', self.turn_left_channels),
            ('turn_right', self.turn_right_channels),
            ('attack', self.attack_channels),
        ]
        for event_name, cfg in self.event_feedback_settings.items():
            channel_groups.append((f'event_{event_name}', cfg.channels))

        for group_name, channels in channel_groups:
            for ch in channels:
                if not (0 <= ch < self.num_channels):
                    raise ValueError(f"Channel {ch} in group '{group_name}' exceeds available range 0-{self.num_channels - 1}.")
                if ch in forbidden_channels:
                    raise ValueError(f"Channel {ch} is reserved and cannot be used (group '{group_name}').")
                if ch in channel_usage:
                    raise ValueError(
                        f"Channel {ch} is shared between '{channel_usage[ch]}' and '{group_name}', which is not allowed."
                    )
                channel_usage[ch] = group_name


# ============================================================================
# Neural Networks
# ============================================================================

class EncoderNetwork(nn.Module):
    """Encodes game state into CL SDK stimulation parameters."""

    def __init__(self, obs_dim: int, config: PPOConfig, num_channel_sets: int):
        super().__init__()
        self.config = config
        self.num_channel_sets = num_channel_sets
        self.trainable = bool(getattr(config, 'encoder_trainable', False))

        self.use_cnn = bool(getattr(config, 'encoder_use_cnn', False))
        self.scalar_dim = int(getattr(config, 'scalar_obs_dim', obs_dim))
        hidden = config.hidden_size
        self._shape_warning_logged = False

        if self.use_cnn:
            cnn_shape = getattr(config, 'cnn_obs_shape', None)
            if cnn_shape is None:
                raise ValueError("encoder_use_cnn is True but config.cnn_obs_shape is not set")
            self.cnn_shape = cnn_shape
            channels, height, width = self.cnn_shape
            base_channels = int(getattr(config, 'encoder_cnn_channels', 16))
            self.cnn = nn.Sequential(
                nn.Conv2d(channels, base_channels, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.cnn_output_dim = base_channels * 4
            shared_input_dim = self.scalar_dim + self.cnn_output_dim
        else:
            self.cnn_shape = None
            self.cnn = None
            self.cnn_output_dim = 0
            shared_input_dim = obs_dim
            self.scalar_dim = obs_dim

        self.shared = nn.Sequential(
            nn.Linear(shared_input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )

        if self.trainable:
            self.freq_alpha_head = nn.Linear(hidden, num_channel_sets)
            self.freq_beta_head = nn.Linear(hidden, num_channel_sets)
            self.amp_alpha_head = nn.Linear(hidden, num_channel_sets)
            self.amp_beta_head = nn.Linear(hidden, num_channel_sets)
            self._freq_range = float(self.config.max_frequency - self.config.min_frequency)
            self._amp_range = float(self.config.max_amplitude - self.config.min_amplitude)
            self._freq_log_scale = math.log(self._freq_range + 1e-8)
            self._amp_log_scale = math.log(self._amp_range + 1e-8)
        else:
            self.freq_head = nn.Linear(hidden, num_channel_sets)
            self.amp_head = nn.Linear(hidden, num_channel_sets)

    def _combine_features(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            scalar = obs[:, :self.scalar_dim]
            image_flat = obs[:, self.scalar_dim:].contiguous()
            image = image_flat.view(-1, *self.cnn_shape)
            if image.size(1) > 1:
                image = image.mean(dim=1, keepdim=True)
            cnn_features = self.cnn(image).view(obs.size(0), -1)
            combined = torch.cat([scalar, cnn_features], dim=-1)
        else:
            combined = obs
        expected_dim = self.shared[0].in_features
        if combined.size(-1) != expected_dim and not self._shape_warning_logged:
            warnings.warn(
                f"Encoder input dimension mismatch: expected {expected_dim}, got {combined.size(-1)}. "
                f"Observation will be {'truncated' if combined.size(-1) > expected_dim else 'zero-padded'} to fit."
            )
            self._shape_warning_logged = True

        if combined.size(-1) > expected_dim:
            combined = combined[:, :expected_dim]
        elif combined.size(-1) < expected_dim:
            pad = torch.zeros(combined.size(0), expected_dim - combined.size(-1), device=combined.device, dtype=combined.dtype)
            combined = torch.cat([combined, pad], dim=-1)
        features = self.shared(combined)
        return features

    def _beta_params(self, alpha_head: torch.Tensor, beta_head: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softplus(alpha_head) + 1.0
        beta = F.softplus(beta_head) + 1.0
        return alpha, beta

    def _distributions(self, obs: torch.Tensor) -> Tuple[Beta, Beta]:
        features = self._combine_features(obs)
        freq_alpha, freq_beta = self._beta_params(
            self.freq_alpha_head(features),
            self.freq_beta_head(features)
        )
        amp_alpha, amp_beta = self._beta_params(
            self.amp_alpha_head(features),
            self.amp_beta_head(features)
        )
        return Beta(freq_alpha, freq_beta), Beta(amp_alpha, amp_beta)

    def _scale_freq(self, u: torch.Tensor) -> torch.Tensor:
        return self.config.min_frequency + u * self._freq_range

    def _scale_amp(self, u: torch.Tensor) -> torch.Tensor:
        return self.config.min_amplitude + u * self._amp_range

    def _unscale_freq(self, freq: torch.Tensor) -> torch.Tensor:
        return torch.clamp((freq - self.config.min_frequency) / (self._freq_range + 1e-8), 1e-6, 1 - 1e-6)

    def _unscale_amp(self, amp: torch.Tensor) -> torch.Tensor:
        return torch.clamp((amp - self.config.min_amplitude) / (self._amp_range + 1e-8), 1e-6, 1 - 1e-6)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.trainable:
            freq_dist, amp_dist = self._distributions(obs)
            frequencies = self._scale_freq(freq_dist.mean)
            amplitudes = self._scale_amp(amp_dist.mean)
            return frequencies, amplitudes

        features = self._combine_features(obs)
        freq_raw = torch.sigmoid(self.freq_head(features))
        amp_raw = torch.sigmoid(self.amp_head(features))

        frequencies = (
            self.config.min_frequency +
            freq_raw * (self.config.max_frequency - self.config.min_frequency)
        )
        amplitudes = (
            self.config.min_amplitude +
            amp_raw * (self.config.max_amplitude - self.config.min_amplitude)
        )
        return frequencies, amplitudes

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.trainable:
            frequencies, amplitudes = self.forward(obs)
            zeros = torch.zeros_like(frequencies)
            return frequencies, amplitudes, zeros, zeros, zeros, zeros

        freq_dist, amp_dist = self._distributions(obs)
        if deterministic:
            freq_u = freq_dist.mean
            amp_u = amp_dist.mean
        else:
            freq_u = freq_dist.rsample()
            amp_u = amp_dist.rsample()

        frequencies = self._scale_freq(freq_u)
        amplitudes = self._scale_amp(amp_u)
        freq_log_prob = freq_dist.log_prob(freq_u) - self._freq_log_scale
        amp_log_prob = amp_dist.log_prob(amp_u) - self._amp_log_scale
        freq_entropy = freq_dist.entropy() + self._freq_log_scale
        amp_entropy = amp_dist.entropy() + self._amp_log_scale
        return frequencies, amplitudes, freq_log_prob, amp_log_prob, freq_entropy, amp_entropy

    def log_prob(self, obs: torch.Tensor, frequencies: torch.Tensor, amplitudes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.trainable:
            zeros = torch.zeros_like(frequencies)
            return zeros, zeros
        freq_dist, amp_dist = self._distributions(obs)
        freq_u = self._unscale_freq(frequencies)
        amp_u = self._unscale_amp(amplitudes)
        freq_log_prob = freq_dist.log_prob(freq_u) - self._freq_log_scale
        amp_log_prob = amp_dist.log_prob(amp_u) - self._amp_log_scale
        return freq_log_prob, amp_log_prob

    def entropy(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.trainable:
            zeros = torch.zeros((obs.size(0), self.num_channel_sets), device=obs.device, dtype=obs.dtype)
            return zeros, zeros
        freq_dist, amp_dist = self._distributions(obs)
        freq_entropy = freq_dist.entropy() + self._freq_log_scale
        amp_entropy = amp_dist.entropy() + self._amp_log_scale
        return freq_entropy, amp_entropy

    def create_stim_commands(
        self,
        frequencies: torch.Tensor,
        amplitudes: torch.Tensor,
        channel_sets: List
    ) -> List[Tuple]:
        """
        Convert encoder outputs to CL SDK stimulation commands.

        Args:
            frequencies: (batch_size, num_channel_sets) Hz values
            amplitudes: (batch_size, num_channel_sets) μA values
            channel_sets: List of cl.ChannelSet objects

        Returns:
            List of (ChannelSet, StimDesign, BurstDesign) tuples for each channel set
        """
        if not CL_AVAILABLE:
            return None

        stim_commands = []
        for i, channel_set in enumerate(channel_sets):
            # Create biphasic StimDesign
            stim_design = cl.StimDesign(
                self.config.phase1_duration,
                -amplitudes[i].item(),  # Negative phase
                self.config.phase2_duration,
                amplitudes[i].item()     # Positive phase
            )

            # Create BurstDesign
            burst_design = cl.BurstDesign(
                self.config.burst_count,
                int(frequencies[i].item())
            )

            stim_commands.append((channel_set, stim_design, burst_design))

        return stim_commands


class LinearReadoutHead(nn.Module):
    """
    Lightweight linear readout with optional non-negative weight constraint.

    Keeps the decoder as a simple weighted sum over spike features so the
    downstream behavior is governed by the neural responses themselves.
    """

    def __init__(self, in_features: int, out_features: int, enforce_nonnegative: bool):
        super().__init__()
        self.enforce_nonnegative = enforce_nonnegative
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.enforce_nonnegative:
            weight = F.softplus(weight)
        return F.linear(inputs, weight, self.bias)

    def effective_weight(self) -> torch.Tensor:
        if self.enforce_nonnegative:
            return F.softplus(self.weight)
        return self.weight

    def weight_l2(self) -> torch.Tensor:
        return self.effective_weight().pow(2).sum()

    def bias_l2(self) -> torch.Tensor:
        return self.bias.pow(2).sum()


class DecoderNetwork(nn.Module):
    """Decoder that emits logits over the full combinatorial action space."""

    def __init__(self, spike_feature_dim: int, config: PPOConfig, num_joint_actions: int):
        super().__init__()
        self.config = config
        self.use_mlp = bool(getattr(config, 'decoder_use_mlp', False))
        self._head_names = ['joint']
        self._zero_bias_enabled = bool(getattr(config, 'decoder_zero_bias', False))

        if self.use_mlp:
            hidden = int(getattr(config, 'decoder_mlp_hidden', None) or config.hidden_size)
            hidden = max(hidden, 16)
            mid_hidden = max(hidden // 2, 8)
            self.shared = nn.Sequential(
                nn.Linear(spike_feature_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, mid_hidden),
                nn.SiLU()
            )
            feature_dim = mid_hidden
            self.joint_head = nn.Linear(feature_dim, num_joint_actions)
        else:
            enforce_positive = getattr(config, 'decoder_enforce_nonnegative', True)
            self.shared = None
            self.joint_head = LinearReadoutHead(spike_feature_dim, num_joint_actions, enforce_positive)

        self.heads = {'joint': self.joint_head}

        if self._zero_bias_enabled:
            self._zero_decoder_biases()
            self.register_load_state_dict_post_hook(self._zero_bias_load_hook)

        if getattr(config, 'decoder_freeze_weights', False):
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, spike_features: torch.Tensor) -> torch.Tensor:
        head_input = self.shared(spike_features) if self.use_mlp else spike_features
        return self.joint_head(head_input)

    def l2_penalties(self, effective: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        param_ref = next(self.parameters(), None)
        if param_ref is None:
            zero = torch.tensor(0.0)
            return zero, zero
        weight_total = torch.zeros((), device=param_ref.device, dtype=param_ref.dtype)
        bias_total = torch.zeros((), device=param_ref.device, dtype=param_ref.dtype)

        if self.use_mlp:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    weight_total = weight_total + module.weight.pow(2).sum()
                    if module.bias is not None:
                        bias_total = bias_total + module.bias.pow(2).sum()
        else:
            for head in self.heads.values():
                if isinstance(head, LinearReadoutHead):
                    if effective:
                        weight_total = weight_total + head.effective_weight().pow(2).sum()
                    else:
                        weight_total = weight_total + head.weight.pow(2).sum()
                    bias_total = bias_total + head.bias.pow(2).sum()
        return weight_total, bias_total

    def _zero_decoder_biases(self) -> None:
        for module in self.modules():
            if isinstance(module, LinearReadoutHead):
                module.bias.data.zero_()
                module.bias.requires_grad = False
            elif isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                module.bias.requires_grad = False

    def _zero_bias_load_hook(self, module: nn.Module, incompatible_keys) -> None:
        self._zero_decoder_biases()

    def compute_weight_bias_metrics(self, spike_features: torch.Tensor) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if spike_features.numel() == 0:
            return metrics
        eps = 1e-6
        with torch.no_grad():
            head_input = self.shared(spike_features) if self.use_mlp else spike_features
            for name, head in self.heads.items():
                if isinstance(head, LinearReadoutHead):
                    weight = head.effective_weight()
                    wx = torch.matmul(head_input, weight.t()).abs().mean()
                    bias_mean = head.bias.abs().mean()
                elif isinstance(head, nn.Linear):
                    weight = head.weight
                    wx = torch.matmul(head_input, weight.t()).abs().mean()
                    bias_mean = head.bias.abs().mean() if head.bias is not None else torch.tensor(0.0, device=head_input.device)
                else:
                    continue
                ratio = float((wx / (bias_mean + eps)).item())
                metrics[f'Decoder/{name}_wx_bias_ratio'] = ratio
                metrics[f'Decoder/{name}_bias_abs_mean'] = float(bias_mean.item())
                metrics[f'Decoder/{name}_wx_abs_mean'] = float(wx.item())

            weight_l2_eff, bias_l2_eff = self.l2_penalties(effective=not self.use_mlp)
            metrics['Decoder/weight_l2'] = float(weight_l2_eff.item())
            metrics['Decoder/bias_l2'] = float(bias_l2_eff.item())
        return metrics


class ValueNetwork(nn.Module):
    """Estimates state value for PPO critic."""

    def __init__(self, obs_dim: int, config: PPOConfig):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch_size, obs_dim)

        Returns:
            value: (batch_size, 1)
        """
        return self.network(obs)


class PPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, config: PPOConfig):
        super().__init__()
        self.config = config
        self.num_actions = num_actions

        # Channel grouping: (name, channel_list, channel_set)
        self.channel_groups: List[Tuple[str, List[int], Optional['cl.ChannelSet']]] = []

        def add_group(name: str, channels: List[int], channel_set: Optional['cl.ChannelSet']):
            if channels is None:
                return
            self.channel_groups.append((name, channels, channel_set))

        add_group('encoding', config.encoding_channels, getattr(config, 'encoding_channels_set', None))
        add_group('move_forward', config.move_forward_channels, getattr(config, 'move_forward_channels_set', None))
        add_group('move_backward', config.move_backward_channels, getattr(config, 'move_backward_channels_set', None))
        add_group('move_left', config.move_left_channels, getattr(config, 'move_left_channels_set', None))
        add_group('move_right', config.move_right_channels, getattr(config, 'move_right_channels_set', None))
        add_group('turn_left', config.turn_left_channels, getattr(config, 'turn_left_channels_set', None))
        add_group('turn_right', config.turn_right_channels, getattr(config, 'turn_right_channels_set', None))
        add_group('attack', config.attack_channels, getattr(config, 'attack_channels_set', None))

        self.num_channel_sets = len(self.channel_groups)

        self.forward_options = ['none', 'forward', 'backward']
        self.strafe_options = ['none', 'left', 'right']
        self.camera_options = ['none', 'turn_left', 'turn_right']
        self.attack_options = ['idle', 'attack']
        self.speed_options = ['off']  # Speed action removed - always off

        self.combinatorial_action_defs: List[Dict[str, Any]] = []
        for forward_idx, forward_name in enumerate(self.forward_options):
            for strafe_idx, strafe_name in enumerate(self.strafe_options):
                for turn_idx, turn_name in enumerate(self.camera_options):
                    for attack_idx, attack_name in enumerate(self.attack_options):
                        for speed_idx, speed_name in enumerate(self.speed_options):
                            action_name = (
                                f"{forward_name}_{strafe_name}_{turn_name}_"
                                f"{attack_name}_{speed_name}"
                            )
                            self.combinatorial_action_defs.append({
                                'name': action_name,
                                'forward': forward_idx,
                                'strafe': strafe_idx,
                                'turn': turn_idx,
                                'attack': attack_idx,
                                'speed': speed_idx,
                            })
        self.num_joint_actions = len(self.combinatorial_action_defs)
        print(f"[INFO] Action space size: {self.num_joint_actions} (all actions allowed while shooting)")

        # Components
        self.encoder = EncoderNetwork(obs_dim, config, num_channel_sets=self.num_channel_sets)
        self.decoder = DecoderNetwork(
            spike_feature_dim=self.num_channel_sets,
            config=config,
            num_joint_actions=self.num_joint_actions
        )
        self.value_net = ValueNetwork(obs_dim, config)
        self._stim_cache = LRUCache(maxsize=256)

        # Lookup from channel index to group index for spike counting
        self.channel_lookup: Dict[int, int] = {}
        for idx, (_, channel_list, _) in enumerate(self.channel_groups):
            for ch in channel_list:
                self.channel_lookup[ch] = idx

        forward_codes = torch.tensor([action['forward'] for action in self.combinatorial_action_defs], dtype=torch.long)
        strafe_codes = torch.tensor([action['strafe'] for action in self.combinatorial_action_defs], dtype=torch.long)
        turn_codes = torch.tensor(
            [action['turn'] for action in self.combinatorial_action_defs],
            dtype=torch.long
        )
        attack_codes = torch.tensor(
            [action['attack'] for action in self.combinatorial_action_defs],
            dtype=torch.long
        )
        speed_codes = torch.tensor(
            [action['speed'] for action in self.combinatorial_action_defs],
            dtype=torch.long
        )
        self.register_buffer('joint_forward_map', forward_codes, persistent=False)
        self.register_buffer('joint_strafe_map', strafe_codes, persistent=False)
        self.register_buffer('joint_turn_map', turn_codes, persistent=False)
        self.register_buffer('joint_attack_map', attack_codes, persistent=False)
        self.register_buffer('joint_speed_map', speed_codes, persistent=False)
        self._debug_joint_prints = 0

    def decode_spikes_to_action(
        self,
        spike_features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode spike features to actions.

        Args:
            spike_features: (batch_size, num_channel_sets) spike counts per channel set
            deterministic: If True, take argmax action

        Returns:
            forward_actions: (batch_size,) categorical over forward/back/idle
            strafe_actions: (batch_size,) categorical over strafe left/right/idle
            camera_actions: (batch_size,) discrete turn options (0=none, 1=left, 2=right)
            attack_actions: (batch_size,)
            speed_actions: (batch_size,)
            log_probs: (batch_size,)
            entropy: (batch_size,)
        """
        joint_logits = self.decoder(spike_features)
        joint_dist = Categorical(logits=joint_logits)
        if deterministic:
            joint_actions = joint_logits.argmax(dim=-1)
        else:
            joint_actions = joint_dist.sample()

        forward_map = self.joint_forward_map.to(spike_features.device)
        strafe_map = self.joint_strafe_map.to(spike_features.device)
        turn_map = self.joint_turn_map.to(spike_features.device)
        attack_map = self.joint_attack_map.to(spike_features.device)
        speed_map = self.joint_speed_map.to(spike_features.device)

        forward_actions = forward_map[joint_actions]
        strafe_actions = strafe_map[joint_actions]
        camera_actions = turn_map[joint_actions]
        attack_actions = attack_map[joint_actions]
        speed_actions = speed_map[joint_actions]

        log_probs = joint_dist.log_prob(joint_actions)
        entropy = joint_dist.entropy()

        if getattr(self.config, 'debug_joint_actions', False):
            limit = int(getattr(self.config, 'debug_joint_actions_limit', 50))
            if self._debug_joint_prints < limit and joint_actions.numel() > 0:
                sample_idx = 0
                attack_ratio = float(attack_actions.float().mean().item())
                joint_idx = int(joint_actions[sample_idx].item())
                print(
                    "[DEBUG] joint_action="
                    f"{joint_idx} "
                    f"(fwd={int(forward_actions[sample_idx])}, "
                    f"strafe={int(strafe_actions[sample_idx])}, "
                    f"turn={int(camera_actions[sample_idx])}, "
                    f"attack={int(attack_actions[sample_idx])}, "
                    f"speed={int(speed_actions[sample_idx])}) "
                    f"| attack_ratio={attack_ratio:.2f}"
                )
                self._debug_joint_prints += 1

        return (
            forward_actions,
            strafe_actions,
            camera_actions,
            attack_actions,
            speed_actions,
            log_probs,
            entropy
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for observations."""
        return self.value_net(obs)

    def ablate_spike_features_tensor(self, spike_features: torch.Tensor) -> torch.Tensor:
        mode = getattr(self.config, 'decoder_ablation_mode', 'none')
        if mode == 'zero':
            return torch.zeros_like(spike_features)
        if mode == 'random':
            return torch.rand_like(spike_features)
        return spike_features

    def ablate_spike_features_numpy(self, spike_features: np.ndarray) -> np.ndarray:
        mode = getattr(self.config, 'decoder_ablation_mode', 'none')
        if mode == 'zero':
            return np.zeros_like(spike_features)
        if mode == 'random':
            return np.random.rand(*spike_features.shape).astype(spike_features.dtype, copy=False)
        return spike_features

    def sample_encoder(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frequencies, amplitudes, freq_log_prob, amp_log_prob, freq_entropy, amp_entropy = self.encoder.sample(obs, deterministic=deterministic)
        encoder_log_prob = (freq_log_prob + amp_log_prob).sum(dim=-1)
        encoder_entropy = (freq_entropy + amp_entropy).sum(dim=-1)
        return frequencies, amplitudes, encoder_log_prob, encoder_entropy

    def evaluate_actions(
        self,
        spike_features: torch.Tensor,
        forward_actions: torch.Tensor,
        strafe_actions: torch.Tensor,
        camera_actions: torch.Tensor,
        attack_actions: torch.Tensor,
        speed_actions: torch.Tensor,
        obs: torch.Tensor,
        stim_frequencies: Optional[torch.Tensor] = None,
        stim_amplitudes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probs and entropy for given spike features and actions.
        Used during PPO update.

        Args:
            spike_features: (batch_size, num_channel_sets) spike counts from CL SDK
            forward_actions: categorical indices for forward/back/idle
            strafe_actions: categorical indices for strafe left/right/idle
            camera_actions: discrete turn indices
            attack_actions: binary attack selections
            speed_actions: binary indices for slow/fast
            obs: (batch_size, obs_dim) observations (for value estimation)

        Returns:
            log_probs: (batch_size,)
            values: (batch_size, 1)
            entropy: (batch_size,)
        """
        joint_logits = self.decoder(spike_features)
        joint_dist = Categorical(logits=joint_logits)
        forward_map = self.joint_forward_map.to(spike_features.device)
        strafe_map = self.joint_strafe_map.to(spike_features.device)
        turn_map = self.joint_turn_map.to(spike_features.device)
        attack_map = self.joint_attack_map.to(spike_features.device)
        speed_map = self.joint_speed_map.to(spike_features.device)

        forward_eq = forward_actions.unsqueeze(-1) == forward_map
        strafe_eq = strafe_actions.unsqueeze(-1) == strafe_map
        turn_eq = camera_actions.unsqueeze(-1) == turn_map
        attack_eq = attack_actions.unsqueeze(-1) == attack_map
        speed_eq = speed_actions.unsqueeze(-1) == speed_map
        matches = forward_eq & strafe_eq & turn_eq & attack_eq & speed_eq
        if not torch.all(matches.any(dim=-1)):
            raise ValueError("Encountered action tuple without combinatorial mapping.")
        joint_indices = matches.float().argmax(dim=-1)

        log_probs = joint_dist.log_prob(joint_indices)
        entropy = joint_dist.entropy()
        encoder_log_prob = torch.zeros_like(log_probs)
        encoder_entropy = torch.zeros_like(log_probs)
        if getattr(self.config, 'encoder_trainable', False):
            if stim_frequencies is None or stim_amplitudes is None:
                raise ValueError("Stim frequencies/amplitudes required when encoder is trainable.")
            freq_log_prob, amp_log_prob = self.encoder.log_prob(obs, stim_frequencies, stim_amplitudes)
            freq_entropy, amp_entropy = self.encoder.entropy(obs)
            encoder_log_prob = (freq_log_prob + amp_log_prob).sum(dim=-1)
            encoder_entropy = (freq_entropy + amp_entropy).sum(dim=-1)
            log_probs = log_probs + encoder_log_prob

        values = self.value_net(obs)

        return log_probs, values, entropy, encoder_log_prob, encoder_entropy

    def apply_stimulation(
        self,
        stim_socket: socket.socket,
        frequencies: np.ndarray,
        amplitudes: np.ndarray
    ):
        """
        Send stimulation commands to CL1 device via UDP.

        Args:
            stim_socket: UDP socket for sending stimulation commands
            frequencies: (num_channel_sets,) array of Hz values
            amplitudes: (num_channel_sets,) array of μA values
        """
        # Pack and send stimulation command via UDP
        packet = udp_protocol.pack_stimulation_command(frequencies, amplitudes)
        bytes_sent = stim_socket.sendto(packet, (self.config.cl1_host, self.config.cl1_stim_port))

        # Debug: Log first few sends
        if not hasattr(self, '_stim_send_count'):
            self._stim_send_count = 0
        self._stim_send_count += 1
        if self._stim_send_count <= 5:
            print(f"  [DEBUG] Sent stim packet #{self._stim_send_count}: {bytes_sent} bytes to {self.config.cl1_host}:{self.config.cl1_stim_port}")

    def collect_spikes(self, spike_socket: socket.socket) -> np.ndarray:
        """
        Receive spike counts from CL1 device via UDP.

        Args:
            spike_socket: UDP socket for receiving spike data

        Returns:
            spike_counts: (num_channel_sets,) array of spike counts per channel set
        """
        try:
            # Receive spike data packet
            packet, addr = spike_socket.recvfrom(udp_protocol.SPIKE_PACKET_SIZE)
            timestamp, spike_counts = udp_protocol.unpack_spike_data(packet)

            # Debug: Log first few receives
            if not hasattr(self, '_spike_recv_count'):
                self._spike_recv_count = 0
            self._spike_recv_count += 1
            if self._spike_recv_count <= 5:
                print(f"  [DEBUG] Received spike packet #{self._spike_recv_count}: {len(packet)} bytes from {addr}")

            return spike_counts
        except socket.timeout:
            # If no packet received, return zeros
            if not hasattr(self, '_timeout_warned'):
                print(f"  [WARNING] Socket timeout waiting for spikes - using zeros")
                self._timeout_warned = True
            return np.zeros(self.num_channel_sets, dtype=np.float32)
        except Exception as e:
            print(f"  [ERROR] Error receiving spikes: {e}")
            return np.zeros(self.num_channel_sets, dtype=np.float32)

    def clear_stim_cache(self):
        """Release cached stimulation designs to avoid excessive memory use."""
        self._stim_cache.clear()


# ============================================================================
# VizDoom Environment
# ============================================================================

class VizDoomEnv:
    """
    Gym-compatible wrapper for VizDoom with neural interface.
    """

    def __init__(self, config: PPOConfig, render: bool = False):
        self.config = config
        self.render_enabled = render

        # Create game
        self.game = DoomGame()
        self.game.load_config(config.doom_config)
        self.game.set_window_visible(render)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_kill_reward(200) # This can probably be modified but 200 felt ok in training
        # Enable game state features
        self.game.set_labels_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        if getattr(config, 'encoder_use_cnn', False):
            config.use_screen_buffer = True

        if config.use_screen_buffer:
            resolution = getattr(vizdoom.ScreenResolution, config.screen_resolution)
            self.game.set_screen_resolution(resolution)

        self.game.init()

        self.available_game_variables = list(self.game.get_available_game_variables())
        self.game_var_to_index = {
            var: idx for idx, var in enumerate(self.available_game_variables)
        }

        self.buttons = list(self.game.get_available_buttons())
        self.num_actions = len(self.buttons)
        self.action_space_size = self.num_actions
        self.screen_width = self.game.get_screen_width()
        self.screen_height = self.game.get_screen_height()

        # Enemy tracking
        self.max_tracked_enemies = 5
        self.enemy_slot_map: Dict[int, int] = {}

        # Observation space configuration
        # [killcount, health, selected_weapon_ammo, position_x, position_y, sin(angle), cos(angle), velocity_x, velocity_y]
        # + [5 enemies × 10 features: dist, sin(angle_to), cos(angle_to), sin(enemy_facing), cos(enemy_facing), position_x, position_y, velocity_x, velocity_y, active_flag]
        base_features = 9  # Added velocity_x and velocity_y
        enemy_features = self.max_tracked_enemies * 10  # 10 features per enemy (relative + absolute position + velocity + active flag)
        self.scalar_feature_dim = base_features + enemy_features

        if getattr(config, 'encoder_use_cnn', False):
            self.cnn_downsample = max(1, int(getattr(config, 'encoder_cnn_downsample', 4)))
            self.screen_height = self.game.get_screen_height()
            self.screen_width = self.game.get_screen_width()
            self.cnn_height = max(1, self.screen_height // self.cnn_downsample)
            self.cnn_width = max(1, self.screen_width // self.cnn_downsample)
            self.cnn_flat_dim = self.cnn_height * self.cnn_width
            self.obs_dim = self.scalar_feature_dim + self.cnn_flat_dim
            config.scalar_obs_dim = self.scalar_feature_dim
            config.cnn_obs_shape = (1, self.cnn_height, self.cnn_width)
        else:
            self.cnn_downsample = 1
            self.screen_height = self.game.get_screen_height()
            self.screen_width = self.game.get_screen_width()
            self.cnn_height = 0
            self.cnn_width = 0
            self.cnn_flat_dim = 0
            self.obs_dim = self.scalar_feature_dim
            config.scalar_obs_dim = self.scalar_feature_dim
            config.cnn_obs_shape = None

        self.ammo1_capacity = 25.0

        # Tracking
        self.episode_reward = 0
        self.episode_length = 0
        self.last_state_health = 100.0
        self.last_state_ammo = 0.0
        self.last_state_killcount = 0.0
        self.last_state_position_x = 0.0
        self.last_state_position_y = 0.0
        self.last_state_angle = 0.0
        self.last_state_velocity_x = 0.0
        self.last_state_velocity_y = 0.0

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.game.new_episode()
        self.episode_reward = 0
        self.episode_length = 0
        self.enemy_slot_map.clear()  # Clear enemy tracking for new episode

        state = self.game.get_state()
        self.last_state_health = self._state_game_variable(
            state,
            GameVariable.HEALTH,
            100.0
        )
        self.last_state_ammo = self._state_game_variable(
            state,
            GameVariable.SELECTED_WEAPON_AMMO,
            0.0
        )
        self.last_state_killcount = self._state_game_variable(
            state,
            GameVariable.KILLCOUNT,
            0.0
        )
        self.last_state_position_x = self._state_game_variable(
            state,
            GameVariable.POSITION_X,
            0.0
        )
        self.last_state_position_y = self._state_game_variable(
            state,
            GameVariable.POSITION_Y,
            0.0
        )
        self.last_state_angle = self._state_game_variable(
            state,
            GameVariable.ANGLE,
            0.0
        )
        self.last_state_velocity_x = self._state_game_variable(
            state,
            GameVariable.VELOCITY_X,
            0.0
        )
        self.last_state_velocity_y = self._state_game_variable(
            state,
            GameVariable.VELOCITY_Y,
            0.0
        )
        return self._get_observation(state)

    def step(self, action: Tuple[int, int, int, int, int]) -> Tuple[np.ndarray, float, bool, Dict, None | np.ndarray]:
        """
        Execute action and return (observation, reward, done, info, screen_buffer).

        Args:
            action: Tuple of (forward_idx, strafe_idx, turn_idx, attack_flag, speed_flag)
        """
        forward_choice = 0
        strafe_choice = 0
        turn_choice = 0
        attack_flag = False
        speed_choice = 0

        # Normalize action tuple lengths for backward compatibility
        if isinstance(action, tuple):
            forward_choice = action[0] if len(action) > 0 else 0
            strafe_choice = action[1] if len(action) > 1 else 0
            camera_action = action[2] if len(action) > 2 else 0
            attack_action = action[3] if len(action) > 3 else 0
            speed_choice = action[4] if len(action) > 4 else 0
        else:
            camera_action = 0
            attack_action = 0
            speed_choice = 0
            forward_choice = int(action)

        forward_choice = int(np.clip(int(forward_choice), 0, 2))
        strafe_choice = int(np.clip(int(strafe_choice), 0, 2))
        if isinstance(camera_action, (int, float, np.floating, np.integer)):
            turn_choice = int(np.clip(int(camera_action), 0, 2))
        else:
            turn_choice = 0
        attack_flag = bool(attack_action)
        speed_choice = int(np.clip(int(speed_choice), 0, 1))

        action_vector = np.zeros(len(self.buttons), dtype=np.float32)

        forward_to_button = {
            1: Button.MOVE_FORWARD,
            2: Button.MOVE_BACKWARD
        }

        strafe_to_button = {
            1: Button.MOVE_LEFT,
            2: Button.MOVE_RIGHT
        }

        button_index = {button: idx for idx, button in enumerate(self.buttons)}
        state_before = self.game.get_state()

        if forward_choice in forward_to_button:
            btn = forward_to_button[forward_choice]
            if btn in button_index:
                action_vector[button_index[btn]] = 1
        if strafe_choice in strafe_to_button:
            btn = strafe_to_button[strafe_choice]
            if btn in button_index:
                action_vector[button_index[btn]] = 1

        turn_step = float(getattr(self.config, 'turn_step_degrees', 30.0))

        if turn_choice == 1:
            if Button.TURN_LEFT in button_index:
                action_vector[button_index[Button.TURN_LEFT]] = 1
            else:
                turn_delta_idx = button_index.get(Button.TURN_LEFT_RIGHT_DELTA)
                if turn_delta_idx is not None:
                    action_vector[turn_delta_idx] = -turn_step
        elif turn_choice == 2:
            if Button.TURN_RIGHT in button_index:
                action_vector[button_index[Button.TURN_RIGHT]] = 1
            else:
                turn_delta_idx = button_index.get(Button.TURN_LEFT_RIGHT_DELTA)
                if turn_delta_idx is not None:
                    action_vector[turn_delta_idx] = turn_step

        if attack_flag and Button.ATTACK in button_index:
            action_vector[button_index[Button.ATTACK]] = 1

        if speed_choice == 1 and Button.SPEED in button_index:
            action_vector[button_index[Button.SPEED]] = 1
        # Execute action
        health_before, _ = self._extract_health_armor(state_before)
        ammo_before = self._state_game_variable(
            state_before,
            GameVariable.SELECTED_WEAPON_AMMO,
            self.last_state_ammo
        )
        killcount_before = self._state_game_variable(
            state_before,
            GameVariable.KILLCOUNT,
            self.last_state_killcount
        )
        self.last_state_health = health_before
        self.game.set_action(action_vector)
        self.game.advance_action(4) # This can be changed but most stable RL implementation advance 4 ticks at a time

        # Get new state
        state = self.game.get_state()
        screen_buffer = getattr(state, "screen_buffer", None) # VizDoom screen buffer is (channels, H, W)
        done = self.game.is_episode_finished()

        current_health, _ = self._extract_health_armor(state)
        ammo_after = self._state_game_variable(
            state,
            GameVariable.SELECTED_WEAPON_AMMO,
            ammo_before
        )
        killcount_after = self._state_game_variable(
            state,
            GameVariable.KILLCOUNT,
            killcount_before
        )
        damage_taken = max(0.0, health_before - current_health)
        ammo_spent = max(0.0, ammo_before - ammo_after)
        kills_delta = max(0.0, killcount_after - killcount_before)
        kill_reward = kills_delta * 100.0
        damage_penalty = -damage_taken * 5.0
        ammo_penalty = -ammo_spent * 10.0
        reward = kill_reward + damage_penalty + ammo_penalty
        enemy_kill_event = kills_delta > 0.0
        took_damage_event = damage_taken > 0.0
        ammo_waste_event = ammo_spent > 0.0

        # Always penalize remaining HP at episode end to discourage stalling/running away
        # This treats HP as a resource to use, not preserve
        if done:
            remaining_hp_penalty = -current_health * 5.0
            reward += remaining_hp_penalty
            obs = np.zeros(self.obs_dim, dtype=np.float32)
        else:
            obs = self._get_observation(state)

        # Update tracked state variables
        self.last_state_health = current_health
        self.last_state_ammo = ammo_after
        self.last_state_killcount = killcount_after

        # Update position and angle tracking
        if state is not None:
            self.last_state_position_x = self._state_game_variable(state, GameVariable.POSITION_X, self.last_state_position_x)
            self.last_state_position_y = self._state_game_variable(state, GameVariable.POSITION_Y, self.last_state_position_y)
            self.last_state_angle = self._state_game_variable(state, GameVariable.ANGLE, self.last_state_angle)
            self.last_state_velocity_x = self._state_game_variable(state, GameVariable.VELOCITY_X, self.last_state_velocity_x)
            self.last_state_velocity_y = self._state_game_variable(state, GameVariable.VELOCITY_Y, self.last_state_velocity_y)

        self.episode_reward += reward
        self.episode_length += 1

        info = {
            'episode_reward': self.episode_reward if done else None,
            'episode_length': self.episode_length if done else None,
            'game_reward': reward,
            'kill_reward': kill_reward,
            'damage_penalty': damage_penalty,
            'ammo_penalty': ammo_penalty,
            'event_enemy_kill': enemy_kill_event,
            'event_took_damage': took_damage_event,
            'event_ammo_waste': ammo_waste_event,
            'killcount_final': self.last_state_killcount if done else None
        }

        return obs, reward, done, info, screen_buffer

    def _state_game_variable(
        self,
        state: Optional[GameState],
        var: GameVariable,
        default: float = 0.0
    ) -> float:
        """Return a specific game variable either from the state or directly from the game."""
        mapping = getattr(self, 'game_var_to_index', None)
        if state is not None and mapping:
            idx = mapping.get(var)
            if idx is not None:
                game_vars = getattr(state, 'game_variables', None)
                if game_vars is not None and len(game_vars) > idx:
                    return float(game_vars[idx])
        game = getattr(self, 'game', None)
        if game is not None:
            try:
                return float(game.get_game_variable(var))
            except Exception:
                return default
        return default

    def _get_observation(self, state: GameState) -> np.ndarray:
        """Extract fully normalized observation vector for PPO."""
        if state is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        HEALTH_MAX = 100.0
        KILLCOUNT_MAX = 32.0
        AMMO_MAX = max(1.0, float(getattr(self, 'ammo1_capacity', 25.0)))
        POSITION_MAX = 1024.0  # Max position for normalization

        health_raw = self._state_game_variable(state, GameVariable.HEALTH, self.last_state_health)
        ammo_raw = self._state_game_variable(state, GameVariable.SELECTED_WEAPON_AMMO, self.last_state_ammo)
        killcount_raw = self._state_game_variable(state, GameVariable.KILLCOUNT, self.last_state_killcount)
        position_x_raw = self._state_game_variable(state, GameVariable.POSITION_X, self.last_state_position_x)
        position_y_raw = self._state_game_variable(state, GameVariable.POSITION_Y, self.last_state_position_y)
        angle_raw = self._state_game_variable(state, GameVariable.ANGLE, self.last_state_angle)

        health_frac = np.clip(health_raw / HEALTH_MAX, 0.0, 1.0)
        ammo_frac = np.clip(ammo_raw / AMMO_MAX, 0.0, 1.0)
        kills_frac = np.clip(killcount_raw / KILLCOUNT_MAX, 0.0, 1.0)

        health = health_frac * 2.0 - 1.0
        ammo_norm = ammo_frac * 2.0 - 1.0
        kills = kills_frac * 2.0 - 1.0

        # Normalize positions to [-1, 1]
        position_x_norm = np.clip(position_x_raw / POSITION_MAX, -1.0, 1.0)
        position_y_norm = np.clip(position_y_raw / POSITION_MAX, -1.0, 1.0)

        # Convert angle to sin/cos (angle is in degrees)
        angle_rad = np.deg2rad(angle_raw)
        sin_angle = np.sin(angle_rad)
        cos_angle = np.cos(angle_rad)

        # Extract and normalize velocity
        velocity_x_raw = self._state_game_variable(state, GameVariable.VELOCITY_X, self.last_state_velocity_x)
        velocity_y_raw = self._state_game_variable(state, GameVariable.VELOCITY_Y, self.last_state_velocity_y)
        # Normalize velocity (typical Doom velocity range ~0-40 units/tick)
        VELOCITY_MAX = 50.0
        velocity_x_norm = np.clip(velocity_x_raw / VELOCITY_MAX, -1.0, 1.0)
        velocity_y_norm = np.clip(velocity_y_raw / VELOCITY_MAX, -1.0, 1.0)

        # Scalar features: kills, health, ammo, position_x, position_y, sin(angle), cos(angle), velocity_x, velocity_y
        obs_components = [
            kills,  # [-1, 1]
            health,  # [-1, 1]
            ammo_norm,  # [-1, 1]
            position_x_norm,  # [-1, 1]
            position_y_norm,  # [-1, 1]
            sin_angle,  # [-1, 1]
            cos_angle,  # [-1, 1]
            velocity_x_norm,  # [-1, 1]
            velocity_y_norm  # [-1, 1]
        ]

        # Add enemy features
        enemy_features = self._get_enemy_features(state)
        obs_components.extend(enemy_features.tolist())

        scalar_vector = np.array(obs_components, dtype=np.float32)

        if getattr(self.config, 'encoder_use_cnn', False):
            image_flat = self._extract_cnn_features(state)
            obs = np.concatenate([scalar_vector, image_flat], axis=0).astype(np.float32)
        else:
            obs = scalar_vector

        # Safety: remove NaNs/Infs
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _extract_health_armor(self, state: Optional[GameState]) -> Tuple[float, float]:
        """Return current health and armor with safe fallbacks."""
        last_health = getattr(self, 'last_state_health', 100.0)
        if state is None:
            return last_health, 0.0

        health = self._state_game_variable(state, GameVariable.HEALTH, last_health)
        return health, 0.0

    def _update_enemy_slot_map(self, enemies: list):
        """
        Assign persistent slot indices to enemies.
        Reuses slots when enemies disappear.
        """
        current_ids = {getattr(e, "id", None) for e in enemies if getattr(e, "id", None) is not None}

        # Remove stale entries
        stale_ids = set(self.enemy_slot_map.keys()) - current_ids
        for stale_id in stale_ids:
            del self.enemy_slot_map[stale_id]

        # Assign new enemies to available slots
        used_slots = set(self.enemy_slot_map.values())
        available_slots = [i for i in range(self.max_tracked_enemies) if i not in used_slots]

        for enemy in enemies:
            enemy_id = getattr(enemy, "id", None)
            if enemy_id is None or enemy_id in self.enemy_slot_map:
                continue
            if available_slots:
                self.enemy_slot_map[enemy_id] = available_slots.pop(0)

    def _get_enemy_features(self, state: GameState) -> np.ndarray:
        """
        Return stacked features for up to max_tracked_enemies.
        Each enemy: [dist_norm, sin(angle_to), cos(angle_to), sin(enemy_facing), cos(enemy_facing), position_x_norm, position_y_norm, velocity_x_norm, velocity_y_norm, active_flag] in [-1,1].
        Missing enemies default to [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (far distance, center position, zero velocity, inactive).
        """
        features = np.zeros((self.max_tracked_enemies, 10), dtype=np.float32)
        features[:, 0] = 1.0  # default far distance
        features[:, 5] = 0.0  # default position_x (center)
        features[:, 6] = 0.0  # default position_y (center)
        features[:, 7] = 0.0  # default velocity_x (stationary)
        features[:, 8] = 0.0  # default velocity_y (stationary)
        features[:, 9] = 0.0  # default inactive (no enemy in slot)

        if (
            state is None
            or not hasattr(state, "objects")
            or not state.objects
            or not hasattr(state, "game_variables")
            or len(state.game_variables) < 3
        ):
            return features.flatten()

        # Player position from game variables (POSITION_X, POSITION_Y are indices 0, 1)
        player_pos = np.array([
            self._state_game_variable(state, GameVariable.POSITION_X, self.last_state_position_x),
            self._state_game_variable(state, GameVariable.POSITION_Y, self.last_state_position_y)
        ], dtype=np.float32)

        # Filter for only Imp enemies
        filtered = {"DoomImp"}
        enemies = [
            obj for obj in state.objects
            if obj is not None and getattr(obj, "name", None) in filtered
        ]

        if not enemies:
            self.enemy_slot_map.clear()
            return features.flatten()

        self._update_enemy_slot_map(enemies)
        max_dist = 1024.0  # Distance normalization constant
        denom = max(1e-6, 0.5 * max_dist)

        for enemy_obj in enemies:
            enemy_id = getattr(enemy_obj, "id", None)
            if enemy_id is None:
                continue
            slot_idx = self.enemy_slot_map.get(enemy_id)
            if slot_idx is None or slot_idx >= self.max_tracked_enemies:
                continue

            # Enemy position
            enemy_pos = np.array([enemy_obj.position_x, enemy_obj.position_y], dtype=np.float32)
            dx, dy = enemy_pos - player_pos
            distance = float(np.hypot(dx, dy))
            dist_norm = float(np.tanh(distance / denom))

            # Angle from player to enemy
            angle_to = np.arctan2(dy, dx)
            sin_angle_to = float(np.clip(np.sin(angle_to), -1.0, 1.0))
            cos_angle_to = float(np.clip(np.cos(angle_to), -1.0, 1.0))

            # Enemy facing direction
            enemy_angle = getattr(enemy_obj, "angle", 0.0)
            enemy_angle_rad = np.deg2rad(enemy_angle)
            sin_enemy_facing = float(np.clip(np.sin(enemy_angle_rad), -1.0, 1.0))
            cos_enemy_facing = float(np.clip(np.cos(enemy_angle_rad), -1.0, 1.0))

            # Absolute enemy position (normalized to [-1, 1])
            POSITION_MAX = 1024.0
            enemy_x_norm = float(np.clip(enemy_obj.position_x / POSITION_MAX, -1.0, 1.0))
            enemy_y_norm = float(np.clip(enemy_obj.position_y / POSITION_MAX, -1.0, 1.0))

            # Enemy velocity (normalized to [-1, 1])
            VELOCITY_MAX = 50.0
            enemy_vx = getattr(enemy_obj, "velocity_x", 0.0)
            enemy_vy = getattr(enemy_obj, "velocity_y", 0.0)
            enemy_vx_norm = float(np.clip(enemy_vx / VELOCITY_MAX, -1.0, 1.0))
            enemy_vy_norm = float(np.clip(enemy_vy / VELOCITY_MAX, -1.0, 1.0))

            features[slot_idx, 0] = dist_norm
            features[slot_idx, 1] = sin_angle_to
            features[slot_idx, 2] = cos_angle_to
            features[slot_idx, 3] = sin_enemy_facing
            features[slot_idx, 4] = cos_enemy_facing
            features[slot_idx, 5] = enemy_x_norm
            features[slot_idx, 6] = enemy_y_norm
            features[slot_idx, 7] = enemy_vx_norm
            features[slot_idx, 8] = enemy_vy_norm
            features[slot_idx, 9] = 1.0  # active flag (enemy present in this slot)

        return features.flatten()

    def _extract_cnn_features(self, state: Optional[GameState]) -> np.ndarray:
        if state is None:
            return np.zeros(self.cnn_flat_dim, dtype=np.float32)

        buffer = getattr(state, 'screen_buffer', None)
        if buffer is None or buffer.size == 0:
            return np.zeros(self.cnn_flat_dim, dtype=np.float32)

        # VizDoom screen buffer is (channels, H, W)
        if buffer.ndim == 3:
            image = buffer.mean(axis=0)
        else:
            image = buffer

        if image.ndim > 2:
            image = image.mean(axis=0)

        downsample = max(1, self.cnn_downsample)
        image_ds = image[::downsample, ::downsample]
        if image_ds.shape[0] < self.cnn_height or image_ds.shape[1] < self.cnn_width:
            padded = np.zeros((self.cnn_height, self.cnn_width), dtype=np.float32)
            h = min(image_ds.shape[0], self.cnn_height)
            w = min(image_ds.shape[1], self.cnn_width)
            padded[:h, :w] = image_ds[:h, :w]
            image_ds = padded
        else:
            image_ds = image_ds[:self.cnn_height, :self.cnn_width]

        image_norm = (image_ds / 255.0).astype(np.float32)
        return image_norm.reshape(-1)

    # NOTE: _extract_wall_ranges method removed - wall rays disabled
    # def _extract_wall_ranges(self, state: Optional[GameState]) -> np.ndarray:
    #     num_rays = max(1, self.wall_ray_count)
    #     ranges = np.ones(num_rays, dtype=np.float32)
    #     if state is None or not hasattr(state, 'depth_buffer'):
    #         return ranges
    #
    #     depth_buffer = state.depth_buffer
    #     if depth_buffer is None or len(depth_buffer) == 0:
    #         return ranges
    #
    #     try:
    #         depth_map = np.array(depth_buffer, dtype=np.float32).reshape(self.screen_height, self.screen_width)
    #     except ValueError:
    #         return ranges
    #
    #     max_distance = max(1e-3, self.wall_depth_max_distance)
    #     columns = np.linspace(0, self.screen_width - 1, num_rays, dtype=int)
    #     horizon_row = min(self.screen_height - 1, int(self.screen_height * 0.6))
    #
    #     for idx, col in enumerate(columns):
    #         column_depth = depth_map[:horizon_row, col]
    #         valid = column_depth[column_depth > 0.0]
    #         if valid.size == 0:
    #             distance = max_distance
    #         else:
    #             distance = float(np.clip(valid.min(), 0.0, max_distance))
    #         ranges[idx] = distance / max_distance
    #
    #     return ranges

    def close(self):
        """Clean up resources."""
        self.game.close()

# ============================================================================
# Multiprocessing Worker
# ============================================================================

def _vizdoom_worker(
    remote,
    parent_remote,
    config: PPOConfig,
    worker_id: int,
    render: bool = False
):
    """Run a VizDoom environment inside a separate process."""
    parent_remote.close()

    # Basic seeding for reproducibility across workers
    seed = (int(time.time() * 1000) + worker_id * 97) % 2**32
    np.random.seed(seed)
    random.seed(seed)

    env = VizDoomEnv(config, render=render)
    observation = env.reset()
    remote.send(observation)

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == 'step':
                action = data
                next_obs, reward, done, info, screen_buffer = env.step(action)

                if done:
                    info = info or {}
                    info['terminal_observation'] = next_obs
                    next_obs = env.reset()

                remote.send((next_obs, reward, done, info))

            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)

            elif cmd == 'close':
                env.close()
                remote.close()
                break

            else:
                raise NotImplementedError(f"Unknown command '{cmd}' sent to VizDoom worker")

    except Exception:
        env.close()
        remote.close()
        raise


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    def __init__(
        self,
        config:            PPOConfig,
        tick_frequency_hz: int  = 10,
        recording_path:    str  = "/data/recordings/seandoom",
        show_window:       bool = False,
        device:            str  = 'cuda'
        ):
        self.config            = config
        self.device            = device
        self.tick_frequency_hz = tick_frequency_hz
        self.recording_path    = recording_path

        # UDP sockets for CL1 communication
        self.stim_socket: Optional[socket.socket] = None
        self.spike_socket: Optional[socket.socket] = None
        self.event_socket: Optional[socket.socket] = None

        if self.config.use_hardware:
            print(f"Will connect to CL1 device at {config.cl1_host}:{config.cl1_stim_port}/{config.cl1_spike_port}/{config.cl1_event_port}")

        # Environment / workers
        self.env: Optional[VizDoomEnv] = None
        self.remotes: Optional[List[Connection]] = None
        self.worker_processes: List[mp.Process] = []
        self.vector_obs: Optional[np.ndarray] = None

        if self.config.use_hardware:
            # Single environment with rendering for hardware training
            self.env = VizDoomEnv(config, render=show_window) # Set to False to save hardware resources
            obs_dim = self.env.obs_dim
            num_actions = self.env.num_actions
        else:
            obs_dim, num_actions = self._init_vectorized_envs()

        # Create policy
        self.policy = PPOPolicy(
            obs_dim=obs_dim,
            num_actions=num_actions,
            config=config
        ).to(device)

        # Running statistics for surprise-based feedback
        self._surprise_ema = 0.0
        self._surprise_beta = 0.999 # Tune if too spiky/smooth, from testing its usually quite spiky (probably because of random spikes) and 0.999 felt good
        self.surprise_sum = 0.0
        self.surprise_count = 0
        self._surprise_pos_ema = 0.0
        self._surprise_neg_ema = 0.0
        self._surprise_pos_beta = self._surprise_beta
        self._surprise_neg_beta = self._surprise_beta
        self.last_positive_freq_scale = 1.0
        self.last_positive_amp_scale = 1.0
        self.last_negative_freq_scale = 1.0
        self.last_negative_amp_scale = 1.0
        self.positive_freq_scale_sum = 0.0
        self.positive_amp_scale_sum = 0.0
        self.negative_freq_scale_sum = 0.0
        self.negative_amp_scale_sum = 0.0
        self.positive_freq_scale_count = 0
        self.positive_amp_scale_count = 0
        self.negative_freq_scale_count = 0
        self.negative_amp_scale_count = 0
        self._feedback_stim_cache: Dict[Tuple[str, int, float, int], Tuple[Any, Any]] = {}
        self._single_channel_cache: Dict[int, Any] = {}
        self._update_surprise_scaling()
        self._event_feedback_specs: Dict[str, Dict[str, Any]] = {}
        self._episode_value_prediction: Optional[float] = None
        self._episode_surprise_ema = 0.0
        self._episode_surprise_beta = 0.9
        self._last_episode_surprise = 0.0
        self._init_event_feedback_specs()

        # Optimizer
        decoder_lr = config.learning_rate
        encoder_lr = decoder_lr * 4.0 # Arbitary value but 4x felt ok in testing, might be a little high with as ppo was clipping ~30% at 0.2 eps_clip

        encoder_params = list(self.policy.encoder.parameters())
        decoder_params = list(self.policy.decoder.parameters())
        value_params = list(self.policy.value_net.parameters())

        param_groups = []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': encoder_lr})
        if decoder_params:
            param_groups.append({'params': decoder_params, 'lr': decoder_lr})
        if value_params:
            param_groups.append({'params': value_params, 'lr': decoder_lr})

        self.optimizer = torch.optim.Adam(
            param_groups,
            lr=decoder_lr
        )

        # Logging
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)
        self.log_dir = getattr(self.writer, 'log_dir', config.log_dir)
        self._log_surprise_scaling_config()

        # Metrics tracking
        self.episode_rewards = deque(maxlen=25)
        self.episode_lengths = deque(maxlen=25)
        self.episode_killcounts = deque(maxlen=25)
        self.total_episodes = 0
        self.total_steps = 0
        self.last_checkpoint_episode = 0
        self.successful_actions = 0
        self.unsuccessful_actions = 0
        self.neutral_actions = 0
        self.positive_feedback_count = 0
        self.negative_feedback_count = 0
        self.total_feedback_actions = 0
        self._feedback_plans_ready = False
        self._positive_episode_plan = None
        self._negative_episode_plan = None
        self.vector_reward_sums: Optional[np.ndarray] = None
        self.vector_episode_lengths: Optional[np.ndarray] = None
        self.training_log_path = os.path.join(self.config.checkpoint_dir, 'training_log.jsonl')
        self.episode_success_actions = 0
        self.episode_fail_actions = 0
        self.episode_neutral_actions = 0
        self._unpredictable_active = False
        self._unpredictable_phase: str = 'idle'
        self._unpredictable_phase_end: float = 0.0
        self._unpredictable_next_pulse: float = 0.0
        self._unpredictable_params: Dict[str, Any] = {}
        if os.path.exists(self.training_log_path):
            with open(self.training_log_path, 'r', encoding='utf-8') as log_file:
                lines = log_file.readlines()
                if lines:
                    try:
                        last = json.loads(lines[-1])
                        self.total_episodes = int(last.get('total_episodes', self.total_episodes))
                        self.total_steps = int(last.get('total_steps', self.total_steps))
                    except json.JSONDecodeError:
                        pass
        else:
            with open(self.training_log_path, 'w', encoding='utf-8') as _:
                pass
        if self.config.save_interval > 0:
            self.last_checkpoint_episode = (self.total_episodes // self.config.save_interval) * self.config.save_interval

    def setup_udp_sockets(self):
        """Set up UDP sockets for CL1 communication."""
        print(f"\n[UDP Setup]")
        print(f"  CL1 Host: {self.config.cl1_host}")
        print(f"  Stim Port (send to CL1): {self.config.cl1_stim_port}")
        print(f"  Spike Port (receive from CL1): {self.config.cl1_spike_port}")
        print(f"  Event Port (send to CL1): {self.config.cl1_event_port}")
        print(f"  Feedback Port (send to CL1): {self.config.cl1_feedback_port}")
        print(f"\n[Feedback Configuration]")
        print(f"  Episode Feedback: {getattr(self.config, 'use_episode_feedback', 'NOT SET')}")
        print(f"  Episode Surprise Scaling: {getattr(self.config, 'episode_feedback_surprise_scaling', 'NOT SET')}")
        print(f"  Reward Feedback: {getattr(self.config, 'use_reward_feedback', 'NOT SET')}")

        # Socket for sending stimulation commands to CL1
        self.stim_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[SUCCESS] Created UDP socket for sending stim commands to {self.config.cl1_host}:{self.config.cl1_stim_port}")

        # Socket for receiving spike data from CL1
        self.spike_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.spike_socket.bind(("0.0.0.0", self.config.cl1_spike_port))
        self.spike_socket.settimeout(0.1)  # 100ms timeout for receiving spikes
        print(f"[SUCCESS] Listening for spike data on port {self.config.cl1_spike_port}")

        # Socket for sending event metadata to CL1
        self.event_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[SUCCESS] Created UDP socket for sending event metadata to {self.config.cl1_host}:{self.config.cl1_event_port}")

        # Socket for sending feedback stimulation commands to CL1
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[SUCCESS] Created UDP socket for sending feedback commands to {self.config.cl1_host}:{self.config.cl1_feedback_port}\n")

    def set_up_image_stream_server(self):
        """ Set up a MJPEGServer to run on a separate process for visualising the game frame buffer. """
        print(f"\n[MJPEGServer Setup]")
        print(f"  Visualisations hosting at {self.config.vis_host}:{self.config.vis_port}{self.config.vis_path}")
        self.image_stream = MJPEGServer(
            host = self.config.vis_host,
            port = self.config.vis_port,
            path = self.config.vis_path
            )

    def _init_vectorized_envs(self) -> Tuple[int, int]:
        """Spin up VizDoom environments across multiple processes."""
        ctx = mp.get_context('spawn')
        self.remotes = []
        self.worker_processes = []

        for idx in range(self.config.num_envs):
            parent_remote, child_remote = ctx.Pipe()
            process = ctx.Process(
                target=_vizdoom_worker,
                args=(child_remote, parent_remote, self.config, idx, idx == 0),
                daemon=True
            )
            process.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.worker_processes.append(process)

        # Receive initial observations
        initial_obs = [remote.recv() for remote in self.remotes]
        self.vector_obs = np.stack(initial_obs).astype(np.float32)
        self.vector_reward_sums = np.zeros(self.config.num_envs, dtype=np.float32)
        self.vector_episode_lengths = np.zeros(self.config.num_envs, dtype=np.int32)

        # Determine action space size using a temporary env instance (avoid persistent window)
        temp_env = VizDoomEnv(self.config, render=True)
        num_actions = temp_env.num_actions
        obs_dim = temp_env.obs_dim
        temp_env.close()

        return obs_dim, num_actions

    def _close_vectorized_envs(self):
        if self.remotes is None:
            return
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except (BrokenPipeError, EOFError):
                pass
            try:
                remote.close()
            except OSError:
                pass

        for p in self.worker_processes:
            p.join(timeout=5)
        for p in self.worker_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        self.remotes = None
        self.worker_processes = []
        self.vector_obs = None
        self.vector_reward_sums = None
        self.vector_episode_lengths = None

    def _resolve_episode_feedback_target(self, positive: bool) -> Optional[Dict[str, Any]]:
        """Choose the episode feedback definition, defaulting to reward-based channels."""
        event_attr = 'episode_positive_feedback_event' if positive else 'episode_negative_feedback_event'
        event_name = getattr(self.config, event_attr, None)
        if event_name:
            spec = self._event_feedback_specs.get(event_name)
            if spec and spec.get('channel_set') is not None:
                cfg: EventFeedbackConfig = spec['config']
                return {
                    'type': 'event',
                    'name': event_name,
                    'channel_set': spec['channel_set'],
                    'base_frequency': float(cfg.base_frequency),
                    'base_amplitude': float(cfg.base_amplitude),
                    'base_pulses': int(cfg.base_pulses)
                }

        channel_set = (
            self.config.reward_positive_channel_set
            if positive else self.config.reward_negative_channel_set
        )
        if channel_set is None:
            return None
        return {
            'type': 'reward',
            'name': 'reward_positive' if positive else 'reward_negative',
            'channel_set': channel_set,
            'base_frequency': float(
                self.config.feedback_episode_positive_frequency
                if positive else self.config.feedback_episode_negative_frequency
            ),
            'base_amplitude': float(
                self.config.feedback_positive_amplitude
                if positive else self.config.feedback_negative_amplitude
            ),
            'base_pulses': int(
                self.config.feedback_episode_positive_pulses
                if positive else self.config.feedback_episode_negative_pulses
            )
        }

    def _init_event_feedback_specs(self):
        """Prepare event-to-channel mappings for hardware feedback."""
        self._event_feedback_specs = {}
        for name, cfg in self.config.event_feedback_settings.items():
            channel_set = None
            if self.config.use_hardware:
                channel_set = self.config.event_feedback_channel_sets.get(name)
            self._event_feedback_specs[name] = {
                'config': cfg,
                'channel_set': channel_set,
                'ema_pos': 0.0,
                'ema_neg': 0.0,
                'count': 0,
                'last_scales': (1.0, 1.0, 1.0)
            }

    def _ensure_feedback_plans(self, neurons: 'cl.Neurons'):
        """Create reusable episode-level feedback plans if not already set."""
        if self._feedback_plans_ready:
            return

        if getattr(self.config, 'use_reward_feedback', False):
            self._positive_episode_plan = None
            self._negative_episode_plan = None
            self._feedback_plans_ready = True
            return

        self._positive_episode_plan = None
        self._negative_episode_plan = None

        pos_target = self._resolve_episode_feedback_target(True)
        if pos_target and pos_target.get('channel_set') is not None:
            positive_plan = neurons.create_stim_plan()
            if self.config.all_channels_set is not None:
                positive_plan.interrupt(self.config.all_channels_set)
            positive_plan.stim(
                pos_target['channel_set'],
                cl.StimDesign(
                    self.config.phase1_duration,
                    -self.config.feedback_positive_amplitude,
                    self.config.phase2_duration,
                    self.config.feedback_positive_amplitude
                ),
                cl.BurstDesign(
                    self.config.feedback_episode_positive_pulses,
                    int(self.config.feedback_episode_positive_frequency)
                )
            )
            self._positive_episode_plan = positive_plan

        neg_target = self._resolve_episode_feedback_target(False)
        if neg_target and neg_target.get('channel_set') is not None:
            negative_plan = neurons.create_stim_plan()
            if self.config.all_channels_set is not None:
                negative_plan.interrupt(self.config.all_channels_set)
            negative_plan.stim(
                neg_target['channel_set'],
                cl.StimDesign(
                    self.config.phase1_duration,
                    -self.config.feedback_negative_amplitude,
                    self.config.phase2_duration,
                    self.config.feedback_negative_amplitude
                ),
                cl.BurstDesign(
                    self.config.feedback_episode_negative_pulses,
                    int(self.config.feedback_episode_negative_frequency)
                )
            )
            self._negative_episode_plan = negative_plan

        self._feedback_plans_ready = True

    def _update_surprise_scaling(self):
        """Resolve surprise scaling configuration into cached floats."""
        base_gain = float(getattr(self.config, 'feedback_surprise_gain', 0.5))
        base_max_scale = float(getattr(self.config, 'feedback_surprise_max_scale', 2.0))
        freq_gain_cfg = getattr(self.config, 'feedback_surprise_freq_gain', None)
        amp_gain_cfg = getattr(self.config, 'feedback_surprise_amp_gain', None)
        freq_max_cfg = getattr(self.config, 'feedback_surprise_freq_max_scale', None)
        amp_max_cfg = getattr(self.config, 'feedback_surprise_amp_max_scale', None)

        self._surprise_base_gain = base_gain
        self._surprise_base_max_scale = base_max_scale
        self._surprise_freq_gain = float(freq_gain_cfg if freq_gain_cfg is not None else base_gain)
        self._surprise_amp_gain = float(amp_gain_cfg if amp_gain_cfg is not None else base_gain)
        self._surprise_freq_max_scale = float(freq_max_cfg if freq_max_cfg is not None else base_max_scale)
        self._surprise_amp_max_scale = float(amp_max_cfg if amp_max_cfg is not None else base_max_scale)

    def _log_surprise_scaling_config(self):
        """Placeholder to keep backward compatibility after removing static config logs."""
        return

    def _clear_stim_caches(self):
        """Flush cached stimulation designs to release native resources."""
        if hasattr(self, 'policy') and hasattr(self.policy, 'clear_stim_cache'):
            self.policy.clear_stim_cache()
        self._feedback_stim_cache.clear()
        self._single_channel_cache.clear()
        gc.collect()

    def _get_single_channel_set(self, channel: int) -> 'cl.ChannelSet':
        if channel not in self._single_channel_cache:
            self._single_channel_cache[channel] = cl.ChannelSet(channel)
        return self._single_channel_cache[channel]

    def _get_feedback_stim(
        self,
        stim_type: str,
        freq_hz: int,
        amplitude_value: float,
        pulse_count: int
    ) -> Tuple['cl.StimDesign', 'cl.BurstDesign']:
        cache_key = (stim_type, freq_hz, round(float(amplitude_value), 4), pulse_count)
        if cache_key not in self._feedback_stim_cache:
            stim_design = cl.StimDesign(
                self.config.phase1_duration,
                -amplitude_value,
                self.config.phase2_duration,
                amplitude_value
            )
            burst_design = cl.BurstDesign(
                pulse_count,
                int(freq_hz)
            )
            self._feedback_stim_cache[cache_key] = (stim_design, burst_design)
        return self._feedback_stim_cache[cache_key]

    def _limit_scaled_amplitude(
        self,
        base_amplitude: float,
        amp_scale: float,
        limit: float = 2.5
    ) -> Tuple[float, float]:
        """Reduce base amplitude so scaled value never exceeds limit."""
        base = float(base_amplitude)
        scale = float(amp_scale)
        if scale <= 0.0:
            scaled = max(self.config.min_amplitude, 0.0)
            return base, scaled
        max_base = limit / scale
        if base > max_base:
            base = max_base
        scaled = base * scale
        scaled = min(limit, scaled)
        scaled = min(self.config.max_amplitude, scaled)
        scaled = max(self.config.min_amplitude, scaled)
        return base, scaled


    def _reward_feedback_scales(
        self,
        magnitude: float,
        baseline: float,
        is_positive: bool,
        conflict_weight: float
    ) -> Tuple[float, float, float]:
        """Convert TD magnitude into scaling factors for reward-based feedback."""
        self._update_surprise_scaling()
        if magnitude <= 0.0:
            return 1.0, 1.0, 1.0
        baseline = max(baseline, 1e-3)
        ratio = min(magnitude / baseline, self._surprise_base_max_scale)

        compression_k = float(getattr(self.config, 'surprise_compression_k', 1.0))
        freq_delta = self._surprise_freq_gain * (1.0 - math.exp(-compression_k * ratio)) * conflict_weight
        freq_delta_cap = max(0.0, self._surprise_freq_max_scale - 1.0)
        freq_delta = min(freq_delta, freq_delta_cap)
        if is_positive:
            freq_scale = 1.0 - freq_delta
        else:
            freq_scale = 1.0 + freq_delta
        freq_scale = max(0.5, min(self._surprise_freq_max_scale, freq_scale))

        amp_delta = self._surprise_amp_gain * (1.0 - math.exp(-compression_k * ratio))
        amp_scale = min(self._surprise_amp_max_scale, 1.0 + amp_delta)
        amp_scale = max(1.0, amp_scale)

        pulse_delta = self._surprise_freq_gain * (1.0 - math.exp(-compression_k * ratio))
        pulse_scale = min(self._surprise_base_max_scale, 1.0 + pulse_delta)
        pulse_scale = max(1.0, pulse_scale)
        return freq_scale, amp_scale, pulse_scale

    def _append_episode_log(
        self,
        episode_reward:  float,
        episode_length:  int,
        success_actions: int,
        fail_actions:    int,
        neutral_actions: int,
        kill_count:      float = 0.0
    ):
        if self.training_log_path is None:
            return
        total_feedback = success_actions + fail_actions + neutral_actions
        success_rate = (success_actions / total_feedback) if total_feedback > 0 else 0.0

        record = {
            "total_episodes":  self.total_episodes,
            "episode_reward":  episode_reward,
            "kill_count":      kill_count,
            "episode_length":  episode_length,
            "success_actions": success_actions,
            "fail_actions":    fail_actions,
            "neutral_actions": neutral_actions,
            "success_rate":    success_rate,
            "total_steps":     self.total_steps
        }
        with open(self.training_log_path, 'a', encoding='utf-8') as log_file:
            json.dump(record, log_file)
            log_file.write('\n')

    def _send_feedback_command(
        self,
        feedback_type: str,
        channels: List[int],
        frequency: int,
        amplitude: float,
        pulses: int,
        unpredictable: bool = False,
        event_name: str = ""
    ):
        """Send feedback stimulation command to CL1 via UDP."""
        if not self.feedback_socket:
            return

        try:
            packet = udp_protocol.pack_feedback_command(
                feedback_type,
                channels,
                frequency,
                amplitude,
                pulses,
                unpredictable,
                event_name
            )
            self.feedback_socket.sendto(packet, (self.config.cl1_host, self.config.cl1_feedback_port))
        except Exception as e:
            print(f"[WARNING] Failed to send feedback command: {e}")

    def _send_episode_feedback(self, episode_reward: float):
        """
        Send episode-level feedback stimulation via UDP.

        Args:
            episode_reward: Total reward for the completed episode
        """
        if not getattr(self.config, 'use_episode_feedback', True):
            print(f"  [DEBUG] Episode feedback disabled (use_episode_feedback={getattr(self.config, 'use_episode_feedback', 'NOT SET')})")
            return

        # Determine if episode is positive or negative
        if episode_reward == 0.0:
            print(f"  [DEBUG] Skipping episode feedback (reward=0.0)")
            return  # No feedback for neutral episodes

        positive = episode_reward > 0.0

        # Get channels for this episode type
        if positive:
            channels = list(self.config.reward_feedback_positive_channels)
            base_frequency = self.config.feedback_episode_positive_frequency
            base_amplitude = self.config.feedback_positive_amplitude
            base_pulses = self.config.feedback_episode_positive_pulses
            event_name = "episode_positive"
        else:
            channels = list(self.config.reward_feedback_negative_channels)
            base_frequency = self.config.feedback_episode_negative_frequency
            base_amplitude = self.config.feedback_negative_amplitude
            base_pulses = self.config.feedback_episode_negative_pulses
            event_name = "episode_negative"

        if not channels:
            print(f"  [DEBUG] No channels configured for {event_name} (channels={channels})")
            return  # No channels configured

        print(f"  [EPISODE FEEDBACK] Sending {event_name} (reward={episode_reward:.2f}, channels={channels})")

        # Calculate surprise scaling if enabled
        freq_scale = 1.0
        amp_scale = 1.0
        pulse_scale = 1.0

        if getattr(self.config, 'episode_feedback_surprise_scaling', True):
            surprise_magnitude = self._compute_episode_surprise_magnitude(episode_reward)
            freq_scale, amp_scale = self._episode_surprise_scales(surprise_magnitude)
            pulse_scale = freq_scale  # Pulse count scales with frequency

            # Log surprise metrics
            if self.writer is not None:
                self.writer.add_scalar('feedback/episode_surprise_magnitude', surprise_magnitude, self.total_episodes)
                self.writer.add_scalar('feedback/episode_freq_scale', freq_scale, self.total_episodes)
                self.writer.add_scalar('feedback/episode_amp_scale', amp_scale, self.total_episodes)

        # Calculate scaled values
        freq_hz = max(
            self.config.min_frequency,
            int(base_frequency * freq_scale)
        )
        freq_hz = min(freq_hz, self.config.max_frequency)

        _, amplitude = self._limit_scaled_amplitude(base_amplitude, amp_scale)

        pulses = max(1, int(base_pulses * pulse_scale))

        # Send interrupt first for negative feedback
        if not positive:
            self._send_feedback_command(
                "interrupt",
                channels,
                0, 0.0, 0,
                unpredictable=False,
                event_name=event_name
            )

        # Send episode feedback stimulation
        self._send_feedback_command(
            "reward",  # Use "reward" type for episode feedback
            channels,
            freq_hz,
            amplitude,
            pulses,
            unpredictable=(not positive),  # Negative episodes use unpredictable mode
            event_name=event_name
        )

        print(f"  [EPISODE FEEDBACK] Sent {event_name}: {freq_hz}Hz, {amplitude:.2f}μA, {pulses} pulses")

        # Log to TensorBoard
        if self.writer is not None:
            tag_base = f'feedback/episode/{event_name}'
            self.writer.add_scalar(f'{tag_base}/frequency_hz', freq_hz, self.total_episodes)
            self.writer.add_scalar(f'{tag_base}/amplitude_ua', amplitude, self.total_episodes)
            self.writer.add_scalar(f'{tag_base}/pulses', pulses, self.total_episodes)
            if getattr(self.config, 'episode_feedback_surprise_scaling', True):
                self.writer.add_scalar(f'{tag_base}/freq_scale', freq_scale, self.total_episodes)
                self.writer.add_scalar(f'{tag_base}/amp_scale', amp_scale, self.total_episodes)

    def _track_action_feedback(
        self,
        info: Dict[str, Any],
        td_error: float,
        surprise_baseline: float,
        reward: float
    ):
        """Track action feedback and send stimulation commands via UDP (for UDP mode)."""
        if getattr(self.config, 'use_reward_feedback', False):
            # Handle reward-based feedback
            if reward == 0.0:
                return

            positive = reward > 0.0
            magnitude = max(td_error, 0.0) if positive else max(-td_error, 0.0)
            if magnitude <= 0.0:
                magnitude = abs(reward)

            td_sign_conflict = (td_error < 0.0 and positive) or (td_error > 0.0 and not positive)
            conflict_weight = 1.0 / (1.0 + abs(td_error)) if td_sign_conflict else 1.0

            freq_scale, amp_scale, pulse_scale = self._reward_feedback_scales(
                magnitude,
                surprise_baseline,
                positive,
                conflict_weight
            )

            if not positive:
                # Negative feedback
                neg_channels = list(self.config.reward_feedback_negative_channels)
                if not neg_channels:
                    neg_channels = list(self.config.encoding_channels)

                freq_hz = max(
                    self.config.min_frequency,
                    int(self.config.feedback_negative_frequency * freq_scale)
                )
                freq_hz = min(freq_hz, self.config.max_frequency)

                _, amplitude = self._limit_scaled_amplitude(
                    self.config.feedback_negative_amplitude,
                    amp_scale
                )

                pulses = max(1, int(self.config.feedback_negative_pulses * pulse_scale))

                # Send interrupt first for negative feedback
                if neg_channels:
                    self._send_feedback_command(
                        "interrupt",
                        neg_channels,
                        0, 0.0, 0,
                        unpredictable=False,
                        event_name="reward_negative"
                    )

                # Send feedback stimulation
                self._send_feedback_command(
                    "reward",
                    neg_channels,
                    freq_hz,
                    amplitude,
                    pulses,
                    unpredictable=True,  # Negative feedback uses unpredictable mode
                    event_name="reward_negative"
                )

                self.unsuccessful_actions += 1
                self.episode_fail_actions += 1
                self.negative_feedback_count += 1
                self.total_feedback_actions += 1
                self.last_negative_freq_scale = float(freq_scale)
                self.last_negative_amp_scale = float(amp_scale)
                self.negative_freq_scale_sum += freq_scale
                self.negative_amp_scale_sum += amp_scale
                self.negative_freq_scale_count += 1
                self.negative_amp_scale_count += 1

                if self.writer is not None:
                    self.writer.add_scalar('feedback/reward_negative/freq_scale', freq_scale, self.total_steps)
                    self.writer.add_scalar('feedback/reward_negative/amp_scale', amp_scale, self.total_steps)
            else:
                # Positive feedback
                pos_channels = list(self.config.reward_feedback_positive_channels)
                if not pos_channels:
                    return

                freq_hz = max(
                    self.config.min_frequency,
                    int(self.config.feedback_positive_frequency * freq_scale)
                )
                freq_hz = min(freq_hz, self.config.max_frequency)

                _, amplitude = self._limit_scaled_amplitude(
                    self.config.feedback_positive_amplitude,
                    amp_scale
                )

                pulses = max(1, int(self.config.feedback_positive_pulses * pulse_scale))

                # Send feedback stimulation
                self._send_feedback_command(
                    "reward",
                    pos_channels,
                    freq_hz,
                    amplitude,
                    pulses,
                    unpredictable=False,
                    event_name="reward_positive"
                )

                self.successful_actions += 1
                self.episode_success_actions += 1
                self.positive_feedback_count += 1
                self.total_feedback_actions += 1
                self.last_positive_freq_scale = float(freq_scale)
                self.last_positive_amp_scale = float(amp_scale)
                self.positive_freq_scale_sum += freq_scale
                self.positive_amp_scale_sum += amp_scale
                self.positive_freq_scale_count += 1
                self.positive_amp_scale_count += 1

                if self.writer is not None:
                    self.writer.add_scalar('feedback/reward_positive/freq_scale', freq_scale, self.total_steps)
                    self.writer.add_scalar('feedback/reward_positive/amp_scale', amp_scale, self.total_steps)
            return

        # Track event-based feedback
        events_triggered = []
        for name, spec in self._event_feedback_specs.items():
            cfg = spec['config']
            if info.get(cfg.info_key, False):
                events_triggered.append((name, spec))

        if not events_triggered:
            return

        delivered_count = 0
        for name, spec in events_triggered:
            cfg: EventFeedbackConfig = spec['config']
            magnitude, td_sign = self._event_td_component(cfg, td_error)
            if magnitude <= 0.0:
                continue

            freq_scale, amp_scale, pulse_scale = self._compute_event_feedback_scales(
                name,
                magnitude,
                cfg,
                surprise_baseline,
                td_sign
            )

            # Calculate stimulation parameters
            freq_hz = max(
                self.config.min_frequency,
                int(cfg.base_frequency * freq_scale)
            )
            freq_hz = min(freq_hz, self.config.max_frequency)

            _, amplitude = self._limit_scaled_amplitude(cfg.base_amplitude, amp_scale)
            pulse_count = max(1, int(cfg.base_pulses * pulse_scale))

            # Get channels for this event
            event_channels = list(cfg.channels)

            # Send interrupt for negative feedback
            if td_sign == 'negative' and event_channels:
                self._send_feedback_command(
                    "interrupt",
                    event_channels,
                    0, 0.0, 0,
                    unpredictable=False,
                    event_name=name
                )

            # Send event feedback stimulation
            if event_channels:
                self._send_feedback_command(
                    "event",
                    event_channels,
                    freq_hz,
                    amplitude,
                    pulse_count,
                    unpredictable=getattr(cfg, 'unpredictable', False),
                    event_name=name
                )

            # Track statistics
            delivered_count += 1
            self.surprise_sum += magnitude
            self.surprise_count += 1

            if td_sign == 'negative':
                self.unsuccessful_actions += 1
                self.negative_feedback_count += 1
                self.episode_fail_actions += 1
                self.last_negative_freq_scale = float(freq_scale)
                self.last_negative_amp_scale = float(amp_scale)
                self.negative_freq_scale_sum += freq_scale
                self.negative_amp_scale_sum += amp_scale
                self.negative_freq_scale_count += 1
                self.negative_amp_scale_count += 1
            else:
                self.successful_actions += 1
                self.positive_feedback_count += 1
                self.episode_success_actions += 1
                self.last_positive_freq_scale = float(freq_scale)
                self.last_positive_amp_scale = float(amp_scale)
                self.positive_freq_scale_sum += freq_scale
                self.positive_amp_scale_sum += amp_scale
                self.positive_freq_scale_count += 1
                self.positive_amp_scale_count += 1

            if self.writer is not None:
                tag_base = f'feedback/event/{name}'
                self.writer.add_scalar(f'{tag_base}/magnitude', magnitude, self.total_steps)
                self.writer.add_scalar(f'{tag_base}/freq_scale', freq_scale, self.total_steps)
                self.writer.add_scalar(f'{tag_base}/amp_scale', amp_scale, self.total_steps)
                self.writer.add_scalar(f'{tag_base}/pulse_scale', pulse_scale, self.total_steps)
                self.writer.add_scalar(f'{tag_base}/frequency', freq_hz, self.total_steps)
                self.writer.add_scalar(f'{tag_base}/amplitude', amplitude, self.total_steps)

        if delivered_count == 0:
            self.neutral_actions += 1
            self.episode_neutral_actions += 1
        else:
            self.total_feedback_actions += delivered_count

    def _event_td_component(
        self,
        cfg: EventFeedbackConfig,
        td_error: float
    ) -> Tuple[float, str]:
        """Select the TD-error component and sign to scale a feedback event."""
        if cfg.td_sign == 'negative':
            magnitude = max(-td_error, 0.0)
            return magnitude, 'negative'
        if cfg.td_sign == 'absolute':
            if td_error >= 0.0:
                return td_error, 'positive'
            return max(-td_error, 0.0), 'negative'
        magnitude = max(td_error, 0.0)
        return magnitude, 'positive'

    def _compute_event_feedback_scales(
        self,
        event_name: str,
        magnitude: float,
        cfg: EventFeedbackConfig,
        surprise_baseline: float,
        sign: str
    ) -> Tuple[float, float, float]:
        """Compute frequency, amplitude, and pulse scaling for a feedback event."""
        stats = self._event_feedback_specs[event_name]
        ema_key = 'ema_pos' if sign != 'negative' else 'ema_neg'
        beta = cfg.ema_beta
        stats[ema_key] = beta * stats[ema_key] + (1.0 - beta) * magnitude

        baseline_default = max(surprise_baseline, 1e-3)
        baseline = max(stats[ema_key], baseline_default)
        ratio = (magnitude / baseline) if magnitude > 0.0 else 0.0

        compression_k = float(getattr(self.config, 'surprise_compression_k', 1.0))

        def _delta(max_delta: float) -> float:
            if max_delta <= 0.0 or ratio <= 0.0:
                return 0.0
            return max_delta * (1.0 - math.exp(-compression_k * ratio))

        freq_delta = _delta(cfg.freq_gain)
        if sign == 'negative':
            freq_scale = 1.0 + freq_delta
        else:
            freq_scale = 1.0 - freq_delta
        freq_scale = max(0.5, min(cfg.freq_max_scale, freq_scale))

        amp_delta = _delta(cfg.amp_gain)
        amp_scale = min(cfg.amp_max_scale, 1.0 + amp_delta)
        amp_scale = max(1.0, amp_scale)

        pulse_delta = _delta(cfg.pulse_gain)
        pulse_scale = min(cfg.pulse_max_scale, 1.0 + pulse_delta)
        pulse_scale = max(1.0, pulse_scale)
        return freq_scale, amp_scale, pulse_scale

    def _compute_episode_surprise_magnitude(self, episode_reward: float) -> float:
        """Measure episode-level surprise from value prediction error."""
        value_pred = self._episode_value_prediction
        self._episode_value_prediction = None
        if value_pred is None:
            return 0.0

        magnitude = abs(float(episode_reward) - float(value_pred))
        beta = getattr(self, '_episode_surprise_beta', 0.9)
        self._episode_surprise_ema = (
            beta * self._episode_surprise_ema
            + (1.0 - beta) * magnitude
        )
        self._last_episode_surprise = magnitude
        return magnitude

    def _episode_surprise_scales(self, magnitude: float) -> Tuple[float, float]:
        """Convert episode surprise magnitude into frequency and amplitude scaling."""
        self._update_surprise_scaling()
        baseline = max(self._episode_surprise_ema, 1e-3)
        if magnitude <= 0.0:
            ratio = 0.0
        else:
            ratio = min(magnitude / baseline, self._surprise_base_max_scale)

        freq_scale = max(
            0.5,
            min(self._surprise_freq_max_scale, 1.0 + self._surprise_freq_gain * ratio)
        )
        amp_scale = max(
            0.5,
            min(self._surprise_amp_max_scale, 1.0 + self._surprise_amp_gain * ratio)
        )
        return freq_scale, amp_scale

    # MARK: main function
    def _collect_rollouts_hardware(
        self,
        num_steps: int,
        tick_frequency_hz: int
    ) -> Dict[str, torch.Tensor]:
        """
        Collect rollouts using CL1 hardware via UDP communication.

        Args:
            num_steps: Number of steps to collect
            tick_frequency_hz: Frequency to run the game loop

        Returns:
            Dictionary of collected data including spike_features
        """
        observations = []
        spike_features_list = []
        forward_actions = []
        strafe_actions = []
        camera_actions = []
        attack_actions = []
        speed_actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        stim_frequencies_list = []
        stim_amplitudes_list = []
        encoder_log_probs = []
        encoder_entropies = []
        self.successful_actions = 0
        self.unsuccessful_actions = 0
        self.neutral_actions = 0
        self.total_feedback_actions = 0
        self.positive_feedback_count = 0
        self.negative_feedback_count = 0
        self.surprise_sum = 0.0
        self.surprise_count = 0
        self._surprise_pos_ema = 0.0
        self._surprise_neg_ema = 0.0
        self.last_positive_freq_scale = 1.0
        self.last_positive_amp_scale = 1.0
        self.last_negative_freq_scale = 1.0
        self.last_negative_amp_scale = 1.0
        self.positive_freq_scale_sum = 0.0
        self.positive_amp_scale_sum = 0.0
        self.negative_freq_scale_sum = 0.0
        self.negative_amp_scale_sum = 0.0
        self.positive_freq_scale_count = 0
        self.positive_amp_scale_count = 0
        self.negative_freq_scale_count = 0
        self.negative_amp_scale_count = 0
        self.episode_success_actions = 0
        self.episode_fail_actions = 0
        self.episode_neutral_actions = 0
        self._episode_value_prediction = None

        obs = self.env.reset()
        step_count = 0
        game_reward_sum = 0.0
        quota_reached = False
        finished_collection = False

        print(f"\nCollecting at least {num_steps} steps with CL1 hardware via UDP (finishing current episode)...")

        # Timing loop - no chunking needed for UDP
        tick_interval = 1.0 / tick_frequency_hz  # seconds per tick
        collection_start_time = time.time()
        next_tick_time = collection_start_time

        try:
            while step_count < num_steps or not quota_reached:
                # Check if we've collected enough steps
                if quota_reached and info.get('episode_reward') is not None:
                    finished_collection = True
                    break

                # Get observation tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                # Get encoder outputs and value estimate
                with torch.no_grad():
                    frequencies, amplitudes, enc_log_prob_tensor, enc_entropy_tensor = self.policy.sample_encoder(obs_tensor)
                    value = self.policy.value_net(obs_tensor)

                freq_np = frequencies[0].cpu().numpy()
                amp_np = amplitudes[0].cpu().numpy()

                # Send stimulation command via UDP
                self.policy.apply_stimulation(self.stim_socket, freq_np, amp_np)

                # Wait for tick timing
                current_time = time.time()
                sleep_time = next_tick_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_tick_time += tick_interval

                # Receive spikes from CL1 via UDP
                spike_counts = self.policy.collect_spikes(self.spike_socket)
                spike_counts = self.policy.ablate_spike_features_numpy(spike_counts)

                # Decode spikes to action
                spike_tensor = torch.FloatTensor(spike_counts).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    (
                        forward_action,
                        strafe_action,
                        camera_action,
                        attack_action,
                        speed_action,
                        log_prob,
                        entropy
                    ) = self.policy.decode_spikes_to_action(spike_tensor)

                action_tuple = (
                    int(forward_action.item()),
                    int(strafe_action.item()),
                    int(camera_action.item()),
                    int(attack_action.item()),
                    int(speed_action.item())
                )

                # Execute action in VizDoom
                # MARK: Get obs from game environment
                next_obs, reward, done, info, screen_buffer = self.env.step(action_tuple)
                game_reward_sum += info.get('game_reward', 0.0)

                # Store data (including spike_features for policy update)
                observations.append(obs) # MARK: append obs
                spike_features_list.append(spike_counts)
                forward_actions.append(int(forward_action.item()))
                strafe_actions.append(int(strafe_action.item()))
                camera_actions.append(int(camera_action.item()))
                attack_actions.append(int(attack_action.item()))
                speed_actions.append(int(speed_action.item()))
                rewards.append(reward)
                dones.append(done)
                value_scalar = float(value.item())
                if self._episode_value_prediction is None:
                    self._episode_value_prediction = value_scalar
                values.append(value_scalar)
                encoder_log_prob_value = float(enc_log_prob_tensor.reshape(-1)[0].item())
                encoder_entropy_value = float(enc_entropy_tensor.reshape(-1)[0].item() / max(1, self.policy.num_channel_sets))
                stim_frequencies_list.append(freq_np)
                stim_amplitudes_list.append(amp_np)

                total_log_prob = float(log_prob.item()) + encoder_log_prob_value
                log_probs.append(total_log_prob)
                encoder_log_probs.append(encoder_log_prob_value)
                encoder_entropies.append(encoder_entropy_value)

                next_value_scalar = 0.0
                if not done:
                    next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        next_value_scalar = float(self.policy.value_net(next_obs_tensor).item())

                td_error = reward + self.config.gamma * next_value_scalar - value_scalar
                baseline = max(self._surprise_ema, 1e-3)
                magnitude = abs(td_error)
                self._surprise_ema = (
                    self._surprise_beta * self._surprise_ema
                    + (1 - self._surprise_beta) * magnitude
                )

                # Track action feedback (without applying neural stimulation in UDP mode)
                self._track_action_feedback(
                    info,
                    td_error,
                    baseline,
                    reward
                )

                # NOTE: (2025-11-28) Update image stream using screen buffer
                if hasattr(self, "image_stream") and screen_buffer is not None:
                    screen_buffer = screen_buffer.transpose(1, 2, 0) # (channels, H, W) -> (H, W, channels)
                    self.image_stream.update(screen_buffer)

                # Track episodes
                if info['episode_reward'] is not None:
                    # NOTE: (2025-11-19, jz/al) only gets here if "done" in VisDoom.step() and done = self.game.is_episode_finished()
                    episode_reward = info['episode_reward']
                    episode_length = info['episode_length']
                    episode_killcount = info.get('killcount_final')
                    if episode_killcount is None:
                        episode_killcount = 0.0
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.episode_killcounts.append(episode_killcount)
                    self.total_episodes += 1 # MARK: increase episode
                    kill_count = self.env.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)
                    print(
                        f"  Episode {self.total_episodes} finished: Reward={episode_reward:.2f} "
                        f"(Game={game_reward_sum:.2f}) | Length={episode_length} | Killcount={kill_count}"
                    )

                    # Send event metadata to CL1 for datastream logging
                    if self.event_socket:
                        try:
                            success_rate = (
                                self.successful_actions / self.total_feedback_actions * 100.0
                                if self.total_feedback_actions > 0 else 0.0
                            )
                            event_data = {
                                "episode": self.total_episodes,
                                "episode_length": episode_length,
                                "episode_reward": episode_reward,
                                "kill_count": kill_count,
                                "game_reward_sum": game_reward_sum,
                                "killcount": episode_killcount,
                                "episode_actions": {
                                    "success": self.episode_success_actions,
                                    "fail": self.episode_fail_actions,
                                    "neutral": self.episode_neutral_actions
                                },
                                "success_rate": success_rate
                            }
                            event_packet = udp_protocol.pack_event_metadata("episode_end", event_data)
                            self.event_socket.sendto(event_packet, (self.config.cl1_host, self.config.cl1_event_port))
                        except Exception as e:
                            print(f"  [WARNING] Failed to send event metadata: {e}")

                    # Send episode-level feedback via UDP
                    self._send_episode_feedback(episode_reward)

                    self._append_episode_log(
                        episode_reward,
                        episode_length,
                        self.episode_success_actions,
                        self.episode_fail_actions,
                        self.episode_neutral_actions,
                        kill_count
                    )
                    armor_dist = info.get('armor_distance_final')
                    if armor_dist is not None and self.writer is not None:
                        self.writer.add_scalar('Armor/distance_final', float(armor_dist), self.total_episodes)
                    if self.writer is not None:
                        self.writer.add_scalar('Reward/episode', episode_reward, self.total_episodes)
                        self.writer.add_scalar('Episode/length', episode_length, self.total_episodes)
                        self.writer.add_scalar('Reward/game_sum', game_reward_sum, self.total_episodes)
                        self.writer.add_scalar('Killcount/episode', episode_killcount, self.total_episodes)
                    self.episode_success_actions = 0
                    self.episode_fail_actions = 0
                    self.episode_neutral_actions = 0
                    self._episode_value_prediction = None
                    game_reward_sum = 0.0

                step_count += 1
                self.total_steps += 1

                if not quota_reached and step_count >= num_steps:
                    quota_reached = True

                if not done:
                    obs = next_obs
                else:
                    obs = self.env.reset()
                    self._episode_value_prediction = None
                    if quota_reached:
                        finished_collection = True
                        break

                if step_count % 100 == 0 and not quota_reached:
                    # Calculate actual collection rate
                    elapsed_since_start = time.time() - collection_start_time
                    actual_hz = step_count / elapsed_since_start if elapsed_since_start > 0 else 0

                    print(
                        f"  Collected {step_count}/{num_steps} steps ({step_count*100//num_steps}%) | "
                        f"Rate: {actual_hz:.1f} Hz | Episodes: {self.total_episodes}"
                    )

                if quota_reached and finished_collection:
                    break

        except KeyboardInterrupt:
            print("\n  Collection interrupted by user")
            if len(observations) == 0:
                raise
        except Exception as e:
            traceback.print_exc()
            print(f"  Warning: UDP collection error at step {step_count}: {e}")
            if len(observations) == 0:
                raise

        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values + [0.0], dones + [True])

        return {
            'observations': torch.from_numpy(np.asarray(observations, dtype=np.float32)).to(self.device),
            'spike_features': torch.from_numpy(np.asarray(spike_features_list, dtype=np.float32)).to(self.device),
            'forward_actions': torch.from_numpy(np.asarray(forward_actions, dtype=np.int64)).to(self.device),
            'strafe_actions': torch.from_numpy(np.asarray(strafe_actions, dtype=np.int64)).to(self.device),
            'camera_actions': torch.from_numpy(np.asarray(camera_actions, dtype=np.int64)).to(self.device),
            'attack_actions': torch.from_numpy(np.asarray(attack_actions, dtype=np.int64)).to(self.device),
            'speed_actions': torch.from_numpy(np.asarray(speed_actions, dtype=np.int64)).to(self.device),
            'old_log_probs': torch.from_numpy(np.asarray(log_probs, dtype=np.float32)).to(self.device),
            'advantages': torch.from_numpy(advantages.astype(np.float32)).to(self.device),
            'returns': torch.from_numpy(returns.astype(np.float32)).to(self.device),
            'stim_frequencies': torch.from_numpy(np.asarray(stim_frequencies_list, dtype=np.float32)).to(self.device),
            'stim_amplitudes': torch.from_numpy(np.asarray(stim_amplitudes_list, dtype=np.float32)).to(self.device),
            'encoder_log_probs': torch.from_numpy(np.asarray(encoder_log_probs, dtype=np.float32)).to(self.device),
            'encoder_entropies': torch.from_numpy(np.asarray(encoder_entropies, dtype=np.float32)).to(self.device)
        }

    def _collect_rollouts_vectorized(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """
        Collect rollouts from multiple VizDoom instances running in parallel.
        """
        if self.remotes is None or self.vector_obs is None:
            raise RuntimeError("Vectorized environments not initialized.")

        observations: List[np.ndarray] = []
        spike_features_list: List[np.ndarray] = []
        forward_list: List[np.ndarray] = []
        strafe_list: List[np.ndarray] = []
        camera_list: List[np.ndarray] = []
        attack_list: List[np.ndarray] = []
        speed_list: List[np.ndarray] = []
        log_probs_list: List[np.ndarray] = []
        rewards_list: List[np.ndarray] = []
        dones_list: List[np.ndarray] = []
        values_list: List[np.ndarray] = []
        valid_masks: List[np.ndarray] = []
        stim_freq_list: List[np.ndarray] = []
        stim_amp_list: List[np.ndarray] = []
        encoder_log_prob_list: List[np.ndarray] = []
        encoder_entropy_list: List[np.ndarray] = []

        iterations = 0
        quota_reached = False
        finish_mask = np.ones(self.config.num_envs, dtype=bool)
        collecting = True

        while collecting:
            obs = self.vector_obs
            obs_tensor = torch.from_numpy(obs).float().to(self.device)

            with torch.no_grad():
                frequencies, amplitudes, enc_log_prob, enc_entropy = self.policy.sample_encoder(obs_tensor)
                spike_tensor = self.policy.ablate_spike_features_tensor(frequencies)
                (
                    forward_actions,
                    strafe_actions,
                    camera_actions,
                    attack_actions,
                    speed_actions,
                    log_probs,
                    _
                ) = self.policy.decode_spikes_to_action(spike_tensor)
                values = self.policy.value_net(obs_tensor).squeeze(-1)

            for env_idx, remote in enumerate(self.remotes):
                action_tuple = (
                    int(forward_actions[env_idx].item()),
                    int(strafe_actions[env_idx].item()),
                    int(camera_actions[env_idx].item()),
                    int(attack_actions[env_idx].item()),
                    int(speed_actions[env_idx].item())
                )
                remote.send(('step', action_tuple))

            results = [remote.recv() for remote in self.remotes]

            next_obs = np.stack([result[0] for result in results]).astype(np.float32)
            rewards = np.array([result[1] for result in results], dtype=np.float32)
            dones = np.array([result[2] for result in results], dtype=np.float32)
            infos = [result[3] for result in results]

            observations.append(obs.copy())
            spike_features_list.append(spike_tensor.cpu().numpy())
            forward_list.append(forward_actions.cpu().numpy())
            strafe_list.append(strafe_actions.cpu().numpy())
            camera_list.append(camera_actions.cpu().numpy())
            attack_list.append(attack_actions.cpu().numpy())
            speed_list.append(speed_actions.cpu().numpy())
            total_log_prob = (log_probs + enc_log_prob).cpu().numpy()
            log_probs_list.append(total_log_prob)
            rewards_list.append(rewards)
            dones_list.append(dones)
            values_list.append(values.cpu().numpy())
            stim_freq_list.append(frequencies.cpu().numpy())
            stim_amp_list.append(amplitudes.cpu().numpy())
            encoder_log_prob_list.append(enc_log_prob.cpu().numpy())
            encoder_entropy_list.append((enc_entropy / max(1, self.policy.num_channel_sets)).cpu().numpy())
            record_mask = finish_mask if quota_reached else np.ones(self.config.num_envs, dtype=bool)
            valid_masks.append(record_mask.astype(bool).copy())

            self.vector_obs = next_obs
            self.total_steps += self.config.num_envs

            if self.vector_reward_sums is not None:
                self.vector_reward_sums += rewards
            if self.vector_episode_lengths is not None:
                self.vector_episode_lengths += 1

            for idx, info in enumerate(infos):
                if info.get('episode_reward') is not None:
                    episode_reward = info['episode_reward']
                    episode_length = info.get(
                        'episode_length',
                        int(self.vector_episode_lengths[idx]) if self.vector_episode_lengths is not None else 0
                    )
                    episode_killcount = info.get('killcount_final')
                    if episode_killcount is None:
                        episode_killcount = 0.0
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.episode_killcounts.append(episode_killcount)
                    self.total_episodes += 1

                    if self.vector_reward_sums is not None:
                        self.vector_reward_sums[idx] = 0.0
                    if self.vector_episode_lengths is not None:
                        self.vector_episode_lengths[idx] = 0

                    self._append_episode_log(
                        episode_reward,
                        episode_length,
                        0,
                        0,
                        0
                    )
                    if self.writer is not None:
                        self.writer.add_scalar('Killcount/episode', episode_killcount, self.total_episodes)

            iterations += 1
            if not quota_reached and iterations >= num_steps:
                quota_reached = True
                finish_mask = np.ones(self.config.num_envs, dtype=bool)

            if quota_reached:
                finish_mask = np.logical_and(finish_mask, np.logical_not(dones.astype(bool)))
                if not finish_mask.any():
                    collecting = False
            else:
                finish_mask = np.ones(self.config.num_envs, dtype=bool)

        rewards_arr = np.stack(rewards_list)  # (iterations, num_envs)
        values_arr = np.stack(values_list)
        dones_arr = np.stack(dones_list)
        obs_arr = np.stack(observations)
        spike_arr = np.stack(spike_features_list)
        forward_arr = np.stack(forward_list)
        strafe_arr = np.stack(strafe_list)
        camera_arr = np.stack(camera_list)
        attack_arr = np.stack(attack_list)
        speed_arr = np.stack(speed_list)
        log_probs_arr = np.stack(log_probs_list)
        valid_arr = np.stack(valid_masks).astype(bool)
        stim_freq_arr = np.stack(stim_freq_list)
        stim_amp_arr = np.stack(stim_amp_list)
        encoder_log_prob_arr = np.stack(encoder_log_prob_list)
        encoder_entropy_arr = np.stack(encoder_entropy_list)

        obs_samples: List[np.ndarray] = []
        spike_samples: List[np.ndarray] = []
        forward_samples: List[np.ndarray] = []
        strafe_samples: List[np.ndarray] = []
        camera_samples: List[np.ndarray] = []
        attack_samples: List[np.ndarray] = []
        speed_samples: List[np.ndarray] = []
        log_prob_samples: List[np.ndarray] = []
        advantage_samples: List[np.ndarray] = []
        return_samples: List[np.ndarray] = []
        stim_freq_samples: List[np.ndarray] = []
        stim_amp_samples: List[np.ndarray] = []
        encoder_log_prob_samples: List[np.ndarray] = []
        encoder_entropy_samples: List[np.ndarray] = []

        num_envs = self.config.num_envs
        for env_idx in range(num_envs):
            mask = valid_arr[:, env_idx]
            if not mask.any():
                continue

            env_obs = obs_arr[:, env_idx][mask]
            env_spike = spike_arr[:, env_idx][mask]
            env_forward = forward_arr[:, env_idx][mask]
            env_strafe = strafe_arr[:, env_idx][mask]
            env_camera = camera_arr[:, env_idx][mask]
            env_attack = attack_arr[:, env_idx][mask]
            env_speed = speed_arr[:, env_idx][mask]
            env_log_prob = log_probs_arr[:, env_idx][mask]
            env_stim_freq = stim_freq_arr[:, env_idx][mask]
            env_stim_amp = stim_amp_arr[:, env_idx][mask]
            env_enc_log_prob = encoder_log_prob_arr[:, env_idx][mask]
            env_enc_entropy = encoder_entropy_arr[:, env_idx][mask]
            env_rewards = rewards_arr[:, env_idx][mask]
            env_dones = dones_arr[:, env_idx][mask]
            env_values = values_arr[:, env_idx][mask]

            rewards_list_env = env_rewards.tolist()
            values_list_env = env_values.tolist() + [0.0]
            dones_list_env = env_dones.tolist() + [True]

            advantages_env, returns_env = self._compute_gae(
                rewards_list_env,
                values_list_env,
                dones_list_env
            )

            obs_samples.append(env_obs)
            spike_samples.append(env_spike)
            forward_samples.append(env_forward)
            strafe_samples.append(env_strafe)
            camera_samples.append(env_camera)
            attack_samples.append(env_attack)
            speed_samples.append(env_speed)
            log_prob_samples.append(env_log_prob)
            advantage_samples.append(advantages_env.astype(np.float32))
            return_samples.append(returns_env.astype(np.float32))
            stim_freq_samples.append(env_stim_freq.astype(np.float32))
            stim_amp_samples.append(env_stim_amp.astype(np.float32))
            encoder_log_prob_samples.append(env_enc_log_prob.astype(np.float32))
            encoder_entropy_samples.append(env_enc_entropy.astype(np.float32))

        if not obs_samples:
            raise RuntimeError("No rollout samples collected; check environment configuration.")

        obs_batch = np.concatenate(obs_samples, axis=0).astype(np.float32)
        spike_batch = np.concatenate(spike_samples, axis=0).astype(np.float32)
        forward_batch = np.concatenate(forward_samples, axis=0).astype(np.int64)
        strafe_batch = np.concatenate(strafe_samples, axis=0).astype(np.int64)
        camera_batch = np.concatenate(camera_samples, axis=0).astype(np.int64)
        attack_batch = np.concatenate(attack_samples, axis=0).astype(np.int64)
        speed_batch = np.concatenate(speed_samples, axis=0).astype(np.int64)
        log_probs_batch = np.concatenate(log_prob_samples, axis=0).astype(np.float32)
        advantages_batch = np.concatenate(advantage_samples, axis=0).astype(np.float32)
        returns_batch = np.concatenate(return_samples, axis=0).astype(np.float32)
        stim_freq_batch = np.concatenate(stim_freq_samples, axis=0).astype(np.float32)
        stim_amp_batch = np.concatenate(stim_amp_samples, axis=0).astype(np.float32)
        encoder_log_prob_batch = np.concatenate(encoder_log_prob_samples, axis=0).astype(np.float32)
        encoder_entropy_batch = np.concatenate(encoder_entropy_samples, axis=0).astype(np.float32)

        return {
            'observations': torch.from_numpy(obs_batch).to(self.device),
            'spike_features': torch.from_numpy(spike_batch).to(self.device),
            'forward_actions': torch.from_numpy(forward_batch).to(self.device),
            'strafe_actions': torch.from_numpy(strafe_batch).to(self.device),
            'camera_actions': torch.from_numpy(camera_batch).to(self.device),
            'attack_actions': torch.from_numpy(attack_batch).to(self.device),
            'speed_actions': torch.from_numpy(speed_batch).to(self.device),
            'old_log_probs': torch.from_numpy(log_probs_batch).to(self.device),
            'advantages': torch.from_numpy(advantages_batch).to(self.device),
            'returns': torch.from_numpy(returns_batch).to(self.device),
            'stim_frequencies': torch.from_numpy(stim_freq_batch).to(self.device),
            'stim_amplitudes': torch.from_numpy(stim_amp_batch).to(self.device),
            'encoder_log_probs': torch.from_numpy(encoder_log_prob_batch).to(self.device),
            'encoder_entropies': torch.from_numpy(encoder_entropy_batch).to(self.device)
        }

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation with explicit terminal value."""
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        num_steps = rewards.shape[0]
        advantages = np.zeros(num_steps, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(num_steps)):
            next_value = values[t + 1]
            mask = 1.0 - dones[t]

            delta = rewards[t] + self.config.gamma * next_value * mask - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]
        return advantages, returns

    def update_policy(self, rollout_data: Dict[str, torch.Tensor]):
        """
        Update policy using PPO clipped objective.

        Decoder parameters are always updated; encoder gradients are included only when
        `config.encoder_trainable` is enabled.
        """
        obs = rollout_data['observations']
        spike_features = rollout_data['spike_features']
        forward_actions = rollout_data['forward_actions']
        strafe_actions = rollout_data['strafe_actions']
        camera_actions = rollout_data['camera_actions']
        attack_actions = rollout_data['attack_actions']
        speed_actions = rollout_data['speed_actions']
        old_log_probs = rollout_data['old_log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        stim_frequencies = rollout_data['stim_frequencies']
        stim_amplitudes = rollout_data['stim_amplitudes']
        encoder_old_log_probs = rollout_data['encoder_log_probs']
        encoder_old_entropies = rollout_data['encoder_entropies']

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Multiple epochs of updates
        last_policy_loss = None
        last_value_loss = None
        last_entropy_loss = None
        clip_eps = self.config.clip_epsilon
        max_ratio_upper = 1 + clip_eps
        min_ratio_lower = 1 - clip_eps
        ratio_records = []
        clipped_counts = []
        kl_records = []
        encoder_ratio_records = []
        encoder_entropy_records = []

        for epoch in range(self.config.num_epochs):
            # Create mini-batches
            indices = torch.randperm(obs.size(0))

            for start in range(0, obs.size(0), self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]

                # Get current policy outputs
                # Only computes gradients through decoder (spike_features -> actions)
                log_probs, values, entropy, encoder_log_prob, encoder_entropy = self.policy.evaluate_actions(
                    spike_features[batch_idx],
                    forward_actions[batch_idx],
                    strafe_actions[batch_idx],
                    camera_actions[batch_idx],
                    attack_actions[batch_idx],
                    speed_actions[batch_idx],
                    obs[batch_idx],
                    stim_frequencies[batch_idx],
                    stim_amplitudes[batch_idx]
                )
                encoder_entropy_mean = encoder_entropy / max(1, self.policy.num_channel_sets)

                # PPO clipped objective
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * advantages[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                predicted_values = values.squeeze(-1)
                target_returns = returns[batch_idx]
                if target_returns.dim() > 1:
                    target_returns = target_returns.squeeze(-1)
                normalized_preds = predicted_values
                normalized_targets = target_returns
                if self.config.normalize_returns:
                    mean = target_returns.mean()
                    std = target_returns.std(unbiased=False)
                    if std < 1e-6:
                        std = torch.ones((), device=target_returns.device, dtype=target_returns.dtype)
                    normalized_preds = (predicted_values - mean) / std
                    normalized_targets = (target_returns - mean) / std
                value_loss = F.mse_loss(normalized_preds, normalized_targets)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                if (
                    self.config.decoder_weight_l2_coef > 0.0
                    or self.config.decoder_bias_l2_coef > 0.0
                ):
                    weight_l2_term, bias_l2_term = self.policy.decoder.l2_penalties(
                        effective=not self.policy.decoder.use_mlp
                    )
                    if self.config.decoder_weight_l2_coef > 0.0:
                        loss = loss + self.config.decoder_weight_l2_coef * weight_l2_term
                    if self.config.decoder_bias_l2_coef > 0.0:
                        loss = loss + self.config.decoder_bias_l2_coef * bias_l2_term

                if getattr(self.config, 'encoder_trainable', False) and self.config.encoder_entropy_coef > 0.0:
                    encoder_entropy_penalty = encoder_entropy_mean.mean() * self.config.encoder_entropy_coef
                    loss = loss - encoder_entropy_penalty

                # Optimize (only decoder and value_net get gradients)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                last_policy_loss = policy_loss
                last_value_loss = value_loss
                last_entropy_loss = entropy_loss

                with torch.no_grad():
                    ratio_records.append(ratio.detach().cpu())
                    clipped = ((ratio > max_ratio_upper) | (ratio < min_ratio_lower)).sum().item()
                    clipped_counts.append(clipped)
                    approx_kl = (old_log_probs[batch_idx] - log_probs).mean().item()
                    kl_records.append(approx_kl)
                    if getattr(self.config, 'encoder_trainable', False):
                        encoder_ratio = torch.exp(encoder_log_prob - encoder_old_log_probs[batch_idx])
                        encoder_ratio_records.append(encoder_ratio.detach().cpu())
                        encoder_entropy_records.append((encoder_entropy / max(1, self.policy.num_channel_sets)).detach().cpu())

        # Log metrics
        if last_policy_loss is not None:
            self.writer.add_scalar('Loss/policy', last_policy_loss.item(), self.total_steps)
        if last_value_loss is not None:
            self.writer.add_scalar('Loss/value', last_value_loss.item(), self.total_steps)
        if last_entropy_loss is not None:
            self.writer.add_scalar('Loss/entropy', last_entropy_loss.item(), self.total_steps)
        if adv_std is not None:
            self.writer.add_scalar('PPO/adv_std', float(adv_std), self.total_steps)
        if ratio_records:
            ratio_tensor = torch.cat(ratio_records).cpu()
            self.writer.add_scalar('PPO/ratio_mean', ratio_tensor.mean().item(), self.total_steps)
            self.writer.add_scalar('PPO/ratio_median', ratio_tensor.median().item(), self.total_steps)
            clipped_total = sum(clipped_counts)
            total_samples = ratio_tensor.numel()
            clipped_pct = 100.0 * clipped_total / max(total_samples, 1)
            self.writer.add_scalar('PPO/ratio_clipped_pct', clipped_pct, self.total_steps)
        if kl_records:
            self.writer.add_scalar('PPO/approx_kl', float(np.mean(kl_records)), self.total_steps)

        if encoder_ratio_records and getattr(self.config, 'encoder_trainable', False):
            enc_ratio_tensor = torch.cat(encoder_ratio_records).cpu()
            self.writer.add_scalar('Encoder/ratio_mean', enc_ratio_tensor.mean().item(), self.total_steps)
            self.writer.add_scalar('Encoder/ratio_median', enc_ratio_tensor.median().item(), self.total_steps)
        if encoder_entropy_records and getattr(self.config, 'encoder_trainable', False):
            enc_entropy_tensor = torch.cat(encoder_entropy_records).cpu()
            self.writer.add_scalar('Encoder/entropy_mean', enc_entropy_tensor.mean().item(), self.total_steps)
        if getattr(self.config, 'encoder_trainable', False) and encoder_old_log_probs.numel() > 0:
            self.writer.add_scalar('Encoder/old_log_prob_mean', encoder_old_log_probs.mean().item(), self.total_steps)
        if getattr(self.config, 'encoder_trainable', False) and encoder_old_entropies.numel() > 0:
            self.writer.add_scalar('Encoder/old_entropy_mean', encoder_old_entropies.mean().item(), self.total_steps)

        with torch.no_grad():
            weight_l2_eff, bias_l2_eff = self.policy.decoder.l2_penalties(
                effective=not self.policy.decoder.use_mlp
            )
        self.writer.add_scalar('Decoder/weight_l2_current', float(weight_l2_eff.item()), self.total_steps)
        self.writer.add_scalar('Decoder/bias_l2_current', float(bias_l2_eff.item()), self.total_steps)
        if self.config.decoder_weight_l2_coef > 0.0:
            self.writer.add_scalar(
                'Decoder/weight_l2_penalty',
                float(weight_l2_eff.item() * self.config.decoder_weight_l2_coef),
                self.total_steps
            )
        if self.config.decoder_bias_l2_coef > 0.0:
            self.writer.add_scalar(
                'Decoder/bias_l2_penalty',
                float(bias_l2_eff.item() * self.config.decoder_bias_l2_coef),
                self.total_steps
            )

        sample_features = spike_features
        if sample_features.size(0) > 1024:
            sample_features = sample_features[:1024]
        decoder_metrics = self.policy.decoder.compute_weight_bias_metrics(sample_features)
        for key, value in decoder_metrics.items():
            self.writer.add_scalar(key, value, self.total_steps)

        with torch.no_grad():
            value_predictions = self.policy.value_net(obs).squeeze(-1)
            returns_flat = returns if returns.dim() == 1 else returns.squeeze(-1)
            var_returns = torch.var(returns_flat, unbiased=False)
            if var_returns > 1e-8:
                residual_var = torch.var(returns_flat - value_predictions, unbiased=False)
                explained_variance = 1.0 - (residual_var / (var_returns + 1e-8))
            else:
                explained_variance = torch.tensor(0.0, device=value_predictions.device)
        self.writer.add_scalar('Value/explained_variance', explained_variance.item(), self.total_steps)

        if self.config.use_hardware:
            self.writer.add_scalar('Feedback/success_actions', self.successful_actions, self.total_steps)
            self.writer.add_scalar('Feedback/unsuccessful_actions', self.unsuccessful_actions, self.total_steps)
            self.writer.add_scalar('Feedback/neutral_actions', self.neutral_actions, self.total_steps)
            success_rate = (
                self.successful_actions / self.total_feedback_actions
                if self.total_feedback_actions > 0 else 0.0
            )
            self.writer.add_scalar('Feedback/success_rate', success_rate, self.total_steps)
            if self.surprise_count > 0:
                mean_surprise = self.surprise_sum / self.surprise_count
                self.writer.add_scalar('Feedback/surprise_mean', mean_surprise, self.total_steps)
            if self.positive_freq_scale_count > 0:
                mean_pos_freq = self.positive_freq_scale_sum / self.positive_freq_scale_count
                self.writer.add_scalar('Feedback/positive_freq_scale', mean_pos_freq, self.total_steps)
            if self.positive_amp_scale_count > 0:
                mean_pos_amp = self.positive_amp_scale_sum / self.positive_amp_scale_count
                self.writer.add_scalar('Feedback/positive_amp_scale', mean_pos_amp, self.total_steps)
            if self.negative_freq_scale_count > 0:
                mean_neg_freq = self.negative_freq_scale_sum / self.negative_freq_scale_count
                self.writer.add_scalar('Feedback/negative_freq_scale', mean_neg_freq, self.total_steps)
            if self.negative_amp_scale_count > 0:
                mean_neg_amp = self.negative_amp_scale_sum / self.negative_amp_scale_count
                self.writer.add_scalar('Feedback/negative_amp_scale', mean_neg_amp, self.total_steps)
            print(
                f"Feedback summary → Success: {self.successful_actions}, "
                f"Miss: {self.unsuccessful_actions}, Neutral: {self.neutral_actions}, "
                f"Success Rate: {success_rate * 100:.1f}%"
            )

        self._clear_stim_caches()

    def _log_progress(self):
        """Log mean reward and episode length statistics."""
        if len(self.episode_rewards) == 0:
            return

        mean_reward = np.mean(self.episode_rewards)
        mean_length = np.mean(self.episode_lengths)
        mean_killcount = np.mean(self.episode_killcounts) if self.episode_killcounts else 0.0

        self.writer.add_scalar('Reward/mean', mean_reward, self.total_episodes)
        self.writer.add_scalar('Episode/length', mean_length, self.total_episodes)
        if self.episode_killcounts:
            self.writer.add_scalar('Killcount/mean', mean_killcount, self.total_episodes)

        print(f"\n{'='*70}")
        kill_str = f" | Kills: {mean_killcount:.2f}" if self.episode_killcounts else ""
        print(f"Episodes: {self.total_episodes} | Reward: {mean_reward:.2f} | Length: {mean_length:.1f}{kill_str}")
        print(f"{'='*70}\n")

    def _maybe_save_checkpoint(self):
        """Persist a checkpoint whenever required intervals are crossed."""
        if self.config.save_interval <= 0 or self.total_episodes <= 0:
            return
        target_episode = (self.total_episodes // self.config.save_interval) * self.config.save_interval
        if target_episode > 0 and target_episode > self.last_checkpoint_episode:
            checkpoint_name = f"checkpoint_{target_episode}.pt"
            print(f"Saving checkpoint: {checkpoint_name}")
            self.last_checkpoint_episode = target_episode
            self.save_checkpoint(checkpoint_name)

    def train(self):
        print("\n" + "="*70)
        print(f"Steps per update: {self.config.steps_per_update}")
        print(f"Max episodes: {self.config.max_episodes}")
        print("="*70 + "\n")

        start_time = time.time()

        try:
            if self.config.use_hardware:
                # Setup UDP sockets for CL1 communication
                self.setup_udp_sockets()

                print("[SUCCESS] Connected to CL1 device via UDP")
                event_channels = []
                for cfg in self.config.event_feedback_settings.values():
                    event_channels.extend(cfg.channels)
                used_channels = (
                    self.config.encoding_channels
                    + self.config.move_forward_channels
                    + self.config.move_backward_channels
                    + self.config.move_left_channels
                    + self.config.move_right_channels
                    + self.config.turn_left_channels
                    + self.config.turn_right_channels
                    + self.config.attack_channels
                    + event_channels
                )
                used_channels = sorted(dict.fromkeys(used_channels))
                print(f"Using {len(used_channels)} channels: {used_channels[:20]}...")
                print("")

                # Recording and datastream handled on CL1 device side
                print("NOTE: Recording and datastream managed by CL1 device\n")

                # Setup image stream server for visualisation
                self.set_up_image_stream_server()

                while self.total_episodes < self.config.max_episodes:
                    rollout_data = self._collect_rollouts_hardware(
                        self.config.steps_per_update,
                        self.tick_frequency_hz
                    )

                    print("Updating policy...")
                    self.update_policy(rollout_data)
                    self._log_progress()
                    self._maybe_save_checkpoint()
                    if self.total_steps % 5000 == 0:
                        self.writer.flush()

                # Training complete - notify CL1 to stop recording
                if self.event_socket:
                    try:
                        completion_data = {
                            "total_episodes": self.total_episodes,
                            "total_steps": self.total_steps,
                            "reason": "max_episodes_reached"
                        }
                        event_packet = udp_protocol.pack_event_metadata("training_complete", completion_data)
                        self.event_socket.sendto(event_packet, (self.config.cl1_host, self.config.cl1_event_port))
                        print("\n[SUCCESS] Sent training completion signal to CL1")
                        time.sleep(0.5)  # Give CL1 time to process
                    except Exception as e:
                        print(f"\n[WARNING] Failed to send training completion signal: {e}")

            else:
                while self.total_episodes < self.config.max_episodes:
                    rollout_data = self._collect_rollouts_vectorized(self.config.steps_per_update)
                    self.update_policy(rollout_data)
                    self._log_progress()
                    self._maybe_save_checkpoint()
                    if self.total_steps % 5000 == 0:
                        self.writer.flush()


        finally:
            if self.config.use_hardware:
                # Close UDP sockets
                if self.stim_socket:
                    self.stim_socket.close()
                    print("[SUCCESS] Closed stimulation socket")
                if self.spike_socket:
                    self.spike_socket.close()
                    print("[SUCCESS] Closed spike socket")
                if self.event_socket:
                    self.event_socket.close()
                    print("[SUCCESS] Closed event socket")
                if self.feedback_socket:
                    self.feedback_socket.close()
                    print("[SUCCESS] Closed feedback socket")
                if hasattr(self, "image_stream"):
                    self.image_stream.close()
                    print("[SUCCESS] Closed image stream server")
                if self.env is not None:
                    self.env.close()
            else:
                self._close_vectorized_envs()

        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Total steps: {self.total_steps}")
        print(f"Total episodes: {self.total_episodes}")
        print(f"{'='*70}\n")

        # Save final model
        self.save_checkpoint("final_model.pt")
        self.writer.close()

    def watch(self):
        """
        NOTE: (2025-11-28, jz) Adapted from self.train() and removing policy updates
        """
        start_time = time.time()

        try:
            if self.config.use_hardware:
                # Setup UDP sockets for CL1 communication
                self.setup_udp_sockets()

                print("[SUCCESS] Connected to CL1 device via UDP")
                event_channels = []
                for cfg in self.config.event_feedback_settings.values():
                    event_channels.extend(cfg.channels)
                used_channels = (
                    self.config.encoding_channels
                    + self.config.move_forward_channels
                    + self.config.move_backward_channels
                    + self.config.move_left_channels
                    + self.config.move_right_channels
                    + self.config.turn_left_channels
                    + self.config.turn_right_channels
                    + self.config.attack_channels
                    + event_channels
                )
                used_channels = sorted(dict.fromkeys(used_channels))
                print(f"Using {len(used_channels)} channels: {used_channels[:20]}...")
                print("")

                # Recording and datastream handled on CL1 device side
                print("NOTE: Recording and datastream managed by CL1 device\n")

                # Setup image stream server for visualisation
                self.set_up_image_stream_server()

                while self.total_episodes < self.config.max_episodes:
                    rollout_data = self._collect_rollouts_hardware(
                        self.config.steps_per_update,
                        self.tick_frequency_hz
                    )

                    self._log_progress()
                    if self.total_steps % 5000 == 0:
                        self.writer.flush()

                # Training complete - notify CL1 to stop recording
                if self.event_socket:
                    try:
                        completion_data = {
                            "total_episodes": self.total_episodes,
                            "total_steps": self.total_steps,
                            "reason": "max_episodes_reached"
                        }
                        event_packet = udp_protocol.pack_event_metadata("training_complete", completion_data)
                        self.event_socket.sendto(event_packet, (self.config.cl1_host, self.config.cl1_event_port))
                        print("\n[SUCCESS] Sent training completion signal to CL1")
                        time.sleep(0.5)  # Give CL1 time to process
                    except Exception as e:
                        print(f"\n[WARNING] Failed to send training completion signal: {e}")

            else:
                while self.total_episodes < self.config.max_episodes:
                    rollout_data = self._collect_rollouts_vectorized(self.config.steps_per_update)
                    self._log_progress()
                    if self.total_steps % 5000 == 0:
                        self.writer.flush()

        finally:
            if self.config.use_hardware:
                # Close UDP sockets
                if self.stim_socket:
                    self.stim_socket.close()
                    print("[SUCCESS] Closed stimulation socket")
                if self.spike_socket:
                    self.spike_socket.close()
                    print("[SUCCESS] Closed spike socket")
                if self.event_socket:
                    self.event_socket.close()
                    print("[SUCCESS] Closed event socket")
                if self.feedback_socket:
                    self.feedback_socket.close()
                    print("[SUCCESS] Closed feedback socket")
                if hasattr(self, "image_stream"):
                    self.image_stream.close()
                    print("[SUCCESS] Closed image stream server")
                if self.env is not None:
                    self.env.close()
            else:
                self._close_vectorized_envs()

        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Eval completed!")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Total steps: {self.total_steps}")
        print(f"Total episodes: {self.total_episodes}")
        print(f"{'='*70}\n")

        # Save final model
        self.save_checkpoint("final_model.pt")
        self.writer.close()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'config': self.config,
            'log_dir': getattr(self, 'log_dir', self.config.log_dir),
            'last_checkpoint_episode': self.last_checkpoint_episode
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model and optimizer state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_episodes = int(checkpoint.get('total_episodes', self.total_episodes))
        saved_steps = int(checkpoint.get('total_steps', self.total_steps))
        self.total_episodes = max(self.total_episodes, saved_episodes)
        self.total_steps = max(self.total_steps, saved_steps)
        saved_last_checkpoint = checkpoint.get('last_checkpoint_episode')
        saved_config = checkpoint.get('config')
        saved_log_dir: Optional[str] = checkpoint.get('log_dir')
        if saved_config is not None:
            saved_checkpoint_dir = getattr(saved_config, 'checkpoint_dir', None)
            if saved_checkpoint_dir and saved_checkpoint_dir != self.config.checkpoint_dir:
                self.config.checkpoint_dir = saved_checkpoint_dir
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                self.training_log_path = os.path.join(self.config.checkpoint_dir, 'training_log.jsonl')
            if saved_log_dir is None:
                saved_log_dir = getattr(saved_config, 'log_dir', None)
        if saved_log_dir:
            if self.writer is not None:
                self.writer.close()
            os.makedirs(saved_log_dir, exist_ok=True)
            self.config.log_dir = saved_log_dir
            self.writer = SummaryWriter(saved_log_dir, purge_step=self.total_steps)
            self.log_dir = getattr(self.writer, 'log_dir', saved_log_dir)
            self._update_surprise_scaling()
            self._log_surprise_scaling_config()
        else:
            self.log_dir = getattr(self.writer, 'log_dir', self.config.log_dir)
        if saved_last_checkpoint is None and self.config.save_interval > 0:
            saved_last_checkpoint = (self.total_episodes // self.config.save_interval) * self.config.save_interval
        if saved_last_checkpoint is not None:
            self.last_checkpoint_episode = max(self.last_checkpoint_episode, int(saved_last_checkpoint))
        self._update_surprise_scaling()
        if getattr(self, 'training_log_path', None) and not os.path.exists(self.training_log_path):
            with open(self.training_log_path, 'w', encoding='utf-8'):
                pass
        print(f"Checkpoint loaded: {path} (episodes={self.total_episodes}, steps={self.total_steps})")

# TODO: deprecate watch()
def watch(checkpoint_path: str, config: PPOConfig, device: str = 'cpu'):
    """Run a trained policy with CL1 hardware, mirroring the training loop."""
    if not CL_AVAILABLE:
        raise RuntimeError("cl-sdk is required for watch mode.")

    print("Initializing watch mode...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('policy_state_dict', {})
    if any(key.startswith('encoder.freq_alpha_head') for key in state_dict):
        config.encoder_trainable = True
    if any(key.startswith('encoder.cnn') for key in state_dict):
        config.encoder_use_cnn = True

    config = dataclasses.replace(
        config,
        use_hardware=True,
        decoder_ablation_mode=config.decoder_ablation_mode,
        encoder_use_cnn=config.encoder_use_cnn
    )
    trainer = PPOTrainer(config, device=device)
    if trainer.writer is not None:
        trainer.writer.close()
        trainer.writer = None
    trainer.training_log_path = None

    trainer.policy.load_state_dict(state_dict, strict=False)
    trainer.policy.eval()

    print(f"Checkpoint loaded: {checkpoint_path}")
    print("Press Ctrl+C to stop watching.\n")

    try:
        with cl.open() as neurons:
            print("Connected")
            event_channels = []
            for cfg in config.event_feedback_settings.values():
                event_channels.extend(cfg.channels)
            used_channels = (
                config.encoding_channels
                + config.move_forward_channels
                + config.move_backward_channels
                + config.move_left_channels
                + config.move_right_channels
                + config.turn_left_channels
                + config.turn_right_channels
                + config.attack_channels
                + event_channels
            )
            used_channels = sorted(dict.fromkeys(used_channels))
            print(f"Using channels: {used_channels}\n")

            while True:
                trainer._collect_rollouts_hardware(neurons, trainer.config.steps_per_update)

    except KeyboardInterrupt:
        print("\nWatch mode stopped by user.")
    finally:
        if trainer.env is not None:
            trainer.env.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function with CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description='PPO Neural Controller for VizDoom with CL1 Hardware')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'deploy', 'watch'],
                        help='Mode: train (hardware), deploy, or watch')
    parser.add_argument('--checkpoint', type=str, default=None,#"deadly_checkpoints/l5_2048_rand/checkpoint_7900.pt",
                        help='Checkpoint path for deployment')
    parser.add_argument('--max-episodes', type=int, default=65000,
                        help='Maximum training episodes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for gradient computation: cpu or cuda')
    parser.add_argument('--decoder-ablation', type=str, default='none',
                        choices=['none', 'zero', 'random'],
                        help='Ablate spike features before decoder (diagnostic)')
    parser.add_argument('--encoder-use-cnn', action='store_true',
                        help='Use CNN encoder over screen buffer in addition to scalar features')

    # NOTE: (2025-19-11, jz/al) Add additional configuration, including visualisation
    parser.add_argument('--show_window',
                        action="store_true", default=False,
                        help='Show the vizdoom window')
    parser.add_argument('--recording_path', type=str, default='/data/recordings/seandoom',
                        help='Path for saving recordings (managed by CL1 device)')
    parser.add_argument('--tick_frequency_hz', type=int, default=10,
                        help='Frequency to run the game loop in Hz')
    parser.add_argument('--visualisation-port', type=int, default=12349,
                        help='Port to use for accessing the visualisation image stream')

    # UDP Configuration for remote CL1 hardware
    parser.add_argument('--cl1-host', type=str, default='localhost',
                        help='IP address of CL1 device')
    parser.add_argument('--cl1-stim-port', type=int, default=12345,
                        help='UDP port for sending stimulation commands to CL1')
    parser.add_argument('--cl1-spike-port', type=int, default=12346,
                        help='UDP port for receiving spike data from CL1')
    parser.add_argument('--cl1-event-port', type=int, default=12347,
                        help='UDP port for sending event metadata to CL1')
    parser.add_argument('--cl1-feedback-port', type=int, default=12348,
                        help='UDP port for sending feedback stimulation to CL1')
    parser.add_argument('--use-episode-feedback', action='store_true', default=True,
                        help='Enable episode-level feedback stimulation')
    parser.add_argument('--no-episode-feedback', action='store_false', dest='use_episode_feedback',
                        help='Disable episode-level feedback stimulation')
    parser.add_argument('--episode-feedback-surprise-scaling', action='store_true', default=True,
                        help='Scale episode feedback by surprise magnitude')
    parser.add_argument('--no-episode-feedback-surprise-scaling', action='store_false',
                        dest='episode_feedback_surprise_scaling',
                        help='Disable surprise scaling for episode feedback')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"PPO Neural Controller for VizDoom")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")

    if args.mode == 'train':
        print(f"Device: {args.device} (for gradient computation)")
        print(f"Max Episodes: {args.max_episodes}")
    elif args.mode == 'watch':
        print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    # Create config
    if args.mode == 'watch':
        config = PPOConfig(max_episodes=args.max_episodes, use_hardware=True)
    else:
        config = PPOConfig(max_episodes=args.max_episodes)

    config.decoder_ablation_mode = args.decoder_ablation
    if args.encoder_use_cnn:
        config.encoder_use_cnn = True

    # Set UDP configuration from CLI arguments
    config.cl1_host = args.cl1_host
    config.cl1_stim_port = args.cl1_stim_port
    config.cl1_spike_port = args.cl1_spike_port
    config.cl1_event_port = args.cl1_event_port
    config.cl1_feedback_port = args.cl1_feedback_port
    config.use_episode_feedback = args.use_episode_feedback
    config.episode_feedback_surprise_scaling = args.episode_feedback_surprise_scaling
    config.vis_port = args.visualisation_port

    trainer = PPOTrainer(
        config,
        tick_frequency_hz = args.tick_frequency_hz,
        recording_path    = args.recording_path,
        show_window       = args.show_window,
        device            = args.device
        )
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)
    if args.mode == 'train':
        trainer.train()
    # elif args.mode == 'deploy':
    #     if args.checkpoint is None:
    #         checkpoint_path = os.path.join(config.checkpoint_dir, 'final_model.pt')
    #     else:
    #         checkpoint_path = args.checkpoint
    #
    #     deploy(checkpoint_path, config)

    elif args.mode == 'watch':
        trainer.watch()


if __name__ == '__main__':
    main()
