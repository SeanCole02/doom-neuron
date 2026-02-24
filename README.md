# ppo_doom.py Quickstart

## Default Parameters
- **PPO**: `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95` (many RL implementations actually use far lower gamma 0.95 and lambda 0.90 for GAE but this can severely affect training on lower levels due to the long range dependencies since you take less damage and therefore live longer), `clip_epsilon=0.2`, `entropy_coef=0.02`, `steps_per_update=2048` (use higher for stability but it can be really slow without parallelization of some sort, that would probably require GRPO instead of PPO), `batch_size=256`, `num_epochs=4`.
- **Observation & Action**: Screen buffer enabled at `RES_320X240`; hybrid action spaces are used (and greatly preferred) unless `use_discrete_action_set=True`. Realistically you only flip this if all else fails to reduce entropy as it greatly reduces the movement fidelity of the agent and just doesn't look as cool.
- **Scenario Config**: `doom_config` defaults to `progressive_deathmatch.cfg` (`progressive_deathmatch.wad`) — similar to survival but kills don't reset ammo count (encouraging proper ammo management) with movement tweaks to make movement easier to train. Also available: `survival.cfg` (`survival.wad`) and the deadly corridor curriculum (`deadly_corridor_1.cfg` through `deadly_corridor_5.cfg`, all using `deadly_corridor.wad`). Files `deadly_corridor_1.cfg` to `deadly_corridor_4.cfg` ramp difficulty gradually, but `deadly_corridor_5.cfg` is a significant jump (and the actual benchmark). Progress through 1-4 builds basic policies yet may result in movement habits that underperform on 5 (straight running toward armor). Adjust curriculum pacing accordingly.
- **Feedback Defaults**: Episode feedback now follows overall reward unless `episode_positive_feedback_event`/`episode_negative_feedback_event` are set. Reward channels use `feedback_positive_amplitude=2.0` and `feedback_negative_amplitude=2.0`, dynamically scaled but clipped via `_limit_scaled_amplitude`.

## Architecture & Feedback Tuning (Deadly Corridor)

> The specific values below are tuned for the deadly corridor scenario (`deadly_corridor_1.cfg` – `deadly_corridor_5.cfg`). Treat them as a starting point only — other scenarios (progressive deathmatch, survival) will likely require different values for feedback scaling, reward shaping, ray-cast geometry, and curriculum pacing.

- `use_reward_feedback`: Uses rewards to drive postive/negative feedback rather than action specific feedback, if you do decide to use action feedback, tweak `event_feedback_settings` accordingly, these were values were arbitarily set.
- `decoder_enforce_nonnegative=False`, `decoder_freeze_weights=False`: decoder stays free to mirror encoder intent; set to True if you need tight control over decoded spike weights.
- `decoder_zero_bias=True`: keeps bias at zero so decoded actions depend solely on encoder output; helped prevent a lot of decoder-sided learning in testing but may be different on actual hardware since the sdk spikes were random; this should definitely be tested with ablations!
- `decoder_use_mlp=False`, `decoder_mlp_hidden=32`: default linear decoder keeps hardware mapping transparent; enable the MLP when you require richer non-linear policies (expect higher sample complexity, decoder also tends to start becoming a policy head but might be due to random spike noise from the SDK).
- `decoder_weight_l2_coef=0.0`, `decoder_bias_l2_coef=0.0`: L2 regularization hooks, raise these only if decoder weights diverge on long runs.
- `wall_ray_count=12`, `wall_ray_max_range=64`, `wall_depth_max_distance=18.0`: ray-cast features tuned for corridor geometry; adjust if you alter field-of-view or scenario scale.
- `encoder_trainable=True`, `encoder_entropy_coef=-0.10`: encoder keeps learning with a negative entropy coefficient that encourages confident (low-variance) stimulation because of the Beta distribution head.
- `decoder_ablation_mode='none'`: swap to `random` or `zero` to test policy robustness when decoder contributions are removed.
- `encoder_use_cnn=True`, `encoder_cnn_channels=16`, `encoder_cnn_downsample=4`: lightweight CNN for spatial features; bump channels/downsample when raising resolution, can disable if needed to rely soley on raycasting data.
- `episode_positive_feedback_event=None`, `episode_negative_feedback_event=None`: default to global reward feedback; set to event keys if you need action-specific filters.
- Surprise scaling knobs (`feedback_surprise_gain`, `_max_scale`, `_freq_gain`, `_amp_gain` variants): modulate how unexpected TD errors boost frequency/amplitude; higher values emphasize novel negative events.
- `enemy_distance_normalization=1312.0`: normalization constant for distance-based shaping; leave untouched unless you change WAD geometry or measurement units.
- We use SiLU instead of GELU for the encoder/decoder, as GELU can be a tad less efficient, especially if the tanh approximation isn’t vectorized, and we don't particularly need the encoder to be perfect, just adaptive enough for the CL1 neurons.


## Getting Started
1. **Install Requirements**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Torch 2.10 was used with CUDA 13.0 in testing but the version does not matter too much here, use whatever is compatible with the hardware available.
2. **Pick a Scenario**
   - Default is `PPOConfig.doom_config = "progressive_deathmatch.cfg"`. Also available: `"survival.cfg"`.
   - For the deadly corridor curriculum, start with `"deadly_corridor_1.cfg"` and advance sequentially; consider fine-tuning on `deadly_corridor_5.cfg` with a lower learning rate to adapt movement behavior.
3. **Run Training**
   ```bash
   python3 ppo_doom.py
   ```
   - Checkpoints land in `PPOConfig.checkpoint_dir`, logs in `PPOConfig.log_dir`.
   - Use TensorBoard to monitor: `tensorboard --logdir checkpoints/l5_2048_rand/logs`.
4. **Tweak Defaults**
   - Override fields when instantiating `PPOConfig`, e.g. `PPOConfig(doom_config="deadly_corridor_1.cfg")`.
   - For event-specific episode feedback, set `episode_positive_feedback_event`/`episode_negative_feedback_event` to keys defined in `event_feedback_settings`.
5. **Parameter Monitoring**
   ```
   tensorboard --logdir checkpoints/l5_2048_rand/logs --port 6006
   ```

## Running with increased configuration (2025-11-19)

```python
# Run with cpu and a tick rate of 240 Hz
python3 ppo_doom.py \
   --device "cpu" \
   --tick_frequency_hz 10

# Run locally and showing the window
python3 ppo_doom.py \
   --device "cpu" \
   --tick_frequency_hz 10 \
   --recording_path ./recordings \
   --show_window
```

## Running Local Server + Connecting to CL1 ##

See [USAGE.md](USAGE.md)
