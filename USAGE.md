# Usage Guide

## Quick Start

### 1. Start CL1 Neural Interface (on CL1 device)

```bash
python cl1_neural_interface.py --training-host <TRAINING_IP>
```

### 2. Start Training Server (on training machine)

```bash
python training_server.py --mode train --device cuda --cl1-host <CL1_IP>
```

## CL1 Neural Interface

### Basic Command

```bash
python cl1_neural_interface.py --training-host 192.168.1.100
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--training-host` | str | **required** | IP address of training system |
| `--stim-port` | int | 12345 | Port for receiving stimulation commands |
| `--spike-port` | int | 12346 | Port for sending spike data |
| `--event-port` | int | 12347 | Port for receiving event metadata |
| `--feedback-port` | int | 12348 | Port for receiving feedback commands |
| `--tick-frequency` | int | 240 | Neural loop frequency in Hz |
| `--recording-path` | str | ./recordings | Directory for saving recordings |

### Example with Custom Ports

```bash
python cl1_neural_interface.py \
    --training-host 192.168.1.100 \
    --stim-port 12345 \
    --spike-port 12346 \
    --event-port 12347 \
    --feedback-port 12348 \
    --tick-frequency 240 \
    --recording-path /data/recordings
```

## Training Server

### Basic Command

```bash
python training_server.py --mode train --device cuda --cl1-host 192.168.1.50
```

### Modes

- `train` - Run PPO training with neural hardware
- `watch` - Run trained policy (inference only, uses direct hardware access)

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | **required** | Operation mode (train, watch) |
| `--device` | str | cpu | PyTorch device (cpu, cuda) |
| `--cl1-host` | str | localhost | IP address of CL1 device |
| `--max-episodes` | int | 100000 | Maximum training episodes |
| `--checkpoint` | str | None | Path to checkpoint for resuming/watching |

### UDP Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cl1-stim-port` | int | 12345 | Port for sending stimulation to CL1 |
| `--cl1-spike-port` | int | 12346 | Port for receiving spikes from CL1 |
| `--cl1-event-port` | int | 12347 | Port for sending events to CL1 |
| `--cl1-feedback-port` | int | 12348 | Port for sending feedback to CL1 |

### Feedback Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-episode-feedback` | flag | True | Enable episode-level feedback |
| `--no-episode-feedback` | flag | - | Disable episode-level feedback |
| `--episode-feedback-surprise-scaling` | flag | True | Scale episode feedback by surprise |
| `--no-episode-feedback-surprise-scaling` | flag | - | Use static episode feedback values |

### Display & Recording

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--show_window` | flag | False | Show VizDoom game window |
| `--recording_path` | str | /data/recordings/seandoom | Recording directory path |
| `--tick_frequency_hz` | int | 240 | Game loop frequency in Hz |

### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--decoder-ablation` | str | none | Ablation mode (none, zero, random) |
| `--encoder-use-cnn` | flag | False | Enable CNN encoder for screen buffer |

## Common Usage Scenarios

### Local Development (Same Machine)

**CL1 Interface:**
```bash
python cl1_neural_interface.py --training-host localhost
```

**Training Server:**
```bash
python training_server.py --mode train --device cuda --cl1-host localhost
```

### Remote Training (Separate Machines)

**CL1 Interface (on CL1 device at 192.168.1.50):**
```bash
python cl1_neural_interface.py --training-host 192.168.1.100
```

**Training Server (on training machine at 192.168.1.100):**
```bash
python training_server.py \
    --mode train \
    --device cuda \
    --cl1-host 192.168.1.50
```

### Resume Training from Checkpoint

```bash
python training_server.py \
    --mode train \
    --device cuda \
    --cl1-host 192.168.1.50 \
    --checkpoint checkpoints/episode_5000.pt
```

### Watch Trained Policy

```bash
python training_server.py \
    --mode watch \
    --device cuda \
    --checkpoint checkpoints/final_model.pt
```

Note: Watch mode uses direct hardware access, I have not ported it to UDP.

### Custom Feedback Configuration

**Episode feedback only (no step-level rewards):**
```bash
python training_server.py \
    --mode train \
    --device cuda \
    --cl1-host 192.168.1.50 \
    --use-episode-feedback \
    --no-episode-feedback-surprise-scaling
```

**Disable episode feedback:**
```bash
python training_server.py \
    --mode train \
    --device cuda \
    --cl1-host 192.168.1.50 \
    --no-episode-feedback
```

### Show Game Window (Debug Mode)

```bash
python training_server.py \
    --mode train \
    --device cuda \
    --cl1-host 192.168.1.50 \
    --show_window
```

### Custom Recording Location

**CL1 Interface:**
```bash
python cl1_neural_interface.py \
    --training-host 192.168.1.100 \
    --recording-path /mnt/data/doom_recordings
```

**Training Server:**
```bash
python training_server.py \
    --mode train \
    --device cuda \
    --cl1-host 192.168.1.50 \
    --recording_path /mnt/data/doom_recordings
```

## Network Requirements

**Ports Required:**
- 12345 (stimulation commands: training → CL1)
- 12346 (spike data: CL1 → training)
- 12347 (event metadata: training → CL1)
- 12348 (feedback commands: training → CL1)

## Stopping Training

Press `Ctrl+C` in either terminal to gracefully shutdown:
- Training server sends completion signal
- CL1 interface saves recording and exits
- Both processes cleanup sockets

## Output Files

**Training Server:**
- `checkpoints/episode_*.pt` - Model checkpoints
- `runs/*/` - TensorBoard logs
- `training_log.jsonl` - Episode statistics

**CL1 Interface:**
- `<recording_path>/*.cl1` - Neural recordings with metadata
