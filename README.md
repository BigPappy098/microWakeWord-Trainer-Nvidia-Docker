# microWakeWord Trainer (RunPod CLI)

Train custom wake word models for ESP32/ESPHome on RunPod GPU pods. No web UI — just SSH in and run commands.

Personal voice recordings are optional but improve accuracy. You store them in your GitHub fork and they're pulled automatically during training.

---

## Complete Setup Guide

### Step 1: Fork This Repo

1. Click the **Fork** button at the top of this GitHub page
2. This gives you your own copy at `github.com/<your-username>/microWakeWord-Trainer-Nvidia-Docker`
3. You'll use this fork to store personal voice recordings and receive trained models

### Step 2: Create a GitHub Personal Access Token

This lets the trainer pull your recordings and push finished models back to your fork.

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **Generate new token** → **Generate new token (classic)**
3. Give it a name like `mww-trainer`
4. Check the **repo** scope (full control of private repositories)
5. Click **Generate token**
6. **Copy the token now** — you won't see it again

### Step 3: (Optional) Add Personal Voice Recordings

Recording your own voice saying the wake word improves detection accuracy. These samples get weighted 3x during training.

1. On your computer, record yourself saying your wake word
   - Format: `.wav`, 16kHz sample rate, 16-bit PCM, mono
   - Most voice recorder apps work — just convert to `.wav` afterward if needed
2. Name the files: `speaker01_take01.wav`, `speaker01_take02.wav`, etc.
   - For multiple people: `speaker02_take01.wav`, `speaker02_take02.wav`, etc.
   - Aim for **10+ takes per speaker**
3. Add the `.wav` files to the `personal_samples/` folder in your fork
4. Commit and push them to GitHub

**Tips for good recordings:**
- Quiet room, minimal background noise
- Vary your distance from the mic (near, mid, far)
- Speak naturally — don't over-enunciate
- Record at different times of day (your voice changes)
- Get family members or friends to record too

> **No recordings?** That's fine — the trainer generates thousands of synthetic voice samples using TTS. Personal recordings just make the model better at recognizing *your* voice.

### Step 4: Create a RunPod Account

1. Go to [runpod.io](https://www.runpod.io/) and sign up
2. Add credits or a payment method (GPU time costs ~$0.20-0.80/hr depending on the GPU)

### Step 5: Create a Network Volume

The network volume stores everything (datasets, Python environment, models). It persists when your pod is stopped, so you only download datasets once.

1. In RunPod, go to **Storage** in the sidebar
2. Click **+ Network Volume**
3. Set size to **100 GB** (training datasets are ~50GB, plus room for models and working files)
4. Pick a datacenter region — **remember this**, your pod must be in the same region
5. Name it `mww-data`
6. Click **Create**

### Step 6: Deploy a Pod

1. Go to **Pods** → **+ Deploy**
2. Pick any NVIDIA GPU (RTX 3090, RTX 4090, A40, etc. — any will work)
3. Click **Edit Template** and set the **Docker image** to:
   ```
   ghcr.io/bigpappy098/microwakeword-trainer-nvidia-docker:latest
   ```
4. Attach your **network volume** (`mww-data`) and set the mount path to `/data`
5. Add these **environment variables** (under "Environment Variables"):
   - `GITHUB_TOKEN` = the token you created in Step 2
   - `GITHUB_REPO` = `<your-username>/microWakeWord-Trainer-Nvidia-Docker` (your fork)
6. Click **Deploy**

### Step 7: Connect to Your Pod

Once the pod shows **Running**:

- **Web Terminal** (easiest): Click **Connect** → **Start Web Terminal** → **Connect to Web Terminal**
- **SSH** (if you prefer): Click **Connect**, copy the SSH command, run it in your terminal

You'll see a welcome banner with available commands.

### Step 8: Run First-Time Setup

```bash
setup
```

This does two things:
1. Creates a Python virtual environment with TensorFlow, PyTorch, and all dependencies
2. Downloads ~50GB of training datasets (negative samples the model needs to learn what *isn't* your wake word)

**This takes 30-60+ minutes** on the first run. Go grab a coffee.

Everything is saved to your network volume — you only do this once. Next time you start the pod, it's all still there.

### Step 9: Train Your Wake Word

```bash
train_wake_word "hey jarvis"
```

This runs the full pipeline:
1. **Pulls personal recordings** from your GitHub fork (if configured)
2. **Generates** 50,000 synthetic TTS samples of your wake phrase
3. **Augments** all samples (pitch shifting, background noise, room acoustics)
4. **Trains** the neural network (~40,000 steps)
5. **Outputs** a quantized `.tflite` model ready for ESP32
6. **Pushes** the model to your GitHub fork (if configured)

Training takes roughly 1-3 hours depending on the GPU.

### Step 10: Get Your Model

After training, your files are in two places:

**On the pod:**
```
/data/output/<timestamp>-<wake_word>/
  <wake_word>.tflite    # The model file for ESP32/ESPHome
  <wake_word>.json      # ESPHome metadata
  logs/                 # Training logs and metrics
```

**On GitHub** (if you configured `GITHUB_TOKEN` and `GITHUB_REPO`):
The `.tflite` and `.json` files are automatically pushed to your fork.

Flash the `.tflite` file to your ESP32 device via ESPHome.

---

## Training Again (Next Time)

Just start your pod, connect, and go:

```bash
train_wake_word "ok google"
```

No setup needed — the network volume has everything cached. Train as many wake words as you want back-to-back. Each run gets its own output folder.

---

## Training Options

```bash
train_wake_word [options] <wake_word> [<wake_word_title>]

Options:
  --samples=<N>           TTS samples to generate (default: 50000)
  --batch-size=<N>        Samples per batch (default: 100)
  --training-steps=<N>    Training iterations (default: 40000)
  --language=<lang>       TTS language: "en", "nl", etc. (default: en)
  --cleanup-work-dir      Delete intermediate files after training
```

Examples:
```bash
# Quick test run (fast but lower quality — good for testing your setup)
train_wake_word --samples=1000 --training-steps=500 "hey jarvis"

# Full quality training
train_wake_word "hey jarvis"

# Custom display title (what shows in ESPHome)
train_wake_word "hey jarvis" "Hey Jarvis"

# Dutch language
train_wake_word --language=nl "hallo computer"
```

---

## Environment Variables Reference

Set these in your RunPod pod settings so they persist across restarts.

| Variable | Required | Description |
|---|---|---|
| `GITHUB_TOKEN` | Yes | Personal access token with `repo` scope |
| `GITHUB_REPO` | Yes | Your fork in `owner/repo` format |
| `GITHUB_BRANCH` | No | Branch to use (default: `main`) |
| `GITHUB_PATH` | No | Directory in repo for trained models (default: `.`) |
| `GITHUB_RECORDINGS_PATH` | No | Directory in repo for recordings (default: `personal_samples`) |

If `GITHUB_TOKEN` and `GITHUB_REPO` aren't set, recording pull and model push are silently skipped. Everything else still works.

---

## All Commands

```bash
setup                       # Full first-time setup (venv + datasets)
train_wake_word <word>      # Train a wake word model
pull_personal_recordings    # Manually pull recordings from your GitHub fork
setup_python_venv           # Reinstall just the Python environment
setup_training_datasets     # Redownload just the datasets
cudainfo                    # Show GPU compute capability
system_summary              # Show CPU, RAM, disk, GPU stats
nvidia-smi                  # NVIDIA GPU status
```

---

## Troubleshooting

**"Python venv not found"** — Run `setup` first.

**Training fails with GPU errors** — The trainer automatically retries with different GPU profiles and falls back to CPU if needed. Check the log file path printed in the error output.

**"No personal_samples/ directory found"** — This is fine. It just means you haven't added recordings to your fork. Training continues with synthetic samples only.

**Pod runs out of disk** — Make sure your network volume is at least 100GB. Run `train_wake_word --cleanup-work-dir "..."` to clean up intermediate files after training.

---

## Storage Breakdown

| What | Size |
|---|---|
| Python environment | ~5 GB |
| Training datasets | ~40 GB |
| Tools & TTS models | ~3 GB |
| Work files (per run) | ~10 GB |
| **Total** | **~60 GB** |

100 GB network volume recommended.

---

## Resetting Everything

```bash
rm -rf /data/*
```

Then run `setup` again.

---

## Credits

Built on [microWakeWord](https://github.com/kahrendt/microWakeWord) by Kevin Ahrendt.
