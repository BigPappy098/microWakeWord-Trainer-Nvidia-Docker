# microWakeWord Trainer (RunPod)

Train custom **microWakeWord** wake word models on **RunPod GPU pods**. SSH in, set up, train.

---

## Step 1: Create a RunPod Account

Go to [runpod.io](https://www.runpod.io/) and sign up. Add credits or a payment method.

## Step 2: Create a Network Volume

This is where your datasets, Python environment, and trained models are stored. It persists even when your pod is stopped.

1. Go to **Storage** in the sidebar
2. Click **+ Network Volume**
3. Set size to **100 GB**
4. Pick a datacenter region (remember this — your pod needs to be in the same region)
5. Name it `mww-data`
6. Click **Create**

## Step 3: Deploy a Pod

1. Go to **Pods** → **+ Deploy**
2. Pick any NVIDIA GPU (RTX 3090, RTX 4090, A40, etc.)
3. Click **Edit Template** and set the **Docker image** to:
   ```
   ghcr.io/bigpappy098/microwakeword-trainer-nvidia-docker:latest
   ```
4. Attach your **network volume** from Step 2 and set the mount path to `/data`
5. **(Optional)** Add environment variables to auto-push models to GitHub after training:
   - `GITHUB_TOKEN` = your personal access token ([how to create one](#github-integration))
   - `GITHUB_REPO` = `owner/repo` (e.g. `myuser/my-wakewords`)
6. Click **Deploy**

## Step 4: Connect to Your Pod

Once the pod shows **Running**:

- **Web Terminal**: Click **Connect** → **Start Web Terminal** → **Connect to Web Terminal**
- **SSH**: Click **Connect**, copy the SSH command, run it locally

You'll see a welcome screen with available commands.

## Step 5: Run Setup (First Time Only)

```bash
setup
```

This installs the Python environment and downloads ~50GB of training datasets. It takes 30-60+ minutes on the first run.

Everything is saved to your network volume — **you only do this once**. Next time you start the pod, it's all still there.

## Step 6: Train a Wake Word

```bash
train_wake_word "hey jarvis"
```

Done. Your model files will be in `/data/output/`.

---

## Next Time You Use It

Just start your pod, connect, and train:

```bash
train_wake_word "ok google"
```

No setup needed — the network volume has everything cached.

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
# Quick test (fast, lower quality)
train_wake_word --samples=1000 --training-steps=500 "hey jarvis"

# Full quality training
train_wake_word "hey jarvis"

# Custom title
train_wake_word "hey jarvis" "Hey Jarvis"

# Dutch
train_wake_word --language=nl "hallo computer"
```

---

## Output Files

```
/data/output/<timestamp>-<wake_word>/
  <wake_word>.tflite    # Model for ESP32 / ESPHome
  <wake_word>.json      # ESPHome metadata
  logs/                 # Training logs
```

Flash the `.tflite` file to your device via ESPHome.

---

## Personal Voice Samples (Optional)

Recording your own voice improves accuracy. Record `.wav` files locally (16kHz, PCM 16-bit) and upload them:

```bash
# From your local machine
scp -P <port> *.wav root@<pod-ip>:/data/personal_samples/
```

Name them like `speaker01_take01.wav`, `speaker01_take02.wav`, etc.

They're automatically detected and weighted 3x during training.

---

## GitHub Integration

Auto-push trained models to a GitHub repo after each training run.

### Create a Token

1. Go to **GitHub** → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. Generate a new token with the **repo** scope
3. Copy it

### Set Environment Variables

Add these in your RunPod pod settings (so they persist), or export them:

```bash
export GITHUB_TOKEN=ghp_your_token_here
export GITHUB_REPO=yourusername/your-wakewords-repo
```

| Variable | Required | Description |
|---|---|---|
| `GITHUB_TOKEN` | Yes | Personal access token with `repo` scope |
| `GITHUB_REPO` | Yes | Target repo (`owner/repo` format) |
| `GITHUB_BRANCH` | No | Branch to push to (default: `main`) |
| `GITHUB_PATH` | No | Directory in repo for models (default: `.`) |

If these aren't set, the GitHub push step is silently skipped.

---

## Other Commands

```bash
setup                    # Full setup (venv + datasets)
setup_python_venv        # Just the Python environment
setup_training_datasets  # Just the datasets
cudainfo                 # GPU info
system_summary           # CPU, RAM, disk, GPU stats
nvidia-smi               # NVIDIA GPU status
```

---

## Multiple Wake Words

Train as many as you want, back-to-back. No cleanup needed — each run gets its own output folder.

## Resetting Everything

```bash
rm -rf /data/*
```

Then run `setup` again.

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

## Credits

Built on [microWakeWord](https://github.com/kahrendt/microWakeWord) by Kevin Ahrendt.
