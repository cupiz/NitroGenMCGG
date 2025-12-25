# NitroGen Fine-Tuning

Fine-tune [NVIDIA NitroGen](https://huggingface.co/nvidia/NitroGen) for coordinate prediction in Magic Chess: Go Go.

## What This Does

Takes gameplay videos and trains a model to predict where to click on screen. The vision encoder stays frozen while a new coordinate head learns to output (x, y) positions.

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the base model:

```bash
pip install huggingface-hub
huggingface-cli download nvidia/NitroGen ng.pt --local-dir weights
```

## Usage

### 1. Generate Dataset from Video

The video needs to have a visible mouse cursor. YouTube gameplay recordings usually work.

```bash
python auto_label.py --video gameplay.mp4 --output-dir data --fps 5
```

This extracts frames and auto-detects cursor positions using CV.

### 2. Train

```bash
python train.py \
    --frames-dir data/frames \
    --labels-file data/labels.json \
    --pretrained weights/ng.pt \
    --batch-size 8 \
    --epochs 100
```

Training saves checkpoints every 30 minutes. If interrupted, just run the same command again - it picks up where it left off.

## Files

```
├── config.py           # paths, hyperparameters
├── model.py            # architecture
├── dataset.py          # data loading
├── auto_label.py       # cursor detection
├── train.py            # training loop
├── utils.py            # checkpointing
└── test_model.py       # validation
```

## Architecture

```
Frame (256x256)
    ↓
Vision Encoder (SigLip2, frozen)
    ↓
Coordinate Head (trainable)
    ↓
(x, y) in [0, 1]
```

Only the coordinate head gets trained. The vision encoder keeps its pretrained weights from 40k hours of gameplay data.

## Cloud Training

Works on Paperspace/similar with 6-hour timeout. Checkpoints go to `/storage/checkpoints/` which persists across restarts.

```bash
python train.py --checkpoint-dir /storage/checkpoints
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- GPU with 16GB+ VRAM

## License

Uses NitroGen weights under [NVIDIA Non-Commercial License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf).
