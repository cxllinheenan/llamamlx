# LLaMA Model Training with MLX

This repository contains an implementation of a LLaMA-style language model that can be trained on the FineWeb dataset using Apple's MLX framework. The code is optimized for Apple Silicon (M1/M2/M3) processors.

## Features

- LLaMA architecture implementation with MLX
- Training on FineWeb or similar text datasets
- Memory-efficient training with gradient checkpointing
- Checkpoint saving and resuming
- Weights & Biases integration for experiment tracking
- Learning rate schedulers (cosine decay and linear warmup)

## Requirements

- macOS with Apple Silicon 
- Python 3.11+
- MLX
- Hugging Face `datasets` and `transformers`
- (Optional) Weights & Biases for experiment tracking

## Installation

```bash
uv pip install mlx datasets transformers wandb
```

## Usage

### Basic Training

```bash
uv run  models/train.py --gpu --checkpoint --batch_size 8 --dim 256 --num_heads 4 --num_blocks 4
```

### Memory-Efficient Training

For systems with limited memory:

```bash
python train.py --gpu --checkpoint --batch_size 1 --dim 256 --num_heads 4 --num_blocks 4 --context_size 128
```

### Training with W&B Tracking

```bash
python train.py --gpu --checkpoint --batch_size 8 --dim 256 --num_heads 4 --num_blocks 4 --use_wandb
```

### Resuming from Checkpoint

```bash
python train.py --gpu --checkpoint --resume_from ./llama_mlx_fineweb/checkpoint-1000
```

## Command Line Arguments

- `--dim`: Model embedding dimension
- `--num_blocks`: Number of transformer layers
- `--num_heads`: Number of attention heads
- `--checkpoint`: Enable gradient checkpointing
- `--dataset_name`: FineWeb dataset configuration
- `--tokenizer`: HuggingFace tokenizer to use
- `--context_size`: Training context window size
- `--max_tokens`: Maximum tokens to process during training
- `--batch_size`: Training batch size
- `--learning_rate`: AdamW learning rate
- `--weight_decay`: AdamW weight decay
- `--lr_warmup`: Learning rate warmup steps
- `--lr_schedule`: Learning rate schedule (linear or cosine)
- `--output_dir`: Directory to save checkpoints
- `--gpu`: Use Metal GPU backend
- `--debug`: Enable extra debugging information

## Model Architecture

The implementation follows the LLaMA architecture with:
- RMSNorm instead of LayerNorm
- Rotary positional embeddings (RoPE)
- SwiGLU activations
- Pre-normalization architecture
- Linear layers without bias terms

## License

MIT
