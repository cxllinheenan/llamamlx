import math
import time
import os
import json
import argparse
import logging
from functools import partial
from typing import Optional, Tuple
from pydantic_core.core_schema import float_schema
import wandb
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from data import FineWebDataLoader, evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Custom activation functions
def silu(x):
    """SiLU/Swish activation function: x * sigmoid(x)"""
    return x * mx.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        # RMSNorm uses root mean square normalization
        x_squared = x * x
        mean_squared = mx.mean(x_squared, axis=-1, keepdims=True)
        norm_x = x * mx.rsqrt(mean_squared + self.eps)
        return norm_x * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) as used in LLaMA."""
    
    def __init__(self, dims: int, base: int = 10000):
        super().__init__()
        self.dims = dims
        self.base = base
        
    def __call__(self, positions, dims: Optional[int] = None):
        """Apply rotary embeddings to input tensor."""
        dims = dims or self.dims
        half_dims = dims // 2
        
        # Create theta parameter
        theta = 1.0 / (self.base ** (mx.arange(0, half_dims, 1) / half_dims))
        
        # Create position-dependent rotation matrix components
        positions = positions.reshape(-1, 1)  # [seq_len, 1]
        freqs = positions * theta  # [seq_len, half_dims]
        
        # Create complex rotations
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        
        return cos, sin
        
    def rotate_embedding(self, x, cos, sin):
        # Reshape for rotation
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        
        # Apply rotation using complex multiplication
        result = mx.concatenate([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], axis=-1)
        
        return result


class LLaMAAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        
        # LLaMA uses separate key, query, value projections
        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.o_proj = nn.Linear(dims, dims, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim)
    
    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        queries = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        positions = mx.arange(seq_len)
        cos, sin = self.rope(positions)
        
        # Reshape queries and keys for rotation
        queries = queries.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        keys = keys.transpose(0, 2, 1, 3)        # [batch, heads, seq, head_dim]
        
        # Apply rotations - core part of RoPE
        queries = self.rope.rotate_embedding(queries, cos, sin)
        keys = self.rope.rotate_embedding(keys, cos, sin)
        
        # Reshape for attention computation
        values = values.transpose(0, 2, 1, 3)    # [batch, heads, seq, head_dim]
        
        # Compute scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * scale
        
        # Apply causal mask
        causal_mask = mx.tril(mx.ones((seq_len, seq_len)))
        scores = scores + (1 - causal_mask) * -10000.0
        
        attention = mx.softmax(scores, axis=-1)
        output = mx.matmul(attention, values)
        
        # Reshape and project back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dims)
        return self.o_proj(output)


class LLaMAFeedForward(nn.Module):
    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()
        # LLaMA uses SwiGLU activation
        self.gate_proj = nn.Linear(dims, hidden_dims, bias=False)
        self.up_proj = nn.Linear(dims, hidden_dims, bias=False)
        self.down_proj = nn.Linear(hidden_dims, dims, bias=False)
    
    def __call__(self, x):
        # SwiGLU activation function
        gate = self.gate_proj(x)
        gate = silu(gate)  # Use SiLU/Swish activation
        up = self.up_proj(x)
        intermediate = gate * up
        return self.down_proj(intermediate)


class LLaMABlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, ffn_dims: int):
        super().__init__()
        # LLaMA uses pre-normalization with RMSNorm
        self.attention_norm = RMSNorm(dims)
        self.attention = LLaMAAttention(dims, num_heads)
        self.ffn_norm = RMSNorm(dims)
        self.feed_forward = LLaMAFeedForward(dims, ffn_dims)
    
    def __call__(self, x):
        # Pre-normalization architecture
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LLaMAModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        ffn_dims: int,
        checkpoint: bool,
    ):
        super().__init__()
        
        # LLaMA doesn't use bias terms in linear layers
        self.embedding = nn.Embedding(vocab_size, dims)
        
        # LLaMA transformer blocks with checkpointing support
        self.blocks = [
            LLaMABlock(dims, num_heads, ffn_dims)
            for _ in range(num_layers)
        ]
        
        # Final normalization
        self.norm = RMSNorm(dims)
        
        # Output projection
        self.out_proj = nn.Linear(dims, vocab_size, bias=False)
    
    def __call__(self, x):
        # Token embeddings (without positional encodings)
        x = self.embedding(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        return self.out_proj(x)


def save_checkpoint(model, optimizer, args, step, tokens_processed):
    """Save model and optimizer state using npz format."""
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Save model parameters using savez
        logger.info("Saving model parameters...")
        flat_params = tree_flatten(model.parameters())
        mx.savez(os.path.join(checkpoint_dir, "model.npz"), **dict(flat_params))
        
        # Save optimizer state
        logger.info("Saving optimizer state...")
        flat_optimizer_state = tree_flatten(optimizer.state)
        mx.savez(os.path.join(checkpoint_dir, "optimizer.npz"), **dict(flat_optimizer_state))
        
        # Save training arguments and progress
        training_info = {
            "step": step,
            "tokens_processed": tokens_processed,
            "args": vars(args)
        }
        
        with open(os.path.join(checkpoint_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        logger.error("Training will continue without saving checkpoint.")


def load_checkpoint(model, optimizer, checkpoint_dir):
    """Load model and optimizer state from npz format checkpoint."""
    # Load model weights
    model_path = os.path.join(checkpoint_dir, "model.npz")
    if os.path.exists(model_path):
        try:
            logger.info(f"Loading model weights from {model_path}")
            params = dict(np.load(model_path))
            # Convert numpy arrays to mx arrays
            params = {k: mx.array(v) for k, v in params.items()}
            model.update(params)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model weights: {str(e)}")
    
    # Load optimizer state
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.npz")
    if os.path.exists(optimizer_path):
        try:
            logger.info(f"Loading optimizer state from {optimizer_path}")
            opt_state = dict(np.load(optimizer_path))
            # Convert numpy arrays to mx arrays
            opt_state = {k: mx.array(v) for k, v in opt_state.items()}
            optimizer.state.update(opt_state)
            logger.info("Optimizer state loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load optimizer state: {str(e)}")
    
    # Load training info
    training_info_path = os.path.join(checkpoint_dir, "training_info.json")
    if os.path.exists(training_info_path):
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
        logger.info(f"Loaded training info from {training_info_path}")
        return training_info["step"], training_info["tokens_processed"]
    
    return 0, 0


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"llama-mlx-{args.dim}-{args.num_blocks}-{args.dataset_name}",
            config=vars(args)
        )
    
    # Initialize dataset
    train_dataloader = FineWebDataLoader(
        tokenizer_name=args.tokenizer,
        context_size=args.context_size,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        dataset_name=args.dataset_name,
        seed=args.seed
    )
    
    # For evaluation, we'll use a separate instance with a small buffer
    eval_dataloader = FineWebDataLoader(
        tokenizer_name=args.tokenizer,
        context_size=args.context_size,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        seed=args.seed + 1  # Use different seed for validation
    )
    
    # Get vocab size from tokenizer
    vocab_size = train_dataloader.vocab_size
    
    # Initialize model:
    ffn_dims = args.dim * 4  # LLaMA typically uses 4x hidden dims
    
    logger.info(f"Initializing LLaMA model with {args.num_blocks} layers, "
                f"{args.dim} dimensions, {args.num_heads} heads")
    
    model = LLaMAModel(
        vocab_size=vocab_size,
        num_layers=args.num_blocks, 
        dims=args.dim, 
        num_heads=args.num_heads, 
        ffn_dims=ffn_dims,
        checkpoint=args.checkpoint
    )
    
    mx.eval(model.parameters())
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    logger.info(f"Model initialized with {nparams / 1024**2:.3f} M parameters")

    # Initialize optimizer
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)  # LLaMA paper uses beta2=0.95
    )
    
    # Load checkpoint if specified
    start_step = 0
    tokens_processed = 0
    if args.resume_from:
        start_step, tokens_processed = load_checkpoint(model, optimizer, args.resume_from)
        logger.info(f"Resuming from step {start_step} with {tokens_processed} tokens processed")
    
    # Training loss function - FIXED LOSS FUNCTION
    def loss_fn(model, x, y):
        logits = model(x)
        # Make sure logits shape is (batch_size, seq_len, vocab_size)
        # and targets shape is (batch_size, seq_len)
        if args.debug:
            logger.info(f"Logits shape: {logits.shape}, Targets shape: {y.shape}")
        
        # Ensure we're using the proper dimension and reduction for cross entropy
        # For language modeling, we want to average over all tokens
        return nn.losses.cross_entropy(logits, y, reduction="mean")
    
    # Compile the training step function
    state = [model.state, optimizer.state]
    
    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss
    
    # Training loop
    losses = []
    tic = time.perf_counter()
    step_count = start_step
    
    logger.info(f"Starting training from step {start_step}")
    logger.info(f"Training for {args.max_tokens} tokens")
    
    for inputs, targets in train_dataloader:
        # Debug shapes if requested
        if args.debug and step_count == 0:
            logger.info(f"Input tensor shape: {inputs.shape}")
            logger.info(f"Target tensor shape: {targets.shape}")
        
        # Update learning rate (cosine or linear schedule)
        if args.lr_schedule == "cosine" and args.num_iters > 0:
            progress = min(1.0, step_count / args.num_iters)
            lr_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            optimizer.learning_rate = args.learning_rate * lr_factor
        else:
            # Linear warmup
            optimizer.learning_rate = min(1, step_count / args.lr_warmup) * args.learning_rate
        
        try:
            # Perform optimization step
            loss = step(inputs, targets)
            mx.eval(state)
            losses.append(loss.item())
            
            step_count += 1
            
            # Reporting
            if step_count % args.steps_per_report == 0:
                train_loss = np.mean(losses)
                tokens_per_sec = (inputs.size * args.steps_per_report) / (time.perf_counter() - tic)
                
                logger.info(
                    f"Step {step_count}: Train loss {train_loss:.3f}, "
                    f"LR {optimizer.learning_rate:.6f}, "
                    f"Tokens/sec {tokens_per_sec:.1f}, "
                    f"Tokens processed {train_dataloader.tokens_processed:,}"
                )
                
                # Log to wandb if enabled
                if args.use_wandb:
                    wandb.log({
                        "train/loss": float(train_loss),
                        "train/learning_rate": float(optimizer.learning_rate),
                        "train/tokens_per_sec": float(tokens_per_sec),
                        "train/tokens_processed": int(train_dataloader.tokens_processed),
                        "train/step": int(step_count)
                    })
                
                losses = []
                tic = time.perf_counter()
            
            # Evaluation
            if step_count % args.steps_per_eval == 0:
                logger.info(f"Evaluating at step {step_count}...")
                eval_loss, eval_ppl = evaluate_model(model, eval_dataloader)
                
                logger.info(
                    f"Step {step_count}: "
                    f"Eval loss {eval_loss:.3f}, "
                    f"Eval ppl {eval_ppl:.3f}"
                )
                
                # Log to wandb if enabled
                if args.use_wandb:
                    wandb.log({
                        "eval/loss": float(eval_loss),
                        "eval/perplexity": float(eval_ppl),
                        "eval/step": int(step_count)
                    })
                
                # Save checkpoint
                save_checkpoint(
                    model, 
                    optimizer, 
                    args, 
                    step_count,
                    train_dataloader.tokens_processed
                )
        
        except Exception as e:
            logger.error(f"Error during training step: {str(e)}")
            if args.debug:
                logger.error(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
                logger.error(f"Input dtype: {inputs.dtype}, Target dtype: {targets.dtype}")
                # Print first few elements to help with debugging
                logger.error(f"Input sample: {inputs[0, :10]}")
                logger.error(f"Target sample: {targets[0, :10]}")
            raise e
            
        # Check if we're done
        if args.num_iters > 0 and step_count >= args.num_iters:
            logger.info(f"Reached max iterations ({args.num_iters}). Training complete.")
            break
            
        if args.max_tokens and train_dataloader.tokens_processed >= args.max_tokens:
            logger.info(f"Reached max tokens ({args.max_tokens}). Training complete.")
            break
    
    # Final evaluation and save
    logger.info(f"Final evaluation...")
    eval_loss, eval_ppl = evaluate_model(model, eval_dataloader)
    logger.info(f"Final: Eval loss {eval_loss:.3f}, Eval ppl {eval_ppl:.3f}")
    
    # Log final results to wandb if enabled
    if args.use_wandb:
        wandb.log({
            "eval/final_loss": float(eval_loss),
            "eval/final_perplexity": float(eval_ppl),
            "train/total_steps": int(step_count),
            "train/total_tokens": int(train_dataloader.tokens_processed
        })
    
    # Save final model
    save_checkpoint(
        model, 
        optimizer, 
        args, 
        step_count,
        train_dataloader.tokens_processed
    )
    
    logger.info("Training completed successfully!")
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a LLaMA-style Model with MLX on FineWeb Dataset")
    
    # Model architecture
    parser.add_argument("--dim", type=int, default=1024, 
                        help="Model dimension")
    parser.add_argument("--num_blocks", type=int, default=16, 
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, 
                        help="Number of attention heads")
    parser.add_argument("--checkpoint", action="store_true", 
                        help="Use gradient checkpointing to save memory")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="sample-10BT", 
                        help="FineWeb dataset config to use")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Tokenizer to use")
    parser.add_argument("--context_size", type=int, default=2048, 
                        help="Context window size")
    parser.add_argument("--max_tokens", type=int, default=2_500_000_000,  # 2.5B tokens
                        help="Maximum tokens to process during training")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                        help="AdamW learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, 
                        help="AdamW weight decay")
    parser.add_argument("--lr_warmup", type=int, default=2000, 
                        help="Learning rate warmup steps")
    parser.add_argument("--lr_schedule", type=str, choices=["linear", "cosine"], default="cosine", 
                        help="Learning rate schedule")
    parser.add_argument("--num_iters", type=int, default=0, 
                        help="Max training iterations (0 = no limit, use max_tokens instead)")
    
    # Training logistics
    parser.add_argument("--steps_per_report", type=int, default=10, 
                        help="Steps between training reports")
    parser.add_argument("--steps_per_eval", type=int, default=1000, 
                        help="Steps between evaluations and checkpoints")
    parser.add_argument("--output_dir", type=str, default="./llama_mlx_fineweb", 
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--resume_from", type=str, default="", 
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--gpu", action="store_true", 
                        help="Use Metal GPU backend")
    parser.add_argument("--debug", action="store_true",
                        help="Enable extra debugging information")
    
    # Weights & Biases integration
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llama-mlx",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="",
                        help="W&B run name (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    
    # Set device
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    else:
        logger.info("Using Metal GPU backend")
    
    main(args)
