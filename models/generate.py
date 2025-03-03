# generate.py - Run inference with trained LLaMA MLX model
import os
import argparse
import logging
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer

# Import the model architecture from train.py
from train import LLaMAModel, silu, RMSNorm, RotaryEmbedding, LLaMAAttention, LLaMAFeedForward, LLaMABlock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, tokenizer_name, dim, num_blocks, num_heads):
    """Load the trained model from checkpoint."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    
    # Initialize model with the same architecture as during training
    ffn_dims = dim * 4
    model = LLaMAModel(
        vocab_size=vocab_size, 
        num_layers=num_blocks, 
        dims=dim, 
        num_heads=num_heads, 
        ffn_dims=ffn_dims,
        checkpoint=False  # Not needed for inference
    )
    
    # Load trained parameters
    model_path = os.path.join(checkpoint_path, "model.npz")
    if not os.path.exists(model_path):
        raise ValueError(f"Model checkpoint not found at {model_path}")
    
    # Load params and convert to MLX arrays
    logger.info(f"Loading model weights from {model_path}")
    params_dict = dict(np.load(model_path))
    params = {k: mx.array(v) for k, v in params_dict.items()}
    
    # Update model with loaded parameters
    model.update(params)
    logger.info(f"Model loaded successfully from {checkpoint_path}")
    
    return model, tokenizer


def generate(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate text from the model with the given prompt."""
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    
    # Store generated token ids
    generated_ids = input_ids.tolist()[0]
    
    logger.info(f"Generating with prompt: {prompt}")
    
    # Generation loop
    for _ in range(max_length):
        # Get input context (limit to the model's context size if needed)
        inputs = mx.array([generated_ids[-min(len(generated_ids), 2048):]])
        
        # Forward pass
        logits = model(inputs)
        
        # Get the logits for the last token
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Sample from distribution
        probs = mx.softmax(next_token_logits)
        next_token = int(mx.random.categorical(probs, num_samples=1)[0])
        
        # Append to generated sequence
        generated_ids.append(next_token)
        
        # Stop if we predict an end of sequence token
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser("Generate text with trained LLaMA model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the checkpoint directory")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Tokenizer to use")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_blocks", type=int, default=4, 
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, 
                        help="Number of attention heads")
    parser.add_argument("--prompt", type=str, default="Once upon a time", 
                        help="Prompt to generate from")
    parser.add_argument("--max_length", type=int, default=200, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature")
    parser.add_argument("--gpu", action="store_true", 
                        help="Use Metal GPU backend")
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu:
        mx.set_default_device(mx.gpu)
        logger.info("Using GPU for inference")
    else:
        mx.set_default_device(mx.cpu)
        logger.info("Using CPU for inference")
    
    # Load model and tokenizer
    model, tokenizer = load_model(
        args.checkpoint, 
        args.tokenizer, 
        args.dim, 
        args.num_blocks, 
        args.num_heads
    )
    
    # Generate text - removed the top_p parameter
    generated_text = generate(
        model, 
        tokenizer, 
        args.prompt, 
        max_length=args.max_length, 
        temperature=args.temperature
    )
    
    print("\nGenerated text:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)