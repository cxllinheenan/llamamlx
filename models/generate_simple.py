# generate_simple.py - Enhanced inference script for LLaMA MLX model
import os
import argparse
import logging
import json
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer

# Import model from train.py
from train import LLaMAModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_path, model):
    """Load model weights and optimizer state from checkpoint."""
    checkpoint_info = {
        "model_loaded": False,
        "optimizer_loaded": False,
        "training_info": None
    }
    
    # Load model parameters
    model_path = os.path.join(checkpoint_path, "model.npz")
    print(f"Loading model weights from {model_path}")
    
    if os.path.exists(model_path):
        try:
            params = dict(np.load(model_path))
            params = {k: mx.array(v) for k, v in params.items()}
            model.update(params)
            print("Model weights loaded successfully")
            checkpoint_info["model_loaded"] = True
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
    else:
        print(f"WARNING: Model weights not found at {model_path}")
    
    # Load optimizer state (for reference)
    optimizer_path = os.path.join(checkpoint_path, "optimizer.npz")
    
    if os.path.exists(optimizer_path):
        try:
            print(f"Loading optimizer state from {optimizer_path}")
            opt_state = dict(np.load(optimizer_path))
            print(f"Optimizer state contains {len(opt_state)} parameters")
            checkpoint_info["optimizer_loaded"] = True
        except Exception as e:
            print(f"Error loading optimizer state: {str(e)}")
    
    # Load training info
    training_info_path = os.path.join(checkpoint_path, "training_info.json")
    if os.path.exists(training_info_path):
        try:
            with open(training_info_path, "r") as f:
                training_info = json.load(f)
            print(f"Loaded training info from {training_info_path}")
            checkpoint_info["training_info"] = training_info
            
            # Print some useful information from the training info
            if "step" in training_info:
                print(f"Checkpoint step: {training_info['step']}")
            if "tokens_processed" in training_info:
                print(f"Tokens processed: {training_info['tokens_processed']:,}")
                
        except Exception as e:
            print(f"Error loading training info: {str(e)}")
    
    return checkpoint_info

def main():
    parser = argparse.ArgumentParser("Generate text with trained LLaMA model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the checkpoint directory")
    parser.add_argument("--prompt", type=str, default="Once upon a time", 
                        help="Prompt to generate from")
    parser.add_argument("--max_tokens", type=int, default=50, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering threshold")
    parser.add_argument("--dim", type=int, default=256, 
                        help="Model dimension")
    parser.add_argument("--num_blocks", type=int, default=4, 
                        help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=4, 
                        help="Number of attention heads")
    parser.add_argument("--gpu", action="store_true", 
                        help="Use GPU for inference")
    
    args = parser.parse_args()
    print("Arguments parsed")
    
    # Set device
    if args.gpu:
        mx.set_default_device(mx.gpu)
        print("Using GPU for inference")
    else:
        mx.set_default_device(mx.cpu)
        print("Using CPU for inference")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    vocab_size = len(tokenizer)
    print(f"Tokenizer loaded with vocab size {vocab_size}")
    
    # Initialize model
    ffn_dims = args.dim * 4
    model = LLaMAModel(
        vocab_size=vocab_size,
        num_layers=args.num_blocks,
        dims=args.dim,
        num_heads=args.num_heads,
        ffn_dims=ffn_dims,
        checkpoint=False
    )
    print("Model initialized")
    
    # Load checkpoint
    checkpoint_info = load_checkpoint(args.checkpoint, model)
    
    if not checkpoint_info["model_loaded"]:
        print("Failed to load model weights. Exiting.")
        return
    
    # Generate text
    print(f"Generating text with prompt: '{args.prompt}'")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    print(f"Prompt tokenized to {input_ids.shape} tokens")
    
    # Convert to list for easier manipulation
    generated_ids = input_ids.tolist()[0]
    print(f"Initial tokens: {generated_ids}")
    
    # Generate one token at a time
    for i in range(args.max_tokens):
        print(f"Generating token {i+1}/{args.max_tokens}")
        
        # Prepare input for model
        inputs = mx.array([generated_ids])
        print(f"Input shape: {inputs.shape}")
        
        # Run model inference
        try:
            print("Running model inference...")
            logits = model(inputs)
            print(f"Logits shape: {logits.shape}")
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]
            print(f"Next token logits shape: {next_token_logits.shape}")
            
            # Apply temperature
            next_token_logits = next_token_logits / args.temperature
            
            # Apply top-k filtering
            if args.top_k > 0:
                # Convert to numpy for easier manipulation
                logits_np = next_token_logits.tolist()
                indices_np = np.argsort(logits_np)[::-1]  # Sort in descending order
                
                # Get top-k indices
                top_k_indices = indices_np[:args.top_k]
                
                # Create a filtered logits array
                filtered_logits = [-float('inf')] * len(logits_np)
                for idx in top_k_indices:
                    filtered_logits[idx] = logits_np[idx]
                
                # Convert back to MLX array
                next_token_logits = mx.array(filtered_logits)
                print(f"Applied top-k filtering with k={args.top_k}")
            
            # Sample from distribution
            print("Sampling next token...")
            probs = mx.softmax(next_token_logits)
            next_token = int(mx.random.categorical(probs, num_samples=1)[0])
            next_token_str = tokenizer.decode([next_token])
            print(f"Next token ID: {next_token} ('{next_token_str}')")
            
            # Add to generated ids
            generated_ids.append(next_token)
            
            # Stop if end of sequence token
            if next_token == tokenizer.eos_token_id:
                print("End of sequence token generated, stopping")
                break
                
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            break
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids)
    
    print("\nGenerated text:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)

if __name__ == "__main__":
    main()