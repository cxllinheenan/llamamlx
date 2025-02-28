# data.py - Data loading utilities for training with FineWeb dataset

import logging
from typing import Iterator, Tuple, Optional

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class FineWebDataLoader:
    """Data loader for FineWeb dataset from Hugging Face."""
    
    def __init__(
        self, 
        tokenizer_name: str,
        context_size: int, 
        batch_size: int,
        max_tokens: Optional[int] = None,
        dataset_name: str = "sample-10BT",
        seed: int = 42
    ):
        self.context_size = context_size
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = len(self.tokenizer)
        self.tokens_processed = 0
        
        logger.info(f"Loading FineWeb dataset ({dataset_name})...")
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb", 
            name=dataset_name, 
            split="train", 
            streaming=True
        )
        # Shuffle with buffer_size for streaming datasets
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10000)
        
        logger.info(f"Tokenizer vocabulary size: {self.vocab_size}")
    
    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=False, padding=False)
    
    def _create_samples(self, token_ids, context_size):
        """Create context-target pairs from tokenized text."""
        # For causal language modeling, inputs are first n-1 tokens, targets are last n-1
        for i in range(0, len(token_ids) - context_size, context_size // 2):  # 50% overlap
            chunk = token_ids[i:i + context_size + 1]  # +1 to include the next token as target
            if len(chunk) < context_size + 1:  # Skip if too short
                continue
            yield chunk[:-1], chunk[1:]
    
    def __iter__(self) -> Iterator[Tuple[mx.array, mx.array]]:
        """Iterate through batches of data."""
        buffer_inputs, buffer_targets = [], []
        
        for example in self.dataset:
            # Skip empty examples
            if not example["text"].strip():
                continue
                
            # Tokenize the text
            tokens = self.tokenizer.encode(example["text"])
            
            # Create samples from the tokenized text
            for inputs, targets in self._create_samples(tokens, self.context_size):
                buffer_inputs.append(inputs)
                buffer_targets.append(targets)
                
                # When we have enough for a batch, yield it
                if len(buffer_inputs) >= self.batch_size:
                    # Convert to arrays
                    inputs_array = mx.array(buffer_inputs[:self.batch_size])
                    targets_array = mx.array(buffer_targets[:self.batch_size])
                    
                    # Update tokens processed count
                    tokens_in_batch = inputs_array.size
                    self.tokens_processed += tokens_in_batch
                    
                    # Check if we've hit the max tokens limit
                    if self.max_tokens and self.tokens_processed >= self.max_tokens:
                        logger.info(f"Reached max tokens limit: {self.tokens_processed}")
                        return
                    
                    # Clear the buffers and keep any extras for the next batch
                    buffer_inputs = buffer_inputs[self.batch_size:]
                    buffer_targets = buffer_targets[self.batch_size:]
                    
                    yield inputs_array, targets_array


def evaluate_model(model, eval_dataloader, num_eval_batches=50):
    """Evaluate the model on validation data."""
    import mlx.nn as nn
    import math
    
    total_loss = 0.0
    total_tokens = 0
    
    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="sum")
    
    for i, (inputs, targets) in enumerate(eval_dataloader):
        if i >= num_eval_batches:
            break
            
        loss = loss_fn(model, inputs, targets)
        total_loss += loss.item()
        total_tokens += targets.size
        
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity
