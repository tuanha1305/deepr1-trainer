from typing import Dict, List, Tuple
import torch
import random
from dataclasses import dataclass

@dataclass
class PretrainingBatch:
    """Batch of data for pretraining"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

class PretrainingCollator:
    """Collate examples for pretraining tasks"""
    def __init__(
        self,
        mlm_probability: float = 0.15,
        pad_token_id: int = 0,
        mask_token_id: int = 103
    ):
        self.mlm_probability = mlm_probability
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
    
    def __call__(self, examples: List[Dict]) -> PretrainingBatch:
        """Create masked language modeling batch"""
        # Stack inputs
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])
        
        # Create MLM inputs and labels
        inputs, labels = self.mask_tokens(input_ids)
        
        return PretrainingBatch(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def mask_tokens(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling
        
        Args:
            inputs: Input token ids [batch_size, seq_length]
            
        Returns:
            Tuple of:
                - Masked input ids
                - Labels with -100 for non-masked tokens
        """
        labels = inputs.clone()
        
        # Sample tokens to mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Ignore padding tokens
        masked_indices[inputs == self.pad_token_id] = False
        
        # Mask input tokens
        inputs[masked_indices] = self.mask_token_id
        
        # Set labels
        labels[~masked_indices] = -100  # Ignore non-masked tokens in loss
        
        return inputs, labels
        