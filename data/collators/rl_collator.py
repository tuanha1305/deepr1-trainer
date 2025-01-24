from typing import Dict, List
import torch
from dataclasses import dataclass

@dataclass
class RLBatch:
    """Batch of data for RL training"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: Optional[torch.Tensor] = None
    metadata: Optional[List[Dict]] = None

class RLCollator:
    """Collate examples into batches for RL training"""
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, examples: List[Dict]) -> RLBatch:
        """
        Collate batch of examples
        
        Args:
            examples: List of examples from dataset
            
        Returns:
            Batched tensors
        """
        # Extract features
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])
        
        # Handle targets if present
        target_ids = None
        if examples[0].get('target_ids') is not None:
            target_ids = torch.stack([ex['target_ids'] for ex in examples])
        
        # Collect metadata
        metadata = [ex.get('metadata', {}) for ex in examples]
        
        return RLBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ids=target_ids,
            metadata=metadata
        )
