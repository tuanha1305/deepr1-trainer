import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
import json
import random
from .processors.text_processor import TextProcessor


class RLDataset(Dataset):
    """Dataset for reinforcement learning tasks"""

    def __init__(
            self,
            data_path: str,
            tokenizer,
            processor: Optional[TextProcessor] = None,
            max_length: int = 512,
            cache_dir: Optional[str] = None
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor or TextProcessor()
        self.max_length = max_length

        # Load and process data
        self.examples = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load data from file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        # Process examples
        processed_data = []
        for example in data:
            processed = self.processor(example)
            if processed is not None:
                processed_data.append(processed)

        return processed_data

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example"""
        example = self.examples[idx]

        # Tokenize input
        inputs = self.tokenizer(
            example['input_text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Prepare target if available
        target = None
        if 'target_text' in example:
            target = self.tokenizer(
                example['target_text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'target_ids': target['input_ids'].squeeze(0) if target else None,
            'metadata': example.get('metadata', {})
        }


class TextDataset(Dataset):
    """Dataset for text processing tasks"""

    def __init__(
            self,
            texts: Union[List[str], str],
            tokenizer,
            max_length: int = 512,
            chunk_size: Optional[int] = None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size

        # Load texts
        if isinstance(texts, str):
            with open(texts, 'r', encoding='utf-8') as f:
                self.texts = f.readlines()
        else:
            self.texts = texts

        # Process into chunks if needed
        if chunk_size:
            self.chunks = self._create_chunks()
        else:
            self.chunks = self.texts

    def _create_chunks(self) -> List[str]:
        """Create overlapping chunks of text"""
        chunks = []
        for text in self.texts:
            words = text.split()
            for i in range(0, len(words), self.chunk_size // 2):
                chunk = ' '.join(words[i:i + self.chunk_size])
                if len(chunk.split()) >= self.chunk_size // 2:  # Avoid too small chunks
                    chunks.append(chunk)
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single chunk of text"""
        text = self.chunks[idx]

        # Tokenize
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'text': text
        }
