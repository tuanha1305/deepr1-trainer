from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import torch
import random
import numpy as np
from transformers import PreTrainedTokenizer, AutoTokenizer

from .base_processor import BaseProcessor, ProcessorConfig, ProcessingError


@dataclass
class MLMConfig(ProcessorConfig):
    """Configuration for MLM text processor"""
    # Tokenizer settings
    tokenizer_name: str = "meta-llama/Llama-2-7b"
    max_length: int = 512
    padding: bool = True
    truncation: bool = True

    # Masking strategy
    mlm_probability: float = 0.15
    mask_probability: float = 0.8  # Probability of replacing with [MASK]
    random_probability: float = 0.1  # Probability of replacing with random token
    keep_probability: float = 0.1  # Probability of keeping original token

    # Special tokens & IDs
    mask_token: str = "[MASK]"
    pad_token: str = "[PAD]"
    mask_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    # Advanced options
    whole_word_masking: bool = True
    max_masks_per_seq: Optional[int] = None
    max_ngram_size: int = 3  # For span masking
    use_span_masking: bool = False

    # Special handling
    skip_special_tokens: bool = True
    preserve_tokens: List[str] = field(default_factory=list)


class MLMTextProcessor(BaseProcessor):
    """
    Processor for Masked Language Modeling (MLM) text processing

    Attributes:
        config: MLM configuration
        tokenizer: Pretrained tokenizer
        vocab_size: Size of tokenizer vocabulary
        special_tokens_mask: Mask for special tokens
    """

    def __init__(
            self,
            config: Optional[Union[Dict[str, Any], MLMConfig]] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        if isinstance(config, dict):
            config = MLMConfig(**config)
        else:
            config = config or MLMConfig()

        super().__init__(config)

        # Initialize tokenizer
        self.tokenizer = tokenizer or self._initialize_tokenizer()
        self.vocab_size = len(self.tokenizer)

        # Set token IDs if not provided
        if self.config.mask_token_id is None:
            self.config.mask_token_id = self.tokenizer.mask_token_id
        if self.config.pad_token_id is None:
            self.config.pad_token_id = self.tokenizer.pad_token_id

    def _initialize(self):
        """Initialize processor specific resources"""
        self.special_tokens_mask = None
        self.word_boundaries = {}

    def _initialize_tokenizer(self) -> PreTrainedTokenizer:
        """Initialize tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

            # Add special tokens if needed
            if self.config.mask_token not in tokenizer.special_tokens_map:
                tokenizer.add_special_tokens({'mask_token': self.config.mask_token})
            if self.config.pad_token not in tokenizer.special_tokens_map:
                tokenizer.add_special_tokens({'pad_token': self.config.pad_token})

            return tokenizer

        except Exception as e:
            raise ProcessingError(f"Failed to initialize tokenizer: {str(e)}")

    def __call__(
            self,
            text: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process text for MLM

        Args:
            text: Input text or list of texts

        Returns:
            Dictionary containing:
                - input_ids: Masked input tokens
                - attention_mask: Attention mask
                - labels: Original token ids
        """
        try:
            # Tokenize input
            encoding = self.tokenizer(
                text,
                padding=self.config.padding,
                truncation=self.config.truncation,
                max_length=self.config.max_length,
                return_tensors="pt"
            )

            # Get input IDs and attention mask
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # Create labels before masking (copy of input_ids)
            labels = input_ids.clone()

            # Get probability mask
            probability_matrix = self._get_mask_probability_matrix(input_ids, attention_mask)

            # Apply masking
            if self.config.use_span_masking:
                masked_inputs = self._apply_span_masking(
                    input_ids,
                    probability_matrix
                )
            else:
                masked_inputs = self._apply_token_masking(
                    input_ids,
                    probability_matrix
                )

            # Set labels for unmasked tokens to -100 (ignored in loss computation)
            labels[masked_inputs == input_ids] = -100

            return {
                "input_ids": masked_inputs,
                "attention_mask": attention_mask,
                "labels": labels
            }

        except Exception as e:
            raise ProcessingError(f"MLM processing failed: {str(e)}")

    def _get_mask_probability_matrix(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Create probability matrix for masking"""
        probability_matrix = torch.full(input_ids.shape, self.config.mlm_probability)

        # Create special tokens mask
        special_tokens_mask = self._get_special_tokens_mask(input_ids)

        # Don't mask special tokens
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Don't mask padding tokens
        padding_mask = input_ids.eq(self.config.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        # Don't mask preserved tokens
        if self.config.preserve_tokens:
            preserve_mask = self._get_preserve_tokens_mask(input_ids)
            probability_matrix.masked_fill_(preserve_mask, value=0.0)

        # Apply whole word masking if enabled
        if self.config.whole_word_masking:
            probability_matrix = self._apply_whole_word_mask(
                input_ids,
                probability_matrix
            )

        return probability_matrix

    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get mask indicating special tokens"""
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids.tolist(),
            already_has_special_tokens=True
        )
        return torch.tensor(special_tokens_mask, dtype=torch.bool)

    def _get_preserve_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get mask for preserved tokens"""
        preserve_ids = set(
            self.tokenizer.convert_tokens_to_ids(self.config.preserve_tokens)
        )
        preserve_mask = torch.tensor(
            [[token_id in preserve_ids for token_id in seq] for seq in input_ids],
            dtype=torch.bool
        )
        return preserve_mask

    def _apply_whole_word_mask(
            self,
            input_ids: torch.Tensor,
            probability_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Apply whole word masking"""
        for i in range(input_ids.size(0)):
            word_ids = self.tokenizer.get_word_ids(input_ids[i].tolist())

            # Group tokens by word ID
            word_groups = {}
            for j, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id not in word_groups:
                        word_groups[word_id] = []
                    word_groups[word_id].append(j)

            # Apply same probability to all tokens in a word
            for positions in word_groups.values():
                prob = probability_matrix[i, positions[0]].item()
                probability_matrix[i, positions] = prob

        return probability_matrix

    def _apply_token_masking(
            self,
            input_ids: torch.Tensor,
            probability_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Apply token-level masking"""
        # Create masking matrix
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Limit number of masks if specified
        if self.config.max_masks_per_seq:
            for i in range(masked_indices.size(0)):
                num_masks = masked_indices[i].sum()
                if num_masks > self.config.max_masks_per_seq:
                    # Randomly remove excess masks
                    masked_positions = masked_indices[i].nonzero().squeeze()
                    remove_positions = masked_positions[
                        torch.randperm(len(masked_positions))
                        [:int(num_masks - self.config.max_masks_per_seq)]
                    ]
                    masked_indices[i, remove_positions] = False

        # Create output tensor
        masked_inputs = input_ids.clone()

        # Get indices for different replacement strategies
        mask_indices = masked_indices & torch.bernoulli(
            torch.full(input_ids.shape, self.config.mask_probability)
        ).bool()
        random_indices = masked_indices & ~mask_indices & torch.bernoulli(
            torch.full(input_ids.shape, self.config.random_probability)
        ).bool()
        keep_indices = masked_indices & ~mask_indices & ~random_indices

        # Apply masking strategies
        masked_inputs[mask_indices] = self.config.mask_token_id
        random_words = torch.randint(
            self.vocab_size,
            input_ids.shape,
            dtype=torch.long
        )
        masked_inputs[random_indices] = random_words[random_indices]

        return masked_inputs

    def _apply_span_masking(
            self,
            input_ids: torch.Tensor,
            probability_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Apply span-based masking"""
        masked_inputs = input_ids.clone()

        for i in range(input_ids.size(0)):
            # Get positions eligible for masking
            eligible_positions = (~self._get_special_tokens_mask(input_ids[i])).nonzero().squeeze()

            if len(eligible_positions) == 0:
                continue

            # Calculate number of spans to mask
            num_tokens_to_mask = int(len(eligible_positions) * self.config.mlm_probability)
            if num_tokens_to_mask == 0:
                continue

            # Generate spans
            spans = []
            current_span = []

            while sum(len(span) for span in spans) < num_tokens_to_mask:
                # Randomly select span size
                span_size = min(
                    random.randint(1, self.config.max_ngram_size),
                    num_tokens_to_mask - sum(len(span) for span in spans)
                )

                # Select random start position
                if len(eligible_positions) == 0:
                    break

                start_idx = random.choice(eligible_positions.tolist())
                span = list(range(start_idx, min(start_idx + span_size, input_ids.size(1))))
                spans.append(span)

                # Remove used positions
                eligible_positions = eligible_positions[
                    ~torch.tensor([pos in span for pos in eligible_positions.tolist()])
                ]

            # Apply masking to spans
            for span in spans:
                # Decide masking strategy for span
                if random.random() < self.config.mask_probability:
                    masked_inputs[i, span] = self.config.mask_token_id
                elif random.random() < self.config.random_probability:
                    masked_inputs[i, span] = torch.randint(
                        self.vocab_size,
                        (len(span),),
                        dtype=torch.long
                    )
                # else keep original tokens

        return masked_inputs
