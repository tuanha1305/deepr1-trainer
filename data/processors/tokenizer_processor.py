from typing import Any, Dict, List, Optional, Union
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from dataclasses import dataclass, field

from .base_processor import BaseProcessor, ProcessorConfig, ProcessingError


@dataclass
class TokenizerConfig(ProcessorConfig):
    """Configuration for tokenizer processor"""
    # Tokenizer settings
    tokenizer_name: str = "meta-llama/Llama-2-7b"
    padding: bool = True
    truncation: bool = True
    max_length: int = 512
    return_tensors: str = "pt"

    # Special token handling
    add_special_tokens: bool = True
    add_prefix_space: bool = False

    # Preprocessing
    clean_text: bool = True
    handle_chinese_chars: bool = True
    strip_accents: Optional[bool] = None

    # Advanced options
    model_max_length: Optional[int] = None
    padding_side: str = "right"
    truncation_side: str = "right"

    # Special tokens
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    pad_token: Optional[str] = None
    unk_token: Optional[str] = None

    # Cache settings
    use_cache: bool = True
    cache_dir: Optional[str] = None


class TokenizerProcessor(BaseProcessor):
    """
    Processor for text tokenization using modern tokenizers

    Attributes:
        tokenizer: Underlying tokenizer instance
        config: Tokenizer configuration
    """

    def __init__(
            self,
            config: Optional[Union[Dict[str, Any], TokenizerConfig]] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        # Convert dict config to TokenizerConfig if needed
        if isinstance(config, dict):
            config = TokenizerConfig(**config)
        else:
            config = config or TokenizerConfig()

        super().__init__(config)

        # Use provided tokenizer or load from config
        self.tokenizer = tokenizer or self._load_tokenizer()

        # Update tokenizer configuration
        self._configure_tokenizer()

        # Initialize cache if needed
        self._initialize_cache()

    def _initialize(self):
        """Initialize processor specific resources"""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer from config"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name,
                cache_dir=self.config.cache_dir,
                add_prefix_space=self.config.add_prefix_space,
                model_max_length=self.config.model_max_length or self.config.max_length
            )

            # Add special tokens if specified
            special_tokens = {}
            if self.config.bos_token:
                special_tokens['bos_token'] = self.config.bos_token
            if self.config.eos_token:
                special_tokens['eos_token'] = self.config.eos_token
            if self.config.pad_token:
                special_tokens['pad_token'] = self.config.pad_token
            if self.config.unk_token:
                special_tokens['unk_token'] = self.config.unk_token

            if special_tokens:
                tokenizer.add_special_tokens(special_tokens)

            return tokenizer

        except Exception as e:
            raise ProcessingError(f"Failed to load tokenizer: {str(e)}")

    def _configure_tokenizer(self):
        """Configure tokenizer settings"""
        self.tokenizer.padding_side = self.config.padding_side
        self.tokenizer.truncation_side = self.config.truncation_side

        # Set default parameters for tokenizer calls
        self.tokenizer_kwargs = {
            "padding": self.config.padding,
            "truncation": self.config.truncation,
            "max_length": self.config.max_length,
            "return_tensors": self.config.return_tensors,
            "add_special_tokens": self.config.add_special_tokens
        }

    def _initialize_cache(self):
        """Initialize tokenizer cache"""
        if self.config.use_cache:
            self.cache = {}
            self.cache_hits = 0
            self.cache_misses = 0

    def __call__(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text

        Args:
            text: Input text or list of texts to tokenize

        Returns:
            Dictionary containing:
                - input_ids: Tensor of token ids
                - attention_mask: Tensor of attention masks
                - token_type_ids: Tensor of token type ids (if applicable)
        """
        # Check cache first
        if self.config.use_cache and isinstance(text, str):
            cached = self.cache.get(text)
            if cached is not None:
                self.cache_hits += 1
                return cached
            self.cache_misses += 1

        try:
            # Tokenize
            outputs = self.tokenizer(
                text,
                **self.tokenizer_kwargs
            )

            # Update cache
            if self.config.use_cache and isinstance(text, str):
                self.cache[text] = outputs

            return outputs

        except Exception as e:
            raise ProcessingError(f"Tokenization failed: {str(e)}")

    def decode(
            self,
            token_ids: Union[torch.Tensor, List[int]],
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        Decode token ids back to text

        Args:
            token_ids: Token ids to decode
            skip_special_tokens: Whether to remove special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces

        Returns:
            Decoded text
        """
        try:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
        except Exception as e:
            raise ProcessingError(f"Decoding failed: {str(e)}")

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        if not self.config.use_cache:
            return {}
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }

    def clear_cache(self):
        """Clear tokenizer cache"""
        if self.config.use_cache:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0

    def get_special_tokens_mask(self, token_ids: List[int]) -> List[int]:
        """Get mask indicating special tokens"""
        return self.tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens=True)

    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)