from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import random
import nltk
from nltk.corpus import wordnet
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .base_processor import BaseProcessor, ProcessorConfig, ProcessingError


@dataclass
class AugmentationConfig(ProcessorConfig):
    """Configuration for data augmentation processor"""
    # General settings
    augmentation_methods: List[str] = field(default_factory=lambda: ["synonym", "back_translation"])
    num_augmentations: int = 2
    prob_augment: float = 0.7

    # Synonym replacement
    synonym_prob: float = 0.3
    max_synonym_replacements: int = 3

    # Back translation
    translation_languages: List[str] = field(default_factory=lambda: ["de", "fr"])
    translation_temperature: float = 0.7

    # Word deletion/insertion
    word_delete_prob: float = 0.1
    word_insert_prob: float = 0.1

    # MLM-based augmentation
    mlm_model_name: str = "bert-base-uncased"
    mlm_mask_prob: float = 0.15
    mlm_top_k: int = 5

    # Preserve words
    preserve_words: List[str] = field(default_factory=list)


class DataAugmentationProcessor(BaseProcessor):
    """
    Processor for text data augmentation using various strategies

    Attributes:
        config: Augmentation configuration
        augmentation_methods: Dictionary of augmentation functions
        mlm_model: Masked language model for MLM-based augmentation
        mlm_tokenizer: Tokenizer for MLM model
    """

    def __init__(
            self,
            config: Optional[Union[Dict[str, Any], AugmentationConfig]] = None
    ):
        if isinstance(config, dict):
            config = AugmentationConfig(**config)
        else:
            config = config or AugmentationConfig()

        super().__init__(config)

        # Initialize NLTK
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            self.logger.warning(f"Failed to download NLTK data: {str(e)}")

        # Initialize MLM model if needed
        self.mlm_model = None
        self.mlm_tokenizer = None
        if "mlm" in self.config.augmentation_methods:
            self._initialize_mlm()

    def _initialize(self):
        """Initialize augmentation methods"""
        self.augmentation_methods = {
            "synonym": self._synonym_replacement,
            "back_translation": self._back_translation,
            "word_deletion": self._word_deletion,
            "word_insertion": self._word_insertion,
            "mlm": self._mlm_augmentation
        }

    def _initialize_mlm(self):
        """Initialize masked language model"""
        try:
            self.mlm_model = AutoModelForMaskedLM.from_pretrained(
                self.config.mlm_model_name
            )
            self.mlm_tokenizer = AutoTokenizer.from_pretrained(
                self.config.mlm_model_name
            )
            self.mlm_model.eval()
        except Exception as e:
            raise ProcessingError(f"Failed to load MLM model: {str(e)}")

    def __call__(self, text: str) -> List[str]:
        """
        Augment input text

        Args:
            text: Input text to augment

        Returns:
            List of augmented texts
        """
        if random.random() > self.config.prob_augment:
            return [text]

        augmented_texts = []

        try:
            # Apply different augmentation methods
            for _ in range(self.config.num_augmentations):
                # Randomly select augmentation method
                method = random.choice(self.config.augmentation_methods)
                augmented = self.augmentation_methods[method](text)
                if augmented and augmented != text:
                    augmented_texts.append(augmented)

            # Add original text if no augmentations were successful
            if not augmented_texts:
                augmented_texts.append(text)

            return augmented_texts

        except Exception as e:
            raise ProcessingError(f"Augmentation failed: {str(e)}")

    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms"""
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)

        # Get replaceable words (exclude preserved words)
        replaceable = [
            (i, word) for i, (word, pos) in enumerate(pos_tags)
            if word.isalnum() and word not in self.config.preserve_words
        ]

        if not replaceable:
            return text

        # Randomly select words to replace
        num_replacements = min(
            len(replaceable),
            random.randint(1, self.config.max_synonym_replacements)
        )

        replace_indices = random.sample(range(len(replaceable)), num_replacements)

        for idx in replace_indices:
            word_idx, word = replaceable[idx]
            synonyms = []

            # Get synonyms from WordNet
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name())

            if synonyms:
                words[word_idx] = random.choice(synonyms)

        return ' '.join(words)

    def _back_translation(self, text: str) -> str:
        """
        Augment text using back translation
        Note: This is a mock implementation. In practice, you would use a translation API
        or a translation model.
        """
        try:
            # Mock translation (replace with actual translation in production)
            translated = text
            for lang in self.config.translation_languages:
                # Simulate translation noise
                words = text.split()
                if len(words) > 3:
                    # Randomly modify word order or drop words
                    if random.random() < 0.3:
                        random.shuffle(words)
                    if random.random() < 0.2:
                        words = words[:-1]
                translated = ' '.join(words)

            return translated
        except Exception:
            return text

    def _word_deletion(self, text: str) -> str:
        """Randomly delete words"""
        words = text.split()
        if len(words) <= 3:
            return text

        kept_words = []
        for word in words:
            if random.random() > self.config.word_delete_prob or word in self.config.preserve_words:
                kept_words.append(word)

        if not kept_words:
            return text

        return ' '.join(kept_words)

    def _word_insertion(self, text: str) -> str:
        """Insert random words from vocabulary"""
        words = text.split()
        if not words:
            return text

        # Get random words from WordNet
        random_words = []
        for _ in range(2):
            synset = random.choice(list(wordnet.all_synsets()))
            random_words.append(synset.lemmas()[0].name())

        # Insert random words
        augmented_words = []
        for word in words:
            augmented_words.append(word)
            if random.random() < self.config.word_insert_prob:
                augmented_words.append(random.choice(random_words))

        return ' '.join(augmented_words)

    def _mlm_augmentation(self, text: str) -> str:
        """Augment text using masked language model"""
        if not self.mlm_model or not self.mlm_tokenizer:
            return text

        try:
            # Tokenize
            tokens = self.mlm_tokenizer.tokenize(text)
            if len(tokens) <= 2:
                return text

            # Randomly mask tokens
            mask_token = self.mlm_tokenizer.mask_token
            masked_tokens = tokens.copy()

            for i in range(len(tokens)):
                if random.random() < self.config.mlm_mask_prob:
                    masked_tokens[i] = mask_token

            # Get model predictions
            inputs = self.mlm_tokenizer.encode(
                ' '.join(masked_tokens),
                return_tensors='pt'
            )
            outputs = self.mlm_model(inputs)[0]

            # Replace masks with predictions
            for i, token in enumerate(masked_tokens):
                if token == mask_token:
                    pred_id = outputs[0, i].topk(self.config.mlm_top_k).indices[0]
                    masked_tokens[i] = self.mlm_tokenizer.convert_ids_to_tokens([pred_id])[0]

            return self.mlm_tokenizer.convert_tokens_to_string(masked_tokens)

        except Exception:
            return text
