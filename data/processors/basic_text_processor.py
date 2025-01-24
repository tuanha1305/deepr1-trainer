from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import re
import html
import unicodedata
import string
from pathlib import Path
import json

from .base_processor import BaseProcessor, ProcessorConfig, ProcessingError


@dataclass
class BasicTextConfig(ProcessorConfig):
    """Configuration for basic text processor"""
    # Text normalization
    lowercase: bool = True
    strip_whitespace: bool = True
    normalize_unicode: bool = True

    # Character handling
    remove_punctuation: bool = False
    remove_numbers: bool = False
    remove_special_chars: bool = False

    # HTML handling
    remove_html: bool = True
    unescape_html: bool = True

    # URL handling
    remove_urls: bool = True
    url_replacement: str = "[URL]"

    # Email handling
    remove_emails: bool = True
    email_replacement: str = "[EMAIL]"

    # Length constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    # Custom replacements
    custom_replacements: Dict[str, str] = field(default_factory=dict)

    # Stop words
    remove_stopwords: bool = False
    stopwords_file: Optional[str] = None
    language: str = "english"


class BasicTextProcessor(BaseProcessor):
    """
    Basic text processor for cleaning and normalizing text

    Attributes:
        config: Text processing configuration
        stopwords: Set of stopwords if enabled
        regex_patterns: Compiled regex patterns
    """

    def __init__(
            self,
            config: Optional[Union[Dict[str, Any], BasicTextConfig]] = None
    ):
        if isinstance(config, dict):
            config = BasicTextConfig(**config)
        else:
            config = config or BasicTextConfig()

        super().__init__(config)

        # Initialize regex patterns
        self._compile_regex_patterns()

        # Load stopwords if needed
        self._load_stopwords()

    def _initialize(self):
        """Initialize processor specific resources"""
        self.stopwords = set()
        self.regex_patterns = {}

    def _compile_regex_patterns(self):
        """Compile regex patterns for text processing"""
        self.regex_patterns = {
            'url': re.compile(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            ),
            'email': re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
            'special_chars': re.compile(r'[^a-zA-Z0-9\s]'),
            'numbers': re.compile(r'\d+'),
            'multiple_spaces': re.compile(r'\s+'),
            'html_tags': re.compile(r'<[^>]+>')
        }

    def _load_stopwords(self):
        """Load stopwords from file or use default set"""
        if not self.config.remove_stopwords:
            return

        try:
            if self.config.stopwords_file:
                # Load custom stopwords file
                with open(self.config.stopwords_file, 'r', encoding='utf-8') as f:
                    self.stopwords = set(line.strip() for line in f)
            else:
                # Use default NLTK stopwords
                import nltk
                try:
                    nltk.download('stopwords', quiet=True)
                    from nltk.corpus import stopwords
                    self.stopwords = set(stopwords.words(self.config.language))
                except Exception as e:
                    self.logger.warning(f"Failed to load NLTK stopwords: {str(e)}")
                    self.stopwords = set()

        except Exception as e:
            raise ProcessingError(f"Failed to load stopwords: {str(e)}")

    def __call__(self, text: str) -> str:
        """
        Process input text

        Args:
            text: Input text to process

        Returns:
            Processed text

        Raises:
            ProcessingError: If processing fails
        """
        if not isinstance(text, str):
            raise ProcessingError(f"Expected string input, got {type(text)}")

        try:
            # Apply text transformations
            text = self._preprocess_text(text)
            text = self._clean_text(text)
            text = self._postprocess_text(text)

            # Validate length constraints
            if not self._validate_length(text):
                return ""

            return text

        except Exception as e:
            raise ProcessingError(f"Text processing failed: {str(e)}")

    def _preprocess_text(self, text: str) -> str:
        """Initial text preprocessing"""
        # Handle HTML
        if self.config.remove_html:
            text = self.regex_patterns['html_tags'].sub(' ', text)
        if self.config.unescape_html:
            text = html.unescape(text)

        # Handle URLs and emails
        if self.config.remove_urls:
            text = self.regex_patterns['url'].sub(self.config.url_replacement, text)
        if self.config.remove_emails:
            text = self.regex_patterns['email'].sub(self.config.email_replacement, text)

        return text

    def _clean_text(self, text: str) -> str:
        """Main text cleaning"""
        # Normalize unicode
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)

        # Apply case transformation
        if self.config.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers
        if self.config.remove_numbers:
            text = self.regex_patterns['numbers'].sub('', text)

        # Remove special characters
        if self.config.remove_special_chars:
            text = self.regex_patterns['special_chars'].sub('', text)

        # Apply custom replacements
        for old, new in self.config.custom_replacements.items():
            text = text.replace(old, new)

        return text

    def _postprocess_text(self, text: str) -> str:
        """Final text postprocessing"""
        # Remove stopwords
        if self.config.remove_stopwords and self.stopwords:
            words = text.split()
            words = [w for w in words if w.lower() not in self.stopwords]
            text = ' '.join(words)

        # Clean whitespace
        if self.config.strip_whitespace:
            text = self.regex_patterns['multiple_spaces'].sub(' ', text)
            text = text.strip()

        return text

    def _validate_length(self, text: str) -> bool:
        """Validate text length constraints"""
        if self.config.min_length and len(text) < self.config.min_length:
            return False
        if self.config.max_length and len(text) > self.config.max_length:
            return False
        return True

    def save_stopwords(self, filepath: str):
        """Save current stopwords to file"""
        if self.stopwords:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(list(self.stopwords), f, ensure_ascii=False, indent=2)

    def add_stopwords(self, words: Union[str, List[str]]):
        """Add new stopwords"""
        if isinstance(words, str):
            words = [words]
        self.stopwords.update(words)

    def remove_stopwords(self, words: Union[str, List[str]]):
        """Remove words from stopwords"""
        if isinstance(words, str):
            words = [words]
        self.stopwords.difference_update(words)
