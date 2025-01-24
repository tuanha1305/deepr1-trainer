from typing import Any, Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
import re
import string
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import contractions
import emoji
from textblob import TextBlob
from collections import Counter

from .base_processor import BaseProcessor, ProcessorConfig, ProcessingError
from .basic_text_processor import BasicTextProcessor, BasicTextConfig


@dataclass
class AdvancedTextConfig(BasicTextConfig):
    """Configuration for advanced text processor"""
    # NLP settings
    use_spacy: bool = True
    spacy_model: str = "en_core_web_sm"
    use_lemmatization: bool = True
    use_stemming: bool = False

    # Entity handling
    extract_entities: bool = True
    entity_types: List[str] = field(default_factory=lambda: ["PERSON", "ORG", "LOC"])
    anonymize_entities: bool = False

    # Content analysis
    analyze_sentiment: bool = False
    detect_language: bool = False
    extract_keywords: bool = False
    num_keywords: int = 5

    # Additional processing
    expand_contractions: bool = True
    handle_emojis: bool = True
    emoji_replacement: str = ""
    remove_hashtags: bool = False
    remove_mentions: bool = False

    # Advanced text cleaning
    correct_spelling: bool = False
    remove_repeating_chars: bool = True
    max_repeating_chars: int = 2

    # Content filtering
    profanity_filter: bool = False
    profanity_list: Optional[str] = None
    min_word_length: int = 2


class AdvancedTextProcessor(BaseProcessor):
    """
    Advanced text processor with NLP capabilities

    Attributes:
        config: Advanced text processing configuration
        nlp: Spacy NLP model
        basic_processor: Basic text processor for initial cleaning
        lemmatizer: WordNet lemmatizer
        stemmer: Porter stemmer
        profanity_words: Set of profanity words
    """

    def __init__(
            self,
            config: Optional[Union[Dict[str, Any], AdvancedTextConfig]] = None
    ):
        if isinstance(config, dict):
            config = AdvancedTextConfig(**config)
        else:
            config = config or AdvancedTextConfig()

        super().__init__(config)

        # Initialize basic processor
        self.basic_processor = BasicTextProcessor(config)

        # Initialize NLP components
        if self.config.use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(self.config.spacy_model)
            except Exception as e:
                raise ProcessingError(f"Failed to load spaCy model: {str(e)}")

        # Initialize other components
        if self.config.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        if self.config.use_stemming:
            self.stemmer = PorterStemmer()

        # Load profanity list if needed
        if self.config.profanity_filter:
            self._load_profanity_list()

        # Compile regex patterns
        self._compile_patterns()

    def _initialize(self):
        """Initialize processor specific resources"""
        self.entities_found = []
        self.profanity_words = set()
        self.regex_patterns = {}

    def _compile_patterns(self):
        """Compile regex patterns"""
        self.regex_patterns.update({
            'hashtags': re.compile(r'#\w+'),
            'mentions': re.compile(r'@\w+'),
            'repeating_chars': re.compile(r'(.)\1+'),
            'emoji': re.compile(emoji.get_emoji_regexp())
        })

    def _load_profanity_list(self):
        """Load profanity words from file"""
        if self.config.profanity_list:
            try:
                with open(self.config.profanity_list, 'r') as f:
                    self.profanity_words = set(line.strip().lower() for line in f)
            except Exception as e:
                self.logger.warning(f"Failed to load profanity list: {str(e)}")

    def __call__(self, text: str) -> Dict[str, Any]:
        """
        Process text with advanced features

        Args:
            text: Input text to process

        Returns:
            Dictionary containing:
                - processed_text: Cleaned and processed text
                - entities: Found entities (if enabled)
                - sentiment: Sentiment analysis (if enabled)
                - language: Detected language (if enabled)
                - keywords: Extracted keywords (if enabled)
        """
        try:
            # Basic cleaning first
            text = self.basic_processor(text)

            # Advanced processing
            processed = self._process_text(text)

            # Additional analysis
            results = {
                'processed_text': processed,
                'metadata': {}
            }

            # Add additional analyses if enabled
            if self.config.extract_entities:
                results['metadata']['entities'] = self._extract_entities(text)

            if self.config.analyze_sentiment:
                results['metadata']['sentiment'] = self._analyze_sentiment(text)

            if self.config.detect_language:
                results['metadata']['language'] = self._detect_language(text)

            if self.config.extract_keywords:
                results['metadata']['keywords'] = self._extract_keywords(text)

            return results

        except Exception as e:
            raise ProcessingError(f"Advanced processing failed: {str(e)}")

    def _process_text(self, text: str) -> str:
        """Main text processing pipeline"""
        # Handle contractions
        if self.config.expand_contractions:
            text = contractions.fix(text)

        # Handle emojis
        if self.config.handle_emojis:
            text = self.regex_patterns['emoji'].sub(self.config.emoji_replacement, text)

        # Remove hashtags and mentions
        if self.config.remove_hashtags:
            text = self.regex_patterns['hashtags'].sub('', text)
        if self.config.remove_mentions:
            text = self.regex_patterns['mentions'].sub('', text)

        # Handle repeating characters
        if self.config.remove_repeating_chars:
            text = self._remove_repeating_chars(text)

        # Spelling correction
        if self.config.correct_spelling:
            text = str(TextBlob(text).correct())

        # Lemmatization/Stemming
        if self.config.use_spacy:
            doc = self.nlp(text)
            if self.config.use_lemmatization:
                text = ' '.join([token.lemma_ for token in doc])
        elif self.config.use_lemmatization:
            words = word_tokenize(text)
            text = ' '.join([self.lemmatizer.lemmatize(word) for word in words])
        elif self.config.use_stemming:
            words = word_tokenize(text)
            text = ' '.join([self.stemmer.stem(word) for word in words])

        # Profanity filtering
        if self.config.profanity_filter:
            text = self._filter_profanity(text)

        return text

    def _remove_repeating_chars(self, text: str) -> str:
        """Remove repeating characters beyond threshold"""

        def replace(match):
            char = match.group(1)
            return char * min(len(match.group(0)), self.config.max_repeating_chars)

        return self.regex_patterns['repeating_chars'].sub(replace, text)

    def _filter_profanity(self, text: str) -> str:
        """Filter out profanity words"""
        if not self.profanity_words:
            return text
        words = text.split()
        return ' '.join(w for w in words if w.lower() not in self.profanity_words)

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities"""
        if not self.config.use_spacy:
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.config.entity_types:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

        return entities

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze text sentiment"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def _detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            return TextBlob(text).detect_language()
        except:
            return 'unknown'

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        if not self.config.use_spacy:
            return []

        doc = self.nlp(text)
        words = []

        # Get important words (nouns and proper nouns)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > self.config.min_word_length:
                words.append(token.text.lower())

        # Get most common words
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(self.config.num_keywords)]
