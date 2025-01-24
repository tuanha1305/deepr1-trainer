from typing import Dict, Optional
import re
import html
import unicodedata


class TextProcessor:
    """Process text data for training"""

    def __init__(
            self,
            lowercase: bool = True,
            normalize_unicode: bool = True,
            remove_html: bool = True,
            max_length: Optional[int] = None
    ):
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.remove_html = remove_html
        self.max_length = max_length

    def __call__(self, example: Dict) -> Optional[Dict]:
        """
        Process a single example

        Returns:
            Processed example or None if example should be filtered out
        """
        # Get text fields
        input_text = example.get('input_text', '')
        target_text = example.get('target_text', '')

        # Basic processing
        input_text = self._clean_text(input_text)
        target_text = self._clean_text(target_text)

        # Length filtering
        if self.max_length and (
                len(input_text.split()) > self.max_length or
                len(target_text.split()) > self.max_length
        ):
            return None

        # Return processed example
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': example.get('metadata', {})
        }

    def _clean_text(self, text: str) -> str:
        """Clean text with selected options"""
        if self.remove_html:
            text = html.unescape(text)
            text = re.sub(r'<[^>]+>', '', text)

        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)

        if self.lowercase:
            text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
