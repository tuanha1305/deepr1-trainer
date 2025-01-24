from typing import Optional, Dict, Any, List
import re
from .base_reward import BaseReward


class FormatReward(BaseReward):
    """Reward based on output format adherence"""

    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None,
            required_formats: Optional[List[str]] = None,
            format_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(config)
        self.required_formats = required_formats or ["<think>", "</think>"]
        self.format_weights = format_weights or {fmt: 1.0 for fmt in self.required_formats}

    def __call__(
            self,
            response: str,
            **kwargs
    ) -> float:
        """
        Calculate format adherence reward

        Args:
            response: Generated text response
        """
        total_reward = 0.0
        max_reward = sum(self.format_weights.values())

        # Check each required format
        for fmt, weight in self.format_weights.items():
            if fmt in response:
                total_reward += weight

                # Check for proper tag closure if it's an XML-like tag
                if fmt.startswith('<') and fmt.endswith('>'):
                    tag_name = fmt[1:-1]
                    closing_tag = f"</{tag_name}>"
                    if closing_tag in response:
                        # Bonus for proper tag closure
                        total_reward += 0.2 * weight

        # Additional format checks
        reward_multiplier = 1.0

        # Check for balanced tags
        if self._check_balanced_tags(response):
            reward_multiplier *= 1.2

        # Check for proper nesting
        if self._check_proper_nesting(response):
            reward_multiplier *= 1.1

        # Normalize reward
        final_reward = (total_reward / max_reward) * reward_multiplier
        return self.clip_reward(final_reward)

    def _check_balanced_tags(self, text: str) -> bool:
        """Check if all tags are properly balanced"""
        stack = []
        tag_pattern = r'<[/]?\w+>'

        for tag in re.finditer(tag_pattern, text):
            tag = tag.group()
            if tag.startswith('</'):
                if not stack or stack[-1] != tag[2:-1]:
                    return False
                stack.pop()
            else:
                stack.append(tag[1:-1])

        return len(stack) == 0

    def _check_proper_nesting(self, text: str) -> bool:
        """Check if tags are properly nested"""
        # Simple check for common nesting patterns
        patterns = [
            r'<think>.*</think>',
            r'<reasoning>.*</reasoning>',
            r'<output>.*</output>'
        ]

        for pattern in patterns:
            if not re.search(pattern, text, re.DOTALL):
                continue
            nested_content = re.findall(pattern, text, re.DOTALL)[0]
            if '</' in nested_content and not self._check_balanced_tags(nested_content):
                return False

        return True
