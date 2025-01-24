from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
import logging
import traceback


@dataclass
class ProcessorConfig:
    """Configuration for processors"""
    # Basic settings
    max_length: Optional[int] = None
    min_length: Optional[int] = None

    # Preprocessing flags
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_numbers: bool = False

    # Special handling
    special_tokens: List[str] = field(default_factory=list)
    ignore_tokens: List[str] = field(default_factory=list)

    # Processing options
    num_workers: int = 1
    batch_size: int = 32

    # Error handling
    skip_on_error: bool = False
    error_value: Any = None


class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass


class BaseProcessor(ABC):
    """
    Base class for all data processors

    Attributes:
        config: Processor configuration
        logger: Logger instance
        statistics: Processing statistics
    """

    def __init__(
            self,
            config: Optional[Union[Dict[str, Any], ProcessorConfig]] = None,
            logger: Optional[logging.Logger] = None
    ):
        # Initialize configuration
        if isinstance(config, dict):
            self.config = ProcessorConfig(**config)
        else:
            self.config = config or ProcessorConfig()

        # Setup logging
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Initialize statistics
        self.statistics = {
            'processed_count': 0,
            'error_count': 0,
            'skipped_count': 0
        }

        # Validate configuration
        self._validate_config()

        # Initialize processor
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """Initialize processor specific resources"""
        pass

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """
        Process a single piece of data

        Args:
            data: Input data to process

        Returns:
            Processed data

        Raises:
            ProcessingError: If processing fails
        """
        pass

    def process_batch(self, batch: List[Any]) -> List[Any]:
        """
        Process a batch of data

        Args:
            batch: List of inputs to process

        Returns:
            List of processed outputs
        """
        results = []
        for item in batch:
            try:
                processed = self(item)
                if processed is not None:
                    results.append(processed)
                    self.statistics['processed_count'] += 1
            except Exception as e:
                self.statistics['error_count'] += 1
                if self.config.skip_on_error:
                    self.statistics['skipped_count'] += 1
                    self.logger.warning(
                        f"Error processing item: {str(e)}\n{traceback.format_exc()}"
                    )
                    if self.config.error_value is not None:
                        results.append(self.config.error_value)
                else:
                    raise ProcessingError(f"Processing failed: {str(e)}") from e
        return results

    def _validate_config(self):
        """Validate processor configuration"""
        if self.config.max_length is not None and self.config.min_length is not None:
            if self.config.max_length < self.config.min_length:
                raise ValueError(
                    f"max_length ({self.config.max_length}) must be >= "
                    f"min_length ({self.config.min_length})"
                )

        if self.config.num_workers < 1:
            raise ValueError(
                f"num_workers must be >= 1, got {self.config.num_workers}"
            )

        if self.config.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {self.config.batch_size}"
            )

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics"""
        return self.statistics

    def reset_statistics(self):
        """Reset processing statistics"""
        self.statistics = {
            'processed_count': 0,
            'error_count': 0,
            'skipped_count': 0
        }

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is not None:
            self.logger.error(
                f"Error during processing: {exc_type.__name__}: {str(exc_val)}"
            )
            return False  # Re-raise exception
        return True

    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(config={self.config})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(config={self.config}, stats={self.statistics})"
