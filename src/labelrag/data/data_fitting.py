"""Pipeline fitting utilities for data loading."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from labelgen import Paragraph
    from labelrag import RAGPipeline
    from labelrag.data.loader import DataLoader, DataLoaderConfig


class DataFittingHelper:
    """Helper class for fitting RAGPipeline with data loading.

    This class provides static methods to fit a RAGPipeline using data from
    JSON files with TechQA structure.
    """

    @staticmethod
    def load_paragraphs_from_json(
        data_path: str | Path,
        **kwargs: Any,
    ) -> list[Paragraph]:
        """Load paragraphs from JSON file with TechQA structure.

        Args:
            data_path: Path to JSON file with TechQA structure (str or Path).
            **kwargs: Additional arguments passed to DataLoaderConfig
                (max_paragraph_length, overlap_sentences, etc.).

        Returns:
            List of Paragraph objects ready for pipeline.fit().

        Raises:
            RuntimeError: If data_path is invalid or data cannot be loaded.
        """
        from labelrag.data.loader import DataLoader, DataLoaderConfig

        path = Path(data_path) if isinstance(data_path, str) else data_path
        config = DataLoaderConfig(data_path=path, **kwargs)
        loader = DataLoader(config)
        return loader.load_paragraphs()

    @staticmethod
    def fit_pipeline_from_json(
        pipeline: RAGPipeline,
        data_path: str | Path,
        **kwargs: Any,
    ) -> RAGPipeline:
        """Fit pipeline from JSON file with TechQA structure.

        Args:
            pipeline: RAGPipeline instance to fit.
            data_path: Path to JSON file with TechQA structure (str or Path).
            **kwargs: Additional arguments passed to DataLoaderConfig
                (max_paragraph_length, overlap_sentences, etc.).

        Returns:
            Fitted RAGPipeline instance.

        Raises:
            RuntimeError: If data_path is invalid or data cannot be loaded.
        """
        paragraphs = DataFittingHelper.load_paragraphs_from_json(data_path, **kwargs)
        return pipeline.fit(paragraphs)

    @staticmethod
    def fit_pipeline_with_loader(
        pipeline: RAGPipeline,
        loader: DataLoader,
    ) -> RAGPipeline:
        """Fit pipeline using an existing DataLoader instance.

        Args:
            pipeline: RAGPipeline instance to fit.
            loader: DataLoader instance already configured with data path.

        Returns:
            Fitted RAGPipeline instance.
        """
        paragraphs = loader.load_paragraphs()
        return pipeline.fit(paragraphs)