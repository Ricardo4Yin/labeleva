"""Create test data file from TechQA dataset for labelrag examples."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypedDict, Any

# 定义 TechQA 文档类型
class TechQADocument(TypedDict):
    """TechQA document format from training_dev_technotes.json."""
    id: str
    text: str
    title: str
    metadata: dict[str, Any]


def create_techqa_test_file(
    source_path: Path,
    output_path: Path,
    num_samples: int = 5,
    seed: int = 42,
) -> None:
    """Create test data file by sampling from TechQA dataset.

    Args:
        source_path: Path to training_dev_technotes.json
        output_path: Path to save techqa_test.json
        num_samples: Number of documents to sample
        seed: Random seed for reproducibility
    """
    # 设置随机种子
    random.seed(seed)

    try:
        # 加载原始数据
        with open(source_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        # 随机选择文档ID
        doc_ids = list(data.keys())
        if len(doc_ids) < num_samples:
            raise ValueError(
                f"Dataset has only {len(doc_ids)} documents, "
                f"requested {num_samples}"
            )

        selected_ids = random.sample(doc_ids, num_samples)

        # 提取所需字段，构建新数据
        test_data: dict[str, TechQADocument] = {}
        for doc_id in selected_ids:
            doc = data[doc_id]
            test_data[doc_id] = {
                "id": doc["id"],
                "text": doc["text"],
                "title": doc.get("title", ""),
                "metadata": doc.get("metadata", {}),
            }

        # 保存测试数据
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)

        print(f"Created {output_path} with {num_samples} documents")
        print("Selected document IDs:", selected_ids)

    except Exception as e:
        raise RuntimeError(f"Failed to create test data file: {e}") from e


def main() -> None:
    """Create techqa_test.json from TechQA dataset."""
    # 源数据路径
    source_path = Path(
        "C:/Users/Lenovo/Desktop/Research/TechQA/training_and_dev/training_dev_technotes.json"
    )

    # 输出路径（当前目录）
    output_path = Path(__file__).parent / "techqa_test.json"

    try:
        create_techqa_test_file(source_path, output_path)
        print("Test data file created successfully!")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()