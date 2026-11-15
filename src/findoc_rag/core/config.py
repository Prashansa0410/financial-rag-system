"""
Configuration management for FinDoc-RAG.
Handles device detection, file paths, model configurations, and hyperparameters.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
import torch


def get_project_root() -> Path:
    """Resolve the project root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "data").is_dir() and (current / "src").is_dir():
            return current
        current = current.parent
    return Path.cwd().resolve()


def detect_device() -> tuple[str, str, torch.dtype]:
    """Auto-detect optimal compute device and precision."""
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        name = "Apple Silicon GPU (MPS)"
        dtype = torch.float16
    else:
        device = "cpu"
        name = "CPU"
        dtype = torch.float32
    return device, name, dtype


@dataclass
class Settings:
    """Global configuration settings for FinDoc-RAG."""
    # Paths
    project_root: Path = field(default_factory=get_project_root)
    corpus_dir: Path = field(init=False)
    chunks_dir: Path = field(init=False)
    eval_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    indexes_dir: Path = field(init=False)
    benchmarks_dir: Path = field(init=False)
    adapters_dir: Path = field(init=False)

    # Compute Hardware
    device: str = field(init=False)
    device_name: str = field(init=False)
    torch_dtype: torch.dtype = field(init=False)

    # Models
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_max_tokens: int = 512
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    judge_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    adapter_path: str = "models/adapters/tinyllama_financial_qlora_BF16"

    # Retrieval Defaults
    default_chunk_size: int = 400
    default_chunk_overlap: int = 50
    default_top_k: int = 3
    candidate_pool_size: int = 10
    rrf_k: int = 60

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Serverless Cloud LLM Providers (for free GPU-less hosting)
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    def __post_init__(self):
        self.corpus_dir = self.project_root / "data" / "corpus"
        self.chunks_dir = self.project_root / "data" / "chunks"
        self.eval_dir = self.project_root / "data" / "eval"
        self.outputs_dir = self.project_root / "outputs"
        self.indexes_dir = self.outputs_dir / "indexes"
        self.benchmarks_dir = self.outputs_dir / "benchmarks"
        self.adapters_dir = self.project_root / "models" / "adapters"

        self.device, self.device_name, self.torch_dtype = detect_device()

        # Ensure output directories exist
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

        # Set Hugging Face offline flags if network is unavailable
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        if not os.environ.get("HF_HUB_ENABLE_ONLINE"):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")


settings = Settings()
