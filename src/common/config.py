"""
Centralized configuration for the SUT Evaluation Framework.

Loads from a TOML file (default: config.toml at project root).
Secrets stay in .env only. Precedence: dataclass defaults < config.toml < CLI --config.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.toml"


@dataclass
class ClientConfig:
    type: str = "script"
    mode: str = "single_round"
    script_path: str = "conversation_service.py"
    cwd: str = str(_PROJECT_ROOT / "src" / "sut" / "services")
    timeout: int = 30
    opensource_model: str = "gpt-5-mini"


@dataclass
class EvaluationConfig:
    dataset_name: str = "SUT Benchmark"
    experiment_prefix: str = "sut-script-eval"
    max_concurrency: int = 1
    judge_model: str = "gpt-5-mini"
    evaluators: list[str] = field(default_factory=list)
    client: ClientConfig = field(default_factory=ClientConfig)
    judge_provider: str = "openai"  # "openai" | "azure_openai"
    query_generator_provider: str = ""  # defaults to judge_provider when empty
    # AI provider evaluation settings
    ai_provider: bool = False
    max_turns: int = 10
    query_generator_model: str = ""  # defaults to judge_model when empty
    query_generator_temperature: float = 0.7
    query_generator_user_style: str = "normal"


@dataclass
class DatasetSyncConfig:
    delete_orphans: bool = False


@dataclass
class DatasetConfig:
    name: str = "SUT Benchmark"
    description: str = "Benchmark dataset for SUT evaluation"
    csv_files: list[str] = field(default_factory=lambda: ["SUTBenchmark.csv"])
    sync: DatasetSyncConfig = field(default_factory=DatasetSyncConfig)


@dataclass
class DataGenConfig:
    mcp_server_url: str = "http://127.0.0.1:8000/mcp/"
    model: str = "claude-opus-4-6"
    batch_size: int = 5
    temperature: float = 0.8
    max_tokens: int = 16384


@dataclass
class ClientServerConfig:
    raw_data_dir: str = "data/azure_blob_storage/client_server_chat_data"
    processed_data_dir: str = (
        "data/azure_blob_storage/processed_client_server_chat_data"
    )
    default_year: str = "2026"
    default_count: int = 0  # 0 = process all conversations
    classifier_model: str = "gpt-5-mini"
    classifier_chunk_size: int = 20


@dataclass
class Config:
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    datagen: DataGenConfig = field(default_factory=DataGenConfig)
    client_server: ClientServerConfig = field(default_factory=ClientServerConfig)


def _merge_client_config(data: dict) -> ClientConfig:
    cfg = ClientConfig()
    for key in ("type", "mode", "script_path", "cwd"):
        if key in data:
            setattr(cfg, key, str(data[key]))
    if "timeout" in data:
        cfg.timeout = int(data["timeout"])

    opensource = data.get("opensource", {})
    if isinstance(opensource, dict) and "model_name" in opensource:
        cfg.opensource_model = str(opensource["model_name"])
    return cfg


def _merge_evaluation_config(data: dict) -> EvaluationConfig:
    cfg = EvaluationConfig()
    for key in (
        "dataset_name",
        "experiment_prefix",
        "judge_model",
        "judge_provider",
        "query_generator_provider",
    ):
        if key in data:
            setattr(cfg, key, str(data[key]))
    if "max_concurrency" in data:
        cfg.max_concurrency = int(data["max_concurrency"])

    if "evaluators" in data:
        raw = data["evaluators"]
        if isinstance(raw, list):
            cfg.evaluators = [str(name) for name in raw]

    client_data = data.get("client", {})
    if isinstance(client_data, dict):
        cfg.client = _merge_client_config(client_data)

    # AI provider settings
    if "ai_provider" in data:
        cfg.ai_provider = bool(data["ai_provider"])
    if "max_turns" in data:
        cfg.max_turns = int(data["max_turns"])
    if "query_generator_model" in data:
        cfg.query_generator_model = str(data["query_generator_model"])
    if "query_generator_temperature" in data:
        cfg.query_generator_temperature = float(data["query_generator_temperature"])
    if "query_generator_user_style" in data:
        cfg.query_generator_user_style = (
            str(data["query_generator_user_style"]).strip().lower()
        )

    return cfg


def _merge_dataset_config(data: dict) -> DatasetConfig:
    cfg = DatasetConfig()
    for key in ("name", "description"):
        if key in data:
            setattr(cfg, key, str(data[key]))
    if "csv_files" in data:
        raw = data["csv_files"]
        if isinstance(raw, list):
            cfg.csv_files = [str(f) for f in raw]

    sync_data = data.get("sync", {})
    if isinstance(sync_data, dict):
        cfg.sync = DatasetSyncConfig(
            delete_orphans=bool(sync_data.get("delete_orphans", False)),
        )
    return cfg


def _merge_datagen_config(data: dict) -> DataGenConfig:
    cfg = DataGenConfig()
    for key in ("mcp_server_url", "model"):
        if key in data:
            setattr(cfg, key, str(data[key]))
    if "batch_size" in data:
        cfg.batch_size = int(data["batch_size"])
    if "temperature" in data:
        cfg.temperature = float(data["temperature"])
    if "max_tokens" in data:
        cfg.max_tokens = int(data["max_tokens"])
    return cfg


def _merge_client_server_config(data: dict) -> ClientServerConfig:
    cfg = ClientServerConfig()
    for key in (
        "raw_data_dir",
        "processed_data_dir",
        "default_year",
        "classifier_model",
    ):
        if key in data:
            setattr(cfg, key, str(data[key]))
    if "default_count" in data:
        cfg.default_count = int(data["default_count"])
    if "classifier_chunk_size" in data:
        cfg.classifier_chunk_size = int(data["classifier_chunk_size"])
    return cfg


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from a TOML file. Returns defaults if file not found."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    cfg = Config()

    evaluation_data = raw.get("evaluation", {})
    if isinstance(evaluation_data, dict):
        cfg.evaluation = _merge_evaluation_config(evaluation_data)

    dataset_data = raw.get("dataset", {})
    if isinstance(dataset_data, dict):
        cfg.dataset = _merge_dataset_config(dataset_data)

    datagen_data = raw.get("datagen", {})
    if isinstance(datagen_data, dict):
        cfg.datagen = _merge_datagen_config(datagen_data)

    client_server_data = raw.get("client_server", {})
    if isinstance(client_server_data, dict):
        cfg.client_server = _merge_client_server_config(client_server_data)

    return cfg


_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global Config singleton. Loads on first call."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def set_config(config: Config) -> None:
    """Replace the global Config singleton (useful for testing)."""
    global _config
    _config = config
