import pytest
import os
import sys

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.config import (
    load_config,
    get_config,
    set_config,
    Config,
    ClientConfig,
    EvaluationConfig,
    DatasetConfig,
    DatasetSyncConfig,
)
import src.common.config as config_module


class TestDefaultValues:
    """Test that default values are used when config is missing or empty."""

    def test_load_config_with_missing_file(self, tmp_path):
        """load_config returns defaults when file doesn't exist."""
        non_existent = tmp_path / "nonexistent.toml"
        cfg = load_config(str(non_existent))
        assert isinstance(cfg, Config)
        assert cfg.evaluation.dataset_name == "SUT Benchmark"
        assert cfg.evaluation.max_concurrency == 1
        assert cfg.evaluation.client.type == "script"
        assert cfg.dataset.name == "SUT Benchmark"

    def test_load_config_with_empty_file(self, tmp_path):
        """load_config returns defaults when file is empty."""
        empty_file = tmp_path / "empty.toml"
        empty_file.write_text("")
        cfg = load_config(str(empty_file))
        assert isinstance(cfg, Config)
        assert cfg.evaluation.dataset_name == "SUT Benchmark"

    def test_client_config_defaults(self):
        """ClientConfig has correct defaults."""
        cfg = ClientConfig()
        assert cfg.type == "script"
        assert cfg.mode == "single_round"
        assert cfg.script_path == "conversation_service.py"
        assert cfg.timeout == 30
        assert cfg.opensource_model == "gpt-5-mini"

    def test_evaluation_config_defaults(self):
        """EvaluationConfig has correct defaults."""
        cfg = EvaluationConfig()
        assert cfg.dataset_name == "SUT Benchmark"
        assert cfg.experiment_prefix == "sut-script-eval"
        assert cfg.max_concurrency == 1
        assert cfg.judge_model == "gpt-5-mini"
        assert cfg.evaluators == []
        assert isinstance(cfg.client, ClientConfig)

    def test_dataset_config_defaults(self):
        """DatasetConfig has correct defaults."""
        cfg = DatasetConfig()
        assert cfg.name == "SUT Benchmark"
        assert cfg.description == "Benchmark dataset for SUT evaluation"
        assert cfg.csv_files == ["SUTBenchmark.csv"]
        assert isinstance(cfg.sync, DatasetSyncConfig)
        assert cfg.sync.delete_orphans is False


class TestTOMLLoading:
    """Test TOML file loading and parsing."""

    def test_load_valid_toml(self, tmp_path):
        """load_config parses valid TOML correctly."""
        toml_content = """
[evaluation]
dataset_name = "Test Dataset"
max_concurrency = 5
judge_model = "gpt-4"
evaluators = ["relevance", "correctness"]

[evaluation.client]
type = "opensource"
mode = "multi_round"
timeout = 60

[dataset]
name = "Test Dataset"
csv_files = ["test1.csv", "test2.csv"]
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.evaluation.dataset_name == "Test Dataset"
        assert cfg.evaluation.max_concurrency == 5
        assert cfg.evaluation.judge_model == "gpt-4"
        assert cfg.evaluation.evaluators == ["relevance", "correctness"]
        assert cfg.evaluation.client.type == "opensource"
        assert cfg.evaluation.client.mode == "multi_round"
        assert cfg.evaluation.client.timeout == 60
        assert cfg.dataset.name == "Test Dataset"
        assert cfg.dataset.csv_files == ["test1.csv", "test2.csv"]

    def test_load_partial_config(self, tmp_path):
        """load_config merges partial config with defaults."""
        toml_content = """
[evaluation]
dataset_name = "Partial Dataset"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.evaluation.dataset_name == "Partial Dataset"
        # Other values should be defaults
        assert cfg.evaluation.max_concurrency == 1
        assert cfg.evaluation.client.type == "script"
        assert cfg.dataset.name == "SUT Benchmark"

    def test_load_invalid_toml_raises(self, tmp_path):
        """load_config raises exception for invalid TOML syntax."""
        config_file = tmp_path / "invalid.toml"
        config_file.write_text("[evaluation]\ndataset_name = invalid syntax here")
        # tomllib.load raises tomllib.TOMLDecodeError for invalid syntax
        with pytest.raises(Exception):  # TOMLDecodeError or similar
            load_config(str(config_file))

    def test_load_nonexistent_section(self, tmp_path):
        """load_config ignores unknown sections."""
        toml_content = """
[unknown_section]
key = "value"

[evaluation]
dataset_name = "Test"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))
        # Should load successfully, ignoring unknown_section
        assert cfg.evaluation.dataset_name == "Test"


class TestTypeConversions:
    """Test that type conversions work correctly."""

    def test_int_conversion(self, tmp_path):
        """Numeric strings and ints are converted correctly."""
        toml_content = """
[evaluation]
max_concurrency = 10

[evaluation.client]
timeout = "45"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.evaluation.max_concurrency == 10
        assert cfg.evaluation.client.timeout == 45  # Should convert string to int

    def test_list_conversion(self, tmp_path):
        """List values are converted correctly."""
        toml_content = """
[evaluation]
evaluators = ["relevance", "correctness", "agent_match"]

[dataset]
csv_files = ["file1.csv", "file2.csv"]
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.evaluation.evaluators == ["relevance", "correctness", "agent_match"]
        assert cfg.dataset.csv_files == ["file1.csv", "file2.csv"]
        assert all(isinstance(e, str) for e in cfg.evaluation.evaluators)

    def test_bool_conversion(self, tmp_path):
        """Boolean values are converted correctly."""
        toml_content = """
[dataset.sync]
delete_orphans = true
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.dataset.sync.delete_orphans is True

    def test_string_coercion(self, tmp_path):
        """Non-string values are coerced to strings where needed."""
        toml_content = """
[evaluation]
dataset_name = 12345
experiment_prefix = 67890
judge_model = true

[evaluation.client]
type = 999
script_path = 111
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.evaluation.dataset_name == "12345"
        assert cfg.evaluation.experiment_prefix == "67890"
        assert cfg.evaluation.judge_model == "True"
        assert cfg.evaluation.client.type == "999"
        assert cfg.evaluation.client.script_path == "111"


class TestNestedConfigSections:
    """Test nested configuration sections."""

    def test_evaluation_client_nested(self, tmp_path):
        """Nested [evaluation.client] section is parsed correctly."""
        toml_content = """
[evaluation]
dataset_name = "Test"

[evaluation.client]
type = "script"
mode = "multi_round"
script_path = "custom_script.py"
cwd = "/custom/path"
timeout = 90

[evaluation.client.opensource]
model_name = "gpt-4"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.evaluation.client.type == "script"
        assert cfg.evaluation.client.mode == "multi_round"
        assert cfg.evaluation.client.script_path == "custom_script.py"
        assert cfg.evaluation.client.cwd == "/custom/path"
        assert cfg.evaluation.client.timeout == 90
        assert cfg.evaluation.client.opensource_model == "gpt-4"

    def test_dataset_sync_nested(self, tmp_path):
        """Nested [dataset.sync] section is parsed correctly."""
        toml_content = """
[dataset]
name = "Test Dataset"

[dataset.sync]
delete_orphans = true
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.dataset.sync.delete_orphans is True

    def test_partial_nested_config(self, tmp_path):
        """Partial nested config merges with defaults."""
        toml_content = """
[evaluation.client]
timeout = 120
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.evaluation.client.timeout == 120
        # Other client fields should be defaults
        assert cfg.evaluation.client.type == "script"
        assert cfg.evaluation.client.mode == "single_round"


class TestSingletonPattern:
    """Test get_config and set_config singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        config_module._config = None

    def test_get_config_loads_on_first_call(self, tmp_path, monkeypatch):
        """get_config loads config on first call."""
        toml_content = """
[evaluation]
dataset_name = "Singleton Test"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        # Patch the default config path
        monkeypatch.setattr(config_module, "_DEFAULT_CONFIG_PATH", config_file)

        cfg1 = get_config()
        assert cfg1.evaluation.dataset_name == "Singleton Test"

    def test_get_config_returns_same_instance(self, tmp_path, monkeypatch):
        """get_config returns the same instance on subsequent calls."""
        toml_content = """
[evaluation]
dataset_name = "Singleton Test"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        monkeypatch.setattr(config_module, "_DEFAULT_CONFIG_PATH", config_file)

        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_overrides_singleton(self):
        """set_config allows overriding the singleton."""
        custom_cfg = Config()
        custom_cfg.evaluation.dataset_name = "Custom Dataset"

        set_config(custom_cfg)
        cfg = get_config()
        assert cfg is custom_cfg
        assert cfg.evaluation.dataset_name == "Custom Dataset"

    def test_get_config_with_path(self, tmp_path):
        """get_config accepts config_path parameter on first call."""
        toml_content = """
[evaluation]
dataset_name = "Path Test"
"""
        config_file = tmp_path / "custom.toml"
        config_file.write_text(toml_content)

        cfg = get_config(str(config_file))
        assert cfg.evaluation.dataset_name == "Path Test"

    def test_get_config_ignores_path_after_first_call(self, tmp_path):
        """get_config ignores config_path after singleton is loaded."""
        toml_content1 = """
[evaluation]
dataset_name = "First Load"
"""
        toml_content2 = """
[evaluation]
dataset_name = "Second Load"
"""
        config_file1 = tmp_path / "config1.toml"
        config_file2 = tmp_path / "config2.toml"
        config_file1.write_text(toml_content1)
        config_file2.write_text(toml_content2)

        cfg1 = get_config(str(config_file1))
        cfg2 = get_config(str(config_file2))
        # Should still be from first file
        assert cfg1 is cfg2
        assert cfg2.evaluation.dataset_name == "First Load"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_list_evaluators(self, tmp_path):
        """Empty evaluators list is handled correctly."""
        toml_content = """
[evaluation]
evaluators = []
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.evaluation.evaluators == []

    def test_empty_list_csv_files(self, tmp_path):
        """Empty csv_files list is handled correctly."""
        toml_content = """
[dataset]
csv_files = []
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        assert cfg.dataset.csv_files == []

    def test_missing_nested_section_uses_defaults(self, tmp_path):
        """Missing nested sections use defaults."""
        toml_content = """
[evaluation]
dataset_name = "Test"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        # Client should have defaults since [evaluation.client] is missing
        assert cfg.evaluation.client.type == "script"
        assert cfg.evaluation.client.timeout == 30

    def test_non_dict_section_ignored(self, tmp_path):
        """Non-dict sections are ignored gracefully."""
        toml_content = """
[evaluation]
dataset_name = "Test"
client = "not a dict"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        # Should use default client config since client is not a dict
        assert cfg.evaluation.client.type == "script"

    def test_non_list_evaluators_ignored(self, tmp_path):
        """Non-list evaluators are ignored."""
        toml_content = """
[evaluation]
evaluators = "not a list"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        cfg = load_config(str(config_file))

        # Should use default empty list
        assert cfg.evaluation.evaluators == []

    def test_relative_path(self, tmp_path):
        """load_config handles relative paths."""
        toml_content = """
[evaluation]
dataset_name = "Relative Path Test"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        # Change to tmp_path directory and use relative path
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = load_config("config.toml")
            assert cfg.evaluation.dataset_name == "Relative Path Test"
        finally:
            os.chdir(original_cwd)
