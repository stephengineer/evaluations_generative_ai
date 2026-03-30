"""
Version management for SUT Evaluation Framework components.

Versions are read from pyproject.toml [tool.sut-eval.versions] when available
(development/source tree). When the package is installed and pyproject.toml
is not shipped, falls back to importlib.metadata for the project version and
uses that for component versions. Data is loaded and cached on first use (no I/O at import time).
"""

from pathlib import Path
from typing import Any, cast

# Project root (assumes this file is at src/common/version.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PYPROJECT_PATH = _PROJECT_ROOT / "pyproject.toml"

# Lazy cache: None = not loaded, dict = from pyproject, _USE_METADATA = fallback to metadata
_USE_METADATA: Any = object()
_version_data: dict[str, Any] | Any | None = None
_cached_project_version: str | None = None  # used when _version_data is _USE_METADATA


def _load_data() -> dict[str, Any] | Any:
    """Load version data once: from pyproject.toml if present, else use metadata fallback. Cached."""
    global _version_data
    if _version_data is not None:
        return _version_data

    if _PYPROJECT_PATH.exists():
        import tomllib

        with _PYPROJECT_PATH.open("rb") as f:
            _version_data = tomllib.load(f)
        return _version_data

    _version_data = _USE_METADATA
    return _version_data


def get_project_version() -> str:
    """Get the main project version from [project].version or from package metadata when installed."""
    global _cached_project_version
    data = _load_data()

    if data is _USE_METADATA:
        if _cached_project_version is not None:
            return _cached_project_version
        from importlib.metadata import PackageNotFoundError, version

        try:
            _cached_project_version = version("sut-eval")
        except PackageNotFoundError:
            _cached_project_version = "0.0.0"
        return _cached_project_version

    return cast(str, data["project"]["version"])


def get_component_version(component: str) -> str:
    """
    Get version for a specific component.

    Args:
        component: Component name ('evaluation', 'data_management')

    Returns:
        Component version string. When pyproject.toml is not available (installed package),
        returns the project version from package metadata.
    """
    data = _load_data()

    if data is _USE_METADATA:
        return get_project_version()

    try:
        return cast(str, data["tool"]["sut-eval"]["versions"][component])
    except KeyError:
        available = list(
            data.get("tool", {}).get("sut-eval", {}).get("versions", {}).keys()
        )
        raise KeyError(
            f"Component '{component}' not found. Available: {available}"
        ) from None


def __getattr__(name: str) -> str:
    """Lazy-load constants on first access so no I/O at import time."""
    if name == "EVALUATION_VERSION":
        return get_component_version("evaluation")
    if name == "DATA_MANAGEMENT_VERSION":
        return get_component_version("data_management")
    if name == "PROJECT_VERSION":
        return get_project_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# EVALUATION_VERSION, DATA_MANAGEMENT_VERSION, PROJECT_VERSION
# are provided by __getattr__ (lazy); import explicitly, e.g. from src.common.version import _VERSION
__all__ = [
    "get_project_version",
    "get_component_version",
]
