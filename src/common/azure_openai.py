"""
Factory functions for Azure OpenAI clients.

Model name → env var suffix mapping:
  "gpt-5-mini" → "GPT_5_MINI"
  Reads AZURE_OPENAI_ENDPOINT_{suffix}, AZURE_OPENAI_API_KEY_{suffix},
        AZURE_OPENAI_DEPLOYMENT_{suffix}, AZURE_OPENAI_API_VERSION_{suffix}
"""

import os


def _get_azure_env_vars(model_name: str) -> tuple[str, str, str, str]:
    """Read per-model Azure env vars. Raises ValueError if any are missing."""
    suffix = model_name.upper().replace("-", "_")
    keys = {
        "endpoint": f"AZURE_OPENAI_ENDPOINT_{suffix}",
        "api_key": f"AZURE_OPENAI_API_KEY_{suffix}",
        "deployment": f"AZURE_OPENAI_DEPLOYMENT_{suffix}",
        "api_version": f"AZURE_OPENAI_API_VERSION_{suffix}",
    }
    values: dict[str, str] = {}
    missing: list[str] = []
    for attr, env_key in keys.items():
        val = os.environ.get(env_key, "")
        if not val:
            missing.append(env_key)
        values[attr] = val

    if missing:
        raise ValueError(
            f"Missing Azure OpenAI env vars for model '{model_name}': {missing}"
        )

    return (
        values["endpoint"],
        values["api_key"],
        values["deployment"],
        values["api_version"],
    )


def create_azure_chat_model(model_name: str):  # type: ignore[return]
    """Build a LangChain AzureChatOpenAI from per-model env vars.

    Used by evaluators (via openevals judge parameter).
    """
    from langchain_openai import AzureChatOpenAI

    endpoint, api_key, deployment, api_version = _get_azure_env_vars(model_name)
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,  # type: ignore[arg-type]
        azure_deployment=deployment,
        api_version=api_version,
    )


def create_azure_openai_client(model_name: str) -> tuple:
    """Build a raw openai.AzureOpenAI SDK client and deployment name.

    Used by QueryGenerator which calls client.chat.completions.create() directly.
    Returns (client, deployment_name).
    """
    from openai import AzureOpenAI

    endpoint, api_key, deployment, api_version = _get_azure_env_vars(model_name)
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return client, deployment
