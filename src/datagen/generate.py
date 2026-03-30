"""
CLI entry point for AI benchmark data generation.

Usage:
    uv run -m src.datagen.generate --type promotion_products --count 100
    uv run -m src.datagen.generate --type subscription --count 50 --output data/generated_subscriptions.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.common.config import load_config
from src.common.logger import setup_logging
from src.datagen.config import OBJECT_TYPES
from src.datagen.mcp_client import McpDataClient
from src.datagen.generator import BenchmarkGenerator

load_dotenv(override=True)

logger = setup_logging("datagen")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AI benchmark datasets using MCP server data and LLM"
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=list(OBJECT_TYPES.keys()),
        help="Creation-object type to generate test cases for",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of test cases to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: data/generated_{type}.json)",
    )
    parser.add_argument(
        "--mcp-url",
        type=str,
        default=None,
        help="MCP server URL (default: from config or http://127.0.0.1:8000/mcp/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Anthropic model name (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Samples per LLM call (default: from config or 5)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (default: config.toml)",
    )
    return parser.parse_args()


def _load_reference_samples(filename: str) -> list[dict]:
    """Load reference samples from a JSON dataset in the data/ directory."""
    path = DATA_DIR / filename
    if not path.exists():
        logger.warning("Reference dataset '%s' not found at %s", filename, path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.warning("Reference dataset '%s' is not a JSON array", filename)
        return []
    logger.info("Loaded %d reference samples from '%s'", len(data), filename)
    return data


async def _fetch_mcp_data(
    mcp_url: str, schema_tool: str, data_tools: list[str]
) -> tuple[dict, dict]:
    """Connect to MCP server and fetch schema + business data."""
    async with McpDataClient(mcp_url) as client:
        schema = await client.fetch_schema(schema_tool)
        business_data = await client.fetch_business_data(data_tools)
    return schema, business_data


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    dg_cfg = cfg.datagen

    obj_type_cfg = OBJECT_TYPES[args.type]

    mcp_url = args.mcp_url or dg_cfg.mcp_server_url
    model = args.model or dg_cfg.model
    batch_size = args.batch_size or dg_cfg.batch_size
    output_path = args.output or str(DATA_DIR / f"generated_{args.type}.json")

    logger.info("=" * 60)
    logger.info("AI Benchmark Data Generator")
    logger.info("=" * 60)
    logger.info("  Type:       %s (%s)", args.type, obj_type_cfg.description)
    logger.info("  Count:      %d", args.count)
    logger.info("  Model:      %s", model)
    logger.info("  Batch size: %d", batch_size)
    logger.info("  MCP URL:    %s", mcp_url)
    logger.info("  Output:     %s", output_path)
    logger.info("=" * 60)

    # 1. Fetch schema and business data from MCP server
    logger.info("Step 1: Fetching schema and business data from MCP server...")
    schema, business_data = asyncio.run(
        _fetch_mcp_data(mcp_url, obj_type_cfg.schema_tool, obj_type_cfg.data_tools)
    )
    logger.info(
        "  Schema keys: %s",
        list(schema.get("properties", schema).keys())[:10],
    )
    logger.info("  Business data tools: %s", list(business_data.keys()))

    # 2. Load reference samples
    logger.info("Step 2: Loading reference samples...")
    reference_samples = _load_reference_samples(obj_type_cfg.reference_dataset)
    if not reference_samples:
        logger.error(
            "No reference samples found. Cannot generate without examples. "
            "Ensure '%s' exists in %s",
            obj_type_cfg.reference_dataset,
            DATA_DIR,
        )
        sys.exit(1)

    # 3. Generate
    logger.info("Step 3: Generating %d test cases...", args.count)
    generator = BenchmarkGenerator(
        model=model,
        temperature=dg_cfg.temperature,
        max_tokens=dg_cfg.max_tokens,
    )
    samples = generator.generate(
        schema=schema,
        business_data=business_data,
        reference_samples=reference_samples,
        total_count=args.count,
        batch_size=batch_size,
        object_type=obj_type_cfg.description,
        id_prefix=obj_type_cfg.id_prefix,
        existing_count=len(reference_samples),
    )

    # 4. Write output
    logger.info("Step 4: Writing %d samples to %s", len(samples), output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)

    logger.info("Done! Generated %d test cases -> %s", len(samples), output_path)


if __name__ == "__main__":
    main()
