"""Synchronise local CSV/JSON dataset files to LangSmith.

Reads dataset definitions from the TOML config, computes content hashes to
detect changes, and creates/updates/deletes LangSmith examples accordingly.
Run as a script (``python -m src.dataset.sync``) or call :func:`main`.
"""

import argparse
import hashlib
import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from langsmith import Client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.common.config import load_config
from src.common.logger import setup_logging

logger = setup_logging("data_sync")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

load_dotenv(override=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync CSV data to LangSmith")
    parser.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Delete LangSmith examples not found in CSV. "
            "By default, orphaned examples are logged but not deleted unless "
            "this flag is provided or dataset.sync.delete_orphans is set in "
            "the config."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (default: config.toml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg.dataset

    dataset_name = dataset_cfg.name
    description = dataset_cfg.description
    csv_files = dataset_cfg.csv_files

    client = Client()

    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if not datasets:
        logger.info(f"Dataset '{dataset_name}' not found. Creating...")
        dataset = client.create_dataset(
            dataset_name=dataset_name, description=description
        )
    else:
        dataset = datasets[0]
        logger.info(f"Dataset '{dataset_name}' found (ID: {dataset.id}).")

    logger.info("Fetching existing examples...")
    existing_examples = list(client.list_examples(dataset_id=dataset.id))
    id_to_example_map: dict = {}
    for ex in existing_examples:
        if not ex.inputs:
            continue
        ex_id = (ex.metadata or {}).get("id")
        if ex_id in id_to_example_map:
            logger.warning(
                f"Duplicate metadata.id '{ex_id}' found in LangSmith "
                f"(example_ids: {id_to_example_map[ex_id].id}, {ex.id}). "
                "Run with --delete to remove orphans after re-sync."
            )
        id_to_example_map[ex_id] = ex

    csv_ids: set[str] = set()

    for filename in csv_files:
        if filename.endswith(".json"):
            json_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(json_path):
                logger.info(f"Processing {filename}...")
                file_ids = upsert_from_json(
                    client, dataset, json_path, id_to_example_map
                )
                csv_ids.update(file_ids)
            else:
                logger.warning(f"File {filename} not found. Skipping.")
        elif filename.endswith(".csv"):
            csv_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(csv_path):
                logger.info(f"Processing {filename}...")
                file_ids = _upsert_from_csv(
                    client, dataset, csv_path, id_to_example_map
                )
                csv_ids.update(file_ids)
            else:
                logger.warning(f"File {filename} not found. Skipping.")

    delete = args.delete or dataset_cfg.sync.delete_orphans
    if csv_ids:
        _handle_deletions(client, id_to_example_map, csv_ids, delete=delete)
    else:
        logger.warning(
            "No CSV files were successfully processed. Skipping orphan detection "
            "to prevent accidental deletion of all examples."
        )


def _normalize_value(value):
    """Normalise a cell value: None/NaN become '', strings are stripped."""
    if value is None:
        return ""
    # Handle lists - don't try to use pd.isna on them
    if isinstance(value, list):
        return value
    try:
        if pd.isna(value):
            return ""
    except TypeError as exc:
        # Some complex or non-standard types may cause pd.isna to raise TypeError;
        # in that case, fall back to the generic normalization logic below.
        logger.debug("pd.isna raised TypeError for value %r: %s", value, exc)
    if isinstance(value, str):
        return value.strip()
    return value


REQUIRED_COLUMNS = {"id", "question", "answer"}
CONTENT_HASH_KEY = "content_hash"


def _value_for_hash(value) -> str:
    """Normalize once and return string form for use in content hash (e.g. numbers as strings)."""
    return str(_normalize_value(value))


def _compute_content_hash(inputs: dict, outputs: dict, metadata: dict) -> str:
    """SHA-256 hash of inputs, outputs, and metadata for change detection.

    All values are normalized (None/NaN -> "", strings stripped) then stringified
    so the hash is deterministic and order-independent. Numeric metadata is hashed
    as its string representation.
    """
    normalized = {
        "inputs": {k: _value_for_hash(v) for k, v in sorted(inputs.items())},
        "outputs": {k: _value_for_hash(v) for k, v in sorted(outputs.items())},
        "metadata": {
            k: _value_for_hash(v)
            for k, v in sorted(metadata.items())
            if k != CONTENT_HASH_KEY
        },
    }
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _upsert_from_csv(client, dataset, csv_path, id_to_example_map) -> set[str]:
    """Create or update LangSmith examples from a CSV file. Returns processed IDs."""
    # Try UTF-8 first, fall back to Windows-1252 for files with curly quotes
    # Read id column as string to prevent float inference (e.g., 1.0 -> "1.0")
    try:
        df = pd.read_csv(
            csv_path,
            skipinitialspace=True,
            encoding="utf-8",
            dtype={"id": "string"},
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            csv_path,
            skipinitialspace=True,
            encoding="cp1252",
            dtype={"id": "string"},
        )

    # Validate required columns
    csv_columns = set(df.columns)
    missing = REQUIRED_COLUMNS - csv_columns
    if missing:
        logger.error(f"CSV missing required columns: {missing}. Skipping {csv_path}.")
        return set()

    # All columns beyond question/answer are metadata
    metadata_cols = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    if metadata_cols:
        logger.info(f"  Metadata columns detected: {metadata_cols}")

    created_count = 0
    updated_count = 0
    skipped_count = 0
    processed_ids: set[str] = set()

    for _, row in df.iterrows():
        raw_id = row["id"]
        if pd.isna(raw_id):
            logger.warning(f"Skipping row with missing id in {csv_path}.")
            continue
        # Normalize ID: strip whitespace to ensure stable matching across CSV edits
        # dtype="string" ensures we get a string even if CSV had numeric-looking IDs
        row_id = str(raw_id).strip()
        if not row_id:
            logger.warning(
                f"Skipping row with empty id after normalization in {csv_path}."
            )
            continue

        question = row["question"]
        if pd.isna(question):
            continue

        normalized_question = str(question).strip()
        if not normalized_question:
            continue

        # Only add to processed_ids after all validation passes.
        # This ensures invalid rows (missing/empty question) don't prevent orphan detection.
        processed_ids.add(row_id)

        inputs = {"question": normalized_question}
        # Normalize answer before storing to ensure hash and LangSmith values match
        normalized_answer = (
            _normalize_value(row["answer"]) if pd.notna(row["answer"]) else ""
        )
        outputs = {"answer": normalized_answer}
        # Include agent in outputs so evaluators (e.g. agent_name_evaluator) receive it
        # as reference_outputs; otherwise it only lives in metadata and is unavailable.
        if "agent" in row and pd.notna(row["agent"]):
            outputs["agent"] = _normalize_value(row["agent"])

        # Normalize all metadata values before storing to ensure hash and LangSmith values match
        metadata = {}
        for col in metadata_cols:
            value = row[col]
            metadata[col] = _normalize_value(value)
        metadata["id"] = row_id

        content_hash = _compute_content_hash(inputs, outputs, metadata)
        metadata[CONTENT_HASH_KEY] = content_hash

        if row_id in id_to_example_map:
            example = id_to_example_map[row_id]
            existing_hash = (example.metadata or {}).get(CONTENT_HASH_KEY)
            if existing_hash != content_hash:
                client.update_example(
                    example_id=example.id,
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                )
                updated_count += 1
            else:
                skipped_count += 1
        else:
            new_example = client.create_example(
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                dataset_id=dataset.id,
            )
            id_to_example_map[row_id] = new_example
            created_count += 1

    logger.info(f"  - Created: {created_count}")
    logger.info(f"  - Updated: {updated_count}")
    logger.info(f"  - Unchanged (skipped): {skipped_count}")

    return processed_ids


def _handle_deletions(client, id_to_example_map: dict, csv_ids: set[str], delete: bool):
    """Log orphaned LangSmith examples and optionally delete them."""
    orphaned_ids = set(id_to_example_map.keys()) - csv_ids
    if not orphaned_ids:
        logger.info("No orphaned examples found.")
        return

    for orphaned_id in orphaned_ids:
        example = id_to_example_map[orphaned_id]
        question = (example.inputs or {}).get("question", "N/A")
        logger.warning(
            f'  Orphaned: id={orphaned_id}, question="{question}", '
            f"example_id={example.id}"
        )

    if delete:
        for orphaned_id in orphaned_ids:
            example = id_to_example_map[orphaned_id]
            client.delete_example(example_id=example.id)
        logger.info(f"Deleted {len(orphaned_ids)} orphaned example(s).")
    else:
        logger.info(
            f"Found {len(orphaned_ids)} orphaned example(s). "
            f"Run with --delete to remove them."
        )


def upsert_from_json(client, dataset, json_path, id_to_example_map) -> set[str]:
    """Create or update LangSmith examples from a JSON file. Returns processed IDs."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    created_count = 0
    updated_count = 0
    skipped_count = 0
    processed_ids: set[str] = set()

    for item in data:
        raw_id = item.get("id")
        if raw_id is None:
            logger.warning(f"Skipping row with missing id in {json_path}.")
            continue
        row_id = str(raw_id).strip()
        if not row_id:
            logger.warning(
                f"Skipping row with empty id after normalization in {json_path}."
            )
            continue

        # Handle conversation format (client-server chat data)
        if "messages" in item:
            messages = item.get("messages", [])
            # Collect paired messages - user message must have corresponding server response
            user_messages = []
            server_messages = []

            pending_user = None

            for msg in messages:
                if "user_message" in msg and msg["user_message"]:
                    pending_user = msg["user_message"]
                elif "server_message" in msg and msg["server_message"]:
                    if pending_user is not None:
                        user_messages.append(pending_user)
                        server_messages.append(msg["server_message"])
                        pending_user = None
                elif "human_agent_message" in msg and msg["human_agent_message"]:
                    # Discard pending user message (it won't get a server response)
                    # Stop collecting - conversation transferred to human
                    pending_user = None
                    break

            # Discard any remaining pending user message that never got a server response
            question = user_messages if user_messages else None
            expected_output = server_messages if server_messages else []
            description = f"Chat conversation (ID: {item.get('livechat_id', row_id)})"
        else:
            # Standard format with input_query/expected_output
            question = item.get("input_query")
            expected_output = item.get("expected_output")
            description = item.get("description")

        if not question:
            logger.warning(f"Skipping row {row_id}: no question found.")
            continue

        # For list-based questions, check if list is empty
        if isinstance(question, list) and len(question) == 0:
            logger.warning(f"Skipping row {row_id}: empty question list.")
            continue

        # Enforce type contract: conversation-format items must produce list questions
        if "messages" in item and not isinstance(question, list):
            logger.error(
                f"Skipping row {row_id}: conversation format produced a non-list "
                f"question (got {type(question).__name__!r}). This is a bug — "
                f"check the messages branch in upsert_from_json."
            )
            continue

        processed_ids.add(row_id)

        inputs = {"question": question}
        # Use baseline_output as ordered list of baseline messages
        outputs = {"baseline_output": expected_output}

        # Handle fields_to_compare if present (specific to JSON datasets)
        if "fields_to_compare" in item:
            outputs["fields_to_compare"] = _normalize_value(item["fields_to_compare"])

        # Extract metadata
        metadata = {
            "id": row_id,
            "description": _normalize_value(description),
        }
        # Add conversation metadata if available
        if "conversation_id" in item:
            metadata["conversation_id"] = item["conversation_id"]
        if "livechat_id" in item:
            metadata["livechat_id"] = item["livechat_id"]
        if "messages" in item:
            metadata["message_count"] = len(item["messages"])
            # Add counts for the paired messages that were actually uploaded
            metadata["user_message_count"] = len(user_messages)
            metadata["server_message_count"] = len(server_messages)

        content_hash = _compute_content_hash(inputs, outputs, metadata)
        metadata[CONTENT_HASH_KEY] = content_hash

        if row_id in id_to_example_map:
            # Update existing example
            example = id_to_example_map[row_id]
            existing_hash = (example.metadata or {}).get(CONTENT_HASH_KEY)
            if existing_hash != content_hash:
                client.update_example(
                    example_id=example.id,
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                )
                updated_count += 1
            else:
                skipped_count += 1
        else:
            # Create new example
            new_example = client.create_example(
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                dataset_id=dataset.id,
            )
            id_to_example_map[row_id] = new_example
            created_count += 1

    logger.info(f"  - Created: {created_count}")
    logger.info(f"  - Updated: {updated_count}")
    logger.info(f"  - Unchanged (skipped): {skipped_count}")

    return processed_ids


if __name__ == "__main__":
    main()
