# SUT Evaluation Framework

A Python-based evaluation platform for benchmarking real-time, multi-turn AI conversational assistants. Supports single-round and multi-turn evaluation with LLM-as-judge scoring, deterministic JSON comparison, pairwise A/B testing, automated query generation, and dataset management via LangSmith.

> **Paper:** [How to Evaluate a Real-Time Multi-Turn Chatbot](docs/technical_report.pdf)

---

## Key Features

- **Single-round & multi-turn evaluation** -- Evaluate one-shot responses or full multi-turn conversations with persistent SUT sessions
- **AI provider evaluation** -- LLM-based query generator simulates realistic users, adapting follow-up queries based on live SUT responses
- **7 built-in evaluators** -- Relevance, correctness, agent routing match, pairwise comparison, JSON structure match, JSON semantic match, report agent data
- **A/B comparison with position randomization** -- Eliminates order bias when comparing SUT vs. baseline responses
- **Synthetic data generation** -- Generate evaluation datasets using Claude via MCP server schemas and real business data
- **Dataset sync** -- Content-hash-based sync between local CSV/JSON files and LangSmith datasets with change detection
- **LangSmith integration** -- Experiment tracking, result dashboards, and dataset management
- **Pluggable SUT clients** -- Script (subprocess), MCP, and OpenAI-compatible client types
- **Configurable query styles** -- Simulate brief, normal, or verbose user personas during AI provider evaluation

## Architecture

```
                        Evaluation Pipeline
                    ┌─────────────────────────┐
                    │  LangSmith Experiments  │
  SUT               │                         │         Dashboard
 ┌───────┐          │  ┌─────────┐            │        (LangSmith)
 │script │─request─▶│  │evaluator│◀─LLM Judge │─────▶  engineers
 │mcp    │◀response─│  │provider │            │        product
 │openai │          │  └─────────┘            │        security
 └───────┘          │                         │
                    │  client manager         │
                    │  single-turn│multi-turn │
                    └─────────────────────────┘
                              ▲
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────────┐   ┌──────────┐    ┌──────────┐
        │benchmark│   │  AI Gen  │    │ data mgr │
        │ (CSV/   │   │ (Claude  │    │(blob,sync│
        │  JSON)  │   │  + MCP)  │    │ process) │
        └─────────┘   └──────────┘    └──────────┘
              └───────────┼───────────────┘
                     LangSmith
```

## Project Structure

```
src/
  evaluation/           # Core evaluation pipeline
    main.py             # Orchestrator (single-round & AI provider modes)
    client.py           # SUT client implementations (script, MCP, opensource)
    evaluators.py       # Evaluator registry and implementations
    json_evaluators.py  # Deterministic and semantic JSON comparison
    query_generator.py  # LLM-based user query simulation
    prompts/            # Prompt templates for judges and generators
  datagen/              # Synthetic data generation via Claude + MCP
    generate.py         # CLI entry point
    generator.py        # Batch generation with Anthropic API
    mcp_client.py       # MCP server communication
    config.py           # Object type registry (8 creation types)
  dataset/              # Dataset management
    sync.py             # CSV/JSON to LangSmith sync with content hashing
  blob_storage/         # Azure Blob Storage client for chat data
  common/               # Shared utilities
    config.py           # TOML-based configuration with dataclass hierarchy
    logger.py           # Rotating file + console logging
    azure_openai.py     # Azure OpenAI model factory
scripts/                # Analysis utilities
  analyze_turn_distribution.py   # Multi-turn conversation depth analysis
  analyze_parallel_usage.py      # Concurrent user pattern analysis
  filter_sut_losses.py           # Post-evaluation loss filtering
tests/                  # Unit and integration tests
```

## Evaluators

| Evaluator | Type | Description |
|-----------|------|-------------|
| `relevance` | LLM-as-judge | Answer relevance scoring (0-1) via openevals |
| `correctness` | LLM-as-judge | Intent/scope/detail match (0, 0.5, 1) |
| `agent_match` | Deterministic | Agent routing validation for multi-agent systems |
| `comparison` | LLM-as-judge | Pairwise A/B with position randomization (50% relevance, 30% completeness, 20% clarity) |
| `json_structure_match` | Deterministic | Order-independent recursive JSON comparison with leaf-counting |
| `json_llm_match` | LLM-as-judge | Semantic field-level JSON evaluation |
| `report_agent_data` | Deterministic | Report agent data extraction validation |

## Setup

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/) (recommended)

1. Install dependencies:
   ```bash
   pip install -e .
   # or with uv
   uv sync
   ```

2. Copy and configure environment:
   ```bash
   cp .env.example .env
   cp config.toml.example config.toml
   ```

3. Fill in API keys in `.env`:
   - `LANGCHAIN_API_KEY` -- LangSmith API key
   - `AZURE_OPENAI_*` -- Azure OpenAI endpoints (per model)
   - `ANTHROPIC_API_KEY` -- For data generation with Claude

4. Configure evaluation settings in `config.toml`.

## Usage

### Run Evaluation (Single-Round)

```bash
python -m src.evaluation.main --config config.toml
```

### Run AI Provider Evaluation (Multi-Turn)

Set `ai_provider = true` in `config.toml` under `[evaluation]`, then:

```bash
python -m src.evaluation.main --config config.toml
```

The query generator will simulate a user, adapting each follow-up based on the SUT's live responses.

### Sync Dataset to LangSmith

```bash
python -m src.dataset.sync --config config.toml
python -m src.dataset.sync --delete  # Remove orphaned examples
```

### Generate Synthetic Data

```bash
uv run -m src.datagen.generate --type promotion_products --count 100
uv run -m src.datagen.generate --type subscription --count 50 --output data/generated_subscriptions.json
```

Supported types: `promotion_products`, `promotion_subscriptions`, `promotion_giftcard`, `subscription`, `bundle`, `product`, `category`

### Run Tests

```bash
pytest
```

## Configuration

All settings live in `config.toml` (see `config.toml.example`):

```toml
[evaluation]
dataset_name = "SUT Benchmark"
experiment_prefix = "sut-script-eval"
max_concurrency = 1
judge_model = "gpt-5-mini"
evaluators = ["relevance", "correctness", "agent_match"]

# AI Provider (multi-turn) settings
ai_provider = false
max_turns = 10
query_generator_model = "gpt-5-mini"
query_generator_user_style = "normal"  # "brief" | "normal" | "verbose"

[evaluation.client]
type = "script"        # "script" | "mcp" | "opensource"
mode = "single_round"  # "single_round" | "multi_round"
script_path = "conversation_service.py"
timeout = 30

[dataset]
name = "SUT Benchmark"
csv_files = ["SUTBenchmark.csv"]

[dataset.sync]
delete_orphans = false

[datagen]
mcp_server_url = "http://127.0.0.1:8000/mcp/"
model = "claude-opus-4-6"
batch_size = 5
```

## How It Works

### Single-Round Evaluation
1. Load dataset examples from LangSmith
2. For each example, send `inputs.question` to SUT client
3. Collect response and run configured evaluators
4. Results tracked as LangSmith experiment

### AI Provider (Multi-Turn) Evaluation
1. Load dataset with scenario and reference output
2. For each example, create a fresh SUT session
3. Query generator LLM produces the first user message from the scenario
4. Send to SUT, receive response
5. Query generator reads conversation history and generates next message
6. Repeat until `is_done=true` or `max_turns` reached
7. Evaluate final state with configured evaluators

### Dataset Sync
- Computes SHA-256 content hash over normalized inputs, outputs, and metadata
- Creates new examples, updates changed ones, skips unchanged
- Detects orphaned examples (in LangSmith but not in local files)
- Supports both CSV (with UTF-8/Windows-1252 fallback) and JSON (standard + conversation format)

## Gaps Between Report and Codebase

The following features exist in the codebase but are not fully covered in the [technical report](docs/technical_report.md):

- **JSON evaluators** (`json_structure_match`, `json_llm_match`) -- deterministic and semantic JSON field comparison with order-independent list matching
- **A/B position randomization** in comparison evaluator to eliminate order bias
- **Query generator user styles** (brief/normal/verbose personas)
- **Content hash change detection** for efficient dataset sync
- **Multiple SUT client types** (script subprocess, MCP agent, OpenAI-compatible) with factory pattern
- **Azure Blob Storage integration** for client-server chat data management
- **Conversation format** dataset with paired user/server message extraction and human-handoff detection
- **TOML-based configuration system** with dataclass hierarchy and singleton pattern

## Contact Information

- **Stephen Wang**
  - Email: zhongqi1112@gmail.com
  - Website: https://stephengineer.github.io
- **Lohith Kumar Bhambore**
  - Email: lohithkumar24@duck.com
  - Github: https://github.com/EmileS24

## License

[MIT](LICENSE)
