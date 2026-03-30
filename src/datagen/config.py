"""
Configuration for the AI benchmark data generator.

Defines the ObjectTypeConfig registry (which MCP tools + reference data to use
per creation-object type) and the DataGenConfig (MCP URL, model, batch params).
"""

from dataclasses import dataclass


@dataclass
class ObjectTypeConfig:
    """Maps a creation-object type to its MCP tools and reference dataset."""

    schema_tool: str
    data_tools: list[str]
    reference_dataset: str
    id_prefix: str
    description: str


OBJECT_TYPES: dict[str, ObjectTypeConfig] = {
    "promotion_products": ObjectTypeConfig(
        schema_tool="get_promotion_products_schema",
        data_tools=["get_products", "get_categories", "get_employees"],
        reference_dataset="productsAndCategoriesPromotionDataset.json",
        id_prefix="promotion_test_case",
        description="Products & Categories Promotion",
    ),
    "promotion_subscriptions": ObjectTypeConfig(
        schema_tool="get_promotion_subscriptions_schema",
        data_tools=["get_products", "get_categories"],
        reference_dataset="productsAndCategoriesPromotionDataset-2.json",
        id_prefix="promotion_subscription_test_case",
        description="Subscriptions & Bundles Promotion",
    ),
    "promotion_giftcard": ObjectTypeConfig(
        schema_tool="get_promotion_giftcard_schema",
        data_tools=[],
        reference_dataset="productsAndCategoriesPromotionDataset.json",
        id_prefix="promotion_giftcard_test_case",
        description="Gift Cards Promotion",
    ),
    "subscription": ObjectTypeConfig(
        schema_tool="get_subscription_schema",
        data_tools=["get_products", "get_categories", "get_items"],
        reference_dataset="subscriptionDataset.json",
        id_prefix="test_case",
        description="Subscription",
    ),
    "bundle": ObjectTypeConfig(
        schema_tool="get_bundle_schema",
        data_tools=["get_products", "get_categories", "get_items"],
        reference_dataset="bundleDataset.json",
        id_prefix="bundle_test_case",
        description="Bundle",
    ),
    "product": ObjectTypeConfig(
        schema_tool="get_product_schema",
        data_tools=[
            "get_employees",
            "get_business_resources",
            "get_taxes_fees",
            "get_product_add_ons",
        ],
        reference_dataset="product_eval_dataset.json",
        id_prefix="product_test",
        description="Product",
    ),
    "category": ObjectTypeConfig(
        schema_tool="get_category_schema",
        data_tools=[
            "get_employees",
            "get_business_resources",
            "get_taxes_fees",
            "get_category_add_ons",
        ],
        reference_dataset="category_eval_dataset.json",
        id_prefix="category_test",
        description="Category",
    ),
}


@dataclass
class DataGenConfig:
    """Runtime configuration for the data generator."""

    mcp_server_url: str = "http://127.0.0.1:8000/mcp/"
    model: str = "claude-opus-4-6"
    batch_size: int = 5
    temperature: float = 0.8
    max_tokens: int = 16384


def merge_datagen_config(data: dict) -> DataGenConfig:
    """Merge a raw TOML dict into a DataGenConfig."""
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
