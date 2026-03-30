#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting pre-push checks..."

echo ""
echo "📦 Running Ruff (Linting)..."
uv run ruff check . --fix

echo ""
echo "⚫ Running Black (Formatting Check)..."
uv run black .

echo ""
echo "📦 Running Deptry (Dependency Check)..."
uv run deptry .

echo ""
echo "🔍 Running Mypy (Type Checking)..."
uv run mypy .

echo ""
echo "🧪 Running Pytest (Unit Tests)..."
uv run pytest

echo ""
echo "✅ All checks passed! You are ready to push."
