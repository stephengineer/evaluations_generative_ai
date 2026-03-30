"""
Analyze turn distribution from processed client-server chat transcripts.

Counts user messages before human agent takes over to understand
conversation depth with the bot.

Run from repo root:
  uv run python scripts/analyze_turn_distribution.py --input data/processed_2026.json
  uv run python scripts/analyze_turn_distribution.py --input data/processed_2026.json --output-csv turns.csv --output-plot turns.png
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)


def count_bot_only_turns(conversation: Dict[str, Any]) -> int:
    """Count turns (user messages before agent handoff)."""
    messages = conversation.get("messages", [])
    bot_only_turns = 0

    for msg in messages:
        if "human_agent_message" in msg:
            break
        if "user_message" in msg:
            bot_only_turns += 1

    return bot_only_turns


def load_conversations(input_path: Path) -> List[Dict[str, Any]]:
    """Load processed conversations from JSON file."""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            print(f"[ERROR] Unexpected data format in {input_path}")
            return []
    except Exception as e:
        print(f"[ERROR] Failed to load {input_path}: {e}")
        return []


def calculate_percentile(values: List[int], percentile: float) -> float:
    """Calculate percentile value from a list of integers."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    index = (percentile / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1

    if upper >= n:
        return float(sorted_values[-1])

    fraction = index - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def analyze_turn_distribution(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze turn distribution across all conversations."""
    turn_counts = []

    for conv in conversations:
        turns = count_bot_only_turns(conv)
        turn_counts.append(turns)

    if not turn_counts:
        return {
            "total_conversations": 0,
            "total_turns": 0,
            "average_turns": 0.0,
            "median_turns": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max_turns": 0,
            "distribution": {},
        }

    # Calculate distribution
    distribution = Counter(turn_counts)
    total_turns = sum(turn_counts)

    return {
        "total_conversations": len(turn_counts),
        "total_turns": total_turns,
        "average_turns": total_turns / len(turn_counts),
        "median_turns": calculate_percentile(turn_counts, 50),
        "p90": calculate_percentile(turn_counts, 90),
        "p95": calculate_percentile(turn_counts, 95),
        "p99": calculate_percentile(turn_counts, 99),
        "max_turns": max(turn_counts),
        "distribution": dict(sorted(distribution.items())),
    }


def write_csv_output(stats: Dict[str, Any], output_path: Path) -> None:
    """Write turn distribution to CSV file."""
    try:
        distribution = stats["distribution"]
        total = stats["total_conversations"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["turn_count", "num_conversations", "percentage"])

            for turn_count, num_convs in sorted(distribution.items()):
                percentage = (num_convs / total) * 100 if total > 0 else 0
                writer.writerow([turn_count, num_convs, f"{percentage:.2f}"])

        print(f"[SAVED] CSV output: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")


def generate_visualization(stats: Dict[str, Any], output_path: Path) -> None:
    """Generate histogram visualization of turn distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[WARNING] matplotlib not available. Install with: uv add --dev matplotlib"
        )
        return

    try:
        distribution = stats["distribution"]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Plot 1: Full distribution
        turn_counts = sorted(distribution.keys())

        fig, ax = plt.subplots(figsize=(14, 8))

        ax.axvline(
            stats["average_turns"],
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Average: {stats['average_turns']:.1f}",
        )
        ax.axvline(
            stats["median_turns"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median (p50): {stats['median_turns']:.1f}",
        )
        ax.axvline(
            stats["p90"],
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"p90: {stats['p90']:.1f}",
        )
        ax.axvline(
            stats["p95"],
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"p95: {stats['p95']:.1f}",
        )
        ax.axvline(
            stats["p99"],
            color="purple",
            linestyle="--",
            linewidth=2,
            label=f"p99: {stats['p99']:.1f}",
        )

        ax.set_title(
            "Turn Distribution (Full View)", fontsize=16, fontweight="bold", pad=20
        )
        ax.set_xlabel("Number of Turns", fontsize=13)
        ax.set_ylabel("Number of Conversations", fontsize=13)
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.tick_params(axis="both", which="major", labelsize=11)

        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[SAVED] Full distribution: {output_path}")

        # Plot 2: Zoomed view (turns 1-10 only)
        fig2, ax2 = plt.subplots(figsize=(12, 8))

        # Filter data for turns 1-10
        zoomed_turns = [t for t in turn_counts if t <= 10]
        zoomed_counts = [distribution[t] for t in zoomed_turns]

        # Calculate percentage for 11+ turns
        turns_11_plus = sum(distribution[t] for t in turn_counts if t > 10)
        if turns_11_plus > 0:
            zoomed_turns.append(11)
            zoomed_counts.append(turns_11_plus)

        bars2 = ax2.bar(
            zoomed_turns,
            zoomed_counts,
            color="#2E86AB",
            edgecolor="black",
            alpha=0.8,
            width=0.7,
        )

        # Color the 11+ bar differently
        if turns_11_plus > 0 and len(bars2) > 0:
            bars2[-1].set_color("#A23B72")
            bars2[-1].set_alpha(0.6)

        # Add percentile lines (only if they fall within 0-11)
        if stats["average_turns"] <= 11:
            ax2.axvline(
                stats["average_turns"],
                color="red",
                linestyle="-",
                linewidth=2.5,
                label=f"Average: {stats['average_turns']:.1f}",
            )
        if stats["median_turns"] <= 11:
            ax2.axvline(
                stats["median_turns"],
                color="green",
                linestyle="--",
                linewidth=2.5,
                label=f"Median: {stats['median_turns']:.1f}",
            )
        if stats["p90"] <= 11:
            ax2.axvline(
                stats["p90"],
                color="blue",
                linestyle="--",
                linewidth=2.5,
                label=f"p90: {stats['p90']:.1f}",
            )

        ax2.set_title(
            "Turn Distribution (Turns 1-10 Focus)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax2.set_xlabel("Number of Turns", fontsize=13)
        ax2.set_ylabel("Number of Conversations", fontsize=13)
        ax2.set_xticks(range(1, 12))
        ax2.set_xticklabels([str(i) for i in range(1, 11)] + ["11+"])
        ax2.legend(loc="upper right", fontsize=12)
        ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax2.tick_params(axis="both", which="major", labelsize=11)

        # Add value labels on top of bars
        for bar, count in zip(bars2, zoomed_counts):
            height = bar.get_height()
            percentage = (count / stats["total_conversations"]) * 100
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{count:,}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        zoomed_path = output_path.parent / (output_path.stem + "_zoomed.png")
        plt.savefig(zoomed_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[SAVED] Zoomed view (1-10): {zoomed_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate visualization: {e}")


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print summary statistics in a formatted way."""
    print()
    print("=" * 60)
    print("BOT-ONLY TURN DISTRIBUTION")
    print("=" * 60)
    print(f"Total Conversations: {stats['total_conversations']:,}")
    print(f"Total Bot-Only Turns: {stats['total_turns']:,}")
    print()
    print("STATISTICS")
    print("-" * 40)
    print(f"  Average turns:           {stats['average_turns']:>6.1f}")
    print(f"  Median turns:            {stats['median_turns']:>6.1f}")
    print(f"  90th percentile:         {stats['p90']:>6.1f}")
    print(f"  95th percentile:         {stats['p95']:>6.1f}")
    print(f"  99th percentile:         {stats['p99']:>6.1f}")
    print(f"  Maximum turns:           {stats['max_turns']:>6.0f}")
    print()

    if stats["distribution"]:
        print("DISTRIBUTION")
        print("-" * 40)
        total = stats["total_conversations"]

        # Show top turn counts (up to 10, then group the rest)
        sorted_items = sorted(stats["distribution"].items())
        shown = 0
        for turn_count, num_convs in sorted_items[:10]:
            percentage = (num_convs / total) * 100
            bar = "#" * int(percentage / 2)
            print(
                f"  {turn_count:>2} turn{'s' if turn_count != 1 else ' '}: {num_convs:>7,} ({percentage:>5.1f}%) {bar}"
            )
            shown += num_convs

        # Show remaining as group if there are more
        if len(sorted_items) > 10:
            remaining = total - shown
            if remaining > 0:
                percentage = (remaining / total) * 100
                print(f"  11+ turns: {remaining:>7,} ({percentage:>5.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze turn distribution from client-server chat transcripts"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to processed transcripts JSON file",
    )
    parser.add_argument(
        "--output-csv", type=str, help="Output CSV file path for turn distribution data"
    )
    parser.add_argument(
        "--output-plot", type=str, help="Output PNG file path for visualization"
    )

    args = parser.parse_args()

    # Parse input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    # Load conversations
    print(f"[INFO] Loading conversations from: {input_path}")
    conversations = load_conversations(input_path)
    print(f"[INFO] Loaded {len(conversations):,} conversations")

    if not conversations:
        print("[WARNING] No conversations found. Exiting.")
        sys.exit(0)

    # Analyze turn distribution
    print("[INFO] Analyzing turn distribution...")
    stats = analyze_turn_distribution(conversations)

    # Print results
    print_statistics(stats)

    # Write CSV if requested
    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv_output(stats, output_csv_path)

    # Generate visualization if requested
    if args.output_plot:
        output_plot_path = Path(args.output_plot)
        generate_visualization(stats, output_plot_path)

    print("[COMPLETE] Analysis finished")


if __name__ == "__main__":
    main()
