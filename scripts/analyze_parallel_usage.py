"""
Analyze parallel (concurrent) usage of client-server bot from processed transcript data.

Calculates how many users were using the bot simultaneously by analyzing
conversation time ranges and counting overlaps within configurable time buckets.

Run from repo root:
  uv run python scripts/analyze_parallel_usage.py --input data/processed_transcripts.json
  uv run python scripts/analyze_parallel_usage.py --input data/processed_transcripts.json --bucket-size 5m
  uv run python scripts/analyze_parallel_usage.py --input data/processed_transcripts.json --output-csv concurrent_users.csv
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)


def parse_bucket_size(bucket_str: str) -> timedelta:
    """Parse bucket size string like '1m', '5m', '15m', '1h' into timedelta."""
    bucket_str = bucket_str.strip().lower()

    # Extract numeric part and unit
    if bucket_str.endswith("m"):
        minutes = int(bucket_str[:-1])
        return timedelta(minutes=minutes)
    elif bucket_str.endswith("h"):
        hours = int(bucket_str[:-1])
        return timedelta(hours=hours)
    elif bucket_str.endswith("s"):
        seconds = int(bucket_str[:-1])
        return timedelta(seconds=seconds)
    else:
        # Assume minutes if just a number
        try:
            return timedelta(minutes=int(bucket_str))
        except ValueError:
            raise ValueError(
                f"Invalid bucket size: {bucket_str}. Use format like '1m', '5m', '15m', '1h'"
            )


def parse_iso_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO 8601 timestamp string to datetime."""
    if not ts_str:
        return None

    try:
        # Handle various ISO 8601 formats
        # With timezone: 2024-01-15T10:30:00Z or 2024-01-15T10:30:00+00:00
        # Without timezone: 2024-01-15T10:30:00
        ts_str = ts_str.strip()

        # Replace Z with +00:00 for Python < 3.11 compatibility
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"

        # Try parsing with fromisoformat
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        # Try alternative formats
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"]:
            try:
                return datetime.strptime(ts_str[: len(fmt)], fmt)
            except ValueError:
                continue
        return None


def load_conversations(input_path: Path) -> List[Dict[str, Any]]:
    """Load processed conversations from JSON file."""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both single conversation and list of conversations
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


def extract_conversation_ranges(
    conversations: List[Dict[str, Any]], date_filter: Optional[str] = None
) -> List[Tuple[str, datetime, datetime]]:
    """
    Extract conversation time ranges from processed data.

    Returns list of (conversation_id, start_time, end_time) tuples.
    Filters out conversations without valid timestamps.
    """
    ranges = []
    skipped = 0

    for conv in conversations:
        conv_id = conv.get("conversation_id", "unknown")
        start_str = conv.get("conversation_start", "")
        end_str = conv.get("conversation_end", "")

        start_dt = parse_iso_timestamp(start_str)
        end_dt = parse_iso_timestamp(end_str)

        if not start_dt or not end_dt:
            skipped += 1
            continue

        # Apply date filter if specified
        if date_filter:
            filter_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
            if start_dt.date() != filter_date:
                continue

        # Ensure start <= end (handle edge cases)
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt

        ranges.append((conv_id, start_dt, end_dt))

    if skipped > 0:
        print(f"[INFO] Skipped {skipped} conversations with missing/invalid timestamps")

    return ranges


def calculate_concurrent_users(
    conversation_ranges: List[Tuple[str, datetime, datetime]], bucket_size: timedelta
) -> Dict[datetime, int]:
    """
    Calculate concurrent users for each time bucket.

    Uses interval overlap counting: a conversation is active in a bucket
    if its time range overlaps with the bucket's time range.
    """
    if not conversation_ranges:
        return {}

    # Find global time range
    all_starts = [r[1] for r in conversation_ranges]
    all_ends = [r[2] for r in conversation_ranges]
    min_time = min(all_starts)
    max_time = max(all_ends)

    # Round min_time down to bucket boundary
    if bucket_size.total_seconds() >= 3600:
        # Hour buckets
        min_time = min_time.replace(minute=0, second=0, microsecond=0)
    elif bucket_size.total_seconds() >= 60:
        # Minute buckets
        min_time = min_time.replace(second=0, microsecond=0)
    else:
        # Second buckets - keep as is
        min_time = min_time.replace(microsecond=0)

    # Generate time buckets and count overlaps
    concurrent_counts = {}
    current_time = min_time

    while current_time <= max_time:
        bucket_end = current_time + bucket_size

        # Count conversations active in this bucket
        count = 0
        for _, conv_start, conv_end in conversation_ranges:
            # Conversation is active if:
            # - It started before the bucket ends AND
            # - It ended after the bucket starts
            if conv_start < bucket_end and conv_end > current_time:
                count += 1

        concurrent_counts[current_time] = count
        current_time = bucket_end

    return concurrent_counts


def calculate_percentile(values: List[int], percentile: float) -> float:
    """Calculate percentile value from a list of integers."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    # Use linear interpolation
    index = (percentile / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1

    if upper >= n:
        return float(sorted_values[-1])

    fraction = index - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def compute_statistics(concurrent_counts: Dict[datetime, int]) -> Dict[str, Any]:
    """Compute summary statistics from concurrent user counts."""
    if not concurrent_counts:
        return {
            "total_buckets": 0,
            "max_concurrent": 0,
            "avg_concurrent": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    values = list(concurrent_counts.values())

    # Find busy periods (top 5 buckets)
    sorted_by_count = sorted(
        concurrent_counts.items(), key=lambda x: x[1], reverse=True
    )
    busy_periods = sorted_by_count[:5]

    return {
        "total_buckets": len(values),
        "max_concurrent": max(values),
        "avg_concurrent": sum(values) / len(values),
        "p50": calculate_percentile(values, 50),
        "p90": calculate_percentile(values, 90),
        "p95": calculate_percentile(values, 95),
        "p99": calculate_percentile(values, 99),
        "busy_periods": busy_periods,
    }


def write_csv_output(concurrent_counts: Dict[datetime, int], output_path: Path) -> None:
    """Write concurrent user counts to CSV file."""
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "concurrent_count"])

            for timestamp, count in sorted(concurrent_counts.items()):
                writer.writerow([timestamp.isoformat(), count])

        print(f"[SAVED] CSV output written to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")


def generate_visualizations(
    concurrent_counts: Dict[datetime, int],
    stats: Dict[str, Any],
    output_dir: Path,
    bucket_size: timedelta,
) -> None:
    """Generate separate visualization plots from concurrent user data."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as e:
        print(f"[WARNING] Cannot generate visualizations - missing dependency: {e}")
        print("[INFO] Install with: uv add --dev matplotlib pandas")
        return

    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert data to DataFrame
        df = pd.DataFrame(
            [
                {"timestamp": ts, "concurrent_count": count}
                for ts, count in sorted(concurrent_counts.items())
            ]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Chart 1: Time Series - Concurrent Users Over Time
        fig1, ax1 = plt.subplots(figsize=(16, 8))
        ax1.plot(
            df["timestamp"], df["concurrent_count"], linewidth=1.5, color="#2E86AB"
        )
        ax1.fill_between(
            df["timestamp"], df["concurrent_count"], alpha=0.3, color="#2E86AB"
        )

        # Add peak annotations
        if "busy_periods" in stats and stats["busy_periods"]:
            for i, (timestamp, count) in enumerate(stats["busy_periods"][:3]):
                ax1.annotate(
                    f"Peak {i+1}: {count} users",
                    xy=(timestamp, count),
                    xytext=(10, 30),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="yellow",
                        alpha=0.8,
                        edgecolor="black",
                    ),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0",
                        color="black",
                        lw=1.5,
                    ),
                    fontsize=11,
                    fontweight="bold",
                )

        ax1.set_title(
            "Concurrent Users Over Time", fontsize=16, fontweight="bold", pad=20
        )
        ax1.set_xlabel("Time", fontsize=13)
        ax1.set_ylabel("Concurrent Users", fontsize=13)
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax1.tick_params(axis="both", which="major", labelsize=11)

        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add stats box
        stats_text = (
            f"Max: {stats['max_concurrent']:.0f} | Avg: {stats['avg_concurrent']:.1f} | "
            f"Median: {stats['p50']:.1f} | p95: {stats['p95']:.1f}"
        )
        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        output_path1 = output_dir / "01_concurrent_users_timeseries.png"
        plt.savefig(output_path1, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[SAVED] Time Series: {output_path1}")

        # Chart 2: Distribution Histogram
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.hist(
            df["concurrent_count"],
            bins=50,
            edgecolor="black",
            color="#A23B72",
            alpha=0.7,
        )

        # Add vertical lines for statistics
        ax2.axvline(
            stats["avg_concurrent"],
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"Average: {stats['avg_concurrent']:.1f} users",
        )
        ax2.axvline(
            stats["p50"],
            color="green",
            linestyle="--",
            linewidth=2.5,
            label=f"Median (p50): {stats['p50']:.1f} users",
        )
        ax2.axvline(
            stats["p95"],
            color="orange",
            linestyle="--",
            linewidth=2.5,
            label=f"95th percentile: {stats['p95']:.1f} users",
        )

        ax2.set_title(
            "Distribution of Concurrent User Counts",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax2.set_xlabel("Concurrent Users", fontsize=13)
        ax2.set_ylabel("Frequency (Number of Time Buckets)", fontsize=13)
        ax2.legend(loc="upper right", fontsize=12, framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax2.tick_params(axis="both", which="major", labelsize=11)

        output_path2 = output_dir / "02_distribution_histogram.png"
        plt.savefig(output_path2, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[SAVED] Distribution: {output_path2}")

        # Chart 3: Percentile Bar Chart
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        percentiles = ["p50\n(Median)", "p90", "p95", "p99", "Max"]
        values = [
            stats["p50"],
            stats["p90"],
            stats["p95"],
            stats["p99"],
            stats["max_concurrent"],
        ]
        colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]

        bars = ax3.bar(
            percentiles, values, color=colors, edgecolor="black", linewidth=1.5
        )

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

        ax3.set_title(
            "Concurrent Users by Percentile", fontsize=16, fontweight="bold", pad=20
        )
        ax3.set_xlabel("Percentile", fontsize=13)
        ax3.set_ylabel("Concurrent Users", fontsize=13)
        ax3.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax3.tick_params(axis="both", which="major", labelsize=12)

        # Add explanation text
        expl_text = f"Based on {stats['total_buckets']:,} time buckets (bucket size: {bucket_size})"
        ax3.text(
            0.5,
            -0.12,
            expl_text,
            transform=ax3.transAxes,
            ha="center",
            fontsize=11,
            style="italic",
        )

        output_path3 = output_dir / "03_percentile_chart.png"
        plt.savefig(output_path3, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[SAVED] Percentile Chart: {output_path3}")

        # Chart 4: Hourly Pattern
        fig4, ax4 = plt.subplots(figsize=(14, 8))
        df["hour"] = df["timestamp"].dt.hour
        hourly_avg = df.groupby("hour")["concurrent_count"].mean().reset_index()

        bars = ax4.bar(
            hourly_avg["hour"],
            hourly_avg["concurrent_count"],
            color="#6C5B7B",
            edgecolor="black",
            alpha=0.8,
            linewidth=1,
        )

        ax4.set_title(
            "Average Concurrent Users by Hour of Day",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax4.set_xlabel("Hour of Day (24-hour format)", fontsize=13)
        ax4.set_ylabel("Average Concurrent Users", fontsize=13)
        ax4.set_xticks(range(0, 24))
        ax4.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax4.tick_params(axis="both", which="major", labelsize=11)

        # Highlight peak hours
        max_hour_idx = hourly_avg["concurrent_count"].idxmax()
        max_hour = (
            int(hourly_avg.loc[max_hour_idx, "hour"]) if max_hour_idx is not None else 0
        )
        ax4.axvline(
            float(max_hour),
            color="red",
            linestyle="--",
            alpha=0.5,
            linewidth=2,
            label=f"Peak Hour: {max_hour:02d}:00",
        )
        ax4.legend(loc="upper right", fontsize=12)

        output_path4 = output_dir / "04_hourly_pattern.png"
        plt.savefig(output_path4, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[SAVED] Hourly Pattern: {output_path4}")

        print(f"\n[COMPLETE] All visualizations saved to: {output_dir}")

    except Exception as e:
        print(f"[ERROR] Failed to generate visualizations: {e}")
        import traceback

        traceback.print_exc()


def print_statistics(stats: Dict[str, Any], bucket_size: timedelta) -> None:
    """Print summary statistics in a formatted way."""
    print()
    print("=" * 60)
    print("PARALLEL USAGE ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Time bucket size: {bucket_size}")
    print(f"Total time buckets analyzed: {stats['total_buckets']}")
    print()
    print("CONCURRENT USERS STATISTICS")
    print("-" * 40)
    print(f"  Maximum:           {stats['max_concurrent']:>6.0f} users")
    print(f"  Average:           {stats['avg_concurrent']:>6.1f} users")
    print(f"  Median (p50):      {stats['p50']:>6.1f} users")
    print(f"  90th percentile:   {stats['p90']:>6.1f} users")
    print(f"  95th percentile:   {stats['p95']:>6.1f} users")
    print(f"  99th percentile:   {stats['p99']:>6.1f} users")
    print()

    if "busy_periods" in stats and stats["busy_periods"]:
        print("BUSIEST TIME PERIODS (Top 5)")
        print("-" * 40)
        for i, (timestamp, count) in enumerate(stats["busy_periods"], 1):
            print(
                f"  {i}. {timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {count} concurrent users"
            )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parallel (concurrent) usage of client-server bot"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to processed transcripts JSON file",
    )
    parser.add_argument(
        "--bucket-size",
        type=str,
        default="1m",
        help="Time bucket size for analysis (e.g., '1m', '5m', '15m', '1h'). Default: 1m",
    )
    parser.add_argument(
        "--output-csv", type=str, help="Output CSV file path for time-series data"
    )
    parser.add_argument(
        "--date-filter", type=str, help="Filter to specific date (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed processing information"
    )
    parser.add_argument(
        "--output-plots",
        type=str,
        help="Directory to save visualization plots (PNG files)",
    )

    args = parser.parse_args()

    # Parse input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    # Parse bucket size
    try:
        bucket_size = parse_bucket_size(args.bucket_size)
        if args.verbose:
            print(f"[INFO] Using bucket size: {bucket_size}")
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Load conversations
    print(f"[INFO] Loading conversations from: {input_path}")
    conversations = load_conversations(input_path)
    print(f"[INFO] Loaded {len(conversations)} conversations")

    if not conversations:
        print("[WARNING] No conversations found. Exiting.")
        sys.exit(0)

    # Extract conversation time ranges
    conversation_ranges = extract_conversation_ranges(
        conversations, date_filter=args.date_filter
    )
    print(f"[INFO] Extracted {len(conversation_ranges)} valid conversation time ranges")

    if not conversation_ranges:
        print(
            "[WARNING] No valid conversation ranges found. Check that your data has 'conversation_start' and 'conversation_end' timestamps."
        )
        sys.exit(0)

    # Calculate concurrent users
    print(f"[INFO] Calculating concurrent users with {args.bucket_size} buckets...")
    concurrent_counts = calculate_concurrent_users(conversation_ranges, bucket_size)

    # Compute statistics
    stats = compute_statistics(concurrent_counts)

    # Print results
    print_statistics(stats, bucket_size)

    # Write CSV if requested
    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        # Ensure parent directory exists
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv_output(concurrent_counts, output_csv_path)

    # Generate visualizations if requested
    if args.output_plots:
        output_plots_dir = Path(args.output_plots)
        generate_visualizations(concurrent_counts, stats, output_plots_dir, bucket_size)

    print("[COMPLETE] Analysis finished")


if __name__ == "__main__":
    main()
