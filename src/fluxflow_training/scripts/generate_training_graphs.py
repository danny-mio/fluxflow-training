#!/usr/bin/env python3
"""
Generate training progress diagrams from logged metrics.

Usage:
    python scripts/generate_training_graphs.py <output_path>

Where <output_path> is the path to the training output directory
containing the 'graph' folder with training_metrics.jsonl.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib is required for diagram generation.")
    print("Install it with: pip install matplotlib")
    sys.exit(1)


def load_metrics(metrics_file: Path) -> List[Dict[str, Any]]:
    """
    Load metrics from JSON Lines file.

    Args:
        metrics_file: Path to the training_metrics.jsonl file

    Returns:
        List of metric dictionaries
    """
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        sys.exit(1)

    metrics = []
    with open(metrics_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue

    if not metrics:
        print("Error: No valid metrics found in the file.")
        sys.exit(1)

    return metrics


def extract_metric_series(metrics: List[Dict[str, Any]], metric_name: str) -> tuple:
    """
    Extract a time series for a specific metric.

    Args:
        metrics: List of metric dictionaries
        metric_name: Name of the metric to extract

    Returns:
        Tuple of (steps, values) where both are lists
    """
    steps = []
    values = []

    for entry in metrics:
        if metric_name in entry.get("metrics", {}):
            steps.append(entry["global_step"])
            values.append(entry["metrics"][metric_name])

    return steps, values


def plot_kl_loss(
    metrics: List[Dict[str, Any]], output_dir: Path, verbose: bool = True, prefix: str = ""
):
    """
    Generate KL divergence loss plot with beta warmup (separate due to different scales).

    Args:
        metrics: List of metric dictionaries
        output_dir: Directory to save the plot
        verbose: Whether to print status messages
        prefix: Filename prefix (e.g., "step1_" for pipeline mode)
    """
    steps_kl, kl_vals = extract_metric_series(metrics, "kl_loss")
    steps_beta, beta_vals = extract_metric_series(metrics, "kl_beta")

    if not steps_kl or not kl_vals:
        if verbose:
            print("Warning: No KL loss data found to plot.")
        return

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot KL loss on left y-axis with log scale
    color = "#ff7f0e"
    ax1.set_xlabel("Global Step", fontsize=12)
    ax1.set_ylabel("KL Divergence (log scale)", fontsize=12, color=color)
    ax1.plot(steps_kl, kl_vals, label="KL Divergence", color=color, linewidth=2, alpha=0.8)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot KL beta on right y-axis if available
    if steps_beta and beta_vals:
        ax2 = ax1.twinx()
        color = "#2ca02c"
        ax2.set_ylabel("KL Beta (β)", fontsize=12, color=color)
        ax2.plot(
            steps_beta,
            beta_vals,
            label="KL Beta (β)",
            color=color,
            linewidth=2,
            alpha=0.8,
            linestyle="--",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)
    else:
        ax1.legend(loc="best", fontsize=10)

    ax1.set_title("KL Divergence Loss and Beta Warmup", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_file = output_dir / f"{prefix}kl_loss.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"✓ Generated: {output_file}")
    plt.close(fig)


def plot_losses(
    metrics: List[Dict[str, Any]], output_dir: Path, verbose: bool = True, prefix: str = ""
):
    """
    Generate loss curves plot (excluding KL loss which has different scale).

    Args:
        metrics: List of metric dictionaries
        output_dir: Directory to save the plot
        verbose: Whether to print status messages
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define all possible loss metrics (excluding KL which is plotted separately)
    loss_metrics = {
        "vae_loss": {"label": "VAE Loss", "color": "#1f77b4", "linestyle": "-"},
        "recon_loss": {"label": "Reconstruction Loss", "color": "#17becf", "linestyle": "-"},
        "lpips_loss": {"label": "LPIPS (Perceptual)", "color": "#bcbd22", "linestyle": "-"},
        "flow_loss": {"label": "Flow Loss", "color": "#2ca02c", "linestyle": "-"},
        "discriminator_loss": {
            "label": "Discriminator Loss",
            "color": "#d62728",
            "linestyle": "--",
        },
        "generator_loss": {"label": "Generator Loss", "color": "#9467bd", "linestyle": "--"},
    }

    plotted_any = False
    for metric_name, props in loss_metrics.items():
        steps, values = extract_metric_series(metrics, metric_name)
        if steps and values:
            ax.plot(
                steps,
                values,
                label=props["label"],
                color=props["color"],
                linestyle=props["linestyle"],
                linewidth=2,
                alpha=0.8,
            )
            plotted_any = True

    if not plotted_any:
        if verbose:
            print("Warning: No loss metrics found to plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Loss (log scale)", fontsize=12)
    ax.set_title("Training Loss Curves (excluding KL)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Always use log scale for losses to prevent spikes from hiding other behavior
    ax.set_yscale("log")

    plt.tight_layout()
    output_file = output_dir / f"{prefix}training_losses.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"✓ Generated: {output_file}")
    plt.close(fig)


def plot_learning_rates(
    metrics: List[Dict[str, Any]], output_dir: Path, verbose: bool = True, prefix: str = ""
):
    """
    Generate learning rate curves plot.

    Args:
        metrics: List of metric dictionaries
        output_dir: Directory to save the plot
        verbose: Whether to print status messages
    """
    steps = []
    lr_main = []
    lr_vae = []

    for entry in metrics:
        if "learning_rates" in entry:
            lrs = entry["learning_rates"]
            if "flow_lr" in lrs or "vae_lr" in lrs:
                steps.append(entry["global_step"])
                lr_main.append(lrs.get("flow_lr", 0))
                lr_vae.append(lrs.get("vae_lr", 0))

    if not steps:
        if verbose:
            print("Warning: No learning rate data found to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    if lr_main:
        ax.plot(
            steps,
            lr_main,
            label="Flow Model LR",
            color="#1f77b4",
            linewidth=2,
            marker="o",
            markersize=3,
            alpha=0.8,
        )
    if lr_vae:
        ax.plot(
            steps,
            lr_vae,
            label="VAE LR",
            color="#ff7f0e",
            linewidth=2,
            marker="s",
            markersize=3,
            alpha=0.8,
        )

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    output_file = output_dir / f"{prefix}learning_rates.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"✓ Generated: {output_file}")
    plt.close(fig)


def plot_batch_times(
    metrics: List[Dict[str, Any]], output_dir: Path, verbose: bool = True, prefix: str = ""
):
    """
    Generate batch time plot.

    Args:
        metrics: List of metric dictionaries
        output_dir: Directory to save the plot
        verbose: Whether to print status messages
    """
    steps = []
    batch_times = []

    for entry in metrics:
        if "extras" in entry and "batch_time" in entry["extras"]:
            steps.append(entry["global_step"])
            batch_times.append(entry["extras"]["batch_time"])

    if not steps:
        if verbose:
            print("Warning: No batch time data found to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw batch times
    ax.plot(steps, batch_times, color="#1f77b4", alpha=0.3, linewidth=1)

    # Plot moving average if we have enough data points
    if len(batch_times) > 10:
        window_size = min(50, len(batch_times) // 5)
        batch_times_ma = np.convolve(batch_times, np.ones(window_size) / window_size, mode="valid")
        steps_ma = steps[window_size - 1 :]
        ax.plot(
            steps_ma,
            batch_times_ma,
            label=f"Moving Average (window={window_size})",
            color="#d62728",
            linewidth=2,
        )

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Batch Time (seconds)", fontsize=12)
    ax.set_title("Training Speed", fontsize=14, fontweight="bold")
    if len(batch_times) > 10:
        ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"{prefix}batch_times.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"✓ Generated: {output_file}")
    plt.close(fig)


def plot_combined_overview(
    metrics: List[Dict[str, Any]], output_dir: Path, verbose: bool = True, prefix: str = ""
):
    """
    Generate a combined overview plot with multiple subplots.

    Args:
        metrics: List of metric dictionaries
        output_dir: Directory to save the plot
        verbose: Whether to print status messages
    """
    # Determine how many subplots we need based on available data
    has_vae = any("vae_loss" in entry.get("metrics", {}) for entry in metrics)
    has_flow = any("flow_loss" in entry.get("metrics", {}) for entry in metrics)
    has_lr = any("learning_rates" in entry for entry in metrics)

    num_plots = sum([has_vae, has_flow, has_lr])
    if num_plots == 0:
        if verbose:
            print("Warning: No data available for combined overview.")
        return

    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # VAE losses subplot (excluding KL which has different scale)
    if has_vae:
        ax = axes[plot_idx]
        plot_idx += 1

        steps, vae_vals = extract_metric_series(metrics, "vae_loss")
        if steps and vae_vals:
            ax.plot(steps, vae_vals, label="VAE Loss", color="#1f77b4", linewidth=2)

        steps, disc_vals = extract_metric_series(metrics, "discriminator_loss")
        if steps and disc_vals:
            ax.plot(
                steps,
                disc_vals,
                label="Discriminator Loss",
                color="#d62728",
                linewidth=2,
                linestyle="--",
            )

        ax.set_xlabel("Global Step", fontsize=11)
        ax.set_ylabel("Loss (log scale)", fontsize=11)
        ax.set_title("VAE Training Metrics (excluding KL)", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    # Flow loss subplot
    if has_flow:
        ax = axes[plot_idx]
        plot_idx += 1

        steps, flow_vals = extract_metric_series(metrics, "flow_loss")
        if steps and flow_vals:
            ax.plot(steps, flow_vals, label="Flow Loss", color="#2ca02c", linewidth=2)

        ax.set_xlabel("Global Step", fontsize=11)
        ax.set_ylabel("Loss (log scale)", fontsize=11)
        ax.set_title("Flow Model Training Metrics", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    # Learning rates subplot
    if has_lr:
        ax = axes[plot_idx]

        steps = []
        lr_main = []
        lr_vae = []

        for entry in metrics:
            if "learning_rates" in entry:
                lrs = entry["learning_rates"]
                if "flow_lr" in lrs or "vae_lr" in lrs:
                    steps.append(entry["global_step"])
                    lr_main.append(lrs.get("flow_lr", 0))
                    lr_vae.append(lrs.get("vae_lr", 0))

        if steps:
            if lr_main:
                ax.plot(steps, lr_main, label="Flow Model LR", color="#1f77b4", linewidth=2)
            if lr_vae:
                ax.plot(steps, lr_vae, label="VAE LR", color="#ff7f0e", linewidth=2)

            ax.set_xlabel("Global Step", fontsize=11)
            ax.set_ylabel("Learning Rate", fontsize=11)
            ax.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

    plt.tight_layout()
    output_file = output_dir / f"{prefix}training_overview.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"✓ Generated: {output_file}")
    plt.close(fig)


def generate_summary_stats(
    metrics: List[Dict[str, Any]], output_dir: Path, verbose: bool = True, prefix: str = ""
):
    """
    Generate a text summary of training statistics.

    Args:
        metrics: List of metric dictionaries
        output_dir: Directory to save the summary
        verbose: Whether to print status messages
    """
    if not metrics:
        return

    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("TRAINING SESSION SUMMARY")
    summary_lines.append("=" * 60)

    # Session info
    first_entry = metrics[0]
    last_entry = metrics[-1]

    summary_lines.append(f"\nSession ID: {first_entry.get('session_id', 'unknown')}")
    summary_lines.append(f"First logged: {first_entry.get('timestamp', 'unknown')}")
    summary_lines.append(f"Last logged: {last_entry.get('timestamp', 'unknown')}")
    summary_lines.append(f"Total logged steps: {len(metrics)}")
    summary_lines.append(
        f"Global steps: {first_entry.get('global_step', 0)} → {last_entry.get('global_step', 0)}"
    )

    # Metric statistics
    summary_lines.append("\n" + "=" * 60)
    summary_lines.append("METRIC STATISTICS")
    summary_lines.append("=" * 60)

    # Collect all available metrics
    all_metric_names = set()
    for entry in metrics:
        all_metric_names.update(entry.get("metrics", {}).keys())

    for metric_name in sorted(all_metric_names):
        steps, values = extract_metric_series(metrics, metric_name)
        if values:
            summary_lines.append(f"\n{metric_name}:")
            summary_lines.append(f"  Initial: {values[0]:.6f}")
            summary_lines.append(f"  Final: {values[-1]:.6f}")
            summary_lines.append(f"  Min: {min(values):.6f}")
            summary_lines.append(f"  Max: {max(values):.6f}")
            summary_lines.append(f"  Mean: {np.mean(values):.6f}")
            summary_lines.append(f"  Std: {np.std(values):.6f}")

            # Improvement percentage
            if values[0] != 0:
                improvement = ((values[0] - values[-1]) / abs(values[0])) * 100
                summary_lines.append(f"  Change: {improvement:+.2f}%")

    # Learning rates
    if any("learning_rates" in entry for entry in metrics):
        summary_lines.append("\n" + "=" * 60)
        summary_lines.append("LEARNING RATES")
        summary_lines.append("=" * 60)

        for entry in [first_entry, last_entry]:
            if "learning_rates" in entry:
                lrs = entry["learning_rates"]
                label = "Initial" if entry == first_entry else "Final"
                summary_lines.append(f"\n{label}:")
                for lr_name, lr_value in lrs.items():
                    summary_lines.append(f"  {lr_name}: {lr_value:.2e}")

    summary_lines.append("\n" + "=" * 60)

    # Save summary
    output_file = output_dir / f"{prefix}training_summary.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(summary_lines))

    if verbose:
        print(f"✓ Generated: {output_file}")
        # Also print to console
        print("\n" + "\n".join(summary_lines))


def generate_all_diagrams(output_path: Path, verbose: bool = True) -> bool:
    """
    Generate all training diagrams for a given output path.

    Supports both legacy mode (training_metrics.jsonl) and pipeline mode
    (training_metrics_<step_name>.jsonl for each step).

    Args:
        output_path: Path to the training output directory
        verbose: Whether to print progress messages

    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path).resolve()
        if not output_path.exists():
            if verbose:
                print(f"Error: Output path does not exist: {output_path}")
            return False

        graph_dir = output_path / "graph"
        if not graph_dir.exists():
            if verbose:
                print(f"Error: Graph directory not found: {graph_dir}")
            return False

        # Find all metrics files (legacy or pipeline mode)
        metrics_files = list(graph_dir.glob("training_metrics*.jsonl"))

        if not metrics_files:
            if verbose:
                print(f"Error: No metrics files found in: {graph_dir}")
            return False

        # Generate diagrams for each metrics file
        success_count = 0
        for metrics_file in sorted(metrics_files):
            # Extract step name from filename (if present)
            filename = metrics_file.stem  # "training_metrics" or "training_metrics_step_name"
            if filename == "training_metrics":
                step_name = None  # Legacy mode
                output_prefix = ""
            else:
                # Extract step name: "training_metrics_gan_warmup" → "gan_warmup"
                step_name = filename.replace("training_metrics_", "")
                output_prefix = f"{step_name}_"

            if verbose:
                if step_name:
                    print(f"\n{'='*60}")
                    print(f"Processing step: {step_name}")
                    print(f"{'='*60}")
                print(f"Loading metrics from: {metrics_file}")

            metrics = load_metrics(metrics_file)

            if not metrics:
                if verbose:
                    print(f"No metrics found in {metrics_file}")
                continue

            if verbose:
                print(f"Loaded {len(metrics)} metric entries")
                print(f"\nGenerating diagrams in: {graph_dir}")
                print("-" * 60)

            # Generate all plots with step-specific prefixes
            plot_kl_loss(metrics, graph_dir, verbose, prefix=output_prefix)
            plot_losses(metrics, graph_dir, verbose, prefix=output_prefix)
            plot_learning_rates(metrics, graph_dir, verbose, prefix=output_prefix)
            plot_batch_times(metrics, graph_dir, verbose, prefix=output_prefix)
            plot_combined_overview(metrics, graph_dir, verbose, prefix=output_prefix)
            generate_summary_stats(metrics, graph_dir, verbose, prefix=output_prefix)

            success_count += 1

            if verbose:
                print("-" * 60)
                if step_name:
                    print(f"✓ Step '{step_name}' diagrams generated successfully!")
                else:
                    print("✓ Diagrams generated successfully!")

        if verbose:
            print(f"\n{'='*60}")
            print(f"✓ Generated diagrams for {success_count} file(s)")
            print(f"View your results in: {graph_dir}")
            print(f"{'='*60}")

        return success_count > 0

    except Exception as e:
        if verbose:
            print(f"Error generating diagrams: {e}")
        return False


def main():
    """Main entry point for diagram generation."""
    parser = argparse.ArgumentParser(
        description="Generate training progress diagrams from logged metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_training_graphs.py ./output
  python scripts/generate_training_graphs.py /path/to/training/output
        """,
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the training output directory containing the graph folder",
    )

    args = parser.parse_args()

    success = generate_all_diagrams(Path(args.output_path), verbose=True)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
