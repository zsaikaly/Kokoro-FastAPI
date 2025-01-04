"""Shared plotting utilities for benchmarks and tests."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Common style configurations
STYLE_CONFIG = {
    "background_color": "#1a1a2e",
    "primary_color": "#ff2a6d",
    "secondary_color": "#05d9e8",
    "grid_color": "#ffffff",
    "text_color": "#ffffff",
    "font_sizes": {
        "title": 16,
        "label": 14,
        "tick": 12,
        "text": 10
    }
}

def setup_plot(fig, ax, title, xlabel=None, ylabel=None):
    """Configure plot styling with consistent theme.
    
    Args:
        fig: matplotlib figure object
        ax: matplotlib axis object
        title: str, plot title
        xlabel: str, optional x-axis label
        ylabel: str, optional y-axis label
    
    Returns:
        tuple: (fig, ax) with applied styling
    """
    # Grid styling
    ax.grid(True, linestyle="--", alpha=0.3, color=STYLE_CONFIG["grid_color"])
    
    # Title and labels
    ax.set_title(title, pad=20, 
                fontsize=STYLE_CONFIG["font_sizes"]["title"], 
                fontweight="bold", 
                color=STYLE_CONFIG["text_color"])
    
    if xlabel:
        ax.set_xlabel(xlabel, 
                     fontsize=STYLE_CONFIG["font_sizes"]["label"], 
                     fontweight="medium", 
                     color=STYLE_CONFIG["text_color"])
    if ylabel:
        ax.set_ylabel(ylabel, 
                     fontsize=STYLE_CONFIG["font_sizes"]["label"], 
                     fontweight="medium", 
                     color=STYLE_CONFIG["text_color"])
    
    # Tick styling
    ax.tick_params(labelsize=STYLE_CONFIG["font_sizes"]["tick"], 
                  colors=STYLE_CONFIG["text_color"])
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(STYLE_CONFIG["text_color"])
        spine.set_alpha(0.3)
        spine.set_linewidth(0.5)
    
    # Background colors
    ax.set_facecolor(STYLE_CONFIG["background_color"])
    fig.patch.set_facecolor(STYLE_CONFIG["background_color"])
    
    return fig, ax

def plot_system_metrics(metrics_data, output_path):
    """Create plots for system metrics over time.
    
    Args:
        metrics_data: list of dicts containing system metrics
        output_path: str, path to save the output plot
    """
    df = pd.DataFrame(metrics_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    elapsed_time = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    
    # Get baseline values
    baseline_cpu = df["cpu_percent"].iloc[0]
    baseline_ram = df["ram_used_gb"].iloc[0]
    baseline_gpu = df["gpu_memory_used"].iloc[0] / 1024 if "gpu_memory_used" in df.columns else None
    
    # Convert GPU memory to GB if present
    if "gpu_memory_used" in df.columns:
        df["gpu_memory_gb"] = df["gpu_memory_used"] / 1024
    
    plt.style.use("dark_background")
    
    # Create subplots based on available metrics
    has_gpu = "gpu_memory_used" in df.columns
    num_plots = 3 if has_gpu else 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots))
    fig.patch.set_facecolor(STYLE_CONFIG["background_color"])
    
    # Smoothing window
    window = min(5, len(df) // 2)
    
    # Plot CPU Usage
    smoothed_cpu = df["cpu_percent"].rolling(window=window, center=True).mean()
    sns.lineplot(x=elapsed_time, y=smoothed_cpu, ax=axes[0], 
                color=STYLE_CONFIG["primary_color"], linewidth=2)
    axes[0].axhline(y=baseline_cpu, color=STYLE_CONFIG["secondary_color"], 
                    linestyle="--", alpha=0.5, label="Baseline")
    setup_plot(fig, axes[0], "CPU Usage Over Time", 
              xlabel="Time (seconds)", ylabel="CPU Usage (%)")
    axes[0].set_ylim(0, max(df["cpu_percent"]) * 1.1)
    axes[0].legend()
    
    # Plot RAM Usage
    smoothed_ram = df["ram_used_gb"].rolling(window=window, center=True).mean()
    sns.lineplot(x=elapsed_time, y=smoothed_ram, ax=axes[1], 
                color=STYLE_CONFIG["secondary_color"], linewidth=2)
    axes[1].axhline(y=baseline_ram, color=STYLE_CONFIG["primary_color"], 
                    linestyle="--", alpha=0.5, label="Baseline")
    setup_plot(fig, axes[1], "RAM Usage Over Time", 
              xlabel="Time (seconds)", ylabel="RAM Usage (GB)")
    axes[1].set_ylim(0, max(df["ram_used_gb"]) * 1.1)
    axes[1].legend()
    
    # Plot GPU Memory if available
    if has_gpu:
        smoothed_gpu = df["gpu_memory_gb"].rolling(window=window, center=True).mean()
        sns.lineplot(x=elapsed_time, y=smoothed_gpu, ax=axes[2], 
                    color=STYLE_CONFIG["primary_color"], linewidth=2)
        axes[2].axhline(y=baseline_gpu, color=STYLE_CONFIG["secondary_color"], 
                        linestyle="--", alpha=0.5, label="Baseline")
        setup_plot(fig, axes[2], "GPU Memory Usage Over Time", 
                  xlabel="Time (seconds)", ylabel="GPU Memory (GB)")
        axes[2].set_ylim(0, max(df["gpu_memory_gb"]) * 1.1)
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_correlation(df, x, y, title, xlabel, ylabel, output_path):
    """Create correlation plot with regression line and correlation coefficient.
    
    Args:
        df: pandas DataFrame containing the data
        x: str, column name for x-axis
        y: str, column name for y-axis
        title: str, plot title
        xlabel: str, x-axis label
        ylabel: str, y-axis label
        output_path: str, path to save the output plot
    """
    plt.style.use("dark_background")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    sns.scatterplot(data=df, x=x, y=y, s=100, alpha=0.6, 
                    color=STYLE_CONFIG["primary_color"])
    
    # Regression line
    sns.regplot(data=df, x=x, y=y, scatter=False, 
                color=STYLE_CONFIG["secondary_color"], 
                line_kws={"linewidth": 2})
    
    # Add correlation coefficient
    corr = df[x].corr(df[y])
    plt.text(0.05, 0.95, f"Correlation: {corr:.2f}", 
             transform=ax.transAxes, 
             fontsize=STYLE_CONFIG["font_sizes"]["text"], 
             color=STYLE_CONFIG["text_color"],
             bbox=dict(facecolor=STYLE_CONFIG["background_color"], 
                      edgecolor=STYLE_CONFIG["text_color"], 
                      alpha=0.7))
    
    setup_plot(fig, ax, title, xlabel=xlabel, ylabel=ylabel)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
