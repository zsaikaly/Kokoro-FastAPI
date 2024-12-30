import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def setup_plot(fig, ax, title):
    """Configure plot styling"""
    # Improve grid
    ax.grid(True, linestyle='--', alpha=0.3, color='#ffffff')
    
    # Set title and labels with better fonts
    ax.set_title(title, pad=20, fontsize=16, fontweight='bold', color='#ffffff')
    ax.set_xlabel(ax.get_xlabel(), fontsize=12, fontweight='medium', color='#ffffff')
    ax.set_ylabel(ax.get_ylabel(), fontsize=12, fontweight='medium', color='#ffffff')
    
    # Improve tick labels
    ax.tick_params(labelsize=10, colors='#ffffff')
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color('#ffffff')
        spine.set_alpha(0.3)
        spine.set_linewidth(0.5)
    
    # Set background colors
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    return fig, ax

def main():
    # Load benchmark results
    with open('examples/benchmark_results.json', 'r') as f:
        results = json.load(f)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Set the style
    plt.style.use('dark_background')
    
    # Plot 1: Processing Time vs Output Length
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with custom styling
    scatter = sns.scatterplot(data=df, x='output_length', y='processing_time', 
                            s=100, alpha=0.6, color='#ff2a6d')  # Neon pink
    
    # Add regression line with confidence interval
    sns.regplot(data=df, x='output_length', y='processing_time', 
                scatter=False, color='#05d9e8',  # Neon blue
                line_kws={'linewidth': 2})
    
    # Calculate correlation
    corr = df['output_length'].corr(df['processing_time'])
    
    # Add correlation annotation
    plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
             transform=ax.transAxes, fontsize=10, color='#ffffff',
             bbox=dict(facecolor='#1a1a2e', edgecolor='#ffffff', alpha=0.7))
    
    setup_plot(fig, ax, 'Processing Time vs Output Length')
    ax.set_xlabel('Output Audio Length (seconds)')
    ax.set_ylabel('Processing Time (seconds)')
    
    plt.savefig('examples/time_vs_output.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Processing Time vs Token Count
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with custom styling
    scatter = sns.scatterplot(data=df, x='tokens', y='processing_time', 
                            s=100, alpha=0.6, color='#ff2a6d')  # Neon pink
    
    # Add regression line with confidence interval
    sns.regplot(data=df, x='tokens', y='processing_time', 
                scatter=False, color='#05d9e8',  # Neon blue
                line_kws={'linewidth': 2})
    
    # Calculate correlation
    corr = df['tokens'].corr(df['processing_time'])
    
    # Add correlation annotation
    plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
             transform=ax.transAxes, fontsize=10, color='#ffffff',
             bbox=dict(facecolor='#1a1a2e', edgecolor='#ffffff', alpha=0.7))
    
    setup_plot(fig, ax, 'Processing Time vs Token Count')
    ax.set_xlabel('Number of Input Tokens')
    ax.set_ylabel('Processing Time (seconds)')
    
    plt.savefig('examples/time_vs_tokens.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
