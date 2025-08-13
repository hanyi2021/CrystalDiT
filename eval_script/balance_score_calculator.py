import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re

def calculate_balance_score(un_rate, structural_validity, chemical_validity, 
                          density_diff, elements_diff, alpha=1.0):
    """
    Calculate BALANCE = UN_rate × quality_composite^alpha
    
    Args:
        un_rate: UN rate (0-1, higher is better)
        structural_validity: Structural validity (0.95-1.0, higher is better)
        chemical_validity: Chemical validity (0.8-1.0, higher is better) 
        density_diff: Density distribution difference (0.1-1.0, lower is better)
        elements_diff: Elements distribution difference (0.1-1.0, lower is better)
        alpha: Quality weight exponent (default 1.0)
    
    Returns:
        balance_score: Final balance score
        components: Dictionary with component scores for analysis
    """
    
    # 1. Structural validity score (95-100% range)
    struct_score = max(0, min(1, (structural_validity - 0.95) / 0.05))
    
    # 2. Chemical validity score (80-100% range) 
    chem_score = max(0, min(1, (chemical_validity - 0.8) / 0.2))
    
    # 3. Density distribution score (1.0-0.1 range, lower is better)
    density_score = max(0, min(1, (1.0 - density_diff) / 0.9))
    
    # 4. Elements distribution score (1.0-0.1 range, lower is better)
    elements_score = max(0, min(1, (1.0 - elements_diff) / 0.9))
    
    # 5. Quality composite (geometric mean - any poor score hurts overall)
    quality_composite = (struct_score * chem_score * density_score * elements_score) ** (1/4)
    
    # 6. Final balance score: UN_rate × quality_composite^alpha
    balance_score = un_rate * (quality_composite ** alpha)
    
    components = {
        'un_rate': un_rate,
        'structural_score': struct_score,
        'chemical_score': chem_score,
        'density_score': density_score,
        'elements_score': elements_score,
        'quality_composite': quality_composite,
        'alpha': alpha,
        'balance_score': balance_score
    }
    
    return balance_score, components

def extract_epoch_from_filename(filename):
    """Extract epoch number from filename like 'noema_checkpoint_epoch_23000_results.json'"""
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else 0

def smooth_scores(results_dict, window_size=5):
    """
    Apply moving average smoothing to balance scores for all alpha values.
    
    Args:
        results_dict: Dictionary with alpha as key and DataFrame as value
        window_size: Number of neighboring points to average (default: 5)
        
    Returns:
        results_dict: Dictionary with smoothed scores added
    """
    for alpha, results_df in results_dict.items():
        if results_df.empty or len(results_df) < window_size:
            results_df['balance_score_smoothed'] = results_df['balance_score']
            results_df['un_rate_smoothed'] = results_df['un_rate']
            continue
        
        # Sort by epoch first
        df_sorted = results_df.sort_values('epoch').copy()
        
        # Calculate moving averages
        df_sorted['balance_score_smoothed'] = df_sorted['balance_score'].rolling(
            window=window_size, center=True, min_periods=1).mean()
        df_sorted['un_rate_smoothed'] = df_sorted['un_rate'].rolling(
            window=window_size, center=True, min_periods=1).mean()
        df_sorted['quality_composite_smoothed'] = df_sorted['quality_composite'].rolling(
            window=window_size, center=True, min_periods=1).mean()
        
        # Also smooth component scores for better analysis
        df_sorted['structural_score_smoothed'] = df_sorted['structural_score'].rolling(
            window=window_size, center=True, min_periods=1).mean()
        df_sorted['chemical_score_smoothed'] = df_sorted['chemical_score'].rolling(
            window=window_size, center=True, min_periods=1).mean()
        df_sorted['density_score_smoothed'] = df_sorted['density_score'].rolling(
            window=window_size, center=True, min_periods=1).mean()
        df_sorted['elements_score_smoothed'] = df_sorted['elements_score'].rolling(
            window=window_size, center=True, min_periods=1).mean()
        
        # Restore original order and add smoothed columns
        results_df_updated = results_df.merge(
            df_sorted[['filename', 'balance_score_smoothed', 'un_rate_smoothed', 
                      'quality_composite_smoothed', 'structural_score_smoothed',
                      'chemical_score_smoothed', 'density_score_smoothed', 
                      'elements_score_smoothed']], 
            on='filename'
        )
        
        results_dict[alpha] = results_df_updated
    
    return results_dict
    """Extract epoch number from filename like 'noema_checkpoint_epoch_23000_results.json'"""
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else 0

def process_results_folder(folder_path, alpha_values=None, max_epoch=50000):
    """
    Process all JSON result files in the folder and calculate balance scores.
    
    Args:
        folder_path: Path to folder containing JSON result files
        alpha_values: List of alpha values to calculate, or single float value
        max_epoch: Maximum epoch to include (default: 50000)
        
    Returns:
        results_dict: Dictionary with alpha as key and DataFrame as value
    """
    folder_path = Path(folder_path)
    
    # Default alpha values if not specified
    if alpha_values is None:
        alpha_values = [0.5, 1.0, 1.5, 2.0]
    elif isinstance(alpha_values, (int, float)):
        alpha_values = [float(alpha_values)]
    
    # First, extract basic data from all JSON files
    basic_results = []
    
    # Process all JSON files in the folder
    for json_file in folder_path.glob("*.json"):
        # Skip best and latest checkpoints
        if 'best' in json_file.name.lower() or 'latest' in json_file.name.lower():
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract epoch from filename
            epoch = extract_epoch_from_filename(json_file.name)
            
            # Filter out epochs beyond max_epoch
            if epoch > max_epoch:
                continue
            
            # Extract required metrics
            un_rate = data['un_metrics']['un_rate']
            structural_validity = data['validity_metrics']['structural_validity']
            chemical_validity = data['validity_metrics']['chemical_validity']
            density_diff = data['distribution_metrics']['d_density_train']
            elements_diff = data['distribution_metrics']['d_elements_train']
            
            # Store basic results
            basic_result = {
                'filename': json_file.name,
                'epoch': epoch,
                'un_rate': un_rate,
                'structural_validity': structural_validity,
                'chemical_validity': chemical_validity,
                'density_diff': density_diff,
                'elements_diff': elements_diff
            }
            basic_results.append(basic_result)
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Now calculate balance scores for each alpha value
    results_dict = {}
    
    for alpha in alpha_values:
        results = []
        
        for basic_result in basic_results:
            # Calculate balance score for this alpha
            balance_score, components = calculate_balance_score(
                basic_result['un_rate'], 
                basic_result['structural_validity'], 
                basic_result['chemical_validity'],
                basic_result['density_diff'], 
                basic_result['elements_diff'], 
                alpha
            )
            
            # Store complete results
            result = {**basic_result, **components}
            results.append(result)
        
        # Convert to DataFrame and sort by balance score
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('balance_score', ascending=False)
        
        results_dict[alpha] = results_df
    
    return results_dict

def create_quality_plots(results_dict, folder_path, show_smoothed=True):
    """Create quality metric plots vs epoch for alpha=1.0."""
    
    # Use alpha=1.0 for quality plots (or first available alpha)
    alpha_values = sorted(results_dict.keys())
    if 1.0 in alpha_values:
        alpha = 1.0
    else:
        alpha = alpha_values[0]
    
    results_df = results_dict[alpha]
    if results_df.empty:
        print("No data for quality plots")
        return []
    
    df_sorted = results_df.sort_values('epoch')
    plot_paths = []
    
    # 1. Overall Quality Composite vs Epoch
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    if show_smoothed and 'quality_composite_smoothed' in df_sorted.columns:
        ax.plot(df_sorted['epoch'], df_sorted['quality_composite'], 
                'o-', linewidth=1, markersize=3, color='#2E86AB', alpha=0.3, label='Raw')
        ax.plot(df_sorted['epoch'], df_sorted['quality_composite_smoothed'], 
                'o-', linewidth=2, markersize=4, color='#2E86AB', label='Smoothed')
    else:
        ax.plot(df_sorted['epoch'], df_sorted['quality_composite'], 
                'o-', linewidth=2, markersize=4, color='#2E86AB', label='Quality Composite')
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Quality Composite Score')
    ax.set_title('Quality Composite vs Training Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    quality_path = Path(folder_path) / "quality_composite_vs_epoch.png"
    plt.savefig(quality_path, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(quality_path)
    print(f"Quality composite plot saved to: {quality_path}")
    
    # 2. Individual Quality Metrics (2x2 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = [
        ('structural_score', 'Structural Validity Score', axes[0, 0]),
        ('chemical_score', 'Chemical Validity Score', axes[0, 1]),
        ('density_score', 'Density Distribution Score', axes[1, 0]),
        ('elements_score', 'Elements Distribution Score', axes[1, 1])
    ]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (metric, title, ax) in enumerate(metrics):
        color = colors[i]
        
        if show_smoothed and f'{metric}_smoothed' in df_sorted.columns:
            ax.plot(df_sorted['epoch'], df_sorted[metric], 
                    'o-', linewidth=1, markersize=2, color=color, alpha=0.3, label='Raw')
            ax.plot(df_sorted['epoch'], df_sorted[f'{metric}_smoothed'], 
                    'o-', linewidth=2, markersize=3, color=color, label='Smoothed')
        else:
            ax.plot(df_sorted['epoch'], df_sorted[metric], 
                    'o-', linewidth=2, markersize=3, color=color, label=title)
        
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    individual_path = Path(folder_path) / "individual_quality_metrics_vs_epoch.png"
    plt.savefig(individual_path, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(individual_path)
    print(f"Individual quality metrics plot saved to: {individual_path}")
    
    return plot_paths
def create_visualizations(results_dict, folder_path, show_smoothed=True):
    """Create visualizations for multiple alpha values."""
    
    if not results_dict:
        print("No data to visualize")
        return []
    
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    
    plot_paths = []
    alpha_values = sorted(results_dict.keys())
    
    # First create quality plots
    print("Creating quality metric plots...")
    quality_paths = create_quality_plots(results_dict, folder_path, show_smoothed)
    plot_paths.extend(quality_paths)
    
    # Create individual plots for each alpha
    for alpha in alpha_values:
        results_df = results_dict[alpha]
        
        if results_df.empty:
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        df_sorted = results_df.sort_values('epoch')
        
        # Plot balance score
        score_col = 'balance_score_smoothed' if show_smoothed and 'balance_score_smoothed' in df_sorted.columns else 'balance_score'
        if show_smoothed and 'balance_score_smoothed' in df_sorted.columns:
            ax.plot(df_sorted['epoch'], df_sorted['balance_score'], 
                    'o-', linewidth=1, markersize=3, color='#2E86AB', alpha=0.3, label='Raw')
            ax.plot(df_sorted['epoch'], df_sorted['balance_score_smoothed'], 
                    'o-', linewidth=2, markersize=4, color='#2E86AB', label='Smoothed')
        else:
            ax.plot(df_sorted['epoch'], df_sorted['balance_score'], 
                    'o-', linewidth=2, markersize=4, color='#2E86AB', label='Balance Score')
        
        # Plot UN rate on secondary axis
        ax2 = ax.twinx()
        if show_smoothed and 'un_rate_smoothed' in df_sorted.columns:
            ax2.plot(df_sorted['epoch'], df_sorted['un_rate'], 
                     's--', linewidth=1, markersize=2, color='#A23B72', alpha=0.3)
            ax2.plot(df_sorted['epoch'], df_sorted['un_rate_smoothed'], 
                     's-', linewidth=2, markersize=3, color='#A23B72', label='UN Rate')
        else:
            ax2.plot(df_sorted['epoch'], df_sorted['un_rate'], 
                     's--', linewidth=2, markersize=3, color='#A23B72', label='UN Rate')
        
        # Calculate stage boundaries and mark best points in each stage
        min_epoch = results_df['epoch'].min()
        max_epoch = results_df['epoch'].max()
        epoch_range = max_epoch - min_epoch
        
        stage_1_max = min_epoch + 0.3 * epoch_range
        stage_2_max = min_epoch + 0.6 * epoch_range
        
        stages = [
            ("Early (0-30%)", results_df[results_df['epoch'] <= stage_1_max], 'red'),
            ("Mid (31-60%)", results_df[(results_df['epoch'] > stage_1_max) & (results_df['epoch'] <= stage_2_max)], 'orange'),
            ("Late (61-100%)", results_df[results_df['epoch'] > stage_2_max], 'green')
        ]
        
        # Mark best point in each stage
        for i, (stage_name, stage_df, color) in enumerate(stages):
            if stage_df.empty:
                continue
            
            best_row = stage_df.loc[stage_df[score_col].idxmax()]
            ax.scatter(best_row['epoch'], best_row[score_col], 
                      s=120, c=color, marker='*', zorder=10, edgecolors='black', linewidth=1,
                      label=f'Best {stage_name}')
        
        # Add vertical lines for stage boundaries
        ax.axvline(x=stage_1_max, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=stage_2_max, color='gray', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Balance Score', color='#2E86AB')
        ax2.set_ylabel('UN Rate', color='#A23B72')
        ax.set_title(f'Balance Score vs Epoch (alpha = {alpha})\nFormula: UN_rate × quality_composite^{alpha}')
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='#2E86AB')
        ax2.tick_params(axis='y', labelcolor='#A23B72')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(folder_path) / f"balance_score_alpha_{alpha:.1f}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plot_paths.append(plot_path)
        print(f"Plot for alpha={alpha} saved to: {plot_path}")
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, alpha in enumerate(alpha_values):
        results_df = results_dict[alpha]
        if results_df.empty:
            continue
            
        df_sorted = results_df.sort_values('epoch')
        score_col = 'balance_score_smoothed' if show_smoothed and 'balance_score_smoothed' in df_sorted.columns else 'balance_score'
        
        color = colors[i % len(colors)]
        ax.plot(df_sorted['epoch'], df_sorted[score_col], 
                'o-', linewidth=2, markersize=4, color=color, label=f'alpha = {alpha}')
        
        # Mark best point for each alpha
        best_idx = df_sorted[score_col].idxmax()
        best_row = df_sorted.loc[best_idx]
        ax.scatter(best_row['epoch'], best_row[score_col], 
                  s=100, c=color, marker='*', zorder=10, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Balance Score')
    ax.set_title('Balance Score Comparison: UN_rate × quality_composite^alpha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    comparison_path = Path(folder_path) / "balance_score_alpha_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    plot_paths.append(comparison_path)
    print(f"Comparison plot saved to: {comparison_path}")
    
    # Print summary
    print(f"\nSUMMARY: Alpha values {alpha_values}")
    for alpha in alpha_values:
        results_df = results_dict[alpha]
        if results_df.empty:
            continue
        score_col = 'balance_score_smoothed' if show_smoothed and 'balance_score_smoothed' in results_df.columns else 'balance_score'
        best_score = results_df[score_col].max()
        best_epoch = results_df.loc[results_df[score_col].idxmax(), 'epoch']
        print(f"Alpha {alpha}: Best Score = {best_score:.4f} (Epoch {best_epoch})")
    
    return plot_paths

def print_top_results(results_dict, top_n=3, use_smoothed=True):
    """Print the best results from different training stages for each alpha value."""
    
    if not results_dict:
        print("No results to display")
        return
    
    alpha_values = sorted(results_dict.keys())
    
    for alpha in alpha_values:
        results_df = results_dict[alpha]
        
        if results_df.empty:
            print(f"\nNo results for alpha = {alpha}")
            continue
        
        score_col = 'balance_score_smoothed' if use_smoothed and 'balance_score_smoothed' in results_df.columns else 'balance_score'
        
        print(f"\n{'='*80}")
        print(f"BEST MODELS BY TRAINING STAGE (Alpha = {alpha})")
        print(f"{'='*80}")
        
        # Calculate stage boundaries
        min_epoch = results_df['epoch'].min()
        max_epoch = results_df['epoch'].max()
        epoch_range = max_epoch - min_epoch
        
        stage_1_max = min_epoch + 0.3 * epoch_range
        stage_2_max = min_epoch + 0.6 * epoch_range
        
        stages = [
            ("Early (0-30%)", results_df[results_df['epoch'] <= stage_1_max]),
            ("Mid (31-60%)", results_df[(results_df['epoch'] > stage_1_max) & (results_df['epoch'] <= stage_2_max)]),
            ("Late (61-100%)", results_df[results_df['epoch'] > stage_2_max])
        ]
        
        for stage_name, stage_df in stages:
            if stage_df.empty:
                print(f"\n{stage_name}: No checkpoints")
                continue
            
            best_row = stage_df.loc[stage_df[score_col].idxmax()]
            
            print(f"\n{stage_name}: {best_row['filename']}")
            print(f"Score: {best_row[score_col]:.4f} | UN: {best_row['un_rate']:.3f} | Epoch: {best_row['epoch']:,}")
            print(f"Structural: {best_row['structural_validity']:.3f} | Chemical: {best_row['chemical_validity']:.3f}")
            print(f"Density Diff: {best_row['density_diff']:.3f} | Elements Diff: {best_row['elements_diff']:.3f}")
        
        # Summary table
        print(f"\nSTAGE COMPARISON (Alpha = {alpha}):")
        print("Stage            | Score   | UN Rate | Epoch   | Progress")
        print("-" * 55)
        
        for stage_name, stage_df in stages:
            if stage_df.empty:
                continue
            
            best_row = stage_df.loc[stage_df[score_col].idxmax()]
            progress = ((best_row['epoch'] - min_epoch) / epoch_range * 100)
            
            print(f"{stage_name:16} | {best_row[score_col]:7.4f} | {best_row['un_rate']:7.3f} | {best_row['epoch']:7,} | {progress:6.1f}%")

def main(folder_path, alpha=None, window_size=5, max_epoch=50000):
    """Main function to process results and generate analysis."""
    
    print("Crystal Generation Model Balance Score Analysis")
    print("Formula: BALANCE = UN_rate × quality_composite^alpha")
    print(f"Max epoch filter: {max_epoch}")
    print("="*60)
    
    # Process JSON files
    print("Processing JSON files...")
    results_dict = process_results_folder(folder_path, alpha, max_epoch)
    
    if not results_dict or all(df.empty for df in results_dict.values()):
        print("ERROR: No valid JSON files found.")
        return
    
    alpha_values = sorted(results_dict.keys())
    total_checkpoints = len(next(iter(results_dict.values())))
    print(f"Processed {total_checkpoints} checkpoint files (epoch <= {max_epoch})")
    print(f"Alpha values: {alpha_values}")
    
    # Apply smoothing
    if window_size > 1:
        print(f"Applying smoothing (window size: {window_size})...")
        results_dict = smooth_scores(results_dict, window_size)
        use_smoothed = True
    else:
        use_smoothed = False
    
    # Sort results
    for alpha in alpha_values:
        score_col = 'balance_score_smoothed' if use_smoothed and 'balance_score_smoothed' in results_dict[alpha].columns else 'balance_score'
        results_dict[alpha] = results_dict[alpha].sort_values(score_col, ascending=False)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_paths = create_visualizations(results_dict, folder_path, use_smoothed)
    
    # Print results
    print_top_results(results_dict, use_smoothed=use_smoothed)
    
    # Save CSV files
    for alpha in alpha_values:
        output_file = Path(folder_path) / f"balance_score_analysis_alpha_{alpha:.1f}.csv"
        results_dict[alpha].to_csv(output_file, index=False)
        print(f"Results for alpha={alpha} saved to: {output_file}")
    
    print(f"\nAnalysis completed: {len(plot_paths)} plots generated")
    
    return results_dict

# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Crystal Generation Model Balance Score Analysis')
    parser.add_argument('folder_path', type=str, 
                       help='Path to folder containing JSON result files')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Quality weight exponent alpha (if not specified, analyzes alpha=0.5,1.0,1.5,2.0)')
    parser.add_argument('--top_n', type=int, default=3,
                       help='Number of top models to display (default: 3)')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Moving average window size for smoothing (default: 5, set to 1 to disable)')
    parser.add_argument('--max_epoch', type=int, default=50000,
                       help='Maximum epoch to include in analysis (default: 50000)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.folder_path):
        print(f"ERROR: Folder '{args.folder_path}' does not exist.")
        exit(1)
        
    if args.window_size < 1:
        print(f"ERROR: Window size must be at least 1.")
        exit(1)
        
    if args.alpha is not None and args.alpha <= 0:
        print(f"ERROR: Alpha must be positive.")
        exit(1)
        
    if args.max_epoch <= 0:
        print(f"ERROR: Max epoch must be positive.")
        exit(1)
    
    # Run analysis
    results = main(args.folder_path, args.alpha, args.window_size, args.max_epoch)
    
    if args.alpha is None:
        print(f"\nMulti-Alpha Analysis completed: 0.5, 1.0, 1.5, 2.0")
    else:
        print(f"\nSingle-Alpha Analysis completed: {args.alpha}")
    
    if args.window_size > 1:
        print(f"Smoothing: {args.window_size}-point moving average")
    else:
        print(f"No smoothing applied")
        
    print(f"Epoch filter: <= {args.max_epoch}")
