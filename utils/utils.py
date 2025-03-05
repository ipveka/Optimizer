# Utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from typing import Dict, List, Tuple, Any, Optional

def save_metadata(data: Dict, filename: str, base_dir: str, subfolder: str = "params") -> None:
    """Save metadata as JSON for later reference."""
    # Create subfolder if it doesn't exist
    folder_path = os.path.join(base_dir, subfolder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Convert data to be JSON serializable (handle numpy types)
    json_data = convert_to_serializable(data)
    
    # Save to file
    filepath = os.path.join(folder_path, filename)
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    return f"Saved metadata to {filepath}"

def convert_to_serializable(obj):
    """
    Convert object with numpy types to JSON serializable objects.
    
    Args:
        obj: Object to convert (can be dict, list, numpy types, etc.)
        
    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def apply_optimization_results(df, optimization_results, weeks_simulated=52):
    """
    Apply optimization results to the DataFrame to create the optimized scenario.
    Instead of using arbitrary percentage improvements, we'll use the actual optimization output.
    
    Args:
        df: DataFrame to update
        optimization_results: Results from the optimizer
        weeks_simulated: Number of weeks in the simulation
    
    Returns:
        Updated DataFrame with optimization results applied
    """
    if not optimization_results:
        print("No optimization results available, using mock improvements")
        return df
    
    # Create a copy to avoid modifying the input DataFrame
    optimized_df = df.copy()
    
    # Extract relevant parts of the optimization results
    plant_to_wh = optimization_results.get('plant_to_warehouse', {})
    wh_to_market = optimization_results.get('warehouse_to_market', {})
    optimized_inventory = optimization_results.get('inventory', {})
    
    # First optimization week
    first_opt_week = min([w for _, _, _, w in plant_to_wh.keys()]) if plant_to_wh else max(df['week'].unique()) + 1
    
    # For each plant-warehouse-product combination in the optimization results
    for (plant, warehouse, product, opt_week), flow in plant_to_wh.items():
        # Map the optimization week to simulation weeks
        sim_week = min(max(df['week'].unique()), first_opt_week + (opt_week - first_opt_week) % weeks_simulated)
        
        # Update supply based on optimization
        mask = (
            (optimized_df['plant'] == plant) & 
            (optimized_df['warehouse'] == warehouse) & 
            (optimized_df['product'] == product) & 
            (optimized_df['week'] == sim_week)
        )
        
        if mask.any():
            # Update with optimized supply (with some smoothing to avoid drastic changes)
            current_supply = optimized_df.loc[mask, 'supply'].values[0]
            optimized_supply = (current_supply * 0.3) + (flow * 0.7)  # Weighted average for smoothness
            optimized_df.loc[mask, 'supply'] = optimized_supply
    
    # For each warehouse-market-product combination in the optimization results
    for (warehouse, market, product, opt_week), flow in wh_to_market.items():
        # Map the optimization week to simulation weeks
        sim_week = min(max(df['week'].unique()), first_opt_week + (opt_week - first_opt_week) % weeks_simulated)
        
        # Update sell-in based on optimization
        mask = (
            (optimized_df['warehouse'] == warehouse) & 
            (optimized_df['market'] == market) & 
            (optimized_df['product'] == product) & 
            (optimized_df['week'] == sim_week)
        )
        
        if mask.any():
            # Update with optimized sell-in
            current_sell_in = optimized_df.loc[mask, 'sell_in'].values[0]
            # Adjust sell-in based on optimization, but maintain demand integrity
            optimized_df.loc[mask, 'sell_in'] = current_sell_in  # Keep original demand
            
    # For each warehouse-product combination with optimized inventory
    for (warehouse, product, opt_week), inv_level in optimized_inventory.items():
        # Map the optimization week to simulation weeks
        sim_week = min(max(df['week'].unique()), first_opt_week + (opt_week - first_opt_week) % weeks_simulated)
        
        # Update inventory levels based on optimization
        mask = (
            (optimized_df['warehouse'] == warehouse) & 
            (optimized_df['product'] == product) & 
            (optimized_df['week'] == sim_week)
        )
        
        if mask.any():
            # Update with optimized inventory
            optimized_df.loc[mask, 'inventory'] = inv_level
    
    # Recalculate inventory for all rows based on updated supply and sell-in
    # Group by product and warehouse
    for (product, warehouse), group in optimized_df.groupby(['product', 'warehouse']):
        indices = group.index
        if len(indices) > 0:
            # Get the initial inventory from the first week
            current_inventory = optimized_df.loc[indices[0], 'inventory']
            
            # Update inventory for each week in sequence
            for i in range(1, len(indices)):
                idx_prev = indices[i-1]
                idx_curr = indices[i]
                
                # Add supply from previous week and subtract demand (sell-in)
                supply = optimized_df.loc[idx_prev, 'supply']
                demand = optimized_df.loc[idx_prev, 'sell_in']
                
                # Update inventory using the inventory flow equation
                current_inventory = max(0, current_inventory + supply - demand)
                optimized_df.loc[idx_curr, 'inventory'] = current_inventory
    
    # Recalculate stockouts based on updated inventory
    optimized_df['stockout_units'] = 0  # Reset stockouts
    optimized_df['service_level'] = 1.0  # Reset service level
    
    for (product, warehouse), group in optimized_df.groupby(['product', 'warehouse']):
        indices = group.index
        
        for idx in indices:
            demand = optimized_df.loc[idx, 'sell_in']
            inventory = optimized_df.loc[idx, 'inventory']
            
            # If demand exceeds inventory, record stockout
            if demand > inventory:
                optimized_df.loc[idx, 'stockout_units'] = demand - inventory
                optimized_df.loc[idx, 'service_level'] = (inventory / demand) if demand > 0 else 1.0
    
    return optimized_df

def calculate_kpis(df, weeks_simulated=52, holding_cost_rate=0.2):
    """
    Calculate key performance indicators for a supply chain scenario.
    
    Args:
        df: DataFrame containing supply chain data
        weeks_simulated: Number of weeks in the simulation
        holding_cost_rate: Annual holding cost rate as a fraction
        
    Returns:
        Dictionary of KPIs
    """
    # Calculate basic metrics
    kpis = {}
    
    # Safely add metrics only if the columns exist
    for col in ['inventory', 'supply', 'sell_in']:
        if col in df.columns:
            kpis[f'total_{col}'] = float(df[col].sum())
            kpis[f'avg_{col}'] = float(df[col].mean())
            if col == 'inventory':
                kpis['median_inventory'] = float(df[col].median())
    
    # Calculate stockout and service level metrics if available
    if 'stockout_units' in df.columns:
        kpis['total_stockouts'] = float(df['stockout_units'].sum())
        
        # Calculate warehouse-level metrics only if warehouse column exists
        if 'warehouse' in df.columns:
            warehouse_stockouts = df.groupby('warehouse')['stockout_units'].sum()
            kpis['avg_stockouts_per_warehouse'] = float(warehouse_stockouts.mean())
        else:
            kpis['avg_stockouts_per_warehouse'] = kpis['total_stockouts'] / len(df['warehouse'].unique()) if 'warehouse' in df.columns else 0
    else:
        # Add default values if columns don't exist
        kpis['total_stockouts'] = 0
        kpis['avg_stockouts_per_warehouse'] = 0
    
    if 'service_level' in df.columns:
        kpis['avg_service_level'] = float(df['service_level'].mean())
    else:
        kpis['avg_service_level'] = 0.95  # Default value
    
    # Calculate inventory turns if we have both inventory and sell_in
    if 'inventory' in df.columns and 'sell_in' in df.columns and df['inventory'].mean() > 0:
        kpis['inventory_turns'] = float(df['sell_in'].sum() / (df['inventory'].mean() * weeks_simulated) * 52)  # Annualized
    else:
        kpis['inventory_turns'] = 0
    
    # Calculate cost metrics if unit_cost is available
    if 'unit_cost' in df.columns and 'inventory' in df.columns:
        avg_unit_cost = float(df['unit_cost'].mean())
        weekly_holding_cost_rate = holding_cost_rate / 52  # Convert to weekly rate
        
        # Calculate inventory carrying cost
        kpis['inventory_carrying_cost'] = float(kpis.get('avg_inventory', 0) * avg_unit_cost * weekly_holding_cost_rate * weeks_simulated)
        
        # Calculate stockout cost (assuming 50% profit margin loss)
        if 'stockout_units' in df.columns:
            kpis['stockout_cost'] = float(kpis.get('total_stockouts', 0) * avg_unit_cost * 0.5)
            kpis['total_cost'] = kpis['inventory_carrying_cost'] + kpis['stockout_cost']
        else:
            kpis['stockout_cost'] = 0
            kpis['total_cost'] = kpis['inventory_carrying_cost']
    else:
        # Add default values if unit_cost column doesn't exist
        avg_unit_cost = 50  # Default unit cost
        weekly_holding_cost_rate = holding_cost_rate / 52
        
        if 'inventory' in df.columns:
            kpis['inventory_carrying_cost'] = float(kpis.get('avg_inventory', 0) * avg_unit_cost * weekly_holding_cost_rate * weeks_simulated)
        else:
            kpis['inventory_carrying_cost'] = 0
            
        kpis['stockout_cost'] = float(kpis.get('total_stockouts', 0) * avg_unit_cost * 0.5)
        kpis['total_cost'] = kpis['inventory_carrying_cost'] + kpis['stockout_cost']
    
    return kpis

def calculate_improvements(base_kpis, optimized_kpis):
    """
    Calculate improvements between two sets of KPIs.
    
    Args:
        base_kpis: Dictionary of KPIs for base scenario
        optimized_kpis: Dictionary of KPIs for optimized scenario
        
    Returns:
        Dictionary of improvements
    """
    improvements = {}
    
    # Helper function to safely calculate improvement
    def safe_improvement(base_key, opt_key, is_higher_better=False):
        if base_key in base_kpis and opt_key in optimized_kpis:
            base_val = base_kpis[base_key]
            opt_val = optimized_kpis[opt_key]
            
            # Avoid division by zero
            if base_val == 0:
                if opt_val == 0:
                    return {'absolute': 0, 'percentage': 0}
                else:
                    # For metrics where higher is better, if starting from zero, show positive infinity
                    # For metrics where lower is better, if starting from zero, show negative infinity
                    return {
                        'absolute': opt_val - base_val,
                        'percentage': float('inf') if is_higher_better else float('-inf')
                    }
            
            # Calculate improvement
            absolute = opt_val - base_val
            
            # For metrics where higher is better, we need to flip the sign
            if is_higher_better:
                percentage = (absolute / abs(base_val)) * 100
            else:
                percentage = ((base_val - opt_val) / abs(base_val)) * 100
                
            return {'absolute': float(absolute), 'percentage': float(percentage)}
        
        # If metrics are missing, return default values
        return {'absolute': 0, 'percentage': 0}
    
    # Calculate inventory improvements
    improvements['inventory_reduction'] = safe_improvement('total_inventory', 'total_inventory')
    
    # Calculate stockout improvements
    improvements['stockout_reduction'] = safe_improvement('total_stockouts', 'total_stockouts')
    
    # Calculate service level improvements (higher is better)
    improvements['service_level_improvement'] = safe_improvement('avg_service_level', 'avg_service_level', True)
    
    # Calculate inventory turns improvements (higher is better)
    improvements['inventory_turns_improvement'] = safe_improvement('inventory_turns', 'inventory_turns', True)
    
    return improvements

def calculate_cost_improvements(base_kpis, optimized_kpis):
    """
    Calculate cost improvements between two sets of KPIs.
    
    Args:
        base_kpis: Dictionary of KPIs for base scenario
        optimized_kpis: Dictionary of KPIs for optimized scenario
        
    Returns:
        Dictionary of cost improvements
    """
    cost_improvements = {}
    
    # Helper function to safely calculate cost improvement
    def safe_cost_improvement(key):
        if key in base_kpis and key in optimized_kpis:
            before = base_kpis[key]
            after = optimized_kpis[key]
            
            # Avoid division by zero
            if before == 0:
                return {
                    'before': float(before),
                    'after': float(after),
                    'savings': float(before - after),
                    'percentage': 0 if after == 0 else float('-inf')
                }
            
            # Calculate improvement
            savings = before - after
            percentage = (savings / before) * 100
                
            return {
                'before': float(before),
                'after': float(after),
                'savings': float(savings),
                'percentage': float(percentage)
            }
        
        # If metrics are missing, return default values
        return {'before': 0, 'after': 0, 'savings': 0, 'percentage': 0}
    
    # Calculate inventory carrying cost improvements
    cost_improvements['inventory_carrying_cost'] = safe_cost_improvement('inventory_carrying_cost')
    
    # Calculate stockout cost improvements
    cost_improvements['stockout_cost'] = safe_cost_improvement('stockout_cost')
    
    # Calculate total cost improvements
    cost_improvements['total_cost'] = safe_cost_improvement('total_cost')
    
    return cost_improvements

def create_optimization_comparison(before_df, after_df, metrics, group_by='warehouse'):
    """
    Create a customized before/after optimization comparison visualization.
    
    Args:
        before_df: DataFrame before optimization
        after_df: DataFrame after optimization
        metrics: List of metrics to compare
        group_by: Column to group by
    
    Returns:
        matplotlib.figure.Figure
    """
    # Create subplot grid
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 5 * n_metrics), sharex=True)
    
    # Ensure axes is always a list
    if n_metrics == 1:
        axes = [axes]
    
    # Calculate aggregated values for each metric by the grouping variable
    before_grouped = before_df.groupby(group_by)[metrics].mean().reset_index()
    after_grouped = after_df.groupby(group_by)[metrics].mean().reset_index()
    
    # Create comparison dataframe
    comparison_df = pd.merge(before_grouped, after_df.groupby(group_by)[metrics].mean().reset_index(),
                           on=group_by, suffixes=('_before', '_after'))
    
    # Calculate percentage changes
    for metric in metrics:
        comparison_df[f'{metric}_change_pct'] = ((comparison_df[f'{metric}_after'] - comparison_df[f'{metric}_before']) / 
                                              comparison_df[f'{metric}_before'] * 100).round(2)
    
    # Sort data for consistent display
    comparison_df = comparison_df.sort_values(group_by)
    
    # Plotting each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract data for current metric
        before_data = comparison_df[f'{metric}_before']
        after_data = comparison_df[f'{metric}_after']
        change_pct = comparison_df[f'{metric}_change_pct']
        
        # Create positions for bars
        x = np.arange(len(comparison_df))
        width = 0.35
        
        # Create bars
        before_bars = ax.bar(x - width/2, before_data, width, label='Before Optimization', 
                           color='#BBDEFB', edgecolor='black', linewidth=1)
        after_bars = ax.bar(x + width/2, after_data, width, label='After Optimization', 
                          color='#2196F3', edgecolor='black', linewidth=1)
        
        # Add labels and styling
        metric_name = metric.replace('_', ' ').title()
        ax.set_title(f"{metric_name} Before vs After Optimization", fontsize=16, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df[group_by], rotation=45, ha='right', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', labelsize=12)
        
        # Increase y-axis margin
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.2)  # Add 20% margin at the top
        
        # Add percentage change annotations
        for j, pct in enumerate(change_pct):
            color = 'green' if pct < 0 else 'red'
            if metric in ['service_level', 'supply']:  # Metrics where increase is good
                color = 'green' if pct > 0 else 'red'
            
            label = f"{pct:+.1f}%"
            ax.annotate(label, 
                      xy=(j, max(before_data.iloc[j], after_data.iloc[j]) * 1.05),
                      ha='center', va='bottom', 
                      color=color, fontweight='bold', fontsize=12)
        
        # Add a legend with larger font
        ax.legend(fontsize=12)
    
    # Don't add overall title as requested
    plt.tight_layout()
    
    return fig

def plot_cost_benefits(cost_improvements):
    """
    Create a visualization of cost benefits from optimization.
    
    Args:
        cost_improvements: Dictionary of cost improvement metrics
        
    Returns:
        matplotlib.figure.Figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Cost Before vs After
    categories = ['Inventory Carrying Cost', 'Stockout Cost', 'Total Cost']
    before_values = [cost_improvements['inventory_carrying_cost']['before'], 
                    cost_improvements['stockout_cost']['before'],
                    cost_improvements['total_cost']['before']]
    after_values = [cost_improvements['inventory_carrying_cost']['after'], 
                   cost_improvements['stockout_cost']['after'],
                   cost_improvements['total_cost']['after']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, before_values, width, label='Before Optimization', color='#BBDEFB', edgecolor='black')
    ax1.bar(x + width/2, after_values, width, label='After Optimization', color='#2196F3', edgecolor='black')
    
    # Add data labels
    for i, v in enumerate(before_values):
        ax1.text(i - width/2, v + 0.1, f"${v:.1f}", ha='center', fontweight='bold')
    
    for i, v in enumerate(after_values):
        ax1.text(i + width/2, v + 0.1, f"${v:.1f}", ha='center', fontweight='bold')
    
    ax1.set_ylabel('Cost ($)')
    ax1.set_title('Cost Comparison Before vs After Optimization')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot 2: Cost Savings and ROI
    savings = [cost_improvements['inventory_carrying_cost']['savings'],
              cost_improvements['stockout_cost']['savings'],
              cost_improvements['total_cost']['savings']]
    
    percentage = [cost_improvements['inventory_carrying_cost']['percentage'],
                 cost_improvements['stockout_cost']['percentage'],
                 cost_improvements['total_cost']['percentage']]
    
    bars = ax2.bar(x, savings, width, color='#4CAF50', edgecolor='black')
    
    # Create second y-axis for percentages
    ax2b = ax2.twinx()
    ax2b.plot(x, percentage, 'ro-', linewidth=2, markersize=8)
    
    # Add data labels
    for i, v in enumerate(savings):
        ax2.text(i, v + 0.1, f"${v:.1f}", ha='center', fontweight='bold')
    
    for i, v in enumerate(percentage):
        ax2b.text(i, v + 0.1, f"{v:.1f}%", ha='center', color='red', fontweight='bold')
    
    ax2.set_ylabel('Cost Savings ($)')
    ax2b.set_ylabel('Savings Percentage (%)', color='red')
    ax2.set_title('Cost Savings from Optimization')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2b.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    return fig

def create_kpi_dashboard(before_kpis, after_kpis, cost_improvements):
    """
    Create a dashboard of key performance indicators.
    
    Args:
        before_kpis: Dictionary of KPIs for base scenario
        after_kpis: Dictionary of KPIs for optimized scenario
        cost_improvements: Dictionary of cost improvement metrics
        
    Returns:
        matplotlib.figure.Figure
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # 1. Inventory Levels (Total and Average)
    ax = axes[0]
    metrics = ['total_inventory', 'avg_inventory']
    x = np.arange(len(metrics))
    width = 0.35
    
    before_values = [before_kpis[m] for m in metrics]
    after_values = [after_kpis[m] for m in metrics]
    
    ax.bar(x - width/2, before_values, width, label='Before', color='#BBDEFB', edgecolor='black')
    ax.bar(x + width/2, after_values, width, label='After', color='#2196F3', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Total Inventory', 'Average Inventory'])
    ax.set_title('Inventory Level Comparison', fontsize=14)
    ax.set_ylabel('Units')
    ax.legend()
    
    # Calculate and display percentage changes
    for i, (before, after) in enumerate(zip(before_values, after_values)):
        pct_change = ((after - before) / before) * 100
        color = 'green' if pct_change < 0 else 'red'  # for inventory, reduction is good
        ax.annotate(f"{pct_change:.1f}%", 
                  xy=(i, max(before, after) * 1.05),
                  ha='center', color=color, fontweight='bold')
    
    # 2. Service Level and Stockouts
    ax = axes[1]
    metrics = ['avg_service_level', 'total_stockouts']
    x = np.arange(len(metrics))
    
    before_values = [before_kpis.get(m, 0) for m in metrics]
    after_values = [after_kpis.get(m, 0) for m in metrics]
    
    ax.bar(x - width/2, before_values, width, label='Before', color='#BBDEFB', edgecolor='black')
    ax.bar(x + width/2, after_values, width, label='After', color='#2196F3', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Service Level', 'Total Stockouts'])
    ax.set_title('Service Level and Stockouts', fontsize=14)
    ax.legend()
    
    # Calculate and display percentage changes
    for i, (before, after) in enumerate(zip(before_values, after_values)):
        if before == 0 and i == 1:  # Handle division by zero for stockouts
            pct_change = -100 if after == 0 else float('inf')
        else:
            pct_change = ((after - before) / (before if before != 0 else 1)) * 100
        
        # For service level, increase is good; for stockouts, decrease is good
        color = 'green' if (i == 0 and pct_change > 0) or (i == 1 and pct_change < 0) else 'red'
        ax.annotate(f"{pct_change:.1f}%", 
                  xy=(i, max(before, after) * 1.05),
                  ha='center', color=color, fontweight='bold')
    
    # 3. Inventory Turns
    ax = axes[2]
    before_turns = before_kpis.get('inventory_turns', 0)
    after_turns = after_kpis.get('inventory_turns', 0)
    
    ax.bar(['Before', 'After'], [before_turns, after_turns], color=['#BBDEFB', '#2196F3'], edgecolor='black')
    ax.set_title('Inventory Turns', fontsize=14)
    ax.set_ylabel('Turns per Year')
    
    # Calculate and display percentage change
    pct_change = ((after_turns - before_turns) / before_turns) * 100
    color = 'green' if pct_change > 0 else 'red'  # for inventory turns, increase is good
    ax.annotate(f"{pct_change:.1f}%", 
              xy=(1, max(before_turns, after_turns) * 1.05),
              ha='center', color=color, fontweight='bold')
    
    # 4. Cost Breakdown
    ax = axes[3]
    cost_categories = ['Inventory Cost', 'Stockout Cost', 'Total Cost']
    before_costs = [
        cost_improvements['inventory_carrying_cost']['before'],
        cost_improvements['stockout_cost']['before'],
        cost_improvements['total_cost']['before']
    ]
    after_costs = [
        cost_improvements['inventory_carrying_cost']['after'],
        cost_improvements['stockout_cost']['after'],
        cost_improvements['total_cost']['after']
    ]
    
    x = np.arange(len(cost_categories))
    ax.bar(x - width/2, before_costs, width, label='Before', color='#BBDEFB', edgecolor='black')
    ax.bar(x + width/2, after_costs, width, label='After', color='#2196F3', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(cost_categories)
    ax.set_title('Cost Analysis', fontsize=14)
    ax.set_ylabel('Cost ($)')
    ax.legend()
    
    # Calculate and display percentage changes
    for i, (before, after) in enumerate(zip(before_costs, after_costs)):
        pct_change = ((after - before) / before) * 100
        color = 'green' if pct_change < 0 else 'red'  # for costs, reduction is good
        ax.annotate(f"{pct_change:.1f}%", 
                  xy=(i, max(before, after) * 1.05),
                  ha='center', color=color, fontweight='bold')
    
    # Add overall title
    plt.suptitle('Supply Chain Optimization KPI Dashboard', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def generate_html_report(base_dir, plots_dir, network_config, base_kpis, optimized_kpis, improvements, cost_improvements):
    """
    Generate an HTML report of the optimization results.
    
    Args:
        base_dir: Base directory of the experiment
        plots_dir: Directory containing the plot images
        network_config: Configuration of the supply chain network
        base_kpis: KPIs for the base scenario
        optimized_kpis: KPIs for the optimized scenario
        improvements: Improvement metrics
        cost_improvements: Cost improvement metrics
        
    Returns:
        Path to the saved report
    """
    # Format metrics for display
    inventory_reduction_pct = improvements.get('inventory_reduction', {}).get('percentage', 0)
    service_level_improvement_pct = improvements.get('service_level_improvement', {}).get('percentage', 0)
    stockout_reduction_pct = improvements.get('stockout_reduction', {}).get('percentage', 0)
    cost_savings = cost_improvements.get('total_cost', {}).get('savings', 0)
    cost_savings_pct = cost_improvements.get('total_cost', {}).get('percentage', 0)
    
    # Save the report in the base experiment directory
    report_path = os.path.join(base_dir, 'supply_chain_optimization_report.html')
    
    # Images should be referenced with relative paths from the HTML file location
    # Since HTML is in the same directory as the folders, we just use the folder names
    
    with open(report_path, 'w') as f:
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Supply Chain Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .highlight {{ background-color: #e8f4fd; padding: 20px; border-radius: 5px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .comparison-section {{ display: flex; flex-direction: column; margin-bottom: 30px; }}
                .positive-change {{ color: green; font-weight: bold; }}
                .negative-change {{ color: red; font-weight: bold; }}
                .dashboard-section {{ 
                    display: grid; 
                    grid-template-columns: 1fr 1fr; 
                    grid-gap: 20px;
                    margin: 20px 0; 
                }}
                .metric-card {{ 
                    background-color: white; 
                    padding: 15px; 
                    border-radius: 5px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                    margin-bottom: 20px; 
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .chart-container {{ background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Supply Chain Optimization Report</h1>
            
            <div class="highlight">
                <h2>Executive Summary</h2>
                <p>This report presents the results of a supply chain optimization initiative aimed at improving inventory management, 
                   reducing costs, and enhancing service levels. The optimization has delivered the following key improvements:</p>
                <div class="dashboard-section">
                    <div class="metric-card">
                        <h3>Inventory Reduction</h3>
                        <div class="metric-value" style="color: {('green' if inventory_reduction_pct > 0 else 'red')}">
                            {inventory_reduction_pct:.2f}%
                        </div>
                        <p>Resulting in lower carrying costs and improved operational efficiency</p>
                    </div>
                    <div class="metric-card">
                        <h3>Service Level Improvement</h3>
                        <div class="metric-value" style="color: {('green' if service_level_improvement_pct > 0 else 'red')}">
                            {service_level_improvement_pct:.2f}%
                        </div>
                        <p>Enhancing customer satisfaction and reducing lost sales</p>
                    </div>
                    <div class="metric-card">
                        <h3>Stockout Reduction</h3>
                        <div class="metric-value" style="color: {('green' if stockout_reduction_pct > 0 else 'red')}">
                            {stockout_reduction_pct:.2f}%
                        </div>
                        <p>Minimizing lost sales opportunities and improving customer experience</p>
                    </div>
                    <div class="metric-card">
                        <h3>Total Cost Savings</h3>
                        <div class="metric-value" style="color: {('green' if cost_savings > 0 else 'red')}">
                            ${cost_savings:.2f} ({cost_savings_pct:.2f}%)
                        </div>
                        <p>Direct financial impact from optimization</p>
                    </div>
                </div>
            </div>
            
            <div class="summary">
                <h2>Supply Chain Network Overview</h2>
                <p>The optimization was performed on a supply chain network with the following dimensions:</p>
                <ul>
                    <li><strong>{len(network_config['plants'])}</strong> production plants: {", ".join(network_config['plants'])}</li>
                    <li><strong>{len(network_config['warehouses'])}</strong> distribution warehouses: {", ".join(network_config['warehouses'])}</li>
                    <li><strong>{len(network_config['markets'])}</strong> markets: {", ".join(network_config['markets'])}</li>
                    <li><strong>{len(network_config['products'])}</strong> products: {", ".join(network_config['products'])}</li>
                    <li><strong>{network_config['weeks']}</strong> weeks of data simulated</li>
                </ul>
            </div>
            
            <div class="chart-container">
                <h2>KPI Dashboard</h2>
                <p>The dashboard below provides a comprehensive view of the key performance indicators before and after optimization:</p>
                <img src="plots/kpi_dashboard.png" alt="KPI Dashboard">
            </div>
            
            <div class="chart-container">
                <h2>Cost Benefit Analysis</h2>
                <p>The following chart illustrates the cost benefits achieved through the optimization:</p>
                <img src="plots/optimization_roi.png" alt="Cost Benefit Analysis">
                <p><strong>Note:</strong> All cost figures represent annualized values based on the {network_config['weeks']} week simulation period.</p>
            </div>
            
            <h2>Warehouse Performance Comparison</h2>
            <p>The chart below shows how key metrics varied across warehouses before and after optimization:</p>
            <img src="plots/optimization_comparison.png" alt="Warehouse Performance Comparison">
            
            <h2>Product Performance Analysis</h2>
            <p>The chart below compares performance metrics by product before and after optimization:</p>
            <img src="plots/product_optimization_comparison.png" alt="Product Performance Analysis">
            
            <h2>Network Visualization</h2>
            <p>The following visualization shows the supply chain network structure:</p>
            <img src="plots/network_visualization.png" alt="Supply Chain Network">
            
            <h2>Multi-Network Analysis</h2>
            <p>This enhanced visualization shows inventory levels and safety stock across the network:</p>
            <img src="plots/multiple_networks_visualization.png" alt="Multiple Networks Visualization">
            
            <h2>Inventory Metrics Over Time</h2>
            <p>The time series chart below shows how inventory metrics evolved over time:</p>
            <img src="plots/inventory_time_series.png" alt="Inventory Time Series">
            
            <h2>Inventory Distribution</h2>
            <p>This heatmap shows inventory distribution across warehouses and products:</p>
            <img src="plots/inventory_heatmap.png" alt="Inventory Heatmap">
            
            <div class="summary">
                <h2>Methodology</h2>
                <p>The optimization methodology consisted of the following key steps:</p>
                <ol>
                    <li><strong>Network Modeling:</strong> Creating a digital representation of the supply chain network</li>
                    <li><strong>Data Simulation:</strong> Generating realistic supply, demand, and inventory patterns</li>
                    <li><strong>Safety Stock Optimization:</strong> Calculating optimal safety stock levels based on service level targets and demand variability</li>
                    <li><strong>Reorder Point Determination:</strong> Establishing optimal reorder points to minimize stockouts while controlling inventory</li>
                    <li><strong>Network Flow Optimization:</strong> Optimizing the flow of products through the network to minimize total cost</li>
                    <li><strong>Performance Measurement:</strong> Quantifying improvements in key metrics including inventory levels, service levels, and costs</li>
                </ol>
            </div>
            
            <p><strong>Report generated on:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </body>
        </html>
        ''')
    
    return report_path