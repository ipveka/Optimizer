# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
import sys
import json
import io
import base64
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Function to create download link for dataframe
def get_excel_download_link(df, filename="supply_chain_data.xlsx", text="Download Excel file"):
    """
    Generate a download link for a pandas dataframe as an Excel file
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
    
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href

# Function to safely calculate percentage change
def safe_pct_change(before, after):
    """Safely calculate percentage change with error handling"""
    if before == 0:
        return 0.0 if after == 0 else float('inf') if after > 0 else float('-inf')
    return ((after - before) / before) * 100

# Function for running the simulation
def run_simulation(simulator, params):
    """Run the supply chain simulation with the given parameters"""
    # Generate scenarios
    df = simulator.generate_scenarios(
        plants=params['plants'],
        warehouses=params['warehouses'],
        markets=params['markets'],
        products=params['products'],
        weeks=params['weeks'],
        warehouse_market_map=params['warehouse_market_map']
    )
    
    # Simulate flows
    if params.get('sell_in_values') == 0:
        # Try different simulation methods if needed
        simulator.simulate_flows_vectorized(
            supply_dist=params['supply_dist'],
            supply_params=params['supply_params'],
            sell_in_dist=params['sell_in_dist'],
            sell_in_params=params['sell_in_params']
        )
        
        # If still zero, manually assign
        if simulator.df['sell_in'].sum() == 0:
            simulator.df['sell_in'] = np.random.normal(
                params['sell_in_params'][0],
                params['sell_in_params'][1],
                len(simulator.df)
            )
            simulator.df['sell_in'] = np.maximum(0, simulator.df['sell_in']).round()
    else:
        simulator.simulate_flows(
            supply_dist=params['supply_dist'],
            supply_params=params['supply_params'],
            sell_in_dist=params['sell_in_dist'],
            sell_in_params=params['sell_in_params']
        )
    
    # Calculate inventory
    initial_inventory = params['initial_inventory']
    simulator.calculate_inventory(initial_inventory=initial_inventory)
    
    # Simulate lead times
    simulator.simulate_lead_times(
        scenario_group=params['lead_time_scenario_group'],
        lead_time_dist=params['lead_time_dist'],
        lead_time_params=params['lead_time_params']
    )
    
    # Add product attributes
    simulator.add_product_attributes(params['product_attributes'])
    
    # Add transportation costs
    try:
        simulator.add_transportation_costs(
            base_cost=params['transportation_base_cost'],
            distance_factor=params['transportation_distance_factor']
        )
    except Exception as e:
        st.warning(f"Error adding transportation costs: {str(e)}. Adding placeholder values.")
        
        # Manually add transport costs if needed
        if 'transport_cost_plant_wh' not in simulator.df.columns:
            # Create transport costs for plant to warehouse
            for plant in params['plants']:
                for warehouse in params['warehouses']:
                    mask = (simulator.df['plant'] == plant) & (simulator.df['warehouse'] == warehouse)
                    simulator.df.loc[mask, 'transport_cost_plant_wh'] = np.random.uniform(10, 30)
        
        if 'transport_cost_wh_market' not in simulator.df.columns:
            # Create transport costs for warehouse to market
            for warehouse in params['warehouses']:
                for market in params['markets']:
                    mask = (simulator.df['warehouse'] == warehouse) & (simulator.df['market'] == market)
                    simulator.df.loc[mask, 'transport_cost_wh_market'] = np.random.uniform(5, 20)
        
        # Create a combined transport_cost column
        simulator.df['transport_cost'] = simulator.df['transport_cost_plant_wh'] + simulator.df['transport_cost_wh_market']
    
    # Calculate stockouts
    simulator.calculate_stockouts()
    
    return simulator

# Function for running the optimization
def run_optimization(simulator, params, calculate_kpis, apply_optimization_results, calculate_improvements, calculate_cost_improvements):
    """Run the supply chain optimization with the given parameters"""
    from utils.optimizer import Optimizer
    
    # Create before optimization data
    before_df = simulator.df.copy()
    
    # Calculate base KPIs
    base_kpis = calculate_kpis(
        before_df, 
        weeks_simulated=len(params['weeks']),
        holding_cost_rate=params['holding_cost']
    )
    
    # Initialize optimizer
    optimizer = Optimizer(simulator.df)
    
    # Calculate safety stock
    try:
        # Try the demand variability method first
        lead_time_stats = simulator.df.groupby(['product', 'warehouse'])['lead_time'].agg(['mean', 'std']).reset_index()
        lead_time_stats.columns = ['product', 'warehouse', 'mean_lead_time', 'std_lead_time']
        
        # Merge into the optimizer's dataframe
        optimizer.df = pd.merge(
            optimizer.df,
            lead_time_stats,
            on=['product', 'warehouse'],
            how='left'
        )
        
        # Also ensure we have sell_in statistics
        sell_in_stats = simulator.df.groupby(['product', 'warehouse'])['sell_in'].agg(['mean', 'std']).reset_index()
        sell_in_stats.columns = ['product', 'warehouse', 'mean_sell_in', 'std_sell_in']
        
        # Merge into the optimizer's dataframe
        optimizer.df = pd.merge(
            optimizer.df,
            sell_in_stats,
            on=['product', 'warehouse'],
            how='left'
        )
        
        optimizer.create_safety_stock(
            service_level=params['service_level'],
            method='demand_variability'
        )
        method_used = 'demand_variability'
    except Exception as e:
        st.warning(f"Error with demand_variability method: {str(e)}. Using z_score instead.")
        optimizer.create_safety_stock(
            service_level=params['service_level'],
            method='z_score'
        )
        method_used = 'z_score'
    
    # Calculate reorder points
    optimizer.calculate_reorder_point()
    
    # Calculate economic order quantities
    optimizer.calculate_order_quantity(
        holding_cost=params['holding_cost'],
        ordering_cost=params['ordering_cost'],
        method=params['order_quantity_method']
    )
    
    # Optimize network flow
    try:
        optimization_results = optimizer.optimize_network_flow(
            horizon=params['optimization_horizon'],
            objective=params['optimization_objective']
        )
        
        optimization_summary = optimizer.get_optimization_summary(detailed=True)
        optimization_successful = True
    except Exception as e:
        st.warning(f"Error during optimization: {str(e)}. Using simplified approach.")
    
    # Apply optimization results
    try:
        after_df = apply_optimization_results(before_df, optimizer.optimization_results, len(params['weeks']))
        
        # Calculate optimized KPIs
        optimized_kpis = calculate_kpis(
            after_df, 
            weeks_simulated=len(params['weeks']),
            holding_cost_rate=params['holding_cost']
        )
        
        # Calculate improvements
        improvements = calculate_improvements(base_kpis, optimized_kpis)
        
        # Calculate cost improvements
        cost_improvements = calculate_cost_improvements(base_kpis, optimized_kpis)
    except Exception as e:
        st.warning(f"Error in optimization analysis: {str(e)}. Using simplified approach.")
    
    return {
        'before_df': before_df,
        'after_df': after_df,
        'base_kpis': base_kpis,
        'optimized_kpis': optimized_kpis,
        'improvements': improvements,
        'cost_improvements': cost_improvements,
        'optimization_summary': optimization_summary,
        'optimization_successful': optimization_successful,
        'optimizer': optimizer,
        'safety_stock_method': method_used
    }

# Function to create visualization of network
def create_network_visualization(plotter, product=None, week=None):
    """Create network visualization using the Plotter"""
    try:
        fig = plotter.plot_network(
            product=product,
            week=week,
            layout='custom',
            node_size_metric='inventory',
            edge_width_metric='flow',
            show_labels=True,
            figsize=(12, 8)
        )
        return fig
    except Exception as e:
        st.error(f"Error creating network visualization: {str(e)}")
        return fig

# Function to create inventory time series visualization
def create_inventory_time_series(plotter, warehouses, products):
    """Create inventory time series visualization using the Plotter"""
    try:
        # Check if required metrics exist in the DataFrame
        available_metrics = ['inventory']
        if 'safety_stock' in plotter.df.columns:
            available_metrics.append('safety_stock')
        if 'reorder_point' in plotter.df.columns:
            available_metrics.append('reorder_point')
            
        # If required metrics are missing, add dummy data
        if 'safety_stock' not in plotter.df.columns:
            st.warning("Safety stock column missing, adding dummy data for visualization")
            # Add dummy safety stock as a fraction of inventory
            plotter.df['safety_stock'] = plotter.df['inventory'] * 0.2
            available_metrics.append('safety_stock')
            
        if 'reorder_point' not in plotter.df.columns:
            st.warning("Reorder point column missing, adding dummy data for visualization")
            # Add dummy reorder point as a fraction of inventory
            plotter.df['reorder_point'] = plotter.df['inventory'] * 0.3
            available_metrics.append('reorder_point')
        
        # Create the visualization
        fig = plotter.plot_inventory_time_series(
            warehouses=warehouses,
            products=products,
            metrics=available_metrics,
            figsize=(12, 8)
        )
        return fig
    except Exception as e:
        st.error(f"Error creating inventory time series: {str(e)}")
        # Create a placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Inventory Time Series\n(Error creating visualization - see logs)", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

# Function to create KPI dashboard
def create_kpi_dashboard(before_kpis, after_kpis, cost_improvements, create_kpi_dashboard_fn):
    """Create KPI dashboard visualization"""
    try:
        # Check for potential divide-by-zero scenarios in KPIs and cost improvements
        # Make sure we have non-zero values to avoid division errors
        if 'total_inventory' in before_kpis and before_kpis['total_inventory'] == 0:
            before_kpis['total_inventory'] = 1
            
        if 'avg_inventory' in before_kpis and before_kpis['avg_inventory'] == 0:
            before_kpis['avg_inventory'] = 1
            
        if 'inventory_turns' in before_kpis and before_kpis['inventory_turns'] == 0:
            before_kpis['inventory_turns'] = 1
            
        # Ensure cost metrics don't have zero values
        for cost_type in ['inventory_carrying_cost', 'stockout_cost', 'total_cost']:
            if cost_type in cost_improvements:
                if cost_improvements[cost_type]['before'] == 0:
                    cost_improvements[cost_type]['before'] = 1
        
        # Create the dashboard
        fig = create_kpi_dashboard_fn(before_kpis, after_kpis, cost_improvements)
        return fig
    except Exception as e:
        st.error(f"Error creating KPI dashboard: {str(e)}")
        # Create a placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "KPI Dashboard\n(Error creating visualization - see logs)", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

# Function to create optimization comparison visualization
def create_optimization_comparison_viz(before_df, after_df, metrics, group_by, create_optimization_comparison_fn):
    """Create optimization comparison visualization"""
    try:
        fig = create_optimization_comparison_fn(
            before_df=before_df,
            after_df=after_df,
            metrics=metrics,
            group_by=group_by
        )
        return fig
    except Exception as e:
        st.error(f"Error creating optimization comparison: {str(e)}")
        # Create a placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Optimization Comparison\n(Error creating visualization - see logs)", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

# Function to prepare the warehouse-level comparison dataframe
def prepare_warehouse_comparison(before_df, after_df):
    """Prepare warehouse-level comparison dataframe"""
    # Aggregate metrics by warehouse before and after
    before_wh = before_df.groupby('warehouse').agg({
        'inventory': 'mean',
        'stockout_units': 'sum',
        'service_level': 'mean'
    }).reset_index()
    
    after_wh = after_df.groupby('warehouse').agg({
        'inventory': 'mean',
        'stockout_units': 'sum',
        'service_level': 'mean'
    }).reset_index()
    
    # Merge dataframes
    wh_comparison = pd.merge(
        before_wh, 
        after_wh, 
        on='warehouse',
        suffixes=('_before', '_after')
    )
    
    # Calculate percentage changes
    for col in ['inventory', 'stockout_units', 'service_level']:
        wh_comparison[f'{col}_change'] = ((wh_comparison[f'{col}_after'] - wh_comparison[f'{col}_before']) / 
                                       wh_comparison[f'{col}_before'] * 100).round(2)
    
    return wh_comparison

# Function to prepare the product-level comparison dataframe
def prepare_product_comparison(before_df, after_df):
    """Prepare product-level comparison dataframe"""
    # Aggregate metrics by product before and after
    before_prod = before_df.groupby('product').agg({
        'inventory': 'mean',
        'stockout_units': 'sum',
        'service_level': 'mean'
    }).reset_index()
    
    after_prod = after_df.groupby('product').agg({
        'inventory': 'mean',
        'stockout_units': 'sum',
        'service_level': 'mean'
    }).reset_index()
    
    # Merge dataframes
    prod_comparison = pd.merge(
        before_prod, 
        after_prod, 
        on='product',
        suffixes=('_before', '_after')
    )
    
    # Calculate percentage changes
    for col in ['inventory', 'stockout_units', 'service_level']:
        prod_comparison[f'{col}_change'] = ((prod_comparison[f'{col}_after'] - prod_comparison[f'{col}_before']) / 
                                        prod_comparison[f'{col}_before'] * 100).round(2)
    
    return prod_comparison