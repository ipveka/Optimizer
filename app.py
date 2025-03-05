# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import io
import base64
from datetime import datetime
import traceback

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import assets
from utils.simulator import Simulator
from utils.optimizer import Optimizer
from utils.plotter import Plotter
from utils.utils import *
from utils.app_utils import *

# Set page config
st.set_page_config(
    page_title="Supply Chain Optimizer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #4285F4;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #4285F4;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .download-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        margin: 10px 0;
    }
    .download-button:hover {
        background-color: #45a049;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .positive {
        color: green;
    }
    .negative {
        color: red;
    }
</style>
""", unsafe_allow_html=True)

# Main App Logic
def main():
    # Main title
    st.markdown('<div class="main-header">Supply Chain Optimization</div>', unsafe_allow_html=True)
    st.markdown("""
    This app simulates a supply chain network and optimizes inventory, safety stock, 
    and network flow. Adjust parameters in the sidebar to configure your simulation.
    """)
    
    # Initialize session state for storing results
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False
    if 'optimization_run' not in st.session_state:
        st.session_state.optimization_run = False
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    
    # Sidebar Configuration
    st.sidebar.markdown('<div class="subsection-header">Network Configuration</div>', unsafe_allow_html=True)
    
    # Network dimensions
    with st.sidebar.expander("Network Dimensions", expanded=True):
        plants = st.text_input("Plants (comma-separated)", "PlantCA, PlantTX, PlantGA").split(',')
        plants = [p.strip() for p in plants]
        
        warehouses = st.text_input("Warehouses (comma-separated)", 
                                  "WH_North, WH_South, WH_East, WH_West, WH_Central").split(',')
        warehouses = [w.strip() for w in warehouses]
        
        markets = st.text_input("Markets (comma-separated)", 
                               "Market_NE, Market_SE, Market_MW, Market_SW, Market_NW, Market_Central").split(',')
        markets = [m.strip() for m in markets]
        
        products = st.text_input("Products (comma-separated)", "ProductA, ProductB, ProductC").split(',')
        products = [p.strip() for p in products]
        
        num_weeks = st.slider("Number of Weeks", min_value=1, max_value=104, value=52)
        weeks = list(range(1, num_weeks + 1))
    
    # Warehouse-Market Map
    with st.sidebar.expander("Warehouse-Market Mapping", expanded=False):
        st.write("Select which markets are served by each warehouse:")
        warehouse_market_map = {}
        
        for wh in warehouses:
            sel_markets = st.multiselect(
                f"Markets served by {wh}",
                options=markets,
                default=markets[:2] if len(markets) > 1 else markets
            )
            warehouse_market_map[wh] = sel_markets
    
    # Simulation parameters
    st.sidebar.markdown('<div class="subsection-header">Simulation Parameters</div>', unsafe_allow_html=True)
    
    with st.sidebar.expander("Supply & Demand", expanded=True):
        supply_dist = st.selectbox(
            "Supply Distribution",
            options=["normal", "uniform", "poisson", "exponential", "lognormal"],
            index=0
        )
        
        supply_mean = st.slider("Supply Mean", min_value=50, max_value=500, value=200)
        supply_std = st.slider("Supply Standard Deviation", min_value=10, max_value=100, value=40)
        supply_params = (supply_mean, supply_std)
        
        sell_in_dist = st.selectbox(
            "Demand Distribution",
            options=["normal", "uniform", "poisson", "exponential", "lognormal"],
            index=0
        )
        
        sell_in_mean = st.slider("Demand Mean", min_value=50, max_value=500, value=150)
        sell_in_std = st.slider("Demand Standard Deviation", min_value=10, max_value=100, value=35)
        sell_in_params = (sell_in_mean, sell_in_std)
        
        initial_inventory = st.slider(
            "Initial Inventory", 
            min_value=0, 
            max_value=5000, 
            value=int(sell_in_mean * 20)
        )
    
    with st.sidebar.expander("Lead Times", expanded=False):
        lead_time_dist = st.selectbox(
            "Lead Time Distribution",
            options=["uniform", "normal", "poisson"],
            index=0
        )
        
        lead_time_min = st.slider("Lead Time Min", min_value=1, max_value=20, value=3)
        lead_time_max = st.slider("Lead Time Max", min_value=lead_time_min, max_value=30, value=10)
        lead_time_params = (lead_time_min, lead_time_max)
        
        lead_time_scenario_group = ["plant", "warehouse", "product"]
    
    with st.sidebar.expander("Product Attributes", expanded=False):
        unit_cost_min = st.slider("Unit Cost Min", min_value=5, max_value=50, value=20)
        unit_cost_max = st.slider("Unit Cost Max", min_value=unit_cost_min, max_value=100, value=80)
        
        weight_mean = st.slider("Weight Mean", min_value=1, max_value=20, value=5)
        weight_std = st.slider("Weight Standard Deviation", min_value=0.5, max_value=10.0, value=2.0)
        
        product_attributes = {
            'unit_cost': {
                'distribution': 'uniform',
                'params': (unit_cost_min, unit_cost_max),
                'min_value': unit_cost_min,
                'max_value': unit_cost_max
            },
            'weight': {
                'distribution': 'normal',
                'params': (weight_mean, weight_std),
                'min_value': 1,
                'max_value': 10
            }
        }
    
    with st.sidebar.expander("Transportation", expanded=False):
        transportation_base_cost = st.slider("Base Cost", min_value=1, max_value=50, value=10)
        transportation_distance_factor = st.slider("Distance Factor", min_value=0.1, max_value=2.0, value=0.5)
    
    # Optimization parameters
    st.sidebar.markdown('<div class="subsection-header">Optimization Parameters</div>', unsafe_allow_html=True)
    
    with st.sidebar.expander("Safety Stock", expanded=True):
        service_level = st.slider("Service Level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    
    with st.sidebar.expander("Order Quantities", expanded=False):
        holding_cost = st.slider("Holding Cost (% of item value per year)", min_value=0.05, max_value=0.5, value=0.2, step=0.01)
        ordering_cost = st.slider("Ordering Cost ($ per order)", min_value=10, max_value=500, value=100)
        
        order_quantity_method = st.selectbox(
            "Order Quantity Calculation Method",
            options=["eoq", "fixed"],
            index=0
        )
    
    with st.sidebar.expander("Network Flow", expanded=False):
        optimization_horizon = st.slider("Optimization Horizon (weeks)", min_value=1, max_value=12, value=4)
        
        optimization_objective = st.selectbox(
            "Optimization Objective",
            options=["min_cost", "max_service"],
            index=0
        )
    
    # Collect all parameters
    params = {
        'plants': plants,
        'warehouses': warehouses,
        'markets': markets,
        'products': products,
        'weeks': weeks,
        'warehouse_market_map': warehouse_market_map,
        'supply_dist': supply_dist,
        'supply_params': supply_params,
        'sell_in_dist': sell_in_dist,
        'sell_in_params': sell_in_params,
        'sell_in_values': 0,  # Flag to check if values are zero
        'initial_inventory': initial_inventory,
        'lead_time_dist': lead_time_dist,
        'lead_time_params': lead_time_params,
        'lead_time_scenario_group': lead_time_scenario_group,
        'product_attributes': product_attributes,
        'transportation_base_cost': transportation_base_cost,
        'transportation_distance_factor': transportation_distance_factor,
        'service_level': service_level,
        'holding_cost': holding_cost,
        'ordering_cost': ordering_cost,
        'order_quantity_method': order_quantity_method,
        'optimization_horizon': optimization_horizon,
        'optimization_objective': optimization_objective
    }
    
    # Buttons to run simulation and optimization
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        run_sim = st.button("Run Simulation", key="run_simulation")
    
    with col2:
        run_opt = st.button("Run Optimization", key="run_optimization", disabled=not st.session_state.simulation_run)
    
    # Run simulation if button is clicked
    if run_sim:
        with st.spinner("Running supply chain simulation..."):
            try:
                simulator = Simulator(log_level=20)  # INFO level
                simulator = run_simulation(simulator, params)
                st.session_state.simulator = simulator
                st.session_state.simulation_run = True
                st.success("Simulation completed successfully!")
            except Exception as e:
                st.error(f"Error during simulation: {str(e)}")
                st.code(traceback.format_exc())
    
    # Run optimization if button is clicked
    if run_opt:
        with st.spinner("Running supply chain optimization..."):
            try:
                optimization_results = run_optimization(
                    st.session_state.simulator, 
                    params,
                    calculate_kpis,
                    apply_optimization_results,
                    calculate_improvements,
                    calculate_cost_improvements
                )
                st.session_state.optimization_results = optimization_results
                st.session_state.optimization_run = True
                st.success("Optimization completed successfully!")
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.simulation_run:
        simulator = st.session_state.simulator
        
        # Create tabs for different results
        tab_simulation, tab_optimization, tab_visualization, tab_download = st.tabs([
            "Simulation Results", 
            "Optimization Results", 
            "Visualizations",
            "Download Data"
        ])
        
        with tab_simulation:
            st.markdown('<div class="section-header">Simulation Results</div>', unsafe_allow_html=True)
            
            # Show summary statistics
            summary = simulator.get_summary()
            st.markdown('<div class="subsection-header">Supply Chain Summary</div>', unsafe_allow_html=True)
            st.dataframe(summary)
            
            # Show sample data
            st.markdown('<div class="subsection-header">Sample Data</div>', unsafe_allow_html=True)
            st.dataframe(simulator.df.head(10))
            
            # Show aggregated statistics
            st.markdown('<div class="subsection-header">Metrics by Warehouse</div>', unsafe_allow_html=True)
            warehouse_stats = simulator.df.groupby('warehouse').agg({
                'inventory': ['mean', 'sum'],
                'supply': ['mean', 'sum'],
                'sell_in': ['mean', 'sum']
            })
            st.dataframe(warehouse_stats)
            
            # Show product statistics
            st.markdown('<div class="subsection-header">Metrics by Product</div>', unsafe_allow_html=True)
            product_stats = simulator.df.groupby('product').agg({
                'inventory': ['mean', 'sum'],
                'supply': ['mean', 'sum'],
                'sell_in': ['mean', 'sum']
            })
            st.dataframe(product_stats)
        
        # Display optimization results if available
        if st.session_state.optimization_run:
            optimization_results = st.session_state.optimization_results
            
            with tab_optimization:
                st.markdown('<div class="section-header">Optimization Results</div>', unsafe_allow_html=True)
                
                # Display key improvement metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    inventory_reduction = optimization_results['improvements'].get('inventory_reduction', {}).get('percentage', 0)
                    st.metric(
                        "Inventory Reduction", 
                        f"{inventory_reduction:.2f}%",
                        delta=f"{inventory_reduction:.2f}%",
                        delta_color="inverse"
                    )
                
                with col2:
                    service_level_improvement = optimization_results['improvements'].get('service_level_improvement', {}).get('percentage', 0)
                    st.metric(
                        "Service Level Improvement", 
                        f"{service_level_improvement:.2f}%",
                        delta=f"{service_level_improvement:.2f}%"
                    )
                
                with col3:
                    stockout_reduction = optimization_results['improvements'].get('stockout_reduction', {}).get('percentage', 0)
                    st.metric(
                        "Stockout Reduction", 
                        f"{stockout_reduction:.2f}%",
                        delta=f"{stockout_reduction:.2f}%",
                        delta_color="inverse"
                    )
                
                with col4:
                    inventory_turns_improvement = optimization_results['improvements'].get('inventory_turns_improvement', {}).get('percentage', 0)
                    st.metric(
                        "Inventory Turns Improvement", 
                        f"{inventory_turns_improvement:.2f}%",
                        delta=f"{inventory_turns_improvement:.2f}%"
                    )
                
                # Optimization status
                st.markdown('<div class="subsection-header">Optimization Status</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-box">
                    <p><strong>Status:</strong> {optimization_results['optimization_summary']['status']}</p>
                    <p><strong>Objective Value:</strong> {optimization_results['optimization_summary']['objective_value']:.2f}</p>
                    <p><strong>Safety Stock Method:</strong> {optimization_results['safety_stock_method']}</p>
                    <p><strong>Total Flow from Plants:</strong> {optimization_results['optimization_summary'].get('total_flow_from_plants', 0):.2f}</p>
                    <p><strong>Total Flow to Markets:</strong> {optimization_results['optimization_summary'].get('total_flow_to_markets', 0):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Comparison of before and after
                st.markdown('<div class="subsection-header">Before vs After Optimization</div>', unsafe_allow_html=True)
                
                # Create comparison dataframe for key metrics
                compare_df = pd.DataFrame({
                    'Metric': ['Total Inventory', 'Average Inventory', 'Total Stockouts', 'Service Level', 'Inventory Turns'],
                    'Before': [
                        optimization_results['base_kpis'].get('total_inventory', 0),
                        optimization_results['base_kpis'].get('avg_inventory', 0),
                        optimization_results['base_kpis'].get('total_stockouts', 0),
                        optimization_results['base_kpis'].get('avg_service_level', 0),
                        optimization_results['base_kpis'].get('inventory_turns', 0)
                    ],
                    'After': [
                        optimization_results['optimized_kpis'].get('total_inventory', 0),
                        optimization_results['optimized_kpis'].get('avg_inventory', 0),
                        optimization_results['optimized_kpis'].get('total_stockouts', 0),
                        optimization_results['optimized_kpis'].get('avg_service_level', 0),
                        optimization_results['optimized_kpis'].get('inventory_turns', 0)
                    ]
                })
                
                # Calculate percentage changes
                compare_df['Change'] = ((compare_df['After'] - compare_df['Before']) / compare_df['Before'] * 100).round(2)
                compare_df['Change'] = compare_df['Change'].apply(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A")
                
                st.dataframe(compare_df)
                
                # Display warehouse-level improvements
                st.markdown('<div class="subsection-header">Warehouse-Level Improvements</div>', unsafe_allow_html=True)
                
                # Prepare warehouse comparison
                wh_comparison = prepare_warehouse_comparison(
                    optimization_results['before_df'], 
                    optimization_results['after_df']
                )
                
                # Display warehouse comparison
                st.dataframe(wh_comparison[[
                    'warehouse', 
                    'inventory_before', 'inventory_after', 'inventory_change',
                    'stockout_units_before', 'stockout_units_after', 'stockout_units_change',
                    'service_level_before', 'service_level_after', 'service_level_change'
                ]])
                
                # Display product-level improvements
                st.markdown('<div class="subsection-header">Product-Level Improvements</div>', unsafe_allow_html=True)
                
                # Prepare product comparison
                prod_comparison = prepare_product_comparison(
                    optimization_results['before_df'], 
                    optimization_results['after_df']
                )
                
                # Display product comparison
                st.dataframe(prod_comparison[[
                    'product', 
                    'inventory_before', 'inventory_after', 'inventory_change',
                    'stockout_units_before', 'stockout_units_after', 'stockout_units_change',
                    'service_level_before', 'service_level_after', 'service_level_change'
                ]])
            
            # Visualizations tab
            with tab_visualization:
                st.markdown('<div class="section-header">Supply Chain Visualizations</div>', unsafe_allow_html=True)
                
                # Network visualization
                st.markdown('<div class="subsection-header">Network Visualization</div>', unsafe_allow_html=True)
                
                # Add selector for product and week
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    selected_product = st.selectbox("Select Product", options=products)
                
                with viz_col2:
                    mid_week = len(weeks) // 2
                    selected_week = st.slider("Select Week", min_value=min(weeks), max_value=max(weeks), value=mid_week)
                
                # Create plotter for visualization
                if st.session_state.optimization_run:
                    plotter = Plotter(optimization_results['after_df'])
                else:
                    plotter = Plotter(simulator.df)
                
                # Create and show network visualization
                network_fig = create_network_visualization(
                    plotter, 
                    product=selected_product, 
                    week=selected_week
                )
                st.pyplot(network_fig)
                
                # Inventory time series visualization
                st.markdown('<div class="subsection-header">Inventory Time Series</div>', unsafe_allow_html=True)
                
                # Add selectors for warehouses and products
                viz_col3, viz_col4 = st.columns(2)
                
                with viz_col3:
                    selected_warehouses = st.multiselect(
                        "Select Warehouses", 
                        options=warehouses,
                        default=warehouses[:2] if len(warehouses) > 1 else warehouses
                    )
                
                with viz_col4:
                    selected_products = st.multiselect(
                        "Select Products", 
                        options=products,
                        default=products[:1] if len(products) > 0 else []
                    )
                
                # Create and show inventory time series
                if selected_warehouses and selected_products:
                    inventory_fig = create_inventory_time_series(
                        plotter, 
                        warehouses=selected_warehouses,
                        products=selected_products
                    )
                    st.pyplot(inventory_fig)
                else:
                    st.warning("Please select at least one warehouse and one product to view the inventory time series.")
                
                # Only show optimization-specific visualizations if optimization was run
                if st.session_state.optimization_run:
                    # KPI Dashboard
                    st.markdown('<div class="subsection-header">KPI Dashboard</div>', unsafe_allow_html=True)
                    
                    kpi_fig = create_kpi_dashboard_wrapper(
                        optimization_results['base_kpis'],
                        optimization_results['optimized_kpis'],
                        optimization_results['cost_improvements'],
                        create_kpi_dashboard
                    )
                    st.pyplot(kpi_fig)
                    
                    # Optimization Comparison
                    st.markdown('<div class="subsection-header">Optimization Comparison by Warehouse</div>', unsafe_allow_html=True)
                    
                    # Check available metrics
                    available_metrics = []
                    for metric in ['inventory', 'stockout_units', 'service_level']:
                        if metric in optimization_results['before_df'].columns and metric in optimization_results['after_df'].columns:
                            available_metrics.append(metric)
                    
                    # Create comparison visualization
                    if available_metrics:
                        comparison_fig = create_optimization_comparison_viz(
                            optimization_results['before_df'],
                            optimization_results['after_df'],
                            metrics=available_metrics,
                            group_by='warehouse',
                            create_optimization_comparison_fn=create_optimization_comparison
                        )
                        st.pyplot(comparison_fig)
                    else:
                        st.warning("No metrics available for comparison.")
            
            # Download tab
            with tab_download:
                st.markdown('<div class="section-header">Download Data</div>', unsafe_allow_html=True)
                
                # Download simulation data
                st.markdown('<div class="subsection-header">Simulation Data</div>', unsafe_allow_html=True)
                st.markdown(get_excel_download_link(simulator.df, filename="simulation_data.xlsx", text="Download Simulation Data"), unsafe_allow_html=True)
                
                # Download detailed summary
                detailed_summary = simulator.get_detailed_summary(
                    group_by=['product', 'warehouse', 'market']
                )
                st.markdown('<div class="subsection-header">Detailed Summary</div>', unsafe_allow_html=True)
                st.markdown(get_excel_download_link(detailed_summary, filename="detailed_summary.xlsx", text="Download Detailed Summary"), unsafe_allow_html=True)
                
                # If optimization was run, download optimization results
                if st.session_state.optimization_run:
                    st.markdown('<div class="subsection-header">Optimization Results</div>', unsafe_allow_html=True)
                    
                    # Convert KPIs to dataframe for download
                    kpi_df = pd.DataFrame([
                        {'Metric': 'Total Inventory', 'Before': optimization_results['base_kpis'].get('total_inventory', 0), 'After': optimization_results['optimized_kpis'].get('total_inventory', 0)},
                        {'Metric': 'Average Inventory', 'Before': optimization_results['base_kpis'].get('avg_inventory', 0), 'After': optimization_results['optimized_kpis'].get('avg_inventory', 0)},
                        {'Metric': 'Total Stockouts', 'Before': optimization_results['base_kpis'].get('total_stockouts', 0), 'After': optimization_results['optimized_kpis'].get('total_stockouts', 0)},
                        {'Metric': 'Service Level', 'Before': optimization_results['base_kpis'].get('avg_service_level', 0), 'After': optimization_results['optimized_kpis'].get('avg_service_level', 0)},
                        {'Metric': 'Inventory Turns', 'Before': optimization_results['base_kpis'].get('inventory_turns', 0), 'After': optimization_results['optimized_kpis'].get('inventory_turns', 0)}
                    ])
                    kpi_df['Improvement (%)'] = ((kpi_df['After'] - kpi_df['Before']) / kpi_df['Before'] * 100).round(2)
                    
                    st.markdown(get_excel_download_link(kpi_df, filename="optimization_kpis.xlsx", text="Download KPI Comparison"), unsafe_allow_html=True)
                    
                    # Download optimized data
                    st.markdown(get_excel_download_link(optimization_results['after_df'], filename="optimized_data.xlsx", text="Download Optimized Data"), unsafe_allow_html=True)
                    
                    # Download warehouse and product level comparisons
                    wh_comparison = prepare_warehouse_comparison(
                        optimization_results['before_df'], 
                        optimization_results['after_df']
                    )
                    
                    prod_comparison = prepare_product_comparison(
                        optimization_results['before_df'], 
                        optimization_results['after_df']
                    )
                    
                    st.markdown(get_excel_download_link(wh_comparison, filename="warehouse_comparison.xlsx", text="Download Warehouse Comparison"), unsafe_allow_html=True)
                    st.markdown(get_excel_download_link(prod_comparison, filename="product_comparison.xlsx", text="Download Product Comparison"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()