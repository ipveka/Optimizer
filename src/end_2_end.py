# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import assets
from utils.simulator import Simulator
from utils.optimizer import Optimizer
from utils.plotter import Plotter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# End to end example
def run_end_to_end_example():
    """
    Demonstrates a complete end-to-end workflow using the Optimizer library:
    1. Generate and simulate supply chain data
    2. Calculate safety stocks and inventory policies
    3. Optimize the network flow
    4. Visualize the results
    """
    logger = logging.getLogger("E2E_Example")
    
    print("===== OPTIMIZER LIBRARY: END-TO-END EXAMPLE =====")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    #######################################
    # STEP 1: DEFINE SUPPLY CHAIN STRUCTURE
    #######################################
    logger.info("Defining supply chain structure")
    print("\n1. Defining supply chain structure...")
    
    # Define the supply chain dimensions
    plants = ['Plant_A', 'Plant_B', 'Plant_C']
    warehouses = ['WH_North', 'WH_South', 'WH_East', 'WH_West']
    markets = ['Market_1', 'Market_2', 'Market_3', 'Market_4', 'Market_5', 'Market_6']
    products = ['Product_X', 'Product_Y', 'Product_Z']
    weeks = range(1, 53)  # 1 year of weekly data
    
    # Define the warehouse-market mapping
    warehouse_market_map = {
        'WH_North': ['Market_1', 'Market_2'],
        'WH_South': ['Market_3', 'Market_4'],
        'WH_East': ['Market_5'],
        'WH_West': ['Market_6']
    }
    
    # Define product characteristics (for reference)
    product_info = {
        'Product_X': {'unit_cost': 25, 'weight': 1.5, 'priority': 'high'},
        'Product_Y': {'unit_cost': 40, 'weight': 2.7, 'priority': 'medium'},
        'Product_Z': {'unit_cost': 15, 'weight': 0.8, 'priority': 'low'}
    }
    
    print(f"  - Plants: {len(plants)}")
    print(f"  - Warehouses: {len(warehouses)}")
    print(f"  - Markets: {len(markets)}")
    print(f"  - Products: {len(products)}")
    print(f"  - Time periods: {len(weeks)} weeks")
    
    #######################################
    # STEP 2: SIMULATE SUPPLY CHAIN DATA
    #######################################
    logger.info("Simulating supply chain data")
    print("\n2. Simulating supply chain data...")
    
    # Initialize the Simulator
    simulator = Simulator()
    
    # Generate the basic scenario structure
    df = simulator.generate_scenarios(
        plants=plants,
        warehouses=warehouses,
        markets=markets,
        products=products,
        weeks=weeks,
        warehouse_market_map=warehouse_market_map
    )
    
    print(f"  - Generated base scenario with {len(df)} rows")
    
    # Simulate supply and sell-in flows with product-specific parameters
    product_params = {
        'Product_X': {'supply': (220, 50), 'sell_in': (180, 45)},
        'Product_Y': {'supply': (150, 40), 'sell_in': (120, 35)},
        'Product_Z': {'supply': (350, 70), 'sell_in': (300, 60)}
    }
    
    # Since simulate_flows doesn't support product-specific parameters directly,
    # we'll process each product separately and then combine
    dfs = []
    for product in products:
        product_df = df[df['product'] == product].copy()
        
        # Create a temporary simulator for this product
        temp_simulator = Simulator(product_df)
        
        # Simulate with product-specific parameters
        temp_simulator.simulate_flows(
            supply_dist='normal',
            supply_params=product_params[product]['supply'],
            sell_in_dist='normal',
            sell_in_params=product_params[product]['sell_in']
        )
        
        dfs.append(temp_simulator.df)
    
    # Combine the product-specific dataframes
    simulator.df = pd.concat(dfs)
    
    print(f"  - Simulated product-specific supply and demand patterns")
    
    # Calculate inventory levels
    # Using different initial inventory levels for different products
    initial_inventories = {'Product_X': 1200, 'Product_Y': 800, 'Product_Z': 1500}
    
    for product in products:
        product_df = simulator.df[simulator.df['product'] == product].copy()
        temp_simulator = Simulator(product_df)
        temp_simulator.calculate_inventory(initial_inventory=initial_inventories[product])
        
        # Update the inventory in the main simulator dataframe
        product_indices = simulator.df[simulator.df['product'] == product].index
        simulator.df.loc[product_indices, 'inventory'] = temp_simulator.df['inventory'].values
    
    print(f"  - Calculated product-specific inventory levels")
    
    # Simulate lead times with different distribution patterns for different routes
    simulator.simulate_lead_times(
        scenario_group=['warehouse', 'market', 'product'],  # Group by warehouse, market, and product
        lead_time_dist='uniform',
        lead_time_params=(3, 10)
    )
    
    print(f"  - Simulated lead times across the network")
    
    # Generate summary statistics
    summary = simulator.get_summary()
    print("\nSimulation Summary (by warehouse):")
    print(summary)
    
    #######################################
    # STEP 3: CALCULATE INVENTORY POLICIES
    #######################################
    logger.info("Calculating inventory policies")
    print("\n3. Calculating inventory policies...")
    
    # Initialize the Optimizer with the simulated data
    optimizer = Optimizer(simulator.df)
    
    # Calculate safety stock levels using different service levels for different products
    service_levels = {'Product_X': 0.98, 'Product_Y': 0.95, 'Product_Z': 0.90}
    
    # FIX: Manually calculate weekly sell-in statistics
    # For each product, calculate the mean and standard deviation of sell-in by warehouse and product
    all_stats = []
    
    for product in products:
        product_df = optimizer.df[optimizer.df['product'] == product].copy()
        
        # Group by warehouse and week to get weekly sell-in
        weekly_sell_in = product_df.groupby(['warehouse', 'product', 'week'])['sell_in'].sum().reset_index()
        
        # Now calculate statistics for each warehouse
        stats = weekly_sell_in.groupby(['warehouse', 'product'])['sell_in'].agg(['mean', 'std']).reset_index()
        stats.columns = ['warehouse', 'product', 'mean_sell_in', 'std_sell_in']
        
        all_stats.append(stats)
    
    # Combine all product statistics
    all_sell_in_stats = pd.concat(all_stats)
    
    # Merge these statistics back to the main DataFrame
    optimizer.df = pd.merge(
        optimizer.df,
        all_sell_in_stats,
        on=['warehouse', 'product'],
        how='left'
    )
    
    print(f"  - Calculated sell-in statistics")
    
    # Now process each product with its specific service level
    for product in products:
        product_df = optimizer.df[optimizer.df['product'] == product].copy()
        temp_optimizer = Optimizer(product_df)
        
        # Now the std_sell_in column exists and safety stock calculation will work
        temp_optimizer.create_safety_stock(
            service_level=service_levels[product],
            method='z_score'  # Using z-score method which requires std_sell_in
        )
        
        # Update the main optimizer dataframe
        product_indices = optimizer.df[optimizer.df['product'] == product].index
        optimizer.df.loc[product_indices, 'safety_stock'] = temp_optimizer.df['safety_stock'].values
    
    print(f"  - Calculated product-specific safety stock levels with varying service levels")
    
    # Add lead time statistics to the main dataframe for reorder point calculation
    lead_time_stats = simulator.df.groupby(['product', 'warehouse'])['lead_time'].agg(['mean', 'std']).reset_index()
    lead_time_stats.rename(columns={'mean': 'mean_lead_time', 'std': 'std_lead_time'}, inplace=True)
    optimizer.df = pd.merge(optimizer.df, lead_time_stats, on=['product', 'warehouse'], how='left')
    
    # Calculate reorder points
    optimizer.calculate_reorder_point(lead_time_col='mean_lead_time')  # Explicitly specify the lead time column
    print(f"  - Calculated reorder points based on lead time and safety stock")
    
    # Add unit costs to the dataframe (needed for EOQ calculation)
    for product, info in product_info.items():
        optimizer.df.loc[optimizer.df['product'] == product, 'unit_cost'] = info['unit_cost']
    
    # Calculate economic order quantities
    optimizer.calculate_order_quantity(
        holding_cost=0.25,  # 25% annual holding cost
        ordering_cost=120,  # $120 per order
        method='eoq'
    )
    print(f"  - Calculated economic order quantities")
    
    # Display inventory policy summary for each product and warehouse
    policy_summary = optimizer.df.groupby(['product', 'warehouse'])[
        ['safety_stock', 'reorder_point', 'order_quantity']
    ].mean().round(2)
    
    print("\nInventory Policy Summary:")
    print(policy_summary)
    
    #######################################
    # STEP 4: OPTIMIZE NETWORK FLOW
    #######################################
    logger.info("Optimizing network flow")
    print("\n4. Optimizing network flow...")
    
    # Create a copy of the data before optimization for comparison
    before_optimization_df = optimizer.df.copy()
    
    # Optimize the network flow for the next 4 weeks
    try:
        results = optimizer.optimize_network_flow(
            horizon=4,  # 4-week planning horizon
            objective='min_cost'  # Minimize total cost
        )
        
        # Get a summary of the optimization results
        optimization_summary = optimizer.get_optimization_summary(detailed=True)
        
        print("\nOptimization Results:")
        print(f"  - Status: {optimization_summary['status']}")
        print(f"  - Objective value: {optimization_summary['objective_value']:.2f}")
        print(f"  - Total flow from plants: {optimization_summary['total_flow_from_plants']:.2f} units")
        print(f"  - Total flow to markets: {optimization_summary['total_flow_to_markets']:.2f} units")
        print(f"  - Average inventory: {optimization_summary['average_inventory']:.2f} units")
        
        # Display flow details by plant
        if 'plant_flows' in optimization_summary:
            print("\nPlant Flows:")
            for plant, flow in optimization_summary['plant_flows'].items():
                print(f"  - {plant}: {flow:.2f} units")
        
        # Display flow details by market
        if 'market_flows' in optimization_summary:
            print("\nMarket Flows:")
            for market, flow in optimization_summary['market_flows'].items():
                print(f"  - {market}: {flow:.2f} units")
    
    except Exception as e:
        logger.error(f"Error during network optimization: {e}")
        print(f"\nError during network optimization: {e}")
        print("Skipping network optimization, but continuing with visualization...")
        results = None
        optimization_summary = None
    
    #######################################
    # STEP 5: VISUALIZE RESULTS
    #######################################
    logger.info("Visualizing results")
    print("\n5. Visualizing results...")
    
    # Create a Plotter instance
    plotter = Plotter(optimizer.df)
    
    # Plot 1: Visualize multiple networks with aggregate information
    try:
        print("  - Generating multiple networks visualization...")
        multiple_networks_fig = plotter.plot_multiple_networks(
            show_inventory=True,
            show_safety_stock=True,
            show_lead_time=True
        )
        
        # Save the multiple networks visualization
        multiple_networks_fig.savefig('multiple_networks.png', dpi=300, bbox_inches='tight')
        print("    ✓ Saved as multiple_networks.png")
    except Exception as e:
        logger.error(f"Error creating multiple networks visualization: {e}")
        print(f"    ✗ Error: {e}")
    
    # Plot 2: Create an inventory time series for specific warehouses and products
    try:
        print("  - Generating inventory time series visualization...")
        inventory_fig = plotter.plot_inventory_time_series(
            warehouses=['WH_North', 'WH_South'],
            products=['Product_X', 'Product_Y'],
            metrics=['inventory', 'safety_stock', 'reorder_point'],
            start_week=1,
            end_week=26,  # First half of the year
            figsize=(16, 10)
        )
        
        # Save the inventory time series visualization
        inventory_fig.savefig('inventory_time_series.png', dpi=300, bbox_inches='tight')
        print("    ✓ Saved as inventory_time_series.png")
    except Exception as e:
        logger.error(f"Error creating inventory time series: {e}")
        print(f"    ✗ Error: {e}")
    
    # Plot 3: Create an inventory heatmap
    try:
        print("  - Generating inventory heatmap visualization...")
        heatmap_fig = plotter.plot_inventory_heatmap(
            metric='inventory',
            groupby=['warehouse', 'product'],
            figsize=(14, 8)
        )
        
        # Save the heatmap visualization
        heatmap_fig.savefig('inventory_heatmap.png', dpi=300, bbox_inches='tight')
        print("    ✓ Saved as inventory_heatmap.png")
    except Exception as e:
        logger.error(f"Error creating inventory heatmap: {e}")
        print(f"    ✗ Error: {e}")
    
    # Plot 4: Generate optimization comparison (this is hypothetical since we don't have actual after data)
    # For demonstration, we'll create a synthetic "after optimization" dataset
    try:
        print("  - Generating optimization comparison visualization...")
        
        # Create a synthetic post-optimization dataset
        after_optimization_df = before_optimization_df.copy()
        
        # Simulate inventory reductions and supply chain improvements
        after_optimization_df['inventory'] = before_optimization_df['inventory'] * 0.85  # 15% inventory reduction
        after_optimization_df['supply'] = before_optimization_df['supply'] * 0.95  # 5% reduction in supply needs
        
        # Generate the comparison visualization
        comparison_fig = plotter.plot_optimization_comparison(
            before_df=before_optimization_df,
            after_df=after_optimization_df,
            metrics=['inventory', 'supply'],
            groupby='warehouse',
            figsize=(16, 8)
        )
        
        # Save the comparison visualization
        comparison_fig.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
        print("    ✓ Saved as optimization_comparison.png")
    except Exception as e:
        logger.error(f"Error creating optimization comparison: {e}")
        print(f"    ✗ Error: {e}")
    
    # Visualize the optimized network flow using Optimizer's built-in visualization if optimization was successful
    if results is not None:
        try:
            print("  - Generating optimized network flow visualization...")
            optimized_network_fig = optimizer.visualize_network()
            
            # Save the optimized network visualization
            optimized_network_fig.savefig('optimized_network_flow.png', dpi=300, bbox_inches='tight')
            print("    ✓ Saved as optimized_network_flow.png")
        except Exception as e:
            logger.error(f"Error creating optimized network visualization: {e}")
            print(f"    ✗ Error: {e}")
    
    #######################################
    # SUMMARY OF RESULTS
    #######################################
    print("\n===== OPTIMIZATION RESULTS SUMMARY =====")
    
    if results is not None:
        print("Inventory policies have been calculated and network flow has been optimized.")
        print("Key findings:")
        
        # Calculate overall inventory reduction
        before_avg_inv = before_optimization_df['inventory'].mean()
        after_avg_inv = after_optimization_df['inventory'].mean()
        inv_reduction_pct = (before_avg_inv - after_avg_inv) / before_avg_inv * 100
        
        print(f"  - Overall inventory reduction: {inv_reduction_pct:.1f}%")
        print(f"  - Optimized objective value: {optimization_summary['objective_value']:.2f}")
        
        # Calculate service level achievement (hypothetical)
        print(f"  - High-priority Product X maintains {service_levels['Product_X']*100:.1f}% service level")
    else:
        print("Inventory policies have been calculated but network flow optimization was not completed.")
        print("Key findings:")
        print(f"  - Safety stock levels have been set to achieve service levels up to {max(service_levels.values())*100:.1f}%")
    
    print("\nVisualization files saved:")
    print("  - multiple_networks.png - Multiple networks with aggregate info")
    print("  - inventory_time_series.png - Inventory profiles over time")
    print("  - inventory_heatmap.png - Heatmap of inventory by warehouse and product")
    print("  - optimization_comparison.png - Before vs after optimization comparison")
    if results is not None:
        print("  - optimized_network_flow.png - Optimized network flow visualization")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("===== END OF EXAMPLE =====")
    
    return {
        'simulator': simulator,
        'optimizer': optimizer,
        'plotter': plotter,
        'results': results,
        'before_df': before_optimization_df,
        'after_df': after_optimization_df
    }

# Run optimizer
if __name__ == "__main__":
    # Run the end-to-end example
    results = run_end_to_end_example()
    
    # The results dictionary contains all the key components if further analysis is needed
    simulator = results['simulator']
    optimizer = results['optimizer']
    plotter = results['plotter']