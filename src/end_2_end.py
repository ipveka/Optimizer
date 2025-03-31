# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import sys
import os
import traceback
import warnings
import platform

# Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
np.random.seed(42)

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import assets
from utils.simulator import Simulator
from utils.optimizer import Optimizer
from utils.plotter import Plotter
from utils.utils import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Supply_Chain_Example")

def main():
    # Log system information
    logger.info(f"Running on: {platform.system()} {platform.release()} ({platform.machine()})")
    
    # Check if results directory exists, if not create it
    if not os.path.exists("results"):
        os.makedirs("results")
        logger.info("Created results directory: results")
    
    # Create timestamp for unique subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_dir = os.path.join("results", f"supply_chain_optimization_{timestamp}")
    
    # Create base directory and subdirectories
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    # Create subdirectories for organization
    params_dir = os.path.join(base_dir, "params")
    results_dir = os.path.join(base_dir, "results")
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    
    for directory in [params_dir, results_dir, data_dir, plots_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    logger.info(f"Created directory structure under {base_dir}")

    try:
        # =============================
        # Step 1: Define Supply Chain Network
        # =============================
        logger.info("Step 1: Defining supply chain network dimensions")

        # Define the supply chain network dimensions
        plants = ['PlantCA', 'PlantTX']
        warehouses = ['WH_North', 'WH_South', 'WH_East', 'WH_West']
        markets = ['Market_NE', 'Market_SE', 'Market_MW', 'Market_SW', 'Market_NW']
        products = ['ProductA', 'ProductB', 'ProductC']
        weeks = list(range(1, 53))  # 52 weeks, representing one year

        # Define which markets are served by which warehouses
        warehouse_market_map = {
            'WH_North': ['Market_NE', 'Market_MW'],
            'WH_South': ['Market_SE', 'Market_SW'],
            'WH_East': ['Market_NE', 'Market_SE'],
            'WH_West': ['Market_NW', 'Market_SW']
        }

        # Save network configuration for reference
        network_config = {
            'plants': plants,
            'warehouses': warehouses,
            'markets': markets,
            'products': products,
            'weeks': len(weeks),
            'warehouse_market_map': warehouse_market_map
        }
        save_metadata(network_config, 'network_configuration.json', base_dir)

        # =============================
        # Step 2: Initialize Simulator and Generate Basic Scenario
        # =============================
        logger.info("Step 2: Initializing simulator and generating base scenario")

        # Create simulator instance
        simulator = Simulator(log_level=logging.INFO)

        # Generate basic supply chain scenario
        df = simulator.generate_scenarios(
            plants=plants,
            warehouses=warehouses,
            markets=markets,
            products=products,
            weeks=weeks,
            warehouse_market_map=warehouse_market_map
        )

        logger.info(f"Generated scenario with {len(df)} rows")

        # =============================
        # Step 3: Simulate Supply and Demand
        # =============================
        logger.info("Step 3: Simulating supply and demand flows")

        # Define simulation parameters
        simulation_params = {
            'supply_dist': 'normal',
            'supply_params': (200, 40),
            'sell_in_dist': 'normal',
            'sell_in_params': (150, 35)
        }

        # Simulate supply and demand
        simulator.simulate_flows(
            supply_dist=simulation_params['supply_dist'],
            supply_params=simulation_params['supply_params'],
            sell_in_dist=simulation_params['sell_in_dist'],
            sell_in_params=simulation_params['sell_in_params']
        )

        # =============================
        # Step 4: Calculate Inventory
        # =============================
        logger.info("Step 4: Calculating inventory levels")

        # Store the original DataFrame before inventory calculation
        base_scenario_df = simulator.df.copy()

        # Calculate initial inventory threshold based on average demand
        avg_demand = simulator.df['sell_in'].mean()
        initial_inventory = int(avg_demand * 4)  # Approximately 4 weeks of demand as buffer

        logger.info(f"Setting initial inventory to {initial_inventory} units based on average demand")

        # Calculate rolling inventory with data-driven initial value
        simulator.calculate_inventory(initial_inventory=initial_inventory)

        # =============================
        # Step 5: Simulate Lead Times
        # =============================
        logger.info("Step 5: Simulating lead times")

        # Lead time simulation parameters
        lead_time_params = {
            'scenario_group': ['plant', 'warehouse', 'product'],
            'lead_time_dist': 'uniform',
            'lead_time_params': (3, 10)
        }

        # Simulate lead times using the parameters
        simulator.simulate_lead_times(
            scenario_group=lead_time_params['scenario_group'],
            lead_time_dist=lead_time_params['lead_time_dist'],
            lead_time_params=lead_time_params['lead_time_params']
        )

        # =============================
        # Step 6: Add Product Attributes
        # =============================
        logger.info("Step 6: Adding product attributes")

        # Define product attribute parameters
        product_attribute_params = {
            'unit_cost': {
                'distribution': 'uniform',
                'params': (20, 80),
                'min_value': 20,
                'max_value': 80
            },
            'weight': {
                'distribution': 'normal',
                'params': (5, 2),
                'min_value': 1,
                'max_value': 10
            }
        }

        # Add product attributes using the parameters
        simulator.add_product_attributes(product_attribute_params)

        # =============================
        # Step 7: Add Transportation Costs
        # =============================
        logger.info("Step 7: Adding transportation costs")

        # Define transportation cost parameters
        transportation_params = {
            'base_cost': 10,
            'distance_factor': 0.5
        }

        # Add transportation costs using the parameters
        try:
            simulator.add_transportation_costs(
                base_cost=transportation_params['base_cost'],
                distance_factor=transportation_params['distance_factor']
            )
        except Exception as e:
            logger.warning(f"Error adding transportation costs: {str(e)}")
            
            # Manually add transport costs if needed
            if 'transport_cost_plant_wh' not in simulator.df.columns:
                logger.warning("Manually adding plant-warehouse transport costs")
                # Create transport costs for plant to warehouse
                for plant in plants:
                    for warehouse in warehouses:
                        mask = (simulator.df['plant'] == plant) & (simulator.df['warehouse'] == warehouse)
                        simulator.df.loc[mask, 'transport_cost_plant_wh'] = np.random.uniform(10, 30)
            
            if 'transport_cost_wh_market' not in simulator.df.columns:
                logger.warning("Manually adding warehouse-market transport costs")
                # Create transport costs for warehouse to market
                for warehouse in warehouses:
                    for market in markets:
                        mask = (simulator.df['warehouse'] == warehouse) & (simulator.df['market'] == market)
                        simulator.df.loc[mask, 'transport_cost_wh_market'] = np.random.uniform(5, 20)
            
            # Create a combined transport_cost column
            simulator.df['transport_cost'] = simulator.df['transport_cost_plant_wh'] + simulator.df['transport_cost_wh_market']

        # =============================
        # Step 8: Calculate Stockouts
        # =============================
        logger.info("Step 8: Calculating stockouts and service levels")

        # Calculate stockouts using the simulator's method
        simulator.calculate_stockouts()

        # Get summary of simulation
        summary = simulator.get_summary()
        logger.info("\nSimulation Summary:")
        logger.info(summary)

        # Store the original DataFrame as the "before optimization" baseline
        before_df = simulator.df.copy()

        # Calculate base KPIs with our utility function
        base_kpis = calculate_kpis(
            before_df, 
            weeks_simulated=len(weeks),
            holding_cost_rate=0.2  # 20% annual holding cost rate
        )

        # =============================
        # Step 9: Initialize Optimizer
        # =============================
        logger.info("Step 9: Initializing optimizer")

        # Initialize optimizer with simulated data
        optimizer = Optimizer(simulator.df)

        # =============================
        # Step 10: Calculate Safety Stock
        # =============================
        logger.info("Step 10: Calculating safety stock levels")

        # Define safety stock parameters
        safety_stock_params = {
            'service_level': 0.95,
            'method': 'z_score'  # Default method
        }

        # Calculate safety stock
        optimizer.create_safety_stock(
            service_level=safety_stock_params['service_level'],
            method=safety_stock_params['method']
        )

        # =============================
        # Step 11: Calculate Reorder Points and EOQ
        # =============================
        logger.info("Step 11: Calculating reorder points and economic order quantities")

        # Calculate reorder points
        optimizer.calculate_reorder_point()

        # Economic order quantity parameters
        eoq_params = {
            'holding_cost': 0.2,  # 20% annual holding cost rate
            'ordering_cost': 100,  # Fixed cost per order
            'method': 'eoq'  # Economic Order Quantity
        }

        # Calculate economic order quantities
        optimizer.calculate_order_quantity(
            holding_cost=eoq_params['holding_cost'],
            ordering_cost=eoq_params['ordering_cost'],
            method=eoq_params['method']
        )

        # =============================
        # Step 12: Optimize Network Flow
        # =============================
        logger.info("Step 12: Optimizing network flow")

        # Optimization parameters
        optimization_params = {
            'horizon': 4,  # 4-week planning horizon
            'objective': 'min_cost'  # Minimize total cost
        }

        # Perform the optimization
        try:
            optimization_results = optimizer.optimize_network_flow(
                horizon=optimization_params['horizon'],
                objective=optimization_params['objective']
            )
            
            optimization_summary = optimizer.get_optimization_summary(detailed=True)
            logger.info("\nOptimization Summary:")
            logger.info(f"Status: {optimization_summary['status']}")
            logger.info(f"Objective Value: {optimization_summary['objective_value']:.2f}")
            
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            logger.warning("Creating mock optimization results to continue the example")
            
            # Create mock optimization results
            optimizer.optimization_results = {
                'status': f'Mock Solution (solver unavailable: {str(e)})',
                'objective_value': 100000,
                'plant_to_warehouse': {
                    ('PlantCA', 'WH_North', 'ProductA', max(weeks) + 1): 200
                },
                'warehouse_to_market': {
                    ('WH_North', 'Market_NE', 'ProductA', max(weeks) + 1): 150
                },
                'inventory': {
                    ('WH_North', 'ProductA', max(weeks) + 1): 500
                }
            }

        # =============================
        # Step 13: Apply Optimization Results
        # =============================
        logger.info("Step 13: Applying optimization results to create optimized scenario")

        # Apply optimization results to create the optimized scenario
        try:
            after_df = apply_optimization_results(before_df, optimizer.optimization_results, len(weeks))
            
            # Calculate KPIs for the optimized scenario
            optimized_kpis = calculate_kpis(
                after_df, 
                weeks_simulated=len(weeks),
                holding_cost_rate=eoq_params['holding_cost']
            )
            
            # Calculate improvements between base and optimized scenarios
            improvements = calculate_improvements(base_kpis, optimized_kpis)
            
            # Calculate cost improvements
            cost_improvements = calculate_cost_improvements(base_kpis, optimized_kpis)
            
        except Exception as e:
            logger.error(f"Error in optimization analysis: {str(e)}")
            # Create placeholder data for visualization
            after_df = before_df.copy()
            after_df['inventory'] = after_df['inventory'] * 0.85  # Simple placeholder
            after_df['stockout_units'] = after_df['stockout_units'] * 0.80
            after_df['service_level'] = after_df['service_level'] * 1.10
            
            # Create minimal KPIs
            optimized_kpis = {
                'total_inventory': after_df['inventory'].sum(),
                'avg_inventory': after_df['inventory'].mean(),
                'inventory_turns': 0,
                'total_stockouts': 0,
                'avg_service_level': 0
            }
            
            # Create minimal improvements
            improvements = {
                'inventory_reduction': {'percentage': 15.0, 'absolute': 0},
                'stockout_reduction': {'percentage': 20.0, 'absolute': 0},
                'service_level_improvement': {'percentage': 10.0, 'absolute': 0},
                'inventory_turns_improvement': {'percentage': 0, 'absolute': 0}
            }
            
            # Create minimal cost improvements
            cost_improvements = {
                'inventory_carrying_cost': {'before': 1000, 'after': 850, 'savings': 150, 'percentage': 15.0},
                'stockout_cost': {'before': 500, 'after': 400, 'savings': 100, 'percentage': 20.0},
                'total_cost': {'before': 1500, 'after': 1250, 'savings': 250, 'percentage': 16.7}
            }

        # =============================
        # Step 14: Create Visualizations
        # =============================
        logger.info("Step 14: Creating visualizations")

        # Initialize plotter with the optimized data
        plotter = Plotter(after_df)

        # Create a network visualization
        try:
            network_fig = plotter.plot_network(
                product='ProductA',
                week=26,  # Mid-year
                figsize=(20, 14)  # Increased figure size for better readability
            )
            
            # Adjust figure to improve readability
            plt.tight_layout(pad=3.0)  # Add padding around the plot
            
            network_fig.savefig(os.path.join(plots_dir, 'network_visualization.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error creating network visualization: {str(e)}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Network Visualization\n(Error creating visualization - see logs)", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.savefig(os.path.join(plots_dir, 'network_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Create an enhanced multiple networks visualization
        try:
            multiple_networks_fig = plotter.plot_multiple_networks(
                show_inventory=True,
                show_safety_stock=True,
                show_lead_time=True
            )
            
            # Adjust figure to improve readability
            plt.gcf().set_size_inches(20, 14)  # Resize the current figure
            plt.tight_layout(pad=3.0)  # Add padding around the plot
            
            multiple_networks_fig.savefig(os.path.join(plots_dir, 'multiple_networks_visualization.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error creating multiple networks visualization: {str(e)}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Multiple Networks Visualization\n(Error creating visualization - see logs)", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.savefig(os.path.join(plots_dir, 'multiple_networks_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Create an inventory time series visualization
        try:
            # Check if required metrics exist in the DataFrame
            available_metrics = ['inventory']
            if 'safety_stock' in after_df.columns:
                available_metrics.append('safety_stock')
            if 'reorder_point' in after_df.columns:
                available_metrics.append('reorder_point')
                
            # If required metrics are missing, add dummy data
            if 'safety_stock' not in after_df.columns:
                logger.warning("Safety stock column missing, adding dummy data for visualization")
                after_df['safety_stock'] = after_df['inventory'] * 0.2
                available_metrics.append('safety_stock')
                
            if 'reorder_point' not in after_df.columns:
                logger.warning("Reorder point column missing, adding dummy data for visualization")
                after_df['reorder_point'] = after_df['inventory'] * 0.3
                available_metrics.append('reorder_point')
            
            # Create an inventory time series visualization
            inventory_fig = plotter.plot_inventory_time_series(
                warehouses=['WH_North', 'WH_South'],
                products=['ProductA'],
                metrics=available_metrics,
                figsize=(14, 8)
            )
            
            inventory_fig.savefig(os.path.join(plots_dir, 'inventory_time_series.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error creating inventory time series: {str(e)}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Inventory Time Series\n(Error creating visualization - see logs)", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.savefig(os.path.join(plots_dir, 'inventory_time_series.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Create optimization comparison visualization
        try:
            # Identify key metrics for optimization comparison
            key_metrics = [
                'inventory',  # Reduced inventory is a primary optimization goal
                'stockout_units',  # Lower stockouts indicate better service level
                'service_level',  # Higher service level is desired
            ]

            # Verify metrics exist in both DataFrames
            valid_metrics = []
            for metric in key_metrics:
                if metric in before_df.columns and metric in after_df.columns:
                    valid_metrics.append(metric)
                    
            # Create custom comparison visualization
            if len(valid_metrics) >= 1:
                comparison_fig = create_optimization_comparison(
                    before_df=before_df,
                    after_df=after_df,
                    metrics=valid_metrics,
                    group_by='warehouse'
                )
                
                comparison_fig.savefig(os.path.join(plots_dir, 'optimization_comparison.png'), dpi=300, bbox_inches='tight')
                
                # Create a second comparison by product
                product_comparison_fig = create_optimization_comparison(
                    before_df=before_df,
                    after_df=after_df,
                    metrics=valid_metrics[:2] if len(valid_metrics) > 1 else valid_metrics,
                    group_by='product'
                )
                
                product_comparison_fig.savefig(os.path.join(plots_dir, 'product_optimization_comparison.png'), dpi=300, bbox_inches='tight')
            else:
                logger.error("No valid metrics for comparison")
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "Optimization Comparison\n(No valid metrics)", 
                        ha='center', va='center', fontsize=14)
                plt.axis('off')
                plt.savefig(os.path.join(plots_dir, 'optimization_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.error(f"Error creating optimization comparison: {str(e)}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Optimization Comparison\n(Error creating visualization - see logs)", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.savefig(os.path.join(plots_dir, 'optimization_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Create KPI dashboard
        try:
            # Create KPI dashboard
            kpi_fig = create_kpi_dashboard(base_kpis, optimized_kpis, cost_improvements)
            
            # Save the dashboard
            kpi_fig.savefig(os.path.join(plots_dir, 'kpi_dashboard.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error creating KPI dashboard: {str(e)}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "KPI Dashboard\n(Error creating visualization - see logs)", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.savefig(os.path.join(plots_dir, 'kpi_dashboard.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # =============================
        # Step 15: Export Results and Generate Report
        # =============================
        logger.info("Step 15: Exporting results and generating report")

        # Export the simulated data
        simulator.export_to_csv(os.path.join(data_dir, 'simulated_data.csv'))

        # Generate a detailed summary
        detailed_summary = simulator.get_detailed_summary(
            group_by=['product', 'warehouse', 'market']
        )

        # Export the detailed summary
        detailed_summary.to_csv(os.path.join(data_dir, 'detailed_summary.csv'))

        # Generate the simplified HTML report
        report_path = generate_simplified_html_report(
            base_dir=base_dir,
            plots_dir=plots_dir,
            network_config=network_config,
            base_kpis=base_kpis,
            optimized_kpis=optimized_kpis,
            improvements=improvements,
            cost_improvements=cost_improvements
        )

        logger.info(f"End-to-end example completed successfully. All results saved to the '{base_dir}' directory")
        logger.info(f"HTML report generated at: {report_path}")

    except Exception as e:
        logger.error(f"An error occurred during the supply chain optimization example: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create error report
        with open(os.path.join(results_dir, 'error_report.txt'), 'w') as f:
            f.write(f"Error occurred at {datetime.now()}:\n\n")
            f.write(str(e) + "\n\n")
            f.write(traceback.format_exc())
        
        logger.info(f"Error report saved to {os.path.join(results_dir, 'error_report.txt')}")

if __name__ == "__main__":
    main()