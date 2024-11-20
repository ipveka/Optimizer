# General libraries
from datetime import date
from datetime import *
import pandas as pd
import numpy as np
import warnings
import logging
import mlflow
import sys
import os

# Plots
import networkx as nx
import matplotlib.pyplot as plt

# Stats
from scipy.stats import norm

# LP
import pulp

# Simulator class
import pandas as pd
import numpy as np

class Simulator:
    def __init__(self, df=None):
        """
        Initialize the Simulator with a DataFrame.
        
        Args:
            df (pd.DataFrame, optional): Input DataFrame. Creates an empty DataFrame if None.
        
        Initializes additional columns:
        - 'supply': Quantity of products supplied
        - 'sell_in': Quantity of products sold
        - 'inventory': Current inventory levels
        """
        # Use provided DataFrame or create an empty one
        self.df = df if df is not None else pd.DataFrame()
        
        # Initialize key columns with zero values
        self.df['supply'] = 0
        self.df['sell_in'] = 0
        self.df['inventory'] = 0

    def generate_scenarios(self, plants, warehouses, markets, products, weeks, warehouse_market_map):
        """
        Generate a comprehensive supply chain dataset with all dimension combinations.
        
        Args:
            plants (list): List of production plants
            warehouses (list): List of warehouses
            markets (list): List of markets
            products (list): List of products
            weeks (list): List of weeks to simulate
            warehouse_market_map (dict): Mapping of warehouses to their associated markets
        
        Returns:
            pd.DataFrame: Generated supply chain dataset
        """
        data = []
        
        # Create a cross-product of all dimensions
        for product in products:
            for week in weeks:
                for plant in plants:
                    for warehouse in warehouses:
                        # Get markets associated with the current warehouse
                        market_list = warehouse_market_map.get(warehouse)
                        
                        if market_list:
                            for market in market_list:
                                data.append({
                                    'plant': plant,
                                    'warehouse': warehouse,
                                    'market': market,
                                    'product': product,
                                    'week': week
                                })
                        else:
                            print(f"Warning: Warehouse '{warehouse}' has no market in warehouse_market_map.")
        
        # Create DataFrame and sort for consistent ordering
        self.df = pd.DataFrame(data)
        self.df.sort_values(by=['product', 'warehouse', 'week'], inplace=True)
        
        return self.df

    def simulate_flows(self, supply_dist='normal', supply_params=(200, 50), 
                       sell_in_dist='normal', sell_in_params=(150, 40)):
        """
        Simulate supply and sell-in data using various probability distributions.
        
        Args:
            supply_dist (str): Distribution type for supply ('normal', 'uniform', 'poisson')
            supply_params (tuple): Parameters for supply distribution
            sell_in_dist (str): Distribution type for sell-in ('normal', 'uniform', 'poisson')
            sell_in_params (tuple): Parameters for sell-in distribution
        
        Returns:
            pd.DataFrame: DataFrame with simulated supply and sell-in columns
        """
        # Mapping of distribution generators
        dist_generators = {
            'normal': lambda params, size: np.abs(np.random.normal(params[0], params[1], size)),
            'uniform': lambda params, size: np.random.uniform(params[0], params[1], size),
            'poisson': lambda params, size: np.random.poisson(params[0], size)
        }
        
        # Validate distribution types
        if supply_dist not in dist_generators or sell_in_dist not in dist_generators:
            raise ValueError(f"Invalid distribution. Supported: {list(dist_generators.keys())}")
        
        # Generate supply data
        supply_generator = dist_generators[supply_dist]
        self.df['supply'] = np.round(supply_generator(supply_params, len(self.df)))
        
        # Generate sell-in data
        sell_in_generator = dist_generators[sell_in_dist]
        self.df['sell_in'] = np.round(sell_in_generator(sell_in_params, len(self.df)))
        
        return self.df

    def calculate_inventory(self, initial_inventory=1000):
        """
        Calculate rolling inventory for each product-warehouse combination.
        
        Args:
            initial_inventory (int): Starting inventory level (default: 1000)
        
        Returns:
            pd.DataFrame: DataFrame with calculated inventory levels
        """
        # Sort DataFrame to ensure correct inventory calculation
        self.df = self.df.sort_values(['product', 'warehouse', 'week'])
        
        # Calculate inventory for each product-warehouse group
        for (product, warehouse), group in self.df.groupby(['product', 'warehouse']):
            indices = group.index
            current_inventory = initial_inventory
            
            # Update inventory based on supply and sell-in
            for idx in indices:
                current_inventory += self.df.loc[idx, 'supply'] - self.df.loc[idx, 'sell_in']
                self.df.loc[idx, 'inventory'] = current_inventory
        
        return self.df

    def simulate_lead_times(self, scenario_group=['plant', 'warehouse', 'market', 'product'], 
                            lead_time_dist='uniform', lead_time_params=(3, 10)):
        """
        Assign random lead times to unique combinations of specified scenario groups.
        
        Args:
            scenario_group (list): Dimensions to group by for unique lead time assignment
            lead_time_dist (str): Distribution type for lead times ('normal', 'uniform', 'poisson')
            lead_time_params (tuple): Parameters for lead time distribution
        
        Returns:
            pd.DataFrame: DataFrame with added lead time column
        """
        # Mapping of distribution generators (ensuring integer outputs)
        dist_generators = {
            'normal': lambda params, size: np.round(np.abs(np.random.normal(params[0], params[1], size))).astype(int),
            'uniform': lambda params, size: np.round(np.random.uniform(params[0], params[1], size)).astype(int),
            'poisson': lambda params, size: np.random.poisson(params[0], size)
        }
        
        # Validate distribution types
        if lead_time_dist not in dist_generators:
            raise ValueError(f"Invalid distribution. Supported: {list(dist_generators.keys())}")
        
        # Get unique combinations based on specified scenario groups
        unique_combinations = self.df[scenario_group].drop_duplicates()
        
        # Generate lead times using specified distribution
        lead_time_generator = dist_generators[lead_time_dist]
        unique_combinations['lead_time'] = lead_time_generator(lead_time_params, len(unique_combinations))
        
        # Merge lead times back to the main DataFrame
        self.df = pd.merge(
            self.df, 
            unique_combinations, 
            on=scenario_group, 
            how='left'
        )
        
        return self.df

    def get_summary(self):
        """
        Generate a summary of inventory levels, supply, sell-in, and lead times.
        
        Returns:
            pd.DataFrame or str: Aggregated summary statistics or message if no valid columns
        """
        # Prepare aggregation dictionary dynamically
        agg_dict = {}
        for col in ['inventory', 'supply', 'sell_in', 'lead_time']:
            if col in self.df.columns:
                agg_dict[col] = ['min', 'max', 'mean', 'std']
        
        # Return summary if aggregation columns exist
        return (
            self.df.groupby(['warehouse']).agg(agg_dict).round(2) 
            if agg_dict 
            else "No valid columns for summary."
        )
    