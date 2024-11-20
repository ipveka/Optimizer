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
class Simulator:
    def __init__(self, df=None):
        """
        Initializes the Simulator with a DataFrame.
        If no DataFrame is provided, an empty DataFrame is initialized.
        """
        self.df = df if df is not None else pd.DataFrame()
        self.df['supply'] = 0
        self.df['sell_in'] = 0
        self.df['inventory'] = 0

    def generate_supply_chain_data(self, plants, warehouses, markets, products, weeks, warehouse_market_map):
        """
        Generates a supply chain dataset with all combinations of dimensions.
        Ensures that each warehouse is paired with a specific market as per the warehouse_market_map.
        """
        data = []
        for product in products:
            for week in weeks:
                for plant in plants:
                    for warehouse in warehouses:
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
        
        self.df = pd.DataFrame(data)
        self.df.sort_values(by=['product', 'warehouse', 'week'], inplace=True)
        return self.df

    def simulate_data_normal(self, supply_params=(200, 50), sell_in_params=(150, 40)):
        """
        Simulates data for supply and sell_in columns using normal distribution.
        """
        self.df['supply'] = np.round(
            np.abs(np.random.normal(supply_params[0], supply_params[1], len(self.df)))
        )
        self.df['sell_in'] = np.round(
            np.abs(np.random.normal(sell_in_params[0], sell_in_params[1], len(self.df)))
        )
        return self.df

    def calculate_rolling_inventory(self, initial_inventory=1000):
        """
        Calculates the rolling inventory for each product-warehouse combination.
        """
        self.df = self.df.sort_values(['product', 'warehouse', 'week'])
        for (product, warehouse), group in self.df.groupby(['product', 'warehouse']):
            indices = group.index
            current_inventory = initial_inventory
            for idx in indices:
                current_inventory += self.df.loc[idx, 'supply'] - self.df.loc[idx, 'sell_in']
                self.df.loc[idx, 'inventory'] = current_inventory
        return self.df

    def set_lead_times(self):
        """
        Assigns random lead times to each combination of plant, warehouse, market, and product.
        """
        unique_combinations = self.df[['plant', 'warehouse', 'market', 'product']].drop_duplicates()
        unique_combinations['lead_time'] = np.random.randint(3, 11, size=len(unique_combinations))
        self.df = pd.merge(self.df, unique_combinations, on=['plant', 'warehouse', 'market', 'product'], how='left')
        return self.df

    def get_summary(self):
        """
        Returns a summary of inventory levels by product and warehouse.
        """
        agg_dict = {}
        for col in ['inventory', 'supply', 'sell_in', 'lead_time']:
            if col in self.df.columns:
                agg_dict[col] = ['min', 'max', 'mean', 'std']
        return self.df.groupby(['warehouse']).agg(agg_dict).round(2) if agg_dict else "No valid columns for summary."
