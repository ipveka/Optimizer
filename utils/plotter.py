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

# Plotter
class Plotter:
    def __init__(self, df):
        self.df = df

    def plot_multiple_networks(self):
        """
        Plots a combined network visualization for all products with centered layout.
        This includes the average supply, sell_in, and lead_time values aggregated across all products.
        """
        # Create a directed graph
        G = nx.DiGraph()

        # Calculate average inventory, safety stock, and lead time per warehouse/market
        avg_inventory = self.df.groupby('warehouse')['inventory'].mean().to_dict()
        avg_safety_stock = self.df.groupby('warehouse')['safety_stock'].mean().to_dict() if 'safety_stock' in self.df.columns else {}
        avg_lead_time = self.df.groupby('market')['lead_time'].mean().to_dict() if 'lead_time' in self.df.columns else {}

        # Add nodes and calculate average supply and sell_in for all connections
        for _, row in self.df.iterrows():
            plant = row['plant']
            warehouse = row['warehouse']
            market = row['market']
            
            # Add nodes to the graph
            G.add_node(plant, node_type='plant')
            G.add_node(warehouse, node_type='warehouse')
            G.add_node(market, node_type='market')

            # Add edges with average supply
            if not G.has_edge(plant, warehouse):
                G.add_edge(plant, warehouse, supply_list=[], sell_in_list=[])
            G.edges[plant, warehouse]['supply_list'].append(row['supply'])

            # Add edges with average sell_in
            if not G.has_edge(warehouse, market):
                G.add_edge(warehouse, market, supply_list=[], sell_in_list=[])
            G.edges[warehouse, market]['sell_in_list'].append(row['sell_in'])

        # Calculate average values for edge labels
        for u, v, data in G.edges(data=True):
            avg_supply = np.mean(data['supply_list']) if 'supply_list' in data and data['supply_list'] else 0
            avg_sell_in = np.mean(data['sell_in_list']) if 'sell_in_list' in data and data['sell_in_list'] else 0
            label = f"Avg Supply: {avg_supply:.2f}" if avg_supply > 0 else ""
            if avg_sell_in > 0:
                label += f" | Avg Sell-In: {avg_sell_in:.2f}"
            data['label'] = label

        # Get unique nodes for plants, warehouses, and markets and sort in descending order
        plants = sorted(self.df['plant'].unique(), reverse=True)
        warehouses = sorted(self.df['warehouse'].unique(), reverse=True)
        markets = sorted(self.df['market'].unique(), reverse=True)

        # Calculate centered positions
        pos = {}
        plant_x = 0.2 
        warehouse_x = 0.5
        market_x = 0.8

        # Calculate vertical positions with centering
        max_nodes = max(len(plants), len(warehouses), len(markets))
        vertical_spacing = 1.0 / (max_nodes + 1)

        # Center plants vertically
        plant_start = (1 - (len(plants) - 1) * vertical_spacing) / 2
        for i, plant in enumerate(plants):
            pos[plant] = (plant_x, plant_start + i * vertical_spacing)

        # Center warehouses vertically
        warehouse_start = (1 - (len(warehouses) - 1) * vertical_spacing) / 2
        for i, warehouse in enumerate(warehouses):
            pos[warehouse] = (warehouse_x, warehouse_start + i * vertical_spacing)

        # Center markets vertically
        market_start = (1 - (len(markets) - 1) * vertical_spacing) / 2
        for i, market in enumerate(markets):
            pos[market] = (market_x, market_start + i * vertical_spacing)

        # Plot with adjusted figure size and margins
        plt.figure(figsize=(20, 12))
        
        # Adjust margins to prevent cutoff
        plt.margins(x=0.2, y=0.2)

        # Draw nodes with type-based colors
        nx.draw_networkx_nodes(G, pos, nodelist=plants, node_color='lightblue', node_size=12000, label='Plants')
        nx.draw_networkx_nodes(G, pos, nodelist=warehouses, node_color='lightgreen', node_size=12000, label='Warehouses')
        nx.draw_networkx_nodes(G, pos, nodelist=markets, node_color='lightcoral', node_size=12000, label='Markets')

        # Draw edges with labels
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=11, edge_color='black')

        # Draw edge labels with dynamic position adjustments
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True) if d['label']}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=14)

        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

        # Add average inventory and safety stock as part of the warehouse labels
        warehouse_labels = {}
        for warehouse in warehouses:
            label = ""
            
            # Append safety stock information first if it exists
            if warehouse in avg_safety_stock:
                avg_safety_stock_val = avg_safety_stock[warehouse]
                label += f"Safety Stock: {avg_safety_stock_val:.2f}\n"
            
            # Append average inventory information next if it exists
            avg_inventory_val = avg_inventory.get(warehouse, 0)
            label += f"Avg Inv: {avg_inventory_val:.2f}"
            
            warehouse_labels[warehouse] = label

        # Adjust the position of the warehouse inventory and safety stock labels to appear below the node labels
        warehouse_label_pos = {}
        for warehouse, (x, y) in pos.items():
            if warehouse in warehouse_labels:
                warehouse_label_pos[warehouse] = (x, y - 0.05)

        # Draw the warehouse labels with safety stock first and average inventory second, both below the node label
        nx.draw_networkx_labels(G, warehouse_label_pos, labels=warehouse_labels, font_size=14, font_weight='normal', verticalalignment='center')

        # Add average lead time as part of the market labels
        market_labels = {}
        for market in markets:
            label = ""
            
            # Append average lead time information if it exists
            if market in avg_lead_time:
                avg_lead_time_val = avg_lead_time[market]
                label += f"Avg LT: {avg_lead_time_val:.2f}"

            market_labels[market] = label

        # Adjust the position of the market lead time labels to appear below the node labels
        market_label_pos = {}
        for market, (x, y) in pos.items():
            if market in market_labels:
                market_label_pos[market] = (x, y - 0.05)

        # Draw the market labels with average lead time below the node labels
        nx.draw_networkx_labels(G, market_label_pos, labels=market_labels, font_size=14, font_weight='normal', verticalalignment='center')

        # Title
        plt.title("Supply Chain Flow", fontsize=32, pad=20, color='black', fontweight='bold')
        
        # Remove axes
        plt.axis('off')
        
        # Show
        plt.tight_layout()
        plt.show()