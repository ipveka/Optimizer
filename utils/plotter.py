# Libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from matplotlib.lines import Line2D

# Plotter
class Plotter:
    """
    Plotter class for visualizing supply chain networks, inventory levels, and optimization results.
    
    This class provides comprehensive visualization tools for supply chain analysis, including:
    - Network visualizations showing plants, warehouses, and markets
    - Time series analysis of inventory, supply, and demand
    - Optimization result comparisons
    - Heatmaps for identifying bottlenecks and opportunities
    
    Attributes:
        df (pd.DataFrame): The supply chain data containing plant, warehouse, market dimensions
                          along with metrics like inventory, supply, and sell_in
    """
    
    def __init__(self, df):
        """
        Initialize the Plotter with a supply chain DataFrame.
        
        Args:
            df (pd.DataFrame): Supply chain data with plant, warehouse, market dimensions
                              and associated metrics
        """
        self.df = df
        # Set default style for consistent visualization
        plt.style.use('seaborn-v0_8-whitegrid')
        self.color_palette = {
            'plant': '#4285F4',      # Google Blue
            'warehouse': '#34A853',  # Google Green
            'market': '#EA4335',     # Google Red
            'inventory': '#FBBC05',  # Google Yellow
            'safety_stock': '#46BDC6',  # Teal
            'reorder_point': '#7E57C2',  # Purple
            'lead_time': '#F06292'   # Pink
        }
    
    def plot_network(self, product=None, week=None, figsize=(16, 12)):
        """
        Plot a supply chain network visualization for a specific product and/or time period.
        
        Args:
            product (str, optional): Filter visualization for a specific product
            week (int, optional): Filter visualization for a specific week
            figsize (tuple): Figure dimensions (width, height)
            
        Returns:
            matplotlib.figure.Figure: The network visualization figure
        """
        import numpy as np  # Make sure numpy is imported
        
        # Filter data if needed
        df = self.df.copy()
        if product:
            df = df[df['product'] == product]
        if week:
            df = df[df['week'] == week]
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Extract node sets
        plants = sorted(df['plant'].unique())
        warehouses = sorted(df['warehouse'].unique())
        markets = sorted(df['market'].unique())
        
        # Add nodes with type information
        for plant in plants:
            G.add_node(plant, node_type='plant')
        for warehouse in warehouses:
            G.add_node(warehouse, node_type='warehouse')
        for market in markets:
            G.add_node(market, node_type='market')
        
        # Calculate aggregated metrics for nodes
        warehouse_metrics = df.groupby('warehouse')['inventory'].mean().to_dict()
        
        # Add edges with flow information
        for plant in plants:
            for warehouse in warehouses:
                plant_wh_data = df[(df['plant'] == plant) & (df['warehouse'] == warehouse)]
                if not plant_wh_data.empty:
                    avg_supply = plant_wh_data['supply'].mean()
                    G.add_edge(plant, warehouse, weight=avg_supply, edge_type='supply')
        
        for warehouse in warehouses:
            for market in markets:
                wh_market_data = df[(df['warehouse'] == warehouse) & (df['market'] == market)]
                if not wh_market_data.empty:
                    avg_sell_in = wh_market_data['sell_in'].mean()
                    avg_lead_time = wh_market_data['lead_time'].mean() if 'lead_time' in wh_market_data.columns else 0
                    G.add_edge(warehouse, market, weight=avg_sell_in, lead_time=avg_lead_time, edge_type='sell_in')
        
        # Create custom layout with plants on left, warehouses in middle, markets on right
        pos = {}
        # Position plants on the left
        for i, plant in enumerate(plants):
            pos[plant] = (0.1, 0.9 - (i * 0.8 / max(len(plants), 1)))
            
        # Position warehouses in the middle
        for i, warehouse in enumerate(warehouses):
            pos[warehouse] = (0.5, 0.9 - (i * 0.8 / max(len(warehouses), 1)))
            
        # Position markets on the right
        for i, market in enumerate(markets):
            pos[market] = (0.9, 0.9 - (i * 0.8 / max(len(markets), 1)))
        
        # Create the figure
        fig = plt.figure(figsize=figsize)
        
        # Draw nodes with different colors based on node type
        node_sizes = {
            'plant': 8000,      # INCREASED SIZE MORE
            'warehouse': 8000,  # INCREASED SIZE MORE
            'market': 8000      # INCREASED SIZE MORE
        }
        
        # Draw plants
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=plants, 
                            node_color=self.color_palette['plant'],
                            node_size=node_sizes['plant'],
                            alpha=0.8,
                            edgecolors='black',
                            linewidths=2)
        
        # Draw warehouses
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=warehouses, 
                            node_color=self.color_palette['warehouse'],
                            node_size=node_sizes['warehouse'],
                            alpha=0.8,
                            edgecolors='black',
                            linewidths=2)
        
        # Draw markets
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=markets, 
                            node_color=self.color_palette['market'],
                            node_size=node_sizes['market'],
                            alpha=0.8,
                            edgecolors='black',
                            linewidths=2)
        
        # Draw edges with appropriate styles
        supply_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'supply']
        sell_in_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'sell_in']
        
        # OPTION 1: Use straight lines instead of curved
        # Draw supply edges with straight lines
        nx.draw_networkx_edges(G, pos, 
                            edgelist=supply_edges,
                            width=2.0,
                            edge_color='gray',
                            alpha=0.7,
                            arrowstyle='-|>',
                            arrowsize=20,
                            connectionstyle='arc3,rad=0.0')  # Changed from 0.1 to 0.0
        
        # Draw sell-in edges with straight lines
        nx.draw_networkx_edges(G, pos, 
                            edgelist=sell_in_edges,
                            width=2.0,
                            edge_color='gray',
                            alpha=0.7,
                            arrowstyle='-|>',
                            arrowsize=20,
                            connectionstyle='arc3,rad=0.0')  # Changed from 0.1 to 0.0
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')  # INCREASED SIZE
        
        # Create labels for warehouses with inventory
        warehouse_labels = {}
        for warehouse in warehouses:
            label_parts = []
            if warehouse in warehouse_metrics:
                label_parts.append(f"Inv: {warehouse_metrics[warehouse]:.0f}")
                # Add safety stock if available (calculated as 20% of inventory for simplicity)
                safety_stock = warehouse_metrics[warehouse] * 0.2
                label_parts.append(f"SS: {safety_stock:.0f}")
            
            if label_parts:
                warehouse_labels[warehouse] = "\n".join(label_parts)
        
        # Draw warehouse inventory labels
        label_pos = {n: (pos[n][0], pos[n][1] - 0.05) for n in warehouse_labels}
        nx.draw_networkx_labels(G, label_pos, labels=warehouse_labels, 
                            font_size=14, font_weight='normal')  # INCREASED SIZE
        
        # Add edge labels (flow volumes) - DIRECTLY ABOVE THE EDGES
        for u, v, data in G.edges(data=True):
            # Calculate the midpoint of the edge
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Calculate the midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Create the label text
            if data.get('edge_type') == 'supply':
                label_text = f"Supply: {data['weight']:.0f}"
            else:  # sell_in
                label_text = f"Sell-In: {data['weight']:.0f}"
                if 'lead_time' in data and data['lead_time'] > 0:
                    label_text += f"\nLT: {data['lead_time']:.1f}"
            
            # Position the label above the edge with a fixed offset
            label_y_offset = 0.01  # Increased offset to ensure visibility
            
            # Draw the label with a white background
            plt.text(mid_x, mid_y + label_y_offset, label_text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,  # REDUCED SIZE
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=1.0, ec='black', linewidth=1.5))
        
        # Add a title with information about the visualization
        title = "Supply Chain Network"
        if product:
            title += f" - Product: {product}"
        if week:
            title += f" - Week: {week}"
        
        plt.title(title, fontsize=20, pad=20)  # INCREASED SIZE
        
        plt.axis('off')
        plt.tight_layout()
        
        return fig

    def plot_multiple_networks(self, show_inventory=True, show_safety_stock=True, show_lead_time=True, figsize=(16, 12)):
        """
        Plots a combined network visualization for all products with centered layout.
        This includes the average supply, sell_in, and lead_time values aggregated across all products.
        
        Args:
            show_inventory (bool): Whether to show inventory information on warehouse nodes
            show_safety_stock (bool): Whether to show safety stock information on warehouse nodes
            show_lead_time (bool): Whether to show lead time information on market nodes
            figsize (tuple): Figure dimensions (width, height)
            
        Returns:
            matplotlib.figure.Figure: The network visualization figure
        """
        import numpy as np  # Make sure numpy is imported
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Calculate average inventory per warehouse
        avg_inventory = self.df.groupby('warehouse')['inventory'].mean().to_dict() if show_inventory else {}
        
        # Calculate average safety stock per warehouse if available
        avg_safety_stock = {}
        if 'safety_stock' in self.df.columns and show_safety_stock:
            avg_safety_stock = self.df.groupby('warehouse')['safety_stock'].mean().to_dict()
        
        # Get unique plants, warehouses, and markets
        plants = sorted(self.df['plant'].unique())
        warehouses = sorted(self.df['warehouse'].unique())
        markets = sorted(self.df['market'].unique())
        
        # Add nodes to the graph
        for plant in plants:
            G.add_node(plant, node_type='plant')
        for warehouse in warehouses:
            G.add_node(warehouse, node_type='warehouse')
        for market in markets:
            G.add_node(market, node_type='market')
        
        # Calculate average flows between nodes
        plant_to_warehouse_flow = {}
        for plant in plants:
            for warehouse in warehouses:
                flow_data = self.df[(self.df['plant'] == plant) & (self.df['warehouse'] == warehouse)]
                if not flow_data.empty:
                    avg_flow = flow_data['supply'].mean()
                    plant_to_warehouse_flow[(plant, warehouse)] = avg_flow
                    G.add_edge(plant, warehouse, weight=avg_flow, edge_type='supply')
        
        warehouse_to_market_flow = {}
        warehouse_to_market_lead_time = {}
        for warehouse in warehouses:
            for market in markets:
                flow_data = self.df[(self.df['warehouse'] == warehouse) & (self.df['market'] == market)]
                if not flow_data.empty:
                    avg_flow = flow_data['sell_in'].mean()
                    warehouse_to_market_flow[(warehouse, market)] = avg_flow
                    
                    if 'lead_time' in flow_data.columns and show_lead_time:
                        avg_lead_time = flow_data['lead_time'].mean()
                        warehouse_to_market_lead_time[(warehouse, market)] = avg_lead_time
                        G.add_edge(warehouse, market, weight=avg_flow, lead_time=avg_lead_time, edge_type='sell_in')
                    else:
                        G.add_edge(warehouse, market, weight=avg_flow, edge_type='sell_in')
        
        # Create custom layout with plants on left, warehouses in middle, markets on right
        pos = {}
        # Position plants on the left
        for i, plant in enumerate(plants):
            pos[plant] = (0.1, 0.9 - (i * 0.8 / max(len(plants), 1)))
            
        # Position warehouses in the middle
        for i, warehouse in enumerate(warehouses):
            pos[warehouse] = (0.5, 0.9 - (i * 0.8 / max(len(warehouses), 1)))
            
        # Position markets on the right
        for i, market in enumerate(markets):
            pos[market] = (0.9, 0.9 - (i * 0.8 / max(len(markets), 1)))
        
        # Create the figure
        fig = plt.figure(figsize=figsize)
        
        # Draw nodes with different colors based on node type
        node_sizes = {
            'plant': 8000,      # INCREASED SIZE MORE
            'warehouse': 8000,  # INCREASED SIZE MORE
            'market': 8000      # INCREASED SIZE MORE
        }
        
        # Draw plants
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=plants, 
                            node_color=self.color_palette['plant'],
                            node_size=node_sizes['plant'],
                            alpha=0.8,
                            edgecolors='black',
                            linewidths=2)
        
        # Draw warehouses
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=warehouses, 
                            node_color=self.color_palette['warehouse'],
                            node_size=node_sizes['warehouse'],
                            alpha=0.8,
                            edgecolors='black',
                            linewidths=2)
        
        # Draw markets
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=markets, 
                            node_color=self.color_palette['market'],
                            node_size=node_sizes['market'],
                            alpha=0.8,
                            edgecolors='black',
                            linewidths=2)
        
        # Draw edges with appropriate styles
        supply_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'supply']
        sell_in_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'sell_in']
        
        # OPTION 1: Draw with straight lines
        # Draw supply edges with straight lines
        nx.draw_networkx_edges(G, pos, 
                            edgelist=supply_edges,
                            width=2.0,
                            edge_color='gray',
                            alpha=0.7,
                            arrowstyle='-|>',
                            arrowsize=20,
                            connectionstyle='arc3,rad=0.0')  # Changed from 0.1 to 0.0
        
        # Draw sell-in edges with straight lines
        nx.draw_networkx_edges(G, pos, 
                            edgelist=sell_in_edges,
                            width=2.0,
                            edge_color='gray',
                            alpha=0.7,
                            arrowstyle='-|>',
                            arrowsize=20,
                            connectionstyle='arc3,rad=0.0')  # Changed from 0.1 to 0.0
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')
        
        # Create labels for warehouses with inventory and safety stock
        warehouse_labels = {}
        for warehouse in warehouses:
            label_parts = []
            if warehouse in avg_inventory and show_inventory:
                label_parts.append(f"Inv: {avg_inventory[warehouse]:.0f}")
            if warehouse in avg_safety_stock and show_safety_stock:
                label_parts.append(f"SS: {avg_safety_stock[warehouse]:.0f}")
            
            if label_parts:
                warehouse_labels[warehouse] = "\n".join(label_parts)
        
        # Draw warehouse labels
        label_pos = {n: (pos[n][0], pos[n][1] - 0.05) for n in warehouse_labels}
        nx.draw_networkx_labels(G, label_pos, labels=warehouse_labels, 
                            font_size=14, font_weight='normal')
        
        # Add edge labels with improved positioning for straight lines
        for u, v, data in G.edges(data=True):
            # Calculate the midpoint of the edge
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # For straight lines, midpoint calculation is simple
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Position the label above the edge with a fixed offset
            label_y_offset = 0.01 # Increased offset to ensure visibility
            
            # Create the label text
            if data.get('edge_type') == 'supply':
                label_text = f"Supply: {data['weight']:.0f}"
            else:  # sell_in
                label_text = f"Sell-In: {data['weight']:.0f}"
                if 'lead_time' in data and show_lead_time:
                    label_text += f"\nLT: {data['lead_time']:.1f}"
            
            # Draw the label with a white background
            plt.text(mid_x, mid_y + label_y_offset, label_text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=1.0, ec='black', linewidth=1.5))
        
        # Add a title
        plt.title("Aggregated Supply Chain Network", fontsize=20, pad=20)
        
        plt.axis('off')
        plt.tight_layout()
        
        return fig
    
    def plot_inventory_time_series(self, warehouses=None, products=None, metrics=None, 
                                   start_week=None, end_week=None, figsize=(14, 8)):
        """
        Plot time series of inventory levels and related metrics for selected warehouses and products.
        
        Args:
            warehouses (list, optional): List of warehouses to include (all if None)
            products (list, optional): List of products to include (all if None)
            metrics (list, optional): List of metrics to plot ('inventory', 'supply', 'sell_in', 
                                     'safety_stock', 'reorder_point')
            start_week (int, optional): Starting week for the time series
            end_week (int, optional): Ending week for the time series
            figsize (tuple): Figure dimensions (width, height)
            
        Returns:
            matplotlib.figure.Figure: The time series visualization figure
        """
        # Default metrics if none specified
        if metrics is None:
            metrics = ['inventory', 'supply', 'sell_in']
            if 'safety_stock' in self.df.columns:
                metrics.append('safety_stock')
            if 'reorder_point' in self.df.columns:
                metrics.append('reorder_point')
        
        # Filter the DataFrame
        df = self.df.copy()
        
        if warehouses:
            df = df[df['warehouse'].isin(warehouses)]
        if products:
            df = df[df['product'].isin(products)]
        if start_week is not None:
            df = df[df['week'] >= start_week]
        if end_week is not None:
            df = df[df['week'] <= end_week]
        
        # Check if we have data
        if df.empty:
            print("No data available for the selected filters.")
            return
        
        # Aggregate metrics by week
        time_series = df.groupby(['warehouse', 'product', 'week'])[metrics].mean().reset_index()
        
        # Create subplots for each warehouse-product combination
        wh_prod_combinations = time_series[['warehouse', 'product']].drop_duplicates()
        n_combinations = len(wh_prod_combinations)
        
        if n_combinations > 12:
            print(f"Warning: {n_combinations} warehouse-product combinations found. "
                 "Consider filtering to reduce complexity.")
        
        # Calculate subplot grid dimensions
        n_cols = min(3, n_combinations)
        n_rows = (n_combinations + n_cols - 1) // n_cols
        
        # Create the figure with adjusted height-to-width ratio (less vertical)
        fig_width = figsize[0]
        fig_height = fig_width * 0.75 * n_rows / n_cols  # Reduce height for less vertical plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=True)
        
        # Ensure axes is always a list/array
        if n_combinations > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot each warehouse-product combination
        for i, (_, row) in enumerate(wh_prod_combinations.iterrows()):
            if i >= len(axes):
                break
                
            warehouse = row['warehouse']
            product = row['product']
            
            # Get data for this combination
            combo_data = time_series[(time_series['warehouse'] == warehouse) & 
                                     (time_series['product'] == product)]
            
            ax = axes[i]
            
            # Plot each metric with reduced marker size and thinner lines
            for metric in metrics:
                if metric in combo_data.columns:
                    ax.plot(combo_data['week'], combo_data[metric], 
                           marker='o', 
                           markersize=4,  # Smaller markers
                           linewidth=1.5,  # Thinner lines
                           linestyle='-', 
                           label=metric.replace('_', ' ').title(),
                           color=self.color_palette.get(metric, None))
            
            ax.set_title(f"{warehouse} - {product}", fontsize=12)
            ax.set_xlabel('Week', fontsize=10)
            ax.set_ylabel('Units', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Inventory and Supply Chain Metrics Over Time', fontsize=18, y=1.02)
        
        return fig
    
    def plot_optimization_comparison(self, before_df, after_df, metrics=['inventory', 'supply'], 
                                     groupby='warehouse', figsize=(14, 8)):
        """
        Create bar charts comparing before and after optimization for key metrics.
        
        Args:
            before_df (pd.DataFrame): Data before optimization
            after_df (pd.DataFrame): Data after optimization
            metrics (list): Metrics to compare
            groupby (str): Dimension to group by (e.g., 'warehouse', 'product')
            figsize (tuple): Figure dimensions (width, height)
            
        Returns:
            matplotlib.figure.Figure: The comparison visualization figure
        """
        # Check that both DataFrames have the required columns
        for metric in metrics:
            if metric not in before_df.columns or metric not in after_df.columns:
                raise ValueError(f"Metric '{metric}' not found in one of the DataFrames")
        
        if groupby not in before_df.columns or groupby not in after_df.columns:
            raise ValueError(f"Groupby dimension '{groupby}' not found in one of the DataFrames")
        
        # Calculate aggregated values for each metric
        before_data = before_df.groupby(groupby)[metrics].mean().reset_index()
        after_data = after_df.groupby(groupby)[metrics].mean().reset_index()
        
        # Set up the figure
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
        
        # Ensure axes is always a list/array
        if n_metrics == 1:
            axes = [axes]
        
        # Create a bar chart for each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract and sort data for consistent display
            before_metric = before_data.set_index(groupby)[metric]
            after_metric = after_data.set_index(groupby)[metric]
            
            # Create a merged DataFrame for plotting
            compare_df = pd.DataFrame({
                'Before': before_metric,
                'After': after_metric
            }).reset_index()
            
            # Calculate the percentage change
            compare_df['Change %'] = ((compare_df['After'] - compare_df['Before']) / compare_df['Before'] * 100).round(1)
            
            # Sort by the groupby dimension
            compare_df = compare_df.sort_values(groupby)
            
            # Plot the before and after bars
            x = np.arange(len(compare_df))
            width = 0.35
            
            before_bars = ax.bar(x - width/2, compare_df['Before'], width, 
                               label='Before', color='#BBDEFB', edgecolor='black', linewidth=1)
            after_bars = ax.bar(x + width/2, compare_df['After'], width, 
                              label='After', color='#2196F3', edgecolor='black', linewidth=1)
            
            # Add labels and styling
            ax.set_title(f"{metric.replace('_', ' ').title()} Comparison", fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_xticks(x)
            ax.set_xticklabels(compare_df[groupby], rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add percentage change annotations
            for j, (_, row) in enumerate(compare_df.iterrows()):
                change_label = f"{row['Change %']:+.1f}%"
                color = 'green' if row['Change %'] >= 0 else 'red'
                ax.annotate(change_label, 
                          xy=(j, max(row['Before'], row['After']) * 1.05),
                          ha='center', va='bottom', 
                          color=color, fontweight='bold')
            
            # Add a legend
            ax.legend()
        
        plt.suptitle('Before vs. After Optimization Comparison', fontsize=16, y=1.05)
        plt.tight_layout()
        
        return fig