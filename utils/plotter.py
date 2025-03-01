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
    
    def plot_network(self, product=None, week=None, layout='spring', node_size_metric='inventory', 
                    edge_width_metric='flow', show_labels=True, figsize=(16, 12)):
        """
        Plot a comprehensive supply chain network visualization for a specific product and/or time period.
        
        Args:
            product (str, optional): Filter visualization for a specific product
            week (int, optional): Filter visualization for a specific week
            layout (str): Network layout algorithm ('spring', 'circular', 'spectral', or 'custom')
            node_size_metric (str): Metric to determine node sizes ('inventory', 'safety_stock', 'none')
            edge_width_metric (str): Metric to determine edge widths ('flow', 'lead_time', 'none')
            show_labels (bool): Whether to show detailed metric labels
            figsize (tuple): Figure dimensions (width, height)
            
        Returns:
            matplotlib.figure.Figure: The network visualization figure
        """
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
        node_metrics = {}
        
        # Warehouse metrics
        warehouse_metrics = df.groupby('warehouse').agg({
            'inventory': 'mean',
            'supply': 'mean',
            'sell_in': 'mean'
        })
        if 'safety_stock' in df.columns:
            warehouse_metrics['safety_stock'] = df.groupby('warehouse')['safety_stock'].mean()
        if 'reorder_point' in df.columns:
            warehouse_metrics['reorder_point'] = df.groupby('warehouse')['reorder_point'].mean()
        
        # Add warehouse metrics to node_metrics
        for warehouse in warehouses:
            if warehouse in warehouse_metrics.index:
                node_metrics[warehouse] = warehouse_metrics.loc[warehouse].to_dict()
        
        # Market metrics
        if 'lead_time' in df.columns:
            market_metrics = df.groupby('market')['lead_time'].mean()
            for market in markets:
                if market in market_metrics.index:
                    node_metrics[market] = {'lead_time': market_metrics[market]}
        
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
        
        # Determine node positions based on layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=100)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:  # custom layout with plants on left, warehouses in middle, markets on right
            pos = {}
            # Calculate vertical spacing and centering
            plant_y_positions = np.linspace(0.1, 0.9, len(plants))
            warehouse_y_positions = np.linspace(0.1, 0.9, len(warehouses))
            market_y_positions = np.linspace(0.1, 0.9, len(markets))
            
            for i, plant in enumerate(plants):
                pos[plant] = (0.1, plant_y_positions[i])
            for i, warehouse in enumerate(warehouses):
                pos[warehouse] = (0.5, warehouse_y_positions[i])
            for i, market in enumerate(markets):
                pos[market] = (0.9, market_y_positions[i])
        
        # Create the figure
        plt.figure(figsize=figsize)
        
        # Determine node sizes based on the selected metric
        node_sizes = {}
        if node_size_metric != 'none':
            max_size = 5000  # Maximum node size
            min_size = 1000  # Minimum node size
            
            # Get the metric values for all nodes
            metric_values = []
            for node in G.nodes():
                if node in node_metrics and node_size_metric in node_metrics[node]:
                    metric_values.append(node_metrics[node][node_size_metric])
            
            if metric_values:
                min_val = min(metric_values)
                max_val = max(metric_values)
                range_val = max_val - min_val if max_val > min_val else 1
                
                for node in G.nodes():
                    if node in node_metrics and node_size_metric in node_metrics[node]:
                        val = node_metrics[node][node_size_metric]
                        # Scale node size based on metric value
                        size = min_size + ((val - min_val) / range_val) * (max_size - min_size)
                    else:
                        size = min_size
                    node_sizes[node] = size
            else:
                # Default size if no metric values are available
                for node in G.nodes():
                    node_sizes[node] = min_size
        else:
            # Constant size if no metric is specified
            for node in G.nodes():
                node_type = G.nodes[node]['node_type']
                if node_type == 'plant':
                    node_sizes[node] = 2000
                elif node_type == 'warehouse':
                    node_sizes[node] = 3000
                else:  # market
                    node_sizes[node] = 2000
        
        # Draw nodes with different colors based on node type
        plant_nodes = [node for node in G.nodes() if G.nodes[node]['node_type'] == 'plant']
        warehouse_nodes = [node for node in G.nodes() if G.nodes[node]['node_type'] == 'warehouse']
        market_nodes = [node for node in G.nodes() if G.nodes[node]['node_type'] == 'market']
        
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=plant_nodes, 
                              node_color=self.color_palette['plant'],
                              node_size=[node_sizes[node] for node in plant_nodes],
                              alpha=0.8,
                              edgecolors='black',
                              linewidths=2)
        
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=warehouse_nodes, 
                              node_color=self.color_palette['warehouse'],
                              node_size=[node_sizes[node] for node in warehouse_nodes],
                              alpha=0.8,
                              edgecolors='black',
                              linewidths=2)
        
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=market_nodes, 
                              node_color=self.color_palette['market'],
                              node_size=[node_sizes[node] for node in market_nodes],
                              alpha=0.8,
                              edgecolors='black',
                              linewidths=2)
        
        # Determine edge widths based on the selected metric
        edge_widths = {}
        if edge_width_metric == 'flow':
            # Scale edge width based on flow volume
            max_width = 5.0
            min_width = 0.5
            
            # Get all flow values
            flow_values = [data['weight'] for _, _, data in G.edges(data=True) if 'weight' in data]
            
            if flow_values:
                min_flow = min(flow_values)
                max_flow = max(flow_values)
                flow_range = max_flow - min_flow if max_flow > min_flow else 1
                
                for u, v, data in G.edges(data=True):
                    if 'weight' in data:
                        # Scale width based on flow value
                        width = min_width + ((data['weight'] - min_flow) / flow_range) * (max_width - min_width)
                    else:
                        width = min_width
                    edge_widths[(u, v)] = width
            else:
                # Default width if no flow values are available
                for u, v in G.edges():
                    edge_widths[(u, v)] = min_width
        
        elif edge_width_metric == 'lead_time':
            # Scale edge width based on lead time
            max_width = 5.0
            min_width = 0.5
            
            # Get all lead time values
            lead_time_values = [data['lead_time'] for _, _, data in G.edges(data=True) 
                               if 'lead_time' in data and data['lead_time'] > 0]
            
            if lead_time_values:
                min_lt = min(lead_time_values)
                max_lt = max(lead_time_values)
                lt_range = max_lt - min_lt if max_lt > min_lt else 1
                
                for u, v, data in G.edges(data=True):
                    if 'lead_time' in data and data['lead_time'] > 0:
                        # Scale width based on lead time value
                        width = min_width + ((data['lead_time'] - min_lt) / lt_range) * (max_width - min_width)
                    else:
                        width = min_width
                    edge_widths[(u, v)] = width
            else:
                # Default width if no lead time values are available
                for u, v in G.edges():
                    edge_widths[(u, v)] = min_width
        else:
            # Constant width if no metric is specified
            for u, v in G.edges():
                edge_widths[(u, v)] = 1.5
        
        # Draw edges with appropriate styles
        supply_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'supply']
        sell_in_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'sell_in']
        
        nx.draw_networkx_edges(G, pos, 
                              edgelist=supply_edges,
                              width=[edge_widths[(u, v)] for u, v in supply_edges],
                              edge_color='gray',
                              alpha=0.7,
                              arrowstyle='-|>',
                              arrowsize=15)
        
        nx.draw_networkx_edges(G, pos, 
                              edgelist=sell_in_edges,
                              width=[edge_widths[(u, v)] for u, v in sell_in_edges],
                              edge_color='gray',
                              alpha=0.7,
                              arrowstyle='-|>',
                              arrowsize=15)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Draw detailed metric labels if requested
        if show_labels:
            # Create labels for warehouses with inventory and other metrics
            warehouse_labels = {}
            for warehouse in warehouses:
                if warehouse in node_metrics:
                    metrics = node_metrics[warehouse]
                    label = []
                    
                    if 'inventory' in metrics:
                        label.append(f"Inv: {metrics['inventory']:.1f}")
                    if 'safety_stock' in metrics:
                        label.append(f"SS: {metrics['safety_stock']:.1f}")
                    if 'reorder_point' in metrics:
                        label.append(f"ROP: {metrics['reorder_point']:.1f}")
                    
                    warehouse_labels[warehouse] = '\n'.join(label)
            
            # Create labels for markets with lead time
            market_labels = {}
            for market in markets:
                if market in node_metrics and 'lead_time' in node_metrics[market]:
                    market_labels[market] = f"LT: {node_metrics[market]['lead_time']:.1f}"
            
            # Draw warehouse metric labels
            label_pos = {n: (pos[n][0], pos[n][1] - 0.05) for n in warehouse_labels}
            nx.draw_networkx_labels(G, label_pos, labels=warehouse_labels, 
                                   font_size=8, font_weight='normal')
            
            # Draw market metric labels
            label_pos = {n: (pos[n][0], pos[n][1] - 0.05) for n in market_labels}
            nx.draw_networkx_labels(G, label_pos, labels=market_labels, 
                                   font_size=8, font_weight='normal')
            
            # Add edge labels (flow volumes)
            edge_labels = {}
            for u, v, data in G.edges(data=True):
                if 'weight' in data:
                    if data.get('edge_type') == 'supply':
                        edge_labels[(u, v)] = f"Supply: {data['weight']:.1f}"
                    else:  # sell_in
                        edge_labels[(u, v)] = f"Sell-in: {data['weight']:.1f}"
                        if 'lead_time' in data and data['lead_time'] > 0:
                            edge_labels[(u, v)] += f"\nLT: {data['lead_time']:.1f}"
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                        font_size=7, alpha=0.7)
        
        # Add a title with information about the visualization
        title = "Supply Chain Network"
        if product:
            title += f" - Product: {product}"
        if week:
            title += f" - Week: {week}"
        title += f"\nNode Size: {node_size_metric}, Edge Width: {edge_width_metric}"
        
        plt.title(title, fontsize=16, pad=20)
        
        # Add a legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.color_palette['plant'], 
                  markersize=15, label='Plants'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.color_palette['warehouse'], 
                  markersize=15, label='Warehouses'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.color_palette['market'], 
                  markersize=15, label='Markets')
        ]
        
        plt.legend(handles=legend_elements, loc='best')
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
    
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
        
        # Create the figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
        
        # Flatten axes array for easier indexing
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
            
            # Plot each metric
            for metric in metrics:
                if metric in combo_data.columns:
                    ax.plot(combo_data['week'], combo_data[metric], 
                           marker='o', 
                           linestyle='-', 
                           label=metric.replace('_', ' ').title(),
                           color=self.color_palette.get(metric, None))
            
            ax.set_title(f"{warehouse} - {product}", fontsize=10)
            ax.set_xlabel('Week')
            ax.set_ylabel('Units')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Inventory and Supply Chain Metrics Over Time', fontsize=16, y=1.02)
        
        return fig
    
    def plot_multiple_networks(self, show_inventory=True, show_safety_stock=True, show_lead_time=True):
        """
        Plots a combined network visualization for all products with centered layout.
        This includes the average supply, sell_in, and lead_time values aggregated across all products.
        
        Args:
            show_inventory (bool): Whether to show inventory information on warehouse nodes
            show_safety_stock (bool): Whether to show safety stock information on warehouse nodes
            show_lead_time (bool): Whether to show lead time information on market nodes
            
        Returns:
            matplotlib.figure.Figure: The network visualization figure
        """
        # Create a directed graph
        G = nx.DiGraph()

        # Calculate average inventory, safety stock, and lead time per warehouse/market
        avg_inventory = self.df.groupby('warehouse')['inventory'].mean().to_dict() if show_inventory else {}
        avg_safety_stock = (self.df.groupby('warehouse')['safety_stock'].mean().to_dict() 
                           if 'safety_stock' in self.df.columns and show_safety_stock else {})
        avg_lead_time = (self.df.groupby(['warehouse', 'market'])['lead_time'].mean().reset_index() 
                        if 'lead_time' in self.df.columns and show_lead_time else pd.DataFrame())

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
            
            # Add lead time to the edge if it exists
            if 'lead_time' in row and not pd.isna(row['lead_time']):
                if 'lead_time_list' not in G.edges[warehouse, market]:
                    G.edges[warehouse, market]['lead_time_list'] = []
                G.edges[warehouse, market]['lead_time_list'].append(row['lead_time'])

        # Calculate average values for edge labels
        for u, v, data in G.edges(data=True):
            avg_supply = np.mean(data['supply_list']) if 'supply_list' in data and data['supply_list'] else 0
            avg_sell_in = np.mean(data['sell_in_list']) if 'sell_in_list' in data and data['sell_in_list'] else 0
            avg_lead_time = np.mean(data['lead_time_list']) if 'lead_time_list' in data and data['lead_time_list'] else 0
            
            label = []
            if avg_supply > 0:
                label.append(f"Supply: {avg_supply:.1f}")
            if avg_sell_in > 0:
                label.append(f"Sell-In: {avg_sell_in:.1f}")
            if avg_lead_time > 0 and show_lead_time:
                label.append(f"LT: {avg_lead_time:.1f}")
            
            data['label'] = "\n".join(label)
            data['weight'] = max(avg_supply, avg_sell_in)  # Use the larger value for edge width

        # Get unique nodes for plants, warehouses, and markets
        plants = sorted(self.df['plant'].unique())
        warehouses = sorted(self.df['warehouse'].unique())
        markets = sorted(self.df['market'].unique())

        # Calculate positions with improved spacing
        pos = {}
        plant_x, warehouse_x, market_x = 0.1, 0.5, 0.9
        
        # Calculate vertical positions with better distribution
        max_nodes = max(len(plants), len(warehouses), len(markets))
        
        # Set vertical spacing based on node count (more nodes = less spacing)
        vertical_spacing = 0.8 / max(max_nodes, 1)
        
        # Position plants
        plant_start = 0.1 + (0.8 - (len(plants) - 1) * vertical_spacing) / 2
        for i, plant in enumerate(plants):
            pos[plant] = (plant_x, plant_start + i * vertical_spacing)

        # Position warehouses
        warehouse_start = 0.1 + (0.8 - (len(warehouses) - 1) * vertical_spacing) / 2
        for i, warehouse in enumerate(warehouses):
            pos[warehouse] = (warehouse_x, warehouse_start + i * vertical_spacing)

        # Position markets
        market_start = 0.1 + (0.8 - (len(markets) - 1) * vertical_spacing) / 2
        for i, market in enumerate(markets):
            pos[market] = (market_x, market_start + i * vertical_spacing)

        # Create the figure with responsive sizing
        fig = plt.figure(figsize=(16, 10))
        
        # Draw nodes with type-based colors and improved styling - INCREASED NODE SIZES
        node_sizes = {
            'plant': 6000,      # Increased from 3000
            'warehouse': 8000,  # Increased from 4000
            'market': 6000      # Increased from 3000
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

        # Calculate edge widths based on weights
        weights = [G.edges[u, v]['weight'] for u, v in G.edges()]
        max_weight = max(weights) if weights else 1
        
        # Scale edge widths between 1 and 5 based on the weight
        edge_widths = [1 + 4 * (G.edges[u, v]['weight'] / max_weight) for u, v in G.edges()]
        
        # Draw edges with improved styling
        nx.draw_networkx_edges(G, pos, 
                              width=edge_widths,
                              edge_color='gray',
                              alpha=0.7,
                              arrowstyle='-|>',
                              arrowsize=20,
                              connectionstyle='arc3,rad=0.1')  # Curved edges for better visibility

        # Draw node labels with improved styling - INCREASED FONT SIZE
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

        # Prepare warehouse labels with inventory and safety stock info
        warehouse_labels = {}
        for warehouse in warehouses:
            label_parts = []
            
            if warehouse in avg_inventory and show_inventory:
                label_parts.append(f"Inventory: {avg_inventory[warehouse]:.1f}")
            
            if warehouse in avg_safety_stock and show_safety_stock:
                label_parts.append(f"Safety Stock: {avg_safety_stock[warehouse]:.1f}")
            
            if label_parts:
                warehouse_labels[warehouse] = "\n".join(label_parts)
        
        # Draw warehouse labels with better positioning - INCREASED FONT SIZE & ADJUSTED POSITION
        warehouse_label_pos = {wh: (pos[wh][0], pos[wh][1] - 0.08) for wh in warehouse_labels}
        nx.draw_networkx_labels(G, warehouse_label_pos, labels=warehouse_labels, 
                               font_size=12, font_weight='normal')

        # Draw edge labels with improved positioning and formatting - INCREASED FONT SIZE
        edge_label_pos = 0.3  # Adjust this for label positioning on edges
        nx.draw_networkx_edge_labels(G, pos, 
                                    edge_labels={edge: data['label'] for edge, data in G.edges.items() if data['label']},
                                    label_pos=edge_label_pos, 
                                    font_size=11,
                                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                                    rotate=False)

        # Create a custom legend
        legend_elements = [
            mpatches.Patch(color=self.color_palette['plant'], label='Plants'),
            mpatches.Patch(color=self.color_palette['warehouse'], label='Warehouses'),
            mpatches.Patch(color=self.color_palette['market'], label='Markets'),
        ]
        
        if show_inventory:
            legend_elements.append(Line2D([0], [0], color='black', lw=2, label='Flow Direction'))
        
        plt.legend(handles=legend_elements, loc='best', fontsize=14)

        # Title and styling
        plt.title("Aggregated Supply Chain Network", fontsize=20, pad=20, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Fix to prevent showing the plot twice - return fig directly instead of using plt.gcf()
        plt.close()  # Close the current figure to prevent double display
        return fig
    
    def plot_inventory_heatmap(self, metric='inventory', groupby=None, normalize=False, figsize=(14, 10)):
        """
        Create a heatmap visualization of inventory or other metrics across the supply chain.
        
        Args:
            metric (str): Metric to visualize ('inventory', 'safety_stock', 'supply', 'sell_in', etc.)
            groupby (list, optional): Dimensions to group by (default: ['warehouse', 'product'])
            normalize (bool): Whether to normalize values (useful for comparing different metrics)
            figsize (tuple): Figure dimensions (width, height)
            
        Returns:
            matplotlib.figure.Figure: The heatmap visualization figure
        """
        # Default groupby if not specified
        if groupby is None:
            groupby = ['warehouse', 'product']
        
        # Ensure the metric exists in the DataFrame
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in the DataFrame. Available metrics: {list(self.df.columns)}")
        
        # Group data and calculate mean of the metric
        pivot_data = self.df.groupby(groupby)[metric].mean().reset_index()
        
        # Check if we have enough dimensions for a heatmap
        if len(groupby) < 2:
            raise ValueError("Need at least 2 dimensions for groupby to create a heatmap")
        
        # Create a pivot table for the heatmap
        pivot_table = pivot_data.pivot(index=groupby[0], columns=groupby[1], values=metric)
        
        # Normalize if requested
        if normalize:
            pivot_table = (pivot_table - pivot_table.min().min()) / (pivot_table.max().max() - pivot_table.min().min())
        
        # Create the figure
        plt.figure(figsize=figsize)
        
        # Create a custom colormap from light to dark in the metric's color
        base_color = self.color_palette.get(metric, '#4285F4')  # Default to blue if color not found
        cmap = LinearSegmentedColormap.from_list(
            f"{metric}_cmap", 
            ['#FFFFFF', base_color]
        )
        
        # Create the heatmap
        ax = sns.heatmap(pivot_table, 
                        annot=True, 
                        fmt='.1f' if not normalize else '.2f', 
                        cmap=cmap,
                        cbar_kws={'label': metric.replace('_', ' ').title()},
                        linewidths=0.5,
                        square=True)
        
        # Adjust labels and title
        plt.title(f"{metric.replace('_', ' ').title()} Heatmap by {groupby[0].title()} and {groupby[1].title()}", 
                fontsize=16, pad=20)
        plt.ylabel(groupby[0].replace('_', ' ').title())
        plt.xlabel(groupby[1].replace('_', ' ').title())
        
        # Rotate x-axis labels if there are many
        if len(pivot_table.columns) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return plt.gcf()
    
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