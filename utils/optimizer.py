# Libraries
import pandas as pd
import numpy as np
import pulp
from scipy.stats import norm
import networkx as nx
import matplotlib.pyplot as plt

# Optimizer
class Optimizer:
    def __init__(self, df):
        """
        Initialize the Optimizer with the given DataFrame.
        
        Args:
            df (pd.DataFrame): Supply chain data with plant, warehouse, market, product, 
                              and week dimensions along with inventory, supply, and sell_in data.
        """
        self.df = df
        self.optimization_results = None
        
    def create_safety_stock(self, service_level=0.95, method='z_score', date_col='week'):
        """
        Adds a safety stock column to self.df based on sell_in variability using different methods.
        
        Args:
            service_level (float): Desired service level (default: 0.95)
            method (str): Method to calculate safety stock - 'z_score' or 'demand_variability'
            date_col (str): Column name for time periods (default: 'week')
            
        Returns:
            pd.DataFrame: DataFrame with added safety_stock column
        """
        z_score = norm.ppf(service_level)
        
        # Calculate weekly sell-in statistics
        weekly_sell_in_df = self.df.groupby(['warehouse', 'product', date_col])['sell_in'].sum().reset_index()
        
        # Compute mean and standard deviation for each product-warehouse combination
        policy_df = (
            weekly_sell_in_df.groupby(['product', 'warehouse'])['sell_in']
            .agg(['std', 'mean'])
            .rename(columns={'std': 'std_sell_in', 'mean': 'mean_sell_in'})
            .reset_index()
        )
        
        # Merge statistics back to the main DataFrame
        self.df = pd.merge(self.df, policy_df, on=['product', 'warehouse'], how='left')
        
        # Calculate safety stock based on the chosen method
        if method == 'z_score':
            self.df['safety_stock'] = (z_score * self.df['std_sell_in']).fillna(0).round(2)
            
        elif method == 'demand_variability':
            if 'lead_time' not in self.df.columns:
                raise ValueError("The 'lead_time' column is required for 'demand_variability'.")
                
            lead_time_df = self.df.groupby(['product', 'warehouse'])['lead_time'].agg(['std', 'mean']).reset_index()
            self.df = pd.merge(self.df, lead_time_df, on=['product', 'warehouse'], how='left')
            
            self.df['safety_stock'] = (
                z_score * np.sqrt(
                    (self.df['std_sell_in'] ** 2) * self.df['mean_lead_time'] +
                    (self.df['mean_sell_in'] ** 2) * self.df['std_lead_time'] ** 2
                )
            ).fillna(0).round(2)
            
        else:
            raise ValueError("Invalid method. Choose either 'z_score' or 'demand_variability'.")
            
        return self.df
    
    def calculate_reorder_point(self, lead_time_col='lead_time'):
        """
        Calculate reorder points for each product-warehouse combination.
        
        Args:
            lead_time_col (str): Column name for lead time (default: 'lead_time')
            
        Returns:
            pd.DataFrame: DataFrame with added reorder_point column
        """
        if 'mean_sell_in' not in self.df.columns or 'safety_stock' not in self.df.columns:
            raise ValueError("Please run create_safety_stock() first to generate required columns.")
            
        if lead_time_col not in self.df.columns:
            raise ValueError(f"The '{lead_time_col}' column is required for reorder point calculation.")
            
        # Calculate reorder point as: mean demand during lead time + safety stock
        self.df['reorder_point'] = (self.df['mean_sell_in'] * self.df[lead_time_col] + self.df['safety_stock']).round(2)
        
        return self.df
    
    def calculate_order_quantity(self, holding_cost=0.2, ordering_cost=100, method='eoq'):
        """
        Calculate economic order quantities for each product-warehouse combination.
        
        Args:
            holding_cost (float): Annual holding cost as a fraction of item value (default: 0.2)
            ordering_cost (float): Fixed cost per order (default: 100)
            method (str): Method to calculate order quantity - 'eoq' (Economic Order Quantity) or 'fixed'
            
        Returns:
            pd.DataFrame: DataFrame with added order_quantity column
        """
        if 'mean_sell_in' not in self.df.columns:
            raise ValueError("Please run create_safety_stock() first to generate required columns.")
        
        # Add unit_cost if not present (for simplicity, using random values)
        if 'unit_cost' not in self.df.columns:
            print("Warning: 'unit_cost' column not found. Using placeholder values.")
            # Group by product to ensure consistent unit cost per product
            product_costs = {
                product: np.random.uniform(10, 100)
                for product in self.df['product'].unique()
            }
            self.df['unit_cost'] = self.df['product'].map(product_costs)
        
        if method == 'eoq':
            # Calculate annual demand (assuming weeks as time units)
            self.df['annual_demand'] = self.df['mean_sell_in'] * 52
            
            # Economic Order Quantity formula: Q = sqrt(2DS/H)
            # Where D is annual demand, S is ordering cost, H is holding cost per unit
            self.df['order_quantity'] = np.sqrt(
                (2 * self.df['annual_demand'] * ordering_cost) / 
                (holding_cost * self.df['unit_cost'])
            ).round(2)
            
        elif method == 'fixed':
            # Simple fixed order quantity based on 4 weeks of demand
            self.df['order_quantity'] = (self.df['mean_sell_in'] * 4).round(2)
            
        else:
            raise ValueError("Invalid method. Choose either 'eoq' or 'fixed'.")
            
        return self.df
    
    def optimize_network_flow(self, horizon=4, objective='min_cost'):
        """
        Optimize the flow of products from plants to warehouses to markets over a time horizon.
        
        Args:
            horizon (int): Number of future periods to optimize for (default: 4)
            objective (str): Optimization objective - 'min_cost' or 'max_service'
            
        Returns:
            dict: Optimization results with flows and objective value
        """
        # Extract unique dimensions
        plants = self.df['plant'].unique()
        warehouses = self.df['warehouse'].unique()
        markets = self.df['market'].unique()
        products = self.df['product'].unique()
        
        # Get the maximum week to use as the starting point
        current_week = self.df['week'].max()
        future_weeks = range(current_week + 1, current_week + horizon + 1)
        
        # Create transportation costs if not available (using distances as proxy)
        if 'transport_cost' not in self.df.columns:
            print("Warning: 'transport_cost' column not found. Using placeholder values.")
            
            # Create placeholder transportation costs
            transport_costs = {}
            for plant in plants:
                for warehouse in warehouses:
                    # Randomly assign transportation costs between plant and warehouse
                    transport_costs[(plant, warehouse)] = np.random.uniform(10, 50)
            
            for warehouse in warehouses:
                for market in markets:
                    # Assign transportation costs between warehouse and market
                    transport_costs[(warehouse, market)] = np.random.uniform(5, 25)
        else:
            # Extract transportation costs from the DataFrame
            transport_costs = {}
            for _, row in self.df.drop_duplicates(['plant', 'warehouse']).iterrows():
                transport_costs[(row['plant'], row['warehouse'])] = row['transport_cost']
                
            for _, row in self.df.drop_duplicates(['warehouse', 'market']).iterrows():
                transport_costs[(row['warehouse'], row['market'])] = row['transport_cost']
        
        # Get production capacity at plants (using max historical supply as proxy)
        plant_capacity = {}
        for plant in plants:
            for product in products:
                plant_subset = self.df[(self.df['plant'] == plant) & (self.df['product'] == product)]
                if not plant_subset.empty:
                    max_supply = plant_subset['supply'].max()
                    plant_capacity[(plant, product)] = max_supply * 1.2  # Add 20% buffer
                else:
                    plant_capacity[(plant, product)] = 0
        
        # Get warehouse capacity (using max historical inventory as proxy)
        warehouse_capacity = {}
        for warehouse in warehouses:
            for product in products:
                wh_subset = self.df[(self.df['warehouse'] == warehouse) & (self.df['product'] == product)]
                if not wh_subset.empty:
                    max_inventory = wh_subset['inventory'].max()
                    warehouse_capacity[(warehouse, product)] = max_inventory * 1.5  # Add 50% buffer
                else:
                    warehouse_capacity[(warehouse, product)] = 0
        
        # Project demand for future weeks using mean historical sell-in
        future_demand = {}
        for market in markets:
            for product in products:
                for week in future_weeks:
                    market_subset = self.df[(self.df['market'] == market) & (self.df['product'] == product)]
                    if not market_subset.empty:
                        mean_demand = market_subset['sell_in'].mean()
                        future_demand[(market, product, week)] = mean_demand
                    else:
                        future_demand[(market, product, week)] = 0
        
        # Create a PuLP model
        model = pulp.LpProblem(name="Supply_Chain_Optimization", sense=pulp.LpMinimize)
        
        # Decision variables
        # Flow from plant to warehouse
        plant_to_wh = {}
        for plant in plants:
            for warehouse in warehouses:
                for product in products:
                    for week in future_weeks:
                        plant_to_wh[(plant, warehouse, product, week)] = pulp.LpVariable(
                            name=f"flow_p{plant}_w{warehouse}_prod{product}_week{week}",
                            lowBound=0,
                            cat='Continuous'
                        )
        
        # Flow from warehouse to market
        wh_to_market = {}
        for warehouse in warehouses:
            for market in markets:
                for product in products:
                    for week in future_weeks:
                        wh_to_market[(warehouse, market, product, week)] = pulp.LpVariable(
                            name=f"flow_w{warehouse}_m{market}_prod{product}_week{week}",
                            lowBound=0,
                            cat='Continuous'
                        )
        
        # Inventory at warehouse
        inventory = {}
        for warehouse in warehouses:
            for product in products:
                for week in future_weeks:
                    inventory[(warehouse, product, week)] = pulp.LpVariable(
                        name=f"inv_w{warehouse}_prod{product}_week{week}",
                        lowBound=0,
                        cat='Continuous'
                    )
        
        # Get initial inventory levels (latest available week)
        initial_inventory = {}
        for warehouse in warehouses:
            for product in products:
                wh_product_data = self.df[
                    (self.df['warehouse'] == warehouse) & 
                    (self.df['product'] == product) & 
                    (self.df['week'] == current_week)
                ]
                if not wh_product_data.empty:
                    initial_inventory[(warehouse, product)] = wh_product_data['inventory'].iloc[0]
                else:
                    initial_inventory[(warehouse, product)] = 0
        
        # Objective function
        if objective == 'min_cost':
            # Minimize transportation costs + holding costs
            transport_cost_expr = pulp.lpSum([
                # Plant to warehouse transport
                plant_to_wh[(plant, warehouse, product, week)] * transport_costs.get((plant, warehouse), 0)
                for plant in plants
                for warehouse in warehouses
                for product in products
                for week in future_weeks
            ]) + pulp.lpSum([
                # Warehouse to market transport
                wh_to_market[(warehouse, market, product, week)] * transport_costs.get((warehouse, market), 0)
                for warehouse in warehouses
                for market in markets
                for product in products
                for week in future_weeks
            ])
            
            # Holding cost
            holding_cost_expr = pulp.lpSum([
                inventory[(warehouse, product, week)] * 0.1  # Arbitrary holding cost per unit
                for warehouse in warehouses
                for product in products
                for week in future_weeks
            ])
            
            model += transport_cost_expr + holding_cost_expr
            
        elif objective == 'max_service':
            # Maximize fulfilled demand (or minimize unfulfilled demand)
            fulfilled_demand = pulp.lpSum([
                wh_to_market[(warehouse, market, product, week)]
                for warehouse in warehouses
                for market in markets
                for product in products
                for week in future_weeks
            ])
            
            total_demand = sum(future_demand.values())
            
            # Minimize the negative of fulfilled demand (equivalent to maximizing)
            model += -fulfilled_demand
            
        else:
            raise ValueError("Invalid objective. Choose either 'min_cost' or 'max_service'.")
        
        # Constraints
        # 1. Plant capacity constraints
        for plant in plants:
            for product in products:
                for week in future_weeks:
                    model += (
                        pulp.lpSum([
                            plant_to_wh[(plant, warehouse, product, week)]
                            for warehouse in warehouses
                        ]) <= plant_capacity.get((plant, product), 0),
                        f"Plant_Capacity_{plant}_{product}_{week}"
                    )
        
        # 2. Warehouse capacity constraints
        for warehouse in warehouses:
            for product in products:
                for week in future_weeks:
                    model += (
                        inventory[(warehouse, product, week)] <= warehouse_capacity.get((warehouse, product), 0),
                        f"Warehouse_Capacity_{warehouse}_{product}_{week}"
                    )
        
        # 3. Market demand constraints
        for market in markets:
            for product in products:
                for week in future_weeks:
                    model += (
                        pulp.lpSum([
                            wh_to_market[(warehouse, market, product, week)]
                            for warehouse in warehouses
                        ]) <= future_demand.get((market, product, week), 0),
                        f"Market_Demand_{market}_{product}_{week}"
                    )
        
        # 4. Inventory balance constraints
        for warehouse in warehouses:
            for product in products:
                # First week inventory balance
                model += (
                    inventory[(warehouse, product, future_weeks[0])] == 
                    initial_inventory.get((warehouse, product), 0) + 
                    pulp.lpSum([
                        plant_to_wh[(plant, warehouse, product, future_weeks[0])]
                        for plant in plants
                    ]) - 
                    pulp.lpSum([
                        wh_to_market[(warehouse, market, product, future_weeks[0])]
                        for market in markets
                    ]),
                    f"Inventory_Balance_{warehouse}_{product}_{future_weeks[0]}"
                )
                
                # Subsequent weeks inventory balance
                for week_idx in range(1, len(future_weeks)):
                    week = future_weeks[week_idx]
                    prev_week = future_weeks[week_idx - 1]
                    
                    model += (
                        inventory[(warehouse, product, week)] == 
                        inventory[(warehouse, product, prev_week)] + 
                        pulp.lpSum([
                            plant_to_wh[(plant, warehouse, product, week)]
                            for plant in plants
                        ]) - 
                        pulp.lpSum([
                            wh_to_market[(warehouse, market, product, week)]
                            for market in markets
                        ]),
                        f"Inventory_Balance_{warehouse}_{product}_{week}"
                    )
        
        # Solve the model
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)
        
        # Check solution status
        if model.status != pulp.LpStatusOptimal:
            print(f"Warning: Optimal solution not found. Status: {pulp.LpStatus[model.status]}")
        
        # Store the results
        self.optimization_results = {
            'status': pulp.LpStatus[model.status],
            'objective_value': pulp.value(model.objective),
            'plant_to_warehouse': {
                (plant, warehouse, product, week): pulp.value(plant_to_wh[(plant, warehouse, product, week)])
                for plant in plants
                for warehouse in warehouses
                for product in products
                for week in future_weeks
                if pulp.value(plant_to_wh[(plant, warehouse, product, week)]) > 0.01  # Filter out near-zero flows
            },
            'warehouse_to_market': {
                (warehouse, market, product, week): pulp.value(wh_to_market[(warehouse, market, product, week)])
                for warehouse in warehouses
                for market in markets
                for product in products
                for week in future_weeks
                if pulp.value(wh_to_market[(warehouse, market, product, week)]) > 0.01  # Filter out near-zero flows
            },
            'inventory': {
                (warehouse, product, week): pulp.value(inventory[(warehouse, product, week)])
                for warehouse in warehouses
                for product in products
                for week in future_weeks
            }
        }
        
        return self.optimization_results
    
    def get_optimization_summary(self, detailed=False):
        """
        Generate a summary of optimization results.
        
        Args:
            detailed (bool): Whether to include detailed flow information (default: False)
            
        Returns:
            dict: Summary statistics of optimization results
        """
        if self.optimization_results is None:
            return "No optimization results. Run optimize_network_flow() first."
        
        total_flow = sum(self.optimization_results['plant_to_warehouse'].values())
        total_warehouse_to_market = sum(self.optimization_results['warehouse_to_market'].values())
        
        summary = {
            'status': self.optimization_results['status'],
            'objective_value': self.optimization_results['objective_value'],
            'total_flow_from_plants': total_flow,
            'total_flow_to_markets': total_warehouse_to_market,
            'average_inventory': np.mean(list(self.optimization_results['inventory'].values()))
        }
        
        if detailed:
            # Add detailed flow breakdowns
            plant_flows = {}
            for (plant, _, _, _), flow in self.optimization_results['plant_to_warehouse'].items():
                plant_flows[plant] = plant_flows.get(plant, 0) + flow
            
            market_flows = {}
            for (_, market, _, _), flow in self.optimization_results['warehouse_to_market'].items():
                market_flows[market] = market_flows.get(market, 0) + flow
                
            summary['plant_flows'] = plant_flows
            summary['market_flows'] = market_flows
        
        return summary
    
    def visualize_network(self, time_period=None):
        """
        Visualize the supply chain network with flow volumes.
        
        Args:
            time_period (int, optional): Specific future week to visualize. If None, aggregates all periods.
            
        Returns:
            matplotlib.figure.Figure: Network visualization
        """
        if self.optimization_results is None:
            print("No optimization results. Run optimize_network_flow() first.")
            return None
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for plants, warehouses, and markets
        plants = set()
        warehouses = set()
        markets = set()
        
        # Extract all nodes from optimization results
        for (plant, warehouse, _, _) in self.optimization_results['plant_to_warehouse'].keys():
            plants.add(plant)
            warehouses.add(warehouse)
            
        for (warehouse, market, _, _) in self.optimization_results['warehouse_to_market'].keys():
            warehouses.add(warehouse)
            markets.add(market)
        
        # Add nodes to graph with positions
        # Position plants on the left, warehouses in the middle, and markets on the right
        pos = {}
        
        # Add plant nodes
        for i, plant in enumerate(plants):
            node_id = f"P:{plant}"
            G.add_node(node_id, type='plant')
            pos[node_id] = (0, i - len(plants)/2)
        
        # Add warehouse nodes
        for i, warehouse in enumerate(warehouses):
            node_id = f"W:{warehouse}"
            G.add_node(node_id, type='warehouse')
            pos[node_id] = (1, i - len(warehouses)/2)
        
        # Add market nodes
        for i, market in enumerate(markets):
            node_id = f"M:{market}"
            G.add_node(node_id, type='market')
            pos[node_id] = (2, i - len(markets)/2)
        
        # Add edges with weights based on flow volumes
        # Plant to warehouse flows
        for (plant, warehouse, product, week), flow in self.optimization_results['plant_to_warehouse'].items():
            if time_period is not None and week != time_period:
                continue
                
            source = f"P:{plant}"
            target = f"W:{warehouse}"
            
            if G.has_edge(source, target):
                G[source][target]['weight'] += flow
            else:
                G.add_edge(source, target, weight=flow)
        
        # Warehouse to market flows
        for (warehouse, market, product, week), flow in self.optimization_results['warehouse_to_market'].items():
            if time_period is not None and week != time_period:
                continue
                
            source = f"W:{warehouse}"
            target = f"M:{market}"
            
            if G.has_edge(source, target):
                G[source][target]['weight'] += flow
            else:
                G.add_edge(source, target, weight=flow)
        
        # Create the figure
        plt.figure(figsize=(12, 8))
        
        # Draw the nodes with different colors based on node type
        plant_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'plant']
        warehouse_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'warehouse']
        market_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'market']
        
        nx.draw_networkx_nodes(G, pos, nodelist=plant_nodes, node_color='lightblue', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=warehouse_nodes, node_color='lightgreen', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=market_nodes, node_color='salmon', node_size=500, alpha=0.8)
        
        # Draw the edges with width proportional to flow volume
        # Get the weights
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        # Normalize to get reasonable edge widths
        max_weight = max(weights) if weights else 1
        normalized_weights = [1 + 5 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.7, edge_color='gray', arrows=True, arrowsize=15)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Add a title
        title = f"Supply Chain Network Flow - Week {time_period}" if time_period else "Aggregated Supply Chain Network Flow"
        plt.title(title)
        
        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Plants'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Warehouses'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=10, label='Markets')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()