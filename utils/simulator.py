# Libraries
import pandas as pd
import numpy as np
import logging
import multiprocessing as mp
from functools import partial
from datetime import datetime, timedelta

# Simulator
class Simulator:
    """
    Simulator class for generating realistic supply chain scenarios and simulating inventory flows.
    
    This class provides tools to create test environments for validating optimization strategies
    without requiring actual historical data. It generates a comprehensive supply chain dataset 
    and simulates how products flow through the network.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing the simulated supply chain data
        logger (logging.Logger): Logger for simulation events and errors
        service_summary (pd.DataFrame): Summary of service level metrics, created after stockout calculations
    """
    
    def __init__(self, df=None, log_level=logging.INFO):
        """
        Initialize the Simulator with a DataFrame.
        
        Args:
            df (pd.DataFrame, optional): Input DataFrame. Creates an empty DataFrame if None.
            log_level (int): Logging level for simulator operations
        
        Initializes additional columns:
        - 'supply': Quantity of products supplied
        - 'sell_in': Quantity of products sold
        - 'inventory': Current inventory levels
        """
        # Use provided DataFrame or create an empty one
        self.df = df if df is not None else pd.DataFrame()
        
        # Initialize key columns with zero values if DataFrame is not empty
        if not self.df.empty and 'supply' not in self.df.columns:
            self.df['supply'] = 0
        if not self.df.empty and 'sell_in' not in self.df.columns:
            self.df['sell_in'] = 0
        if not self.df.empty and 'inventory' not in self.df.columns:
            self.df['inventory'] = 0
            
        # Set up logging
        self.setup_logging(log_level)
        
        # Mapping of distribution generators for simulation
        self.dist_generators = {
            'normal': lambda params, size: np.abs(np.random.normal(params[0], params[1], size)),
            'uniform': lambda params, size: np.random.uniform(params[0], params[1], size),
            'poisson': lambda params, size: np.random.poisson(params[0], size),
            'exponential': lambda params, size: np.random.exponential(params[0], size),
            'lognormal': lambda params, size: np.random.lognormal(params[0], params[1], size)
        }
        
        # Service level summary will be created when calculate_stockouts is called
        self.service_summary = None

    def setup_logging(self, log_level=logging.INFO):
        """
        Set up logging for the Simulator.
        
        Args:
            log_level (int): Logging level to use
        """
        self.logger = logging.getLogger(f"Simulator_{id(self)}")
        self.logger.setLevel(log_level)
        
        # Create handler if it doesn't exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.logger.info("Simulator initialized")

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
        self.logger.info(f"Generating scenarios with {len(plants)} plants, {len(warehouses)} warehouses, "
                        f"{len(markets)} markets, {len(products)} products, and {len(weeks)} weeks")
        
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
                                    'week': week,
                                    'supply': 0,
                                    'sell_in': 0,
                                    'inventory': 0
                                })
                        else:
                            self.logger.warning(f"Warehouse '{warehouse}' has no market in warehouse_market_map.")
        
        # Create DataFrame and sort for consistent ordering
        self.df = pd.DataFrame(data)
        self.df.sort_values(by=['product', 'warehouse', 'week'], inplace=True)
        
        self.logger.info(f"Generated scenario with {len(self.df)} rows")
        
        return self.df
        
    def simulate_parallel(self, num_processes=4):
        """
        Run simulation using parallel processing for large datasets.
        
        Args:
            num_processes (int): Number of parallel processes to use
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        self.logger.info(f"Running parallel simulation with {num_processes} processes")
        
        if len(self.df) < 10000:
            self.logger.warning("DataFrame is relatively small, parallel processing may not improve performance")
            
        # Split DataFrame into chunks for parallel processing
        df_split = np.array_split(self.df, num_processes)
        
        # Create a pool of processes
        pool = mp.Pool(processes=num_processes)
        
        # Define the processing function for each chunk
        def process_chunk(chunk, supply_params=(200, 50), sell_in_params=(150, 40)):
            # Create a temporary simulator for this chunk
            temp_sim = Simulator(chunk)
            
            # Simulate flows for this chunk
            temp_sim.simulate_flows(supply_params=supply_params, sell_in_params=sell_in_params)
            
            # Return the processed chunk
            return temp_sim.df
        
        # Process chunks in parallel
        results = pool.map(process_chunk, df_split)
        
        # Combine results
        self.df = pd.concat(results)
        
        # Clean up
        pool.close()
        pool.join()
        
        self.logger.info(f"Parallel simulation complete. DataFrame has {len(self.df)} rows")
        
        return self.df
        
    def add_date_dimension(self, start_date='2023-01-01', week_as_days=7):
        """
        Add date dimension based on week numbers.
        
        Args:
            start_date (str): Starting date in 'YYYY-MM-DD' format
            week_as_days (int): Number of days in a week
            
        Returns:
            pd.DataFrame: DataFrame with added date columns
        """
        self.logger.info(f"Adding date dimension starting from {start_date}")
        
        if 'week' not in self.df.columns:
            self.logger.error("Week column is required to add date dimension")
            raise ValueError("Week column is required to add date dimension")
            
        # Convert start_date to datetime
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Create a function to calculate date from week number
        def calculate_date(week):
            return start_date + timedelta(days=(week - 1) * week_as_days)
        
        # Apply the function to create date column
        self.df['date'] = self.df['week'].apply(calculate_date)
        
        # Add additional date components
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['quarter'] = self.df['date'].dt.quarter
        
        self.logger.info("Date dimension added with year, month, day, day_of_week, and quarter columns")
        
        return self.df
    
    def get_summary(self):
        """
        Generate a summary of inventory levels, supply, sell-in, and lead times.
        
        Returns:
            pd.DataFrame or str: Aggregated summary statistics or message if no valid columns
        """
        self.logger.info("Generating summary statistics")
        
        # Prepare aggregation dictionary dynamically
        agg_dict = {}
        for col in ['inventory', 'supply', 'sell_in', 'lead_time', 'backorders', 'stockout_units']:
            if col in self.df.columns:
                agg_dict[col] = ['min', 'max', 'mean', 'std']
        
        # Check if we have columns to aggregate
        if not agg_dict:
            message = "No valid columns for summary."
            self.logger.warning(message)
            return message
        
        # Generate the summary
        try:
            summary = self.df.groupby(['warehouse']).agg(agg_dict).round(2)
            self.logger.info("Summary statistics generated successfully")
            return summary
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
            
    def get_detailed_summary(self, group_by=['product', 'warehouse']):
        """
        Generate a more detailed summary with custom grouping.
        
        Args:
            group_by (list): Columns to group by
            
        Returns:
            pd.DataFrame: Detailed summary statistics
        """
        self.logger.info(f"Generating detailed summary grouped by {group_by}")
        
        # Verify all group_by columns exist
        missing_cols = [col for col in group_by if col not in self.df.columns]
        if missing_cols:
            self.logger.error(f"Missing columns for grouping: {missing_cols}")
            raise ValueError(f"Missing columns for grouping: {missing_cols}")
        
        # Prepare metrics for aggregation
        metrics = [
            'inventory', 'supply', 'sell_in', 'lead_time', 
            'backorders', 'stockout_units', 'service_level'
        ]
        
        # Filter to metrics that exist in the DataFrame
        existing_metrics = [col for col in metrics if col in self.df.columns]
        
        if not existing_metrics:
            self.logger.warning("No metrics columns available for detailed summary")
            return "No metrics columns available for detailed summary"
        
        # Define aggregations for each metric
        agg_dict = {metric: ['count', 'min', 'max', 'mean', 'std'] for metric in existing_metrics}
        
        # Generate the summary
        detailed_summary = self.df.groupby(group_by).agg(agg_dict).round(2)
        
        self.logger.info(f"Detailed summary generated with shape {detailed_summary.shape}")
        
        return detailed_summary
    
    def export_to_csv(self, filename='simulated_data.csv'):
        """
        Export the simulated data to a CSV file.
        
        Args:
            filename (str): Name of the output file
            
        Returns:
            str: Path to the saved file
        """
        self.logger.info(f"Exporting data to {filename}")
        
        try:
            self.df.to_csv(filename, index=False)
            self.logger.info(f"Successfully exported {len(self.df)} rows to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
            return f"Error: {str(e)}"

    def validate_data(self):
        """
        Validate the DataFrame structure and contents.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        # Check if DataFrame is empty
        if self.df.empty:
            self.logger.error("DataFrame is empty")
            return False
            
        # Check required columns
        required_columns = ['plant', 'warehouse', 'market', 'product', 'week']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for null values in key columns
        null_counts = self.df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            self.logger.warning(f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}")
            
        # Check data types
        if not pd.api.types.is_numeric_dtype(self.df['week']):
            self.logger.error("'week' column must be numeric")
            return False
        
        # Check unique combinations
        duplicate_rows = self.df.duplicated(subset=['plant', 'warehouse', 'market', 'product', 'week']).sum()
        if duplicate_rows > 0:
            self.logger.warning(f"Found {duplicate_rows} duplicate rows in key dimensions")
            
        return True

    def simulate_flows(self, supply_dist='normal', supply_params=(200, 50), sell_in_dist='normal', sell_in_params=(150, 40)):
        """
        Simulate supply and sell-in data using various probability distributions.
        
        Args:
            supply_dist (str): Distribution type for supply ('normal', 'uniform', 'poisson', 'exponential', 'lognormal')
            supply_params (tuple): Parameters for supply distribution
            sell_in_dist (str): Distribution type for sell-in ('normal', 'uniform', 'poisson', 'exponential', 'lognormal')
            sell_in_params (tuple): Parameters for sell-in distribution
        
        Returns:
            pd.DataFrame: DataFrame with simulated supply and sell-in columns
        """
        self.logger.info(f"Simulating flows using {supply_dist} distribution for supply and {sell_in_dist} for sell-in")
        
        # Validate distribution types
        if supply_dist not in self.dist_generators or sell_in_dist not in self.dist_generators:
            valid_dists = list(self.dist_generators.keys())
            self.logger.error(f"Invalid distribution. Supported: {valid_dists}")
            raise ValueError(f"Invalid distribution. Supported: {valid_dists}")
        
        # Generate supply data
        supply_generator = self.dist_generators[supply_dist]
        self.df['supply'] = np.round(supply_generator(supply_params, len(self.df)))
        
        # Generate sell-in data
        sell_in_generator = self.dist_generators[sell_in_dist]
        self.df['sell_in'] = np.round(sell_in_generator(sell_in_params, len(self.df)))
        
        self.logger.info(f"Simulation complete. Average supply: {self.df['supply'].mean():.2f}, "
                         f"Average sell-in: {self.df['sell_in'].mean():.2f}")
        
        return self.df
        
    def simulate_flows_vectorized(self, supply_dist='normal', supply_params=(200, 50), sell_in_dist='normal', sell_in_params=(150, 40)):
        """
        Vectorized version of simulate_flows for better performance with large datasets.
        
        Args:
            supply_dist (str): Distribution type for supply
            supply_params (tuple): Parameters for supply distribution
            sell_in_dist (str): Distribution type for sell-in
            sell_in_params (tuple): Parameters for sell-in distribution
        
        Returns:
            pd.DataFrame: DataFrame with simulated values
        """
        self.logger.info(f"Using vectorized flow simulation with {supply_dist} and {sell_in_dist} distributions")
        
        # Generate random data using numpy's vectorized operations
        n_rows = len(self.df)
        
        if supply_dist == 'normal':
            self.df['supply'] = np.abs(np.random.normal(supply_params[0], supply_params[1], n_rows))
        elif supply_dist == 'uniform':
            self.df['supply'] = np.random.uniform(supply_params[0], supply_params[1], n_rows)
        elif supply_dist == 'poisson':
            self.df['supply'] = np.random.poisson(supply_params[0], n_rows)
        elif supply_dist == 'exponential':
            self.df['supply'] = np.random.exponential(supply_params[0], n_rows)
        elif supply_dist == 'lognormal':
            self.df['supply'] = np.random.lognormal(supply_params[0], supply_params[1], n_rows)
        else:
            self.logger.error(f"Unsupported supply distribution: {supply_dist}")
            raise ValueError(f"Unsupported supply distribution: {supply_dist}")
        
        # Similar vectorized approach for sell_in
        if sell_in_dist == 'normal':
            self.df['sell_in'] = np.abs(np.random.normal(sell_in_params[0], sell_in_params[1], n_rows))
        elif sell_in_dist == 'uniform':
            self.df['sell_in'] = np.random.uniform(sell_in_params[0], sell_in_params[1], n_rows)
        elif sell_in_dist == 'poisson':
            self.df['sell_in'] = np.random.poisson(sell_in_params[0], n_rows)
        elif sell_in_dist == 'exponential':
            self.df['sell_in'] = np.random.exponential(sell_in_params[0], n_rows)
        elif sell_in_dist == 'lognormal':
            self.df['sell_in'] = np.random.lognormal(sell_in_params[0], sell_in_params[1], n_rows)
        else:
            self.logger.error(f"Unsupported sell-in distribution: {sell_in_dist}")
            raise ValueError(f"Unsupported sell-in distribution: {sell_in_dist}")
        
        # Round values
        self.df['supply'] = np.round(self.df['supply'])
        self.df['sell_in'] = np.round(self.df['sell_in'])
        
        self.logger.info(f"Vectorized simulation complete. Average supply: {self.df['supply'].mean():.2f}, "
                       f"Average sell-in: {self.df['sell_in'].mean():.2f}")
        
        return self.df

    def simulate_seasonal_demand(self, base_params=(150, 40), seasonality_factor=0.3, peak_weeks=[13, 26, 39, 52], cycle_length=52):
        """
        Simulate demand with seasonal patterns.
        
        Args:
            base_params (tuple): Base parameters for demand distribution (mean, std)
            seasonality_factor (float): Amplitude of seasonality (0-1)
            peak_weeks (list): Weeks with peak demand within the cycle
            cycle_length (int): Length of a complete seasonal cycle in weeks
            
        Returns:
            pd.DataFrame: DataFrame with seasonally adjusted sell_in
        """
        self.logger.info(f"Simulating seasonal demand with factor {seasonality_factor}")
        
        if 'week' not in self.df.columns:
            self.logger.error("Week column is required for seasonal simulation")
            raise ValueError("Week column is required for seasonal simulation")
            
        # Generate baseline demand
        self.df['sell_in'] = np.random.normal(base_params[0], base_params[1], len(self.df))
        
        # Apply seasonal adjustment - vectorized implementation
        weeks = self.df['week'].values
        
        # Calculate distance to nearest peak for each week
        distances = np.array([
            [abs((week - peak) % cycle_length) for peak in peak_weeks]
            for week in weeks
        ])
        min_distances = np.min(distances, axis=1)
        max_distance = cycle_length / 2
        
        # Calculate seasonal factor (-1 to 1 scale, with 1 at peaks)
        seasonal_positions = 1 - (min_distances / max_distance) * 2
        
        # Apply seasonality adjustment
        self.df['sell_in'] *= (1 + seasonal_positions * seasonality_factor)
        
        # Round values
        self.df['sell_in'] = np.round(self.df['sell_in'])
        
        self.logger.info(f"Seasonal demand simulation complete. Average demand: {self.df['sell_in'].mean():.2f}")
        
        return self.df
        
    def simulate_correlated_products(self, product_groups, correlation_strength=0.7):
        """
        Simulate correlated demand across product groups.
        
        Args:
            product_groups (dict): Dictionary mapping group names to lists of products
            correlation_strength (float): Strength of correlation between products in same group (0-1)
            
        Returns:
            pd.DataFrame: DataFrame with correlated sell_in values across products
        """
        self.logger.info(f"Simulating correlated product demand with strength {correlation_strength}")
        
        # Verify that all products in product_groups are in the DataFrame
        all_products = set(self.df['product'].unique())
        for group, products in product_groups.items():
            missing_products = set(products) - all_products
            if missing_products:
                self.logger.warning(f"Products {missing_products} in group {group} not found in DataFrame")
        
        # For each week and market, generate correlated demand
        for week in self.df['week'].unique():
            for market in self.df['market'].unique():
                for group_name, products in product_groups.items():
                    # Skip if no products in this group
                    if not products:
                        continue
                        
                    # Get all rows for this market, week, and product group
                    mask = (
                        (self.df['week'] == week) & 
                        (self.df['market'] == market) & 
                        (self.df['product'].isin(products))
                    )
                    
                    # Skip if no matching rows
                    if not self.df[mask].any().any():
                        continue
                    
                    # Get current mean and std of demand
                    mean_demand = self.df.loc[mask, 'sell_in'].mean()
                    std_demand = self.df.loc[mask, 'sell_in'].std()
                    
                    # If std is 0 or NaN, use a default value
                    if pd.isna(std_demand) or std_demand == 0:
                        std_demand = mean_demand * 0.1  # Use 10% of mean as default
                        
                    # Create correlation matrix for multivariate normal
                    num_products = len(products)
                    corr_matrix = np.ones((num_products, num_products)) * correlation_strength
                    np.fill_diagonal(corr_matrix, 1.0)
                    
                    # Convert correlation matrix to covariance matrix
                    cov_matrix = corr_matrix * (std_demand ** 2)
                    
                    # Generate correlated random values
                    correlated_demands = np.random.multivariate_normal(
                        mean=np.ones(num_products) * mean_demand,
                        cov=cov_matrix,
                        size=1
                    )[0]
                    
                    # Ensure non-negative values
                    correlated_demands = np.maximum(correlated_demands, 0)
                    
                    # Assign correlated demands to products
                    for i, product in enumerate(products):
                        product_mask = mask & (self.df['product'] == product)
                        if product_mask.any():
                            self.df.loc[product_mask, 'sell_in'] = round(correlated_demands[i])
        
        self.logger.info("Correlated product demand simulation complete")
        return self.df

    def simulate_disruption(self, start_week, duration, affected_plants=None, affected_warehouses=None, impact_severity=0.8):
        """
        Simulate a supply chain disruption event.
        
        Args:
            start_week (int): Week when disruption begins
            duration (int): Duration of disruption in weeks
            affected_plants (list): List of plants affected by the disruption
            affected_warehouses (list): List of warehouses affected by the disruption
            impact_severity (float): Severity factor (0-1) to reduce capacity
            
        Returns:
            pd.DataFrame: DataFrame with disruption effects applied
        """
        self.logger.info(f"Simulating disruption from week {start_week} for {duration} weeks "
                        f"with severity {impact_severity}")
        
        # Identify affected rows
        mask = ((self.df['week'] >= start_week) & 
                (self.df['week'] < start_week + duration))
        
        if affected_plants:
            mask = mask & self.df['plant'].isin(affected_plants)
            self.logger.info(f"Disruption affects plants: {affected_plants}")
            
        if affected_warehouses:
            mask = mask & self.df['warehouse'].isin(affected_warehouses)
            self.logger.info(f"Disruption affects warehouses: {affected_warehouses}")
        
        # Store pre-disruption values for reporting
        pre_disruption_supply = self.df.loc[mask, 'supply'].sum() if mask.any() else 0
        
        # Apply disruption effect (reduce supply)
        self.df.loc[mask, 'supply'] = self.df.loc[mask, 'supply'] * (1 - impact_severity)
        
        # Round values
        self.df['supply'] = np.round(self.df['supply'])
        
        # Report impact
        post_disruption_supply = self.df.loc[mask, 'supply'].sum() if mask.any() else 0
        supply_reduction = pre_disruption_supply - post_disruption_supply
        
        self.logger.info(f"Disruption applied. Supply reduced by {supply_reduction} units "
                      f"({100 * supply_reduction / pre_disruption_supply:.1f}% of affected supply)")
        
        return self.df

    def calculate_inventory(self, initial_inventory=1000):
        """
        Calculate rolling inventory for each product-warehouse combination.
        
        Args:
            initial_inventory (int): Starting inventory level (default: 1000)
        
        Returns:
            pd.DataFrame: DataFrame with calculated inventory levels
        """
        self.logger.info(f"Calculating inventory with initial level {initial_inventory}")
        
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
        
        self.logger.info(f"Inventory calculation complete. Average ending inventory: {self.df.groupby(['product', 'warehouse'])['inventory'].last().mean():.2f}")
        
        return self.df
    
    def calculate_realistic_inventory(self, initial_inventory=1000, warehouse_capacity=None, allow_backorders=True):
        """
        Calculate inventory with warehouse capacity constraints and backorder handling.
        
        Args:
            initial_inventory (int): Starting inventory level
            warehouse_capacity (dict): Maximum capacity for each warehouse
            allow_backorders (bool): Whether to allow negative inventory (backorders)
            
        Returns:
            pd.DataFrame: DataFrame with realistic inventory calculations
        """
        self.logger.info(f"Calculating realistic inventory with constraints. Backorders allowed: {allow_backorders}")
        
        # Sort to ensure correct sequence
        self.df = self.df.sort_values(['product', 'warehouse', 'week'])
        
        # Initialize backorder column if we're tracking them
        if allow_backorders:
            self.df['backorders'] = 0
            
        # Initialize excess inventory column for tracking capacity violations
        if warehouse_capacity:
            self.df['excess_inventory'] = 0
        
        # Calculate inventory for each product-warehouse combination
        for (product, warehouse), group in self.df.groupby(['product', 'warehouse']):
            indices = group.index
            current_inventory = initial_inventory
            current_backorders = 0
            
            # Get capacity for this warehouse if specified
            capacity = None
            if warehouse_capacity and warehouse in warehouse_capacity:
                capacity = warehouse_capacity[warehouse]
            
            for idx in indices:
                # Add new supply
                current_inventory += self.df.loc[idx, 'supply']
                
                # Fulfill backorders first if they exist
                if allow_backorders and current_backorders > 0:
                    backorder_fulfillment = min(current_inventory, current_backorders)
                    current_inventory -= backorder_fulfillment
                    current_backorders -= backorder_fulfillment
                    # Update backorder column
                    self.df.loc[idx, 'backorders'] = current_backorders
                
                # Apply capacity constraint if specified
                if capacity is not None:
                    # If inventory exceeds capacity, reduce it
                    if current_inventory > capacity:
                        # Record the excessive inventory as lost or redirected
                        self.df.loc[idx, 'excess_inventory'] = current_inventory - capacity
                        current_inventory = capacity
                
                # Process demand (sell_in)
                demand = self.df.loc[idx, 'sell_in']
                
                if current_inventory >= demand:
                    # Enough inventory to meet demand
                    current_inventory -= demand
                else:
                    # Not enough inventory
                    unfulfilled = demand - current_inventory
                    
                    if allow_backorders:
                        # Record backorders
                        current_backorders += unfulfilled
                        self.df.loc[idx, 'backorders'] = current_backorders
                        current_inventory = 0
                    else:
                        # Record stockout but don't create backorders
                        self.df.loc[idx, 'stockout_units'] = unfulfilled if 'stockout_units' in self.df.columns else unfulfilled
                        current_inventory = 0
                
                # Update inventory
                self.df.loc[idx, 'inventory'] = current_inventory
        
        # Log results
        if allow_backorders:
            total_backorders = self.df['backorders'].sum()
            self.logger.info(f"Realistic inventory calculation complete. Total backorders: {total_backorders}")
            
        if warehouse_capacity:
            total_excess = self.df['excess_inventory'].sum()
            self.logger.info(f"Total excess inventory (over capacity): {total_excess}")
        
        return self.df
        
    def calculate_stockouts(self):
        """
        Calculate stockouts and service level metrics.
        
        Returns:
            pd.DataFrame: DataFrame with added stockout metrics
        """
        self.logger.info("Calculating stockout metrics")
        
        # Sort to ensure correct inventory calculation
        self.df = self.df.sort_values(['product', 'warehouse', 'week'])
        
        # Initialize stockout columns
        self.df['stockout_units'] = 0
        self.df['service_level'] = 1.0
        
        # Calculate stockouts for each product-warehouse combination
        for (product, warehouse), group in self.df.groupby(['product', 'warehouse']):
            indices = group.index
            
            for i, idx in enumerate(indices):
                demand = self.df.loc[idx, 'sell_in']
                inventory = self.df.loc[idx, 'inventory']
                
                # If demand exceeds inventory, record stockout
                if demand > inventory:
                    self.df.loc[idx, 'stockout_units'] = demand - inventory
                    self.df.loc[idx, 'service_level'] = inventory / demand if demand > 0 else 1.0
        
        # Calculate fill rate (percentage of demand met)
        total_demand = self.df.groupby(['product', 'warehouse'])['sell_in'].sum()
        total_stockout = self.df.groupby(['product', 'warehouse'])['stockout_units'].sum()
        
        # Create a summary DataFrame
        service_summary = pd.DataFrame({
            'total_demand': total_demand,
            'total_stockout': total_stockout,
            'fill_rate': 1 - (total_stockout / total_demand)
        }).reset_index()
        
        self.service_summary = service_summary
        
        # Log results
        overall_fill_rate = 1 - (total_stockout.sum() / total_demand.sum())
        self.logger.info(f"Stockout calculation complete. Overall fill rate: {overall_fill_rate:.4f}")
        
        return self.df

    def simulate_lead_times(self, scenario_group=['plant', 'warehouse', 'market', 'product'], lead_time_dist='uniform', lead_time_params=(3, 10)):
        """
        Assign random lead times to unique combinations of specified scenario groups.
        
        Args:
            scenario_group (list): Dimensions to group by for unique lead time assignment
            lead_time_dist (str): Distribution type for lead times ('normal', 'uniform', 'poisson')
            lead_time_params (tuple): Parameters for lead time distribution
        
        Returns:
            pd.DataFrame: DataFrame with added lead time column
        """
        self.logger.info(f"Simulating lead times using {lead_time_dist} distribution")
        
        # Validate distribution types
        if lead_time_dist not in self.dist_generators:
            valid_dists = list(self.dist_generators.keys())
            self.logger.error(f"Invalid distribution. Supported: {valid_dists}")
            raise ValueError(f"Invalid distribution. Supported: {valid_dists}")
        
        # Get unique combinations based on specified scenario groups
        unique_combinations = self.df[scenario_group].drop_duplicates()
        
        # Generate lead times using specified distribution
        lead_time_generator = self.dist_generators[lead_time_dist]
        unique_combinations['lead_time'] = lead_time_generator(lead_time_params, len(unique_combinations))
        
        # Ensure lead times are integers and positive
        unique_combinations['lead_time'] = np.maximum(1, np.round(unique_combinations['lead_time'])).astype(int)
        
        # Merge lead times back to the main DataFrame
        self.df = pd.merge(
            self.df, 
            unique_combinations, 
            on=scenario_group, 
            how='left'
        )
        
        # Now safely log the average lead time after the merge is complete
        self.logger.info(f"Lead time simulation complete. Average lead time: {self.df['lead_time'].mean():.2f}")
        
        return self.df
        
    def add_product_attributes(self, attributes_dict):
        """
        Add product attributes like cost, weight, volume, etc.
        
        Args:
            attributes_dict (dict): Dictionary of product attributes
                Format: {
                    'attribute_name': {
                        'distribution': 'normal',
                        'params': (mean, std),
                        'min_value': min_allowed,
                        'max_value': max_allowed
                    }
                }
                
        Returns:
            pd.DataFrame: DataFrame with added product attributes
        """
        self.logger.info(f"Adding product attributes: {list(attributes_dict.keys())}")
        
        # Get unique products
        products = self.df['product'].unique()
        
        # Create a DataFrame for product attributes
        product_attrs = pd.DataFrame({'product': products})
        
        # Generate attributes for each product
        for attr_name, attr_config in attributes_dict.items():
            dist_name = attr_config.get('distribution', 'normal')
            params = attr_config.get('params', (0, 1))
            min_val = attr_config.get('min_value', None)
            max_val = attr_config.get('max_value', None)
            
            # Validate distribution
            if dist_name not in self.dist_generators:
                self.logger.warning(f"Invalid distribution {dist_name} for {attr_name}. Using normal instead.")
                dist_name = 'normal'
            
            # Generate values
            values = self.dist_generators[dist_name](params, len(products))
            
            # Apply min/max constraints
            if min_val is not None:
                values = np.maximum(values, min_val)
            if max_val is not None:
                values = np.minimum(values, max_val)
                
            # Add to product attributes DataFrame
            product_attrs[attr_name] = values
        
        # Merge attributes to main DataFrame
        self.df = pd.merge(self.df, product_attrs, on='product', how='left')
        
        self.logger.info(f"Product attributes added to DataFrame")
        
        return self.df
        
    def add_transportation_costs(self, base_cost=10, distance_factor=0.1):
        """
        Add transportation costs based on simulated distances between nodes.
        
        Args:
            base_cost (float): Base cost of transportation
            distance_factor (float): Cost multiplier per unit of distance
            
        Returns:
            pd.DataFrame: DataFrame with added transportation costs
        """
        """
        Add transportation costs based on simulated distances between nodes.
        
        Args:
            base_cost (float): Base cost of transportation
            distance_factor (float): Cost multiplier per unit of distance
            
        Returns:
            pd.DataFrame: DataFrame with added transportation costs
        """
        self.logger.info("Adding transportation costs to supply chain network")
        
        # Generate random coordinates for each location
        plants = self.df['plant'].unique()
        warehouses = self.df['warehouse'].unique()
        markets = self.df['market'].unique()
        
        # Create location coordinates
        locations = {}
        
        # Place plants, warehouses, and markets in a grid-like structure
        # to ensure realistic transportation distances
        for i, plant in enumerate(plants):
            # Place plants on the left side of the grid
            locations[plant] = (0, i * 10)
            
        for i, warehouse in enumerate(warehouses):
            # Place warehouses in the middle of the grid
            locations[warehouse] = (50, i * 10)
            
        for i, market in enumerate(markets):
            # Place markets on the right side of the grid
            locations[market] = (100, i * 10)
            
        # Calculate distances between locations
        distances = {}
        
        # Plant to warehouse distances
        for plant in plants:
            for warehouse in warehouses:
                plant_pos = locations[plant]
                wh_pos = locations[warehouse]
                # Euclidean distance
                distance = np.sqrt((plant_pos[0] - wh_pos[0])**2 + (plant_pos[1] - wh_pos[1])**2)
                distances[(plant, warehouse)] = distance
                
        # Warehouse to market distances
        for warehouse in warehouses:
            for market in markets:
                wh_pos = locations[warehouse]
                market_pos = locations[market]
                # Euclidean distance
                distance = np.sqrt((wh_pos[0] - market_pos[0])**2 + (wh_pos[1] - market_pos[1])**2)
                distances[(warehouse, market)] = distance
                
        # Create transportation costs
        transport_costs_plant_wh = []
        for _, row in self.df.drop_duplicates(['plant', 'warehouse']).iterrows():
            plant = row['plant']
            warehouse = row['warehouse']
            distance = distances.get((plant, warehouse), 0)
            transport_cost = base_cost + distance * distance_factor
            transport_costs_plant_wh.append({
                'plant': plant,
                'warehouse': warehouse,
                'distance': distance,
                'transport_cost_plant_wh': transport_cost
            })
            
        transport_costs_wh_market = []
        for _, row in self.df.drop_duplicates(['warehouse', 'market']).iterrows():
            warehouse = row['warehouse']
            market = row['market']
            distance = distances.get((warehouse, market), 0)
            transport_cost = base_cost + distance * distance_factor
            transport_costs_wh_market.append({
                'warehouse': warehouse,
                'market': market,
                'distance': distance,
                'transport_cost_wh_market': transport_cost
            })
            
        # Convert to DataFrames for merging
        plant_wh_costs_df = pd.DataFrame(transport_costs_plant_wh)
        wh_market_costs_df = pd.DataFrame(transport_costs_wh_market)
        
        # Merge costs to main DataFrame
        self.df = pd.merge(self.df, plant_wh_costs_df, on=['plant', 'warehouse'], how='left')
        self.df = pd.merge(self.df, wh_market_costs_df, on=['warehouse', 'market'], how='left')
        
        self.logger.info(f"Transportation costs added. Average plant-warehouse cost: {self.df['transport_cost_plant_wh'].mean():.2f}, "
                         f"Average warehouse-market cost: {self.df['transport_cost_wh_market'].mean():.2f}")
        
        return self.df