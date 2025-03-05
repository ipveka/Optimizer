# Optimizer: Supply Chain Optimization Library

## Overview

**Optimizer** is a comprehensive Python library for supply chain network optimization, inventory management, and visual analytics. The library provides tools for simulating realistic supply chain scenarios, calculating optimal inventory policies, and optimizing the flow of goods from plants through warehouses to markets.

The library consists of three main components:
- **Simulator**: Generates realistic supply chain scenarios and simulates inventory flows
- **Optimizer**: Calculates optimal inventory policies and optimizes network flows
- **Plotter**: Visualizes supply chain networks, inventory profiles, and optimization results

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/optimizer.git
cd optimizer

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for the Streamlit app
pip install streamlit xlsxwriter
```

## Key Components

### 1. Simulator

The `Simulator` class generates a comprehensive supply chain dataset and simulates product flows. It creates a test environment for validating optimization strategies without requiring actual historical data.

#### Key Features:
- Generate complete supply chain scenarios with customizable dimensions
- Simulate supply and demand patterns with various probability distributions
- Calculate inventory levels based on supply and demand
- Simulate lead times across the supply chain network
- Provide summary statistics for quick analysis

#### Example Usage:

```python
from optimizer.utils.simulator import Simulator

# Define supply chain dimensions
plants = ['PlantA', 'PlantB']
warehouses = ['WarehouseX', 'WarehouseY', 'WarehouseZ']
markets = ['MarketNorth', 'MarketSouth', 'MarketEast', 'MarketWest']
products = ['ProductAlpha', 'ProductBeta']
weeks = list(range(1, 53))  # 52 weeks

# Define warehouse-market mappings
warehouse_market_map = {
    'WarehouseX': ['MarketNorth', 'MarketEast'],
    'WarehouseY': ['MarketSouth'],
    'WarehouseZ': ['MarketWest']
}

# Initialize and run simulation
simulator = Simulator()
df = simulator.generate_scenarios(
    plants=plants, 
    warehouses=warehouses, 
    markets=markets, 
    products=products, 
    weeks=weeks, 
    warehouse_market_map=warehouse_market_map
)

# Simulate supply and demand patterns
simulator.simulate_flows(
    supply_dist='normal', supply_params=(200, 50),
    sell_in_dist='normal', sell_in_params=(150, 40)
)

# Calculate inventory and simulate lead times
simulator.calculate_inventory(initial_inventory=1000)
simulator.simulate_lead_times(
    lead_time_dist='uniform', 
    lead_time_params=(3, 10)
)

# Get summary statistics
summary = simulator.get_summary()
print(summary)
```

### 2. Optimizer

The `Optimizer` class implements advanced inventory optimization and network flow optimization algorithms to minimize costs while maintaining service levels.

#### Key Features:
- Calculate safety stock requirements based on service levels and demand variability
- Determine optimal reorder points considering lead times
- Calculate economic order quantities (EOQ) to balance ordering and holding costs
- Optimize network flow from plants to warehouses to markets
- Support multiple optimization objectives (cost minimization or service level maximization)
- Provide comprehensive optimization summaries and reports

#### Example Usage:

```python
from optimizer.utils.optimizer import Optimizer

# Initialize optimizer with simulator data
optimizer = Optimizer(simulator.df)

# Calculate safety stock based on service level
optimizer.create_safety_stock(
    service_level=0.95,
    method='demand_variability'
)

# Calculate reorder points and order quantities
optimizer.calculate_reorder_point()
optimizer.calculate_order_quantity(
    holding_cost=0.2,
    ordering_cost=100,
    method='eoq'
)

# Optimize network flow for the next 4 weeks
results = optimizer.optimize_network_flow(
    horizon=4,
    objective='min_cost'
)

# Get optimization summary
summary = optimizer.get_optimization_summary(detailed=True)
print(summary)

# Visualize the optimization network
network_fig = optimizer.visualize_network()
network_fig.show()
```

### 3. Plotter

The `Plotter` class provides visualization capabilities for supply chain networks, inventory profiles, and optimization results.

#### Key Features:
- Visualize the supply chain network with flow volumes and key metrics
- Create heatmaps to identify bottlenecks and optimization opportunities
- Display inventory time series with safety stock and reorder points
- Compare pre-optimization and post-optimization scenarios
- Support filtering by product, warehouse, or time period
- Generate professional and customizable visualizations
- High-readability node sizing with clear labels
- Consistent figure handling to prevent duplicate displays

#### Example Usage:

```python
from optimizer.utils.plotter import Plotter
import matplotlib.pyplot as plt

# Initialize plotter with optimizer data
plotter = Plotter(optimizer.df)

# Plot the supply chain network
network_fig = plotter.plot_network(
    product='ProductAlpha',
    week=10,
    layout='custom',
    node_size_metric='inventory',
    edge_width_metric='flow',
    show_labels=True
)

# Display or save the figure
network_fig.savefig('network.png', dpi=300, bbox_inches='tight')
# plt.show()  # Uncomment to display interactively

# Plot multiple networks with additional information
multiple_networks_fig = plotter.plot_multiple_networks(
    show_inventory=True,
    show_safety_stock=True,
    show_lead_time=True
)
multiple_networks_fig.savefig('multiple_networks.png')

# Create an inventory time series visualization
inventory_fig = plotter.plot_inventory_time_series(
    warehouses=['WarehouseX', 'WarehouseY'],
    products=['ProductAlpha'],
    metrics=['inventory', 'safety_stock', 'reorder_point']
)
inventory_fig.savefig('inventory_time_series.png')

# Generate a heatmap visualization
heatmap_fig = plotter.plot_inventory_heatmap(
    metric='inventory',
    groupby=['warehouse', 'product']
)
heatmap_fig.savefig('inventory_heatmap.png')

# Compare before and after optimization
comparison_fig = plotter.plot_optimization_comparison(
    before_df=before_optimization_df,
    after_df=after_optimization_df,
    metrics=['inventory', 'supply'],
    groupby='warehouse'
)
comparison_fig.savefig('optimization_comparison.png')
```

## Utility Modules

The project includes several utility modules to facilitate supply chain optimization and analysis:

### 1. utils.py

This module contains utility functions for the end-to-end supply chain optimization process:

- **Data conversion**: Safely converts complex data types (including NumPy types) to JSON-serializable formats
- **Optimization application**: Applies optimization results to create an optimized scenario
- **KPI calculation**: Computes key performance indicators such as inventory levels, service levels, and costs
- **Improvement measurement**: Calculates improvements between base and optimized scenarios
- **Visualization generation**: Creates comparative visualizations and dashboards

#### Location: `utils/utils.py`

### 2. app_utils.py

Utilities specifically designed for the Streamlit web application:

- **Data download**: Creates Excel download links for simulation and optimization results
- **Simulation workflow**: Functions to run the supply chain simulation with UI parameters
- **Optimization workflow**: Functions to apply optimization algorithms and measure results
- **Visualization generation**: Creates interactive visualizations for the web interface
- **Data preparation**: Functions to prepare comparison dataframes for warehouse and product analysis

#### Location: `utils/app_utils.py`

### End-to-End Example

The repository includes an end-to-end example script that demonstrates the complete workflow from simulation to optimization to visualization:

```python
python src/end_2_end.py
```

#### Features of the end-to-end example:

- Creates a simulated supply chain with configurable dimensions
- Applies inventory optimization algorithms
- Generates visualizations to compare before and after optimization
- Organizes results in a structured directory format:
  - `results/supply_chain_optimization_{timestamp}/`
    - `data/`: Contains raw CSV files with simulation and optimization data
    - `params/`: Stores JSON files with all configuration parameters
    - `plots/`: Includes all visualization files (PNG format)
    - `results/`: Contains summary files with KPIs and metrics
    - `supply_chain_optimization_report.html`: Complete HTML report summarizing the results

### Streamlit Web Application

The repository also includes an interactive Streamlit web application for supply chain optimization:

```bash
# Install streamlit if not already installed
pip install streamlit

# Run the app
streamlit run app.py
```

#### Features of the Streamlit app:

- Interactive UI for configuring all supply chain parameters
- Live simulation and optimization
- Visual results with multiple tabs:
  - Simulation Results: View supply chain data and summary statistics
  - Optimization Results: See improvements and comparison metrics
  - Visualizations: Interactive network, inventory time series, and KPI dashboard
  - Download Data: Export all results in Excel format
- Ability to customize and re-run the simulation and optimization

## Advanced Usage

### End-to-End Workflow

Here's an example of an end-to-end workflow using the Optimizer library:

```python
import pandas as pd
from optimizer.utils.simulator import Simulator
from optimizer.utils.optimizer import Optimizer
from optimizer.utils.plotter import Plotter

# 1. Generate simulated data
simulator = Simulator()
df = simulator.generate_scenarios(plants, warehouses, markets, products, weeks, warehouse_market_map)
simulator.simulate_flows()
simulator.calculate_inventory()
simulator.simulate_lead_times()

# 2. Optimize inventory policies
optimizer = Optimizer(simulator.df)
optimizer.create_safety_stock(service_level=0.95, method='demand_variability')
optimizer.calculate_reorder_point()
optimizer.calculate_order_quantity(method='eoq')

# 3. Optimize network flow
results = optimizer.optimize_network_flow(horizon=4, objective='min_cost')

# 4. Visualize results
plotter = Plotter(optimizer.df)
network_fig = plotter.plot_multiple_networks(show_inventory=True, show_safety_stock=True)
network_fig.savefig('network_visualization.png', dpi=300)

timeseries_fig = plotter.plot_inventory_time_series(
    metrics=['inventory', 'safety_stock', 'reorder_point']
)
timeseries_fig.savefig('inventory_analysis.png', dpi=300)
```

### Using Real Data

While the library includes a robust simulator, it can also work with real supply chain data:

```python
import pandas as pd
from optimizer.utils.optimizer import Optimizer
from optimizer.utils.plotter import Plotter

# Load real supply chain data
df = pd.read_csv('your_supply_chain_data.csv')

# Ensure the data has the required columns
required_columns = [
    'plant', 'warehouse', 'market', 'product', 'week',
    'supply', 'sell_in', 'inventory', 'lead_time'
]

# Run the optimizer on real data
optimizer = Optimizer(df)
optimizer.create_safety_stock()
optimizer.optimize_network_flow()

# Visualize the real supply chain
plotter = Plotter(df)
plotter.plot_network()
```

## Customization and Extension

The Optimizer library is designed to be highly extensible. Here are some ways to customize and extend its functionality:

### Custom Distribution Functions

You can extend the `Simulator` class with additional probability distributions:

```python
# Add custom distribution to the Simulator class
def custom_distribution(params, size):
    # Implement your custom distribution logic
    return custom_values

# Add the custom distribution to the available distributions
simulator.dist_generators['custom'] = custom_distribution
simulator.simulate_flows(supply_dist='custom', supply_params=your_params)
```

### Custom Optimization Objectives

You can add custom optimization objectives to the `Optimizer` class:

```python
# Define a custom objective function for network flow optimization
def minimize_lead_time_objective(model, variables, network_data):
    # Create an objective that prioritizes routes with shorter lead times
    weighted_flows = [
        variables['flow'][i, j, p, t] * network_data['lead_times'][i, j]
        for i, j, p, t in variables['flow']
    ]
    return pulp.lpSum(weighted_flows)

# Use in optimization
optimizer.optimize_network_flow(custom_objective=minimize_lead_time_objective)
```

## Best Practices

1. **Data Preparation**
   - Ensure your data has all required columns
   - Clean and validate data before optimization
   - Handle missing values appropriately

2. **Simulation Parameters**
   - Base distribution parameters on historical data when available
   - Validate simulation outputs against expected patterns
   - Perform sensitivity analysis to understand parameter impact

3. **Optimization Settings**
   - Set service levels to match business requirements
   - Balance inventory costs against service level objectives
   - Consider practical constraints like warehouse capacity and lead times

4. **Visualization**
   - Filter complex networks to focus on key products or locations
   - Use different visualization types for different audiences
   - Create clear visual comparisons between current and optimized states
   - Save figures to files rather than displaying them directly to avoid duplicate rendering
   - Adjust node sizes and font sizes based on the amount of text and visualization complexity

5. **End-to-End Example**
   - The end-to-end example saves all results in a timestamped directory within the `results` folder
   - Examine the HTML report for a comprehensive summary of the optimization results
   - Compare the visualization images to understand the impact of optimization
   - Review the detailed KPI statistics to quantify improvements

6. **Streamlit Application**
   - Use the app for quick experimentation and parameter tuning
   - Download results for detailed analysis in external tools
   - The interactive visualizations help communicate results to stakeholders

## Contributing

Contributions to the Optimizer library are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Add your code with appropriate tests
4. Submit a pull request with a clear description of your changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.