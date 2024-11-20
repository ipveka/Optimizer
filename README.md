# 🛠️ Optimizer

**Optimizer** is a project for inventory optimization in supply chains using advanced simulation and machine learning techniques. It streamlines inventory management by providing tools for supply planning and inventory control. The project includes classes for each stage, from data preparation to evaluation, making it versatile for various supply chain scenarios.

This project uses **simulated data** to demonstrate its capabilities, providing a practical, scalable framework for real-world applications.

---

## 📁 Project Structure

The **Simulator** project is organized into the following folders for ease of navigation and use:

- **`data`**  
  Contains simulated datasets, both raw and processed. The processed datasets are prepped for modeling after data cleaning and feature engineering.

- **`docs`**  
  Comprehensive documentation detailing the project's classes, functions, architecture, and usage instructions.

- **`notebooks`**  
  Jupyter notebooks demonstrating the **Simulator** workflow:
  - **`simulation`**: Step-by-step walkthrough of simulation.
  - **`optimization`**: Demonstrates inventory modeling, optimization and evaluation.
  - **`runner`**: An end-to-end workflow combining all pipeline steps for a complete inventory optimization scenario.

- **`utils`**  
  Core utility folder containing classes, auxiliary functions, and visualization tools for the project:
  - **`simulator.py`**  
    Contains the **Simulator** class, which provides tools for generating and managing simulated supply chain scenarios.  
    - **Key features**:
      - Simulates supply, sell in, inventory and lead times,
      - Offers flexibility for testing different supply chain configurations.
  - **`optimizer.py`**  
    Houses the **Optimizer** class, which focuses on inventory modeling and optimization.  
    - **Key features**:
      - Runs optimization algorithms to minimize costs while ensuring adequate stock levels.
      - Supports scenario analysis for strategic decision-making.
  - **`plotter.py`**  
    Includes the **Plotter** class for creating insightful visualizations to analyze simulation and optimization results.  
    - **Key features**:
      - Generates demand and inventory level charts.
      - Visualizes optimization outcomes, such as cost savings or service level improvements.
      - Provides comparative plots for multiple scenarios.

---

This project is currently in development. If you have suggestions, feedback, or questions, please feel free to reach out!
