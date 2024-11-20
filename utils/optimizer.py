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

# Optimizer class
class Optimizer:
    def __init__(self, df):
        """
        Initializes the Optimizer with the given DataFrame.
        """
        self.df = df

    def create_safety_stock(self, service_level=0.95, method='z_score', date_col='week'):
        """
        Adds a safety stock column to self.df based on sell_in variability using different methods.
        """
        z_score = norm.ppf(service_level)
        weekly_sell_in_df = self.df.groupby(['warehouse', 'product', date_col])['sell_in'].sum().reset_index()
        policy_df = (
            weekly_sell_in_df.groupby(['product', 'warehouse'])['sell_in']
            .agg(['std', 'mean'])
            .rename(columns={'std': 'std_sell_in', 'mean': 'mean_sell_in'})
            .reset_index()
        )
        self.df = pd.merge(self.df, policy_df, on=['product', 'warehouse'], how='left')
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