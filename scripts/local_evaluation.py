#! /usr/bin/env python3

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
)

"""
This script aims to provide you a demonstration of how the scores are actually calculated. 
"""

########################################################
# WarmUp Round: Classification Task
########################################################

"""
Lets assume that we want to only focus on SiteA in the Training set, 
and want to calculate the scores for SiteA against some random predictions.
"""

# Load Training Data
training_df = pd.read_csv("../data/classification/classification-train.csv")
# Filter out only SiteA data
training_df = training_df[training_df["Site"] == "siteA"]

# Gather the ground truth demand response flags for siteA
ground_truth_demand_response_flags = training_df["Demand_Response_Flag"]

# Generate random predictions
random_predictions_demand_response_flags = np.random.choice(
    [-1, 0, 1],
    size=len(training_df),
)

# Calculate the geometric mean score
geometric_mean_score_value = geometric_mean_score(
    ground_truth_demand_response_flags,
    random_predictions_demand_response_flags,
    average="macro",
)

# Calculate the f1 score
f1_score_value = f1_score(
    ground_truth_demand_response_flags,
    random_predictions_demand_response_flags,
    average="macro",
)

########################################################
# Phase 1: Regression Task
########################################################

"""
Phase 1 evaluation metrics principally focus on the regression task on SiteA.
This demonstrates how the regression metrics are calculated for demand response capacity prediction.
"""

# Load Phase 1 data for SiteA
phase1_training_df = pd.read_csv("../data/regression/regression-train.csv")
# Filter for SiteA only
phase1_df = phase1_training_df[phase1_training_df["Site"] == "siteA"]

# Calculate building power statistics for normalization
non_zero_building_power_indices = phase1_df["Building_Power_kW"] != 0
min_building_power = phase1_df["Building_Power_kW"][non_zero_building_power_indices].min()
max_building_power = phase1_df["Building_Power_kW"][non_zero_building_power_indices].max()
range_building_power = max_building_power - min_building_power
mean_building_power = phase1_df["Building_Power_kW"][non_zero_building_power_indices].mean()

print(f"\nBuilding Power Statistics for SiteA:")
print(f"Min Building Power: {min_building_power}")
print(f"Max Building Power: {max_building_power}")
print(f"Range: {range_building_power}")
print(f"Mean: {mean_building_power}")

# Generate random predictions for demand response capacity
ground_truth_capacity = phase1_df["Demand_Response_Capacity_kW"]
random_capacity_predictions = np.random.uniform(
    low=ground_truth_capacity.min(),
    high=ground_truth_capacity.max(),
    size=len(phase1_df)
)

# Calculate regression metrics
mae = mean_absolute_error(ground_truth_capacity, random_capacity_predictions)
rmse = root_mean_squared_error(ground_truth_capacity, random_capacity_predictions)

# Normalized metrics
n_mae_range = (mae / range_building_power) * 100
n_mae_mean = (mae / mean_building_power) * 100
n_rmse_range = (rmse / range_building_power) * 100
n_rmse_mean = (rmse / mean_building_power) * 100
cv_rmse = (rmse / np.mean(ground_truth_capacity)) * 100

print(f"\nRegression Metrics for SiteA:")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"NMAE (range): {n_mae_range}")
print(f"NMAE (mean): {n_mae_mean}")
print(f"NRMSE (range): {n_rmse_range}")
print(f"NRMSE (mean): {n_rmse_mean}")
print(f"CV-RMSE: {cv_rmse}")

# Also calculate classification metrics for SiteA in Phase 1
ground_truth_flags = phase1_df["Demand_Response_Flag"]
random_flags = np.random.choice(
    [-1, 0, 1],
    size=len(phase1_df),
)

geometric_mean_score_phase1 = geometric_mean_score(
    ground_truth_flags,
    random_flags,
    average="macro",
)
f1_score_phase1 = f1_score(
    ground_truth_flags,
    random_flags,
    average="macro",
)

# Combined Classification Metrics (using Phase 1 data as it's more comprehensive)
print(f"\nClassification Metrics for SiteA:")
print(f"Geometric Mean Score: {geometric_mean_score_phase1}")
print(f"F1 Score: {f1_score_phase1}")

print(f"\nNote: This demonstrates the evaluation metrics used in the competition.")
print(f"WarmUp Round focuses on classification (Demand_Response_Flag prediction)")
print(f"Phase 1 focuses on both classification and regression (Demand_Response_Flag and Demand_Response_Capacity_kW prediction)")