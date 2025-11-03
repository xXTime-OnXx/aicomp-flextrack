"""
Simplified competition metrics - only the 4 required metrics
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, f1_score
from imblearn.metrics import geometric_mean_score


def calculate_nmae_nrmse(y_true, y_pred, building_power):
    """
    Calculate NMAE and NRMSE (normalized by range and mean)
    
    Args:
        y_true: ground truth values
        y_pred: predicted values
        building_power: building power values for normalization
    
    Returns:
        dict with nmae_range, nmae_mean, nrmse_range, nrmse_mean
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    # Normalized metrics using building power
    non_zero_power = building_power[building_power != 0]
    if len(non_zero_power) > 0:
        power_range = non_zero_power.max() - non_zero_power.min()
        power_mean = non_zero_power.mean()
        
        nmae_range = (mae / power_range) * 100 if power_range > 0 else 0
        nmae_mean = (mae / power_mean) * 100 if power_mean > 0 else 0
        nrmse_range = (rmse / power_range) * 100 if power_range > 0 else 0
        nrmse_mean = (rmse / power_mean) * 100 if power_mean > 0 else 0
    else:
        nmae_range = nmae_mean = nrmse_range = nrmse_mean = 0
    
    return {
        'nmae_range': nmae_range,
        'nmae_mean': nmae_mean,
        'nrmse_range': nrmse_range,
        'nrmse_mean': nrmse_mean
    }


def calculate_classification_metrics(y_true_flags, y_pred_capacity):
    """
    Calculate Geometric Mean Score and F1 Score
    
    Args:
        y_true_flags: ground truth demand response flags (-1, 0, 1)
        y_pred_capacity: predicted capacity values (will be converted to flags)
    
    Returns:
        dict with geometric_mean_score and f1_score
    """
    # Convert predicted capacity to flags
    y_pred_flags = np.where(y_pred_capacity > 0, 1, 
                           np.where(y_pred_capacity < 0, -1, 0))
    
    try:
        geo_mean = geometric_mean_score(y_true_flags, y_pred_flags, average='macro')
        f1 = f1_score(y_true_flags, y_pred_flags, average='macro')
    except:
        geo_mean = 0.0
        f1 = 0.0
    
    return {
        'geometric_mean_score': geo_mean,
        'f1_score': f1
    }


def evaluate_all_metrics(y_true, y_pred, site_labels, building_power, demand_flags=None):
    """
    Calculate aggregated metrics across all sites
    
    Args:
        y_true: ground truth values
        y_pred: predicted values
        site_labels: array of site labels for each sample
        building_power: building power values
        demand_flags: (optional) demand response flags for classification
    
    Returns:
        dict with only aggregate metrics: nmae_range, nmae_mean, nrmse_range, nrmse_mean, 
        geometric_mean_score, f1_score
    """
    sites = np.unique(site_labels)
    
    # Calculate per-site metrics
    site_nmae_range = []
    site_nmae_mean = []
    site_nrmse_range = []
    site_nrmse_mean = []
    site_geo_mean = []
    site_f1 = []
    
    for site in sites:
        site_mask = site_labels == site
        site_true = y_true[site_mask]
        site_pred = y_pred[site_mask]
        site_power = building_power[site_mask]
        
        # NMAE and NRMSE
        regression_metrics = calculate_nmae_nrmse(site_true, site_pred, site_power)
        site_nmae_range.append(regression_metrics['nmae_range'])
        site_nmae_mean.append(regression_metrics['nmae_mean'])
        site_nrmse_range.append(regression_metrics['nrmse_range'])
        site_nrmse_mean.append(regression_metrics['nrmse_mean'])
        
        # Classification metrics (if flags provided)
        if demand_flags is not None:
            site_flags = demand_flags[site_mask]
            class_metrics = calculate_classification_metrics(site_flags, site_pred)
            site_geo_mean.append(class_metrics['geometric_mean_score'])
            site_f1.append(class_metrics['f1_score'])
    
    # Return only aggregate metrics (average across sites)
    metrics = {
        'nmae_range': np.mean(site_nmae_range),
        'nmae_mean': np.mean(site_nmae_mean),
        'nrmse_range': np.mean(site_nrmse_range),
        'nrmse_mean': np.mean(site_nrmse_mean),
    }
    
    # Add classification metrics if available
    if site_geo_mean:
        metrics['geometric_mean_score'] = np.mean(site_geo_mean)
        metrics['f1_score'] = np.mean(site_f1)
    
    return metrics