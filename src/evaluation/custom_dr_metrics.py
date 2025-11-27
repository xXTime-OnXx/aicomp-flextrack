"""
Custom Metrics for DR-Only Training

This evaluates performance ONLY on DR events (where DR flag != 0)
and provides meaningful metrics that actually reflect model quality.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, f1_score, r2_score
from imblearn.metrics import geometric_mean_score


def evaluate_dr_only_metrics(y_true, y_pred, demand_flags, building_power=None):
    """
    Calculate metrics ONLY on DR events (where demand_flags != 0)
    
    Args:
        y_true: ground truth DR capacity values (all samples)
        y_pred: predicted DR capacity values (all samples)
        demand_flags: DR flags (-1, 0, 1) for each sample
        building_power: (optional) building power for normalization
    
    Returns:
        dict with DR-only metrics
    """
    # Filter to DR events only (where flag is -1 or 1, not 0)
    dr_mask = demand_flags != 0
    
    if not np.any(dr_mask):
        # No DR events in this dataset
        return {
            'dr_samples': 0,
            'mae': 0, 'rmse': 0, 'r2': 0,
            'nmae_capacity': 0, 'nrmse_capacity': 0,
            'nmae_power': 0, 'nrmse_power': 0,
            'mean_abs_error_kw': 0,
            'f1_direction': 0, 'accuracy_direction': 0
        }
    
    y_true_dr = y_true[dr_mask]
    y_pred_dr = y_pred[dr_mask]
    flags_dr = demand_flags[dr_mask]
    
    # Basic regression metrics
    mae = mean_absolute_error(y_true_dr, y_pred_dr)
    rmse = root_mean_squared_error(y_true_dr, y_pred_dr)
    
    # R² score (how much variance explained)
    try:
        r2 = r2_score(y_true_dr, y_pred_dr)
    except:
        r2 = 0.0
    
    # Normalized metrics by DR capacity range (makes sense for DR-only!)
    capacity_range = y_true_dr.max() - y_true_dr.min()
    capacity_mean = np.abs(y_true_dr).mean()
    
    nmae_capacity = (mae / capacity_range * 100) if capacity_range > 0 else 0
    nrmse_capacity = (rmse / capacity_range * 100) if capacity_range > 0 else 0
    
    # Also normalize by capacity mean
    nmae_capacity_mean = (mae / capacity_mean * 100) if capacity_mean > 0 else 0
    nrmse_capacity_mean = (rmse / capacity_mean * 100) if capacity_mean > 0 else 0
    
    # If building power provided, also normalize by that
    nmae_power = 0
    nrmse_power = 0
    if building_power is not None:
        power_dr = building_power[dr_mask]
        power_mean = np.abs(power_dr).mean()
        power_range = power_dr.max() - power_dr.min()
        
        nmae_power = (mae / power_mean * 100) if power_mean > 0 else 0
        nrmse_power = (rmse / power_range * 100) if power_range > 0 else 0
    
    # Direction accuracy (positive/negative)
    # This checks if we got the direction right (increase vs decrease in demand)
    pred_direction = np.sign(y_pred_dr)
    true_direction = np.sign(y_true_dr)
    direction_correct = (pred_direction == true_direction).mean()
    
    # F1 for direction classification
    try:
        f1_dir = f1_score(true_direction, pred_direction, average='macro', zero_division=0)
    except:
        f1_dir = 0.0
    
    return {
        # Dataset info
        'dr_samples': int(np.sum(dr_mask)),
        'dr_percentage': float(np.mean(dr_mask) * 100),
        
        # Raw metrics (in kW)
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        
        # Normalized by DR capacity range (PRIMARY METRICS for DR-only training)
        'nmae_capacity_range': float(nmae_capacity),
        'nrmse_capacity_range': float(nrmse_capacity),
        'nmae_capacity_mean': float(nmae_capacity_mean),
        'nrmse_capacity_mean': float(nrmse_capacity_mean),
        
        # Normalized by building power (for reference)
        'nmae_power': float(nmae_power),
        'nrmse_power': float(nrmse_power),
        
        # Direction metrics
        'accuracy_direction': float(direction_correct * 100),
        'f1_direction': float(f1_dir),
        
        # Interpretable error
        'mean_abs_error_kw': float(mae),
    }


def evaluate_by_site_dr_only(y_true, y_pred, demand_flags, site_labels, building_power=None):
    """
    Calculate DR-only metrics per site, then aggregate
    
    Args:
        y_true: ground truth values
        y_pred: predicted values  
        demand_flags: DR flags for each sample
        site_labels: site identifier for each sample
        building_power: (optional) building power values
    
    Returns:
        dict with aggregated DR-only metrics across sites
    """
    sites = np.unique(site_labels)
    
    site_metrics = []
    
    for site in sites:
        site_mask = site_labels == site
        
        site_true = y_true[site_mask]
        site_pred = y_pred[site_mask]
        site_flags = demand_flags[site_mask]
        site_power = building_power[site_mask] if building_power is not None else None
        
        metrics = evaluate_dr_only_metrics(site_true, site_pred, site_flags, site_power)
        metrics['site'] = site
        site_metrics.append(metrics)
    
    # Aggregate across sites (weighted by number of DR samples)
    total_dr_samples = sum(m['dr_samples'] for m in site_metrics)
    
    if total_dr_samples == 0:
        return {
            'total_dr_samples': 0,
            'mae': 0, 'rmse': 0, 'r2': 0,
            'nmae_capacity_range': 0, 'nrmse_capacity_range': 0,
            'nmae_capacity_mean': 0, 'nrmse_capacity_mean': 0,
            'nmae_power': 0, 'nrmse_power': 0,
            'accuracy_direction': 0, 'f1_direction': 0,
            'per_site': site_metrics
        }
    
    # Weighted average by number of DR samples per site
    weights = np.array([m['dr_samples'] / total_dr_samples for m in site_metrics])
    
    aggregated = {
        'total_dr_samples': int(total_dr_samples),
        
        # Weighted averages
        'mae': float(np.average([m['mae'] for m in site_metrics], weights=weights)),
        'rmse': float(np.average([m['rmse'] for m in site_metrics], weights=weights)),
        'r2': float(np.average([m['r2'] for m in site_metrics], weights=weights)),
        
        'nmae_capacity_range': float(np.average([m['nmae_capacity_range'] for m in site_metrics], weights=weights)),
        'nrmse_capacity_range': float(np.average([m['nrmse_capacity_range'] for m in site_metrics], weights=weights)),
        'nmae_capacity_mean': float(np.average([m['nmae_capacity_mean'] for m in site_metrics], weights=weights)),
        'nrmse_capacity_mean': float(np.average([m['nrmse_capacity_mean'] for m in site_metrics], weights=weights)),
        
        'nmae_power': float(np.average([m['nmae_power'] for m in site_metrics], weights=weights)),
        'nrmse_power': float(np.average([m['nrmse_power'] for m in site_metrics], weights=weights)),
        
        'accuracy_direction': float(np.average([m['accuracy_direction'] for m in site_metrics], weights=weights)),
        'f1_direction': float(np.average([m['f1_direction'] for m in site_metrics], weights=weights)),
        
        # Per-site breakdown
        'per_site': site_metrics
    }
    
    return aggregated


def evaluate_classification_quality(y_true, y_pred, demand_flags):
    """
    Evaluate how well the model classifies DR events vs non-events
    
    This checks: can the model detect WHEN a DR event is happening?
    
    Args:
        y_true: ground truth capacity
        y_pred: predicted capacity
        demand_flags: true DR flags
    
    Returns:
        dict with classification metrics
    """
    # Convert to binary: DR event or not
    true_is_dr = (demand_flags != 0).astype(int)
    
    # Predict DR event if capacity is non-zero (with small threshold)
    threshold = 1.0  # kW threshold for considering it a DR event
    pred_is_dr = (np.abs(y_pred) > threshold).astype(int)
    
    # Calculate classification metrics
    tp = np.sum((true_is_dr == 1) & (pred_is_dr == 1))
    tn = np.sum((true_is_dr == 0) & (pred_is_dr == 0))
    fp = np.sum((true_is_dr == 0) & (pred_is_dr == 1))
    fn = np.sum((true_is_dr == 1) & (pred_is_dr == 0))
    
    accuracy = (tp + tn) / len(true_is_dr) if len(true_is_dr) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'detection_accuracy': float(accuracy * 100),
        'detection_precision': float(precision * 100),
        'detection_recall': float(recall * 100),
        'detection_f1': float(f1 * 100),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def print_metrics_summary(metrics, prefix=""):
    """Pretty print metrics summary"""
    print(f"\n{prefix}{'='*60}")
    print(f"{prefix}DR-ONLY METRICS SUMMARY")
    print(f"{prefix}{'='*60}")
    print(f"{prefix}DR Samples: {metrics['total_dr_samples']}")
    print(f"{prefix}")
    print(f"{prefix}PRIMARY METRICS (normalized by DR capacity):")
    print(f"{prefix}  NMAE (capacity mean):  {metrics['nmae_capacity_mean']:.2f}%")
    print(f"{prefix}  NRMSE (capacity mean): {metrics['nrmse_capacity_mean']:.2f}%")
    print(f"{prefix}  R² Score:              {metrics['r2']:.4f}")
    print(f"{prefix}")
    print(f"{prefix}ABSOLUTE ERRORS:")
    print(f"{prefix}  MAE:  {metrics['mae']:.2f} kW")
    print(f"{prefix}  RMSE: {metrics['rmse']:.2f} kW")
    print(f"{prefix}")
    print(f"{prefix}DIRECTION ACCURACY:")
    print(f"{prefix}  Correct direction: {metrics['accuracy_direction']:.1f}%")
    print(f"{prefix}  F1 (direction):    {metrics['f1_direction']:.3f}")
    print(f"{prefix}{'='*60}\n")


# Example usage function
def evaluate_model_comprehensive(y_true, y_pred, demand_flags, site_labels, building_power):
    """
    Comprehensive evaluation with all custom metrics
    
    Returns:
        dict with all metrics organized by category
    """
    # DR-only regression metrics (by site)
    dr_metrics = evaluate_by_site_dr_only(
        y_true, y_pred, demand_flags, site_labels, building_power
    )
    
    # Classification quality (DR detection)
    classification = evaluate_classification_quality(
        y_true, y_pred, demand_flags
    )
    
    # Combine all metrics
    all_metrics = {
        **dr_metrics,
        **classification
    }
    
    return all_metrics


if __name__ == "__main__":
    # Test with dummy data
    print("Testing custom metrics...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create dummy data
    y_true = np.random.randn(n_samples) * 100
    y_pred = y_true + np.random.randn(n_samples) * 20  # Add some noise
    demand_flags = np.random.choice([-1, 0, 1], n_samples, p=[0.15, 0.70, 0.15])
    site_labels = np.random.choice(['Site A', 'Site B', 'Site C'], n_samples)
    building_power = np.abs(np.random.randn(n_samples) * 500)
    
    # Evaluate
    metrics = evaluate_model_comprehensive(
        y_true, y_pred, demand_flags, site_labels, building_power
    )
    
    print_metrics_summary(metrics)
    
    print("Per-site breakdown:")
    for site_metric in metrics['per_site']:
        print(f"\n{site_metric['site']}:")
        print(f"  DR samples: {site_metric['dr_samples']}")
        print(f"  NMAE: {site_metric['nmae_capacity_mean']:.2f}%")
        print(f"  MAE: {site_metric['mae']:.2f} kW")
