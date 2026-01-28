//! Evaluation metrics for NiV-Bench

/// Compute Area Under the ROC Curve
pub fn auc_roc(predictions: &[f32], labels: &[bool]) -> f32 {
    if predictions.len() != labels.len() || predictions.is_empty() {
        return 0.0;
    }
    
    let mut indices: Vec<usize> = (0..predictions.len()).collect();
    // Sort descending by score
    indices.sort_by(|&a, &b| predictions[b].partial_cmp(&predictions[a]).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tp = 0.0;
    let mut prev_fp = 0.0;
    let mut auc = 0.0;
    
    let total_pos = labels.iter().filter(|&&l| l).count() as f32;
    let total_neg = labels.len() as f32 - total_pos;
    
    if total_pos == 0.0 || total_neg == 0.0 {
        return 0.5; // Undefined, return random
    }

    for &i in &indices {
        if labels[i] {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        
        // Trapezoidal rule
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
        
        prev_tp = tp;
        prev_fp = fp;
    }
    
    auc / (total_pos * total_neg)
}

/// Compute Area Under the Precision-Recall Curve
pub fn auc_pr(predictions: &[f32], labels: &[bool]) -> f32 {
    // Simplified implementation (Average Precision)
     if predictions.len() != labels.len() || predictions.is_empty() {
        return 0.0;
    }
    
    let mut indices: Vec<usize> = (0..predictions.len()).collect();
    indices.sort_by(|&a, &b| predictions[b].partial_cmp(&predictions[a]).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut auc = 0.0;
    let mut prev_recall = 0.0;
    
    let total_pos = labels.iter().filter(|&&l| l).count() as f32;
    if total_pos == 0.0 { return 0.0; }

    for &i in &indices {
        if labels[i] {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        
        let precision = tp / (tp + fp);
        let recall = tp / total_pos;
        
        auc += precision * (recall - prev_recall);
        prev_recall = recall;
    }
    
    auc
}

/// Compute Precision at K
pub fn precision_at_k(predictions: &[f32], labels: &[bool], k: usize) -> f32 {
     if predictions.len() != labels.len() || k == 0 {
        return 0.0;
    }
     let mut indices: Vec<usize> = (0..predictions.len()).collect();
    indices.sort_by(|&a, &b| predictions[b].partial_cmp(&predictions[a]).unwrap_or(std::cmp::Ordering::Equal));
    
    let k = k.min(indices.len());
    let top_k = &indices[0..k];
    
    let tp = top_k.iter().filter(|&&i| labels[i]).count();
    
    tp as f32 / k as f32
}

/// Compute Spearman's Rank Correlation Coefficient
pub fn spearman_rho(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    if n != y.len() || n < 2 {
        return 0.0;
    }
    
    let rank_x = rank(x);
    let rank_y = rank(y);
    
    // Pearson of ranks
    pearson_correlation(&rank_x, &rank_y)
}

fn rank(v: &[f32]) -> Vec<f32> {
    let mut indices: Vec<usize> = (0..v.len()).collect();
    indices.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut ranks = vec![0.0; v.len()];
    for (r, &i) in indices.iter().enumerate() {
        ranks[i] = r as f32 + 1.0;
    }
    // Handle ties if needed, but for now simple rank
    ranks
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;
    
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    
    if den_x == 0.0 || den_y == 0.0 {
        0.0
    } else {
        num / (den_x.sqrt() * den_y.sqrt())
    }
}

/// Compute Root Mean Squared Error
pub fn rmse(predictions: &[f32], targets: &[f32]) -> f32 {
     if predictions.len() != targets.len() || predictions.is_empty() {
        return 0.0;
    }
    
    let mse = predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>() / predictions.len() as f32;
        
    mse.sqrt()
}