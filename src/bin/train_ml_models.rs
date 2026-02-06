use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::error::Error;
use csv::{Writer, StringRecord};
use lightgbm3::{Dataset, Booster};
use serde_json::json;

/// Load classifications from CSV
fn load_classifications(path: &str) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let mut classifications = HashMap::new();
    let file = fs::File::open(path)?;
    let mut reader = csv::ReaderBuilder::new().from_reader(file);
    
    for result in reader.records() {
        let record = result?;
        if record.len() >= 2 {
            let object = record[0].trim().to_string();
            let class = record[1].trim().to_string();
            classifications.insert(object, class);
        }
    }
    
    Ok(classifications)
}

/// Load timescale parameters from CSV and extract labels
fn load_data_with_labels(
    param_file: &str,
    classifications: &HashMap<String, String>,
) -> Result<(Vec<Vec<f64>>, Vec<String>, Vec<String>), Box<dyn Error>> {
    let file = fs::File::open(param_file)?;
    let mut reader = csv::ReaderBuilder::new().from_reader(file);
    
    let headers = reader.headers()?.clone();
    let mut feature_data = Vec::new();
    let mut labels = Vec::new();
    let mut objects = Vec::new();
    
    // Find numeric column indices
    let mut numeric_cols: Vec<usize> = Vec::new();
    for (i, header) in headers.iter().enumerate() {
        if !["object", "band", "method", "variant", "classification", "probability"].contains(&header) {
            numeric_cols.push(i);
        }
    }
    
    for result in reader.records() {
        let record = result?;
        if let Some(obj_str) = record.get(0) {
            let obj = obj_str.trim().to_string();
            if let Some(class) = classifications.get(&obj) {
                // Extract numeric features
                let mut features = Vec::new();
                for &col_idx in &numeric_cols {
                    if let Some(val_str) = record.get(col_idx) {
                        if let Ok(val) = val_str.trim().parse::<f64>() {
                            features.push(val);
                        } else {
                            features.push(f64::NAN);
                        }
                    } else {
                        features.push(f64::NAN);
                    }
                }
                
                if !features.is_empty() {
                    feature_data.push(features);
                    labels.push(class.clone());
                    objects.push(obj.clone());
                }
            }
        }
    }
    
    Ok((feature_data, labels, objects))
}

/// Fill NaN values with column medians
fn fill_nan_with_median(data: &mut Vec<Vec<f64>>) {
    if data.is_empty() {
        return;
    }
    
    let n_features = data[0].len();
    
    // Compute medians for each feature
    for feat_idx in 0..n_features {
        let mut col_values: Vec<f64> = data.iter()
            .filter_map(|row| {
                if feat_idx < row.len() && row[feat_idx].is_finite() {
                    Some(row[feat_idx])
                } else {
                    None
                }
            })
            .collect();
        
        if !col_values.is_empty() {
            col_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if col_values.len() % 2 == 0 {
                (col_values[col_values.len() / 2 - 1] + col_values[col_values.len() / 2]) / 2.0
            } else {
                col_values[col_values.len() / 2]
            };
            
            // Fill NaNs with median
            for row in data.iter_mut() {
                if feat_idx < row.len() && !row[feat_idx].is_finite() {
                    row[feat_idx] = median;
                }
            }
        }
    }
}

/// Compute feature importance based on variance
fn compute_feature_importance(data: &[Vec<f64>]) -> Vec<(usize, f64)> {
    if data.is_empty() || data[0].is_empty() {
        return Vec::new();
    }
    
    let n_features = data[0].len();
    let mut importances = Vec::new();
    
    for feat_idx in 0..n_features {
        let col_values: Vec<f64> = data.iter()
            .filter_map(|row| {
                if feat_idx < row.len() && row[feat_idx].is_finite() {
                    Some(row[feat_idx])
                } else {
                    None
                }
            })
            .collect();
        
        if !col_values.is_empty() {
            let mean = col_values.iter().sum::<f64>() / col_values.len() as f64;
            let variance = col_values.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / col_values.len() as f64;
            importances.push((feat_idx, variance));
        } else {
            importances.push((feat_idx, 0.0));
        }
    }
    
    // Sort by importance descending
    importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    importances
}

/// Compute classification statistics
fn compute_classification_stats(labels: &[String]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for label in labels {
        *counts.entry(label.clone()).or_insert(0) += 1;
    }
    counts
}

/// Train and evaluate classifier using LightGBM
fn train_classifier(
    feature_data: Vec<Vec<f64>>,
    labels: Vec<String>,
    name: &str,
) -> Result<(), Box<dyn Error>> {
    println!("\n{}", "=".repeat(60));
    println!("Processing {} dataset", name.to_uppercase());
    println!("{}", "=".repeat(60));
    
    let n_samples = feature_data.len();
    let n_features = if !feature_data.is_empty() { feature_data[0].len() } else { 0 };
    
    println!("Total samples: {}", n_samples);
    println!("Using {} numeric features", n_features);
    
    // Fill NaN values
    let mut data = feature_data;
    fill_nan_with_median(&mut data);
    
    // Compute class statistics
    let class_counts = compute_classification_stats(&labels);
    let n_classes = class_counts.len();
    println!("Classes: {}", n_classes);
    for (class, count) in class_counts.iter() {
        println!("  {}: {}", class, count);
    }
    
    // Convert string labels to numeric labels
    let mut unique_classes: Vec<&String> = class_counts.keys().collect();
    unique_classes.sort();
    let class_to_label: HashMap<&String, usize> = unique_classes.iter()
        .enumerate()
        .map(|(i, c)| (*c, i))
        .collect();
    
    let y_numeric: Vec<usize> = labels.iter()
        .map(|l| class_to_label[l])
        .collect();
    
    // Compute feature importance
    println!("\nComputing feature importance (variance-based)...");
    let importances = compute_feature_importance(&data);
    
    // Save feature importance
    let mut importance_file = Writer::from_path(format!("{}_feature_importance.csv", name))?;
    importance_file.write_record(&["feature_index", "importance"])?;
    for (idx, imp) in importances.iter().take(20) {
        importance_file.write_record(&[idx.to_string(), imp.to_string()])?;
    }
    importance_file.flush()?;
    println!("✓ Feature importance saved to {}_feature_importance.csv", name);
    
    // Train LightGBM classifier
    println!("\nTraining LightGBM classifier...");
    
    // Prepare data for lightgbm3 (Vec<f64> in row-major order)
    let mut train_data: Vec<f64> = Vec::with_capacity(n_samples * n_features);
    for row in &data {
        train_data.extend(row.iter());
    }
    
    let train_labels: Vec<f64> = y_numeric.iter().map(|&v| v as f64).collect();
    
    // Create dataset
    let dataset = Dataset::from_slice(&train_data, &train_labels, n_features as i32, true)?;
    
    // Set parameters for multi-class classification using JSON
    let params = json!({
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "num_threads": 4,
        "verbosity": -1
    });
    
    // Train the model
    let booster = Booster::train(dataset, &params)?;
    
    // Evaluate training accuracy
    let predictions = booster.predict(&train_data, n_features as i32, true)?;
    let mut correct = 0usize;
    
    // predictions is flat Vec with shape (n_samples * n_classes)
    for i in 0..n_samples {
        let mut best_class = 0usize;
        let mut best_prob = f64::NEG_INFINITY;
        for c in 0..n_classes {
            let prob = predictions[i * n_classes + c];
            if prob > best_prob {
                best_prob = prob;
                best_class = c;
            }
        }
        if best_class == y_numeric[i] {
            correct += 1;
        }
    }
    
    let accuracy = correct as f64 / n_samples as f64;
    println!("✓ Training accuracy: {:.2}% ({}/{})", accuracy * 100.0, correct, n_samples);
    
    // Save the model
    booster.save_file(&format!("{}_model.txt", name))?;
    println!("✓ Model saved to {}_model.txt", name);
    
    // Save processed data
    let mut data_file = Writer::from_path(format!("{}_ml.csv", name))?;
    
    // Write header
    let mut header = vec!["classification".to_string(), "class_numeric".to_string()];
    for i in 0..n_features {
        header.push(format!("feature_{}", i));
    }
    data_file.write_record(&header.iter().map(|s| s.as_str()).collect::<StringRecord>())?;
    
    // Write data
    for (i, row) in data.iter().enumerate() {
        let mut record = vec![labels[i].clone(), y_numeric[i].to_string()];
        for val in row {
            record.push(val.to_string());
        }
        data_file.write_record(&record.iter().map(|s| s.as_str()).collect::<StringRecord>())?;
    }
    data_file.flush()?;
    println!("✓ ML data saved to {}_ml.csv", name);
    
    // Save class statistics
    let mut stats_file = Writer::from_path(format!("{}_class_stats.csv", name))?;
    stats_file.write_record(&["class", "class_numeric", "count"])?;
    
    for (class_idx, class_name) in unique_classes.iter().enumerate() {
        let count = y_numeric.iter().filter(|&&y| y == class_idx).count();
        
        stats_file.write_record(&[
            class_name.to_string(),
            class_idx.to_string(),
            count.to_string(),
        ])?;
    }
    stats_file.flush()?;
    println!("✓ Class statistics saved to {}_class_stats.csv", name);
    
    // Summary statistics
    println!("\nSummary Statistics:");
    for (idx, _) in importances.iter().take(5) {
        let col_values: Vec<f64> = data.iter()
            .filter_map(|row| {
                if *idx < row.len() && row[*idx].is_finite() {
                    Some(row[*idx])
                } else {
                    None
                }
            })
            .collect();
        
        if !col_values.is_empty() {
            let mean = col_values.iter().sum::<f64>() / col_values.len() as f64;
            let var = col_values.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / col_values.len() as f64;
            let std = var.sqrt();
            
            println!("  Feature {}: mean={:.4}, std={:.4}", idx, mean, std);
        }
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("ML Training Pipeline (Rust + LightGBM)");
    println!("====================================\n");
    
    // Load classifications
    if !Path::new("classifications.csv").exists() {
        eprintln!("Error: classifications.csv not found!");
        std::process::exit(1);
    }
    
    let classifications = load_classifications("classifications.csv")?;
    println!("Loaded {} classifications", classifications.len());
    
    // Process parametric data
    if Path::new("parametric_timescale_parameters.csv").exists() {
        println!("\nLoading parametric timescale parameters...");
        let (feature_data, labels, objects) = 
            load_data_with_labels("parametric_timescale_parameters.csv", &classifications)?;
        
        println!("Loaded {} objects with classifications", objects.len());
        
        train_classifier(feature_data, labels, "parametric")?;
    } else {
        println!("Warning: parametric_timescale_parameters.csv not found, skipping...");
    }
    
    // Process nonparametric data
    if Path::new("nonparametric_timescale_parameters.csv").exists() {
        println!("\nLoading nonparametric timescale parameters...");
        let (feature_data, labels, objects) = 
            load_data_with_labels("nonparametric_timescale_parameters.csv", &classifications)?;
        
        println!("Loaded {} objects with classifications", objects.len());
        
        train_classifier(feature_data, labels, "nonparametric")?;
    } else {
        println!("Warning: nonparametric_timescale_parameters.csv not found, skipping...");
    }
    
    println!("\n{}", "=".repeat(60));
    println!("✓ PIPELINE COMPLETE!");
    println!("{}", "=".repeat(60));
    
    Ok(())
}
