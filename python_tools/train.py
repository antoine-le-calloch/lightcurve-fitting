import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
import umap
import matplotlib.pyplot as plt
import pickle

# ------------------ Load master classification ------------------
if not os.path.exists("classifications.csv"):
    print("Error: classifications.csv not found!")
    exit(1)
cls = pd.read_csv("classifications.csv")

# ------------------ Build ML tables ------------------
def build_ml_table(param_file, outname):
    if not os.path.exists(param_file):
        print(f"Warning: {param_file} not found, skipping...")
        return None
    
    df = pd.read_csv(param_file)
    
    # Remove non-numeric columns that shouldn't be pivoted
    non_numeric_cols = ["method", "variant"]
    for col in non_numeric_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Pivot with proper handling of multi-level columns
    wide = df.pivot(index="object", columns="band")
    wide.columns = [f"{col}_{band}" for col, band in wide.columns]
    wide.reset_index(inplace=True)
    
    # Merge with classifications
    ml = cls.merge(wide, on="object", how="inner")
    ml = ml.dropna(subset=["classification"])
    
    # Filter out rare classes (less than 2 samples)
    class_counts = ml["classification"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    ml = ml[ml["classification"].isin(valid_classes)]
    
    # Create numeric labels
    ml["label"] = ml["classification"].astype("category").cat.codes
    
    # Save
    ml.to_csv(outname, index=False)
    print(f"Built {outname}: {len(ml)} samples, {ml['classification'].nunique()} classes")
    return ml

par_ml = build_ml_table("parametric_timescale_parameters.csv", "parametric_ml.csv")
non_ml = build_ml_table("nonparametric_timescale_parameters.csv", "nonparametric_ml.csv")

# Exit if both failed
if par_ml is None and non_ml is None:
    print("Error: No parameter files found!")
    exit(1)

# ------------------ LightGBM trainer with hyperparameter tuning ------------------
def train_lgb(df, name):
    if df is None:
        print(f"Skipping {name} - no data available")
        return None, None
    
    # Drop non-feature columns
    drop_cols = ["object", "classification", "probability", "label"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label"]
    
    # Select only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    
    print(f"\n{'='*60}")
    print(f"Processing {name.upper()} dataset")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Using {len(numeric_cols)} numeric features")
    print(f"Classes: {y.nunique()}")
    
    # Handle infinite values and NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaNs with column medians
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
    
    # Check if we have enough samples for stratified split
    min_class_count = y.value_counts().min()
    use_stratify = min_class_count >= 2
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, 
        stratify=y if use_stratify else None, 
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Hyperparameter tuning with GridSearchCV
    print("\nPerforming hyperparameter tuning...")
    print("This may take a few minutes...")
    
    param_grid = {
        'num_leaves': [31, 63],
        'learning_rate': [0.05, 0.1],
        'max_depth': [6, 8],
        'n_estimators': [100, 200]
    }
    
    print(f"Grid search will test {2*2*2*2} parameter combinations with {min(3, min_class_count)}-fold CV...")
    
    base_model = lgb.LGBMClassifier(
        objective="multiclass",
        metric="multi_logloss",
        num_class=y.nunique(),
        boosting_type='gbdt',
        verbose=-1,
        random_state=42,
        n_jobs=4  # Limit to 4 jobs to see progress
    )
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=min(3, min_class_count),  # Use 3-fold CV or fewer if classes are sparse
        scoring='accuracy',
        n_jobs=1,  # Sequential to show progress
        verbose=3  # Maximum verbosity
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Use best model
    model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{name.upper()} CLASSIFIER RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save model
    with open(f"{name}_lgb.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {name}_lgb.pkl")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv(f"{name}_feature_importance_lgb.csv", index=False)
    print(f"✓ Feature importance saved to {name}_feature_importance_lgb.csv")
    
    return model, X

# Train models
print("\n" + "="*60)
print("TRAINING PARAMETRIC CLASSIFIER (LightGBM)")
print("="*60)
par_model, par_X = train_lgb(par_ml, "parametric")

print("\n" + "="*60)
print("TRAINING NONPARAMETRIC CLASSIFIER (LightGBM)")
print("="*60)
non_model, non_X = train_lgb(non_ml, "nonparametric")

# ------------------ UMAP Projection ------------------
def do_umap(X, df, name):
    if X is None or df is None:
        return None
    
    print(f"\nGenerating UMAP projection for {name}...")
    
    n_neighbors = min(20, len(X)-1)
    if n_neighbors < 2:
        print(f"Not enough samples for UMAP on {name}")
        return None
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.05, metric="euclidean", random_state=42)
    emb = reducer.fit_transform(X)
    
    emb_df = pd.DataFrame(emb, columns=["UMAP1", "UMAP2"])
    emb_df["classification"] = df["classification"].values
    emb_df.to_csv(f"{name}_umap_lgb.csv", index=False)
    
    plt.figure(figsize=(10, 8))
    for cls in emb_df["classification"].unique():
        mask = emb_df["classification"] == cls
        plt.scatter(emb_df.loc[mask, "UMAP1"], emb_df.loc[mask, "UMAP2"], 
                   label=cls, s=20, alpha=0.6)
    
    plt.xlabel("UMAP1", fontsize=14)
    plt.ylabel("UMAP2", fontsize=14)
    plt.title(f"{name.capitalize()} UMAP Projection (LightGBM)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{name}_umap_lgb.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ UMAP plot saved to {name}_umap_lgb.png")
    return emb

par_emb = do_umap(par_X, par_ml, "parametric")
non_emb = do_umap(non_X, non_ml, "nonparametric")

# ------------------ Anomaly Detection ------------------
def find_outliers(emb, df, name):
    if emb is None or df is None:
        return
    
    print(f"\nDetecting anomalies for {name}...")
    
    iso = IsolationForest(contamination=0.02, random_state=42)
    scores = iso.fit_predict(emb)
    
    df_out = df.copy()
    df_out["anomaly"] = scores
    
    anomalies = df_out[df_out["anomaly"] == -1][["object", "classification"]]
    anomalies.to_csv(f"{name}_anomalies_lgb.csv", index=False)
    print(f"✓ Found {len(anomalies)} anomalies, saved to {name}_anomalies_lgb.csv")

find_outliers(par_emb, par_ml, "parametric")
find_outliers(non_emb, non_ml, "nonparametric")

print("\n" + "="*60)
print("✓ PIPELINE COMPLETE!")
print("="*60)
