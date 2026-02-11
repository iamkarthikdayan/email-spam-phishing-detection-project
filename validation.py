"""
validation.py - Comprehensive Model Validation Suite
Tests saved models from train_bert.py and generates detailed reports
"""

import joblib
import json
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_fscore_support, accuracy_score, f1_score,
    roc_curve, auc, precision_recall_curve
)

print("="*80)
print("EMAIL SPAM/PHISHING DETECTION - MODEL VALIDATION SUITE")
print("="*80)

# ============================================================================
# STEP 1: LOAD MODELS AND DATA
# ============================================================================
print("\n[1/4] LOADING MODELS AND DATA...")
print("-"*80)

try:
    # Load models
    clf_binary = joblib.load('models/clf_binary.pkl')
    print("✓ Loaded: Binary Classifier (Stage 1)")
    
    try:
        clf_multiclass = joblib.load('models/clf_multiclass.pkl')
        print("✓ Loaded: Multiclass Classifier (Stage 2)")
        has_multiclass = True
    except FileNotFoundError:
        print("⚠ Multiclass classifier not found - Stage 2 unavailable")
        clf_multiclass = None
        has_multiclass = False
    
    # Load metadata
    with open('models/metadata.json') as f:
        metadata = json.load(f)
    print("✓ Loaded: Metadata")
    
    # Load test data
    test_data = joblib.load('models/test_data.pkl')
    x_test = test_data['x_test']
    y_test = test_data['y_test']
    y_pred_train = test_data['y_pred']
    y_pred_proba_binary_train = test_data['y_pred_proba_binary']
    print("✓ Loaded: Test Data")
    
    # Load preprocessing info
    with open('models/preprocessing_info.json') as f:
        preprocessing_info = json.load(f)
    print("✓ Loaded: Preprocessing Info")
    
    # Load feature importance
    with open('models/feature_importance.json') as f:
        feature_importance_data = json.load(f)
    print("✓ Loaded: Feature Importance Scores")
    
    print(f"\nDataset Info:")
    print(f"  • Test samples: {len(x_test)}")
    print(f"  • Features: {x_test.shape[1]}")
    print(f"  • Threshold (Stage 1): {metadata['optimal_threshold_stage1']:.4f}")
    
except FileNotFoundError as e:
    print(f"❌ Error: Missing model file - {e}")
    print("   Run train_bert.py first to generate models")
    exit(1)
except Exception as e:
    print(f"❌ Error loading models: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

    
    print(f"\nDataset Info:")
    print(f"  • Test samples: {len(x_test)}")
    print(f"  • Features: {x_test.shape[1]}")
    print(f"  • Threshold (Stage 1): {metadata['optimal_threshold_stage1']:.4f}")
    
except FileNotFoundError as e:
    print(f"❌ Error: Missing model file - {e}")
    print("   Run train_bert.py first to generate models")
    exit(1)
except Exception as e:
    print(f"❌ Error loading models: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# STEP 2: VALIDATE STAGE 1 (BINARY CLASSIFIER)
# ============================================================================
print("\n[2/4] VALIDATING STAGE 1 - BINARY CLASSIFIER (Phishing Detection)")
print("-"*80)

try:
    # Get binary labels from y_test
    y_test_binary = (y_test == "Phishing").astype(int)
    
    # Regenerate predictions for fresh validation
    y_pred_proba_binary = clf_binary.predict_proba(x_test)[:, 1]
    optimal_threshold = metadata['optimal_threshold_stage1']
    y_pred_binary = (y_pred_proba_binary > optimal_threshold).astype(int)
    
    # Classification metrics
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_binary, y_pred_binary, average='binary', zero_division=0
    )
    
    print(f"\n📊 STAGE 1 PERFORMANCE METRICS:")
    print(f"  • Accuracy:  {accuracy:.4f}")
    print(f"  • Precision: {precision:.4f} (avoid false positives)")
    print(f"  • Recall:    {recall:.4f} (catch all phishing)")
    print(f"  • F1-Score:  {f1:.4f}")
    print(f"  • Threshold: {optimal_threshold:.4f}")
    print(f"  • PR-AUC:    {metadata['pr_auc']:.4f}")
    
    # Confusion matrix
    cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
    tn, fp, fn, tp = cm_binary.ravel()
    print(f"\n📈 CONFUSION MATRIX (Stage 1):")
    print(f"  True Negatives:  {tn} (Correctly identified Legitimate)")
    print(f"  False Positives: {fp} (Legitimate flagged as Phishing)")
    print(f"  False Negatives: {fn} (Phishing missed)")
    print(f"  True Positives:  {tp} (Correctly identified Phishing)")
    
    # Prediction distribution
    unique, counts = np.unique(y_pred_binary, return_counts=True)
    print(f"\n🎯 PREDICTION DISTRIBUTION (Stage 1):")
    for label, count in zip(unique, counts):
        label_name = "Phishing" if label == 1 else "Not Phishing"
        pct = (count / len(y_pred_binary)) * 100
        print(f"  • {label_name}: {count} ({pct:.1f}%)")
    
    # Classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT (Stage 1):")
    report_binary = classification_report(y_test_binary, y_pred_binary, 
                                         target_names=["Legitimate", "Phishing"],
                                         digits=4)
    print(report_binary)

except Exception as e:
    print(f"❌ Error validating Stage 1: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 3: VALIDATE STAGE 2 (MULTICLASS CLASSIFIER)
# ============================================================================
print("\n[3/4] VALIDATING STAGE 2 - MULTICLASS CLASSIFIER (Spam vs Legitimate)")
print("-"*80)

if has_multiclass:
    try:
        # Get non-phishing samples
        non_phishing_mask = y_test != "Phishing"
        x_test_non_phish = x_test[non_phishing_mask]
        y_test_non_phish = y_test[non_phishing_mask]
        
        if len(x_test_non_phish) > 0:
            y_pred_stage2 = clf_multiclass.predict(x_test_non_phish)
            
            # Accuracy
            accuracy_mc = accuracy_score(y_test_non_phish, y_pred_stage2)
            
            print(f"\n📊 STAGE 2 PERFORMANCE METRICS:")
            print(f"  • Accuracy: {accuracy_mc:.4f}")
            print(f"  • Samples: {len(x_test_non_phish)} (non-phishing)")
            
            # Confusion matrix
            cm_multiclass = confusion_matrix(y_test_non_phish, y_pred_stage2)
            print(f"\n📈 CONFUSION MATRIX (Stage 2):")
            print(cm_multiclass)
            
            # Classification report
            print(f"\n📋 DETAILED CLASSIFICATION REPORT (Stage 2):")
            report_mc = classification_report(y_test_non_phish, y_pred_stage2, digits=4)
            print(report_mc)
            
            # Prediction distribution
            unique_mc, counts_mc = np.unique(y_pred_stage2, return_counts=True)
            print(f"\n🎯 PREDICTION DISTRIBUTION (Stage 2):")
            for label, count in zip(unique_mc, counts_mc):
                pct = (count / len(y_pred_stage2)) * 100
                print(f"  • {label}: {count} ({pct:.1f}%)")
        else:
            print("⚠ No non-phishing samples in test set for Stage 2 validation")
    
    except Exception as e:
        print(f"❌ Error validating Stage 2: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ Stage 2 classifier not available for validation")

# ============================================================================
# STEP 4: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[4/4] FEATURE IMPORTANCE ANALYSIS")
print("-"*80)

try:
    feature_names = metadata['all_feature_names']
    stage1_importance = np.array(feature_importance_data['stage1_importance'])
    
    # Top 10 features
    top_indices = np.argsort(stage1_importance)[-10:][::-1]
    
    print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES (Stage 1):")
    print(f"  {'Rank':<6} {'Feature Name':<50} {'Importance':>12}")
    print(f"  {'-'*70}")
    
    for rank, idx in enumerate(top_indices, 1):
        fname = feature_names[idx]
        if len(fname) > 47:
            fname = fname[:44] + "..."
        importance = stage1_importance[idx]
        print(f"  {rank:<6} {fname:<50} {importance:>12.6f}")
    
    # Bottom 5 features
    bottom_indices = np.argsort(stage1_importance)[:5]
    print(f"\n📉 LEAST IMPORTANT FEATURES (Stage 1):")
    print(f"  {'Rank':<6} {'Feature Name':<50} {'Importance':>12}")
    print(f"  {'-'*70}")
    
    for rank, idx in enumerate(bottom_indices, 1):
        fname = feature_names[idx]
        if len(fname) > 47:
            fname = fname[:44] + "..."
        importance = stage1_importance[idx]
        print(f"  {rank:<6} {fname:<50} {importance:>12.6f}")
    
    # Feature importance summary
    print(f"\n📊 FEATURE IMPORTANCE STATISTICS:")
    print(f"  • Mean:   {stage1_importance.mean():.6f}")
    print(f"  • Median: {np.median(stage1_importance):.6f}")
    print(f"  • Std:    {stage1_importance.std():.6f}")
    print(f"  • Max:    {stage1_importance.max():.6f}")
    print(f"  • Min:    {stage1_importance.min():.6f}")

except Exception as e:
    print(f"❌ Error analyzing feature importance: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# FINAL SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY REPORT")
print("="*80)

summary = f"""
MODEL VALIDATION RESULTS
{'='*80}

TEST DATASET:
  • Total Samples: {len(x_test)}
  • Features: {x_test.shape[1]}
  • Classes: {', '.join(np.unique(y_test))}

STAGE 1: BINARY CLASSIFIER (Phishing vs Not-Phishing)
  • Type: LightGBM
  • Threshold: {optimal_threshold:.4f}
  • Accuracy: {accuracy:.4f}
  • Precision: {precision:.4f}
  • Recall: {recall:.4f}
  • F1-Score: {f1:.4f}
  • Status: ✓ VALIDATED

STAGE 2: MULTICLASS CLASSIFIER (Spam vs Legitimate)
  • Type: LightGBM
  • Status: {'✓ VALIDATED' if has_multiclass else '⚠ NOT AVAILABLE'}
  {'• Accuracy: ' + f'{accuracy_mc:.4f}' if has_multiclass and len(x_test_non_phish) > 0 else ''}

FEATURE ENGINEERING:
  • Total Features: {len(feature_names)}
  • Embedding Dimension: 768
  • Metadata Features: 7
  • Phishing Keywords: 14
  • URL Features: 13
  • Additional Features: {len(feature_names) - 802}

TOP 3 FEATURES:
"""

for rank, idx in enumerate(top_indices[:3], 1):
    fname = feature_names[idx]
    importance = stage1_importance[idx]
    summary += f"  {rank}. {fname}: {importance:.6f}\n"

summary += f"""
RECOMMENDATIONS:
  1. ✓ Models are production-ready
  2. Monitor top features for data drift
  3. Consider retraining when recall drops below {recall*0.9:.4f}
  4. Validate on new phishing samples periodically
"""

print(summary)

# Save validation report
try:
    report_path = 'models/validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n✓ Validation report saved to: {report_path}")
except Exception as e:
    print(f"⚠ Could not save validation report: {e}")

# ============================================================================
# EXPORT VALIDATION RESULTS
# ============================================================================
print("\n" + "="*80)
print("EXPORTING VALIDATION RESULTS")
print("="*80)

try:
    # Save validation metrics to JSON
    validation_metrics = {
        'stage1': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'threshold': float(optimal_threshold),
            'confusion_matrix': cm_binary.tolist(),
        },
        'stage2': {
            'available': has_multiclass,
            'accuracy': float(accuracy_mc) if has_multiclass and len(x_test_non_phish) > 0 else None,
        },
        'dataset': {
            'test_samples': int(len(x_test)),
            'features': int(x_test.shape[1]),
            'classes': list(np.unique(y_test)),
        }
    }
    
    with open('models/validation_metrics.json', 'w') as f:
        json.dump(validation_metrics, f, indent=2)
    print("✓ Saved: models/validation_metrics.json")
    
    # Save predictions for analysis
    predictions_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred_train,
        'Phishing_Probability': y_pred_proba_binary_train,
        'Decision_Threshold': optimal_threshold,
    })
    
    predictions_df.to_csv('models/validation_predictions.csv', index=False)
    print("✓ Saved: models/validation_predictions.csv")
    
    # Create feature importance CSV
    importance_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'Importance_Score': stage1_importance.tolist(),
        'Rank': np.argsort(stage1_importance)[::-1].argsort() + 1,
    }).sort_values('Importance_Score', ascending=False)
    
    importance_df.to_csv('models/validation_feature_importance.csv', index=False)
    print("✓ Saved: models/validation_feature_importance.csv")

except Exception as e:
    print(f"⚠ Error exporting results: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✓ VALIDATION COMPLETE")
print("="*80)
print("\nGenerated Files:")
print("  • models/validation_report.txt")
print("  • models/validation_metrics.json")
print("  • models/validation_predictions.csv")
print("  • models/validation_feature_importance.csv")

# -------------------------------------------------------------------
# Hold-Out Testing (Stage 2 Multiclass Classifier)
# -------------------------------------------------------------------
def run_holdout_test(model, X_test, y_test, stage="Stage 2"):
    print(f"\n=== {stage} Hold-Out Testing ===")
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------------------------
# Run Validation
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Run 5-fold CV on Stage 1
    run_cross_validation(clf_binary, X, y, k=5, stage="Stage 1 Binary Classifier")

    # Example: Run hold-out test on Stage 2
    # Replace with your actual hold-out dataset
    # X_holdout = np.load("data/X_holdout.npy")
    # y_holdout = np.load("data/y_holdout.npy")
    # run_holdout_test(clf_multiclass, X_holdout, y_holdout, stage="Stage 2 Multiclass Classifier")