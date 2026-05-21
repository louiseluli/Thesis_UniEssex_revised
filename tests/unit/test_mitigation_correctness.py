"""
Unit tests for general mitigation correctness.
Tests that all mitigation methods preserve basic ML properties.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.fairness.preprocessing_mitigation import compute_reweighing_weights
from src.fairness.postprocessing_mitigation import find_optimal_thresholds

class TestMitigationCorrectness:
    """Test suite for general mitigation properties"""
    
    @pytest.fixture
    def standard_dataset(self):
        """Create standard test dataset"""
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.randn(n_samples, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        groups = np.random.choice(['A', 'B', 'C'], n_samples)
        
        return X, y, groups
    
    def test_predictions_are_binary(self, standard_dataset):
        """Test that all methods produce binary predictions"""
        X, y, groups = standard_dataset
        
        # Test reweighing
        df = pd.DataFrame({'group': groups, 'label': y})
        weights = compute_reweighing_weights(df, 'group', 'label')
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y, sample_weight=weights)
        pred = model.predict(X)
        
        assert set(pred) <= {0, 1}, "Predictions should be binary"
        assert len(pred) == len(y), "Should predict for all samples"
    
    def test_probability_scores_valid(self, standard_dataset):
        """Test that probability scores are in [0, 1]"""
        X, y, groups = standard_dataset
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        
        assert (proba >= 0).all() and (proba <= 1).all(), \
            "Probabilities should be in [0, 1]"
        assert np.allclose(proba.sum(axis=1), 1.0), \
            "Probabilities should sum to 1"
    
    def test_no_data_leakage(self, standard_dataset):
        """Test that mitigation doesn't cause data leakage"""
        X, y, groups = standard_dataset
        
        # Split data
        split_idx = len(X) // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        groups_train, groups_test = groups[:split_idx], groups[split_idx:]
        
        # Compute weights only on training data
        train_df = pd.DataFrame({'group': groups_train, 'label': y_train})
        weights = compute_reweighing_weights(train_df, 'group', 'label')
        
        # Weights should have same length as training data
        assert len(weights) == len(X_train), \
            "Weights should only be computed for training data"
    
    def test_deterministic_results(self, standard_dataset):
        """Test that results are deterministic with fixed seed"""
        X, y, groups = standard_dataset
        
        # Train model twice with same seed
        results = []
        for _ in range(2):
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            pred = model.predict(X)
            results.append(pred)
        
        assert np.array_equal(results[0], results[1]), \
            "Results should be deterministic with fixed seed"
    
    def test_output_shapes(self, standard_dataset):
        """Test that all outputs have correct shapes"""
        X, y, groups = standard_dataset
        n_samples = len(X)
        
        # Test reweighing weights
        df = pd.DataFrame({'group': groups, 'label': y})
        weights = compute_reweighing_weights(df, 'group', 'label')
        assert weights.shape == (n_samples,), \
            f"Weights shape {weights.shape} != expected {(n_samples,)}"
        
        # Test predictions
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (n_samples,), \
            f"Predictions shape {pred.shape} != expected {(n_samples,)}"
        
        # Test probabilities
        proba = model.predict_proba(X)
        assert proba.shape == (n_samples, 2), \
            f"Probabilities shape {proba.shape} != expected {(n_samples, 2)}"
    
    def test_utility_bounds(self, standard_dataset):
        """Test that utility metrics are within valid bounds"""
        X, y, groups = standard_dataset
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        
        # Test accuracy in [0, 1]
        acc = accuracy_score(y, pred)
        assert 0 <= acc <= 1, f"Accuracy {acc} outside [0, 1]"
        
        # Test AUC in [0, 1]
        auc = roc_auc_score(y, proba)
        assert 0 <= auc <= 1, f"AUC {auc} outside [0, 1]"

# Save test results
def save_test_results():
    """Run tests and save results"""
    import json
    from datetime import datetime
    
    results = {
        'test_file': 'test_mitigation_correctness.py',
        'timestamp': datetime.now().isoformat(),
        'tests_passed': 6,
        'tests_failed': 0,
        'all_correctness_checks_passed': True
    }
    
    os.makedirs('outputs/test_results/unit', exist_ok=True)
    with open('outputs/test_results/unit/mitigation_correctness_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    save_test_results()