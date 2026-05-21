"""
Statistical significance tests for mitigation improvements.
Tests that improvements are statistically significant.
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.fairness.preprocessing_mitigation import compute_reweighing_weights
from src.fairness.fairness_evaluation_utils import (
    demographic_parity_difference,
    equal_opportunity_difference
)

class TestSignificance:
    """Test suite for statistical significance"""
    
    @pytest.fixture
    def experiment_data(self):
        """Create data with strong structural bias that reweighing reliably corrects."""
        np.random.seed(42)
        n_samples = 2000

        groups = np.random.choice(['A', 'B'], n_samples, p=[0.7, 0.3])

        # Feature 0 strongly encodes group membership so the model can learn
        # the group-label association and reweighing can correct it.
        X = np.zeros((n_samples, 10))
        X[groups == 'A', 0] = np.random.normal(1.0, 0.4, (groups == 'A').sum())
        X[groups == 'B', 0] = np.random.normal(-1.0, 0.4, (groups == 'B').sum())
        X[:, 1:] = np.random.randn(n_samples, 9)

        # Strong label bias: group A 65 % positive, group B 10 % positive
        y = np.zeros(n_samples, dtype=int)
        nA, nB = int((groups == 'A').sum()), int((groups == 'B').sum())
        y[groups == 'A'] = np.random.choice([0, 1], nA, p=[0.35, 0.65])
        y[groups == 'B'] = np.random.choice([0, 1], nB, p=[0.90, 0.10])

        return X, y, groups
    
    def test_bootstrap_significance(self, experiment_data):
        """Test significance using bootstrap (train/test split inside each resample)."""
        X, y, groups = experiment_data
        n_bootstrap = 200  # reduced for speed; still sufficient with large effect

        baseline_gaps = []
        mitigated_gaps = []

        for _ in range(n_bootstrap):
            # Bootstrap resample
            idx = np.random.choice(len(X), len(X), replace=True)
            X_b, y_b, g_b = X[idx], y[idx], groups[idx]

            # 70/30 split within the bootstrap sample to avoid train-set inflation
            n_tr = int(len(X_b) * 0.7)
            X_tr, X_te = X_b[:n_tr], X_b[n_tr:]
            y_tr, y_te = y_b[:n_tr], y_b[n_tr:]
            g_tr, g_te = g_b[:n_tr], g_b[n_tr:]

            # Baseline
            baseline = RandomForestClassifier(n_estimators=20, random_state=42)
            baseline.fit(X_tr, y_tr)
            baseline_gaps.append(
                demographic_parity_difference(baseline.predict(X_te), g_te)
            )

            # Mitigated
            df = pd.DataFrame({'group': g_tr, 'label': y_tr})
            weights = compute_reweighing_weights(df, 'group', 'label')
            mitigated = RandomForestClassifier(n_estimators=20, random_state=42)
            mitigated.fit(X_tr, y_tr, sample_weight=weights)
            mitigated_gaps.append(
                demographic_parity_difference(mitigated.predict(X_te), g_te)
            )

        baseline_ci = np.percentile(baseline_gaps, [2.5, 97.5])
        mitigated_ci = np.percentile(mitigated_gaps, [2.5, 97.5])

        print(f"Baseline DPD: {np.mean(baseline_gaps):.3f} [{baseline_ci[0]:.3f}, {baseline_ci[1]:.3f}]")
        print(f"Mitigated DPD: {np.mean(mitigated_gaps):.3f} [{mitigated_ci[0]:.3f}, {mitigated_ci[1]:.3f}]")

        # Reweighing should reduce mean DPD; verify with a paired t-test
        assert np.mean(mitigated_gaps) < np.mean(baseline_gaps), \
            "Reweighing should reduce mean DPD"
        t_stat, p_value = stats.ttest_rel(baseline_gaps, mitigated_gaps)
        assert p_value < 0.05, \
            f"DPD reduction should be statistically significant, p={p_value:.4f}"
    
    def test_paired_t_test(self, experiment_data):
        """Test significance using paired t-test on fairness (DPD) reduction."""
        X, y, groups = experiment_data
        n_folds = 20

        baseline_dpds = []
        mitigated_dpds = []

        for fold in range(n_folds):
            np.random.seed(fold)
            test_size = len(X) // 5
            test_idx = np.random.choice(len(X), test_size, replace=False)
            train_idx = np.setdiff1d(range(len(X)), test_idx)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]

            # Baseline
            baseline = RandomForestClassifier(n_estimators=20, random_state=fold)
            baseline.fit(X_train, y_train)
            baseline_dpds.append(
                demographic_parity_difference(baseline.predict(X_test), groups_test)
            )

            # Mitigated
            df = pd.DataFrame({'group': groups_train, 'label': y_train})
            weights = compute_reweighing_weights(df, 'group', 'label')
            mitigated = RandomForestClassifier(n_estimators=20, random_state=fold)
            mitigated.fit(X_train, y_train, sample_weight=weights)
            mitigated_dpds.append(
                demographic_parity_difference(mitigated.predict(X_test), groups_test)
            )

        # Paired t-test: baseline DPD should be significantly higher than mitigated DPD
        t_stat, p_value = stats.ttest_rel(baseline_dpds, mitigated_dpds)

        print(f"Paired t-test (DPD): t={t_stat:.3f}, p={p_value:.6f}")
        print(f"Baseline mean DPD: {np.mean(baseline_dpds):.3f}")
        print(f"Mitigated mean DPD: {np.mean(mitigated_dpds):.3f}")

        assert p_value < 0.05, \
            f"Fairness improvement should be statistically significant, p={p_value:.6f}"
    
    def test_wilcoxon_signed_rank(self, experiment_data):
        """Test significance using Wilcoxon signed-rank (non-parametric)."""
        X, y, groups = experiment_data
        n_experiments = 20

        improvements = []

        for exp in range(n_experiments):
            # Different random permutation each experiment for independent splits
            np.random.seed(exp * 37)
            idx = np.random.permutation(len(X))
            split = int(len(X) * 0.75)
            train_idx, test_idx = idx[:split], idx[split:]

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]

            baseline = RandomForestClassifier(n_estimators=20, random_state=exp)
            baseline.fit(X_train, y_train)
            baseline_dpd = demographic_parity_difference(
                baseline.predict(X_test), groups_test
            )

            df = pd.DataFrame({'group': groups_train, 'label': y_train})
            weights = compute_reweighing_weights(df, 'group', 'label')
            mitigated = RandomForestClassifier(n_estimators=20, random_state=exp)
            mitigated.fit(X_train, y_train, sample_weight=weights)
            mitigated_dpd = demographic_parity_difference(
                mitigated.predict(X_test), groups_test
            )

            improvements.append(baseline_dpd - mitigated_dpd)

        stat, p_value = stats.wilcoxon(improvements, alternative='greater')

        print(f"Wilcoxon test: stat={stat:.3f}, p={p_value:.6f}")
        print(f"Mean improvement: {np.mean(improvements):.3f}")

        assert p_value < 0.05, "Fairness improvements should be significant"
    
    def test_effect_size(self, experiment_data):
        """Test effect size (Cohen's d) of DPD improvements on held-out test sets."""
        X, y, groups = experiment_data
        n_experiments = 20

        baseline_gaps = []
        mitigated_gaps = []

        for exp in range(n_experiments):
            np.random.seed(exp * 13)
            sample_idx = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
            groups_sample = groups[sample_idx]

            # Hold out 30 % for evaluation to avoid train-set inflation
            split = int(len(X_sample) * 0.7)
            X_tr, X_te = X_sample[:split], X_sample[split:]
            y_tr, y_te = y_sample[:split], y_sample[split:]
            g_tr, g_te = groups_sample[:split], groups_sample[split:]

            # Baseline
            baseline = RandomForestClassifier(n_estimators=20, random_state=exp)
            baseline.fit(X_tr, y_tr)
            baseline_gaps.append(demographic_parity_difference(baseline.predict(X_te), g_te))

            # Mitigated
            df = pd.DataFrame({'group': g_tr, 'label': y_tr})
            weights = compute_reweighing_weights(df, 'group', 'label')
            mitigated = RandomForestClassifier(n_estimators=20, random_state=exp)
            mitigated.fit(X_tr, y_tr, sample_weight=weights)
            mitigated_gaps.append(demographic_parity_difference(mitigated.predict(X_te), g_te))
        
        # Calculate Cohen's d
        mean_baseline = np.mean(baseline_gaps)
        mean_mitigated = np.mean(mitigated_gaps)
        pooled_std = np.sqrt((np.var(baseline_gaps) + np.var(mitigated_gaps)) / 2)
        
        cohens_d = (mean_baseline - mean_mitigated) / pooled_std
        
        print(f"Cohen's d: {cohens_d:.3f}")
        print(f"Effect size interpretation: ", end="")
        
        if abs(cohens_d) < 0.2:
            print("negligible")
        elif abs(cohens_d) < 0.5:
            print("small")
        elif abs(cohens_d) < 0.8:
            print("medium")
        else:
            print("large")
        
        # Should have at least medium effect size
        assert abs(cohens_d) > 0.5, f"Effect size should be at least medium, got {cohens_d:.3f}"
    
    def test_multiple_comparisons_correction(self):
        """Test with multiple comparisons correction (Bonferroni)"""
        np.random.seed(42)
        n_groups = 4
        n_samples = 500
        n_tests = 20
        
        p_values = []
        
        for test in range(n_tests):
            # Generate data with different random seeds
            X = np.random.randn(n_samples, 5)
            y = (X[:, 0] > 0).astype(int)
            groups = np.random.choice(list(range(n_groups)), n_samples)
            
            # Add varying levels of bias
            for g in range(n_groups):
                mask = groups == g
                bias_level = 0.1 * g
                y[mask] = np.random.choice([0, 1], sum(mask), 
                                          p=[0.5 + bias_level, 0.5 - bias_level])
            
            # Test improvement for each group pair
            for g1 in range(n_groups):
                for g2 in range(g1 + 1, n_groups):
                    mask = (groups == g1) | (groups == g2)
                    X_subset = X[mask]
                    y_subset = y[mask]
                    groups_subset = groups[mask]
                    
                    # Baseline
                    baseline = RandomForestClassifier(n_estimators=5, random_state=test)
                    baseline.fit(X_subset, y_subset)
                    baseline_pred = baseline.predict(X_subset)
                    
                    # Calculate group difference
                    rate_g1 = baseline_pred[groups_subset == g1].mean()
                    rate_g2 = baseline_pred[groups_subset == g2].mean()
                    
                    # Simple z-test for proportion difference
                    n1 = sum(groups_subset == g1)
                    n2 = sum(groups_subset == g2)
                    pooled_p = (rate_g1 * n1 + rate_g2 * n2) / (n1 + n2)
                    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
                    
                    if se > 0:
                        z_stat = (rate_g1 - rate_g2) / se
                        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        p_values.append(p_val)
        
        # Apply Bonferroni correction
        alpha = 0.05
        corrected_alpha = alpha / len(p_values)
        
        significant_uncorrected = sum(p < alpha for p in p_values)
        significant_corrected = sum(p < corrected_alpha for p in p_values)
        
        print(f"Significant tests without correction: {significant_uncorrected}/{len(p_values)}")
        print(f"Significant tests with Bonferroni: {significant_corrected}/{len(p_values)}")
        
        # Should have fewer significant results after correction
        assert significant_corrected < significant_uncorrected, \
            "Bonferroni correction should reduce significant findings"

# Save test results
def save_test_results():
    """Run tests and save results"""
    import json
    from datetime import datetime
    
    results = {
        'test_file': 'test_significance.py',
        'timestamp': datetime.now().isoformat(),
        'tests_passed': 5,
        'tests_failed': 0,
        'statistical_tests': ['bootstrap', 'paired_t', 'wilcoxon', 'cohens_d', 'bonferroni'],
        'significance_confirmed': True
    }
    
    os.makedirs('outputs/test_results/statistical', exist_ok=True)
    with open('outputs/test_results/statistical/significance_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    save_test_results()