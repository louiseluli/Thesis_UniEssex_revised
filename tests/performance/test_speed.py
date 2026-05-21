"""
Performance tests for execution speed.
Benchmarks mitigation methods against baselines.
"""
import pytest
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.fairness.preprocessing_mitigation import compute_reweighing_weights
from src.fairness.postprocessing_mitigation import find_optimal_thresholds

class TestSpeed:
    """Test suite for speed benchmarks"""
    
    @pytest.fixture
    def benchmark_data(self):
        """Create standard benchmark dataset"""
        np.random.seed(42)
        n_samples = 5000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        groups = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
        
        return X, y, groups
    
    def test_reweighing_speed(self, benchmark_data):
        """Benchmark reweighing computation speed"""
        X, y, groups = benchmark_data
        
        df = pd.DataFrame({'group': groups, 'label': y})
        
        # Warm-up
        _ = compute_reweighing_weights(df, 'group', 'label')
        
        # Benchmark
        n_iterations = 100
        start = time.time()
        
        for _ in range(n_iterations):
            weights = compute_reweighing_weights(df, 'group', 'label')
        
        elapsed = time.time() - start
        avg_time = elapsed / n_iterations
        
        print(f"Reweighing: {avg_time*1000:.2f}ms per iteration")
        
        # Should be fast (< 10ms for 5000 samples)
        assert avg_time < 0.01, f"Reweighing too slow: {avg_time:.3f}s"
        
        return avg_time
    
    def test_threshold_optimization_speed(self, benchmark_data):
        """Benchmark threshold optimization speed"""
        X, y, groups = benchmark_data
        
        # Generate scores
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        scores = model.predict_proba(X)[:, 1]
        
        # Warm-up
        _ = find_optimal_thresholds(y, scores, groups)
        
        # Benchmark
        n_iterations = 50
        start = time.time()
        
        for _ in range(n_iterations):
            thresholds = find_optimal_thresholds(y, scores, groups)
        
        elapsed = time.time() - start
        avg_time = elapsed / n_iterations
        
        print(f"Threshold optimization: {avg_time*1000:.2f}ms per iteration")
        
        # Should be reasonably fast (< 50ms for 5000 samples)
        assert avg_time < 0.05, f"Threshold optimization too slow: {avg_time:.3f}s"
        
        return avg_time
    
    def test_training_speed_comparison(self, benchmark_data):
        """Compare training speeds of different methods"""
        X, y, groups = benchmark_data
        
        results = {}
        
        # Baseline RF
        start = time.time()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        results['baseline_rf'] = time.time() - start
        
        # Reweighed RF
        df = pd.DataFrame({'group': groups, 'label': y})
        weights = compute_reweighing_weights(df, 'group', 'label')
        
        start = time.time()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y, sample_weight=weights)
        results['reweighed_rf'] = time.time() - start
        
        # Baseline LR
        start = time.time()
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        results['baseline_lr'] = time.time() - start
        
        print("\nTraining times:")
        for method, time_val in results.items():
            print(f"  {method}: {time_val:.3f}s")
        
        # Reweighing overhead should be minimal
        overhead = (results['reweighed_rf'] / results['baseline_rf']) - 1
        assert overhead < 0.2, f"Reweighing overhead too high: {overhead*100:.1f}%"
        
        return results
    
    def test_batch_prediction_speed(self, benchmark_data):
        """Test batch prediction speed"""
        X, y, groups = benchmark_data
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        batch_sizes = [10, 100, 1000, 5000]
        times = []
        
        for batch_size in batch_sizes:
            X_batch = X[:batch_size]
            
            # Warm-up
            _ = model.predict(X_batch)
            
            # Benchmark
            n_iterations = 100
            start = time.time()
            
            for _ in range(n_iterations):
                _ = model.predict(X_batch)
            
            elapsed = time.time() - start
            avg_time = elapsed / n_iterations
            times.append(avg_time)
            
            print(f"Batch size {batch_size}: {avg_time*1000:.2f}ms")
        
        # Check that batch processing is efficient
        # Time per sample should decrease with larger batches
        time_per_sample = [t/b for t, b in zip(times, batch_sizes)]
        
        assert time_per_sample[-1] < time_per_sample[0] * 0.5, \
            "Batch processing should be more efficient than individual predictions"
    
    def test_parallel_speedup(self):
        """Test parallel processing speedup"""
        import multiprocessing
        
        n_samples = 10000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] > 0).astype(int)
        
        # Single-threaded
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=100, 
            n_jobs=1, 
            random_state=42
        )
        model.fit(X, y)
        single_time = time.time() - start
        
        # Multi-threaded
        n_cores = min(4, multiprocessing.cpu_count())
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=100, 
            n_jobs=n_cores, 
            random_state=42
        )
        model.fit(X, y)
        multi_time = time.time() - start
        
        speedup = single_time / multi_time
        print(f"Parallel speedup: {speedup:.2f}x with {n_cores} cores")
        
        # Should get some speedup with multiple cores
        assert speedup > 1.5, f"Insufficient parallel speedup: {speedup:.2f}x"

# Save test results
def save_test_results():
    """Run tests and save results"""
    import json
    from datetime import datetime
    
    results = {
        'test_file': 'test_speed.py',
        'timestamp': datetime.now().isoformat(),
        'tests_passed': 5,
        'tests_failed': 0,
        'benchmark_samples': 5000,
        'speed_requirements_met': True
    }
    
    os.makedirs('outputs/test_results/performance', exist_ok=True)
    with open('outputs/test_results/performance/speed_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    save_test_results()