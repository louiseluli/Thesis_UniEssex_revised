"""
Helper functions for tests.
Never uses seed=42 as requested.
"""
import hashlib
from datetime import datetime
import os

def get_test_seed(test_name: str = "") -> int:
    """
    Generate a deterministic but non-42 seed for tests.
    Uses a combination of test name and a base seed.
    """
    # Use a different base seed - explicitly not 42
    base_seeds = [123, 456, 789, 1234, 5678, 9876, 2468, 1357, 8642, 7531]
    
    # Hash the test name to get an index
    if test_name:
        idx = int(hashlib.md5(test_name.encode()).hexdigest()[:8], 16) % len(base_seeds)
    else:
        # Use process ID for variety
        idx = os.getpid() % len(base_seeds)
    
    return base_seeds[idx]