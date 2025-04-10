"""
Quick test to verify ML environment setup
"""
import sys
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version  # Fixed import
from sklearn.ensemble import RandomForestRegressor
import joblib

def test_environment():
    print("\n🔍 Testing ML Environment Setup...")
    print("-" * 40)
    
    # Test NumPy
    print("✓ NumPy:", np.__version__)
    test_array = np.array([1, 2, 3])
    
    # Test Pandas
    print("✓ Pandas:", pd.__version__)
    test_df = pd.DataFrame({'A': [1, 2, 3]})
    
    # Test Scikit-learn
    print("✓ Scikit-learn:", sklearn_version)  # Fixed reference
    rf = RandomForestRegressor(n_estimators=10)
    
    # Test Joblib
    print("✓ Joblib:", joblib.__version__)
    
    print("\n📍 Python version:", sys.version.split()[0])
    print("-" * 40)
    print("🎉 All ML packages working correctly!")

if __name__ == "__main__":
    test_environment()