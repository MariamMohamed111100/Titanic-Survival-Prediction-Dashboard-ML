"""
Test cases for Titanic Survival Prediction App
Run these tests to verify the app functionality
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

def test_model_loading():
    """Test if the model loads correctly"""
    try:
        model = joblib.load("../models/best_titanic_model.pkl")
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_data_loading():
    """Test if the dataset loads correctly"""
    try:
        df = pd.read_csv('../Data/tested_cleaned.csv')
        print(f"✅ Data loaded successfully - {len(df)} rows, {len(df.columns)} columns")
        
        # Check required columns
        required_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        else:
            print("✅ All required columns present")
            return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_prediction_with_all_features():
    """Test prediction with complete feature set"""
    try:
        model = joblib.load("../models/best_titanic_model.pkl")

        
        # Test cases with different passenger profiles
        test_cases = [
            {
                "name": "First class female",
                "input": {
                    'Pclass': [1],
                    'Age': [25],
                    'SibSp': [0],
                    'Parch': [0],
                    'Fare': [71.2833],
                    'Embarked': [0]
                }
            },
            {
                "name": "Third class male",
                "input": {
                    'Pclass': [3],
                    'Age': [22],
                    'SibSp': [1],
                    'Parch': [0],
                    'Fare': [7.25],
                    'Embarked': [1]
                }
            },
            {
                "name": "Second class family",
                "input": {
                    'Pclass': [2],
                    'Age': [38],
                    'SibSp': [1],
                    'Parch': [1],
                    'Fare': [26.0],
                    'Embarked': [2]
                }
            }
        ]
        
        print("\n🧪 Testing predictions with complete feature sets:")
        for test_case in test_cases:
            input_df = pd.DataFrame(test_case["input"])
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            print(f"✅ {test_case['name']}: Prediction={prediction}, Survival_prob={probability[1]:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def test_missing_feature_detection():
    """Test that missing features are properly detected"""
    try:
        model = joblib.load("../models/best_titanic_model.pkl")
        
        print("\n🧪 Testing missing feature detection:")
        
        # Test with missing Age feature (this should fail gracefully)
        incomplete_input = pd.DataFrame({
            'Pclass': [1],
            'SibSp': [0],
            'Parch': [0],
            'Fare': [71.2833],
            'Embarked': [0]
            # Missing 'Age' column
        })
        
        prediction = model.predict(incomplete_input)
        print("❌ Should have failed with missing feature")
        return False
        
    except ValueError as e:
        if "feature names" in str(e).lower() or "feature" in str(e).lower():
            print("✅ Correctly detected missing feature")
            return True
        else:
            print(f"❌ Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
        return False

def test_feature_ranges():
    """Test that features are within expected ranges"""
    try:
        df = pd.read_csv('../Data/tested_cleaned.csv')
        print("\n🧪 Testing feature ranges:")
        
        
        # Check Pclass values
        pclass_values = df['Pclass'].unique()
        if set(pclass_values).issubset({1, 2, 3}):
            print("✅ Pclass values are valid (1/2/3)")
        else:
            print(f"❌ Unexpected Pclass values: {pclass_values}")
            return False
        
        # Check Embarked values
        embarked_values = df['Embarked'].unique()
        if set(embarked_values).issubset({0, 1, 2}):
            print("✅ Embarked values are valid (0/1/2)")
        else:
            print(f"❌ Unexpected Embarked values: {embarked_values}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Feature range test failed: {e}")
        return False

def run_all_tests():
    """Run all test cases"""
    print("🚀 Starting Titanic App Tests...\n")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Data Loading", test_data_loading),
        ("Feature Ranges", test_feature_ranges),
        ("Missing Feature Detection", test_missing_feature_detection),
        ("Complete Predictions", test_prediction_with_all_features)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ ERROR in {test_name}: {e}")
    
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()
