"""
Test Script for Data Loading and Preprocessing
Run this to verify your data pipeline works correctly.

Usage:
    python test_data_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor


def main():
    print("\n" + "="*70)
    print("TESTING DATA LOADING & PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Load Data
    print("\n[STEP 1] Loading raw data...")
    print("-"*70)
    loader = DataLoader(data_path="data/raw")
    
    try:
        df_raw = loader.load_products("data.tsv")
        print(f"✅ Successfully loaded {len(df_raw)} products")
    except FileNotFoundError:
        print("❌ ERROR: data.tsv not found in data/raw/")
        print("   Please copy your data.tsv file to: data/raw/data.tsv")
        return
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return
    
    # Step 2: Show Data Summary
    print("\n[STEP 2] Data Summary...")
    print("-"*70)
    summary = loader.get_data_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.2f}")
        else:
            print(f"  {key:30s}: {value}")
    
    # Step 3: Preprocess Data
    print("\n[STEP 3] Preprocessing data...")
    print("-"*70)
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_for_recommendations(df_raw)
    
    # Step 4: Show Preprocessing Results
    print("\n[STEP 4] Preprocessing Summary...")
    print("-"*70)
    prep_summary = preprocessor.get_preprocessing_summary(df_raw, df_processed)
    for key, value in prep_summary.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.2f}")
        else:
            print(f"  {key:30s}: {value}")
    
    # Step 5: Sample Data
    print("\n[STEP 5] Sample Processed Products...")
    print("-"*70)
    sample_cols = ['Product Name', 'Product Brand', 'Product Price', 
                   'popularity_score', 'category_level_1']
    print(df_processed[sample_cols].head(10).to_string(index=False))
    
    # Step 6: Save Processed Data
    print("\n[STEP 6] Saving processed data...")
    print("-"*70)
    loader.save_processed_data(df_processed, "products_processed.csv")
    print("✅ Saved to: data/processed/products_processed.csv")
    
    # Step 7: Test Search Functionality
    print("\n[STEP 7] Testing search functionality...")
    print("-"*70)
    search_results = loader.search_products("shampoo", limit=5)
    if len(search_results) > 0:
        print(f"✅ Found {len(search_results)} products matching 'shampoo'")
        print(search_results[['Product Name', 'Product Brand', 'Product Price']].to_string(index=False))
    else:
        print("⚠️  No products found for 'shampoo'")
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ DATA PIPELINE TEST COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Your data is cleaned and ready")
    print("  2. Processed data saved to: data/processed/products_processed.csv")
    print("  3. Ready to build recommendation models!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
