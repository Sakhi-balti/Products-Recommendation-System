"""
Data Preprocessor Module
Cleans and prepares data for recommendation models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from loguru import logger
import re
import sys

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")


class DataPreprocessor:
    """Clean and transform product data for recommendation models."""
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        self.products_clean = None
        
    def clean_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare product data.
        
        Args:
            df: Raw product DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # 1. Handle missing values
        logger.info("Handling missing values...")
        
        # Fill missing text fields with empty string
        text_columns = ['Product Name', 'Product Brand', 'Product Category', 
                       'Product Description', 'Product Tags']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('')
        
        # Fill missing numeric fields
        if 'Product Price' in df_clean.columns:
            df_clean['Product Price'] = pd.to_numeric(df_clean['Product Price'], errors='coerce')
            df_clean['Product Price'] = df_clean['Product Price'].fillna(df_clean['Product Price'].median())
        
        if 'Product Rating' in df_clean.columns:
            df_clean['Product Rating'] = pd.to_numeric(df_clean['Product Rating'], errors='coerce')
            # Don't fill ratings - keep NaN for products without ratings
        
        if 'Product Reviews Count' in df_clean.columns:
            df_clean['Product Reviews Count'] = pd.to_numeric(df_clean['Product Reviews Count'], errors='coerce')
            df_clean['Product Reviews Count'] = df_clean['Product Reviews Count'].fillna(0)
        
        # 2. Remove duplicates based on Product Id
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['Product Id'], keep='first')
        removed = initial_count - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate products")
        
        # 3. Filter out products with missing critical fields
        df_clean = df_clean[df_clean['Product Name'].str.len() > 0]
        df_clean = df_clean[df_clean['Product Id'].notna()]
        
        logger.info(f"Cleaned data: {len(df_clean)} products remaining")
        
        self.products_clean = df_clean
        return df_clean
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create combined text features for content-based filtering.
        
        Args:
            df: Product DataFrame
            
        Returns:
            DataFrame with combined text features
        """
        logger.info("Creating text features for content-based filtering...")
        df_features = df.copy()
        
        # Combine all text fields
        df_features['combined_features'] = (
            df_features['Product Name'].astype(str) + ' ' +
            df_features['Product Brand'].astype(str) + ' ' +
            df_features['Product Category'].astype(str) + ' ' +
            df_features['Product Description'].astype(str) + ' ' +
            df_features['Product Tags'].astype(str)
        )
        
        # Clean text: lowercase, remove special characters
        df_features['combined_features'] = df_features['combined_features'].apply(
            self._clean_text
        )
        
        logger.info("Text features created")
        return df_features
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and extra spaces.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def create_popularity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for popularity-based recommendations.
        
        Args:
            df: Product DataFrame
            
        Returns:
            DataFrame with popularity features
        """
        logger.info("Creating popularity features...")
        df_pop = df.copy()
        
        # Calculate popularity score
        # Normalize rating (0-5 scale) and review count
        max_reviews = df_pop['Product Reviews Count'].max()
        
        if max_reviews > 0:
            df_pop['normalized_reviews'] = df_pop['Product Reviews Count'] / max_reviews
        else:
            df_pop['normalized_reviews'] = 0
        
        # Normalize rating (assuming 1-5 scale)
        df_pop['normalized_rating'] = df_pop['Product Rating'].fillna(0) / 5.0
        
        # Popularity score: weighted combination
        # 60% rating, 40% review count
        df_pop['popularity_score'] = (
            0.6 * df_pop['normalized_rating'] + 
            0.4 * df_pop['normalized_reviews']
        )
        
        # Add rank within category
        df_pop['category_rank'] = df_pop.groupby('Product Category')['popularity_score'].rank(
            ascending=False, method='dense'
        )
        
        logger.info("Popularity features created")
        return df_pop
    
    def extract_category_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract category levels from hierarchical category string.
        
        Args:
            df: Product DataFrame
            
        Returns:
            DataFrame with category levels
        """
        logger.info("Extracting category hierarchy...")
        df_cat = df.copy()
        
        # Split category by '>' to get hierarchy
        # Example: "Beauty > Hair Care > Shampoo" -> Level 1, 2, 3
        
        def extract_levels(category):
            if pd.isna(category) or category == '':
                return ['', '', '']
            
            levels = [c.strip() for c in category.split('>')]
            
            # Ensure at least 3 levels (pad with empty strings)
            while len(levels) < 3:
                levels.append('')
            
            return levels[:3]  # Take first 3 levels
        
        categories = df_cat['Product Category'].apply(extract_levels)
        
        df_cat['category_level_1'] = [c[0] for c in categories]
        df_cat['category_level_2'] = [c[1] for c in categories]
        df_cat['category_level_3'] = [c[2] for c in categories]
        
        logger.info("Category hierarchy extracted")
        return df_cat
    
    def prepare_for_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for recommendation system.
        
        Args:
            df: Raw product DataFrame
            
        Returns:
            Fully preprocessed DataFrame
        """
        logger.info("="*60)
        logger.info("STARTING FULL PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Clean data
        df_processed = self.clean_products(df)
        
        # Step 2: Create text features
        df_processed = self.create_text_features(df_processed)
        
        # Step 3: Create popularity features
        df_processed = self.create_popularity_features(df_processed)
        
        # Step 4: Extract category hierarchy
        df_processed = self.extract_category_hierarchy(df_processed)
        
        logger.info("="*60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"Final dataset: {len(df_processed)} products")
        logger.info(f"Features created: combined_features, popularity_score, category_rank")
        
        return df_processed
    
    def get_preprocessing_summary(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> dict:
        """
        Get summary of preprocessing steps.
        
        Args:
            df_original: Original DataFrame
            df_processed: Processed DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'original_rows': len(df_original),
            'processed_rows': len(df_processed),
            'rows_removed': len(df_original) - len(df_processed),
            'products_with_ratings': df_processed['Product Rating'].notna().sum(),
            'avg_popularity_score': df_processed['popularity_score'].mean(),
            'unique_categories_level1': df_processed['category_level_1'].nunique(),
            'unique_categories_level2': df_processed['category_level_2'].nunique(),
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df_raw = loader.load_products("data.tsv")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_for_recommendations(df_raw)
    
    # Show summary
    summary = preprocessor.get_preprocessing_summary(df_raw, df_processed)
    
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    for key, value in summary.items():
        print(f"{key:30s}: {value}")
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE PROCESSED DATA")
    print("="*60)
    print(df_processed[['Product Name', 'Product Brand', 'popularity_score', 'category_level_1']].head(10))
    
    # Save
    loader.save_processed_data(df_processed, "products_processed.csv")
