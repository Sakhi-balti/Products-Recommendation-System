"""
Data Loader Module
Handles loading and basic operations for the e-commerce product dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
import sys

# Setup basic logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")


class DataLoader:
    """Load and manage e-commerce product data."""
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to raw data directory
        """
        self.data_path = Path(data_path)
        self.products_df = None
        
    def load_products(self, filename: str = "data.tsv") -> pd.DataFrame:
        """
        Load product catalog from TSV file.
        
        Args:
            filename: Name of the TSV file
            
        Returns:
            DataFrame with product data
        """
        filepath = self.data_path / filename
        logger.info(f"Loading product data from {filepath}")
        
        try:
            # Read TSV file
            df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
            logger.info(f"Successfully loaded {len(df)} products")
            
            # Display basic info
            logger.info(f"Columns: {len(df.columns)}")
            logger.info(f"Unique Products: {df['Product Id'].nunique()}")
            logger.info(f"Unique Brands: {df['Product Brand'].nunique()}")
            logger.info(f"Unique Categories: {df['Product Category'].nunique()}")
            
            self.products_df = df
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            logger.error("Please ensure your data.tsv file is in the data/raw/ directory")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_product_by_id(self, product_id: str) -> Optional[pd.Series]:
        """
        Get product details by ID.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Product data as Series or None
        """
        if self.products_df is None:
            logger.warning("No data loaded. Call load_products() first.")
            return None
        
        product = self.products_df[self.products_df['Product Id'] == product_id]
        
        if len(product) == 0:
            logger.warning(f"Product {product_id} not found")
            return None
        
        return product.iloc[0]
    
    def get_products_by_category(self, category: str) -> pd.DataFrame:
        """
        Get all products in a category.
        
        Args:
            category: Category name (can be partial match)
            
        Returns:
            DataFrame with products in category
        """
        if self.products_df is None:
            logger.warning("No data loaded. Call load_products() first.")
            return pd.DataFrame()
        
        # Case-insensitive partial match
        mask = self.products_df['Product Category'].str.contains(
            category, 
            case=False, 
            na=False
        )
        
        products = self.products_df[mask]
        logger.info(f"Found {len(products)} products in category '{category}'")
        
        return products
    
    def get_products_by_brand(self, brand: str) -> pd.DataFrame:
        """
        Get all products from a brand.
        
        Args:
            brand: Brand name
            
        Returns:
            DataFrame with products from brand
        """
        if self.products_df is None:
            logger.warning("No data loaded. Call load_products() first.")
            return pd.DataFrame()
        
        products = self.products_df[
            self.products_df['Product Brand'].str.lower() == brand.lower()
        ]
        
        logger.info(f"Found {len(products)} products from brand '{brand}'")
        
        return products
    
    def search_products(self, query: str, limit: int = 20) -> pd.DataFrame:
        """
        Search products by name or description.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            DataFrame with matching products
        """
        if self.products_df is None:
            logger.warning("No data loaded. Call load_products() first.")
            return pd.DataFrame()
        
        # Search in product name and description
        mask = (
            self.products_df['Product Name'].str.contains(query, case=False, na=False) |
            self.products_df['Product Description'].str.contains(query, case=False, na=False)
        )
        
        results = self.products_df[mask].head(limit)
        logger.info(f"Found {len(results)} products matching '{query}'")
        
        return results
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the dataset.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.products_df is None:
            logger.warning("No data loaded. Call load_products() first.")
            return {}
        
        df = self.products_df
        
        summary = {
            'total_products': len(df),
            'unique_products': df['Product Id'].nunique(),
            'unique_brands': df['Product Brand'].nunique(),
            'unique_categories': df['Product Category'].nunique(),
            'products_with_ratings': df['Product Rating'].notna().sum(),
            'products_with_reviews': df['Product Reviews Count'].notna().sum(),
            'avg_rating': df['Product Rating'].mean(),
            'avg_price': df['Product Price'].mean(),
            'min_price': df['Product Price'].min(),
            'max_price': df['Product Price'].max(),
        }
        
        return summary
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


# Example usage
if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    # Load data
    df = loader.load_products("data.tsv")
    
    # Get summary
    summary = loader.get_data_summary()
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    for key, value in summary.items():
        print(f"{key:30s}: {value}")
    
    # Test search
    print("\n" + "="*60)
    print("TESTING SEARCH")
    print("="*60)
    results = loader.search_products("shampoo", limit=5)
    print(results[['Product Name', 'Product Brand', 'Product Price']].to_string())
