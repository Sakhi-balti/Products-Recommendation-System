"""
Base Recommender Class
Abstract base class for all recommendation models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
import sys

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")


class BaseRecommender(ABC):
    """Abstract base class for all recommendation models."""
    
    def __init__(self, name: str):
        """
        Initialize base recommender.
        
        Args:
            name: Name of the recommender model
        """
        self.name = name
        self.is_fitted = False
        self.products_df = None
        
    @abstractmethod
    def fit(self, products_df: pd.DataFrame) -> 'BaseRecommender':
        """
        Train the recommendation model.
        
        Args:
            products_df: DataFrame with product information
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def recommend(
        self, 
        product_id: str = None,
        n_recommendations: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations.
        
        Args:
            product_id: Optional product ID to base recommendations on
            n_recommendations: Number of items to recommend
            **kwargs: Additional parameters
            
        Returns:
            List of recommended products with details
        """
        pass
    
    def save_model(self, filepath: str = None):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model (default: models/{name}.pkl)
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet. Nothing to save.")
            return
        
        if filepath is None:
            filepath = f"models/{self.name}/{self.name}_model.pkl"
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseRecommender':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model instance
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_product_details(self, product_id: str) -> Dict[str, Any]:
        """
        Get details for a specific product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Dictionary with product details
        """
        if self.products_df is None:
            return {}
        
        product = self.products_df[self.products_df['Product Id'] == product_id]
        
        if len(product) == 0:
            return {}
        
        product = product.iloc[0]
        
        return {
            'product_id': product['Product Id'],
            'name': product['Product Name'],
            'brand': product.get('Product Brand', 'Unknown'),
            'category': product.get('Product Category', 'Unknown'),
            'price': product.get('Product Price', 0),
            'rating': product.get('Product Rating', None),
            'reviews_count': product.get('Product Reviews Count', 0),
            'image_url': product.get('Product Image Url', ''),
            'product_url': product.get('Product Url', '')
        }
    
    def format_recommendations(
        self, 
        product_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Format list of product IDs into detailed recommendations.
        
        Args:
            product_ids: List of product IDs
            
        Returns:
            List of product details
        """
        recommendations = []
        
        for product_id in product_ids:
            details = self.get_product_details(product_id)
            if details:
                recommendations.append(details)
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'n_products': len(self.products_df) if self.products_df is not None else 0
        }
