"""
Popularity-Based Recommender
Recommends products based on popularity metrics (ratings, reviews, etc.)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
import sys

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

# Import base class (adjust import based on your structure)
try:
    from .base_recommender import BaseRecommender
except:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from base_recommender import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommend products based on popularity scores."""
    
    def __init__(self):
        """Initialize popularity-based recommender."""
        super().__init__(name="popularity")
        self.popularity_scores = None
        self.category_rankings = {}
        
    def fit(self, products_df: pd.DataFrame) -> 'PopularityRecommender':
        """
        Calculate popularity scores for all products.
        
        Args:
            products_df: DataFrame with product data (must include popularity_score)
            
        Returns:
            Self for method chaining
        """
        logger.info("Training popularity-based recommender...")
        
        self.products_df = products_df.copy()
        
        # Check if popularity_score exists (should be created by preprocessor)
        if 'popularity_score' not in products_df.columns:
            logger.warning("popularity_score not found. Calculating now...")
            self._calculate_popularity_scores()
        
        # Create overall popularity ranking
        self.popularity_scores = self.products_df[[
            'Product Id', 'Product Name', 'Product Brand', 
            'Product Category', 'popularity_score'
        ]].copy()
        
        self.popularity_scores = self.popularity_scores.sort_values(
            'popularity_score', 
            ascending=False
        )
        
        # Create category-wise rankings
        self._create_category_rankings()
        
        self.is_fitted = True
        logger.info(f"Fitted popularity model with {len(self.products_df)} products")
        
        return self
    
    def _calculate_popularity_scores(self):
        """Calculate popularity scores if not already present."""
        # Normalize rating (0-5 scale)
        self.products_df['normalized_rating'] = (
            self.products_df['Product Rating'].fillna(0) / 5.0
        )
        
        # Normalize review count
        max_reviews = self.products_df['Product Reviews Count'].max()
        if max_reviews > 0:
            self.products_df['normalized_reviews'] = (
                self.products_df['Product Reviews Count'] / max_reviews
            )
        else:
            self.products_df['normalized_reviews'] = 0
        
        # Popularity score: 60% rating, 40% review count
        self.products_df['popularity_score'] = (
            0.6 * self.products_df['normalized_rating'] + 
            0.4 * self.products_df['normalized_reviews']
        )
    
    def _create_category_rankings(self):
        """Create popularity rankings within each category."""
        for category in self.products_df['Product Category'].unique():
            if pd.isna(category) or category == '':
                continue
            
            category_products = self.products_df[
                self.products_df['Product Category'] == category
            ].copy()
            
            category_products = category_products.sort_values(
                'popularity_score', 
                ascending=False
            )
            
            self.category_rankings[category] = category_products['Product Id'].tolist()
    
    def recommend(
        self, 
        product_id: str = None,
        n_recommendations: int = 10,
        category: str = None,
        min_rating: float = None,
        max_price: float = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get popular product recommendations.
        
        Args:
            product_id: Not used in popularity (for interface consistency)
            n_recommendations: Number of products to recommend
            category: Filter by category (optional)
            min_rating: Minimum rating filter (optional)
            max_price: Maximum price filter (optional)
            **kwargs: Additional filters
            
        Returns:
            List of popular products
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Start with all products
        candidates = self.products_df.copy()
        
        # Apply filters
        if category:
            candidates = candidates[
                candidates['Product Category'].str.contains(category, case=False, na=False)
            ]
        
        if min_rating:
            candidates = candidates[
                candidates['Product Rating'] >= min_rating
            ]
        
        if max_price:
            candidates = candidates[
                candidates['Product Price'] <= max_price
            ]
        
        # Sort by popularity
        candidates = candidates.sort_values('popularity_score', ascending=False)
        
        # Get top N
        top_products = candidates.head(n_recommendations)
        
        # Format recommendations
        recommendations = self.format_recommendations(
            top_products['Product Id'].tolist()
        )
        
        # Add popularity score to results
        for i, rec in enumerate(recommendations):
            product_data = top_products.iloc[i]
            rec['popularity_score'] = product_data['popularity_score']
            rec['rank'] = i + 1
        
        logger.info(f"Generated {len(recommendations)} popular product recommendations")
        
        return recommendations
    
    def get_top_products_by_category(
        self, 
        category: str,
        n_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top products in a specific category.
        
        Args:
            category: Category name
            n_recommendations: Number of products to return
            
        Returns:
            List of top products in category
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Find exact or partial match
        category_products = self.products_df[
            self.products_df['Product Category'].str.contains(
                category, case=False, na=False
            )
        ]
        
        if len(category_products) == 0:
            logger.warning(f"No products found in category: {category}")
            return []
        
        # Sort by popularity
        category_products = category_products.sort_values(
            'popularity_score', 
            ascending=False
        )
        
        # Get top N
        top_products = category_products.head(n_recommendations)
        
        recommendations = self.format_recommendations(
            top_products['Product Id'].tolist()
        )
        
        # Add rank
        for i, rec in enumerate(recommendations):
            rec['category_rank'] = i + 1
        
        return recommendations
    
    def get_trending_products(
        self, 
        n_recommendations: int = 10,
        min_reviews: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get trending products (high rating + reasonable review count).
        
        Args:
            n_recommendations: Number of products to return
            min_reviews: Minimum number of reviews
            
        Returns:
            List of trending products
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Filter for products with sufficient reviews
        trending = self.products_df[
            self.products_df['Product Reviews Count'] >= min_reviews
        ]
        
        # Sort by popularity
        trending = trending.sort_values('popularity_score', ascending=False)
        
        # Get top N
        top_trending = trending.head(n_recommendations)
        
        recommendations = self.format_recommendations(
            top_trending['Product Id'].tolist()
        )
        
        return recommendations
    
    def get_popularity_stats(self) -> Dict[str, Any]:
        """
        Get statistics about popularity distribution.
        
        Returns:
            Dictionary with popularity statistics
        """
        if not self.is_fitted:
            return {}
        
        return {
            'total_products': len(self.products_df),
            'avg_popularity_score': self.products_df['popularity_score'].mean(),
            'median_popularity_score': self.products_df['popularity_score'].median(),
            'products_with_ratings': self.products_df['Product Rating'].notna().sum(),
            'avg_rating': self.products_df['Product Rating'].mean(),
            'total_categories': len(self.category_rankings),
            'avg_products_per_category': len(self.products_df) / max(len(self.category_rankings), 1)
        }


# Example usage
if __name__ == "__main__":
    # Test with sample data
    print("\n" + "="*70)
    print("TESTING POPULARITY RECOMMENDER")
    print("="*70)
    
    # Load processed data
    try:
        df = pd.read_csv("data/processed/products_processed.csv")
        print(f"\n✅ Loaded {len(df)} products")
        
        # Train model
        model = PopularityRecommender()
        model.fit(df)
        
        # Get overall popular products
        print("\n" + "-"*70)
        print("TOP 10 POPULAR PRODUCTS")
        print("-"*70)
        recommendations = model.recommend(n_recommendations=10)
        for rec in recommendations:
            print(f"{rec['rank']:2d}. {rec['name'][:50]:50s} | Score: {rec['popularity_score']:.3f}")
        
        # Get popular in category
        print("\n" + "-"*70)
        print("TOP 5 POPULAR IN 'BEAUTY' CATEGORY")
        print("-"*70)
        beauty_recs = model.get_top_products_by_category("Beauty", n_recommendations=5)
        for rec in beauty_recs:
            print(f"{rec['category_rank']:2d}. {rec['name'][:50]:50s}")
        
        # Get stats
        print("\n" + "-"*70)
        print("POPULARITY STATISTICS")
        print("-"*70)
        stats = model.get_popularity_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.2f}")
            else:
                print(f"  {key:30s}: {value}")
        
        # Save model
        model.save_model()
        print("\n✅ Model saved to models/popularity/")
        
    except FileNotFoundError:
        print("\n❌ Error: products_processed.csv not found")
        print("   Run test_data_pipeline.py first to create processed data")
