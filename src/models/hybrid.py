"""
Hybrid Recommender
Combines popularity-based and content-based recommendations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
import sys

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

# Import base class and other recommenders
try:
    from .base_recommender import BaseRecommender
    from .popularity_recommender import PopularityRecommender
    from .content_based import ContentBasedRecommender
except:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from base_recommender import BaseRecommender
    from popularity_recommender import PopularityRecommender
    from content_based import ContentBasedRecommender


class HybridRecommender(BaseRecommender):
    """Combine multiple recommendation strategies."""
    
    def __init__(
        self,
        popularity_weight: float = 0.4,
        content_weight: float = 0.6
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            popularity_weight: Weight for popularity-based recommendations
            content_weight: Weight for content-based recommendations
        """
        super().__init__(name="hybrid")
        
        # Normalize weights
        total = popularity_weight + content_weight
        self.popularity_weight = popularity_weight / total
        self.content_weight = content_weight / total
        
        # Initialize sub-models
        self.popularity_model = PopularityRecommender()
        self.content_model = ContentBasedRecommender()
        
        logger.info(f"Hybrid weights: Popularity={self.popularity_weight:.2f}, Content={self.content_weight:.2f}")
    
    def fit(self, products_df: pd.DataFrame) -> 'HybridRecommender':
        """
        Train all component models.
        
        Args:
            products_df: DataFrame with product data
            
        Returns:
            Self for method chaining
        """
        logger.info("Training hybrid recommender...")
        
        self.products_df = products_df.copy()
        
        # Train popularity model
        logger.info("Training popularity component...")
        self.popularity_model.fit(products_df)
        
        # Train content-based model
        logger.info("Training content-based component...")
        self.content_model.fit(products_df)
        
        self.is_fitted = True
        logger.info("Hybrid model training complete")
        
        return self
    
    def recommend(
        self, 
        product_id: str = None,
        n_recommendations: int = 10,
        strategy: str = 'hybrid',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations.
        
        Args:
            product_id: Product ID to base recommendations on (optional)
            n_recommendations: Number of products to recommend
            strategy: 'hybrid', 'popularity', or 'content'
            **kwargs: Additional parameters
            
        Returns:
            List of recommended products
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Strategy selection
        if strategy == 'popularity':
            return self.popularity_model.recommend(
                n_recommendations=n_recommendations,
                **kwargs
            )
        elif strategy == 'content':
            if product_id is None:
                raise ValueError("product_id required for content-based recommendations")
            return self.content_model.recommend(
                product_id=product_id,
                n_recommendations=n_recommendations,
                **kwargs
            )
        elif strategy == 'hybrid':
            return self._hybrid_recommend(
                product_id=product_id,
                n_recommendations=n_recommendations,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _hybrid_recommend(
        self,
        product_id: Optional[str] = None,
        n_recommendations: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations by combining both models.
        
        Args:
            product_id: Optional product ID for content-based component
            n_recommendations: Number of recommendations
            **kwargs: Additional parameters
            
        Returns:
            List of hybrid recommendations
        """
        # Get more recommendations from each model to combine
        n_fetch = n_recommendations * 3
        
        # Get popularity recommendations
        pop_recs = self.popularity_model.recommend(
            n_recommendations=n_fetch,
            **kwargs
        )
        
        # Get content recommendations if product_id provided
        if product_id:
            content_recs = self.content_model.recommend(
                product_id=product_id,
                n_recommendations=n_fetch
            )
        else:
            content_recs = []
        
        # Combine recommendations
        combined_scores = {}
        
        # Add popularity scores
        for i, rec in enumerate(pop_recs):
            pid = rec['product_id']
            # Score based on rank (higher rank = higher score)
            rank_score = 1.0 / (i + 1)
            combined_scores[pid] = {
                'product_id': pid,
                'popularity_score': rank_score * self.popularity_weight,
                'content_score': 0.0,
                'details': rec
            }
        
        # Add content scores
        for i, rec in enumerate(content_recs):
            pid = rec['product_id']
            rank_score = 1.0 / (i + 1)
            
            if pid in combined_scores:
                combined_scores[pid]['content_score'] = rank_score * self.content_weight
            else:
                combined_scores[pid] = {
                    'product_id': pid,
                    'popularity_score': 0.0,
                    'content_score': rank_score * self.content_weight,
                    'details': rec
                }
        
        # Calculate final scores
        for pid in combined_scores:
            combined_scores[pid]['final_score'] = (
                combined_scores[pid]['popularity_score'] +
                combined_scores[pid]['content_score']
            )
        
        # Sort by final score
        sorted_products = sorted(
            combined_scores.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        # Format recommendations
        recommendations = []
        for item in sorted_products[:n_recommendations]:
            rec = item['details'].copy()
            rec['hybrid_score'] = item['final_score']
            rec['popularity_contribution'] = item['popularity_score']
            rec['content_contribution'] = item['content_score']
            recommendations.append(rec)
        
        logger.info(f"Generated {len(recommendations)} hybrid recommendations")
        
        return recommendations
    
    def recommend_cold_start(
        self,
        n_recommendations: int = 10,
        category: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Recommend products for new users (cold start problem).
        Uses popularity-based approach.
        
        Args:
            n_recommendations: Number of products to recommend
            category: Optional category filter
            **kwargs: Additional filters
            
        Returns:
            List of popular products
        """
        logger.info("Using cold-start recommendations (popularity-based)")
        
        return self.popularity_model.recommend(
            n_recommendations=n_recommendations,
            category=category,
            **kwargs
        )
    
    def recommend_similar_to_cart(
        self,
        cart_product_ids: List[str],
        n_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommend products similar to items in shopping cart.
        
        Args:
            cart_product_ids: List of product IDs in cart
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended products
        """
        if not cart_product_ids:
            return self.recommend_cold_start(n_recommendations=n_recommendations)
        
        # Use content-based for multiple products
        return self.content_model.recommend_for_multiple(
            product_ids=cart_product_ids,
            n_recommendations=n_recommendations,
            aggregation='mean'
        )
    
    def update_weights(
        self,
        popularity_weight: float,
        content_weight: float
    ):
        """
        Update the weights for combining models.
        
        Args:
            popularity_weight: New weight for popularity
            content_weight: New weight for content
        """
        # Normalize
        total = popularity_weight + content_weight
        self.popularity_weight = popularity_weight / total
        self.content_weight = content_weight / total
        
        logger.info(f"Updated weights: Popularity={self.popularity_weight:.2f}, Content={self.content_weight:.2f}")
    
    def save_model(self, filepath: str = None):
        """Save hybrid model and all components."""
        # Save main model
        super().save_model(filepath)
        
        # Save components
        self.popularity_model.save_model()
        self.content_model.save_model()
        
        logger.info("Saved hybrid model and all components")


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING HYBRID RECOMMENDER")
    print("="*70)
    
    try:
        # Load processed data
        df = pd.read_csv("data/processed/products_processed.csv")
        print(f"\n✅ Loaded {len(df)} products")
        
        # Train hybrid model
        model = HybridRecommender(
            popularity_weight=0.4,
            content_weight=0.6
        )
        model.fit(df)
        
        # Test 1: Cold start (no product_id)
        print("\n" + "-"*70)
        print("TEST 1: COLD START RECOMMENDATIONS (Popular Products)")
        print("-"*70)
        cold_start_recs = model.recommend_cold_start(n_recommendations=5)
        for i, rec in enumerate(cold_start_recs, 1):
            print(f"{i}. {rec['name'][:50]:50s}")
        
        # Test 2: Content-based (with product_id)
        sample_product = df.iloc[0]
        sample_id = sample_product['Product Id']
        
        print(f"\n" + "-"*70)
        print(f"TEST 2: CONTENT-BASED RECOMMENDATIONS")
        print("-"*70)
        print(f"Base Product: {sample_product['Product Name']}")
        print("-"*70)
        
        content_recs = model.recommend(
            product_id=sample_id,
            n_recommendations=5,
            strategy='content'
        )
        for i, rec in enumerate(content_recs, 1):
            print(f"{i}. {rec['name'][:50]:50s}")
        
        # Test 3: Hybrid recommendations
        print(f"\n" + "-"*70)
        print("TEST 3: HYBRID RECOMMENDATIONS")
        print("-"*70)
        
        hybrid_recs = model.recommend(
            product_id=sample_id,
            n_recommendations=5,
            strategy='hybrid'
        )
        for i, rec in enumerate(hybrid_recs, 1):
            print(f"{i}. {rec['name'][:45]:45s} | Score: {rec['hybrid_score']:.3f}")
            print(f"    Pop: {rec['popularity_contribution']:.3f} | Content: {rec['content_contribution']:.3f}")
        
        # Test 4: Cart recommendations
        cart_items = [df.iloc[0]['Product Id'], df.iloc[5]['Product Id']]
        print(f"\n" + "-"*70)
        print("TEST 4: SHOPPING CART RECOMMENDATIONS")
        print("-"*70)
        print(f"Cart Items: {len(cart_items)}")
        
        cart_recs = model.recommend_similar_to_cart(
            cart_product_ids=cart_items,
            n_recommendations=5
        )
        for i, rec in enumerate(cart_recs, 1):
            print(f"{i}. {rec['name'][:50]:50s}")
        
        # Save model
        model.save_model()
        print("\n✅ Hybrid model and components saved")
        
    except FileNotFoundError:
        print("\n❌ Error: products_processed.csv not found")
        print("   Run test_data_pipeline.py first")
