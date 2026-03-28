"""
Content-Based Recommender
Recommends products based on product feature similarity.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import sys

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

# Import base class
try:
    from .base_recommender import BaseRecommender
except:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from base_recommender import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """Recommend products based on content similarity."""
    
    def __init__(self, max_features: int = 5000):
        """
        Initialize content-based recommender.
        
        Args:
            max_features: Maximum number of features for TF-IDF
        """
        super().__init__(name="content_based")
        self.max_features = max_features
        self.vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        
    def fit(self, products_df: pd.DataFrame) -> 'ContentBasedRecommender':
        """
        Train content-based model using product features.
        
        Args:
            products_df: DataFrame with product data (must include combined_features)
            
        Returns:
            Self for method chaining
        """
        logger.info("Training content-based recommender...")
        
        self.products_df = products_df.copy()
        
        # Check for combined_features
        if 'combined_features' not in products_df.columns:
            logger.warning("combined_features not found. Creating now...")
            self._create_combined_features()
        
        # Create product ID mappings
        self.product_id_to_idx = {
            pid: idx for idx, pid in enumerate(self.products_df['Product Id'])
        }
        self.idx_to_product_id = {
            idx: pid for pid, idx in self.product_id_to_idx.items()
        }
        
        # Create TF-IDF matrix
        logger.info("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.products_df['combined_features'].fillna('')
        )
        
        # Calculate similarity matrix
        logger.info("Calculating similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        self.is_fitted = True
        logger.info(f"Fitted content-based model with {len(self.products_df)} products")
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        return self
    
    def _create_combined_features(self):
        """Create combined text features if not present."""
        self.products_df['combined_features'] = (
            self.products_df['Product Name'].astype(str) + ' ' +
            self.products_df['Product Brand'].astype(str) + ' ' +
            self.products_df['Product Category'].astype(str) + ' ' +
            self.products_df['Product Description'].astype(str)
        )
    
    def recommend(
        self, 
        product_id: str,
        n_recommendations: int = 10,
        exclude_same_product: bool = True,
        min_similarity: float = 0.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get similar product recommendations.
        
        Args:
            product_id: Product ID to find similar products for
            n_recommendations: Number of products to recommend
            exclude_same_product: Whether to exclude the input product
            min_similarity: Minimum similarity threshold
            **kwargs: Additional parameters
            
        Returns:
            List of similar products
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Check if product exists
        if product_id not in self.product_id_to_idx:
            logger.warning(f"Product {product_id} not found in catalog")
            return []
        
        # Get product index
        product_idx = self.product_id_to_idx[product_id]
        
        # Get similarity scores for this product
        similarity_scores = self.similarity_matrix[product_idx]
        
        # Create list of (index, score) pairs
        product_scores = list(enumerate(similarity_scores))
        
        # Sort by similarity score (descending)
        product_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter and collect recommendations
        recommendations = []
        for idx, score in product_scores:
            # Skip if below threshold
            if score < min_similarity:
                continue
            
            # Skip the same product if requested
            if exclude_same_product and idx == product_idx:
                continue
            
            # Get product ID
            rec_product_id = self.idx_to_product_id[idx]
            
            # Get product details
            details = self.get_product_details(rec_product_id)
            if details:
                details['similarity_score'] = float(score)
                recommendations.append(details)
            
            # Stop when we have enough
            if len(recommendations) >= n_recommendations:
                break
        
        logger.info(f"Generated {len(recommendations)} content-based recommendations")
        
        return recommendations
    
    def recommend_for_multiple(
        self,
        product_ids: List[str],
        n_recommendations: int = 10,
        aggregation: str = 'mean'
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on multiple products (user profile).
        
        Args:
            product_ids: List of product IDs
            n_recommendations: Number of products to recommend
            aggregation: How to combine scores ('mean' or 'max')
            
        Returns:
            List of recommended products
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get indices for all products
        valid_indices = []
        for pid in product_ids:
            if pid in self.product_id_to_idx:
                valid_indices.append(self.product_id_to_idx[pid])
        
        if len(valid_indices) == 0:
            logger.warning("No valid products found")
            return []
        
        # Get similarity scores for all products
        similarity_scores = self.similarity_matrix[valid_indices]
        
        # Aggregate scores
        if aggregation == 'mean':
            combined_scores = similarity_scores.mean(axis=0)
        elif aggregation == 'max':
            combined_scores = similarity_scores.max(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Create list of (index, score) pairs
        product_scores = list(enumerate(combined_scores))
        
        # Sort by score
        product_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Collect recommendations (excluding input products)
        recommendations = []
        for idx, score in product_scores:
            # Skip if this was one of the input products
            if idx in valid_indices:
                continue
            
            rec_product_id = self.idx_to_product_id[idx]
            details = self.get_product_details(rec_product_id)
            
            if details:
                details['similarity_score'] = float(score)
                recommendations.append(details)
            
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations
    
    def get_similar_by_category(
        self,
        product_id: str,
        n_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get similar products from the same category.
        
        Args:
            product_id: Product ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of similar products from same category
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get product category
        product = self.products_df[self.products_df['Product Id'] == product_id]
        if len(product) == 0:
            return []
        
        category = product.iloc[0]['Product Category']
        
        # Get all recommendations
        all_recs = self.recommend(
            product_id=product_id,
            n_recommendations=n_recommendations * 3  # Get more to filter
        )
        
        # Filter by same category
        category_recs = [
            rec for rec in all_recs 
            if rec['category'] == category
        ]
        
        return category_recs[:n_recommendations]
    
    def get_feature_importance(self, product_id: str, top_n: int = 10) -> List[tuple]:
        """
        Get most important features for a product.
        
        Args:
            product_id: Product ID
            top_n: Number of top features to return
            
        Returns:
            List of (feature, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if product_id not in self.product_id_to_idx:
            return []
        
        # Get product index
        product_idx = self.product_id_to_idx[product_id]
        
        # Get TF-IDF scores for this product
        feature_scores = self.tfidf_matrix[product_idx].toarray()[0]
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Create (feature, score) pairs
        feature_importance = list(zip(feature_names, feature_scores))
        
        # Sort by score
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:top_n]


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING CONTENT-BASED RECOMMENDER")
    print("="*70)
    
    try:
        # Load processed data
        df = pd.read_csv("data/processed/products_processed.csv")
        print(f"\n✅ Loaded {len(df)} products")
        
        # Train model
        model = ContentBasedRecommender(max_features=3000)
        model.fit(df)
        
        # Get a sample product
        sample_product = df.iloc[0]
        sample_id = sample_product['Product Id']
        
        print(f"\n" + "-"*70)
        print(f"BASE PRODUCT:")
        print("-"*70)
        print(f"Name: {sample_product['Product Name']}")
        print(f"Brand: {sample_product['Product Brand']}")
        print(f"Category: {sample_product['Product Category']}")
        
        # Get recommendations
        print(f"\n" + "-"*70)
        print("TOP 10 SIMILAR PRODUCTS")
        print("-"*70)
        recommendations = model.recommend(sample_id, n_recommendations=10)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['name'][:45]:45s} | Similarity: {rec['similarity_score']:.3f}")
            print(f"    Brand: {rec['brand']:20s} | Category: {rec['category'][:40]:40s}")
        
        # Get important features
        print(f"\n" + "-"*70)
        print("TOP FEATURES FOR BASE PRODUCT")
        print("-"*70)
        features = model.get_feature_importance(sample_id, top_n=10)
        for feature, score in features:
            print(f"  {feature:30s}: {score:.4f}")
        
        # Save model
        model.save_model()
        print("\n✅ Model saved to models/content_based/")
        
    except FileNotFoundError:
        print("\n❌ Error: products_processed.csv not found")
        print("   Run test_data_pipeline.py first")
