
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.base_recommender import BaseRecommender
from src.models.popularity_recommender import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def print_recommendations(recommendations, show_scores=False):
    """Print formatted recommendations."""
    for i, rec in enumerate(recommendations, 1):
        name = rec['name'][:50]
        brand = rec.get('brand', 'Unknown')[:20]
        price = rec.get('price', 0)
        
        print(f"{i:2d}. {name:50s}")
        print(f"    Brand: {brand:20s} | Price: ${price:6.2f}")
        
        if show_scores:
            if 'popularity_score' in rec:
                print(f"    Popularity Score: {rec['popularity_score']:.3f}")
            if 'similarity_score' in rec:
                print(f"    Similarity Score: {rec['similarity_score']:.3f}")
            if 'hybrid_score' in rec:
                print(f"    Hybrid Score: {rec['hybrid_score']:.3f}")


def main():
    print_section("RECOMMENDATION MODELS - COMPLETE TEST")
    
    # Load data
    print("\n[STEP 1] Loading processed data...")
    print("-"*70)
    
    try:
        df = pd.read_csv("data/processed/products_processed.csv")
        print(f"✅ Loaded {len(df)} products")
    except FileNotFoundError:
        print("❌ ERROR: products_processed.csv not found")
        print("   Please run test_data_pipeline.py first to create processed data")
        return
    
    # Get a sample product for testing
    sample_product = df.iloc[0]
    sample_id = sample_product['Product Id']
    sample_name = sample_product['Product Name']
    
    print(f"\nSample Product for Testing:")
    print(f"  ID: {sample_id}")
    print(f"  Name: {sample_name}")
    print(f"  Brand: {sample_product['Product Brand']}")
    print(f"  Category: {sample_product['Product Category']}")
    
    # ========================================================================
    # MODEL 1: POPULARITY-BASED RECOMMENDER
    # ========================================================================
    print_section("[MODEL 1] POPULARITY-BASED RECOMMENDER")
    
    print("\nTraining popularity model...")
    pop_model = PopularityRecommender()
    pop_model.fit(df)
    print("✅ Training complete")
    
    # Test 1: Overall popular products
    print("\n" + "-"*70)
    print("Top 10 Popular Products (Overall)")
    print("-"*70)
    pop_recs = pop_model.recommend(n_recommendations=10)
    print_recommendations(pop_recs, show_scores=True)
    
    # Test 2: Popular in category
    print("\n" + "-"*70)
    print("Top 5 Popular Products in 'Beauty' Category")
    print("-"*70)
    beauty_recs = pop_model.get_top_products_by_category("Beauty", n_recommendations=5)
    print_recommendations(beauty_recs)
    
    # Test 3: Trending products
    print("\n" + "-"*70)
    print("Trending Products (High ratings + Reviews)")
    print("-"*70)
    trending_recs = pop_model.get_trending_products(n_recommendations=5, min_reviews=10)
    print_recommendations(trending_recs)
    
    # Save model
    pop_model.save_model()
    print("\n✅ Popularity model saved to models/popularity/")
    
    # ========================================================================
    # MODEL 2: CONTENT-BASED RECOMMENDER
    # ========================================================================
    print_section("[MODEL 2] CONTENT-BASED RECOMMENDER")
    
    print("\nTraining content-based model...")
    content_model = ContentBasedRecommender(max_features=3000)
    content_model.fit(df)
    print("✅ Training complete")
    
    # Test 1: Similar products
    print("\n" + "-"*70)
    print(f"Top 10 Similar Products to: {sample_name[:50]}")
    print("-"*70)
    similar_recs = content_model.recommend(sample_id, n_recommendations=10)
    print_recommendations(similar_recs, show_scores=True)
    
    # Test 2: Similar products in same category
    print("\n" + "-"*70)
    print("Similar Products (Same Category Only)")
    print("-"*70)
    category_similar = content_model.get_similar_by_category(sample_id, n_recommendations=5)
    print_recommendations(category_similar)
    
    # Test 3: Important features
    print("\n" + "-"*70)
    print(f"Top Features for: {sample_name[:50]}")
    print("-"*70)
    features = content_model.get_feature_importance(sample_id, top_n=10)
    for feature, score in features:
        print(f"  {feature:30s}: {score:.4f}")
    
    # Save model
    content_model.save_model()
    print("\n✅ Content-based model saved to models/content_based/")
    
    # ========================================================================
    # MODEL 3: HYBRID RECOMMENDER
    # ========================================================================
    print_section("[MODEL 3] HYBRID RECOMMENDER")
    
    print("\nTraining hybrid model...")
    print("Weights: 40% Popularity + 60% Content-Based")
    hybrid_model = HybridRecommender(
        popularity_weight=0.4,
        content_weight=0.6
    )
    hybrid_model.fit(df)
    print("✅ Training complete")
    
    # Test 1: Cold start (new user)
    print("\n" + "-"*70)
    print("Cold Start Recommendations (New User)")
    print("-"*70)
    cold_start_recs = hybrid_model.recommend_cold_start(n_recommendations=5)
    print_recommendations(cold_start_recs)
    
    # Test 2: Hybrid recommendations
    print("\n" + "-"*70)
    print(f"Hybrid Recommendations for: {sample_name[:50]}")
    print("-"*70)
    hybrid_recs = hybrid_model.recommend(
        product_id=sample_id,
        n_recommendations=10,
        strategy='hybrid'
    )
    
    for i, rec in enumerate(hybrid_recs, 1):
        name = rec['name'][:45]
        print(f"{i:2d}. {name:45s} | Hybrid: {rec['hybrid_score']:.3f}")
        print(f"    Pop: {rec['popularity_contribution']:.3f} | Content: {rec['content_contribution']:.3f}")
    
    # Test 3: Shopping cart recommendations
    cart_items = [df.iloc[i]['Product Id'] for i in range(3)]
    print("\n" + "-"*70)
    print(f"Shopping Cart Recommendations ({len(cart_items)} items in cart)")
    print("-"*70)
    cart_recs = hybrid_model.recommend_similar_to_cart(
        cart_product_ids=cart_items,
        n_recommendations=5
    )
    print_recommendations(cart_recs)
    
    # Save model
    hybrid_model.save_model()
    print("\n✅ Hybrid model saved to models/hybrid/")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("✅ ALL MODELS TRAINED AND TESTED SUCCESSFULLY!")
    
    print("\nModels saved:")
    print("  1. models/popularity/popularity_model.pkl")
    print("  2. models/content_based/content_based_model.pkl")
    print("  3. models/hybrid/hybrid_model.pkl")
    
    print("\nModel Capabilities:")
    print("  📊 Popularity Model:")
    print("     - Overall popular products")
    print("     - Popular by category")
    print("     - Trending products")
    
    print("\n  🔍 Content-Based Model:")
    print("     - Find similar products")
    print("     - Similar in same category")
    print("     - Multi-product recommendations")
    
    print("\n  🎯 Hybrid Model:")
    print("     - Best of both worlds")
    print("     - Cold start recommendations")
    print("     - Shopping cart recommendations")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. Build web application (Streamlit)")
    print("  2. Create evaluation metrics")
    print("  3. Deploy the system")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
