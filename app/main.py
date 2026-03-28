
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import random

# Fix import paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.models.popularity_recommender import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender

# Page configuration
st.set_page_config(
    page_title="ShopSmart - Your Online Shopping Destination",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ''

# Global CSS
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
<style>

    /* ── GLOBAL BACKGROUND ── */
    .main {
        background: linear-gradient(180deg,
            #DDAED3 20%,
            #E8C5DF 10%,
            #EAEFEF 30%,
            #EAEFEF 70%,
            #F5F7F7 100%
        ) !important;
        padding: 0 !important;
        min-height: 100vh;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    section[data-testid="stAppViewContainer"] {
        background: transparent !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── NAVBAR COLUMN ROW ── */
    div[data-testid="stHorizontalBlock"]:first-of-type {
        background: #131921 !important;
        gap: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 56px !important;
        align-items: center !important;
    }
    div[data-testid="stHorizontalBlock"]:first-of-type > div {
        background: #131921 !important;
        padding: 0 !important;
        margin: 0 !important;
        gap: 0 !important;
        display: flex !important;
        align-items: center !important;
        min-height: 56px !important;
    }

    /* ── RESTORE GAP FOR ALL NON-NAVBAR HORIZONTAL BLOCKS ── */
    div[data-testid="stHorizontalBlock"]:not(:first-of-type) > div {
        background: transparent !important;
        padding: revert !important;
        margin: revert !important;
        min-height: revert !important;
    }

    /* ── REMOVE SPACE ABOVE NAVBAR ── */
    .block-container > div:first-child {
        padding: 0 !important;
        margin: 0 !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }

    /* ── NAVBAR ITEM STYLES ── */
    .nav-logo {
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        white-space: nowrap;
        cursor: pointer;
        padding: 4px 10px;
        border: 1px solid transparent;
        border-radius: 3px;
    }
    .nav-logo:hover { border-color: white; }

    .nav-location {
        display: flex;
        align-items: center;
        gap: 4px;
        color: white;
        white-space: nowrap;
        cursor: pointer;
        padding: 4px 6px;
        border: 1px solid transparent;
        border-radius: 3px;
    }
    .nav-location:hover { border-color: white; }
    .nav-location .label { font-size: 0.65rem; color: #ccc; display: block; }
    .nav-location .value { font-size: 0.82rem; font-weight: 600; display: block; }

    .nav-actions-wrap {
        display: flex;
        align-items: center;
        gap: 2px;
        height: 56px;
        background: #131921;
    }
    .nav-item {
        color: white;
        cursor: pointer;
        padding: 4px 6px;
        border: 1px solid transparent;
        border-radius: 3px;
        white-space: nowrap;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .nav-item:hover { border-color: white; }
    .nav-item .label { font-size: 0.65rem; color: #ccc; display: block; }
    .nav-item .value { font-size: 0.82rem; font-weight: 600; display: block; }

    .nav-cart {
        display: flex;
        align-items: center;
        gap: 3px;
        color: white;
        cursor: pointer;
        padding: 4px 6px;
        border: 1px solid transparent;
        border-radius: 3px;
        position: relative;
    }
    .nav-cart:hover { border-color: white; }
    .nav-cart .cart-count {
        position: absolute;
        top: 2px;
        left: 20px;
        background: #FF9900;
        color: #131921;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 1px 4px;
        border-radius: 3px;
    }
    .nav-cart .cart-label {
        font-size: 0.82rem;
        font-weight: 600;
        padding-top: 12px;
    }

    /* ── FORM INSIDE NAVBAR ── */
    div[data-testid="stForm"] {
        background: #131921 !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
    }
    div[data-testid="stForm"] > div {
        background: #131921 !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
    }
    section[data-testid="stForm"] {
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        background: #131921 !important;
    }

    /* ── INNER COLUMNS INSIDE FORM ── */
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
        background: #131921 !important;
        gap: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: unset !important;
        align-items: center !important;
    }
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div {
        background: #131921 !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: unset !important;
        display: flex !important;
        align-items: center !important;
    }

    /* ── SEARCH TEXT INPUT ── */
    div[data-testid="stTextInput"] {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }
    div[data-testid="stTextInput"] > div {
        background: #131921 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTextInput"] input {
        height: 36px !important;
        border-radius: 4px 0 0 4px !important;
        border: none !important;
        padding: 0 10px !important;
        font-size: 0.9rem !important;
        background: white !important;
        color: #131921 !important;
        margin: 0 !important;
    }
    div[data-testid="stTextInput"] input:focus {
        outline: 2px solid #FF9900 !important;
        box-shadow: none !important;
    }

    /* ── SUBMIT BUTTON ── */
    div[data-testid="stFormSubmitButton"] {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stFormSubmitButton"] > button {
        height: 36px !important;
        background: #FEBD69 !important;
        border: none !important;
        border-radius: 0 4px 4px 0 !important;
        padding: 0 14px !important;
        color: #131921 !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #F3A847 !important;
    }

    /* ── HERO ── */
    .hero-section {
        background: #DDAED3;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #131921;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .hero-subtitle {
        font-size: 1.3rem;
        color: #232f3e;
    }

    /* ── CATEGORY CARDS ── */
    .category-cards-container {
        padding: 1.5rem 2rem;
        max-width: 1500px;
        margin: 0 auto;
    }
    .category-card {
        background: white;
        padding: 1.5rem;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .category-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .category-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F1111;
        margin-bottom: 1rem;
    }
    .category-images-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .category-image-box {
        width: 100%;
        height: 110px;
        background: #f8f8f8;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border-radius: 4px;
    }
    .category-image-box img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    .category-link {
        color: #007185;
        font-size: 0.9rem;
        cursor: pointer;
    }
    .category-link:hover {
        color: #C7511F;
        text-decoration: underline;
    }

    /* ── PRODUCT CARDS ── */
    .products-section {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1500px;
        margin: 0 auto;
    }
    .product-card {
        background: white;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        cursor: pointer;
        transition: box-shadow 0.2s;
        margin-bottom: 1rem;
    }
    .product-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .product-image {
        width: 100%;
        height: 220px;
        background: #f8f8f8;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        overflow: hidden;
        border-radius: 4px;
    }
    .product-image img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    .product-title {
        color: #0F1111;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        min-height: 40px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .product-rating { color: #FF9900; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .product-price { color: #0F1111; font-size: 1.4rem; font-weight: 700; }
    .product-price-symbol { font-size: 0.8rem; vertical-align: super; }

    /* ── SECTION HEADER ── */
    .section-header {
        background: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F1111;
        padding: 1.5rem 2rem 1rem 2rem;
        max-width: 1500px;
        margin: 0 auto;
    }

    /* ── PRODUCT DETAIL ── */
    .product-detail-container {
        background: white;
        padding: 2rem;
        max-width: 1500px;
        margin: 2rem auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-radius: 8px;
    }
    .product-detail-image {
        width: 100%;
        height: 400px;
        background: #f8f8f8;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #ddd;
        border-radius: 8px;
    }
    .product-detail-image img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    .product-detail-rating { color: #FF9900; margin-bottom: 1rem; }
    .product-detail-price { font-size: 2rem; color: #B12704; font-weight: 700; margin: 1rem 0; }
    .product-detail-description { color: #0F1111; line-height: 1.6; margin-top: 1.5rem; }
    .product-buy-box {
        background: #f7fafa;
        border: 1px solid #D5D9D9;
        padding: 1.5rem;
        border-radius: 8px
        
    }
    .buy-box-price { font-size: 1.8rem; color: #B12704; font-weight: 700; margin-bottom: 1rem; }
    .buy-button {
        background: #FFD814;
        color: #0F1111;
        padding: 0.8rem;
        text-align: center;
        font-weight: 600;
        cursor: pointer;
        margin-bottom: 0.5rem;
        border: 1px solid #FCD200;
        border-radius: 8px;
    }
    .buy-button:hover { background: #0000; }
    .cart-button {
        background: #FFA41C;
        color: #0F1111;
        padding: 0.8rem;
        text-align: center;
        font-weight: 600;
        cursor: pointer;
        border: 1px solid #FF8F00;
        border-radius: 8px;
    }
    .cart-button:hover { background: #0000; }

    /* ── NO IMAGE ── */
    .no-image-icon { font-size: 4rem; color: #DDAED3; }

    /* ── FOOTER ── */
    .footer {
        background: linear-gradient(180deg, #232F3E 0%, #131921 100%);
        color: white;
        padding: 3rem 2rem;
        margin-top: 4rem;
    }
    .footer-content { max-width: 1500px; margin: 0 auto; }
    .footer-links {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 2rem;
        margin-bottom: 2rem;
    }
    .footer-column h3 { font-size: 1rem; margin-bottom: 1rem; color: white; }
    .footer-link {
        color: #DDD;
        font-size: 0.9rem;
        display: block;
        margin-bottom: 0.5rem;
        cursor: pointer;
    }
    .footer-link:hover { text-decoration: underline; }
    .footer-bottom {
        text-align: center;
        padding-top: 2rem;
        border-top: 1px solid #3a4553;
        color: #DDD;
    }

</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/products_processed.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Data not found.")
        return None


@st.cache_resource
def load_models():
    try:
        models = {
            'popularity': PopularityRecommender.load_model("models/popularity/popularity_model.pkl"),
            'content': ContentBasedRecommender.load_model("models/content_based/content_based_model.pkl"),
            'hybrid': HybridRecommender.load_model("models/hybrid/hybrid_model.pkl")
        }
        return models
    except:
        return None


def get_image_url(product_data):
    if pd.notna(product_data.get('Product Image Url')):
        urls = str(product_data['Product Image Url']).split('|')
        return urls[0].strip() if urls else None
    return None


def render_modern_navbar():

    params = st.query_params
    if "q" in params and params["q"] != st.session_state.get("search_query", ""):
        st.session_state.search_query = params["q"]
        st.session_state.current_page = 'search'
        st.rerun()

    col_logo, col_loc, col_search, col_actions = st.columns([1, 1.2, 5, 2.8])

    with col_logo:
        st.markdown("""
        <div style="background:#131921; height:56px; display:flex;
                    align-items:center; padding:0 8px;">
            <span class="nav-logo">🛒 ShopSmart</span>
        </div>
        """, unsafe_allow_html=True)

    with col_loc:
        st.markdown("""
        <div style="background:#131921; height:56px; display:flex; align-items:center;">
            <div class="nav-location">
                <i class="bi bi-geo-alt-fill" style="font-size:1.1rem;"></i>
                <div>
                    <span class="label">Deliver to</span>
                    <span class="value">Pakistan</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_search:
        with st.form(key="search_form", clear_on_submit=False):
            s_col, b_col = st.columns([11, 1])
            with s_col:
                query = st.text_input(
                    "Search",
                    placeholder="Search ShopSmart...",
                    label_visibility="collapsed",
                    key="search_input"
                )
            with b_col:
                submitted = st.form_submit_button("🔍")

        if submitted and query:
            st.session_state.search_query = query
            st.session_state.current_page = 'search'
            st.rerun()
        elif submitted and not query:
            st.session_state.current_page = 'home'
            st.rerun()

    with col_actions:
        st.markdown("""
        <div class="nav-actions-wrap">
            <div class="nav-item">
                <i class="bi bi-person" style="font-size:1.3rem;"></i>
                <div>
                    <span class="label">Hello, sign in</span>
                    <span class="value">Account &amp; Lists</span>
                </div>
            </div>
            <div class="nav-item">
                <i class="bi bi-box-seam" style="font-size:1.3rem;"></i>
                <div>
                    <span class="label">Returns</span>
                    <span class="value">&amp; Orders</span>
                </div>
            </div>
            <div class="nav-cart">
                <i class="bi bi-cart3" style="font-size:1.7rem;"></i>
                <span class="cart-count">0</span>
                <span class="cart-label">Cart</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_hero():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Welcome to ShopSmart</div>
        <div class="hero-subtitle">Discover amazing products with AI-powered recommendations</div>
    </div>
    """, unsafe_allow_html=True)


def create_category_card(title, category_name, products, df):
    category_products = df[df['Product Category'].str.contains(category_name, case=False, na=False)]
    sample_products = category_products.sample(min(4, len(category_products))) if len(category_products) > 0 else pd.DataFrame()

    images_html = ""
    for _, product in sample_products.iterrows():
        img_url = get_image_url(product)
        if img_url:
            images_html += f'<div class="category-image-box"><img src="{img_url}" onerror="this.style.display=\'none\'; this.parentElement.innerHTML=\'<div class=\\\'no-image-icon\\\'>🛍️</div>\'"></div>'
        else:
            images_html += '<div class="category-image-box"><div class="no-image-icon">🛍️</div></div>'

    remaining = 4 - len(sample_products)
    for _ in range(remaining):
        images_html += '<div class="category-image-box"><div class="no-image-icon">🛍️</div></div>'

    card_html = f"""
    <div class="category-card">
        <div class="category-card-title">{title}</div>
        <div class="category-images-grid">{images_html}</div>
        <div class="category-link">See more</div>
    </div>
    """

    if st.button(f"Browse {category_name}", key=f"cat_{category_name}", use_container_width=True):
        st.session_state.current_page = 'category'
        st.session_state.selected_category = category_name
        st.rerun()

    st.markdown(card_html, unsafe_allow_html=True)


def render_home_page(df, models):
    render_hero()

    popular = models['popularity'].recommend(n_recommendations=100)

    categories = [
        {"title": "Top Kitchen Appliances",  "category": "Kitchen"},
        {"title": "Shop Fashion for Less",   "category": "Fashion"},
        {"title": "Fashion Trends",          "category": "Clothing"},
        {"title": "Home Refresh Ideas",      "category": "Home"},
        {"title": "Beauty Picks",            "category": "Beauty"},
        {"title": "Electronics",             "category": "Electronics"},
        {"title": "Toys under $25",          "category": "Toys"},
        {"title": "Explore Laptops",         "category": "Computers"}
    ]

    st.markdown('<div class="category-cards-container">', unsafe_allow_html=True)
    cols = st.columns(4, gap="large")
    for i, cat_info in enumerate(categories):
        with cols[i % 4]:
            create_category_card(cat_info["title"], cat_info["category"], popular, df)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔥 Best Sellers</div>', unsafe_allow_html=True)
    bestsellers = models['popularity'].recommend(n_recommendations=8)
    render_product_grid(bestsellers, df)


def render_product_grid(products, df):
    st.markdown('<div class="products-section">', unsafe_allow_html=True)
    cols = st.columns(4, gap="large")
    for i, product in enumerate(products):
        with cols[i % 4]:
            render_product_card(product, df)
    st.markdown('</div>', unsafe_allow_html=True)


def render_product_card(product, df):
    product_id = product['product_id']
    product_row = df[df['Product Id'] == product_id]

    if len(product_row) == 0:
        return

    product_data = product_row.iloc[0]
    img_url = get_image_url(product_data)

    if img_url:
        img_html = f'<img src="{img_url}" onerror="this.style.display=\'none\'; this.parentElement.innerHTML=\'<div class=\\\'no-image-icon\\\'>🛍️</div>\'">'
    else:
        img_html = '<div class="no-image-icon">🛍️</div>'

    rating = product.get('rating', 0)
    rating_html = f'<div class="product-rating">{"⭐" * int(rating)} {rating:.1f}</div>' if rating > 0 else ''

    card_html = f"""
    <div class="product-card">
        <div class="product-image">{img_html}</div>
        <div class="product-title">{product.get('name', 'Unknown')[:60]}</div>
        {rating_html}
        <div class="product-price">
            <span class="product-price-symbol">$</span>{product.get('price', 0):.2f}
        </div>
    </div>
    """

    if st.button("View", key=f"prod_{product_id}_{random.randint(1000,9999)}", use_container_width=True):
        st.session_state.current_page = 'product_detail'
        st.session_state.selected_product = product_id
        st.rerun()

    st.markdown(card_html, unsafe_allow_html=True)


def render_search_page(df, models):
    query = st.session_state.search_query

    if not query:
        render_home_page(df, models)
        return

    mask = df['Product Name'].str.contains(query, case=False, na=False)
    results = df[mask].head(20)

    st.markdown(f'<div class="section-header">Search results for "{query}"</div>', unsafe_allow_html=True)

    if len(results) > 0:
        products_list = []
        for _, row in results.iterrows():
            products_list.append({
                'product_id': row['Product Id'],
                'name': row['Product Name'],
                'brand': row['Product Brand'],
                'price': row['Product Price'],
                'rating': row.get('Product Rating')
            })

        render_product_grid(products_list[:8], df)

        st.markdown('<div class="section-header">✨ Customers who searched for this also bought</div>',
                    unsafe_allow_html=True)
        similar = models['content'].recommend(
            product_id=products_list[0]['product_id'],
            n_recommendations=8
        )
        render_product_grid(similar, df)
    else:
        st.info(f"No results found for '{query}'")


def render_product_detail(df, models):
    product_id = st.session_state.selected_product
    product_row = df[df['Product Id'] == product_id]

    if len(product_row) == 0:
        st.error("Product not found")
        return

    product = product_row.iloc[0]

    if st.button("← Back", key="back_btn"):
        st.session_state.current_page = 'home'
        st.session_state.selected_product = None
        st.rerun()

    st.markdown('<div class="product-detail-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 3, 2], gap="large")

    with col1:
        img_url = get_image_url(product)
        img_html = f'<img src="{img_url}" onerror="this.style.display=\'none\'; this.parentElement.innerHTML=\'<div class=\\\'no-image-icon\\\'>🛍️</div>\'">' if img_url else '<div class="no-image-icon">🛍️</div>'
        st.markdown(f'<div class="product-detail-image">{img_html}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<h1>{product["Product Name"]}</h1>', unsafe_allow_html=True)
        st.markdown(f'**Brand:** {product["Product Brand"]}')
        rating = product.get('Product Rating', 0)
        if rating > 0:
            stars = '⭐' * int(rating)
            st.markdown(f'<div class="product-detail-rating">{stars} {rating:.1f} out of 5</div>',
                        unsafe_allow_html=True)
        st.markdown(f'<div class="product-detail-price">${product["Product Price"]:.2f}</div>',
                    unsafe_allow_html=True)
        st.markdown('**About this item:**')
        description = product.get('Product Description', 'No description available.')
        if len(str(description)) > 500:
            description = str(description)[:500] + '...'
        st.write(description)
        st.markdown(f"**Category:** {product['Product Category']}")

    with col3:
        st.markdown(f"""
        <div class="product-buy-box">
            <div class="buy-box-price">${product["Product Price"]:.2f}</div>
            <div style="margin-bottom:1rem; color:#007600; font-weight:600;">✓ In Stock</div>
            <div style="margin-bottom:0.5rem; color:#565959;">FREE delivery</div>
            <div class="buy-button">Buy Now</div>
            <div class="cart-button">Add to Cart</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Customers who viewed this also viewed</div>',
                unsafe_allow_html=True)
    similar = models['content'].recommend(product_id=product_id, n_recommendations=8)
    render_product_grid(similar, df)


def render_footer():
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <div class="footer-links">
                <div class="footer-column">
                    <h3>Get to Know Us</h3>
                    <span class="footer-link">About ShopSmart</span>
                    <span class="footer-link">Careers</span>
                    <span class="footer-link">Press Releases</span>
                </div>
                <div class="footer-column">
                    <h3>Make Money with Us</h3>
                    <span class="footer-link">Sell on ShopSmart</span>
                    <span class="footer-link">Become an Affiliate</span>
                    <span class="footer-link">Advertise</span>
                </div>
                <div class="footer-column">
                    <h3>Let Us Help You</h3>
                    <span class="footer-link">Your Account</span>
                    <span class="footer-link">Returns Centre</span>
                    <span class="footer-link">Help</span>
                </div>
                <div class="footer-column">
                    <h3>Customer Service</h3>
                    <span class="footer-link">Contact Us</span>
                    <span class="footer-link">Privacy Policy</span>
                    <span class="footer-link">Terms of Service</span>
                </div>
            </div>
            <div class="footer-bottom">
                <div style="margin-bottom:1rem;">
                    <strong>🛒 ShopSmart</strong> - AI-Powered Shopping Experience
                </div>
                <div>© 2024 ShopSmart, Inc. All rights reserved.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    df = load_data()
    models = load_models()

    if df is None or models is None:
        st.error("Please run data pipeline and train models first.")
        return

    render_modern_navbar()

    if st.session_state.current_page == 'home':
        render_home_page(df, models)
    elif st.session_state.current_page == 'search':
        render_search_page(df, models)
    elif st.session_state.current_page == 'product_detail':
        render_product_detail(df, models)

    render_footer()


if __name__ == "__main__":
    main()