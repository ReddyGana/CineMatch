import streamlit as st
import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(
    page_title="🎬 CineMatch",
    page_icon="📽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding: 2rem 1rem;
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Genre card styling */
    .genre-card {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        text-align: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
    }
    
    .genre-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.25);
    }
    
    /* Movie card styling */
    .movie-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Search section styling */
    .search-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Stats section */
    .stats-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Loading animation */
    .loading-text {
        text-align: center;
        font-size: 1.2rem;
        color: #667eea;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Recommendation section */
    .recommendation-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Simple movie card for better rendering */
    .simple-movie-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .movie-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .movie-meta {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .movie-description {
        color: #555;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .movie-details {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Helper functions
# -----------------------
def clean_data(x):
    """Extract names from stringified list format"""
    if isinstance(x, str) and x.strip():
        try:
            return " ".join([i['name'] for i in ast.literal_eval(x)])
        except:
            return x
    else:
        return ""

def get_top_cast(x, limit=3):
    """Get top cast members"""
    if isinstance(x, str) and x.strip():
        try:
            cast_list = ast.literal_eval(x)
            return " ".join([i['name'] for i in cast_list[:limit]])
        except:
            names = str(x).split(',')[:limit]
            return " ".join([name.strip() for name in names])
    return ""

def get_director(x):
    """Extract director from crew data"""
    if isinstance(x, str) and x.strip():
        try:
            crew_list = ast.literal_eval(x)
            director = next((i['name'] for i in crew_list if i.get('job') == 'Director'), "")
            return director
        except:
            return ""
    return ""

def normalize_genre(genre_str):
    """Normalize genre names for consistent grouping"""
    if pd.isna(genre_str) or not str(genre_str).strip():
        return "Other"
    
    genre_str = str(genre_str).lower().strip()
    
    genre_map = {
        'thriller': 'Thriller & Mystery',
        'mystery': 'Thriller & Mystery',
        'crime': 'Thriller & Mystery',
        'suspense': 'Thriller & Mystery',
        'romance': 'Romance & Drama',
        'drama': 'Romance & Drama',
        'romantic': 'Romance & Drama',
        'love': 'Romance & Drama',
        'comedy': 'Comedy',
        'humor': 'Comedy',
        'funny': 'Comedy',
        'comic': 'Comedy',
        'action': 'Action & Adventure',
        'adventure': 'Action & Adventure',
        'war': 'Action & Adventure',
        'fight': 'Action & Adventure',
        'horror': 'Horror',
        'scary': 'Horror',
        'supernatural': 'Horror',
        'ghost': 'Horror',
        'fantasy': 'Fantasy & Sci-Fi',
        'science fiction': 'Fantasy & Sci-Fi',
        'sci-fi': 'Fantasy & Sci-Fi',
        'scifi': 'Fantasy & Sci-Fi',
        'magic': 'Fantasy & Sci-Fi',
        'animation': 'Family & Animation',
        'family': 'Family & Animation',
        'kids': 'Family & Animation',
        'children': 'Family & Animation',
        'documentary': 'Documentary',
        'biography': 'Documentary',
        'history': 'Documentary',
        'historical': 'Documentary'
    }
    
    for key, value in genre_map.items():
        if key in genre_str:
            return value
    
    return "Other"

def get_language_flag(language):
    """Get emoji flag for language"""
    if not language or pd.isna(language):
        return '🌐'
        
    language_flags = {
        'english': '🇺🇸',
        'hindi': '🇮🇳',
        'tamil': '🇮🇳',
        'telugu': '🇮🇳',
        'malayalam': '🇮🇳',
        'kannada': '🇮🇳',
        'bengali': '🇮🇳',
        'marathi': '🇮🇳',
        'punjabi': '🇮🇳',
        'gujarati': '🇮🇳',
        'french': '🇫🇷',
        'spanish': '🇪🇸',
        'german': '🇩🇪',
        'italian': '🇮🇹',
        'japanese': '🇯🇵',
        'korean': '🇰🇷',
        'chinese': '🇨🇳'
    }
    return language_flags.get(str(language).lower(), '🌐')

def safe_get(dictionary, key, default="N/A"):
    """Safely get value from dictionary with default"""
    try:
        if isinstance(dictionary, dict):
            return dictionary.get(key, default)
        elif hasattr(dictionary, key):
            value = getattr(dictionary, key, default)
            return value if value is not None and str(value).strip() else default
        else:
            return default
    except:
        return default

def safe_numeric_convert(value, default=0):
    """Safely convert value to numeric"""
    try:
        if pd.isna(value) or value is None or str(value).strip() == '':
            return default
        return float(str(value))
    except:
        return default

@st.cache_data
def load_movie_metadata():
    """Load just the metadata (no similarity computation)"""
    all_movies = []
    
    try:
        movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
        credits_df = pd.read_csv("data/tmdb_5000_credits.csv")
        
        english_movies = movies_df.merge(credits_df, left_on='id', right_on='movie_id')
        
        if 'title_x' in english_movies.columns:
            english_movies.rename(columns={'title_x': 'title'}, inplace=True)
        if 'title_y' in english_movies.columns:
            english_movies.drop(columns=['title_y'], inplace=True)
        
        english_movies['keywords'] = english_movies['keywords'].apply(clean_data)
        english_movies['genres'] = english_movies['genres'].apply(clean_data)
        english_movies['cast'] = english_movies['cast'].apply(get_top_cast)
        english_movies['crew'] = english_movies['crew'].apply(get_director)
        english_movies['overview'] = english_movies['overview'].fillna('No description available.')
        english_movies['language'] = 'English'
        english_movies['year'] = pd.to_datetime(english_movies['release_date'], errors='coerce').dt.year
        english_movies['rating'] = english_movies['vote_average']
        
        for _, row in english_movies.iterrows():
            description = str(row['overview']) if pd.notna(row['overview']) and str(row['overview']).strip() else "No description available."
            if len(description) > 200:
                description = description[:200] + "..."
            
            cast = str(row['cast']) if pd.notna(row['cast']) and str(row['cast']).strip() else "Cast information not available"
            director = str(row['crew']) if pd.notna(row['crew']) and str(row['crew']).strip() else "Director information not available"
            
            movie_data = {
                'title': str(row['title']),
                'genre_raw': str(row['genres']),
                'language': 'English',
                'year': row['year'] if pd.notna(row['year']) else 'Unknown',
                'rating': row['rating'] if pd.notna(row['rating']) else 'N/A',
                'cast': cast,
                'director': director,
                'description': description,
                'plot_tags': f"{row['genres']} {row['keywords']} {row['overview']}",
                'cast_director_tags': f"{cast} {director}",
                'flag': get_language_flag('English')
            }
            all_movies.append(movie_data)
        
    except FileNotFoundError:
        st.warning("⚠️ English movie dataset not found. Skipping...")
    except Exception as e:
        st.error(f"Error loading English movies: {str(e)}")
    
    try:
        indian_df = pd.read_csv("data/indian_movies_filtered.csv")
        indian_df.columns = indian_df.columns.str.strip()
        
        title_col = None
        for col in ['Movie Name', 'title', 'Title', 'movie name']:
            if col in indian_df.columns:
                title_col = col
                break
        
        if title_col:
            indian_df.rename(columns={title_col: 'title'}, inplace=True)
        
        for _, row in indian_df.iterrows():
            genre_raw = row.get('Genre', row.get('genre', 'Other'))
            language = row.get('Language', row.get('language', 'Hindi'))
            
            description = str(row.get('Description', row.get('Plot', 'No description available.')))
            if description == 'nan' or not description.strip():
                description = "No description available."
            if len(description) > 200:
                description = description[:200] + "..."
            
            cast = str(row.get('Cast', row.get('cast', 'Cast information not available')))
            if cast == 'nan' or not cast.strip():
                cast = "Cast information not available"
            
            director = str(row.get('Director', row.get('director', 'Director information not available')))
            if director == 'nan' or not director.strip():
                director = "Director information not available"
            
            movie_data = {
                'title': str(row['title']),
                'genre_raw': str(genre_raw),
                'language': str(language),
                'year': row.get('Year', row.get('year', 'Unknown')),
                'rating': row.get('Rating', row.get('rating', 'N/A')),
                'cast': cast,
                'director': director,
                'description': description,
                'plot_tags': f"{genre_raw} {row['title']} {description}",
                'cast_director_tags': f"{cast} {director}",
                'flag': get_language_flag(language)
            }
            all_movies.append(movie_data)
        
    except FileNotFoundError:
        st.warning("⚠️ Indian movie dataset not found. Skipping...")
    except Exception as e:
        st.error(f"Error loading Indian movies: {str(e)}")
    
    if not all_movies:
        st.error("❌ No movie datasets found! Please add movie data files.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_movies)
    df['genre_category'] = df['genre_raw'].apply(normalize_genre)
    
    required_columns = {
        'flag': lambda x: get_language_flag(x.get('language', '')),
        'genre': 'genre_category'
    }
    
    for col, default_func in required_columns.items():
        if col not in df.columns:
            if callable(default_func):
                df[col] = df.apply(default_func, axis=1)
            else:
                df[col] = df[default_func] if default_func in df.columns else 'N/A'
    
    return df

def compute_similarity_for_genre_with_mode(df, selected_genre, rec_mode, max_movies=1000):
    """Compute similarity only for movies in selected genre based on recommendation mode"""
    genre_movies = df[df['genre_category'] == selected_genre].copy()
    
    if len(genre_movies) > max_movies:
        st.warning(f"⚠️ Too many movies in {selected_genre} ({len(genre_movies)}). Using top {max_movies} rated movies.")
        genre_movies['rating_numeric'] = pd.to_numeric(genre_movies['rating'], errors='coerce')
        genre_movies = genre_movies.nlargest(max_movies, 'rating_numeric', keep='all')
    
    if rec_mode == "Plot-focused":
        tags_col = 'plot_tags'
    elif rec_mode == "Cast-focused":
        tags_col = 'cast_director_tags'
    else: # "Balanced" mode
        genre_movies['combined_tags'] = genre_movies['plot_tags'].fillna('') + ' ' + genre_movies['cast_director_tags'].fillna('')
        tags_col = 'combined_tags'

    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(genre_movies[tags_col].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    return genre_movies, cosine_sim

def recommend_movies_in_genre(movie_title, genre_df, cosine_sim, num_recommendations=5):
    """Get movie recommendations within the same genre"""
    try:
        idx = genre_df[genre_df['title'].str.lower() == movie_title.lower()].index
        if len(idx) == 0:
            return []
        
        movie_pos = genre_df.index.get_loc(idx[0])
        
        sim_scores = list(enumerate(cosine_sim[movie_pos]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        movie_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
        
        recommendations = []
        for idx in movie_indices:
            movie = genre_df.iloc[idx]
            movie_dict = {
                'title': safe_get(movie, 'title'),
                'genre': safe_get(movie, 'genre_category'),
                'language': safe_get(movie, 'language'),
                'year': safe_get(movie, 'year'),
                'rating': safe_get(movie, 'rating'),
                'flag': safe_get(movie, 'flag', get_language_flag(safe_get(movie, 'language'))),
                'description': safe_get(movie, 'description', 'No description available.'),
                'cast': safe_get(movie, 'cast', 'Cast information not available'),
                'director': safe_get(movie, 'director', 'Director information not available')
            }
            recommendations.append(movie_dict)
        
        return recommendations
    
    except (IndexError, KeyError) as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

def display_movie_card_simple(movie, index, similarity_score=None):
    """Display a simple movie recommendation card without complex HTML"""
    
    title = safe_get(movie, 'title', 'Unknown Title')
    flag = safe_get(movie, 'flag', get_language_flag(safe_get(movie, 'language', '')))
    language = safe_get(movie, 'language', 'Unknown')
    year = safe_get(movie, 'year', 'Unknown')
    rating = safe_get(movie, 'rating', 'N/A')
    genre = safe_get(movie, 'genre', safe_get(movie, 'genre_category', 'Unknown'))
    description = safe_get(movie, 'description', 'No description available.')
    director = safe_get(movie, 'director', 'Director information not available')
    cast = safe_get(movie, 'cast', 'Cast information not available')
    
    rating_display = "N/A"
    if rating != 'N/A' and str(rating).replace('.','').replace('-','').isdigit():
        try:
            rating_display = f"{float(rating):.1f}"
        except:
            rating_display = str(rating)
    
    similarity_display = f" • 🎯 {similarity_score:.1%} Match" if similarity_score else ""
    
    with st.container():
        col1, col2 = st.columns([1, 10])
        
        with col1:
            st.markdown(f"<div style='text-align: center; padding: 0.5rem; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 1rem;'>{index}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"### {title}")
            st.markdown(f"{flag} **{language}** • 📅 **{year}** • ⭐ **{rating_display}**{similarity_display}")
            
            st.markdown(f"<span style='background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: bold;'>🎭 {genre}</span>", unsafe_allow_html=True)
            
            st.markdown(f"**📝 Description:** {description}")
            
            col_cast, col_dir = st.columns(2)
            with col_cast:
                st.markdown(f"**🎭 Cast:** {cast[:50]}{'...' if len(str(cast)) > 50 else ''}")
            with col_dir:
                st.markdown(f"**🎬 Director:** {director}")
        
        st.divider()

def create_genre_visualization(df):
    """Create interactive genre distribution chart"""
    try:
        genre_counts = df['genre_category'].value_counts()
        
        fig = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            title="🎭 Movie Distribution by Genre",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            font=dict(size=14),
            title_font=dict(size=18, color='#333'),
            showlegend=True
        )
        return fig
    except Exception as e:
        st.error(f"Error creating genre visualization: {str(e)}")
        return px.pie(title="🎭 Movie Distribution by Genre (No data)")

def create_language_chart(df):
    """Create language distribution chart"""
    try:
        lang_counts = df['language'].value_counts().head(10)
        
        fig = px.bar(
            x=lang_counts.values,
            y=lang_counts.index,
            orientation='h',
            title="🌍 Top 10 Languages in Database",
            color=lang_counts.values,
            color_continuous_scale="viridis"
        )
        fig.update_layout(
            font=dict(size=12),
            title_font=dict(size=16, color='#333'),
            xaxis_title="Number of Movies",
            yaxis_title="Language"
        )
        return fig
    except Exception as e:
        st.error(f"Error creating language chart: {str(e)}")
        return px.bar(title="🌍 Top 10 Languages in Database (No data)")

def create_rating_distribution(df):
    """Create rating distribution chart"""
    try:
        df_copy = df.copy()
        df_copy['rating_numeric'] = pd.to_numeric(df_copy['rating'], errors='coerce')
        df_clean = df_copy.dropna(subset=['rating_numeric'])
        
        if len(df_clean) == 0:
            fig = px.bar(title="⭐ Rating Distribution (No data available)")
            return fig
        
        fig = px.histogram(
            df_clean,
            x='rating_numeric',
            nbins=20,
            title="⭐ Rating Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            font=dict(size=12),
            title_font=dict(size=16, color='#333'),
            xaxis_title="Rating",
            yaxis_title="Number of Movies"
        )
        return fig
    except Exception as e:
        st.error(f"Error creating rating distribution: {str(e)}")
        return px.histogram(title="⭐ Rating Distribution (No data)")

# -----------------------
# Sidebar Functions
# -----------------------
def create_sidebar(df):
    """Create enhanced sidebar with filters and stats"""
    st.sidebar.markdown("## 🎬 CineMatch")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 📊 Quick Stats")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.sidebar.metric("🎬 Movies", f"{len(df):,}")
    
    with col2:
        st.sidebar.metric("🌍 Languages", f"{df['language'].nunique()}")
    
    st.sidebar.markdown("### 🔍 Quick Search")
    search_term = st.sidebar.text_input("Search movies...")
    
    if search_term:
        search_results = df[df['title'].str.contains(search_term, case=False, na=False)]
        if len(search_results) > 0:
            st.sidebar.markdown(f"**Found {len(search_results)} movies:**")
            for _, movie in search_results.head(5).iterrows():
                flag = safe_get(movie, 'flag', get_language_flag(safe_get(movie, 'language', '')))
                title = safe_get(movie, 'title', 'Unknown')
                language = safe_get(movie, 'language', 'Unknown')
                st.sidebar.markdown(f"• {flag} **{title}** ({language})")
            if len(search_results) > 5:
                st.sidebar.markdown(f"... and {len(search_results) - 5} more")
        else:
            st.sidebar.markdown("No movies found")
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 🎛️ Global Filters")
    
    languages = ['All Languages'] + sorted(df['language'].unique().tolist())
    selected_language = st.sidebar.selectbox("Filter by Language:", languages)
    
    df['year_numeric'] = pd.to_numeric(df['year'], errors='coerce')
    min_year = int(df['year_numeric'].min()) if not df['year_numeric'].isna().all() else 1900
    max_year = int(df['year_numeric'].max()) if not df['year_numeric'].isna().all() else 2024
    
    year_range = st.sidebar.slider(
        "Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
    if not df['rating_numeric'].isna().all():
        min_rating = st.sidebar.slider(
            "Minimum Rating:",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )
    else:
        min_rating = 0.0
    
    return selected_language, year_range, min_rating, search_term

def set_random_movie(movie_list):
    """Callback function to set a random movie in session state."""
    random_movie = np.random.choice(movie_list)
    st.session_state.movie_selector = random_movie

# -----------------------
# Main Streamlit App
# -----------------------
def main():
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 3rem;">🎬 CineMatch</h1>
        <h3 style="margin: 0.5rem 0 0 0; font-weight: 300;">AI-Powered Movie Recommendation Engine</h3>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Discover your next favorite movie from a curated collection of global cinema</p>
    </div>
    """, unsafe_allow_html=True)

    # --- NEW: Exciting Loading Screen ---
    fun_facts = [
        "Did you know? The longest movie ever made is 85 hours long!",
        "Lights, camera, data! Our AI is preparing your movie matches...",
        "The first film with a rating system was 'The Pawnbroker' (1966).",
        "Popcorn was first served in movie theaters in 1912.",
        "Almost ready! Our algorithms are analyzing plots, casts, and genres.",
        "Fun fact: The 'Wilhelm scream' has been used in over 200 films!",
        "Just a moment... Our movie database is being curated for you."
    ]

    loading_placeholder = st.empty()
    for fact in fun_facts:
        loading_placeholder.info(f"💡 {fact}")
        time.sleep(0.3)
    loading_placeholder.empty()

    df = load_movie_metadata()
    # --- END NEW ---

    if df.empty:
        st.error("❌ No movies loaded. Please check your data files.")
        return
    
    selected_language, year_range, min_rating, search_term = create_sidebar(df)
    
    filtered_df = df.copy()
    
    if selected_language != 'All Languages':
        filtered_df = filtered_df[filtered_df['language'] == selected_language].copy()
    
    filtered_df['year_numeric'] = pd.to_numeric(filtered_df['year'], errors='coerce')
    filtered_df = filtered_df[
        (filtered_df['year_numeric'] >= year_range[0]) & 
        (filtered_df['year_numeric'] <= year_range[1])
    ]
    
    filtered_df['rating_numeric'] = pd.to_numeric(filtered_df['rating'], errors='coerce')
    filtered_df = filtered_df[
        (filtered_df['rating_numeric'] >= min_rating) | 
        (filtered_df['rating_numeric'].isna())
    ].copy()
    
    genre_counts = filtered_df['genre_category'].value_counts()
    
    st.markdown("## 🎭 Explore by Genre")
    st.markdown(f"*Currently showing {len(filtered_df):,} movies based on your filters*")
    
    if 'selected_genre' not in st.session_state:
        st.session_state.selected_genre = None

    genres_with_emoji = {
        'Thriller & Mystery': '🔍',
        'Romance & Drama': '❤️',
        'Comedy': '😂',
        'Action & Adventure': '💥',
        'Horror': '👻',
        'Fantasy & Sci-Fi': '🚀',
        'Family & Animation': '👨‍👩‍👧‍👦',
        'Documentary': '📺',
        'Other': '🎬'
    }
    
    cols = st.columns(3)
    genre_list = list(genre_counts.items())
    
    selected_genre_click = None
    
    for i, (genre, count) in enumerate(genre_list):
        col_idx = i % 3
        emoji = genres_with_emoji.get(genre, '🎬')
        
        with cols[col_idx]:
            if st.button(
                f"{emoji} {genre}\n({count:,} movies)", 
                key=f"genre_{i}", 
                use_container_width=True,
                help=f"Explore {count} movies in {genre} category"
            ):
                selected_genre_click = genre
    
    if selected_genre_click:
        st.session_state.selected_genre = selected_genre_click
    
    if st.session_state.selected_genre:
        current_genre = st.session_state.selected_genre
        
        col1, col2 = st.columns([3, 1])
        with col1:
            emoji = genres_with_emoji.get(current_genre, '🎬')
            st.markdown(f"""
            <div class="search-section">
                <h2 style="margin: 0; color: #333;">{emoji} {current_genre} Collection</h2>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Discover amazing {current_genre.lower()} movies from around the world</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("← Back to Genres", type="secondary"):
                st.session_state.selected_genre = None
                st.rerun()

        current_filtered_df = filtered_df[filtered_df['genre_category'] == current_genre].copy()
        
        if current_filtered_df.empty:
            st.warning(f"No {current_genre} movies found with current filters.")
            return
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 AI Recommendations", 
            "📋 Browse Collection", 
            "📊 Analytics Dashboard",
            "🔥 Trending & Popular"
        ])
        
        with tab1:
            st.markdown("""
            <div class="recommendation-section">
                <h3 style="margin: 0; color: #333;">🤖 AI-Powered Movie Recommendations</h3>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Our advanced algorithm analyzes plot, cast, and director to find your perfect match</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Use two columns for selectbox and button
            col1, col2 = st.columns([2, 1])
            
            movie_options = [""] + sorted(current_filtered_df['title'].unique())
            
            if 'movie_selector' not in st.session_state:
                st.session_state.movie_selector = ""

            if st.session_state.movie_selector not in movie_options:
                st.session_state.movie_selector = ""

            with col1:
                selected_movie_title = st.selectbox(
                    f"🎬 Select a {current_genre} movie you enjoyed:",
                    options=movie_options,
                    index=movie_options.index(st.session_state.movie_selector),
                    key="movie_selector",
                    help="Choose a movie to get AI-powered similar recommendations"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True) # Adds a small vertical space
                st.button(
                    "🎲 Surprise Me!", 
                    on_click=set_random_movie, 
                    args=[current_filtered_df['title'].unique()],
                    use_container_width=True
                )
            
            num_recs = st.slider("📊 Number of recommendations:", 3, 15, 8)
            
            # Define the recommendation mode here, since the selectbox is removed
            rec_mode = "Balanced"

            if selected_movie_title and selected_movie_title != "":
                selected_movie_data = current_filtered_df[current_filtered_df['title'] == selected_movie_title].iloc[0]
                
                st.markdown("### 🎯 Your Selected Movie")
                movie_dict = selected_movie_data.to_dict()
                movie_dict['flag'] = safe_get(movie_dict, 'flag', get_language_flag(safe_get(movie_dict, 'language', '')))
                movie_dict['genre'] = safe_get(movie_dict, 'genre', safe_get(movie_dict, 'genre_category', 'Unknown'))
                display_movie_card_simple(movie_dict, "★")
                
                if st.button("🚀 Generate AI Recommendations", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing movie patterns and generating recommendations..."):
                        genre_df, cosine_sim = compute_similarity_for_genre_with_mode(
                            current_filtered_df, 
                            current_genre,
                            rec_mode
                        )
                        
                        recommendations = recommend_movies_in_genre(selected_movie_title, genre_df, cosine_sim, num_recs)
                        
                        if recommendations:
                            st.markdown(f"""
                            <div class="recommendation-section">
                                <h3 style="margin: 0; color: #333;">🎉 Your {current_genre} Recommendations</h3>
                                <p style="margin: 0.5rem 0 0 0; color: #666;">Based on "{selected_movie_title}", here are {len(recommendations)} movies you'll love:</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            languages_in_recs = set([r['language'] for r in recommendations])
                            ratings = [r['rating'] for r in recommendations if r['rating'] != 'N/A' and str(r['rating']).replace('.','').replace('-','').isdigit()]
                            avg_rating = safe_numeric_convert(np.mean([safe_numeric_convert(r) for r in ratings]), 0) if ratings else 0
                            recent_movies = sum([1 for r in recommendations if str(r['year']).isdigit() and int(r['year']) >= 2010])
                            
                            with col1:
                                st.metric("🎬 Recommendations", len(recommendations))
                            with col2:
                                st.metric("🌍 Languages", len(languages_in_recs))
                            with col3:
                                st.metric("⭐ Avg Rating", f"{avg_rating:.1f}" if avg_rating > 0 else "N/A")
                            with col4:
                                st.metric("🆕 Recent Movies", f"{recent_movies}/{len(recommendations)}")
                            
                            st.markdown("---")
                            
                            for i, movie in enumerate(recommendations, 1):
                                similarity_score = 0.95 - (i * 0.05)
                                display_movie_card_simple(movie, i, similarity_score)
                                
                        else:
                            st.error("😔 Sorry, couldn't find recommendations for this movie. Try selecting a different movie!")
                
            else:
                st.info("👆 Select a movie above to get personalized AI recommendations!")
        
        with tab2:
            st.markdown("### 📚 Browse Complete Collection")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                genre_languages = ['All Languages'] + sorted(current_filtered_df['language'].unique().tolist())
                genre_language_filter = st.selectbox("🌍 Filter by Language:", genre_languages, key="genre_lang")
            
            with col2:
                sort_options = {
                    "Rating (High → Low)": ("rating_numeric", False),
                    "Rating (Low → High)": ("rating_numeric", True),
                    "Year (Recent → Old)": ("year_numeric", False),
                    "Year (Old → Recent)": ("year_numeric", True),
                    "Title (A → Z)": ("title", True),
                    "Title (Z → A)": ("title", False)
                }
                sort_choice = st.selectbox("📊 Sort by:", list(sort_options.keys()))
            
            with col3:
                display_mode = st.selectbox("👁️ View Mode:", ["Detailed Cards", "Compact List"])
            
            browse_df = current_filtered_df.copy()
            
            if genre_language_filter != 'All Languages':
                browse_df = browse_df[browse_df['language'] == genre_language_filter].copy()
            
            sort_col, ascending = sort_options[sort_choice]
            if sort_col in ['rating_numeric', 'year_numeric']:
                browse_df.loc[:, sort_col] = pd.to_numeric(browse_df[sort_col] if sort_col in browse_df.columns else browse_df['rating' if 'rating' in sort_col else 'year'], errors='coerce')
            
            browse_df = browse_df.sort_values(sort_col, ascending=ascending, na_position='last')
            
            st.markdown(f"**Showing {len(browse_df):,} movies** ({genre_language_filter})")
            
            movies_per_page = 20 if display_mode == "Detailed Cards" else 50
            total_pages = (len(browse_df) - 1) // movies_per_page + 1 if len(browse_df) > 0 else 1
            
            if total_pages > 1:
                page = st.slider(f"📄 Page (1-{total_pages}):", 1, total_pages, 1)
                start_idx = (page - 1) * movies_per_page
                end_idx = start_idx + movies_per_page
                page_df = browse_df.iloc[start_idx:end_idx].copy()
            else:
                page_df = browse_df.head(movies_per_page).copy()
                page = 1
            
            if display_mode == "Detailed Cards":
                for i, (_, movie) in enumerate(page_df.iterrows(), 1):
                    movie_dict = movie.to_dict()
                    movie_dict['flag'] = safe_get(movie_dict, 'flag', get_language_flag(safe_get(movie_dict, 'language', '')))
                    movie_dict['genre'] = safe_get(movie_dict, 'genre', safe_get(movie_dict, 'genre_category', 'Unknown'))
                    display_movie_card_simple(movie_dict, i + (page-1) * movies_per_page)
            
            else:
                for i, (_, movie) in enumerate(page_df.iterrows(), 1):
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        title = safe_get(movie, 'title', 'Unknown')
                        flag = safe_get(movie, 'flag', get_language_flag(safe_get(movie, 'language', '')))
                        language = safe_get(movie, 'language', 'Unknown')
                        year = safe_get(movie, 'year', 'Unknown')
                        st.markdown(f"**{i + (page-1) * movies_per_page}. {title}**")
                        st.markdown(f"{flag} {language} • 📅 {year}")
                    with col2:
                        genre = safe_get(movie, 'genre_category', 'Unknown')
                        rating = safe_get(movie, 'rating', 'N/A')
                        st.markdown(f"🎭 {genre}")
                        if rating != 'N/A':
                            try:
                                rating_val = safe_numeric_convert(rating)
                                if rating_val > 0:
                                    st.markdown(f"⭐ {rating_val:.1f}")
                            except:
                                st.markdown(f"⭐ {rating}")
                    with col3:
                        if st.button("👁️ Details", key=f"details_{i}_{page}"):
                            description = safe_get(movie, 'description', 'No description available.')
                            st.info(f"**{title}**\n\n{description}")
                    st.divider()
        
        with tab3:
            st.markdown("### 📊 Advanced Analytics Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rating = create_rating_distribution(current_filtered_df)
                fig_rating.update_layout(title=f"⭐ {current_genre} Rating Distribution")
                st.plotly_chart(fig_rating, use_container_width=True)
                
                try:
                    lang_counts = current_filtered_df['language'].value_counts().head(8)
                    if len(lang_counts) > 0:
                        fig_lang = px.bar(
                            x=lang_counts.values,
                            y=lang_counts.index,
                            orientation='h',
                            title=f"🌍 Languages in {current_genre}",
                            color=lang_counts.values,
                            color_continuous_scale="plasma"
                        )
                        st.plotly_chart(fig_lang, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating language chart: {str(e)}")
            
            with col2:
                try:
                    current_filtered_df.loc[:, 'year_numeric'] = pd.to_numeric(current_filtered_df['year'], errors='coerce')
                    genre_df_clean = current_filtered_df.dropna(subset=['year_numeric']).copy()
                    
                    if len(genre_df_clean) > 0:
                        fig_year = px.histogram(
                            genre_df_clean,
                            x='year_numeric',
                            nbins=20,
                            title=f"📅 {current_genre} Movies by Year",
                            color_discrete_sequence=['#f093fb']
                        )
                        st.plotly_chart(fig_year, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating year distribution: {str(e)}")
                
                try:
                    directors = current_filtered_df['director'].value_counts().head(10)
                    directors = directors[directors.index != 'Director information not available']
                    directors = directors[directors.index != 'N/A']
                    if len(directors) > 0:
                        fig_directors = px.bar(
                            x=directors.values,
                            y=directors.index,
                            orientation='h',
                            title=f"🎬 Top Directors in {current_genre}",
                            color=directors.values,
                            color_continuous_scale="viridis"
                        )
                        st.plotly_chart(fig_directors, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating directors chart: {str(e)}")
            
            st.markdown("### 🔍 Detailed Statistics")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                try:
                    current_filtered_df.loc[:, 'rating_numeric'] = pd.to_numeric(current_filtered_df['rating'], errors='coerce')
                    avg_rating = current_filtered_df['rating_numeric'].mean()
                    avg_rating_display = f"{avg_rating:.1f}" if not pd.isna(avg_rating) else 'N/A'
                    st.metric("⭐ Average Rating", avg_rating_display)
                except:
                    st.metric("⭐ Average Rating", "N/A")
            
            with metrics_col2:
                try:
                    latest_year = current_filtered_df['year_numeric'].max()
                    latest_year_display = int(latest_year) if not pd.isna(latest_year) else 'N/A'
                    st.metric("📅 Latest Movie", latest_year_display)
                except:
                    st.metric("📅 Latest Movie", "N/A")
            
            with metrics_col3:
                try:
                    high_rated = len(current_filtered_df[current_filtered_df['rating_numeric'] >= 7.0])
                    st.metric("🏆 High Rated (7+)", high_rated)
                except:
                    st.metric("🏆 High Rated (7+)", 0)
            
            with metrics_col4:
                try:
                    recent_movies = len(current_filtered_df[current_filtered_df['year_numeric'] >= 2015])
                    st.metric("🆕 Recent (2015+)", recent_movies)
                except:
                    st.metric("🆕 Recent (2015+)", 0)
        
        with tab4:
            st.markdown("### 🔥 Trending & Popular Movies")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏆 Highest Rated Movies")
                try:
                    top_rated = current_filtered_df.copy()
                    top_rated['rating_numeric'] = pd.to_numeric(top_rated['rating'], errors='coerce')
                    top_rated = top_rated.dropna(subset=['rating_numeric'])
                    top_rated = top_rated.nlargest(10, 'rating_numeric')
                    
                    for i, (_, movie) in enumerate(top_rated.iterrows(), 1):
                        title = safe_get(movie, 'title', 'Unknown')
                        flag = safe_get(movie, 'flag', get_language_flag(safe_get(movie, 'language', '')))
                        language = safe_get(movie, 'language', 'Unknown')
                        year = safe_get(movie, 'year', 'Unknown')
                        rating = safe_get(movie, 'rating', 'N/A')
                        
                        try:
                            rating_val = safe_numeric_convert(rating)
                            rating_display = f"{rating_val:.1f}" if rating_val > 0 else rating
                        except:
                            rating_display = rating
                        
                        st.markdown(f"**{i}. {title}**")
                        st.markdown(f"{flag} {language} • {year} • ⭐ {rating_display}")
                        st.divider()
                except Exception as e:
                    st.error(f"Error loading highest rated movies: {str(e)}")
            
            with col2:
                st.markdown("#### 🆕 Recent Releases")
                try:
                    recent = current_filtered_df.copy()
                    recent['year_numeric'] = pd.to_numeric(recent['year'], errors='coerce')
                    recent = recent.dropna(subset=['year_numeric'])
                    recent = recent.nlargest(10, 'year_numeric')
                    
                    for i, (_, movie) in enumerate(recent.iterrows(), 1):
                        title = safe_get(movie, 'title', 'Unknown')
                        flag = safe_get(movie, 'flag', get_language_flag(safe_get(movie, 'language', '')))
                        language = safe_get(movie, 'language', 'Unknown')
                        year_numeric = safe_get(movie, 'year_numeric', 0)
                        
                        try:
                            year_display = int(year_numeric) if year_numeric > 0 else 'Unknown'
                        except:
                            year_display = 'Unknown'
                        
                        st.markdown(f"**{i}. {title}**")
                        st.markdown(f"{flag} {language} • 📅 {year_display}")
                        st.divider()
                except Exception as e:
                    st.error(f"Error loading recent releases: {str(e)}")
            
            st.markdown("#### 🎭 Popular Director-Actor Combinations")
            try:
                combos = []
                for _, movie in current_filtered_df.iterrows():
                    director = safe_get(movie, 'director', '')
                    cast = safe_get(movie, 'cast', '')
                    
                    if (director not in ['N/A', 'Director information not available', ''] and
                        cast not in ['N/A', 'Cast information not available', '']):
                        cast_list = str(cast).split()
                        if cast_list:
                            actor = cast_list[0]
                            combos.append({
                                'combo': f"{director} + {actor}",
                                'movie': safe_get(movie, 'title', 'Unknown'),
                                'rating': safe_get(movie, 'rating', 'N/A'),
                                'year': safe_get(movie, 'year', 'Unknown')
                            })
                
                if combos:
                    combos_df = pd.DataFrame(combos)
                    combo_counts = combos_df['combo'].value_counts().head(5)
                    
                    for combo, count in combo_counts.items():
                        movies_with_combo = combos_df[combos_df['combo'] == combo]
                        st.markdown(f"**🎬 {combo}** - {count} movie(s)")
                        for _, combo_movie in movies_with_combo.head(3).iterrows():
                            st.markdown(f"   • {combo_movie['movie']} ({combo_movie['year']})")
                else:
                    st.info("No director-actor combinations found in this genre.")
            except Exception as e:
                st.error(f"Error loading director-actor combinations: {str(e)}")
    
    else:
        st.info("👆 Select a genre above to start exploring movies!")
        
        st.markdown("## 🌟 Database Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎬 Total Movies", f"{len(df):,}")
        
        with col2:
            st.metric("🌍 Languages", f"{df['language'].nunique()}")
        
        with col3:
            st.metric("🎭 Genres", f"{df['genre_category'].nunique()}")
        
        with col4:
            try:
                df['year_numeric'] = pd.to_numeric(df['year'], errors='coerce')
                year_span = int(df['year_numeric'].max() - df['year_numeric'].min()) if not df['year_numeric'].isna().all() else 0
                st.metric("📅 Year Span", f"{year_span}")
            except:
                st.metric("📅 Year Span", "N/A")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_genre = create_genre_visualization(df)
            st.plotly_chart(fig_genre, use_container_width=True)
        
        with col2:
            fig_lang = create_language_chart(df)
            st.plotly_chart(fig_lang, use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 2rem 0;">
            <h3>🎬 CineMatch</h3>
            <p>Powered by Advanced Machine Learning & Natural Language Processing</p>
            <p style="opacity: 0.8; font-size: 0.9rem;">
                Built with Streamlit • Scikit-learn • Plotly • Pandas<br>
                Created for movie enthusiasts worldwide 🌍
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()