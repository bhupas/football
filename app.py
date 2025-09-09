import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="⚽ Fodboldlinjen AI Elite Coach",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced CSS for Better UI and Dark Mode Support ---
st.markdown("""
<style>
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }
        .highlight-box {
            background-color: #2d2d2d !important;
            border-left: 4px solid #4a9eff !important;
            color: #ffffff !important;
        }
        div[data-testid="metric-container"] {
            background-color: #2d2d2d !important;
            border: 1px solid #404040 !important;
            color: #ffffff !important;
        }
        .feedback-card {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
    }
    
    /* Light mode styles */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: 500;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .highlight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    div[data-testid="metric-container"] {
        background-color: #f7f7f7;
        border: 1px solid #e1e1e1;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .logo-container {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .logo-text {
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .feedback-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'feedback_analysis_cache' not in st.session_state:
    st.session_state.feedback_analysis_cache = {}

# --- Sidebar Configuration ---
st.sidebar.markdown("""
<div class="logo-container">
    <div class="logo-text">⚽ FC Elite</div>
    <div style="color: white; font-size: 12px; margin-top: 5px;">AI Coaching Platform</div>
</div>
""", unsafe_allow_html=True)

# --- HELPER & CLEANING FUNCTIONS ---
def clean_text(text):
    if not isinstance(text, str): 
        text = str(text)
    return re.sub(r'\s+', ' ', text.lower().strip())

def find_header_row(file_or_df, expected_cols, sample_rows=15):
    df_sample = None
    if isinstance(file_or_df, pd.DataFrame):
        df_sample = file_or_df.head(sample_rows)
    else:
        try:
            file_or_df.seek(0)
            df_sample = pd.read_csv(file_or_df, header=None, nrows=sample_rows, on_bad_lines='skip', encoding='utf-8')
            file_or_df.seek(0)
        except Exception:
            return None

    cleaned_expected_cols = [clean_text(col) for col in expected_cols]
    best_match_count, header_row_index = 0, None

    for i, row in df_sample.iterrows():
        cleaned_row_values = [clean_text(cell) for cell in row.values]
        match_count = sum(1 for expected_col in cleaned_expected_cols if expected_col in cleaned_row_values)
        if match_count > best_match_count and match_count > len(expected_cols) * 0.5:
            best_match_count, header_row_index = match_count, i
    return header_row_index

# --- ENHANCED DATA LOADING & FEATURE ENGINEERING ---
@st.cache_data
def load_and_prepare_data(uploaded_files):
    all_dfs = []
    column_mapping = {
        'tidsstempel': 'Timestamp', 
        'kamp - hvilket hold spillede du for': 'Team',
        'modstanderen (hvem spillede du mod)': 'Opponent', 
        'navn (fulde navn)': 'Player',
        '#succesfulde pasninger /indlæg': 'Successful_Passes',
        '#total pasninger/indlæg (succesfulde + ikke succesfulde)': 'Total_Passes',
        '#total afslutninger': 'Total_Shots',
        '#succesfulde erobringer på egen bane': 'Tackles_Own_Half',
        '#succesfulde erobringer på deres bane': 'Tackles_Opponent_Half',
        '#total succesfulde erobringer (egen + deres bane)': 'Total_Tackles',
        'hvad vil du gøre bedre i næste kamp ?': 'Feedback'
    }
    expected_cols = list(column_mapping.keys())

    for uploaded_file in uploaded_files:
        try:
            content_bytes = uploaded_file.getvalue()
            if uploaded_file.name.endswith('.xlsx'):
                excel_file = io.BytesIO(content_bytes)
                sheets = pd.read_excel(excel_file, sheet_name=None, header=None, engine='openpyxl')
                for _, sheet_df in sheets.items():
                    header_row = find_header_row(sheet_df, expected_cols)
                    if header_row is not None:
                        sheet_df.columns = [clean_text(col) for col in sheet_df.iloc[header_row]]
                        sheet_df = sheet_df.drop(sheet_df.index[:header_row + 1]).reset_index(drop=True)
                        all_dfs.append(sheet_df)
            elif uploaded_file.name.endswith('.csv'):
                string_io = io.StringIO(content_bytes.decode('utf-8'))
                header_row = find_header_row(string_io, expected_cols)
                if header_row is not None:
                    string_io.seek(0)
                    df = pd.read_csv(string_io, on_bad_lines='skip', header=header_row, encoding='utf-8')
                    df.columns = [clean_text(col) for col in df.columns]
                    all_dfs.append(df)
        except Exception as e:
            st.error(f"❌ Could not process file {uploaded_file.name}. Error: {e}")
            continue

    if not all_dfs: 
        return None

    df = pd.concat(all_dfs, ignore_index=True).rename(columns=column_mapping)
    df.dropna(subset=['Player', 'Opponent'], inplace=True)
    df = df[df['Player'].astype(str).str.strip().str.lower() != 'nan']

    # Convert numeric columns with better error handling
    numeric_cols = list(column_mapping.values())[4:-1]
    for col in numeric_cols:
        if col not in df.columns: 
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Process timestamps
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Clean text columns
    for col in ['Player', 'Opponent', 'Team']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Data corrections and validations
    pass_mask = df['Successful_Passes'] > df['Total_Passes']
    df.loc[pass_mask, 'Total_Passes'] = df.loc[pass_mask, 'Successful_Passes']
    
    tackle_sum = df['Tackles_Own_Half'] + df['Tackles_Opponent_Half']
    tackle_mask = tackle_sum != df['Total_Tackles']
    df.loc[tackle_mask, 'Total_Tackles'] = tackle_sum

    # Create match identifier
    if 'Timestamp' in df.columns and df['Timestamp'].notna().any():
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        df['Match'] = df.apply(lambda r: f"{r['Opponent']} ({r['Timestamp'].strftime('%Y-%m-%d')})" 
                              if pd.notna(r['Timestamp']) else r['Opponent'], axis=1)
    else:
        df['Match'] = df['Opponent']

    # Enhanced Feature Engineering with safeguards
    df['Passing_Accuracy'] = np.where(df['Total_Passes'] > 0, 
                                      (df['Successful_Passes'] / df['Total_Passes']) * 100, 0)
    df['Defensive_Action_Ratio'] = np.where(df['Total_Tackles'] > 0, 
                                           (df['Tackles_Opponent_Half'] / df['Total_Tackles']) * 100, 0)
    df['Offensive_Contribution'] = df['Total_Shots'] * 2 + df['Successful_Passes'] * 0.5
    df['Defensive_Contribution'] = df['Tackles_Opponent_Half'] * 2 + df['Tackles_Own_Half']
    df['Shots_per_Pass'] = np.where(df['Total_Passes'] > 0, 
                                    df['Total_Shots'] / df['Total_Passes'] * 100, 0)
    df['Player_Involvement'] = df['Total_Passes'] + df['Total_Shots'] + df['Total_Tackles']
    df['Overall_Impact'] = df['Offensive_Contribution'] + df['Defensive_Contribution']
    
    # Advanced metrics with proper safeguards
    df['Pressing_Intensity'] = np.where(df['Total_Tackles'] > 0,
                                        df['Tackles_Opponent_Half'] / df['Total_Tackles'] * 100, 0)
    df['Shooting_Efficiency'] = np.where(df['Player_Involvement'] > 0,
                                         df['Total_Shots'] / df['Player_Involvement'] * 100, 0)
    df['Ball_Retention'] = np.where(df['Player_Involvement'] > 0,
                                    df['Passing_Accuracy'] * (df['Total_Passes'] / df['Player_Involvement']), 0)
    df['Defensive_Workrate'] = np.where(df['Player_Involvement'] > 0,
                                        df['Total_Tackles'] / df['Player_Involvement'] * 100, 0)
    
    # Performance rating with percentile capping for outliers
    off_contrib_95 = df['Offensive_Contribution'].quantile(0.95) if len(df) > 10 else df['Offensive_Contribution'].max()
    def_contrib_95 = df['Defensive_Contribution'].quantile(0.95) if len(df) > 10 else df['Defensive_Contribution'].max()
    
    df['Performance_Rating'] = (
        df['Passing_Accuracy'] * 0.25 +
        df['Defensive_Action_Ratio'] * 0.15 +
        np.clip(df['Offensive_Contribution'] / (off_contrib_95 + 0.01) * 100, 0, 100) * 0.3 +
        np.clip(df['Defensive_Contribution'] / (def_contrib_95 + 0.01) * 100, 0, 100) * 0.3
    )
    
    return df

# --- WORDCLOUD GENERATION ---
def generate_wordcloud(text, title="Word Cloud"):
    """Generate a word cloud from text"""
    if not text or text.strip() == "":
        return None
    
    # Danish stop words to exclude
    stop_words = set(['i', 'og', 'at', 'er', 'på', 'med', 'for', 'det', 'som', 'en', 'af', 
                      'til', 'har', 'jeg', 'vi', 'de', 'den', 'der', 'kan', 'vil', 'skal',
                      'være', 'blev', 'blive', 'været', 'var', 'min', 'mit', 'mine',
                      'mere', 'når', 'hvor', 'hvordan', 'hvilken', 'hvis', 'have'])
    
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          stopwords=stop_words,
                          colormap='viridis',
                          relative_scaling=0.5,
                          max_words=100).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    return fig

# --- ENHANCED AI MODULE ---
def get_ai_recommendations(df, full_df, selection_name, analysis_type="general"):
    if not st.session_state.api_key_configured:
        st.error("🔑 AI Coach is disabled. Please configure your Gemini API Key.")
        return None

    with st.spinner("🧠 AI Coach analyserer data..."):
        try:
            # Prepare comprehensive statistics
            feedback_text = ' '.join(df['Feedback'].dropna().astype(str))[:1000]
            
            # Calculate trend data if possible
            trend_info = ""
            if 'Timestamp' in df.columns and df['Timestamp'].notna().any():
                recent_games = df.nlargest(3, 'Timestamp') if len(df) >= 3 else df
                older_games = df.nsmallest(3, 'Timestamp') if len(df) >= 6 else pd.DataFrame()
                
                if not older_games.empty:
                    recent_perf = recent_games['Performance_Rating'].mean()
                    older_perf = older_games['Performance_Rating'].mean()
                    trend = "📈 Opadgående" if recent_perf > older_perf else "📉 Nedadgående"
                    trend_info = f"\n- Performance Trend: {trend} ({older_perf:.1f} → {recent_perf:.1f})"
            
            # Determine analysis focus
            analysis_prompts = {
                "tactical": """
                Fokuser på TAKTISKE elementer:
                - Formation og positionering
                - Pressstrategi og kompakthed
                - Omstillinger (defensiv→offensiv og omvendt)
                - Rumudnyttelse og bevægelsesmønstre
                - Samarbejde mellem kæder
                """,
                "individual": """
                Fokuser på INDIVIDUELLE spillerpræstationer:
                - Tekniske færdigheder der skal forbedres
                - Fysiske aspekter (udholdenhed, hurtighed, styrke)
                - Mentale aspekter (beslutningstagning, mod, lederskab)
                - Specifik rolleforståelse
                """,
                "physical_mental": """
                Fokuser på FYSISKE og MENTALE aspekter:
                - Kondition og udholdenhed gennem kampen
                - Mental styrke og fokus
                - Kommunikation og lederskab
                - Håndtering af pres og modgang
                """,
                "feedback": """
                Fokuser på SPILLERFEEDBACK analyse:
                - Hovedtemaer i spillernes feedback
                - Gentagende udfordringer
                - Motivation og mentalitet
                - Konkrete forbedringspunkter spillerne selv identificerer
                - Anbefalinger baseret på spillernes input
                """,
                "general": """
                Giv en BALANCERET analyse der dækker:
                - Holdets samlede præstation
                - Taktiske observationer
                - Individuelle højdepunkter og udviklingsområder
                - Praktiske træningsøvelser
                """
            }
            
            prompt_focus = analysis_prompts.get(analysis_type, analysis_prompts["general"])

            prompt = f"""
            Du er en erfaren dansk fodboldtræner og taktisk ekspert der analyserer ungdomsfodbold på eliteniveau.
            Din analyse skal være skarp, konkret og handlingsorienteret.

            **DATA FOR: {selection_name}**

            📊 **Kvantitative Nøgletal:**
            - Kampe analyseret: {df['Match'].nunique()}
            - Gennemsnitlig Performance Rating: {df['Performance_Rating'].mean():.1f}/100
            - Pasningspræcision: {df['Passing_Accuracy'].mean():.1f}%
            - Skudfrekvens: {df['Total_Shots'].mean():.1f} per kamp
            - Presintensitet: {df['Pressing_Intensity'].mean():.1f}%
            - Defensiv arbejdsrate: {df['Defensive_Workrate'].mean():.1f}%{trend_info}

            💭 **Spillernes Feedback:**
            "{feedback_text}"

            📋 **ANALYSEOMRÅDE:**
            {prompt_focus}

            **DIN OPGAVE:**
            Skriv en professionel trænerrapport på DANSK med følgende struktur:

            ## 🎯 Hovedkonklusioner
            [3-4 skarpe observationer baseret på data]

            ## 💪 Styrker at Bygge På
            [Konkrete styrker med data-backing]

            ## ⚠️ Kritiske Udviklingsområder
            [Specifikke svagheder der SKAL addresses]

            ## 🏃 Træningsplan (Næste 2 Uger)
            [Konkrete øvelser og fokuspunkter]

            ## 📈 Målsætninger for Næste 3 Kampe
            [Specifikke, målbare mål]

            Vær KONKRET og HANDLINGSORIENTERET.
            """
            
            # Try different model configurations
            model = None
            model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    break
                except Exception:
                    continue
            
            if not model:
                st.error("❌ Kunne ikke initialisere AI model.")
                return None
            
            generation_config = genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            )
            
            response = model.generate_content(prompt, generation_config=generation_config)
            
            return response.text
            
        except Exception as e:
            st.error(f"❌ Kunne ikke generere AI-indsigt: {e}")
            return None

# --- FEEDBACK ANALYSIS AI ---
def analyze_feedback_with_ai(feedback_df, match_filter=None):
    """Analyze feedback using AI for deeper insights"""
    if not st.session_state.api_key_configured:
        return None
    
    # Filter by match if specified
    if match_filter:
        feedback_df = feedback_df[feedback_df['Match'] == match_filter]
    
    # Prepare feedback text
    all_feedback = []
    for _, row in feedback_df.iterrows():
        if pd.notna(row.get('Feedback')):
            all_feedback.append(f"{row['Player']}: {row['Feedback']}")
    
    if not all_feedback:
        return None
    
    feedback_text = "\n".join(all_feedback[:50])  # Limit to 50 entries
    
    prompt = f"""
    Du er en erfaren fodboldtræner der analyserer spillerfeedback.
    
    **SPILLERFEEDBACK:**
    {feedback_text}
    
    **ANALYSER OG GIV:**
    
    ## 📊 Hovedtemaer
    Identificer 3-5 gentagende temaer i spillernes feedback
    
    ## 💡 Nøgleindsigter
    Hvad fortæller feedbacken om holdets mentale tilstand og udviklingsbehov?
    
    ## ⚠️ Kritiske Områder
    Hvilke områder kræver øjeblikkelig opmærksomhed?
    
    ## 🎯 Træneranbefalinger
    5 konkrete handlinger træneren bør tage baseret på denne feedback
    
    ## 🏃 Næste Kamp Fokus
    3 specifikke fokuspunkter for næste kamp baseret på spillernes input
    
    Vær konkret og handlingsorienteret i dine anbefalinger.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Kunne ikke analysere feedback: {e}")
        return None

# --- ENHANCED CLUSTERING ---
def perform_advanced_clustering(df):
    st.header("🤖 AI Spillerprofil Analyse", divider="rainbow")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("Dette AI-værktøj identificerer spillertyper og taktiske roller baseret på præstationsdata.", icon="ℹ️")
    
    player_avg_stats = df.groupby('Player').mean(numeric_only=True)
    
    # Enhanced metrics for clustering
    metrics_for_clustering = [
        'Passing_Accuracy', 'Offensive_Contribution', 'Defensive_Contribution',
        'Pressing_Intensity', 'Ball_Retention', 'Shooting_Efficiency', 
        'Performance_Rating', 'Player_Involvement'
    ]
    
    valid_metrics = [m for m in metrics_for_clustering if m in player_avg_stats.columns]
    
    if len(valid_metrics) < 4 or len(player_avg_stats) < 4:
        st.warning("⚠️ Ikke nok data eller spillere (minimum 4) til meningsfuld clustering.")
        return

    X = player_avg_stats[valid_metrics].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # User input for number of clusters
    k = st.slider("🎯 Antal spillerprofiler at identificere:", 2, min(6, len(player_avg_stats)-1), 3)

    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    player_avg_stats['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    player_avg_stats['pca_one'] = components[:, 0]
    player_avg_stats['pca_two'] = components[:, 1]
    
    # Enhanced cluster interpretation
    cluster_names = interpret_clusters_enhanced(player_avg_stats, valid_metrics, k)
    player_avg_stats['Role'] = player_avg_stats['Cluster'].map(cluster_names)
    
    # Visualization
    st.subheader("📊 Taktisk Spillerkort")
    
    fig = px.scatter(
        player_avg_stats.reset_index(), 
        x="pca_one", 
        y="pca_two", 
        color='Role',
        hover_name='Player',
        title="Spillerprofiler og Taktiske Roller",
        labels={'pca_one': 'Offensiv ← → Defensiv', 'pca_two': 'Teknisk ← → Fysisk'},
        hover_data={
            'Performance_Rating': ':.1f',
            'Passing_Accuracy': ':.1f',
            'pca_one': False, 
            'pca_two': False
        },
        size='Performance_Rating',
        size_max=20
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis per cluster
    st.subheader("📋 Detaljeret Profilanalyse")
    
    for i in range(k):
        cluster_df = player_avg_stats[player_avg_stats['Cluster'] == i]
        role_name = cluster_names[i]
        
        with st.expander(f"**{role_name}** ({len(cluster_df)} spillere)", expanded=(i==0)):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"##### 👥 Spillere i denne rolle:")
                for player in cluster_df.index:
                    rating = cluster_df.loc[player, 'Performance_Rating']
                    st.write(f"• **{player}** (Rating: {rating:.1f})")
                
                # Key characteristics
                st.markdown("##### 📊 Nøglekarakteristika:")
                cluster_avg = cluster_df[valid_metrics].mean()
                team_avg = player_avg_stats[valid_metrics].mean()
                
                for metric in valid_metrics[:5]:
                    diff = (cluster_avg[metric] - team_avg[metric]) / (team_avg[metric] + 0.01) * 100
                    if abs(diff) > 10:
                        emoji = "📈" if diff > 0 else "📉"
                        st.write(f"{emoji} {metric.replace('_', ' ')}: {diff:+.0f}% vs. gennemsnit")
            
            with col2:
                # Radar chart for profile
                fig_radar = create_radar_chart(cluster_avg, team_avg, valid_metrics, role_name)
                st.plotly_chart(fig_radar, use_container_width=True)

def interpret_clusters_enhanced(player_stats, metrics, k):
    """Enhanced cluster interpretation with unique role names"""
    cluster_profiles = []
    
    for i in range(k):
        cluster_df = player_stats[player_stats['Cluster'] == i]
        cluster_avg = cluster_df[metrics].mean()
        
        profile = {
            'cluster': i,
            'offensive': cluster_avg.get('Offensive_Contribution', 0),
            'defensive': cluster_avg.get('Defensive_Contribution', 0),
            'passing': cluster_avg.get('Passing_Accuracy', 0),
            'pressing': cluster_avg.get('Pressing_Intensity', 0),
            'shooting': cluster_avg.get('Shooting_Efficiency', 0),
            'retention': cluster_avg.get('Ball_Retention', 0),
            'involvement': cluster_avg.get('Player_Involvement', 0),
            'rating': cluster_avg.get('Performance_Rating', 0)
        }
        cluster_profiles.append(profile)
    
    # Sort profiles to assign unique names
    cluster_profiles.sort(key=lambda x: (x['offensive'], x['defensive'], x['passing']), reverse=True)
    
    cluster_names = {}
    used_names = set()
    
    for idx, profile in enumerate(cluster_profiles):
        # Determine best fitting role based on stats
        if profile['offensive'] > profile['defensive'] * 1.5:
            if profile['shooting'] > 10:
                name = "⚡ Elite Angriber"
            elif profile['passing'] > 75:
                name = "🎯 Kreativ Playmaker"
            else:
                name = "🎨 Offensiv Tekniker"
        elif profile['defensive'] > profile['offensive'] * 1.5:
            if profile['pressing'] > 60:
                name = "🛡️ Defensiv Anker"
            else:
                name = "🧱 Forsvarsmur"
        elif profile['passing'] > 70 and profile['retention'] > 40:
            name = "🎮 Spilfordeler"
        elif profile['involvement'] > player_stats['Player_Involvement'].mean():
            if profile['pressing'] > 50:
                name = "⚖️ Box-to-Box Motor"
            else:
                name = "🏃 Arbejdshest"
        else:
            name = "🔄 Allround Spiller"
        
        # Ensure unique names
        if name in used_names:
            modifiers = ["Elite", "Dynamisk", "Taktisk", "Teknisk", "Fysisk"]
            for mod in modifiers:
                modified_name = f"{mod} {name.split(' ', 1)[1] if ' ' in name else name}"
                if modified_name not in used_names:
                    name = modified_name
                    break
            else:
                name = f"{name} Type {idx+1}"
        
        used_names.add(name)
        cluster_names[profile['cluster']] = name
    
    return cluster_names

def create_radar_chart(cluster_avg, team_avg, metrics, title):
    """Create an enhanced radar chart for cluster visualization"""
    fig = go.Figure()
    
    # Normalize values for better visualization
    max_vals = team_avg.max()
    cluster_norm = (cluster_avg / (max_vals + 0.01) * 100).values
    team_norm = (team_avg / (max_vals + 0.01) * 100).values
    
    fig.add_trace(go.Scatterpolar(
        r=cluster_norm,
        theta=[m.replace('_', ' ') for m in metrics],
        fill='toself',
        name=title,
        line_color='blue',
        fillcolor='rgba(0, 100, 255, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=team_norm,
        theta=[m.replace('_', ' ') for m in metrics],
        name='Hold Gns.',
        line=dict(dash='dot', color='grey'),
        fillcolor='rgba(128, 128, 128, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# --- MAIN APPLICATION ---
def main():
    # Header with enhanced gradient
    st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); 
    color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    ⚽ Fodboldlinjen AI Elite Coach
    </h1>
    """, unsafe_allow_html=True)
    
    # Sidebar file upload
    st.sidebar.header("📁 Upload Data")
    uploaded_files = st.sidebar.file_uploader(
        "Vælg CSV/Excel filer", 
        type=["csv", "xlsx"], 
        accept_multiple_files=True,
        help="Upload dine kampdata filer her"
    )
    
    # Configuration in sidebar (moved here)
    with st.sidebar.expander("⚙️ Configuration", expanded=False):
        # API Key Configuration
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            st.session_state.api_key_configured = True
            st.success("✅ API Key configured")
        except (KeyError, AttributeError):
            gemini_api_key = st.text_input("🔑 Enter your Gemini API Key", type="password")
            if gemini_api_key:
                try:
                    genai.configure(api_key=gemini_api_key)
                    st.session_state.api_key_configured = True
                    st.success("✅ API Key configured!")
                except Exception as e:
                    st.error(f"❌ Invalid API Key: {e}")
    
    if uploaded_files:
        df = load_and_prepare_data(uploaded_files)
        
        if df is None or df.empty:
            st.error("❌ Kunne ikke indlæse data. Tjek filformat og indhold.")
            st.stop()
        
        # Sidebar filters
        st.sidebar.header("🔍 Filtre")
        
        # Date filter if timestamp exists
        if 'Timestamp' in df.columns and df['Timestamp'].notna().any():
            date_range = st.sidebar.date_input(
                "📅 Vælg datointerval",
                value=(df['Timestamp'].min(), df['Timestamp'].max()),
                min_value=df['Timestamp'].min(),
                max_value=df['Timestamp'].max()
            )
            
            if len(date_range) == 2:
                df = df[(df['Timestamp'].dt.date >= date_range[0]) & 
                       (df['Timestamp'].dt.date <= date_range[1])]
        
        all_players = sorted(df['Player'].unique())
        all_opponents = sorted(df['Opponent'].dropna().unique())
        
        # Advanced filter options
        filter_mode = st.sidebar.radio("Filter Mode:", ["Standard", "Avanceret"])
        
        if filter_mode == "Standard":
            selected_players = st.sidebar.multiselect(
                "👤 Vælg Spillere", 
                options=all_players, 
                default=all_players
            )
            selected_opponents = st.sidebar.multiselect(
                "🆚 Vælg Modstandere", 
                options=all_opponents, 
                default=all_opponents
            )
        else:
            # Advanced filters
            min_rating = st.sidebar.slider(
                "Min. Performance Rating",
                0, 100, 0
            )
            selected_players = st.sidebar.multiselect(
                "👤 Vælg Spillere", 
                options=all_players
            )
            if not selected_players:
                selected_players = all_players
            
            selected_opponents = st.sidebar.multiselect(
                "🆚 Vælg Modstandere", 
                options=all_opponents
            )
            if not selected_opponents:
                selected_opponents = all_opponents
            
            df = df[df['Performance_Rating'] >= min_rating]
        
        # Apply filters
        df_filtered = df[
            df['Player'].isin(selected_players) & 
            df['Opponent'].isin(selected_opponents)
        ]
        
        if df_filtered.empty:
            st.warning("⚠️ Ingen data for valgte filtre.")
            st.stop()
        
        # Create tabs with icons (including new Feedback tab)
        tabs = st.tabs([
            "📊 Dashboard",
            "👤 Spiller Analyse", 
            "⚔️ Head-to-Head",
            "📈 Trends & Udvikling",
            "🎯 Taktisk Analyse",
            "💭 Feedback Analyse",  # NEW TAB
            "🤖 AI Profiler",
            "🧠 AI Coach",
            "📝 Rapporter"
        ])
        
        # Tab 1: Dashboard
        with tabs[0]:
            render_dashboard(df_filtered, df)
        
        # Tab 2: Player Analysis
        with tabs[1]:
            render_player_analysis(df_filtered, df)
        
        # Tab 3: Head-to-Head Comparison
        with tabs[2]:
            render_head_to_head(df_filtered)
        
        # Tab 4: Trends & Development
        with tabs[3]:
            render_trends(df_filtered)
        
        # Tab 5: Tactical Analysis
        with tabs[4]:
            render_tactical_analysis(df_filtered, df)
        
        # Tab 6: Feedback Analysis (NEW)
        with tabs[5]:
            render_feedback_analysis(df_filtered, df)
        
        # Tab 7: AI Clustering
        with tabs[6]:
            perform_advanced_clustering(df_filtered)
        
        # Tab 8: AI Coach
        with tabs[7]:
            render_ai_coach(df_filtered, df)
        
        # Tab 9: Reports
        with tabs[8]:
            render_reports(df_filtered)
        
    else:
        # Enhanced welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2>👋 Velkommen til AI Elite Coach!</h2>
            <p style='font-size: 18px; margin: 20px;'>
                Upload dine kampdata i sidemenuen for at starte din fodboldanalyse.
            </p>
            <p style='color: #f0f0f0;'>
                Understøtter CSV og Excel filer med kampstatistikker.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show demo data structure
        with st.expander("📋 Se forventet dataformat"):
            st.markdown("""
            **Påkrævede kolonner:**
            - Navn (Fulde Navn)
            - Modstanderen (Hvem Spillede Du Mod)
            - #Succesfulde Pasninger /Indlæg
            - #Total Pasninger/Indlæg
            - #Total Afslutninger
            - #Succesfulde Erobringer på EGEN Bane
            - #Succesfulde Erobringer på DERES Bane
            - #Total Succesfulde Erobringer
            - Hvad Vil Du Gøre Bedre i Næste Kamp?
            
            **Valgfrie kolonner:**
            - Tidsstempel
            - Kamp - Hvilket Hold Spillede Du For
            """)

# --- NEW: FEEDBACK ANALYSIS TAB ---
def render_feedback_analysis(df_filtered, df_full):
    """Render comprehensive feedback analysis with wordclouds and AI insights"""
    st.header("💭 Spillerfeedback Analyse", divider="rainbow")
    
    # Check if feedback column exists
    if 'Feedback' not in df_filtered.columns:
        st.warning("⚠️ Ingen feedback data tilgængelig.")
        return
    
    feedback_df = df_filtered[['Player', 'Match', 'Feedback', 'Performance_Rating']].dropna(subset=['Feedback'])
    
    if feedback_df.empty:
        st.warning("⚠️ Ingen feedback fundet for de valgte filtre.")
        return
    
    # Main analysis selector
    analysis_type = st.selectbox(
        "Vælg analysetype:",
        ["📊 Oversigt", "☁️ Ordsky Analyse", "🎯 Kampspecifik Analyse", 
         "👤 Spillerspecifik Feedback", "🤖 AI Feedback Analyse"]
    )
    
    if analysis_type == "📊 Oversigt":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Feedback", len(feedback_df))
            st.metric("Unikke Spillere", feedback_df['Player'].nunique())
        
        with col2:
            st.metric("Kampe med Feedback", feedback_df['Match'].nunique())
            avg_length = feedback_df['Feedback'].str.len().mean()
            st.metric("Gns. Feedback Længde", f"{avg_length:.0f} tegn")
        
        with col3:
            # Most common themes
            all_feedback_text = ' '.join(feedback_df['Feedback'].values)
            common_words = ['bedre', 'mere', 'skal', 'pres', 'pasninger', 'skud', 'spil']
            word_counts = {word: all_feedback_text.lower().count(word) for word in common_words}
            top_word = max(word_counts.items(), key=lambda x: x[1])
            st.metric("Mest Nævnte Ord", top_word[0].capitalize())
            st.metric("Nævnt", f"{top_word[1]} gange")
        
        # Recent feedback display
        st.subheader("📝 Seneste Feedback")
        recent_feedback = feedback_df.tail(5)
        for _, row in recent_feedback.iterrows():
            with st.expander(f"{row['Player']} - {row['Match']}"):
                st.write(row['Feedback'])
                st.caption(f"Performance Rating: {row['Performance_Rating']:.1f}")
    
    elif analysis_type == "☁️ Ordsky Analyse":
        st.subheader("☁️ Feedback Ordsky")
        
        view_type = st.radio(
            "Vælg visning:",
            ["Hele Sæsonen", "Per Kamp", "Sammenlign Perioder"],
            horizontal=True
        )
        
        if view_type == "Hele Sæsonen":
            all_feedback = ' '.join(feedback_df['Feedback'].values)
            fig = generate_wordcloud(all_feedback, "Sæson Feedback Ordsky")
            if fig:
                st.pyplot(fig)
                
                # Key themes
                st.subheader("🎯 Nøgletemaer")
                themes = {
                    "Teknisk": ['pasninger', 'afleveringer', 'første berøring', 'boldkontrol'],
                    "Fysisk": ['pres', 'løb', 'tempo', 'hurtigere', 'aggressiv'],
                    "Taktisk": ['positionering', 'placering', 'rum', 'dybde'],
                    "Mental": ['fokus', 'kommunikation', 'lyd', 'koncentration']
                }
                
                theme_counts = {}
                for theme, keywords in themes.items():
                    count = sum(all_feedback.lower().count(kw) for kw in keywords)
                    theme_counts[theme] = count
                
                fig_themes = px.bar(
                    x=list(theme_counts.keys()),
                    y=list(theme_counts.values()),
                    title="Feedback Temaer",
                    labels={'x': 'Tema', 'y': 'Antal Nævnelser'},
                    color=list(theme_counts.values()),
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_themes, use_container_width=True)
        
        elif view_type == "Per Kamp":
            selected_match = st.selectbox(
                "Vælg kamp:",
                options=sorted(feedback_df['Match'].unique())
            )
            
            match_feedback = feedback_df[feedback_df['Match'] == selected_match]
            match_text = ' '.join(match_feedback['Feedback'].values)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = generate_wordcloud(match_text, f"Feedback: {selected_match}")
                if fig:
                    st.pyplot(fig)
            
            with col2:
                st.metric("Antal Feedback", len(match_feedback))
                st.metric("Gns. Rating", f"{match_feedback['Performance_Rating'].mean():.1f}")
                
                st.markdown("##### Top Spillere")
                for player in match_feedback['Player'].head(3):
                    st.write(f"• {player}")
        
        else:  # Sammenlign Perioder
            col1, col2 = st.columns(2)
            
            with col1:
                period1_matches = st.multiselect(
                    "Vælg første periode:",
                    options=sorted(feedback_df['Match'].unique()),
                    key="feedback_period1"
                )
            
            with col2:
                period2_matches = st.multiselect(
                    "Vælg anden periode:",
                    options=sorted(feedback_df['Match'].unique()),
                    key="feedback_period2"
                )
            
            if period1_matches and period2_matches:
                period1_text = ' '.join(feedback_df[feedback_df['Match'].isin(period1_matches)]['Feedback'].values)
                period2_text = ' '.join(feedback_df[feedback_df['Match'].isin(period2_matches)]['Feedback'].values)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = generate_wordcloud(period1_text, f"Periode 1 ({len(period1_matches)} kampe)")
                    if fig1:
                        st.pyplot(fig1)
                
                with col2:
                    fig2 = generate_wordcloud(period2_text, f"Periode 2 ({len(period2_matches)} kampe)")
                    if fig2:
                        st.pyplot(fig2)
    
    elif analysis_type == "🎯 Kampspecifik Analyse":
        selected_match = st.selectbox(
            "Vælg kamp:",
            options=sorted(feedback_df['Match'].unique())
        )
        
        match_feedback = feedback_df[feedback_df['Match'] == selected_match]
        
        st.subheader(f"Feedback fra {selected_match}")
        
        # Display all feedback for the match
        for _, row in match_feedback.iterrows():
            with st.expander(f"{row['Player']} (Rating: {row['Performance_Rating']:.1f})"):
                st.write(row['Feedback'])
        
        # AI Analysis for match
        if st.button("🤖 Generer AI Analyse for Kampen", key="feedback_match_ai"):
            ai_analysis = analyze_feedback_with_ai(feedback_df, selected_match)
            if ai_analysis:
                st.markdown("### 🧠 AI Træner Analyse")
                st.markdown(ai_analysis)
    
    elif analysis_type == "👤 Spillerspecifik Feedback":
        selected_player = st.selectbox(
            "Vælg spiller:",
            options=sorted(feedback_df['Player'].unique())
        )
        
        player_feedback = feedback_df[feedback_df['Player'] == selected_player]
        
        st.subheader(f"Feedback fra {selected_player}")
        
        # Timeline of feedback
        for _, row in player_feedback.iterrows():
            with st.expander(f"{row['Match']} (Rating: {row['Performance_Rating']:.1f})"):
                st.write(row['Feedback'])
        
        # Word cloud for player
        if len(player_feedback) > 2:
            player_text = ' '.join(player_feedback['Feedback'].values)
            fig = generate_wordcloud(player_text, f"{selected_player}'s Feedback Temaer")
            if fig:
                st.pyplot(fig)
    
    else:  # AI Feedback Analyse
        st.subheader("🤖 AI-Drevet Feedback Analyse")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            analysis_scope = st.radio(
                "Analyseomfang:",
                ["Hele Holdet", "Seneste 5 Kampe", "Specifik Kamp"],
                horizontal=True
            )
        
        with col2:
            if st.button("🚀 Generer AI Analyse", type="primary", key="feedback_ai_generate"):
                with st.spinner("Analyserer feedback med AI..."):
                    if analysis_scope == "Hele Holdet":
                        ai_analysis = analyze_feedback_with_ai(feedback_df)
                    elif analysis_scope == "Seneste 5 Kampe":
                        recent_matches = feedback_df['Match'].unique()[-5:]
                        recent_feedback = feedback_df[feedback_df['Match'].isin(recent_matches)]
                        ai_analysis = analyze_feedback_with_ai(recent_feedback)
                    else:
                        selected_match = st.selectbox(
                            "Vælg kamp:",
                            options=sorted(feedback_df['Match'].unique())
                        )
                        ai_analysis = analyze_feedback_with_ai(feedback_df, selected_match)
                    
                    if ai_analysis:
                        st.markdown("### 🧠 AI Træner Indsigter")
                        st.markdown(ai_analysis)
                        
                        # Save to cache
                        st.session_state.feedback_analysis_cache[analysis_scope] = ai_analysis
        
        # AI Recommendations for next matches
        st.divider()
        st.subheader("🎯 AI Anbefalinger for Næste Kampe")
        
        if st.button("📋 Generer Forbedringsprogram", key="feedback_improvement_program"):
            # Generate comprehensive improvement program
            improvement_analysis = get_ai_recommendations(
                df_filtered, df_full, 
                "Forbedringsprogram baseret på feedback",
                analysis_type="feedback"
            )
            
            if improvement_analysis:
                st.markdown("### 📈 Forbedringsprogram")
                st.markdown(improvement_analysis)
                
                # Create downloadable action plan
                st.download_button(
                    label="📥 Download Handlingsplan",
                    data=improvement_analysis,
                    file_name=f"handlingsplan_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

def render_dashboard(df_filtered, df_full):
    """Render the enhanced main dashboard"""
    st.header("📊 Performance Dashboard", divider="rainbow")
    
    # Key metrics with enhanced styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📝 Evalueringer",
            f"{len(df_filtered)}",
            f"{len(df_filtered) - len(df_full)}" if len(df_filtered) != len(df_full) else None
        )
    
    with col2:
        avg_rating = df_filtered['Performance_Rating'].mean()
        st.metric(
            "⭐ Gns. Rating",
            f"{avg_rating:.1f}/100",
            f"{avg_rating - df_full['Performance_Rating'].mean():.1f}" 
        )
    
    with col3:
        st.metric(
            "🎯 Pasningspræcision",
            f"{df_filtered['Passing_Accuracy'].mean():.1f}%",
            f"{df_filtered['Passing_Accuracy'].mean() - df_full['Passing_Accuracy'].mean():.1f}%"
        )
    
    with col4:
        st.metric(
            "⚡ Presintensitet",
            f"{df_filtered['Pressing_Intensity'].mean():.1f}%",
            help="% af erobringer på modstanderens bane"
        )
    
    with col5:
        st.metric(
            "🔥 Offensiv Output",
            f"{df_filtered['Offensive_Contribution'].mean():.1f}",
            f"{df_filtered['Offensive_Contribution'].mean() - df_full['Offensive_Contribution'].mean():.1f}"
        )
    
    st.divider()
    
    # Enhanced charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performers")
        top_players = df_filtered.groupby('Player')['Performance_Rating'].mean().nlargest(10)
        
        fig = px.bar(
            top_players.reset_index(),
            x='Performance_Rating',
            y='Player',
            orientation='h',
            title="Top 10 Spillere (Performance Rating)",
            color='Performance_Rating',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Performance Distribution")
        fig = px.histogram(
            df_filtered,
            x='Performance_Rating',
            nbins=20,
            title="Distribution af Performance Ratings",
            labels={'count': 'Antal', 'Performance_Rating': 'Rating'}
        )
        fig.add_vline(x=df_filtered['Performance_Rating'].mean(), 
                     line_dash="dash", 
                     annotation_text="Gennemsnit")
        st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced heatmap
    st.subheader("🔗 Korrelationsmatrix")
    correlation_metrics = [
        'Passing_Accuracy', 'Total_Shots', 'Offensive_Contribution',
        'Defensive_Contribution', 'Pressing_Intensity', 'Performance_Rating'
    ]
    
    valid_corr_metrics = [m for m in correlation_metrics if m in df_filtered.columns]
    
    if len(valid_corr_metrics) > 1:
        corr_matrix = df_filtered[valid_corr_metrics].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Korrelation"),
            x=valid_corr_metrics,
            y=valid_corr_metrics,
            color_continuous_scale='RdBu',
            aspect="auto",
            zmin=-1, zmax=1
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_player_analysis(df_filtered, df_full):
    """Render enhanced player analysis with match comparison and trends."""
    st.header("👤 Detaljeret Spilleranalyse", divider="rainbow")
    
    player_options = sorted(df_filtered['Player'].unique())
    if not player_options:
        st.warning("Ingen spillere fundet for de valgte filtre.")
        return

    player = st.selectbox(
        "Vælg spiller:",
        options=player_options,
        format_func=lambda x: f"{x} ({len(df_filtered[df_filtered['Player']==x])} kampe)",
        key="player_analysis_select"
    )
    
    if player:
        player_df = df_filtered[df_filtered['Player'] == player].copy()
        
        # Enhanced player summary card
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px; border-radius: 10px; color: white; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2>{player}</h2>
                <p style='font-size: 24px; margin: 10px;'>
                    ⭐ Gns. Rating: {player_df['Performance_Rating'].mean():.1f}/100
                </p>
                <p>{len(player_df)} kampe analyseret</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Performance metrics vs. team average
        st.subheader("📊 Nøgletal vs. Holdgennemsnit")
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_display = [
            ("🎯 Pasningspræcision", "Passing_Accuracy", "%"),
            ("⚽ Skud per kamp", "Total_Shots", ""),
            ("🛡️ Erobringer p. kamp", "Total_Tackles", ""),
            ("📈 Involvement p. kamp", "Player_Involvement", "")
        ]
        
        for col, (label, metric, suffix) in zip([col1, col2, col3, col4], metrics_display):
            with col:
                player_value = player_df[metric].mean()
                team_avg_value = df_filtered[metric].mean()
                delta = player_value - team_avg_value
                st.metric(
                    label=label,
                    value=f"{player_value:.1f}{suffix}",
                    delta=f"{delta:+.1f}{suffix} vs. hold",
                    delta_color="normal"
                )

        st.divider()

        # Player development chart
        st.subheader("📈 Spillerudvikling Over Tid")

        if 'Timestamp' in player_df.columns and player_df['Timestamp'].notna().any():
            player_df_sorted = player_df.sort_values('Timestamp')

            if len(player_df_sorted) > 1:
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Scatter(x=player_df_sorted['Match'], y=player_df_sorted['Performance_Rating'], name='Performance Rating',
                               mode='lines+markers', line=dict(color='blue', width=3)),
                    secondary_y=False,
                )

                fig.add_trace(
                    go.Scatter(x=player_df_sorted['Match'], y=player_df_sorted['Passing_Accuracy'], name='Pasningspræcision',
                               mode='lines', line=dict(color='green', dash='dot')),
                    secondary_y=True,
                )

                fig.update_layout(
                    title_text=f"<b>{player}'s Udvikling</b>",
                    xaxis_title="Kamp",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_yaxes(title_text="<b>Performance Rating</b>", secondary_y=False, range=[0, 100])
                fig.update_yaxes(title_text="<b>Pasningspræcision (%)</b>", secondary_y=True, range=[0, 100])

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Ikke nok data (mindst 2 kampe) til at vise en udviklingsgraf.")
        else:
            st.warning("⚠️ Tidsstempel data mangler for at vise udvikling over tid.")

        # Match-by-match stats and comparison
        st.subheader("📋 Kamp-for-Kamp Analyse")

        # Compare two matches
        st.markdown("##### 🆚 Sammenlign to kampe")
        col1, col2 = st.columns(2)
        match_options = sorted(player_df['Match'].unique())

        # Set default indices for match comparison
        idx1, idx2 = 0, len(match_options) - 1
        if len(match_options) <= 1:
            idx1, idx2 = None, None

        with col1:
            match1_select = st.selectbox("Vælg første kamp", match_options, index=idx1, key=f"match1_compare_{player}")
        with col2:
            match2_select = st.selectbox("Vælg anden kamp", match_options, index=idx2, key=f"match2_compare_{player}")

        if match1_select and match2_select and match1_select != match2_select:
            match1_data = player_df[player_df['Match'] == match1_select].iloc[0]
            match2_data = player_df[player_df['Match'] == match2_select].iloc[0]

            comparison_metrics = ['Performance_Rating', 'Passing_Accuracy', 'Total_Shots', 'Total_Tackles', 'Player_Involvement', 'Offensive_Contribution', 'Defensive_Contribution']

            delta_data = []
            for metric in comparison_metrics:
                if metric in match1_data and metric in match2_data:
                    val1 = match1_data[metric]
                    val2 = match2_data[metric]
                    delta_data.append({
                        'Metrik': metric.replace('_', ' '),
                        match1_select: val1,
                        match2_select: val2,
                        'Udvikling': val2 - val1
                    })

            delta_df = pd.DataFrame(delta_data)

            # Formatting for the comparison table
            st.dataframe(delta_df.style.format({
                match1_select: '{:.1f}',
                match2_select: '{:.1f}',
                'Udvikling': '{:+.1f}'
            }).background_gradient(
                cmap='RdYlGn', subset=['Udvikling'], vmin=-abs(delta_df['Udvikling']).max(), vmax=abs(delta_df['Udvikling']).max()
            ), use_container_width=True)

        # Full data table for the player
        st.markdown("##### Alle Kampe")
        match_stats_cols = ['Match', 'Performance_Rating', 'Passing_Accuracy', 'Total_Shots', 'Total_Tackles', 'Player_Involvement', 'Feedback']
        display_cols = [col for col in match_stats_cols if col in player_df.columns]

        if 'Timestamp' in player_df.columns:
            player_df = player_df.sort_values(by='Timestamp', ascending=False)

        st.dataframe(
            player_df[display_cols].style.format({
                'Performance_Rating': '{:.1f}',
                'Passing_Accuracy': '{:.1f}%',
            }),
            use_container_width=True,
            hide_index=True
        )

def render_head_to_head(df_filtered):
    """Render enhanced head-to-head comparison"""
    st.header("⚔️ Head-to-Head Sammenligning", divider="rainbow")
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox(
            "Vælg første spiller:",
            options=sorted(df_filtered['Player'].unique()),
            key="h2h_p1"
        )
    
    with col2:
        player2 = st.selectbox(
            "Vælg anden spiller:",
            options=sorted(df_filtered['Player'].unique()),
            key="h2h_p2"
        )
    
    if player1 and player2 and player1 != player2:
        p1_df = df_filtered[df_filtered['Player'] == player1]
        p2_df = df_filtered[df_filtered['Player'] == player2]
        
        # Comparison metrics
        comparison_metrics = [
            'Performance_Rating', 'Passing_Accuracy', 'Total_Shots',
            'Offensive_Contribution', 'Defensive_Contribution',
            'Pressing_Intensity', 'Ball_Retention'
        ]
        
        valid_comparison = [m for m in comparison_metrics if m in p1_df.columns]
        
        # Create comparison dataframe
        comparison_data = pd.DataFrame({
            player1: p1_df[valid_comparison].mean(),
            player2: p2_df[valid_comparison].mean()
        })
        
        # Enhanced radar chart
        fig = go.Figure()
        
        for player, color in [(player1, 'blue'), (player2, 'red')]:
            player_data = comparison_data[player]
            max_vals = comparison_data.max(axis=1)
            normalized_data = (player_data / (max_vals + 0.01) * 100).fillna(0)
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_data.values,
                theta=[m.replace('_', ' ') for m in valid_comparison],
                fill='toself',
                name=player,
                line_color=color,
                fillcolor=f'rgba({255 if color == "red" else 0}, 0, {255 if color == "blue" else 0}, 0.2)'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Sammenligning af Spillerprofiler",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_trends(df_filtered):
    """Render enhanced trends analysis"""
    st.header("📈 Trends & Udvikling", divider="rainbow")
    
    if 'Timestamp' not in df_filtered.columns or df_filtered['Timestamp'].isna().all():
        st.warning("⚠️ Ingen tidsstempel data tilgængelig for trendanalyse.")
        return
    
    # Time-based aggregation
    time_group = st.selectbox(
        "Vælg tidsperiode:",
        ["Pr. Kamp", "Ugentlig", "Månedlig"]
    )
    
    # Prepare data based on time grouping
    df_time = df_filtered.copy()
    df_time = df_time.sort_values('Timestamp')
    
    if time_group == "Ugentlig":
        df_time['Period'] = df_time['Timestamp'].dt.to_period('W')
    elif time_group == "Månedlig":
        df_time['Period'] = df_time['Timestamp'].dt.to_period('M')
    else:
        df_time['Period'] = df_time['Match']
    
    # Team performance over time
    st.subheader("📊 Hold Performance Over Tid")
    
    if time_group != "Pr. Kamp":
        team_trends = df_time.groupby('Period').agg({
            'Performance_Rating': 'mean',
            'Passing_Accuracy': 'mean',
            'Total_Shots': 'sum',
            'Total_Tackles': 'sum',
            'Player': 'nunique'
        }).reset_index()
        team_trends['Period'] = team_trends['Period'].astype(str)
    else:
        team_trends = df_time.groupby('Match').agg({
            'Performance_Rating': 'mean',
            'Passing_Accuracy': 'mean',
            'Total_Shots': 'sum',
            'Total_Tackles': 'sum',
            'Player': 'nunique'
        }).reset_index()
        team_trends.rename(columns={'Match': 'Period'}, inplace=True)
    
    # Enhanced multi-metric trend chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Performance Rating", "Pasningspræcision", 
                       "Offensiv Output", "Defensiv Indsats"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Performance Rating with trend line
    fig.add_trace(
        go.Scatter(x=team_trends['Period'], y=team_trends['Performance_Rating'],
                  mode='lines+markers', name='Performance Rating',
                  line=dict(color='blue', width=2),
                  marker=dict(size=8)),
        row=1, col=1
    )
    
    # Passing Accuracy
    fig.add_trace(
        go.Scatter(x=team_trends['Period'], y=team_trends['Passing_Accuracy'],
                  mode='lines+markers', name='Pasningspræcision',
                  line=dict(color='green', width=2),
                  marker=dict(size=8)),
        row=1, col=2
    )
    
    # Offensive Output
    fig.add_trace(
        go.Bar(x=team_trends['Period'], y=team_trends['Total_Shots'],
               name='Total Skud', marker_color='orange'),
        row=2, col=1
    )
    
    # Defensive Work
    fig.add_trace(
        go.Bar(x=team_trends['Period'], y=team_trends['Total_Tackles'],
               name='Total Erobringer', marker_color='red'),
        row=2, col=2
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_tactical_analysis(df_filtered, df_full):
    """Render enhanced tactical analysis"""
    st.header("🎯 Taktisk Analyse", divider="rainbow")
    
    # Tactical overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚽ Taktisk Profil")
        
        # Calculate tactical metrics
        tactical_metrics = {
            'Possession Style': df_filtered['Ball_Retention'].mean(),
            'Pressing Intensity': df_filtered['Pressing_Intensity'].mean(),
            'Offensive Focus': df_filtered['Offensive_Contribution'].mean(),
            'Defensive Solidarity': df_filtered['Defensive_Contribution'].mean(),
            'Direct Play': df_filtered['Shots_per_Pass'].mean() * 100,
            'Work Rate': df_filtered['Player_Involvement'].mean()
        }
        
        # Enhanced tactical profile chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(tactical_metrics.values()),
                y=list(tactical_metrics.keys()),
                orientation='h',
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
                text=[f'{v:.1f}' for v in tactical_metrics.values()],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Holdets Taktiske DNA",
            xaxis_title="Score",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Taktiske Nøgletal")
        
        st.metric("⚡ Presintensitet", f"{tactical_metrics['Pressing Intensity']:.1f}%")
        st.metric("🎯 Boldbesiddelse", f"{tactical_metrics['Possession Style']:.1f}")
        st.metric("⚽ Direkte Spil", f"{tactical_metrics['Direct Play']:.1f}%")

def render_ai_coach(df_filtered, df_full):
    """Render enhanced AI coaching interface with next match recommendations"""
    st.header("🧠 AI Elite Coach", divider="rainbow")
    
    if not st.session_state.api_key_configured:
        st.error("🔑 AI Coach kræver en Gemini API nøgle. Konfigurer den i sidemenuen.")
        return
    
    # Analysis type selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Vælg Analysefokus")
        
        analysis_scope = st.radio(
            "Hvad skal analyseres?",
            ["Hele Holdet", "Specifik Kamp", "Individuel Spiller", 
             "Sammenlign Perioder", "Næste Kamp Forberedelse"],
            horizontal=True
        )
    
    with col2:
        st.subheader("📋 Analysetype")
        
        analysis_type = st.selectbox(
            "Fokusområde:",
            ["Generel Analyse", "Taktisk Dybdeanalyse", 
             "Individuel Udvikling", "Fysisk & Mental", "Feedback Baseret"]
        )
    
    st.divider()
    
    # Scope-specific inputs
    if analysis_scope == "Hele Holdet":
        selection_df = df_filtered
        selection_name = "Hele Holdet"
        
    elif analysis_scope == "Specifik Kamp":
        selected_match = st.selectbox(
            "Vælg kamp:",
            options=sorted(df_filtered['Match'].unique())
        )
        selection_df = df_filtered[df_filtered['Match'] == selected_match]
        selection_name = f"Kamp: {selected_match}"
        
    elif analysis_scope == "Individuel Spiller":
        selected_player = st.selectbox(
            "Vælg spiller:",
            options=sorted(df_filtered['Player'].unique())
        )
        selection_df = df_filtered[df_filtered['Player'] == selected_player]
        selection_name = f"Spiller: {selected_player}"
        
    elif analysis_scope == "Sammenlign Perioder":
        col1, col2 = st.columns(2)
        with col1:
            period1_matches = st.multiselect(
                "Vælg første periode (kampe):",
                options=sorted(df_filtered['Match'].unique()),
                key="period1"
            )
        with col2:
            period2_matches = st.multiselect(
                "Vælg anden periode (kampe):",
                options=sorted(df_filtered['Match'].unique()),
                key="period2"
            )
        
        if period1_matches and period2_matches:
            selection_df = df_filtered[df_filtered['Match'].isin(period1_matches + period2_matches)]
            selection_name = f"Sammenligning: {len(period1_matches)} vs {len(period2_matches)} kampe"
        else:
            st.warning("Vælg kampe for begge perioder")
            return
    
    else:  # Næste Kamp Forberedelse
        selection_df = df_filtered
        selection_name = "Forberedelse til Næste Kamp"
        
        # Special UI for next match preparation
        st.info("🎯 AI vil analysere seneste præstationer og give konkrete anbefalinger for næste kamp")
    
    # Generate analysis button
    if st.button("🚀 Generer AI Analyse", type="primary", key="ai_coach_generate"):
        if not selection_df.empty:
            # Map analysis type
            type_mapping = {
                "Generel Analyse": "general",
                "Taktisk Dybdeanalyse": "tactical",
                "Individuel Udvikling": "individual",
                "Fysisk & Mental": "physical_mental",
                "Feedback Baseret": "feedback"
            }
            
            ai_response = get_ai_recommendations(
                selection_df, 
                df_full, 
                selection_name,
                analysis_type=type_mapping.get(analysis_type, "general")
            )
            
            if ai_response:
                st.markdown("---")
                st.markdown(ai_response)
                
                # Save to history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'selection': selection_name,
                    'analysis': ai_response
                })
                
                # Download button for the analysis
                st.download_button(
                    label="📥 Download Analyse",
                    data=ai_response,
                    file_name=f"ai_analyse_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Ingen data at analysere for den valgte selektion")

def render_reports(df_filtered):
    """Enhanced reports with all selection options"""
    st.header("📝 Rapporter & Eksport", divider="rainbow")
    
    report_type = st.selectbox(
        "Vælg rapporttype:",
        ["Holdrapport", "Individuelle Spillerrapporter", "Kamprapport", "Udviklingsrapport"]
    )
    
    st.divider()
    
    if report_type == "Holdrapport":
        st.subheader("📊 Holdrapport")
        
        # Team summary statistics
        team_stats = {
            'Antal Spillere': df_filtered['Player'].nunique(),
            'Antal Kampe': df_filtered['Match'].nunique(),
            'Gns. Performance Rating': f"{df_filtered['Performance_Rating'].mean():.1f}",
            'Gns. Pasningspræcision': f"{df_filtered['Passing_Accuracy'].mean():.1f}%",
            'Total Skud': int(df_filtered['Total_Shots'].sum()),
            'Total Erobringer': int(df_filtered['Total_Tackles'].sum()),
            'Gns. Presintensitet': f"{df_filtered['Pressing_Intensity'].mean():.1f}%"
        }
        
        # Display in enhanced format
        col1, col2 = st.columns(2)
        
        with col1:
            for key, value in list(team_stats.items())[:4]:
                st.info(f"**{key}:** {value}")
        
        with col2:
            for key, value in list(team_stats.items())[4:]:
                st.info(f"**{key}:** {value}")
        
        # Top performers
        st.subheader("🏆 Top Performers")
        
        top_players = df_filtered.groupby('Player').agg({
            'Performance_Rating': 'mean',
            'Match': 'count'
        }).sort_values('Performance_Rating', ascending=False).head(5)
        
        top_players.columns = ['Gns. Rating', 'Kampe Spillet']
        st.dataframe(top_players, use_container_width=True)
        
        # Generate downloadable report
        if st.button("📥 Download Holdrapport som CSV", key="download_team_report_csv"):
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"holdrapport_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    elif report_type == "Individuelle Spillerrapporter":
        st.subheader("👤 Individuelle Rapporter")
        
        selected_players = st.multiselect(
            "Vælg spillere:",
            options=sorted(df_filtered['Player'].unique()),
            default=sorted(df_filtered['Player'].unique())[:3] if len(df_filtered['Player'].unique()) >= 3 else sorted(df_filtered['Player'].unique())
        )
        
        if selected_players:
            # Create combined report
            all_player_reports = []
            
            for player in selected_players:
                player_data = df_filtered[df_filtered['Player'] == player]
                
                player_report = {
                    'Spiller': player,
                    'Kampe': len(player_data),
                    'Gns. Rating': round(float(player_data['Performance_Rating'].mean()), 2),
                    'Pasningspræcision': round(float(player_data['Passing_Accuracy'].mean()), 2),
                    'Total Skud': int(player_data['Total_Shots'].sum()),
                    'Total Erobringer': int(player_data['Total_Tackles'].sum()),
                    'Gns. Involvement': round(float(player_data['Player_Involvement'].mean()), 2)
                }
                all_player_reports.append(player_report)
                
                with st.expander(f"📋 {player} - Rapport"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Kampe", player_report['Kampe'])
                        st.metric("Gns. Rating", f"{player_report['Gns. Rating']:.1f}")
                        st.metric("Total Skud", int(player_report['Total Skud']))
                    
                    with col2:
                        st.metric("Pasningspræcision", f"{player_report['Pasningspræcision']:.1f}%")
                        st.metric("Erobringer", int(player_report['Total Erobringer']))
                        st.metric("Gns. Involvement", f"{player_report['Gns. Involvement']:.1f}")
                    
                    # Recent feedback
                    if 'Feedback' in player_data.columns:
                        latest_feedback = player_data[['Match', 'Feedback']].dropna().tail(1)
                        if not latest_feedback.empty:
                            st.info(f"**Seneste feedback:** {latest_feedback.iloc[0]['Feedback']}")
            
            # Download all reports
            if st.button("📥 Download Alle Spillerrapporter", key="download_player_reports_json"):
                import json
                reports_json = json.dumps(all_player_reports, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=reports_json,
                    file_name=f"spillerrapporter_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
    
    elif report_type == "Kamprapport":
        st.subheader("🏟️ Kamprapport")
        
        selected_match = st.selectbox(
            "Vælg kamp:",
            options=sorted(df_filtered['Match'].unique())
        )
        
        if selected_match:
            match_data = df_filtered[df_filtered['Match'] == selected_match]
            
            # Match summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Spillere", match_data['Player'].nunique())
                st.metric("Gns. Rating", f"{match_data['Performance_Rating'].mean():.1f}")
            
            with col2:
                st.metric("Total Skud", int(match_data['Total_Shots'].sum()))
                st.metric("Pasningspræcision", f"{match_data['Passing_Accuracy'].mean():.1f}%")
            
            with col3:
                st.metric("Total Erobringer", int(match_data['Total_Tackles'].sum()))
                st.metric("Presintensitet", f"{match_data['Pressing_Intensity'].mean():.1f}%")
            
            # Player performances in match
            st.subheader("Spillerpræstationer")
            
            match_players = match_data[['Player', 'Performance_Rating', 'Total_Shots', 
                                       'Total_Tackles', 'Passing_Accuracy']].copy()
            match_players.columns = ['Spiller', 'Rating', 'Skud', 'Erobringer', 'Pasning %']
            
            st.dataframe(
                match_players.sort_values('Rating', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            # Download match report
            if st.button("📥 Download Kamprapport", key="download_match_report_csv"):
                csv = match_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"kamprapport_{selected_match.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    else:  # Udviklingsrapport
        st.subheader("📈 Udviklingsrapport")
        
        if 'Timestamp' in df_filtered.columns and df_filtered['Timestamp'].notna().any():
            # Calculate development metrics
            df_sorted = df_filtered.sort_values('Timestamp')
            
            # Split into first and last third
            total_matches = df_sorted['Match'].nunique()
            split_point = max(1, total_matches // 3)
            
            early_matches = df_sorted['Match'].unique()[:split_point]
            recent_matches = df_sorted['Match'].unique()[-split_point:]
            
            early_df = df_sorted[df_sorted['Match'].isin(early_matches)]
            recent_df = df_sorted[df_sorted['Match'].isin(recent_matches)]
            
            # Compare metrics
            comparison = pd.DataFrame({
                'Start Periode': [
                    early_df['Performance_Rating'].mean(),
                    early_df['Passing_Accuracy'].mean(),
                    early_df['Total_Shots'].mean(),
                    early_df['Total_Tackles'].mean(),
                    early_df['Pressing_Intensity'].mean()
                ],
                'Seneste Periode': [
                    recent_df['Performance_Rating'].mean(),
                    recent_df['Passing_Accuracy'].mean(),
                    recent_df['Total_Shots'].mean(),
                    recent_df['Total_Tackles'].mean(),
                    recent_df['Pressing_Intensity'].mean()
                ]
            }, index=['Performance Rating', 'Pasningspræcision %', 'Gns. Skud', 
                     'Gns. Erobringer', 'Presintensitet %'])
            
            comparison['Udvikling %'] = ((comparison['Seneste Periode'] - comparison['Start Periode']) / 
                                        (comparison['Start Periode'] + 0.01) * 100).round(1)
            
            # Style the dataframe
            def color_change(val):
                if val > 5:
                    return 'background-color: lightgreen'
                elif val < -5:
                    return 'background-color: lightcoral'
                return ''
            
            styled_comparison = comparison.style.applymap(color_change, subset=['Udvikling %']).format("{:.1f}")
            st.dataframe(styled_comparison, use_container_width=True)
            
            # Download development report
            if st.button("📥 Download Udviklingsrapport", key="download_dev_report_csv"):
                csv = comparison.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"udviklingsrapport_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Ingen tidsstempel data tilgængelig for udviklingsrapport")

# Run the main application
if __name__ == "__main__":
    main()