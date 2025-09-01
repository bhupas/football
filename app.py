import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Fodboldlinjen Coaching Assistant",
    page_icon="🧠",
    layout="wide",
)

# --- HELPER & CLEANING FUNCTIONS ---
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def find_header_row(file_or_df, expected_cols, sample_rows=15):
    df_sample = None
    if isinstance(file_or_df, pd.DataFrame):
        df_sample = file_or_df.head(sample_rows)
    else:
        try:
            df_sample = pd.read_csv(file_or_df, header=None, nrows=sample_rows, on_bad_lines='skip')
            file_or_df.seek(0)
        except Exception: return None
    cleaned_expected_cols = [clean_text(col) for col in expected_cols]
    best_match_count, header_row_index = 0, None
    for i, row in df_sample.iterrows():
        cleaned_row_values = [clean_text(cell) for cell in row.values]
        match_count = sum(1 for expected_col in cleaned_expected_cols if expected_col in cleaned_row_values)
        if match_count > best_match_count:
            best_match_count, header_row_index = match_count, i
    if best_match_count > 3: return header_row_index
    return None

# --- DATA LOADING & ADVANCED FEATURE ENGINEERING ---
@st.cache_data
def load_and_prepare_data(uploaded_files):
    all_dfs = []
    column_mapping = {
        'Tidsstempel': 'Timestamp', 'Kamp - Hvilket hold spillede du for': 'Team',
        'Modstanderen (hvem spillede du mod)': 'Opponent', 'Navn (fulde navn)': 'Player',
        '#Succesfulde pasninger /indlæg': 'Successful_Passes',
        '#Total pasninger/indlæg (Succesfulde + ikke succesfulde)': 'Total_Passes',
        '#Total afslutninger': 'Total_Shots', '#Succesfulde erobringer på EGEN bane': 'Tackles_Own_Half',
        '#Succesfulde erobringer på DERES bane': 'Tackles_Opponent_Half',
        '#Total succesfulde erobringer (EGEN + DERES bane)': 'Total_Tackles',
        'Hvad vil du gøre bedre i næste kamp ?': 'Feedback'
    }
    expected_cols = list(column_mapping.keys())
    final_numeric_cols = ['Successful_Passes', 'Total_Passes', 'Total_Shots', 'Tackles_Own_Half', 'Tackles_Opponent_Half', 'Total_Tackles']
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                excel_sheets = pd.read_excel(uploaded_file, sheet_name=None, header=None, engine='openpyxl')
                for _, sheet_df in excel_sheets.items():
                    header_row = find_header_row(sheet_df, expected_cols)
                    if header_row is not None:
                        sheet_df.columns = sheet_df.iloc[header_row]
                        sheet_df = sheet_df.drop(sheet_df.index[:header_row + 1]).reset_index(drop=True)
                        all_dfs.append(sheet_df)
            elif uploaded_file.name.endswith('.csv'):
                string_io = io.StringIO(uploaded_file.getvalue().decode('utf-8'))
                header_row = find_header_row(string_io, expected_cols)
                if header_row is not None:
                    string_io.seek(0)
                    df = pd.read_csv(string_io, on_bad_lines='skip', header=header_row)
                    all_dfs.append(df)
        except Exception as e:
            st.error(f"Could not process file {uploaded_file.name}. Error: {e}")
            continue
    if not all_dfs: return None, "No valid data could be loaded."

    combined_df = pd.concat(all_dfs, ignore_index=True).rename(columns=column_mapping)
    for col in final_numeric_cols:
        if col not in combined_df.columns: combined_df[col] = np.nan
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    if 'Timestamp' in combined_df.columns:
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], errors='coerce')
    combined_df.dropna(subset=['Player'], inplace=True)
    combined_df['Player'] = combined_df['Player'].astype(str).str.strip().str.title()
    combined_df['Opponent'] = combined_df['Opponent'].astype(str).str.strip().str.title()
    
    warnings = []
    mask = combined_df['Successful_Passes'] > combined_df['Total_Passes']
    combined_df.loc[mask, 'Total_Passes'] = combined_df.loc[mask, 'Successful_Passes']
    if mask.sum() > 0: warnings.append(f"Found and auto-corrected {mask.sum()} illogical pass entries.")

    passing_accuracy_array = np.where(combined_df['Total_Passes'] > 0, (combined_df['Successful_Passes'] / combined_df['Total_Passes']) * 100, 0)
    combined_df['Passing_Accuracy'] = passing_accuracy_array
    combined_df['Passing_Accuracy'] = combined_df['Passing_Accuracy'].fillna(0)
    
    defensive_ratio_array = np.where(combined_df['Total_Tackles'] > 0, (combined_df['Tackles_Opponent_Half'] / combined_df['Total_Tackles']) * 100, 0)
    combined_df['Defensive_Action_Ratio'] = defensive_ratio_array
    combined_df['Defensive_Action_Ratio'] = combined_df['Defensive_Action_Ratio'].fillna(0)

    passing_style_array = np.where(combined_df['Successful_Passes'] > 0, (combined_df['Total_Passes'] / combined_df['Successful_Passes']), 1)
    combined_df['Passing_Style_Index'] = passing_style_array
    combined_df['Passing_Style_Index'] = combined_df['Passing_Style_Index'].fillna(1)
    
    combined_df['Offensive_Contribution'] = combined_df['Total_Shots'].fillna(0) * 1.2 + combined_df['Successful_Passes'].fillna(0)
    combined_df['Defensive_Contribution'] = combined_df['Tackles_Opponent_Half'].fillna(0) * 1.5 + combined_df['Tackles_Own_Half'].fillna(0)

    if 'Timestamp' in combined_df.columns:
        combined_df = combined_df.sort_values(by='Timestamp').reset_index(drop=True)
    return combined_df, warnings

# --- AI & MACHINE LEARNING MODULE ---
def run_ai_analysis(df):
    st.header("🤖 AI-Powered Insights", divider="rainbow")
    ai_col1, ai_col2 = st.columns(2)
    with ai_col1:
        st.subheader("Player Style Clustering")
        features = ['Passing_Accuracy', 'Total_Shots', 'Defensive_Contribution', 'Offensive_Contribution', 'Passing_Style_Index', 'Defensive_Action_Ratio']
        df_cluster = df.dropna(subset=features).copy()
        if len(df_cluster) < 4: st.warning("Not enough data to perform player clustering."); return
        X_scaled = StandardScaler().fit_transform(df_cluster[features])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(X_scaled)
        df_cluster['Cluster'] = kmeans.labels_
        cluster_fig = px.scatter(df_cluster, x='Offensive_Contribution', y='Defensive_Contribution', color='Cluster', hover_name='Player', title="Player Clusters", labels={'Offensive_Contribution': 'Offensive Score', 'Defensive_Contribution': 'Defensive Score'})
        st.plotly_chart(cluster_fig, use_container_width=True)
        for i in range(3):
            cluster_players = df_cluster[df_cluster['Cluster'] == i]['Player'].unique()
            cluster_stats = df_cluster[df_cluster['Cluster'] == i][features].mean()
            desc = ""
            if cluster_stats['Defensive_Contribution'] > df_cluster['Defensive_Contribution'].mean() * 1.1: desc += "High defensive work rate. "
            if cluster_stats['Offensive_Contribution'] > df_cluster['Offensive_Contribution'].mean() * 1.1: desc += "Key offensive contributor. "
            if cluster_stats['Defensive_Action_Ratio'] > 60: desc += "Proactive in pressing high. "
            if cluster_stats['Passing_Style_Index'] > 1.2: desc += "Attempts risky, progressive passes. "
            else: desc += "Secure in possession. "
            with st.expander(f"**Cluster {i}** ({len(cluster_players)} players)"):
                st.markdown(f"**Style Profile:** *{desc}*")
                st.write(f"**Players:** {', '.join(cluster_players)}")
    with ai_col2:
        st.subheader("Common Feedback Themes (NLP)")
        feedback_text = ' '.join(df['Feedback'].dropna().astype(str))
        if not feedback_text.strip(): st.warning("No feedback data available."); return
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=['og', 'jeg', 'en', 'til', 'er', 'det', 'vil', 'i', 'på', 'mere', 'at', 'have', 'være', 'mig', 'min', 'mit', 'den']).fit(df['Feedback'].dropna())
        word_freq = pd.DataFrame(vectorizer.transform(df['Feedback'].dropna()).toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False).head(10)
        st.markdown("**Top Keywords & Phrases:**")
        st.dataframe(word_freq)

# --- MAIN APP LAYOUT ---
st.title("🧠 Fodboldlinjen Coaching Assistant")
st.sidebar.header("Upload Data")
uploaded_files = st.sidebar.file_uploader("Vælg dine CSV/Excel-filer", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    df, warnings = load_and_prepare_data(uploaded_files)
    if df is None or df.empty: st.error(warnings); st.stop()
    if warnings:
        for warning in warnings: st.warning(warning)

    st.sidebar.header("Filtre")
    all_players = sorted(df['Player'].unique())
    all_opponents = sorted(df['Opponent'].dropna().unique())
    selected_players = st.sidebar.multiselect("Vælg Spiller(e)", options=all_players, default=all_players)
    selected_opponents = st.sidebar.multiselect("Vælg Modstander(e)", options=all_opponents, default=all_opponents)
    
    df_filtered = df[df['Player'].isin(selected_players) & df['Opponent'].isin(selected_opponents)]
    if df_filtered.empty: st.warning("Ingen data fundet for de valgte filtre."); st.stop()

    # **NEW**: Added "Sammenlign Spillere" tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Team Oversigt", "🎯 Spiller Udvikling", "👥 Sammenlign Spillere", "⚔️ Taktisk Analyse", "🤖 AI Analyse"])
    
    with tab1:
        st.header("Team Performance Oversigt", divider="rainbow")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Antal Evalueringer", f"{df_filtered.shape[0]}")
        col2.metric("Gns. Pasningspræcision", f"{df_filtered['Passing_Accuracy'].mean():.1f}%")
        col3.metric("Gns. Proaktivt Forsvar", f"{df_filtered['Defensive_Action_Ratio'].mean():.1f}%", help="Procentdel af erobringer på modstanderens banehalvdel.")
        col4.metric("Gns. Pasningsrisiko", f"{df_filtered['Passing_Style_Index'].mean():.2f}", help="Højere værdi = mere risikobetonede afleveringer.")
        
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.subheader("Top Spillere (Offensivt Bidrag)")
            summary = df_filtered.groupby('Player')['Offensive_Contribution'].sum().nlargest(10).sort_values()
            fig = px.bar(summary, x=summary.values, y=summary.index, orientation='h', text=summary.values, title="Offensivt Bidrag")
            st.plotly_chart(fig, use_container_width=True)
        with viz_col2:
            st.subheader("Top Spillere (Defensivt Bidrag)")
            summary = df_filtered.groupby('Player')['Defensive_Contribution'].sum().nlargest(10).sort_values()
            fig = px.bar(summary, x=summary.values, y=summary.index, orientation='h', text=summary.values, title="Defensivt Bidrag")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Følg en spillers udvikling over tid", divider="rainbow")
        player_to_track = st.selectbox("Vælg en spiller for at se udvikling", options=all_players)
        if player_to_track:
            player_df = df[df['Player'] == player_to_track].copy()
            if 'Timestamp' in player_df.columns and player_df['Timestamp'].notna().any():
                player_df['Game'] = player_df['Opponent'] + ' (' + player_df['Timestamp'].dt.strftime('%d-%m-%Y') + ')'
                if len(player_df) > 1:
                    st.subheader(f"Performance Trajectory for {player_to_track}")
                    fig_dev = go.Figure()
                    fig_dev.add_trace(go.Scatter(x=player_df['Game'], y=player_df['Passing_Accuracy'], mode='lines+markers', name='Pasningspræcision (%)'))
                    fig_dev.add_trace(go.Scatter(x=player_df['Game'], y=player_df['Offensive_Contribution'], mode='lines+markers', name='Offensivt Bidrag'))
                    fig_dev.add_trace(go.Scatter(x=player_df['Game'], y=player_df['Defensive_Contribution'], mode='lines+markers', name='Defensivt Bidrag'))
                    st.plotly_chart(fig_dev, use_container_width=True)
                else:
                    st.info(f"Ikke nok data (mindst 2 kampe) til at vise en udvikling for {player_to_track}.")
            else:
                st.warning("Timestamp data mangler, kan ikke vise udvikling over tid.")

    # **NEW**: Player Comparison Tab
    with tab3:
        st.header("Sammenlign Spiller Performance", divider="rainbow")
        players_to_compare = st.multiselect(
            "Vælg 2 til 5 spillere at sammenligne",
            options=all_players,
            default=all_players[:2] if len(all_players) > 1 else [],
            max_selections=5
        )

        if len(players_to_compare) >= 2:
            df_compare = df_filtered[df_filtered['Player'].isin(players_to_compare)]
            
            # Calculate average stats per game for a fair comparison
            comparison_metrics = ['Passing_Accuracy', 'Offensive_Contribution', 'Defensive_Contribution', 'Defensive_Action_Ratio', 'Passing_Style_Index']
            player_avg_stats = df_compare.groupby('Player')[comparison_metrics].mean().reset_index()

            st.subheader("Stilistisk Profil (Radar Chart)")
            st.markdown("Viser en oversigt over spillernes stil. Hvem er den offensive kraft? Hvem er den stabile forsvarsspiller?")
            
            fig_radar = go.Figure()
            categories = ['Pasningspræcision', 'Offensivt Bidrag', 'Defensivt Bidrag', 'Proaktivt Forsvar (%)', 'Pasningsrisiko']
            
            for _, row in player_avg_stats.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=row[comparison_metrics].values,
                    theta=categories,
                    fill='toself',
                    name=row['Player']
                ))
            st.plotly_chart(fig_radar.update_layout(title="Gennemsnitlig Performance Radar"), use_container_width=True)
            
            st.subheader("Direkte Sammenligning (Bar Chart)")
            metric_to_compare = st.selectbox("Vælg en specifik måling at sammenligne", options=comparison_metrics)
            
            if metric_to_compare:
                fig_bar = px.bar(
                    player_avg_stats.sort_values(by=metric_to_compare, ascending=False),
                    x='Player',
                    y=metric_to_compare,
                    color='Player',
                    text_auto='.2s',
                    title=f"Sammenligning af: {metric_to_compare}"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.info("Vælg venligst mindst to spillere for at starte en sammenligning.")

    with tab4:
        st.header("Taktisk Analyse: Sammenlign Performance mod Modstandere", divider="rainbow")
        if len(all_opponents) > 1:
            col1, col2 = st.columns(2)
            with col1:
                opp1 = st.selectbox("Vælg første modstander", options=all_opponents, index=0, key="opp1")
            with col2:
                opp2 = st.selectbox("Vælg anden modstander", options=all_opponents, index=1, key="opp2")
            
            if opp1 and opp2 and opp1 != opp2:
                stats1 = df[df['Opponent'] == opp1][['Passing_Accuracy', 'Defensive_Action_Ratio', 'Total_Shots']].mean().rename(opp1)
                stats2 = df[df['Opponent'] == opp2][['Passing_Accuracy', 'Defensive_Action_Ratio', 'Total_Shots']].mean().rename(opp2)
                comparison_df = pd.concat([stats1, stats2], axis=1)
                st.subheader(f"Team Gennemsnit: {opp1} vs {opp2}")
                st.dataframe(comparison_df.style.format("{:.1f}"))
                fig_comp = go.Figure(data=[go.Bar(name=opp1, x=comparison_df.index, y=comparison_df[opp1]), go.Bar(name=opp2, x=comparison_df.index, y=comparison_df[opp2])])
                fig_comp.update_layout(barmode='group', title="Sammenligning af Team Performance")
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("Vælg venligst to forskellige modstandere for at sammenligne.")
        else:
            st.info("Upload data fra mindst to forskellige modstandere for at bruge denne funktion.")

    with tab5:
        run_ai_analysis(df_filtered)

    with st.expander("Vis Rådata for Valgte Filtre"):
        st.dataframe(df_filtered)
else:
    st.info("👆 Upload dine datafiler for at starte Coaching Assistant.")