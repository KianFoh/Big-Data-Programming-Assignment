import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Mortality Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stSelectbox > label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    data_path = "/Users/limweiteik/Desktop/Big data/Big-Data-Programming-Assignment/clean_data.csv"
    df = pd.read_csv(data_path)
    return df

@st.cache_data
def get_top_causes(df, start_year, end_year, n_causes=5):
    """Get top N causes of death for the specified year range"""
    filtered_df = df[
        (df['Indicator Name'] != 'All Causes') &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year)
    ]
    
    cause_totals = filtered_df.groupby('Indicator Name')['Number'].sum().sort_values(ascending=False)
    return cause_totals.head(n_causes).index.tolist()

@st.cache_data
def prepare_clustering_data(df, start_year, end_year, top_causes):
    """Prepare data for clustering analysis"""
    clustering_data = df[
        (df['Indicator Name'].isin(top_causes)) & 
        (df['Age Category'] != 'All Ages') &
        (df['Sex'] == 'All') &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year)
    ].copy()
    
    # Create feature matrix
    feature_matrix = clustering_data.groupby(['Age Category', 'Indicator Name']).agg({
        'Number': 'mean',
        'Death Rate': 'mean',
        'Percent of All Causes': 'mean'
    }).reset_index()
    
    # Pivot to create matrix
    death_rate_matrix = feature_matrix.pivot(
        index='Age Category', 
        columns='Indicator Name', 
        values='Death Rate'
    ).fillna(0)
    
    return death_rate_matrix, clustering_data

def perform_clustering(death_rate_matrix, n_clusters):
    """Perform hierarchical clustering"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(death_rate_matrix)
    
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = agg_clustering.fit_predict(scaled_features)
    
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    
    # Add cluster labels to dataframe
    clustered_df = death_rate_matrix.copy()
    clustered_df['Cluster'] = cluster_labels
    clustered_df['Risk_Score'] = death_rate_matrix.sum(axis=1)
    
    return clustered_df, silhouette_avg, scaled_features

def main():
    # Main header
    st.markdown('<h1 class="main-header">Mortality Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar controls
    st.sidebar.header("📋 Dashboard Controls")
    
    # Year range selection
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    
    st.sidebar.subheader("📅 Select Year Range")
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    start_year, end_year = year_range
    
    # Number of causes to analyze
    n_causes = st.sidebar.selectbox(
        "Number of Top Causes to Analyze",
        options=[3, 5, 7, 10],
        index=1
    )
    
    # Number of clusters
    n_clusters = st.sidebar.selectbox(
        "Number of Clusters",
        options=[2, 3, 4, 5, 6],
        index=2
    )
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Focus",
        options=["Overview", "Clustering Analysis", "Temporal Trends", "Demographic Analysis"]
    )
    
    # Get data for selected parameters
    top_causes = get_top_causes(df, start_year, end_year, n_causes)
    
    # Display selected parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Selected Parameters:**")
    st.sidebar.write(f"📅 Years: {start_year} - {end_year}")
    st.sidebar.write(f"🔍 Top Causes: {n_causes}")
    st.sidebar.write(f"🎯 Clusters: {n_clusters}")
    
    # Main content area
    if analysis_type == "Overview":
        show_overview(df, start_year, end_year, top_causes)
    elif analysis_type == "Clustering Analysis":
        show_clustering_analysis(df, start_year, end_year, top_causes, n_clusters)
    elif analysis_type == "Temporal Trends":
        show_temporal_trends(df, start_year, end_year, top_causes)
    elif analysis_type == "Demographic Analysis":
        show_demographic_analysis(df, start_year, end_year, top_causes)

def show_overview(df, start_year, end_year, top_causes):
    """Display overview statistics and visualizations"""
    st.markdown('<h2 class="sub-header">📈 Overview Analysis</h2>', unsafe_allow_html=True)
    
    # Filter data for the selected year range
    filtered_df = df[
        (df['Year'] >= start_year) & 
        (df['Year'] <= end_year) &
        (df['Indicator Name'] != 'All Causes')
    ]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_deaths = filtered_df['Number'].sum()
        st.metric("Total Deaths", f"{total_deaths:,.0f}")
    
    with col2:
        avg_death_rate = filtered_df['Death Rate'].mean()
        st.metric("Avg Death Rate", f"{avg_death_rate:.1f}")
    
    with col3:
        unique_causes = len(filtered_df['Indicator Name'].unique())
        st.metric("Causes Analyzed", unique_causes)
    
    with col4:
        years_span = end_year - start_year + 1
        st.metric("Years Analyzed", years_span)
    
    # Top causes visualization
    st.subheader("🏆 Top Causes of Death")
    
    cause_totals = filtered_df.groupby('Indicator Name')['Number'].sum().sort_values(ascending=False)
    top_causes_data = cause_totals.head(len(top_causes))
    
    fig = px.bar(
        x=top_causes_data.values,
        y=top_causes_data.index,
        orientation='h',
        title=f"Top {len(top_causes)} Causes of Death ({start_year}-{end_year})",
        labels={'x': 'Total Deaths', 'y': 'Cause of Death'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Death rate trends over time
    st.subheader("📊 Death Rate Trends Over Time")
    
    # All causes trend
    all_causes_data = df[
        (df['Indicator Name'] == 'All Causes') &
        (df['Sex'] == 'All') &
        (df['Age Category'] == 'All Ages') &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year)
    ]
    
    fig = px.line(
        all_causes_data,
        x='Year',
        y='Death Rate',
        title='Overall Death Rate Trend (All Causes)',
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_clustering_analysis(df, start_year, end_year, top_causes, n_clusters):
    """Display clustering analysis results"""
    st.markdown('<h2 class="sub-header">🎯 Clustering Analysis</h2>', unsafe_allow_html=True)
    
    # Prepare data
    death_rate_matrix, clustering_data = prepare_clustering_data(df, start_year, end_year, top_causes)
    
    if death_rate_matrix.empty:
        st.error("No data available for the selected parameters.")
        return
    
    # Perform clustering
    clustered_df, silhouette_score_val, scaled_features = perform_clustering(death_rate_matrix, n_clusters)
    
    # Display clustering results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Silhouette Score", f"{silhouette_score_val:.3f}")
    
    with col2:
        st.metric("Age Categories", len(death_rate_matrix))
    
    # Death rate heatmap
    st.subheader("🔥 Death Rate Heatmap")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        death_rate_matrix.T, 
        annot=True, 
        fmt='.1f', 
        cmap='YlOrRd',
        cbar_kws={'label': 'Death Rate per 100,000'}
    )
    plt.title('Death Rates by Age Category and Cause of Death')
    plt.xlabel('Age Category')
    plt.ylabel('Cause of Death')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Clustering results
    st.subheader("🎯 Clustering Results")
    
    # Create cluster visualization
    fig = px.scatter(
        x=clustered_df.index,
        y=clustered_df['Risk_Score'],
        color=clustered_df['Cluster'],
        title=f'Age Categories Clustered by Risk Score (k={n_clusters})',
        labels={'x': 'Age Category', 'y': 'Total Risk Score', 'color': 'Cluster'},
        text=clustered_df.index
    )
    fig.update_traces(textposition='middle right')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster analysis table
    st.subheader("📋 Cluster Analysis")
    
    cluster_analysis = []
    for cluster_id in range(n_clusters):
        cluster_ages = clustered_df[clustered_df['Cluster'] == cluster_id]
        cluster_analysis.append({
            'Cluster': cluster_id,
            'Age Categories': ', '.join(cluster_ages.index),
            'Count': len(cluster_ages),
            'Avg Risk Score': cluster_ages['Risk_Score'].mean(),
            'Min Risk Score': cluster_ages['Risk_Score'].min(),
            'Max Risk Score': cluster_ages['Risk_Score'].max()
        })
    
    cluster_df = pd.DataFrame(cluster_analysis)
    cluster_df = cluster_df.sort_values('Avg Risk Score', ascending=False)
    
    st.dataframe(cluster_df, use_container_width=True)
    
    # Risk level recommendations
    st.subheader("💡 Risk Level Recommendations")
    
    for i, row in cluster_df.iterrows():
        risk_level = "High Risk" if row['Avg Risk Score'] > cluster_df['Avg Risk Score'].median() else "Low Risk"
        
        with st.expander(f"Cluster {row['Cluster']} - {risk_level}"):
            st.write(f"**Age Categories:** {row['Age Categories']}")
            st.write(f"**Average Risk Score:** {row['Avg Risk Score']:.1f}")
            st.write(f"**Risk Score Range:** {row['Min Risk Score']:.1f} - {row['Max Risk Score']:.1f}")
            
            if risk_level == "High Risk":
                st.write("**Recommendations:**")
                st.write("- Intensive health monitoring and screening programs")
                st.write("- Specialized medical facilities and emergency care")
                st.write("- Comprehensive insurance coverage")
            else:
                st.write("**Recommendations:**")
                st.write("- Regular health check-ups and screening")
                st.write("- Preventive care and lifestyle counseling")
                st.write("- Health education programs")

def show_temporal_trends(df, start_year, end_year, top_causes):
    """Display temporal trends analysis"""
    st.markdown('<h2 class="sub-header">📈 Temporal Trends Analysis</h2>', unsafe_allow_html=True)
    
    # Filter data
    filtered_df = df[
        (df['Indicator Name'].isin(top_causes)) &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year) &
        (df['Sex'] == 'All') &
        (df['Age Category'] == 'All Ages')
    ]
    
    # Trends over time
    st.subheader("📊 Death Count Trends")
    
    yearly_trends = filtered_df.groupby(['Year', 'Indicator Name'])['Number'].sum().reset_index()
    
    fig = px.line(
        yearly_trends,
        x='Year',
        y='Number',
        color='Indicator Name',
        title=f'Death Count Trends for Top {len(top_causes)} Causes ({start_year}-{end_year})',
        markers=True
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Death rate trends
    st.subheader("📈 Death Rate Trends")
    
    yearly_rate_trends = filtered_df.groupby(['Year', 'Indicator Name'])['Death Rate'].mean().reset_index()
    
    fig = px.line(
        yearly_rate_trends,
        x='Year',
        y='Death Rate',
        color='Indicator Name',
        title=f'Death Rate Trends for Top {len(top_causes)} Causes ({start_year}-{end_year})',
        markers=True
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Year-over-year change analysis
    st.subheader("📊 Year-over-Year Change Analysis")
    
    # Calculate year-over-year changes
    yearly_changes = []
    for cause in top_causes:
        cause_data = yearly_trends[yearly_trends['Indicator Name'] == cause].sort_values('Year')
        if len(cause_data) > 1:
            cause_data['YoY_Change'] = cause_data['Number'].pct_change() * 100
            yearly_changes.append(cause_data)
    
    if yearly_changes:
        all_changes = pd.concat(yearly_changes)
        
        fig = px.bar(
            all_changes.dropna(),
            x='Year',
            y='YoY_Change',
            color='Indicator Name',
            title='Year-over-Year Percentage Change in Death Counts',
            labels={'YoY_Change': 'Percentage Change (%)'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

def show_demographic_analysis(df, start_year, end_year, top_causes):
    """Display demographic analysis"""
    st.markdown('<h2 class="sub-header">👥 Demographic Analysis</h2>', unsafe_allow_html=True)
    
    # Filter data
    filtered_df = df[
        (df['Indicator Name'].isin(top_causes)) &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year) &
        (df['Age Category'] != 'All Ages') &
        (df['Sex'] != 'All')
    ]
    
    # Gender analysis
    st.subheader("⚥ Gender Distribution")
    
    gender_analysis = filtered_df.groupby(['Indicator Name', 'Sex'])['Number'].sum().reset_index()
    
    fig = px.bar(
        gender_analysis,
        x='Indicator Name',
        y='Number',
        color='Sex',
        title='Death Counts by Cause and Gender',
        barmode='group'
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Age group analysis
    st.subheader("👶👴 Age Group Distribution")
    
    age_analysis = filtered_df.groupby(['Indicator Name', 'Age Category'])['Number'].sum().reset_index()
    
    # Create age order
    age_order = ["Infant", "Toddler", "Child", "Teenager", "Young Adult", 
                 "Adult", "Middle Age", "Senior", "Elderly"]
    
    # Filter and order age categories
    available_ages = age_analysis['Age Category'].unique()
    ordered_ages = [age for age in age_order if age in available_ages]
    
    age_analysis['Age Category'] = pd.Categorical(
        age_analysis['Age Category'], 
        categories=ordered_ages, 
        ordered=True
    )
    age_analysis = age_analysis.sort_values('Age Category')
    
    fig = px.bar(
        age_analysis,
        x='Age Category',
        y='Number',
        color='Indicator Name',
        title='Death Counts by Age Category and Cause',
        barmode='group'
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive heatmap
    st.subheader("🔥 Interactive Demographic Heatmap")
    
    # Create pivot table for heatmap
    heatmap_data = filtered_df.groupby(['Age Category', 'Indicator Name'])['Death Rate'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Age Category', columns='Indicator Name', values='Death Rate')
    
    # Reorder age categories
    heatmap_pivot = heatmap_pivot.reindex(ordered_ages)
    
    fig = px.imshow(
        heatmap_pivot.values,
        labels=dict(x="Cause of Death", y="Age Category", color="Death Rate"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        color_continuous_scale="YlOrRd",
        title="Average Death Rate by Age Category and Cause"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics table
    st.subheader("📊 Summary Statistics")
    
    summary_stats = filtered_df.groupby('Indicator Name').agg({
        'Number': ['sum', 'mean', 'std'],
        'Death Rate': ['mean', 'std'],
        'Percent of All Causes': ['mean']
    }).round(2)
    
    summary_stats.columns = ['Total Deaths', 'Mean Deaths', 'Std Deaths', 
                           'Mean Death Rate', 'Std Death Rate', 'Mean Percentage']
    
    st.dataframe(summary_stats, use_container_width=True)

if __name__ == "__main__":
    main()
