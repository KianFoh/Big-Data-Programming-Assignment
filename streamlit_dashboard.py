import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Mortality Data Analysis Dashboard",
    page_icon=None,
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

def perform_dbscan_clustering(death_rate_matrix):
    """Perform DBSCAN clustering"""
    try:
        from sklearn.neighbors import NearestNeighbors
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(death_rate_matrix)
        
        # Determine optimal eps using k-distance graph
        min_samples = 2
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(scaled_features)
        distances, indices = neighbors_fit.kneighbors(scaled_features)
        distances = np.sort(distances[:, min_samples-1], axis=0)
        
        # Use median distance as eps
        eps = np.median(distances)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_features)
        
        # Create result dataframe
        dbscan_df = death_rate_matrix.copy()
        dbscan_df['Cluster'] = cluster_labels
        dbscan_df['Risk_Score'] = death_rate_matrix.sum(axis=1)
        
        # Calculate silhouette score (only if we have more than 1 cluster and no noise-only result)
        unique_labels = np.unique(cluster_labels)
        non_noise_labels = unique_labels[unique_labels != -1]  # Get non-noise cluster labels
        
        if len(non_noise_labels) > 1:  # Need at least 2 non-noise clusters
            # Exclude noise points for silhouette calculation
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(scaled_features[non_noise_mask], cluster_labels[non_noise_mask])
            else:
                silhouette_avg = 0
        else:
            silhouette_avg = 0
        
        return dbscan_df, silhouette_avg, scaled_features, eps, distances
    except ImportError:
        return None, None, None, None, None

def perform_kmeans_clustering(death_rate_matrix, n_clusters):
    """Perform K-Means clustering with elbow method analysis"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(death_rate_matrix)
    
    # Elbow method to find optimal number of clusters
    max_k = min(10, len(death_rate_matrix))  # Don't exceed number of data points
    inertia_values = []
    K_range = range(1, max_k)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia_values.append(kmeans.inertia_)
    
    # Apply K-Means with specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Calculate silhouette score
    if n_clusters > 1:
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    else:
        silhouette_avg = 0
    
    # Create result dataframe
    kmeans_df = death_rate_matrix.copy()
    kmeans_df['Cluster'] = cluster_labels
    kmeans_df['Risk_Score'] = death_rate_matrix.sum(axis=1)
    
    # Determine risk levels based on average death rates
    cluster_means = kmeans_df.groupby('Cluster')['Risk_Score'].mean()
    sorted_clusters = cluster_means.sort_values(ascending=False)
    
    # Assign risk levels
    risk_mapping = {}
    for i, cluster_id in enumerate(sorted_clusters.index):
        if i == 0:
            risk_mapping[cluster_id] = 'High Risk'
        elif i == len(sorted_clusters) - 1:
            risk_mapping[cluster_id] = 'Low Risk'
        else:
            risk_mapping[cluster_id] = 'Medium Risk'
    
    kmeans_df['Risk_Level'] = kmeans_df['Cluster'].map(risk_mapping)
    
    return kmeans_df, silhouette_avg, scaled_features, inertia_values, K_range, kmeans.cluster_centers_

def main():
    # Main header
    st.markdown('<h1 class="main-header">Mortality Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Year range selection
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    
    st.sidebar.subheader("Select Year Range")
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
        options=["Overview", "Death Count Analysis", "Growth Rate Analysis", "Clustering Analysis", "Demographic Analysis"]
    )
    
    # Get data for selected parameters
    top_causes = get_top_causes(df, start_year, end_year, n_causes)
    
    # Display selected parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Selected Parameters:**")
    st.sidebar.write(f"Years: {start_year} - {end_year}")
    st.sidebar.write(f"Top Causes: {n_causes}")
    st.sidebar.write(f"Clusters: {n_clusters}")
    
    # Main content area
    if analysis_type == "Overview":
        show_overview(df, start_year, end_year, top_causes)
    elif analysis_type == "Death Count Analysis":
        show_death_count_analysis(df, start_year, end_year, top_causes)
    elif analysis_type == "Growth Rate Analysis":
        show_growth_rate_analysis(df, start_year, end_year, top_causes)
    elif analysis_type == "Clustering Analysis":
        show_clustering_analysis(df, start_year, end_year, top_causes, n_clusters)
    elif analysis_type == "Demographic Analysis":
        show_demographic_analysis(df, start_year, end_year, top_causes)

def show_overview(df, start_year, end_year, top_causes):
    """Display overview statistics and visualizations"""
    st.markdown('<h2 class="sub-header">Overview Analysis</h2>', unsafe_allow_html=True)
    
    # Dataset summary
    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        st.metric("Total Records", f"{total_records:,.0f}")
    
    with col2:
        unique_causes = len(df['Indicator Name'].unique())
        st.metric("Unique Causes", unique_causes)
    
    with col3:
        year_range = f"{df['Year'].min()}-{df['Year'].max()}"
        st.metric("Year Range", year_range)
    
    with col4:
        age_categories = len(df['Age Category'].unique())
        st.metric("Age Categories", age_categories)
    
    # Filter data for the selected year range
    filtered_df = df[
        (df['Year'] >= start_year) & 
        (df['Year'] <= end_year) &
        (df['Indicator Name'] != 'All Causes')
    ]
    
    # Key metrics for selected period
    st.subheader(f"Key Metrics ({start_year}-{end_year})")
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
    st.subheader("Top Causes of Death")
    
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
    st.subheader("Death Rate Trends Over Time")
    
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
    """Display enhanced clustering analysis with both hierarchical and DBSCAN methods"""
    st.markdown('<h2 class="sub-header">Clustering Analysis</h2>', unsafe_allow_html=True)
    
    # Prepare data
    death_rate_matrix, clustering_data = prepare_clustering_data(df, start_year, end_year, top_causes)
    
    if death_rate_matrix.empty:
        st.error("No data available for the selected parameters.")
        return
    
    # Clustering method selection
    clustering_method = st.selectbox(
        "Select Clustering Method",
        options=["Hierarchical Clustering", "DBSCAN Clustering", "K-Means Clustering", "All Methods Comparison"]
    )
    
    # Special handling for All Methods Comparison
    if clustering_method == "All Methods Comparison":
        st.subheader("Clustering Methods Comparison")
        
        # Create tabs for organized comparison
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Hierarchical", "DBSCAN", "K-Means"])
        
        with tab1:
            st.write("**Compare all clustering methods side by side**")
            
            # Prepare results from all methods
            hierarchical_results = None
            dbscan_results = None  
            kmeans_results = None
            
            # Perform hierarchical clustering
            try:
                clustered_df, silhouette_score_val, scaled_features = perform_clustering(death_rate_matrix, n_clusters)
                hierarchical_results = {
                    'silhouette': silhouette_score_val,
                    'clusters': n_clusters,
                    'method': 'Hierarchical'
                }
            except Exception as e:
                st.warning(f"Hierarchical clustering failed: {str(e)}")
            
            # Perform DBSCAN clustering  
            try:
                dbscan_result = perform_dbscan_clustering(death_rate_matrix)
                if dbscan_result[0] is not None:
                    dbscan_df, dbscan_silhouette, _, _, _ = dbscan_result
                    unique_clusters = len(dbscan_df['Cluster'].unique())
                    noise_points = sum(dbscan_df['Cluster'] == -1)
                    dbscan_results = {
                        'silhouette': dbscan_silhouette,
                        'clusters': unique_clusters,
                        'noise_points': noise_points,
                        'method': 'DBSCAN'
                    }
            except Exception as e:
                st.warning(f"DBSCAN clustering failed: {str(e)}")
            
            # Perform K-Means clustering
            try:
                kmeans_result = perform_kmeans_clustering(death_rate_matrix, n_clusters)
                if kmeans_result[0] is not None:
                    kmeans_df, kmeans_silhouette, _, _, _, _ = kmeans_result
                    kmeans_results = {
                        'silhouette': kmeans_silhouette,
                        'clusters': n_clusters,
                        'method': 'K-Means'
                    }
            except Exception as e:
                st.warning(f"K-Means clustering failed: {str(e)}")
            
            # Create comparison table
            comparison_data = []
            methods = [hierarchical_results, dbscan_results, kmeans_results]
            
            for result in methods:
                if result is not None:
                    row = {
                        'Method': result['method'],
                        'Silhouette Score': f"{result['silhouette']:.3f}" if result['silhouette'] is not None else "N/A",
                        'Number of Clusters': str(result['clusters'])
                    }
                    if result['method'] == 'DBSCAN':
                        row['Noise Points'] = str(result.get('noise_points', 0))
                    else:
                        row['Noise Points'] = "N/A"
                    comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.astype(str)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Find best method by silhouette score
                valid_scores = [(float(row['Silhouette Score']), row['Method']) 
                               for row in comparison_data 
                               if row['Silhouette Score'] != "N/A"]
                
                if valid_scores:
                    best_score, best_method = max(valid_scores)
                    st.success(f"**Best performing method:** {best_method} (Silhouette Score: {best_score:.3f})")
                
                # Quick insights
                st.write("**Quick Insights:**")
                for row in comparison_data:
                    if row['Method'] == 'DBSCAN' and row['Noise Points'] != "N/A":
                        st.write(f"- {row['Method']}: {row['Number of Clusters']} clusters, {row['Noise Points']} noise points")
                    else:
                        st.write(f"- {row['Method']}: {row['Number of Clusters']} clusters")
            else:
                st.error("No clustering methods completed successfully.")
        
        # Individual method tabs will be handled by the existing conditional logic
        with tab2:
            st.write("Hierarchical clustering results will be shown here when you select the 'Hierarchical Clustering' option above.")
        
        with tab3:
            st.write("DBSCAN clustering results will be shown here when you select the 'DBSCAN Clustering' option above.")
        
        with tab4:
            st.write("K-Means clustering results will be shown here when you select the 'K-Means Clustering' option above.")
        
        # Show a note about individual method selection
        st.info("**Tip:** To see detailed results for each method, select the specific clustering method from the dropdown above.")
    
    if clustering_method == "Hierarchical Clustering":
        st.subheader("Hierarchical Clustering Analysis")
        
        # Perform hierarchical clustering
        clustered_df, silhouette_score_val, scaled_features = perform_clustering(death_rate_matrix, n_clusters)
        
        # Display clustering results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Silhouette Score", f"{silhouette_score_val:.3f}")
        
        with col2:
            st.metric("Age Categories", len(death_rate_matrix))
        
        with col3:
            st.metric("Number of Clusters", n_clusters)
        
        # Death rate heatmap
        st.write("**Death Rate Heatmap**")
        
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
        
        # Clustering visualization
        st.write("**Cluster Visualization**")
        
        fig = px.scatter(
            x=clustered_df.index,
            y=clustered_df['Risk_Score'],
            color=clustered_df['Cluster'].astype(str),
            title=f'Age Categories Clustered by Risk Score (k={n_clusters})',
            labels={'x': 'Age Category', 'y': 'Total Risk Score', 'color': 'Cluster'},
            text=clustered_df.index
        )
        fig.update_traces(textposition='middle right')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster analysis table
        st.write("**Cluster Analysis Summary**")
        
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
        
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)
    
    if clustering_method == "DBSCAN Clustering":
        st.subheader("DBSCAN Clustering Analysis")
        
        # Perform DBSCAN clustering
        dbscan_result = perform_dbscan_clustering(death_rate_matrix)
        
        if dbscan_result[0] is not None:
            dbscan_df, dbscan_silhouette, scaled_features, eps, distances = dbscan_result
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("DBSCAN Silhouette Score", f"{dbscan_silhouette:.3f}")
            
            with col2:
                unique_clusters = len(dbscan_df['Cluster'].unique())
                st.metric("Clusters Found", unique_clusters)
            
            with col3:
                noise_points = sum(dbscan_df['Cluster'] == -1)
                st.metric("Noise Points", noise_points)
            
            # K-distance graph
            st.write("**K-Distance Graph for DBSCAN Parameter Selection**")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(distances))),
                y=distances,
                mode='lines',
                name='K-distance'
            ))
            fig.add_hline(y=eps, line_dash='dash', line_color='red', 
                         annotation_text=f'Selected eps: {eps:.3f}')
            fig.update_layout(
                title='K-distance Graph for DBSCAN eps Determination',
                xaxis_title='Points sorted by distance',
                yaxis_title='2nd nearest neighbor distance',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # DBSCAN Results visualization
            st.write("**DBSCAN Clustering Results**")
            
            # Sort by cluster and risk score
            dbscan_display = dbscan_df.sort_values(['Cluster', 'Risk_Score'], ascending=[True, False])
            
            # Display cluster membership
            unique_clusters = sorted(dbscan_df['Cluster'].unique())
            
            for cluster_id in unique_clusters:
                cluster_ages = dbscan_df[dbscan_df['Cluster'] == cluster_id]
                avg_risk = cluster_ages['Risk_Score'].mean()
                
                if cluster_id == -1:
                    st.write(f"**Noise Points** (Avg Risk Score: {avg_risk:.1f})")
                else:
                    st.write(f"**Cluster {cluster_id}** (Avg Risk Score: {avg_risk:.1f})")
                
                # Create a simple table for this cluster
                cluster_data = []
                for age in cluster_ages.index:
                    risk_score = cluster_ages.loc[age, 'Risk_Score']
                    cluster_data.append({'Age Category': age, 'Risk Score': f"{risk_score:.1f}"})
                
                cluster_table = pd.DataFrame(cluster_data)
                st.dataframe(cluster_table, hide_index=True, use_container_width=True)
            
            # Risk segment analysis
            st.write("**Risk Segment Analysis**")
            
            # Calculate risk levels
            dbscan_cluster_analysis = []
            for cluster_id in unique_clusters:
                cluster_ages = dbscan_df[dbscan_df['Cluster'] == cluster_id]
                
                cluster_info = {
                    'Cluster': cluster_id,
                    'Age_Categories': list(cluster_ages.index),
                    'Count': len(cluster_ages),
                    'Avg_Risk_Score': cluster_ages['Risk_Score'].mean(),
                    'Min_Risk_Score': cluster_ages['Risk_Score'].min(),
                    'Max_Risk_Score': cluster_ages['Risk_Score'].max()
                }
                dbscan_cluster_analysis.append(cluster_info)
            
            # Sort by average risk score
            dbscan_cluster_analysis.sort(key=lambda x: x['Avg_Risk_Score'], reverse=True)
            
            # Assign risk levels
            for i, cluster in enumerate(dbscan_cluster_analysis):
                if cluster['Cluster'] == -1:
                    cluster['Risk_Level'] = 'Outlier'
                elif i == 0:
                    cluster['Risk_Level'] = 'High Risk'
                elif i == len(dbscan_cluster_analysis) - 1:
                    cluster['Risk_Level'] = 'Low Risk'
                else:
                    cluster['Risk_Level'] = 'Medium Risk'
            
            # Create summary visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Risk Level Distribution',
                    'Age Categories by Risk Score',
                    'Cluster Size Distribution',
                    'Average Risk Score by Cluster'
                ),
                specs=[
                    [{"type": "pie"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "bar"}]
                ]
            )
            
            # Risk level pie chart
            risk_counts = {}
            for cluster in dbscan_cluster_analysis:
                risk_level = cluster['Risk_Level']
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + cluster['Count']
            
            fig.add_trace(
                go.Pie(
                    labels=list(risk_counts.keys()), 
                    values=list(risk_counts.values()),
                    hole=0.4
                ),
                row=1, col=1
            )
            
            # Scatter plot of age categories by risk score
            fig.add_trace(
                go.Scatter(
                    x=dbscan_df.index,
                    y=dbscan_df['Risk_Score'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=dbscan_df['Cluster'],
                        colorscale='viridis'
                    ),
                    text=['Cluster: '+str(c) for c in dbscan_df['Cluster']]
                ),
                row=1, col=2
            )
            
            # Cluster size bar chart
            cluster_sizes = dbscan_df['Cluster'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=[f'Cluster {c}' if c != -1 else 'Noise' for c in cluster_sizes.index],
                    y=cluster_sizes.values,
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            # Average risk score by cluster
            cluster_names = []
            cluster_risks = []
            for cluster in dbscan_cluster_analysis:
                cluster_names.append(f'Cluster {cluster["Cluster"]}' if cluster["Cluster"] != -1 else 'Noise')
                cluster_risks.append(cluster['Avg_Risk_Score'])
            
            fig.add_trace(
                go.Bar(
                    x=cluster_names,
                    y=cluster_risks,
                    marker_color='orange'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="DBSCAN Clustering Analysis Summary",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("DBSCAN clustering requires additional dependencies. Showing hierarchical clustering only.")
    
    if clustering_method == "K-Means Clustering":
        st.subheader("K-Means Clustering Analysis")
        
        # Perform K-Means clustering
        kmeans_result = perform_kmeans_clustering(death_rate_matrix, n_clusters)
        
        if kmeans_result[0] is not None:
            kmeans_df, kmeans_silhouette, scaled_features, inertia_values, K_range, cluster_centers = kmeans_result
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("K-Means Silhouette Score", f"{kmeans_silhouette:.3f}")
            
            with col2:
                st.metric("Clusters Found", n_clusters)
            
            with col3:
                st.metric("Max Inertia", f"{inertia_values[-1]:,.0f}")
            
            # Elbow method plot
            st.write("**Elbow Method for Optimal Clusters**")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(K_range),
                y=inertia_values,
                mode='lines+markers',
                name='Inertia'
            ))
            fig.update_layout(
                title='Elbow Method for Optimal Number of Clusters',
                xaxis_title='Number of Clusters',
                yaxis_title='Inertia',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # K-Means Results visualization
            st.write("**K-Means Clustering Results**")
            
            # Sort by cluster and risk score
            kmeans_display = kmeans_df.sort_values(['Cluster', 'Risk_Score'], ascending=[True, False])
            
            # Display cluster membership
            unique_clusters = sorted(kmeans_df['Cluster'].unique())
            
            for cluster_id in unique_clusters:
                cluster_ages = kmeans_df[kmeans_df['Cluster'] == cluster_id]
                avg_risk = cluster_ages['Risk_Score'].mean()
                
                st.write(f"**Cluster {cluster_id}** (Avg Risk Score: {avg_risk:.1f})")
                
                # Create a simple table for this cluster
                cluster_data = []
                for age in cluster_ages.index:
                    risk_score = cluster_ages.loc[age, 'Risk_Score']
                    cluster_data.append({'Age Category': age, 'Risk Score': f"{risk_score:.1f}"})
                
                cluster_table = pd.DataFrame(cluster_data)
                st.dataframe(cluster_table, hide_index=True, use_container_width=True)
            
            # Risk segment analysis
            st.write("**Risk Segment Analysis**")
            
            # Calculate risk levels
            kmeans_cluster_analysis = []
            for cluster_id in unique_clusters:
                cluster_ages = kmeans_df[kmeans_df['Cluster'] == cluster_id]
                
                cluster_info = {
                    'Cluster': cluster_id,
                    'Age_Categories': list(cluster_ages.index),
                    'Count': len(cluster_ages),
                    'Avg_Risk_Score': cluster_ages['Risk_Score'].mean(),
                    'Min_Risk_Score': cluster_ages['Risk_Score'].min(),
                    'Max_Risk_Score': cluster_ages['Risk_Score'].max()
                }
                kmeans_cluster_analysis.append(cluster_info)
            
            # Sort by average risk score
            kmeans_cluster_analysis.sort(key=lambda x: x['Avg_Risk_Score'], reverse=True)
            
            # Assign risk levels
            for i, cluster in enumerate(kmeans_cluster_analysis):
                if i == 0:
                    cluster['Risk_Level'] = 'High Risk'
                elif i == len(kmeans_cluster_analysis) - 1:
                    cluster['Risk_Level'] = 'Low Risk'
                else:
                    cluster['Risk_Level'] = 'Medium Risk'
            
            # Create summary visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Risk Level Distribution',
                    'Age Categories by Risk Score',
                    'Cluster Size Distribution',
                    'Average Risk Score by Cluster'
                ),
                specs=[
                    [{"type": "pie"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "bar"}]
                ]
            )
            
            # Risk level pie chart
            risk_counts = {}
            for cluster in kmeans_cluster_analysis:
                risk_level = cluster['Risk_Level']
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + cluster['Count']
            
            fig.add_trace(
                go.Pie(
                    labels=list(risk_counts.keys()), 
                    values=list(risk_counts.values()),
                    hole=0.4
                ),
                row=1, col=1
            )
            
            # Scatter plot of age categories by risk score
            fig.add_trace(
                go.Scatter(
                    x=kmeans_df.index,
                    y=kmeans_df['Risk_Score'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=kmeans_df['Cluster'],
                        colorscale='viridis'
                    ),
                    text=['Cluster: '+str(c) for c in kmeans_df['Cluster']]
                ),
                row=1, col=2
            )
            
            # Cluster size bar chart
            cluster_sizes = kmeans_df['Cluster'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=[f'Cluster {c}' for c in cluster_sizes.index],
                    y=cluster_sizes.values,
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            # Average risk score by cluster
            cluster_names = []
            cluster_risks = []
            for cluster in kmeans_cluster_analysis:
                cluster_names.append(f'Cluster {cluster["Cluster"]}')
                cluster_risks.append(cluster['Avg_Risk_Score'])
            
            fig.add_trace(
                go.Bar(
                    x=cluster_names,
                    y=cluster_risks,
                    marker_color='orange'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="K-Means Clustering Analysis Summary",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("K-Means clustering requires additional dependencies. Showing hierarchical clustering only.")
    
    # Risk recommendations (only for hierarchical clustering)
    if clustering_method == "Hierarchical Clustering":
        st.subheader("Risk-Based Recommendations")
        
        # Use hierarchical clustering results for recommendations
        if 'clustered_df' in locals():
            cluster_analysis = []
            for cluster_id in range(n_clusters):
                cluster_ages = clustered_df[clustered_df['Cluster'] == cluster_id]
                cluster_analysis.append({
                    'Cluster': cluster_id,
                    'Age Categories': ', '.join(cluster_ages.index),
                    'Count': len(cluster_ages),
                    'Avg Risk Score': cluster_ages['Risk_Score'].mean(),
                })
            
            cluster_df = pd.DataFrame(cluster_analysis)
            cluster_df = cluster_df.sort_values('Avg Risk Score', ascending=False)
            
            for i, row in cluster_df.iterrows():
                risk_level = "High Risk" if row['Avg Risk Score'] > cluster_df['Avg Risk Score'].median() else "Low Risk"
                
                with st.expander(f"Cluster {row['Cluster']} - {risk_level}"):
                    st.write(f"**Age Categories:** {row['Age Categories']}")
                    st.write(f"**Average Risk Score:** {row['Avg Risk Score']:.1f}")
                    
                    if risk_level == "High Risk":
                        st.write("**Recommendations:**")
                        st.write("- Intensive health monitoring and screening programs")
                        st.write("- Specialized medical facilities and emergency care")
                        st.write("- Comprehensive insurance coverage")
                        st.write("- Priority access to preventive care services")
                    else:
                        st.write("**Recommendations:**")
                        st.write("- Regular health check-ups and screening")
                        st.write("- Preventive care and lifestyle counseling")
                        st.write("- Health education programs")
                        st.write("- Community-based wellness initiatives")

def show_demographic_analysis(df, start_year, end_year, top_causes):
    """Display demographic analysis"""
    st.markdown('<h2 class="sub-header">Demographic Analysis</h2>', unsafe_allow_html=True)
    
    # Filter data
    filtered_df = df[
        (df['Indicator Name'].isin(top_causes)) &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year) &
        (df['Age Category'] != 'All Ages') &
        (df['Sex'] != 'All')
    ]
    
    # Gender analysis
    st.subheader("Gender Distribution")
    
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
    st.subheader("Age Group Distribution")
    
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
    st.subheader("Interactive Demographic Heatmap")
    
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
    st.subheader("Summary Statistics")
    
    summary_stats = filtered_df.groupby('Indicator Name').agg({
        'Number': ['sum', 'mean', 'std'],
        'Death Rate': ['mean', 'std'],
        'Percent of All Causes': ['mean']
    }).round(2)
    
    summary_stats.columns = ['Total Deaths', 'Mean Deaths', 'Std Deaths', 
                           'Mean Death Rate', 'Std Death Rate', 'Mean Percentage']
    
    st.dataframe(summary_stats, use_container_width=True)

def show_death_count_analysis(df, start_year, end_year, top_causes):
    """Display death count analysis similar to top5_cod_deathcount.ipynb"""
    st.markdown('<h2 class="sub-header">Death Count Analysis</h2>', unsafe_allow_html=True)
    
    # Overall death count trends
    st.subheader("Overall Death Count Trends (All Causes)")
    
    # Filter for all causes data
    all_causes_data = df[
        (df['Indicator Name'] == 'All Causes') &
        (df['Sex'] == 'All') &
        (df['Age Category'] == 'All Ages') &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year)
    ].sort_values('Year')
    
    if not all_causes_data.empty:
        fig = px.line(
            all_causes_data,
            x='Year',
            y='Number',
            title='Death Count for All Causes (All Ages, All Sexes)',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 highest death count causes
    st.subheader("Top 5 Highest Death Count Causes")
    
    filtered_df = df[
        (df['Indicator Name'] != 'All Causes') &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year)
    ]
    
    # Calculate temporal trends for top causes
    temporal_data = filtered_df[filtered_df['Indicator Name'].isin(top_causes)].copy()
    yearly_trends = (temporal_data
                     .groupby(['Year', 'Indicator Name'])['Number']
                     .sum()
                     .reset_index())
    
    fig = px.line(
        yearly_trends,
        x='Year',
        y='Number',
        color='Indicator Name',
        title=f'Top {len(top_causes)} Causes of Death by Total Death Count ({start_year}–{end_year})',
        markers=True
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Gender distribution analysis
    st.subheader("Distribution Across Genders")
    
    # Filter and prepare gender data
    gender_data = df[
        (df['Indicator Name'].isin(top_causes)) &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year) &
        (df['Sex'].isin(['Male', 'Female', 'All']))
    ]
    
    # Create subplots for each cause
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[cause[:30] + '...' if len(cause) > 30 else cause for cause in top_causes],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, None]]
    )
    
    colors = {'Male': '#4A90E2', 'Female': '#E24A90', 'All': '#888888'}
    
    for idx, cause in enumerate(top_causes):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        cause_data = gender_data[gender_data['Indicator Name'] == cause]
        trend = (cause_data.groupby(['Year', 'Sex'])['Number']
                           .sum()
                           .reset_index())
        
        for gender in ['Male', 'Female', 'All']:
            gender_trend = trend[trend['Sex'] == gender]
            if not gender_trend.empty:
                fig.add_trace(
                    go.Scatter(
                        x=gender_trend['Year'],
                        y=gender_trend['Number'],
                        mode='lines+markers',
                        name=gender,
                        line=dict(color=colors[gender]),
                        showlegend=(idx == 0),
                        legendgroup=gender
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(height=800, title_text="Distribution of Top 5 Causes of Death Across Genders")
    st.plotly_chart(fig, use_container_width=True)
    
    # Age group analysis
    st.subheader("Distribution Across Age Groups")
    
    # Custom age order
    age_order = [
        "Infant", "Toddler", "Child", "Teenager", "Young Adult",
        "Adult", "Middle Age", "Senior", "Elderly"
    ]
    
    age_data = df[
        (df['Indicator Name'].isin(top_causes)) &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year) &
        (df['Age Category'] != 'All Ages')
    ]
    
    demographic_analysis = (
        age_data
        .groupby(['Indicator Name', 'Sex', 'Age Category'])['Number']
        .sum()
        .reset_index()
    )
    
    # Interactive visualization for age groups
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[cause[:30] + '...' if len(cause) > 30 else cause for cause in top_causes],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, None]]
    )
    
    for idx, cause in enumerate(top_causes):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        cause_data = demographic_analysis[demographic_analysis['Indicator Name'] == cause]
        
        male_data = (
            cause_data[cause_data['Sex'] == 'Male']
            .set_index('Age Category')
            .reindex(age_order)['Number']
            .fillna(0)
        )
        
        female_data = (
            cause_data[cause_data['Sex'] == 'Female']
            .set_index('Age Category')
            .reindex(age_order)['Number']
            .fillna(0)
        )
        
        if not male_data.empty:
            fig.add_trace(
                go.Bar(
                    x=male_data.index,
                    y=male_data.values,
                    name='Male',
                    marker_color='#4A90E2',
                    showlegend=(idx == 0),
                    legendgroup='Male'
                ),
                row=row, col=col
            )
        if not female_data.empty:
            fig.add_trace(
                go.Bar(
                    x=female_data.index,
                    y=female_data.values,
                    name='Female',
                    marker_color='#E24A90',
                    showlegend=(idx == 0),
                    legendgroup='Female'
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=900,
        title_text="Top 5 Causes of Death by Age Group and Gender",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 causes for each age category
    st.subheader("Top 5 Causes for Each Age Category")
    
    specific_age_categories = sorted([age for age in age_data['Age Category'].unique() if age in age_order])
    
    # Create tabs for different age categories
    if specific_age_categories:
        age_tabs = st.tabs(specific_age_categories)
        
        for i, age_category in enumerate(specific_age_categories):
            with age_tabs[i]:
                age_subset = age_data[age_data['Age Category'] == age_category]
                top5_causes_age = (
                    age_subset.groupby('Indicator Name')['Number']
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                )
                
                if not top5_causes_age.empty:
                    # Create DataFrame for display
                    df_top5 = pd.DataFrame({
                        'Rank': range(1, len(top5_causes_age) + 1),
                        'Cause of Death': top5_causes_age.index,
                        'Total Deaths': [f"{int(n):,}" for n in top5_causes_age.values]
                    })
                    
                    st.dataframe(df_top5, hide_index=True, use_container_width=True)
                    
                    # Bar chart
                    fig = px.bar(
                        x=top5_causes_age.values,
                        y=top5_causes_age.index,
                        orientation='h',
                        title=f'Top 5 Causes for {age_category}',
                        labels={'x': 'Total Deaths', 'y': 'Cause of Death'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

def show_growth_rate_analysis(df, start_year, end_year, top_causes):
    """Display growth rate analysis similar to top5_cod_growthrate.ipynb"""
    st.markdown('<h2 class="sub-header">Growth Rate Analysis</h2>', unsafe_allow_html=True)
    
    # Death rate per year for all causes
    st.subheader("Death Rate Trends (All Causes)")
    
    all_causes_rate = df[
        (df['Sex'] == 'All') &
        (df['Age Category'] == 'All Ages') &
        (df['Indicator Name'] == 'All Causes') &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year)
    ].copy()
    
    if not all_causes_rate.empty:
        fig = px.line(
            all_causes_rate,
            x='Year',
            y='Death Rate',
            title='Death Rate per Year for All Causes (All Genders, All Ages)',
            markers=True,
            color_discrete_sequence=['darkred']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Growth rate calculation and visualization
    st.subheader("Growth Rate of Death Rate")
    
    if not all_causes_rate.empty:
        # Calculate growth rate
        filtered_rate = all_causes_rate.sort_values('Year').reset_index(drop=True)
        filtered_rate['Death_Rate_Growth_%'] = filtered_rate['Death Rate'].pct_change() * 100
        filtered_rate['Death_Rate_Growth_%'] = filtered_rate['Death_Rate_Growth_%'].round(2)
        filtered_rate.loc[0, 'Death_Rate_Growth_%'] = 0
        
        # Display data table
        result = filtered_rate[['Year', 'Death Rate', 'Death_Rate_Growth_%']].reset_index(drop=True)
        st.dataframe(result, hide_index=True, use_container_width=True)
        
        # Interactive growth rate visualization
        def marker_color(val):
            if val > 0:
                return 'red'
            elif val < 0:
                return 'green'
            else:
                return 'gray'
        
        filtered_rate['Color'] = filtered_rate['Death_Rate_Growth_%'].apply(marker_color)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=filtered_rate['Year'],
            y=filtered_rate['Death_Rate_Growth_%'],
            mode='lines+markers',
            marker=dict(
                color=filtered_rate['Color'],
                size=10,
                line=dict(width=2, color='black')
            ),
            line=dict(width=2, color='lightgray'),
            text=[
                f"{year}<br>Growth Rate: {growth}%<br>Death Rate: {rate:.2f}<br>Deaths: {int(deaths):,}"
                for year, growth, rate, deaths in zip(
                    filtered_rate['Year'],
                    filtered_rate['Death_Rate_Growth_%'],
                    filtered_rate['Death Rate'],
                    filtered_rate['Number']
                )
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Death_Rate_Growth_%'
        ))
        
        fig.add_hline(y=0, line_dash='dash', line_color='gray')
        
        fig.update_layout(
            title='Death Rate Growth for All Causes',
            xaxis_title='Year',
            yaxis_title='Death Rate Growth (%)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 causes by growth rate
    st.subheader("Top 5 Causes by Growth Rate of Death Rate")
    
    # Filter and prepare data for growth rate analysis
    growth_df = df[
        (df['Indicator Name'] != 'All Causes') &
        (df['Sex'] == 'All') &
        (df['Age Category'] == 'All Ages') &
        (df['Year'] >= start_year) &
        (df['Year'] <= end_year)
    ]
    
    if not growth_df.empty:
        # Group by indicator and year
        growth_grouped = growth_df.groupby(['Indicator Name', 'Year'])['Death Rate'].sum().reset_index()
        
        # Find earliest and latest years
        min_year = growth_grouped['Year'].min()
        max_year = growth_grouped['Year'].max()
        
        # Get start and end data
        start_data = growth_grouped[growth_grouped['Year'] == min_year]
        end_data = growth_grouped[growth_grouped['Year'] == max_year]
        
        # Merge and calculate growth rate
        merged = pd.merge(start_data, end_data, on='Indicator Name', suffixes=('_start', '_end'))
        merged['Growth_Rate_%'] = ((merged['Death Rate_end'] - merged['Death Rate_start']) / 
                                   merged['Death Rate_start'].replace(0, 1)) * 100
        merged['Growth_Rate_%'] = merged['Growth_Rate_%'].round(2)
        
        # Get top 5 growth rates
        top5_growth = merged.sort_values(by='Growth_Rate_%', ascending=False).head(5)
        
        # Display table
        display_table = top5_growth[['Indicator Name', 'Death Rate_start', 'Death Rate_end', 'Growth_Rate_%']].copy()
        display_table.columns = ['Cause of Death', f'{min_year} Death Rate', f'{max_year} Death Rate', 'Growth Rate (%)']
        st.dataframe(display_table, hide_index=True, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            top5_growth,
            x='Indicator Name',
            y='Growth_Rate_%',
            title='Top 5 Causes by Growth Rate of Death Rate (%)',
            labels={'Growth_Rate_%': 'Growth Rate (%)', 'Indicator Name': 'Cause of Death'},
            color='Growth_Rate_%',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Between Top 5 Growth Rate Causes")
        
        top5_growth_causes = top5_growth['Indicator Name'].tolist()
        
        # Filter for correlation analysis
        corr_df = df[
            (df['Indicator Name'].isin(top5_growth_causes)) &
            (df['Sex'] == 'All') &
            (df['Age Category'] == 'All Ages') &
            (df['Year'] >= start_year) &
            (df['Year'] <= end_year)
        ]
        
        if not corr_df.empty:
            # Pivot table for correlation
            pivot_df = corr_df.pivot_table(
                index='Year',
                columns='Indicator Name',
                values='Death Rate'
            )
            
            # Calculate correlation
            corr_matrix = pivot_df.corr()
            
            # Heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            plt.title("Correlation of Death Rate Trends Between Top 5 Growth Rate Causes")
            plt.tight_layout()
            st.pyplot(fig)
        
        # Distribution by demographics for top growth causes
        st.subheader("Distribution of Top Growth Rate Causes by Demographics")
        
        demo_df = df[
            (df['Indicator Name'].isin(top5_growth_causes)) &
            (df['Year'] >= start_year) &
            (df['Year'] <= end_year)
        ]
        
        if not demo_df.empty:
            # Group by cause, age, and sex
            grouped_demo = demo_df.groupby(['Indicator Name', 'Age Category', 'Sex'])['Number'].sum().reset_index()
            
            # Create heatmaps for each cause
            for cause in top5_growth_causes:
                st.write(f"**{cause}**")
                
                cause_data = grouped_demo[grouped_demo['Indicator Name'] == cause]
                pivot_demo = cause_data.pivot(index='Age Category', columns='Sex', values='Number').fillna(0)
                
                if not pivot_demo.empty:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(pivot_demo, annot=True, fmt=".0f", cmap='YlGnBu', ax=ax)
                    plt.title(f'Distribution of Deaths by Age & Sex: {cause}')
                    plt.ylabel('Age Category')
                    plt.xlabel('Sex')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Correlation between Male and Female
                    if 'Male' in pivot_demo.columns and 'Female' in pivot_demo.columns:
                        correlation = pivot_demo['Male'].corr(pivot_demo['Female'])
                        st.write(f"Correlation between Male and Female distribution: {correlation:.2f}")

if __name__ == "__main__":
    main()
