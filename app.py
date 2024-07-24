import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dateutil import parser
from sklearn.cluster import KMeans
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_elements import elements, mui, html
import io
import base64

# Set page config
st.set_page_config(page_title="Process Mining App", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Process Map", "Variants", "Statistics", "Root Cause Analysis"])

# Function to load data
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xes'):
        # Basic XES parsing (you might want to use a dedicated XES parser for more complex files)
        import xml.etree.ElementTree as ET
        tree = ET.parse(file)
        root = tree.getroot()
        
        data = []
        for trace in root.findall('{http://www.xes-standard.org/}trace'):
            case_id = trace.find('{http://www.xes-standard.org/}string[@key="concept:name"]').get('value')
            for event in trace.findall('{http://www.xes-standard.org/}event'):
                activity = event.find('{http://www.xes-standard.org/}string[@key="concept:name"]').get('value')
                timestamp = event.find('{http://www.xes-standard.org/}date[@key="time:timestamp"]').get('value')
                data.append({'case:concept:name': case_id, 'concept:name': activity, 'time:timestamp': timestamp})
        
        df = pd.DataFrame(data)
    else:
        st.error("Unsupported file format")
        return None
    return df

# Function to preprocess data
def preprocess_data(df):
    required_columns = ['case:concept:name', 'concept:name', 'time:timestamp']
    if not all(col in df.columns for col in required_columns):
        st.error("Required columns missing. Please ensure your data has case ID, activity, and timestamp columns.")
        return None
    
    df = df.rename(columns={
        'case:concept:name': 'case_id',
        'concept:name': 'activity',
        'time:timestamp': 'timestamp'
    })
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['case_id', 'timestamp'])
    
    return df

# Function to create process map
def create_process_map(df):
    edges = df.groupby('case_id')['activity'].apply(lambda x: list(zip(x, x[1:]))).explode()
    edge_counts = edges.value_counts()

    G = nx.DiGraph()
    for (source, target), weight in edge_counts.items():
        G.add_edge(source, target, weight=weight)

    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right')
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{adjacencies[0]}<br># of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Process Map',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Process Map",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    return fig

# Function to analyze variants
def analyze_variants(df):
    variants = df.groupby('case_id')['activity'].agg(lambda x: '->'.join(x)).value_counts()
    return variants

# Function to calculate statistics
def calculate_statistics(df):
    case_durations = df.groupby('case_id').apply(lambda x: (x['timestamp'].max() - x['timestamp'].min()).total_seconds() / 86400)
    stats = {
        'Total cases': df['case_id'].nunique(),
        'Total events': len(df),
        'Unique activities': df['activity'].nunique(),
        'Average case duration (days)': case_durations.mean(),
        'Median case duration (days)': case_durations.median(),
        'Min case duration (days)': case_durations.min(),
        'Max case duration (days)': case_durations.max()
    }
    return stats

# Function for root cause analysis
def root_cause_analysis(df, target_activity):
    # Prepare data for clustering
    cases = df.groupby('case_id').agg({
        'activity': lambda x: '->'.join(x),
        'timestamp': ['min', 'max']
    })
    cases.columns = ['sequence', 'start_time', 'end_time']
    cases['duration'] = (cases['end_time'] - cases['start_time']).dt.total_seconds() / 86400
    cases['target'] = cases['sequence'].str.contains(target_activity).astype(int)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    cases['cluster'] = kmeans.fit_predict(cases[['duration', 'target']])

    return cases

# Function to export data
def export_data(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="process_mining_data.csv">Download CSV File</a>'
    return href

# Main content
if page == "Upload Data":
    st.title("Upload Event Log")
    uploaded_file = st.file_uploader("Choose a CSV or XES file", type=["csv", "xes"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            df = preprocess_data(df)
            if df is not None:
                st.session_state['data'] = df
                st.success("Data loaded successfully!")
                
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_pagination()
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                grid_options = gb.build()
                
                AgGrid(df, gridOptions=grid_options, enable_enterprise_modules=True)
                
                st.markdown(export_data(df), unsafe_allow_html=True)

elif page == "Process Map":
    st.title("Process Map")
    if 'data' in st.session_state:
        df = st.session_state['data']
        fig = create_process_map(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive filtering
        st.subheader("Filter Data")
        start_date = st.date_input("Start Date", df['timestamp'].min().date())
        end_date = st.date_input("End Date", df['timestamp'].max().date())
        activities = st.multiselect("Select Activities", df['activity'].unique())
        
        filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
        if activities:
            filtered_df = filtered_df[filtered_df['activity'].isin(activities)]
        
        st.write(f"Filtered data contains {filtered_df['case_id'].nunique()} cases and {len(filtered_df)} events.")
    else:
        st.warning("Please upload data first")

elif page == "Variants":
    st.title("Variant Analysis")
    if 'data' in st.session_state:
        df = st.session_state['data']
        variants = analyze_variants(df)
        st.write("Top 10 Variants:")
        st.write(variants.head(10))
        
        fig = px.bar(variants.head(10), x=variants.head(10).index, y=variants.head(10).values)
        fig.update_layout(title="Top 10 Variants", xaxis_title="Variant", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        
        # Variant details
        st.subheader("Variant Details")
        selected_variant = st.selectbox("Select a variant", variants.index)
        variant_cases = df.groupby('case_id').filter(lambda x: '->'.join(x['activity']) == selected_variant)
        st.write(f"Cases with this variant: {variant_cases['case_id'].nunique()}")
        st.write(variant_cases)
    else:
        st.warning("Please upload data first")

elif page == "Statistics":
    st.title("Process Statistics")
    if 'data' in st.session_state:
        df = st.session_state['data']
        stats = calculate_statistics(df)
        
        # Use Streamlit Elements to create a more professional-looking UI
        with elements("stats_dashboard"):
            mui.Box(
                mui.Grid(
                    mui.Grid(
                        mui.Paper(
                            mui.Typography(key, variant="h6"),
                            mui.Typography(f"{value:.2f}" if isinstance(value, float) else value, variant="h4"),
                            sx={"p": 2, "textAlign": "center"}
                        ),
                        xs=12, sm=6, md=4, key=key
                    ) for key, value in stats.items()
                ),
                sx={"flexGrow": 1, "p": 2}
            )
        
        # Activity frequency chart
        activity_freq = df['activity'].value_counts()
        fig = px.bar(x=activity_freq.index, y=activity_freq.values)
        fig.update_layout(title="Activity Frequency", xaxis_title="Activity", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload data first")

elif page == "Root Cause Analysis":
    st.title("Root Cause Analysis")
    if 'data' in st.session_state:
        df = st.session_state['data']
        target_activity = st.selectbox("Select target activity", df['activity'].unique())
        if st.button("Perform Root Cause Analysis"):
            results = root_cause_analysis(df, target_activity)
            
            fig = px.scatter(results, x='duration', y='target', color='cluster', hover_data=['sequence'])
            fig.update_layout(title=f"Root Cause Analysis for {target_activity}", xaxis_title="Case Duration (days)", yaxis_title="Contains Target Activity")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Cluster Analysis")
            for cluster in results['cluster'].unique():
                st.write(f"Cluster {cluster}:")
                cluster_data = results[results['cluster'] == cluster]
                st.write(f"Average duration: {cluster_data['duration'].mean():.2f} days")
                st.write(f"Percentage containing target activity: {(cluster_data['target'].mean() * 100):.2f}%")
                st.write("Most common sequences:")
                st.write(cluster_data['sequence'].value_counts().head())
                st.write("---")
    else:
        st.warning("Please upload data first")
