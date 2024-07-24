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
import graphviz
from io import BytesIO

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
page = st.sidebar.radio("Go to", ["Upload Data", "Process Map", "Time Analysis", "Bottleneck Analysis", "Variants", "Statistics", "Root Cause Analysis"])

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
def preprocess_data(df, case_id_col, activity_col, timestamp_col):
    df = df.rename(columns={
        case_id_col: 'case_id',
        activity_col: 'activity',
        timestamp_col: 'timestamp'
    })
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['case_id', 'timestamp'])
    
    return df

# Function to create an enhanced process map
def create_enhanced_process_map(df):
    edges = df.groupby('case_id')['activity'].apply(lambda x: list(zip(x, x[1:])))
    edge_counts = edges.explode().value_counts()
    
    G = nx.DiGraph()
    for (source, target), weight in edge_counts.items():
        G.add_edge(source, target, weight=weight)
    
    # Calculate node sizes based on frequency
    node_sizes = df['activity'].value_counts()
    node_sizes = (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min()) * 50 + 10
    
    # Calculate edge widths based on frequency
    edge_widths = (edge_counts - edge_counts.min()) / (edge_counts.max() - edge_counts.min()) * 5 + 1
    
    # Calculate layout
    pos = nx.spring_layout(G)
    
    # Create edges
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if (edge[0], edge[1]) in edge_widths:
            weight = edge_widths[(edge[0], edge[1])]
        else:
            weight = 1  # Default width if edge is not in edge_widths
        edge_trace.append(
            go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                       line=dict(width=weight, color='#888'),
                       hoverinfo='none',
                       mode='lines')
        )
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(node_sizes.get(node, 10))  # Default size if node is not in node_sizes
        node_color.append(G.degree(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='top center',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=node_size,
            color=node_color,
            line_width=2,
            colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right')
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title='Enhanced Process Map',
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

# Function to create Petri net style process map
def create_petri_net_map(df, bottlenecks):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges
    for case_id, case_df in df.groupby('case_id'):
        activities = case_df['activity'].tolist()
        for i in range(len(activities) - 1):
            G.add_edge(activities[i], activities[i + 1])

    # Create a graphviz graph
    dot = graphviz.Digraph(comment='Petri Net Style Process Map')
    dot.attr(rankdir='LR')  # Left to right layout

    # Add nodes (activities)
    for node in G.nodes():
        if node in bottlenecks:
            dot.node(node, node, shape='box', style='filled', fillcolor='red')
        else:
            dot.node(node, node, shape='box')

    # Add edges (transitions)
    for edge in G.edges():
        dot.edge(edge[0], edge[1])

    # Render the graph
    dot_data = dot.pipe(format='svg')
    return dot_data

# Function for time analysis
def perform_time_analysis(df):
    # Calculate activity durations
    df['next_timestamp'] = df.groupby('case_id')['timestamp'].shift(-1)
    df['duration'] = (df['next_timestamp'] - df['timestamp']).dt.total_seconds() / 3600  # in hours
    
    # Aggregate durations by activity
    activity_durations = df.groupby('activity').agg({
        'duration': ['mean', 'median', 'min', 'max', 'count']
    })
    activity_durations.columns = ['mean_duration', 'median_duration', 'min_duration', 'max_duration', 'frequency']
    activity_durations = activity_durations.sort_values('mean_duration', ascending=False)
    
    return activity_durations

# Function for bottleneck analysis
def perform_bottleneck_analysis(df, activity_durations):
    # Identify bottlenecks based on mean duration and frequency
    bottlenecks = activity_durations[
        (activity_durations['mean_duration'] > activity_durations['mean_duration'].mean()) &
        (activity_durations['frequency'] > activity_durations['frequency'].mean())
    ]
    
    return bottlenecks

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
    cases = df.groupby('case_id').agg({
        'activity': lambda x: '->'.join(x),
        'timestamp': ['min', 'max']
    })
    cases.columns = ['sequence', 'start_time', 'end_time']
    cases['duration'] = (cases['end_time'] - cases['start_time']).dt.total_seconds() / 86400
    cases['target'] = cases['sequence'].str.contains(target_activity).astype(int)

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
            st.success("Data loaded successfully!")
            st.write(df.head())
            
            # Column selection
            st.subheader("Select Columns")
            case_id_col = st.selectbox("Select Case ID column", df.columns)
            activity_col = st.selectbox("Select Activity column", df.columns)
            timestamp_col = st.selectbox("Select Timestamp column", df.columns)
            
            if st.button("Process Data"):
                df = preprocess_data(df, case_id_col, activity_col, timestamp_col)
                st.session_state['data'] = df
                st.success("Data processed successfully!")
                
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_pagination()
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
                grid_options = gb.build()
                
                AgGrid(df, gridOptions=grid_options, enable_enterprise_modules=True)
                
                st.markdown(export_data(df), unsafe_allow_html=True)

elif page == "Process Map":
    st.title("Process Maps")
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Network Chart", "Petri Net Style"])
        
        with tab1:
            fig = create_enhanced_process_map(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Perform bottleneck analysis
            activity_durations = perform_time_analysis(df)
            bottlenecks = perform_bottleneck_analysis(df, activity_durations)
            
            # Create and display Petri net style map
            petri_net_svg = create_petri_net_map(df, bottlenecks.index)
            st.image(petri_net_svg)
        
        # Interactive filtering
        st.subheader("Filter Data")
        start_date = st.date_input("Start Date", df['timestamp'].min().date())
        end_date = st.date_input("End Date", df['timestamp'].max().date())
        activities = st.multiselect("Select Activities", df['activity'].unique())
        
        filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
        if activities:
            filtered_df = filtered_df[filtered_df['activity'].isin(activities)]
        
        st.write(f"Filtered data contains {filtered_df['case_id'].nunique()} cases and {len(filtered_df)} events.")
        
        if st.button("Update Process Maps"):
            tab1, tab2 = st.tabs(["Network Chart", "Petri Net Style"])
            
            with tab1:
                fig = create_enhanced_process_map(filtered_df)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                activity_durations = perform_time_analysis(filtered_df)
                bottlenecks = perform_bottleneck_analysis(filtered_df, activity_durations)
                petri_net_svg = create_petri_net_map(filtered_df, bottlenecks.index)
                st.image(petri_net_svg)
    else:
        st.warning("Please upload and process data first")

elif page == "Time Analysis":
    st.title("Time Analysis")
    if 'data' in st.session_state:
        df = st.session_state['data']
        activity_durations = perform_time_analysis(df)
        
        st.subheader("Activity Durations")
        st.write(activity_durations)
        
        # Visualize mean durations
        fig = px.bar(activity_durations, x=activity_durations.index, y='mean_duration',
                     labels={'mean_duration': 'Mean Duration (hours)', 'index': 'Activity'},
                     title="Mean Duration by Activity")
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualize frequency
        fig = px.bar(activity_durations, x=activity_durations.index, y='frequency',
                     labels={'frequency': 'Frequency', 'index': 'Activity'},
                     title="Activity Frequency")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Please upload and process data first")

elif page == "Bottleneck Analysis":
    st.title("Bottleneck Analysis")
    if 'data' in st.session_state:
        df = st.session_state['data']
        activity_durations = perform_time_analysis(df)
        bottlenecks = perform_bottleneck_analysis(df, activity_durations)
        
        st.subheader("Identified Bottlenecks")
        st.write(bottlenecks)
        
        # Visualize bottlenecks
        fig = px.scatter(activity_durations, x='frequency', y='mean_duration',
                         text=activity_durations.index,
                         labels={'mean_duration': 'Mean Duration (hours)', 'frequency': 'Frequency'},
                         title="Bottleneck Analysis")
        fig.add_shape(type="rect",
                      x0=activity_durations['frequency'].mean(), y0=activity_durations['mean_duration'].mean(),
                      x1=activity_durations['frequency'].max(), y1=activity_durations['mean_duration'].max(),
                      line=dict(color="Red", width=2, dash="dash"))
        fig.add_annotation(x=activity_durations['frequency'].max(), y=activity_durations['mean_duration'].max(),
                           text="Bottleneck Zone", showarrow=False, yshift=10)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Please upload and process data first")

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
        st.warning("Please upload and process data first")

elif page == "Statistics":
    st.title("Process Statistics")
    if 'data' in st.session_state:
        df = st.session_state['data']
        stats = calculate_statistics(df)
        
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
        st.warning("Please upload and process data first")

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
        st.warning("Please upload and process data first")
