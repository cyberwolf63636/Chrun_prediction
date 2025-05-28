import os
import streamlit as st
import pandas as pd
from auth import authenticate_user, login_form, signup_form, logout
from churn_prediction import predict_churn, train_model, evaluate_model
from data_visualizations import (
    plot_churn_distribution, 
    plot_feature_importance, 
    plot_churn_by_categorical, 
    plot_numerical_features,
    plot_correlation_heatmap
)
from ollama_integration import analyze_with_gemini
from utils import load_data, preprocess_data
from prediction_history import display_prediction_history
from database import init_database, save_customer_segments
from export_utils import add_export_section
from gtts import gTTS
import base64
import tempfile
from io import BytesIO
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from datetime import datetime

# Set page configuration with modern theme (must be first Streamlit command)
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for modern, glossy effects
st.markdown("""
<style>
    /* Main container styling */
    .main {
         background: linear-gradient(135deg, #04080f 0%, #000000 100%);
        color: #8ae028;
    }
    
    /* Card-like containers */
    .stApp {
        background: rgba(10, 25, 47, 0.9);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(10, 25, 47, 0.95) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(100, 255, 218, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #0a192f, #112240);
        color: #64ffda;
        border: 1px solid #64ffda;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(100, 255, 218, 0.2);
        background: linear-gradient(45deg, #112240, #0a192f);
    }
    
    /* Selectbox and input fields */
    .stSelectbox, .stTextInput, .stNumberInput {
        background: rgba(10, 25, 47, 0.8);
        border-radius: 8px;
        border: 1px solid #64ffda;
        color: #ffffff;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(10, 25, 47, 0.8);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        transition: all 0.3s ease;
        color: #64ffda;
        border: 1px solid #64ffda;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(100, 255, 218, 0.1);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(10, 25, 47, 0.8);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #64ffda;
    }
    
    /* Metric cards */
    .stMetric {
        background: rgba(10, 25, 47, 0.8);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid #64ffda;
        color: #ffffff;
    }
    
    /* Success and error messages */
    .stAlert {
        border-radius: 8px;
        backdrop-filter: blur(5px);
        border: 1px solid #64ffda;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(10, 25, 47, 0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #0a192f, #64ffda);
        border-radius: 4px;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #64ffda;
    }
    
    p, div {
        color: #ffffff;
    }
    
    /* Markdown styling */
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Selectbox options */
    .stSelectbox > div > div {
        background-color: #0a192f;
        color: #ffffff;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #ffffff;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: #64ffda;
    }
    
    /* Radio button styling */
    .stRadio > label {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_database()

# Initialize Gemini API
GEMINI_API_KEY = "AIzaSyDknTUXRYblk3LsAO-EGqcJm_i_Xq7zJPU"
genai.configure(api_key=GEMINI_API_KEY)

# Configure safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'ollama_analysis' not in st.session_state:
    st.session_state.ollama_analysis = None
# New features session state initialization
if 'segments' not in st.session_state:
    st.session_state.segments = None
if 'speech_summary' not in st.session_state:
    st.session_state.speech_summary = None
if 'demographic_data' not in st.session_state:
    st.session_state.demographic_data = None

# Authentication section with modern styling
if not st.session_state.authenticated:
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(10, 25, 47, 0.9); 
                border-radius: 15px; backdrop-filter: blur(10px); margin-bottom: 2rem;
                border: 1px solid #64ffda;'>
        <h1 style='color: #64ffda; margin-bottom: 1rem;'>Customer Churn Prediction</h1>
        <p style='color: #ffffff;'>Welcome to the Customer Churn Prediction Application. Please log in or sign up to continue.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        login_form()
    
    with tab2:
        signup_form()
else:
    # Main Application
    # Create a header with user profile
    col1, col2 = st.columns([1, 5])
    
    with col1:
        # Create a circular profile with the first letter of the email
        profile_letter = st.session_state.user_email[0].upper()
        profile_html = f"""
        <div style="width: 50px; height: 50px; border-radius: 50%; background-color: #4f8bf9; 
                    color: white; display: flex; align-items: center; justify-content: center; 
                    font-size: 20px; font-weight: bold;">
            {profile_letter}
        </div>
        """
        st.markdown(profile_html, unsafe_allow_html=True)
    
    with col2:
        st.title("Customer Churn Prediction Dashboard")
        st.write(f"Welcome, {st.session_state.user_email}!")
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", [
            "Data Upload",
            "Predictions",
            "Visualizations",
            "Customer Segmentation",
            "Analysis with AI",
            "Speech Summary",
            "Prediction History",
            "About"
        ])
        
        if st.button("Logout"):
            logout()
            st.rerun()
    
    # Data Upload Page
    if page == "Data Upload":
        st.header("Data Upload and Preprocessing")
        
        st.write("""
        Upload your customer data in CSV format. The file should contain customer information 
        and a target column indicating whether they churned (1) or not (0).
        """)
        
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        use_sample = st.checkbox("Use sample data instead")
        
        if use_sample:
            st.session_state.data = pd.read_csv("sample_data/telecom_customer_churn.csv")
            st.success("Sample data loaded successfully!")
            
            # Display sample data
            st.subheader("Sample Data Preview")
            st.dataframe(st.session_state.data.head())
            
            # Data preprocessing
            if st.button("Preprocess Data"):
                st.session_state.preprocessed_data, st.session_state.features = preprocess_data(st.session_state.data)
                st.success("Data preprocessing completed successfully!")
                st.dataframe(st.session_state.preprocessed_data.head())
        
        elif uploaded_file is not None:
            try:
                st.session_state.data = load_data(uploaded_file)
                st.success("Data loaded successfully!")
                
                # Display uploaded data
                st.subheader("Data Preview")
                st.dataframe(st.session_state.data.head())
                
                # Data preprocessing
                if st.button("Preprocess Data"):
                    st.session_state.preprocessed_data, st.session_state.features = preprocess_data(st.session_state.data)
                    st.success("Data preprocessing completed successfully!")
                    st.dataframe(st.session_state.preprocessed_data.head())
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Predictions Page
    elif page == "Predictions":
        st.header("Churn Predictions")
        
        if st.session_state.preprocessed_data is None:
            st.warning("Please upload and preprocess your data first.")
        else:
            st.write("Train a machine learning model to predict customer churn.")
            
            # Model training section
            st.subheader("Model Training")
            test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                model_type = st.selectbox("Select Model", 
                                          ["RandomForest", "LogisticRegression", "GradientBoosting"])
            with col2:
                target_column = st.selectbox("Target Column", 
                                            [col for col in st.session_state.data.columns],
                                            index=st.session_state.data.columns.get_loc("Churn") if "Churn" in st.session_state.data.columns else 0)
            with col3:
                random_state = st.number_input("Random State", 0, 100, 42)
            
            # Add option to save to database
            save_to_db = st.checkbox("Save predictions to database", value=True)
                
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    st.session_state.model, st.session_state.predictions, st.session_state.metrics = train_model(
                        st.session_state.preprocessed_data, 
                        st.session_state.features,
                        target_column,
                        model_type, 
                        test_size, 
                        random_state,
                        save_to_db=save_to_db,
                        user_email=st.session_state.user_email
                    )
                st.success("Model trained successfully!")
            
            # Display metrics if model is trained
            if st.session_state.metrics is not None:
                st.subheader("Model Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{st.session_state.metrics['accuracy']:.2f}")
                col2.metric("Precision", f"{st.session_state.metrics['precision']:.2f}")
                col3.metric("Recall", f"{st.session_state.metrics['recall']:.2f}")
                col4.metric("F1 Score", f"{st.session_state.metrics['f1']:.2f}")
                
                st.subheader("Confusion Matrix")
                st.write(st.session_state.metrics['confusion_matrix'])
            
            # Display predictions
            if st.session_state.predictions is not None:
                st.subheader("Predictions Preview")
                predictions_df = pd.DataFrame({
                    'Actual': st.session_state.predictions['y_test'],
                    'Predicted': st.session_state.predictions['y_pred'],
                    'Probability': [p[1] for p in st.session_state.predictions['y_prob']]
                })
                st.dataframe(predictions_df)
                
                # Feature importance
                feature_importance_fig = None
                if 'feature_importance' in st.session_state.predictions:
                    st.subheader("Feature Importance")
                    feature_importance_fig = plot_feature_importance(
                        st.session_state.predictions['feature_importance'],
                        st.session_state.features
                    )
                
                # Add export section
                st.markdown("---")
                add_export_section(
                    predictions_df,
                    title="Export Predictions",
                    description="Export your prediction results in CSV or PDF format for further analysis or reporting.",
                    figures=[feature_importance_fig] if feature_importance_fig is not None else None
                )
    
    # Visualizations Page
    elif page == "Visualizations":
        st.header("Data Visualizations")
        
        if st.session_state.data is None:
            st.warning("Please upload data first.")
        else:
            # Tab-based visualizations
            viz_tabs = st.tabs(["Churn Distribution", "Categorical Analysis", "Numerical Analysis", "Correlation Analysis"])
            
            with viz_tabs[0]:
                # Churn distribution
                st.subheader("Churn Distribution")
                target_column = st.selectbox("Target Column for Distribution", 
                                            [col for col in st.session_state.data.columns],
                                            index=st.session_state.data.columns.get_loc("Churn") if "Churn" in st.session_state.data.columns else 0)
                plot_churn_distribution(st.session_state.data, target_column)
            
            with viz_tabs[1]:
                # Categorical features analysis
                st.subheader("Categorical Features Analysis")
                categorical_columns = st.multiselect(
                    "Select categorical columns",
                    [col for col in st.session_state.data.select_dtypes(include=['object', 'category']).columns],
                    default=[col for col in st.session_state.data.select_dtypes(include=['object', 'category']).columns][:min(3, len(st.session_state.data.select_dtypes(include=['object', 'category']).columns))]
                )
                target_column = st.selectbox("Target Column for Categorical Analysis", 
                                            [col for col in st.session_state.data.columns],
                                            index=st.session_state.data.columns.get_loc("Churn") if "Churn" in st.session_state.data.columns else 0)
                
                if categorical_columns:
                    plot_churn_by_categorical(st.session_state.data, categorical_columns, target_column)
                else:
                    st.info("Please select at least one categorical column.")
            
            with viz_tabs[2]:
                # Numerical features analysis
                st.subheader("Numerical Features Analysis")
                numerical_columns = st.multiselect(
                    "Select numerical columns",
                    [col for col in st.session_state.data.select_dtypes(include=['int64', 'float64']).columns if col != "Churn"],
                    default=[col for col in st.session_state.data.select_dtypes(include=['int64', 'float64']).columns if col != "Churn"][:min(3, len(st.session_state.data.select_dtypes(include=['int64', 'float64']).columns))]
                )
                target_column = st.selectbox("Target Column for Numerical Analysis", 
                                            [col for col in st.session_state.data.columns],
                                            index=st.session_state.data.columns.get_loc("Churn") if "Churn" in st.session_state.data.columns else 0)
                
                if numerical_columns:
                    plot_numerical_features(st.session_state.data, numerical_columns, target_column)
                else:
                    st.info("Please select at least one numerical column.")
            
            with viz_tabs[3]:
                # Correlation heatmap
                st.subheader("Correlation Analysis")
                
                if st.button("Show Correlation Heatmap"):
                    plot_correlation_heatmap(st.session_state.data)
    
    # Customer Segmentation Page
    elif page == "Customer Segmentation":
        st.header("Customer Segmentation")
        
        if st.session_state.data is None:
            st.warning("Please upload data first.")
        else:
            st.write("""
            This section helps you segment your customers into different groups based on their characteristics.
            Customer segmentation can help you target specific groups with tailored retention strategies.
            """)
            
            segmentation_tabs = st.tabs(["K-Means Clustering", "RFM Analysis", "Manual Segmentation"])
            
            with segmentation_tabs[0]:
                st.subheader("K-Means Clustering")
                
                # Select features for clustering
                cluster_features = st.multiselect(
                    "Select features for clustering:",
                    [col for col in st.session_state.data.select_dtypes(include=['int64', 'float64']).columns],
                    default=[col for col in st.session_state.data.select_dtypes(include=['int64', 'float64']).columns][:min(3, len(st.session_state.data.select_dtypes(include=['int64', 'float64']).columns))]
                )
                
                # Number of clusters
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                
                if st.button("Run K-Means Clustering") and cluster_features:
                    try:
                        from sklearn.cluster import KMeans
                        from sklearn.preprocessing import StandardScaler
                        import numpy as np
                        import pandas as pd
                        import plotly.express as px
                        
                        # Handle data types - check and convert non-numeric columns
                        for col in cluster_features:
                            if st.session_state.data[col].dtype == 'object':
                                st.warning(f"Converting non-numeric column '{col}' to numeric. This may lead to unexpected results.")
                                try:
                                    # For binary Yes/No columns
                                    if set(st.session_state.data[col].unique()) == {"Yes", "No"} or set(st.session_state.data[col].unique()) == {0, 1}:
                                        st.session_state.data[col] = st.session_state.data[col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
                                    else:
                                        # For other categorical columns, try to convert to numeric or use dummy encoding
                                        st.session_state.data[col] = pd.to_numeric(st.session_state.data[col], errors='coerce')
                                except Exception as e:
                                    st.error(f"Could not convert column '{col}' to numeric: {str(e)}")
                                    # Remove problematic feature
                                    cluster_features.remove(col)
                        
                        if not cluster_features:
                            st.error("No valid numeric features left for clustering.")
                            st.stop()
                            
                        # Extract features - only numeric features
                        X = st.session_state.data[cluster_features].select_dtypes(include=['number']).fillna(0)
                        
                        # Scale the features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Apply K-means
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        st.session_state.data['Cluster'] = kmeans.fit_predict(X_scaled)
                        
                        # Create segments dataframe
                        segments_df = pd.DataFrame({
                            'segment': 'Cluster ' + st.session_state.data['Cluster'].astype(str)
                        })
                        st.session_state.segments = segments_df
                        
                        # Save to database option
                        if st.checkbox("Save segments to database", value=True):
                            try:
                                save_customer_segments(
                                    st.session_state.user_email,
                                    segments_df,
                                    "KMeans"
                                )
                                st.success("Customer segments saved to database successfully!")
                            except Exception as e:
                                st.warning(f"Could not save segments to database: {str(e)}")
                        
                        # Display cluster statistics - ensure we're only using numeric columns
                        numeric_features = []
                        for col in cluster_features:
                            # Convert columns to numeric if possible
                            try:
                                if st.session_state.data[col].dtype == 'object':
                                    st.session_state.data[col] = pd.to_numeric(st.session_state.data[col], errors='coerce')
                                if st.session_state.data[col].dtype in ['int64', 'float64']:
                                    numeric_features.append(col)
                            except Exception as e:
                                st.warning(f"Could not convert column {col} to numeric: {str(e)}")
                        
                        # Handle Churn column - ensure it's numeric for aggregation
                        if 'Churn' in st.session_state.data.columns:
                            if st.session_state.data['Churn'].dtype == 'object':
                                # Convert Churn to numeric if it's an object type
                                churn_mapping = {'Yes': 1, 'No': 0, True: 1, False: 0}
                                st.session_state.data['Churn'] = st.session_state.data['Churn'].map(lambda x: churn_mapping.get(x, x))
                            
                            # Perform aggregation with numeric columns only
                            cluster_stats = st.session_state.data.groupby('Cluster').agg({
                                **{col: 'mean' for col in numeric_features},
                                'Churn': 'mean',
                                'Cluster': 'count'
                            })
                        else:
                            # No Churn column
                            cluster_stats = st.session_state.data.groupby('Cluster').agg({
                                **{col: 'mean' for col in numeric_features},
                                'Cluster': 'count'
                            })
                        
                        st.subheader("Cluster Statistics")
                        st.dataframe(cluster_stats)
                        
                        # Visualize clusters
                        if len(cluster_features) >= 2:
                            fig = px.scatter(
                                st.session_state.data, 
                                x=cluster_features[0], 
                                y=cluster_features[1],
                                color='Cluster',
                                title=f'Clusters based on {cluster_features[0]} and {cluster_features[1]}',
                                hover_data=cluster_features + (['Churn'] if 'Churn' in st.session_state.data.columns else [])
                            )
                            st.plotly_chart(fig)
                        
                        # Pie chart of cluster distribution
                        fig = px.pie(
                            st.session_state.data, 
                            names='Cluster', 
                            title='Customer Distribution Across Clusters'
                        )
                        st.plotly_chart(fig)
                        
                        # Show churn by cluster if available
                        if 'Churn' in st.session_state.data.columns:
                            fig = px.bar(
                                st.session_state.data.groupby('Cluster')['Churn'].mean().reset_index(), 
                                x='Cluster', 
                                y='Churn',
                                title='Churn Rate by Cluster',
                                labels={'Churn': 'Churn Rate'}
                            )
                            st.plotly_chart(fig)
                        
                        # Add export option for segmented data
                        st.markdown("---")
                        cluster_export_df = st.session_state.data[['Cluster']].join(
                            st.session_state.data.select_dtypes(include=['number']).drop('Cluster', axis=1, errors='ignore')
                        )
                        add_export_section(
                            cluster_export_df,
                            title="Export Cluster Data",
                            description="Export your customer segments in CSV or PDF format for further analysis or reporting."
                        )
                    
                    except Exception as e:
                        st.error(f"Error in clustering: {str(e)}")
                
            with segmentation_tabs[1]:
                st.subheader("RFM Analysis")
                st.info("RFM (Recency, Frequency, Monetary) analysis requires transaction data with customer ID, transaction date, and purchase amount.")
                
                if 'customerID' in st.session_state.data.columns:
                    # Check if we have required columns for RFM
                    has_monetary = any(col in st.session_state.data.columns for col in ['MonthlyCharges', 'TotalCharges'])
                    
                    if has_monetary and 'tenure' in st.session_state.data.columns:
                        st.write("Your data contains some elements needed for RFM analysis. Let's proceed with a simplified version.")
                        
                        # Create RFM segments
                        try:
                            # Recency (inverse of tenure)
                            recency_col = 'tenure'
                            max_tenure = st.session_state.data[recency_col].max()
                            st.session_state.data['Recency_Score'] = pd.qcut(max_tenure - st.session_state.data[recency_col], 
                                                                             q=5, labels=[5, 4, 3, 2, 1])
                            
                            # Frequency (we'll use contract length as a proxy)
                            if 'Contract' in st.session_state.data.columns:
                                contract_scores = {
                                    'Month-to-month': 1,
                                    'One year': 3,
                                    'Two year': 5
                                }
                                st.session_state.data['Frequency_Score'] = st.session_state.data['Contract'].map(
                                    lambda x: contract_scores.get(x, 3))
                            else:
                                # Randomly assign frequency if no proxy is available
                                st.session_state.data['Frequency_Score'] = np.random.randint(1, 6, size=len(st.session_state.data))
                            
                            # Monetary
                            monetary_col = 'TotalCharges' if 'TotalCharges' in st.session_state.data.columns else 'MonthlyCharges'
                            st.session_state.data['Monetary_Score'] = pd.qcut(st.session_state.data[monetary_col].astype(float), 
                                                                              q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                            
                            # Calculate RFM Score
                            st.session_state.data['RFM_Score'] = st.session_state.data[['Recency_Score', 'Frequency_Score', 'Monetary_Score']].sum(axis=1)
                            
                            # Create segments
                            def segment_rfm(rfm_score):
                                if rfm_score >= 13:
                                    return 'Champions'
                                elif rfm_score >= 10:
                                    return 'Loyal Customers'
                                elif rfm_score >= 7:
                                    return 'Potential Loyalists'
                                elif rfm_score >= 5:
                                    return 'At Risk'
                                else:
                                    return 'Need Attention'
                            
                            st.session_state.data['RFM_Segment'] = st.session_state.data['RFM_Score'].apply(segment_rfm)
                            
                            # Display RFM segments
                            st.subheader("RFM Segments")
                            segment_counts = st.session_state.data['RFM_Segment'].value_counts().reset_index()
                            segment_counts.columns = ['Segment', 'Count']
                            
                            # Pie chart of segments
                            fig = px.pie(
                                segment_counts, 
                                values='Count', 
                                names='Segment', 
                                title='Customer Distribution Across RFM Segments'
                            )
                            st.plotly_chart(fig)
                            
                            # Segment statistics
                            segment_stats = st.session_state.data.groupby('RFM_Segment').agg({
                                recency_col: 'mean',
                                monetary_col: 'mean',
                                'RFM_Score': 'mean'
                            })
                            
                            if 'Churn' in st.session_state.data.columns:
                                segment_stats['Churn_Rate'] = st.session_state.data.groupby('RFM_Segment')['Churn'].mean()
                            
                            st.dataframe(segment_stats)
                            
                            # Bar chart of churn by segment if available
                            if 'Churn' in st.session_state.data.columns:
                                fig = px.bar(
                                    st.session_state.data.groupby('RFM_Segment')['Churn'].mean().reset_index(), 
                                    x='RFM_Segment', 
                                    y='Churn',
                                    title='Churn Rate by RFM Segment',
                                    labels={'Churn': 'Churn Rate', 'RFM_Segment': 'Segment'}
                                )
                                st.plotly_chart(fig)
                            
                            # Add export option for RFM segmented data
                            st.markdown("---")
                            rfm_export_df = st.session_state.data[['RFM_Segment', 'RFM_Score', 'Recency_Score', 'Frequency_Score', 'Monetary_Score']].join(
                                st.session_state.data.select_dtypes(include=['number']).drop(['RFM_Score', 'Recency_Score', 'Frequency_Score', 'Monetary_Score'], axis=1, errors='ignore')
                            )
                            add_export_section(
                                rfm_export_df,
                                title="Export RFM Segment Data",
                                description="Export your RFM customer segments in CSV or PDF format for further analysis or reporting."
                            )
                            
                        except Exception as e:
                            st.error(f"Error in RFM analysis: {str(e)}")
                    else:
                        st.warning("Your data doesn't contain all required columns for RFM analysis (tenure and charges).")
                else:
                    st.warning("Your data doesn't have a customer ID column, which is required for RFM analysis.")
            
            with segmentation_tabs[2]:
                st.subheader("Manual Segmentation")
                st.write("Create custom segments by defining rules on your data.")
                
                # Select column for segmentation
                seg_column = st.selectbox(
                    "Select column for segmentation:",
                    [col for col in st.session_state.data.columns],
                    index=0
                )
                
                # Get column type to determine segmentation method
                if st.session_state.data[seg_column].dtype in ['int64', 'float64']:
                    # Numerical column
                    seg_method = st.radio("Segmentation method:", ["Ranges", "Quantiles"])
                    
                    if seg_method == "Ranges":
                        min_val = float(st.session_state.data[seg_column].min())
                        max_val = float(st.session_state.data[seg_column].max())
                        
                        # Let user define ranges
                        seg_ranges = st.slider(
                            f"Define ranges for {seg_column}:",
                            min_val, max_val, (min_val, max_val / 2, max_val)
                        )
                        
                        # Define segment labels
                        if len(seg_ranges) == 3:
                            seg_labels = [
                                f"Low {seg_column} (< {seg_ranges[1]:.2f})",
                                f"High {seg_column} (>= {seg_ranges[1]:.2f})"
                            ]
                        elif len(seg_ranges) == 4:
                            seg_labels = [
                                f"Low {seg_column} (< {seg_ranges[1]:.2f})",
                                f"Medium {seg_column} ({seg_ranges[1]:.2f} - {seg_ranges[2]:.2f})",
                                f"High {seg_column} (>= {seg_ranges[2]:.2f})"
                            ]
                        
                    else:  # Quantiles
                        num_quantiles = st.slider("Number of quantiles:", 2, 5, 3)
                        seg_labels = [f"{seg_column} Q{i+1}" for i in range(num_quantiles)]
                
                else:
                    # Categorical column
                    unique_values = st.session_state.data[seg_column].unique()
                    selected_values = st.multiselect(
                        f"Select values to include from {seg_column}:",
                        unique_values,
                        default=list(unique_values)[:min(5, len(unique_values))]
                    )
                    seg_labels = [f"{seg_column}: {val}" for val in selected_values]
                
                if st.button("Create Segments"):
                    try:
                        if st.session_state.data[seg_column].dtype in ['int64', 'float64']:
                            if seg_method == "Ranges":
                                if len(seg_ranges) == 3:
                                    conditions = [
                                        st.session_state.data[seg_column] < seg_ranges[1],
                                        st.session_state.data[seg_column] >= seg_ranges[1]
                                    ]
                                elif len(seg_ranges) == 4:
                                    conditions = [
                                        st.session_state.data[seg_column] < seg_ranges[1],
                                        (st.session_state.data[seg_column] >= seg_ranges[1]) & (st.session_state.data[seg_column] < seg_ranges[2]),
                                        st.session_state.data[seg_column] >= seg_ranges[2]
                                    ]
                                
                                st.session_state.data['Manual_Segment'] = np.select(conditions, seg_labels, default='Other')
                            
                            else:  # Quantiles
                                st.session_state.data['Manual_Segment'] = pd.qcut(
                                    st.session_state.data[seg_column], 
                                    q=num_quantiles, 
                                    labels=seg_labels
                                )
                        
                        else:
                            # Categorical column
                            st.session_state.data['Manual_Segment'] = st.session_state.data[seg_column].apply(
                                lambda x: f"{seg_column}: {x}" if x in selected_values else "Other"
                            )
                        
                        # Display segment distribution
                        seg_counts = st.session_state.data['Manual_Segment'].value_counts().reset_index()
                        seg_counts.columns = ['Segment', 'Count']
                        
                        # Pie chart of segments
                        fig = px.pie(
                            seg_counts, 
                            values='Count', 
                            names='Segment', 
                            title='Customer Distribution Across Manual Segments'
                        )
                        st.plotly_chart(fig)
                        
                        # Segment statistics
                        num_cols = [col for col in st.session_state.data.select_dtypes(include=['int64', 'float64']).columns 
                                 if col not in ['Manual_Segment']]
                        
                        if num_cols:
                            selected_stats = st.multiselect(
                                "Select columns for segment statistics:",
                                num_cols,
                                default=num_cols[:min(3, len(num_cols))]
                            )
                            
                            if selected_stats:
                                seg_stats = st.session_state.data.groupby('Manual_Segment')[selected_stats].mean()
                                st.dataframe(seg_stats)
                                
                                # Bar chart of churn by segment if available
                                if 'Churn' in selected_stats:
                                    fig = px.bar(
                                        seg_stats.reset_index(), 
                                        x='Manual_Segment', 
                                        y='Churn',
                                        title='Churn Rate by Manual Segment',
                                        labels={'Churn': 'Churn Rate', 'Manual_Segment': 'Segment'}
                                    )
                                    st.plotly_chart(fig)
                                
                                # Add export option for manually segmented data
                                st.markdown("---")
                                manual_seg_export_df = st.session_state.data[['Manual_Segment']].join(
                                    st.session_state.data.select_dtypes(include=['number']).drop('Manual_Segment', axis=1, errors='ignore')
                                )
                                add_export_section(
                                    manual_seg_export_df,
                                    title="Export Segmented Data",
                                    description="Export your manually segmented customer data in CSV or PDF format for further analysis or reporting."
                                )
                    
                    except Exception as e:
                        st.error(f"Error in manual segmentation: {str(e)}")

    # Analysis with AI Page
    elif page == "Analysis with AI":
        st.header("Analysis with Wolf AI")
        
        if st.session_state.data is None:
            st.warning("Please upload data first.")
        else:
            st.write("""
            This section uses the Wolf API with the Wolf:7b model to analyze your customer data 
            and provide insights on churn patterns. You can ask specific questions about your data or 
            get general recommendations to reduce churn.
            """)
            
            # Sample questions
            st.subheader("Sample Questions")
            sample_questions = [
                "What are the main factors contributing to customer churn in this dataset?",
                "How can we reduce customer churn based on this data?",
                "What customer segments are most likely to churn?",
                "What patterns do you see in the customer behavior before churning?",
                "Summarize the key insights from this churn data."
            ]
            
            selected_question = st.selectbox("Select a question or type your own below:", 
                                           [""] + sample_questions)
            
            # Custom question input
            custom_question = st.text_area("Or type your own question:", 
                                         value=selected_question if selected_question else "",
                                         height=100)
            
            if st.button("Analyze with Wolf AI") and custom_question:
                with st.spinner("Analyzing with Wolf Wolf:7b..."):
                    # Get data summary for context
                    data_sample = st.session_state.data.head(10)
                    data_info = {
                        "shape": st.session_state.data.shape,
                        "columns": list(st.session_state.data.columns),
                        "dtypes": {col: str(dtype) for col, dtype in st.session_state.data.dtypes.items()},
                        "missing_values": st.session_state.data.isnull().sum().to_dict(),
                        "summary": st.session_state.data.describe().to_dict()
                    }
                    
                    # If predictions exist, include some metrics
                    if st.session_state.metrics is not None:
                        data_info["model_metrics"] = st.session_state.metrics
                    
                    # Get AI analysis
                    st.session_state.ollama_analysis = analyze_with_gemini(
                        custom_question, 
                        data_info,
                        data_sample
                    )
                
                # Display the analysis
                st.subheader("Ollama Analysis")
                st.markdown(st.session_state.ollama_analysis)
                
                # Initialize speech states if not exists
                if 'speech_playing' not in st.session_state:
                    st.session_state.speech_playing = False
                if 'audio_data' not in st.session_state:
                    st.session_state.audio_data = None
                if 'speech_settings' not in st.session_state:
                    st.session_state.speech_settings = {
                        'voice_type': "Natural Female",
                        'speed': 1.0,
                        'emotion': "Neutral"
                    }
                if 'auto_generate' not in st.session_state:
                    st.session_state.auto_generate = True
                
                # Add speech controls
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Auto-generation toggle
                    auto_generate = st.checkbox(
                        "Auto-generate speech on analysis",
                        value=st.session_state.auto_generate,
                        key="auto_generate_check"
                    )
                    st.session_state.auto_generate = auto_generate
                    
                    # Voice selection
                    voice_type = st.selectbox(
                        "Select Voice",
                        [
                            "Natural Female",
                            "Natural Male",
                            "Professional Female",
                            "Professional Male",
                            "Casual Female",
                            "Casual Male"
                        ],
                        index=0,
                        key="voice_select"
                    )
                    
                    # Speech settings
                    speed = st.slider(
                        "Speech Speed",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        key="speed_slider"
                    )
                    
                    # Emotion selection
                    emotion = st.selectbox(
                        "Speech Emotion",
                        ["Neutral", "Happy", "Serious", "Empathetic", "Confident"],
                        index=0,
                        key="emotion_select"
                    )
                    
                    # Update settings in session state
                    st.session_state.speech_settings = {
                        'voice_type': voice_type,
                        'speed': speed,
                        'emotion': emotion
                    }
                
                with col2:
                    # Auto-generate speech if enabled
                    if st.session_state.auto_generate and not st.session_state.speech_playing:
                        with st.spinner("Auto-generating speech..."):
                            try:
                                # Create a BytesIO object to store the audio
                                audio_buffer = BytesIO()
                                
                                # Generate speech using gTTS with selected parameters
                                tts = gTTS(
                                    text=st.session_state.ollama_analysis,
                                    lang='en',
                                    slow=False
                                )
                                
                                # Save the audio to the buffer
                                tts.write_to_fp(audio_buffer)
                                audio_buffer.seek(0)
                                
                                # Store audio data in session state
                                st.session_state.audio_data = audio_buffer.getvalue()
                                st.session_state.speech_playing = True
                                st.success("Speech auto-generated successfully!")
                                
                                # Auto-download the MP3 file
                                st.download_button(
                                    label="💾 Download Auto-generated MP3",
                                    data=st.session_state.audio_data,
                                    file_name=f"ollama_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                                    mime="audio/mp3"
                                )
                                
                            except Exception as e:
                                st.error(f"Error auto-generating speech: {str(e)}")
                                st.session_state.speech_playing = False
                    
                    # Manual speech control buttons
                    col_start, col_stop = st.columns(2)
                    
                    with col_start:
                        if not st.session_state.speech_playing:
                            if st.button("▶️ Start Speech", key="start_speech", use_container_width=True):
                                with st.spinner("Generating speech..."):
                                    try:
                                        # Create a BytesIO object to store the audio
                                        audio_buffer = BytesIO()
                                        
                                        # Generate speech using gTTS with selected parameters
                                        tts = gTTS(
                                            text=st.session_state.ollama_analysis,
                                            lang='en',
                                            slow=False
                                        )
                                        
                                        # Save the audio to the buffer
                                        tts.write_to_fp(audio_buffer)
                                        audio_buffer.seek(0)
                                        
                                        # Store audio data in session state
                                        st.session_state.audio_data = audio_buffer.getvalue()
                                        st.session_state.speech_playing = True
                                        st.success("Speech generated successfully!")
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"Error generating speech: {str(e)}")
                                        st.session_state.speech_playing = False
                    
                    with col_stop:
                        if st.session_state.speech_playing:
                            if st.button("⏹️ Stop Speech", key="stop_speech", use_container_width=True):
                                st.session_state.speech_playing = False
                                st.session_state.audio_data = None
                                st.info("Speech stopped")
                                st.rerun()
                    
                    # Display audio player if speech is playing
                    if st.session_state.speech_playing and st.session_state.audio_data:
                        audio_base64 = base64.b64encode(st.session_state.audio_data).decode('utf-8')
                        
                        # Create a styled audio player container with live controls
                        st.markdown("""
                        <div style='background: rgba(10, 25, 47, 0.8); padding: 1.5rem; border-radius: 10px; 
                                    border: 1px solid #64ffda; margin-top: 1rem;'>
                            <h4 style='color: #64ffda; margin-bottom: 1rem;'>Live Speech Player</h4>
                            <div style='color: #ffffff; margin-bottom: 1rem;'>
                                Voice: {} | Speed: {}x | Emotion: {}
                            </div>
                            <audio controls autoplay style='width: 100%;'>
                                <source src="data:audio/mp3;base64,{}" type="audio/mp3">
                                Your browser does not support the audio element.
                            </audio>
                        </div>
                        """.format(
                            st.session_state.speech_settings['voice_type'],
                            st.session_state.speech_settings['speed'],
                            st.session_state.speech_settings['emotion'],
                            audio_base64
                        ), unsafe_allow_html=True)
                
                # Add a disclaimer
                st.caption("Note: The analysis is based on the Ollama qwen2.5-coder:7b model and should be reviewed by a human expert.")
    
    # Speech Summary Page
    elif page == "Speech Summary":
        st.header("Speech Summary")
        
        if st.session_state.data is None:
            st.warning("Please upload data first.")
        else:
            st.markdown("""
            <div style='background: rgba(10, 25, 47, 0.9); padding: 1.5rem; border-radius: 10px; 
                        border: 1px solid #64ffda; margin-bottom: 2rem;'>
                <h3 style='color: #64ffda;'>AI-Powered Speech Synthesis</h3>
                <p style='color: #ffffff;'>Generate natural-sounding speech summaries using advanced text-to-speech technology. 
                Create comprehensive reports and share insights with your team through audio format.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create three columns for layout
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Enhanced summary type selection
                summary_type = st.radio(
                    "Select Summary Type",
                    [
                        "Comprehensive Analysis",
                        "Executive Summary",
                        "Technical Deep Dive",
                        "Action Plan",
                        "Customer Insights",
                        "Model Performance",
                        "Custom Report"
                    ],
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                # Generate enhanced summary text based on selection
                if summary_type == "Comprehensive Analysis":
                    try:
                        churn_rate = st.session_state.data['Churn'].mean() * 100 if 'Churn' in st.session_state.data.columns else "unknown"
                        summary_text = f"""
                        Comprehensive Customer Churn Analysis Report:
                        
                        Overview:
                        - Total customers analyzed: {st.session_state.data.shape[0]}
                        - Features considered: {st.session_state.data.shape[1]}
                        - Overall churn rate: {churn_rate if isinstance(churn_rate, str) else f"{churn_rate:.1f}%"}
                        
                        Key Findings:
                        1. Customer Demographics:
                           - Age distribution: {', '.join([f"{col}" for col in st.session_state.data.select_dtypes(include=['int64']).columns[:3]])}
                           - Geographic distribution: {', '.join([f"{col}" for col in st.session_state.data.select_dtypes(include=['object']).columns[:2]])}
                        
                        2. Service Usage Patterns:
                           - Most used services: {', '.join([f"{col}" for col in st.session_state.data.select_dtypes(include=['object']).columns[2:4]])}
                           - Average monthly charges: ${st.session_state.data['MonthlyCharges'].mean():.2f}
                        
                        3. Churn Risk Factors:
                           - Contract type impact
                           - Service quality metrics
                           - Customer support interactions
                        
                        Recommendations:
                        1. Implement targeted retention programs
                        2. Enhance customer support services
                        3. Optimize pricing strategies
                        4. Improve service quality
                        5. Develop customer loyalty programs
                        """
                    except Exception as e:
                        summary_text = "Error generating comprehensive analysis. Please check your data format."
                
                elif summary_type == "Executive Summary":
                    summary_text = """
                    Executive Summary: Customer Churn Analysis
                    
                    Key Highlights:
                    1. Current Churn Status:
                       - Overall churn rate
                       - High-risk customer segments
                       - Revenue impact
                    
                    2. Strategic Insights:
                       - Main churn drivers
                       - Customer behavior patterns
                       - Service usage trends
                    
                    3. Business Impact:
                       - Financial implications
                       - Customer lifetime value
                       - Market position
                    
                    4. Action Items:
                       - Immediate priorities
                       - Long-term strategies
                       - Resource allocation
                    """
                
                elif summary_type == "Technical Deep Dive":
                    if st.session_state.predictions is not None:
                        summary_text = f"""
                        Technical Analysis Report:
                        
                        Model Performance:
                        - Algorithm: {st.session_state.model.__class__.__name__}
                        - Accuracy: {st.session_state.metrics['accuracy']:.2f}
                        - Precision: {st.session_state.metrics['precision']:.2f}
                        - Recall: {st.session_state.metrics['recall']:.2f}
                        - F1 Score: {st.session_state.metrics['f1']:.2f}
                        
                        Feature Importance:
                        {', '.join([f"{feat}: {imp:.2f}" for feat, imp in zip(st.session_state.features, st.session_state.predictions['feature_importance'])[:5]])}
                        
                        Data Quality:
                        - Missing values analysis
                        - Feature correlation
                        - Outlier detection
                        
                        Model Insights:
                        - Key decision boundaries
                        - Prediction confidence
                        - Error analysis
                        """
                    else:
                        summary_text = "Please train a model first to generate technical insights."
                
                elif summary_type == "Action Plan":
                    summary_text = """
                    Customer Retention Action Plan:
                    
                    Immediate Actions (0-30 days):
                    1. High-Risk Customer Outreach
                       - Identify top 100 at-risk customers
                       - Implement personalized retention offers
                       - Schedule proactive support calls
                    
                    2. Service Improvements
                       - Address common service issues
                       - Enhance technical support
                       - Implement quick fixes
                    
                    Short-term Initiatives (30-90 days):
                    1. Customer Experience Enhancement
                       - Improve onboarding process
                       - Streamline support channels
                       - Optimize communication
                    
                    2. Retention Program Development
                       - Design loyalty rewards
                       - Create customer success program
                       - Implement feedback system
                    
                    Long-term Strategy (90+ days):
                    1. Product Development
                       - Address feature gaps
                       - Enhance service quality
                       - Innovate offerings
                    
                    2. Market Positioning
                       - Strengthen brand value
                       - Improve competitive edge
                       - Expand market share
                    """
                
                elif summary_type == "Customer Insights":
                    summary_text = """
                    Customer Behavior Insights:
                    
                    Customer Segments:
                    1. High-Value Retained Customers
                       - Characteristics
                       - Behavior patterns
                       - Success factors
                    
                    2. At-Risk Customers
                       - Warning signs
                       - Common patterns
                       - Intervention opportunities
                    
                    3. Churned Customers
                       - Exit reasons
                       - Service usage patterns
                       - Feedback analysis
                    
                    Behavioral Patterns:
                    1. Service Usage
                       - Peak usage times
                       - Feature preferences
                       - Support interactions
                    
                    2. Communication Preferences
                       - Channel preferences
                       - Response patterns
                       - Engagement levels
                    
                    3. Value Perception
                       - Price sensitivity
                       - Feature importance
                       - Satisfaction drivers
                    """
                
                elif summary_type == "Model Performance":
                    if st.session_state.metrics is not None:
                        summary_text = f"""
                        Model Performance Analysis:
                        
                        Model Details:
                        - Algorithm: {st.session_state.model.__class__.__name__}
                        - Training Date: {datetime.now().strftime('%Y-%m-%d')}
                        - Data Size: {st.session_state.data.shape[0]} records
                        
                        Performance Metrics:
                        - Accuracy: {st.session_state.metrics['accuracy']:.2f}
                        - Precision: {st.session_state.metrics['precision']:.2f}
                        - Recall: {st.session_state.metrics['recall']:.2f}
                        - F1 Score: {st.session_state.metrics['f1']:.2f}
                        
                        Model Strengths:
                        1. High prediction accuracy
                        2. Good balance of precision and recall
                        3. Robust feature importance
                        4. Consistent performance
                        
                        Areas for Improvement:
                        1. Feature engineering
                        2. Data quality
                        3. Model tuning
                        4. Validation strategy
                        """
                    else:
                        summary_text = "Please train a model first to view performance metrics."
                
                else:  # Custom Report
                    summary_text = st.text_area(
                        "Enter your custom report text:",
                        """
                        Custom Analysis Report:
                        
                        Executive Summary:
                        - Key findings
                        - Main recommendations
                        - Business impact
                        
                        Detailed Analysis:
                        1. Customer Behavior
                           - Usage patterns
                           - Service preferences
                           - Support interactions
                        
                        2. Churn Patterns
                           - Common triggers
                           - Warning signs
                           - Prevention opportunities
                        
                        3. Business Impact
                           - Revenue loss
                           - Customer acquisition cost
                           - Market position
                        
                        Recommendations:
                        1. Short-term actions
                        2. Medium-term initiatives
                        3. Long-term strategies
                        """,
                        height=300
                    )
                
                # Display the summary text in a styled container
                st.markdown("""
                <div style='background: rgba(10, 25, 47, 0.8); padding: 1.5rem; border-radius: 10px; 
                            border: 1px solid #64ffda; margin-top: 1rem;'>
                    <h3 style='color: #64ffda; margin-bottom: 1rem;'>Summary Text</h3>
                    <div style='color: #ffffff; white-space: pre-wrap;'>{}</div>
                </div>
                """.format(summary_text), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background: rgba(10, 25, 47, 0.8); padding: 1.5rem; border-radius: 10px; 
                            border: 1px solid #64ffda; margin-bottom: 1rem;'>
                    <h3 style='color: #64ffda; margin-bottom: 1rem;'>Speech Settings</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced voice options
                voice_type = st.selectbox(
                    "Select Voice",
                    [
                        "Natural Female",
                        "Natural Male",
                        "Professional Female",
                        "Professional Male",
                        "Casual Female",
                        "Casual Male",
                        "Technical Female",
                        "Technical Male"
                    ],
                    index=0
                )
                
                # Voice style mapping
                voice_style_map = {
                    "Natural Female": {"gender": "female", "style": "natural"},
                    "Natural Male": {"gender": "male", "style": "natural"},
                    "Professional Female": {"gender": "female", "style": "professional"},
                    "Professional Male": {"gender": "male", "style": "professional"},
                    "Casual Female": {"gender": "female", "style": "casual"},
                    "Casual Male": {"gender": "male", "style": "casual"},
                    "Technical Female": {"gender": "female", "style": "technical"},
                    "Technical Male": {"gender": "male", "style": "technical"}
                }
                
                # Speech settings
                col_speed, col_pitch = st.columns(2)
                with col_speed:
                    speed = st.slider(
                        "Speech Speed",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.1
                    )
                with col_pitch:
                    pitch = st.slider(
                        "Voice Pitch",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.1
                    )
                
                # Emotion and emphasis
                emotion = st.selectbox(
                    "Speech Emotion",
                    ["Neutral", "Happy", "Serious", "Empathetic", "Confident"],
                    index=0
                )
            
            with col3:
                st.markdown("""
                <div style='background: rgba(10, 25, 47, 0.8); padding: 1.5rem; border-radius: 10px; 
                            border: 1px solid #64ffda; margin-bottom: 1rem;'>
                    <h3 style='color: #64ffda; margin-bottom: 1rem;'>Additional Options</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional settings
                add_timestamp = st.checkbox("Add timestamp to summary", value=True)
                include_metrics = st.checkbox("Include performance metrics", value=True)
                add_recommendations = st.checkbox("Add action recommendations", value=True)
                
                # Export options
                export_format = st.selectbox(
                    "Export Format",
                    ["MP3", "WAV", "Text + Audio"],
                    index=0
                )
                
                # Generate speech button with enhanced styling
                if st.button("Generate & Play Speech", key="generate_speech"):
                    with st.spinner("Generating AI-powered speech synthesis..."):
                        try:
                            # Create a BytesIO object to store the audio
                            audio_buffer = BytesIO()
                            
                            # Prepare speech parameters
                            voice_params = voice_style_map[voice_type]
                            
                            # Generate speech using gTTS
                            tts = gTTS(
                                text=summary_text,
                                lang='en',
                                slow=False
                            )
                            
                            # Save the audio to the buffer
                            tts.write_to_fp(audio_buffer)
                            audio_buffer.seek(0)
                            
                            # Convert audio bytes to base64 for playback
                            audio_bytes = audio_buffer.getvalue()
                            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                            
                            # Create a styled audio player container
                            st.markdown("""
                            <div style='background: rgba(10, 25, 47, 0.8); padding: 1.5rem; border-radius: 10px; 
                                        border: 1px solid #64ffda; margin-top: 1rem;'>
                                <h4 style='color: #64ffda; margin-bottom: 1rem;'>Audio Player</h4>
                                <div style='color: #ffffff; margin-bottom: 1rem;'>
                                    Voice: {} | Speed: {}x | Emotion: {}
                                </div>
                                <audio controls autoplay style='width: 100%;'>
                                    <source src="data:audio/mp3;base64,{}" type="audio/mp3">
                                    Your browser does not support the audio element.
                                </audio>
                            </div>
                            """.format(
                                voice_type,
                                speed,
                                emotion,
                                audio_base64
                            ), unsafe_allow_html=True)
                            
                            # Provide a download button
                            st.download_button(
                                label="Download Speech MP3",
                                data=audio_bytes,
                                file_name=f"churn_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                                mime="audio/mp3"
                            )
                            
                            st.success("AI-powered speech generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating speech: {str(e)}")
                            st.error("If the error persists, please try again or contact support.")
                
                # Tips for Effective Speech Summaries
                with st.expander("Tips for Effective Speech Summaries", expanded=True):
                    st.markdown("""
                    <div style='color: #ffffff;'>
                        <h4>Content Structure</h4>
                        - Start with key findings
                        - Present data-driven insights
                        - Include actionable recommendations
                        - End with next steps
                        
                        <h4>Best Practices</h4>
                        - Keep language clear and concise
                        - Use bullet points for clarity
                        - Include specific metrics
                        - Focus on actionable insights
                        
                        <h4>Voice Selection</h4>
                        - Use professional voice for formal reports
                        - Choose natural voice for general summaries
                        - Match voice style to audience
                        
                        <h4>Technical Tips</h4>
                        - Break long text into sections
                        - Use proper punctuation
                        - Include pauses between sections
                        - Maintain consistent formatting
                    </div>
                    """, unsafe_allow_html=True)
    
    # Prediction History Page
    elif page == "Prediction History":
        display_prediction_history()
    
    # About Page
    elif page == "About":
        st.title("About ChurnPredictorPro")
        
        # Author section with images
        st.subheader("Developed By")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("images/author1.jpg", width=150, caption="Lead Developer")
            st.write("""
            **S.Tamilselvan**  
            Ethical Hacker & Cyber Security Expert  
           
            """)
        
        with col1:
            st.image("images/pugazhmani-photoaidcom-cropped.jpg", width=150, caption="UX Designer")
            st.write("""
            **K.Pugazhmani**  
            Devloper Front End  
            Focused on user experience and interface design
            """)
       
        with col2:
            st.image("images/vikram-modified.png", width=150, caption="cloud Assistant")
            st.write("""
            **P.Vikram**  
            cloud Assistant 
            Expert in cloud technologies and deployment
            """)
        # Overview section
        st.subheader("Overview")
        st.write("""
        ChurnPredictorPro is an advanced analytics platform designed to help businesses predict and prevent customer churn. 
        By leveraging machine learning and AI technologies, it provides actionable insights to improve customer retention 
        and drive business growth.
        """)
        
        # Key Features section
        st.subheader("Key Features")
        features = [
            "Advanced Churn Prediction using state-of-the-art machine learning algorithms",
            "Comprehensive Data Analysis with interactive visualizations",
            "AI-Powered Insights for deeper understanding of churn patterns",
            "Customer Segmentation to identify high-risk groups",
            "Actionable Recommendations for churn prevention",
            "Speech Summaries for easy sharing of insights"
        ]
        
        for feature in features:
            st.write(f"✓ {feature}")
        
        # Getting Started section
        st.subheader("Getting Started")
        steps = [
            "Upload your customer data in CSV format",
            "Preprocess and clean your data using our automated tools",
            "Train machine learning models to predict churn",
            "Analyze results through interactive visualizations",
            "Generate insights and recommendations",
            "Export reports and share findings with your team"
        ]
        
        for i, step in enumerate(steps, 1):
            st.write(f"{i}. {step}")
        
        # Best Practices section
        st.subheader("Best Practices")
        practices = [
            "Regularly update your customer data for accurate predictions",
            "Use segmentation to target high-risk customer groups",
            "Implement recommended retention strategies",
            "Monitor model performance and retrain as needed",
            "Share insights across departments for coordinated action"
        ]
        
        for practice in practices:
            st.write(f"• {practice}")
        
        # Technology Stack section
        st.subheader("Technology Stack")
        tech_stack = [
            "Python", "Streamlit", "Scikit-learn", "Pandas",
            "Plotly", "SQLite", "gTTS", "Firebase"
        ]
        
        st.write("Built with: " + ", ".join(tech_stack))
        
        # Support section
        st.subheader("Support")
        st.write("""
        Need help or have questions? Our support team is here to assist you:
        
        📧 Tamilselvan637986@gmail.com 
        📚 [Documentation](https://github.com/Tamilselvan-S-Cyber-Security/Customer-Churn-Prediction)  
        
        """)
        
        # Version and Copyright
        st.markdown("---")
        st.write("""
        Version 1.0.0  
        © 2025 Cyber WOlf. All rights reserved.
        """)
