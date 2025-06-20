import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from PIL import Image
import cv2
import asyncio
import json
import os
from typing import Dict, Any, List
import warnings

# Import custom modules
from modules.chat import IntelligentChatEngine
from modules.visuals import CNNChartClassifier, EnhancedImageProcessor
from modules.machine_learning_models import EnhancedMLPipeline
from modules.eda import ComprehensiveEDA

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Data Science & ML Pipeline",
    page_icon="🤖",
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
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'eda_results' not in st.session_state:
    st.session_state.eda_results = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""

# Helper functions
def load_data():
    """Load and cache data"""
    return st.session_state.uploaded_data

def save_plot_as_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def display_metrics_grid(metrics_dict, title="Metrics"):
    """Display metrics in a grid format"""
    st.markdown(f"### {title}")
    cols = st.columns(len(metrics_dict))
    for i, (key, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            if isinstance(value, float):
                st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
            else:
                st.metric(key.replace('_', ' ').title(), str(value))

# Main Application Header
st.markdown('<h1 class="main-header">🤖 Advanced Data Science & ML Pipeline</h1>', unsafe_allow_html=True)
st.markdown("### Comprehensive solution for EDA, Machine Learning, Chart Analysis, and AI-powered insights")

# Sidebar Navigation
st.sidebar.title("🚀 Navigation")
page = st.sidebar.selectbox(
    "Choose a module:",
    ["🏠 Home", "📊 Data Upload & EDA", "📈 Chart Analysis", "🤖 Machine Learning", "💬 AI Chat Assistant"]
)

# API Configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 API Configuration")
groq_api_key = st.sidebar.text_input(
    "Groq API Key (for AI features)", 
    type="password", 
    value=st.session_state.groq_api_key,
    help="Enter your Groq API key for AI-powered analysis"
)
if groq_api_key:
    st.session_state.groq_api_key = groq_api_key

# ===== HOME PAGE =====
if page == "🏠 Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🎯 Welcome to Your Complete Data Science Toolkit!")
        
        st.markdown("""
        <div class="info-box">
        <h4>🚀 Key Features:</h4>
        <ul>
            <li><strong>📊 Advanced EDA</strong> - Comprehensive exploratory data analysis with automated insights</li>
            <li><strong>📈 Chart Processing</strong> - AI-powered chart classification and data extraction</li>
            <li><strong>🤖 ML Pipeline</strong> - Automated machine learning with 8+ algorithms and hyperparameter tuning</li>
            <li><strong>💬 AI Assistant</strong> - Intelligent chat with Groq API for contextual analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📈 Platform Capabilities")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🔬 EDA Features</h3>
                <p>• Missing value analysis<br>
                • Correlation heatmaps<br>
                • Distribution analysis<br>
                • Outlier detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Chart Analysis</h3>
                <p>• 7+ chart types<br>
                • OCR text extraction<br>
                • Visual data parsing<br>
                • AI-powered insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 ML Models</h3>
                <p>• 8+ algorithms<br>
                • Auto hyperparameter tuning<br>
                • Ensemble methods<br>
                • Performance visualization</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>💬 AI Assistant</h3>
                <p>• Context-aware chat<br>
                • Multi-modal analysis<br>
                • Groq API integration<br>
                • Memory & insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🚀 Get Started")
        st.info("👈 Use the sidebar to navigate between different modules. Start with **Data Upload & EDA** to analyze your dataset!")

# ===== DATA UPLOAD & EDA PAGE =====
elif page == "📊 Data Upload & EDA":
    st.markdown('<h2 class="sub-header">📊 Data Upload & Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown("### 📁 Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload a CSV file to start your analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            
            # Display basic info
            st.markdown("### 📋 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicates", df.duplicated().sum())
            
            # Display first few rows
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # EDA Section
            st.markdown("---")
            st.markdown("### 🔍 Comprehensive EDA Analysis")
            
            # Target column selection
            target_column = st.selectbox(
                "Select target column (optional):",
                ["None"] + df.columns.tolist(),
                help="Select the target variable for supervised learning analysis"
            )
            target_column = None if target_column == "None" else target_column
            
            if st.button("🚀 Run Complete EDA Analysis", type="primary"):
                with st.spinner("🔄 Performing comprehensive EDA analysis..."):
                    try:
                        eda_analyzer = ComprehensiveEDA()
                        eda_results = eda_analyzer.perform_full_eda(df, target_column)
                        st.session_state.eda_results = eda_results
                        
                        st.success("✅ EDA Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during EDA analysis: {str(e)}")
            
            # Display EDA Results
            if st.session_state.eda_results:
                results = st.session_state.eda_results
                
                # Basic Information
                st.markdown("### 📊 Dataset Statistics")
                basic_info = results['basic_info']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.json(basic_info)
                
                with col2:
                    # Missing values chart
                    if results['missing_analysis']['missing_summary']:
                        missing_df = pd.DataFrame(results['missing_analysis']['missing_summary'])
                        fig = px.bar(missing_df, x='Column', y='Missing_Percentage', 
                                   title='Missing Values by Column')
                        st.plotly_chart(fig, use_container_width=True)
                
                # Statistical Summary
                st.markdown("### 📈 Statistical Summary")
                if results['statistical_summary']['numeric_summary']:
                    st.markdown("**Numeric Variables**")
                    numeric_summary_df = pd.DataFrame(results['statistical_summary']['numeric_summary'])
                    st.dataframe(numeric_summary_df, use_container_width=True)
                
                if results['statistical_summary']['categorical_summary']:
                    st.markdown("**Categorical Variables**")
                    cat_summary_df = pd.DataFrame(results['statistical_summary']['categorical_summary'])
                    st.dataframe(cat_summary_df, use_container_width=True)
                
                # Correlation Analysis
                st.markdown("### 🔗 Correlation Analysis")
                if results['correlation_analysis']['correlation_matrix']:
                    corr_df = pd.DataFrame(results['correlation_analysis']['correlation_matrix'])
                    fig = px.imshow(corr_df, 
                                  title="Correlation Heatmap",
                                  color_continuous_scale="RdBu_r",
                                  aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                
                # High Correlations
                if results['correlation_analysis']['high_correlations']:
                    st.markdown("**High Correlations (|r| > 0.7)**")
                    high_corr_df = pd.DataFrame(results['correlation_analysis']['high_correlations'])
                    st.dataframe(high_corr_df, use_container_width=True)
                
                # Distribution Analysis
                st.markdown("### 📊 Distribution Analysis")
                numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns]
                
                if numeric_cols:
                    selected_col = st.selectbox("Select column for distribution plot:", numeric_cols)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.box(df, y=selected_col, title=f'Box Plot of {selected_col}')
                        st.plotly_chart(fig, use_container_width=True)
                
                # Outlier Analysis
                st.markdown("### 🎯 Outlier Analysis")
                if results['outlier_analysis']:
                    outlier_df = pd.DataFrame([
                        {
                            'Column': col, 
                            'Outlier Count': data['outlier_count'],
                            'Outlier %': f"{data['outlier_percentage']:.2f}%"
                        }
                        for col, data in results['outlier_analysis'].items()
                        if data['outlier_count'] > 0
                    ])
                    
                    if not outlier_df.empty:
                        st.dataframe(outlier_df, use_container_width=True)
                    else:
                        st.info("No significant outliers detected in the dataset.")
                
                # Target Analysis
                if results['target_analysis']:
                    st.markdown("### 🎯 Target Variable Analysis")
                    target_info = results['target_analysis']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json(target_info)
                    
                    with col2:
                        if target_info['type'] == 'categorical':
                            # Class distribution
                            class_counts = pd.Series(target_info['value_counts'])
                            fig = px.pie(values=class_counts.values, names=class_counts.index,
                                       title=f'Distribution of {target_column}')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Distribution plot
                            fig = px.histogram(df, x=target_column, 
                                             title=f'Distribution of {target_column}')
                            st.plotly_chart(fig, use_container_width=True)
                
                # Insights and Recommendations
                st.markdown("### 💡 Key Insights")
                for insight in results['insights']:
                    st.markdown(f"• {insight}")
                
                st.markdown("### 🔧 Recommendations")
                for recommendation in results['recommendation']:
                    st.markdown(f"• {recommendation}")
                
                # Visualizations
                if results['visualizations']:
                    st.markdown("### 📊 Generated Visualizations")
                    for viz_name, viz_path in results['visualizations'].items():
                        if os.path.exists(viz_path):
                            st.markdown(f"**{viz_name.replace('_', ' ').title()}**")
                            st.image(viz_path, use_column_width=True)
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to start the analysis")

# ===== CHART ANALYSIS PAGE =====
elif page == "📈 Chart Analysis":
    st.markdown('<h2 class="sub-header">📈 Chart Analysis & Data Extraction</h2>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Upload Chart Images for Analysis")
    
    uploaded_images = st.file_uploader(
        "Choose chart image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload chart images (PNG, JPG, JPEG) for AI-powered analysis"
    )
    
    if uploaded_images:
        # Initialize processors
        chart_classifier = CNNChartClassifier()
        image_processor = EnhancedImageProcessor()
        
        st.markdown(f"### 🔍 Analysis Results ({len(uploaded_images)} images)")
        
        for idx, uploaded_image in enumerate(uploaded_images):
            st.markdown(f"#### 📊 Image {idx + 1}: {uploaded_image.name}")
            
            try:
                # Load and display image
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(image, caption=f"Original Image", use_column_width=True)
                
                with col2:
                    with st.spinner("🔄 Analyzing chart..."):
                        # Convert PIL to OpenCV format
                        if len(image_array.shape) == 3:
                            opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        else:
                            opencv_image = image_array
                        
                        # Classify chart type
                        chart_type, confidence = chart_classifier.classify_chart(opencv_image)
                        
                        # Extract chart data
                        extracted_data = image_processor.extract_chart_data(opencv_image, chart_type)
                        
                        # Display results
                        st.markdown("**📋 Classification Results**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Chart Type", chart_type.title())
                        with col_b:
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Display extracted data
                        st.markdown("**📊 Extracted Data**")
                        
                        if extracted_data.get('text_data'):
                            st.markdown("**Text Data (OCR)**")
                            st.write(extracted_data['text_data'])
                        
                        if extracted_data.get('extracted_values'):
                            st.markdown("**Numeric Values**")
                            st.write(extracted_data['extracted_values'])
                        
                        if extracted_data.get('visual_data'):
                            st.markdown("**Visual Data**")
                            st.json(extracted_data['visual_data'])
                        
                        # Generate AI insights if API key is available
                        if st.session_state.groq_api_key:
                            if st.button(f"🤖 Generate AI Insights for Image {idx + 1}", key=f"ai_insights_{idx}"):
                                with st.spinner("🧠 Generating AI insights..."):
                                    try:
                                        chat_engine = IntelligentChatEngine(
                                            api_key=st.session_state.groq_api_key,
                                            api_url="https://api.groq.com/openai/v1/chat/completions",
                                            model="mixtral-8x7b-32768"
                                        )
                                        
                                        # Generate insights asynchronously
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        insights = loop.run_until_complete(
                                            chat_engine.generate_chart_insights(chart_type, extracted_data)
                                        )
                                        
                                        st.markdown("**🧠 AI-Generated Insights**")
                                        for insight in insights:
                                            st.markdown(f"• {insight}")
                                        
                                    except Exception as e:
                                        st.error(f"❌ Error generating insights: {str(e)}")
                        else:
                            st.info("💡 Add Groq API key in the sidebar to generate AI-powered insights!")
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error processing image {uploaded_image.name}: {str(e)}")
    
    else:
        st.info("👆 Please upload chart images to start the analysis")
        
        # Example charts section
        st.markdown("### 📚 Supported Chart Types")
        chart_types = {
            "Bar Charts": "Vertical and horizontal bar charts with automatic data extraction",
            "Line Charts": "Line plots with trend analysis capabilities", 
            "Pie Charts": "Circular charts with segment detection",
            "Scatter Plots": "Point-based charts with correlation analysis",
            "Histograms": "Distribution visualizations",
            "Box Plots": "Statistical summary visualizations",
            "Area Charts": "Filled area visualizations"
        }
        
        for chart_name, description in chart_types.items():
            st.markdown(f"**{chart_name}**: {description}")

# ===== MACHINE LEARNING PAGE =====
elif page == "🤖 Machine Learning":
    st.markdown('<h2 class="sub-header">🤖 Advanced Machine Learning Pipeline</h2>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        
        st.markdown("### ⚙️ Configure ML Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Task type selection
            task_type = st.selectbox(
                "Select ML Task Type:",
                ["classification", "regression"],
                help="Choose the type of machine learning task"
            )
        
        with col2:
            # Target column selection
            target_column = st.selectbox(
                "Select Target Column:",
                df.columns.tolist(),
                help="Choose the target variable for prediction"
            )
        
        # Display target variable info
        if target_column:
            st.markdown("### 🎯 Target Variable Information")
            target_info = {
                "Data Type": str(df[target_column].dtype),
                "Unique Values": df[target_column].nunique(),
                "Missing Values": df[target_column].isnull().sum(),
                "Sample Values": df[target_column].dropna().head(5).tolist()
            }
            
            col1, col2 = st.columns(2)
            with col1:
                for key, value in list(target_info.items())[:2]:
                    st.metric(key, value)
            with col2:
                for key, value in list(target_info.items())[2:4]:
                    st.metric(key, value)
            
            st.write("**Sample Values:**", target_info["Sample Values"])
        
        # ML Pipeline Configuration
        st.markdown("### 🔧 Pipeline Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        with col2:
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        # Feature selection options
        st.markdown("**Feature Selection (Optional)**")
        feature_columns = st.multiselect(
            "Select features to include (leave empty for all):",
            [col for col in df.columns if col != target_column],
            help="Select specific features or leave empty to use all features"
        )
        
        # Start ML training
        if st.button("🚀 Start ML Training Pipeline", type="primary"):
            if target_column:
                with st.spinner("🔄 Training multiple ML models... This may take a few minutes."):
                    try:
                        # Prepare data
                        if feature_columns:
                            X = df[feature_columns]
                        else:
                            X = df.drop(columns=[target_column])
                        
                        # Create ML pipeline
                        ml_pipeline = EnhancedMLPipeline()
                        
                        # Train and evaluate models
                        results = ml_pipeline.train_and_evaluate(df, task_type, target_column)
                        st.session_state.ml_results = results
                        
                        st.success("✅ ML Training completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during ML training: {str(e)}")
            else:
                st.warning("⚠️ Please select a target column first!")
        
        # Display ML Results
        if st.session_state.ml_results:
            results = st.session_state.ml_results
            
            st.markdown("---")
            st.markdown("### 🏆 Model Performance Results")
            
            # Best Model Summary
            best_model = results['best_model']
            st.markdown("#### 🥇 Best Performing Model")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Name", best_model['name'])
            with col2:
                primary_score = best_model['primary_score']
                st.metric("Primary Score", f"{primary_score:.4f}")
            with col3:
                st.metric("Task Type", results['task_type'].title())
            
            # Model Comparison Table
            st.markdown("#### 📊 Model Comparison")
            comparison_df = pd.DataFrame(results['comparison_table'])
            
            # Format numeric columns
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                comparison_df[col] = comparison_df[col].round(4)
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance Visualizations
            st.markdown("#### 📈 Performance Visualization")
            
            if task_type == "classification":
                # Classification metrics comparison
                metrics_data = []
                for model_result in results['comparison_table']:
                    if 'accuracy' in model_result:
                        metrics_data.append({
                            'Model': model_result['Model'],
                            'Accuracy': model_result.get('accuracy', 0),
                            'F1 Score': model_result.get('f1_macro', 0),
                            'Precision': model_result.get('precision_macro', 0),
                            'Recall': model_result.get('recall_macro', 0)
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Create comparison chart
                    fig = px.bar(metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                               x='Model', y='Score', color='Metric',
                               title='Classification Metrics Comparison',
                               barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Regression metrics comparison
                metrics_data = []
                for model_result in results['comparison_table']:
                    if 'r2_score' in model_result:
                        metrics_data.append({
                            'Model': model_result['Model'],
                            'R² Score': model_result.get('r2_score', 0),
                            'RMSE': model_result.get('rmse', 0),
                            'MAE': model_result.get('mae', 0)
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # R² Score comparison
                    fig = px.bar(metrics_df, x='Model', y='R² Score',
                               title='R² Score Comparison')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Error metrics comparison
                    error_df = metrics_df[['Model', 'RMSE', 'MAE']].melt(
                        id_vars=['Model'], var_name='Error Metric', value_name='Value')
                    fig = px.bar(error_df, x='Model', y='Value', color='Error Metric',
                               title='Error Metrics Comparison (Lower is Better)',
                               barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            if results.get('feature_importance'):
                st.markdown("#### 🔍 Feature Importance")
                importance_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v} 
                    for k, v in results['feature_importance'].items()
                ]).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df.head(15), x='Importance', y='Feature',
                           title='Top 15 Most Important Features',
                           orientation='h')
                st.plotly_chart(fig, use_container_width=True)
            
            # Training Summary
            st.markdown("#### 📋 Training Summary")
            training_summary = results['training_summary']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Models", results['total_models_trained'])
            with col2:
                st.metric("Successful Models", training_summary['successful_models'])
            with col3:
                st.metric("Failed Models", training_summary['failed_models'])
            
            # Model Visualization Links
            st.markdown("#### 🎨 Model Visualizations")
            for model_result in results['comparison_table']:
                if 'Visualization' in model_result and os.path.exists(model_result['Visualization']):
                    st.markdown(f"**{model_result['Model']} Performance Plots**")
                    st.image(model_result['Visualization'], use_column_width=True)
            
            # Generate AI Insights
            if st.session_state.groq_api_key:
                if st.button("🧠 Generate AI Insights for ML Results"):
                    with st.spinner("🤖 Generating AI insights for ML results..."):
                        try:
                            chat_engine = IntelligentChatEngine(
                                api_key=st.session_state.groq_api_key,
                                api_url="https://api.groq.com/openai/v1/chat/completions",
                                model="mixtral-8x7b-32768"
                            )
                            
                            context = {
                                "task_type": results['task_type'],
                                "best_model": best_model,
                                "metrics": best_model['metrics'],
                                "comparison_table": results['comparison_table']
                            }
                            
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            response = loop.run_until_complete(
                                chat_engine.generate_response(
                                    "Analyze these ML results and provide insights", 
                                    context, 
                                    "model_performance"
                                )
                            )
                            
                            st.markdown("#### 🧠 AI Analysis")
                            st.markdown(response)
                            
                        except Exception as e:
                            st.error(f"❌ Error generating AI insights: {str(e)}")
    
    else:
        st.warning("⚠️ Please upload a dataset in the 'Data Upload & EDA' section first!")
        st.info("👈 Navigate to the 'Data Upload & EDA' page to upload your dataset.")

# ===== AI CHAT ASSISTANT PAGE =====
elif page == "💬 AI Chat Assistant":
    st.markdown('<h2 class="sub-header">💬 AI-Powered Data Science Assistant</h2>', unsafe_allow_html=True)
    
    if not st.session_state.groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to use the AI assistant.")
        st.info("Get your API key from: https://console.groq.com/keys")
    else:
    
    # Chat interface
    st.markdown("### 🤖 Chat with Your AI Data Science Assistant")
    
    # Initialize chat engine
    try:
        chat_engine = IntelligentChatEngine(
            api_key=st.session_state.groq_api_key,
            api_url="https://api.groq.com/openai/v1/chat/completions",
            model="mixtral-8x7b-32768"
        )
        
        # Context selection
        context_type = st.selectbox(
            "Select context type:",
            ["general", "chart_analysis", "eda_analysis", "model_performance"],
            help="Choose the context for your conversation"
        )
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Conversation History")
            for i, chat in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**👤 You:** {chat['message']}")
                    st.markdown(f"**🤖 Assistant:** {chat['response']}")
                    st.markdown("---")
        
        # Chat input
        user_message = st.text_area(
            "💬 Ask me anything about data science, your dataset, or analysis results:",
            height=100,
            placeholder="e.g., 'What insights can you provide about my dataset?' or 'How should I interpret these ML results?'"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            send_button = st.button("📤 Send", type="primary")
        
        with col2:
            clear_button = st.button("🗑️ Clear History")
        
        if clear_button:
            st.session_state.chat_history = []
            chat_engine.clear_memory()
            st.rerun()
        
        if send_button and user_message:
            with st.spinner("🤔 AI is thinking..."):
                try:
                    # Prepare context based on available data
                    context = {}
                    
                    if context_type == "eda_analysis" and st.session_state.eda_results:
                        context = st.session_state.eda_results
                    elif context_type == "model_performance" and st.session_state.ml_results:
                        context = st.session_state.ml_results
                    elif st.session_state.uploaded_data is not None:
                        context = {
                            "dataset_shape": st.session_state.uploaded_data.shape,
                            "columns": st.session_state.uploaded_data.columns.tolist(),
                            "missing_values": st.session_state.uploaded_data.isnull().sum().sum()
                        }
                    
                    # Generate response
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        chat_engine.generate_response(user_message, context, context_type)
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "message": user_message,
                        "response": response,
                        "context_type": context_type
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error getting AI response: {str(e)}")
        
        # Quick questions
        st.markdown("### ❓ Quick Questions")
        quick_questions = [
            "What are the key insights from my dataset?",
            "How should I preprocess my data for machine learning?",
            "Which machine learning algorithm would work best for my data?",
            "How can I improve my model performance?",
            "What are the potential issues with my dataset?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(f"💡 {question}", key=f"quick_{i}"):
                    # Auto-fill the question
                    st.session_state.temp_question = question
        
        # Context information
        st.markdown("### 📋 Available Context")
        context_info = []
        
        if st.session_state.uploaded_data is not None:
            context_info.append("✅ Dataset uploaded")
        if st.session_state.eda_results:
            context_info.append("✅ EDA analysis completed")
        if st.session_state.ml_results:
            context_info.append("✅ ML training completed")
        
        if context_info:
            for info in context_info:
                st.markdown(info)
        else:
            st.info("💡 Upload data and run analysis to get more contextual insights!")
        
    except Exception as e:
        st.error(f"❌ Error initializing chat engine: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 Advanced Data Science & ML Pipeline | Built with Streamlit, Groq AI, and ❤️</p>
    <p>For support or questions, please refer to the documentation.</p>
</div>
""", unsafe_allow_html=True)

