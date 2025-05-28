import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_churn_distribution(data, target_column='Churn'):
    """
    Plot the distribution of the churn target variable.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    target_column : str
        Name of the target column
    """
    try:
        # Get churn counts
        churn_counts = data[target_column].value_counts().reset_index()
        churn_counts.columns = [target_column, 'Count']
        
        # Add percentage
        total = churn_counts['Count'].sum()
        churn_counts['Percentage'] = churn_counts['Count'] / total * 100
        
        # Create labels
        churn_counts['Label'] = churn_counts[target_column].astype(str)
        if target_column == 'Churn' and set(data[target_column].unique()) == {0, 1}:
            churn_counts['Label'] = churn_counts[target_column].map({0: 'Not Churned', 1: 'Churned'})
        
        # Create pie chart
        fig1 = px.pie(
            churn_counts, 
            values='Count', 
            names='Label', 
            title=f'{target_column} Distribution',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Create bar chart
        fig2 = px.bar(
            churn_counts, 
            x='Label', 
            y='Count', 
            text='Percentage', 
            title=f'{target_column} Distribution',
            color='Label',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(xaxis_title=target_column, yaxis_title='Count')
        st.plotly_chart(fig2, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error plotting churn distribution: {str(e)}")

def plot_feature_importance(feature_importance, feature_names):
    """
    Plot feature importance from the trained model.
    
    Parameters:
    -----------
    feature_importance : numpy.ndarray
        Feature importance values from the model
    feature_names : list
        Names of the features
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The figure object containing the plot
    """
    try:
        # Create DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Return the figure for export
        return fig
        
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")
        return None

def plot_churn_by_categorical(data, categorical_columns, target_column='Churn'):
    """
    Plot churn distribution by categorical features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    categorical_columns : list
        List of categorical column names
    target_column : str
        Name of the target column
    """
    try:
        for column in categorical_columns:
            # Get value counts with percentages
            grouped = data.groupby([column, target_column]).size().reset_index(name='Count')
            total_counts = data.groupby(column).size().reset_index(name='Total')
            merged = pd.merge(grouped, total_counts, on=column)
            merged['Percentage'] = merged['Count'] / merged['Total'] * 100
            
            # Create labels
            target_labels = {0: 'Not Churned', 1: 'Churned'} if target_column == 'Churn' and set(data[target_column].unique()) == {0, 1} else None
            
            # Plot count distribution
            fig1 = px.bar(
                grouped, 
                x=column, 
                y='Count', 
                color=target_column,
                title=f'Churn Count by {column}',
                barmode='group',
                color_discrete_map=target_labels
            )
            fig1.update_layout(xaxis_title=column, yaxis_title='Count')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Plot percentage distribution
            fig2 = px.bar(
                merged, 
                x=column, 
                y='Percentage', 
                color=target_column,
                title=f'Churn Percentage by {column}',
                barmode='group',
                color_discrete_map=target_labels
            )
            fig2.update_layout(xaxis_title=column, yaxis_title='Percentage (%)')
            st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error plotting categorical analysis: {str(e)}")

def plot_numerical_features(data, numerical_columns, target_column='Churn'):
    """
    Plot distribution of numerical features by churn status.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    numerical_columns : list
        List of numerical column names
    target_column : str
        Name of the target column
    """
    try:
        for column in numerical_columns:
            # Create labels
            target_labels = {0: 'Not Churned', 1: 'Churned'} if target_column == 'Churn' and set(data[target_column].unique()) == {0, 1} else None
            
            # Histogram
            fig1 = px.histogram(
                data, 
                x=column, 
                color=target_column,
                marginal="box", 
                title=f'Distribution of {column} by {target_column}',
                color_discrete_map=target_labels
            )
            fig1.update_layout(xaxis_title=column, yaxis_title='Count')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Box plot
            fig2 = px.box(
                data, 
                x=target_column, 
                y=column, 
                color=target_column,
                title=f'Box Plot of {column} by {target_column}',
                points="all",
                color_discrete_map=target_labels
            )
            fig2.update_layout(xaxis_title=target_column, yaxis_title=column)
            st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error plotting numerical analysis: {str(e)}")

def plot_correlation_heatmap(data):
    """
    Plot correlation heatmap for numerical features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    """
    try:
        # Get numerical columns
        numerical_data = data.select_dtypes(include=['int64', 'float64'])
        
        # Compute correlation matrix
        corr = numerical_data.corr()
        
        # Plot heatmap
        fig = px.imshow(
            corr, 
            text_auto='.2f', 
            aspect="auto",
            title='Correlation Heatmap',
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error plotting correlation heatmap: {str(e)}")
