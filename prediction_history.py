import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import get_user_predictions, get_user_model_metrics, get_prediction_history_stats

def display_prediction_history():
    """Display prediction history page content"""
    st.header("Prediction History & Database")
    
    # Check if user is authenticated
    if not st.session_state.authenticated or not st.session_state.user_email:
        st.warning("Please log in to view prediction history.")
        return
    
    # Show prediction history statistics
    st.subheader("Prediction Statistics")
    
    try:
        # Get prediction history stats
        stats = get_prediction_history_stats(st.session_state.user_email)
        
        if stats["total_predictions"] > 0:
            # Display key metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", f"{stats['total_predictions']:,}")
            
            with col2:
                st.metric("Churn Rate", f"{stats['churn_percentage']:.1f}%")
            
            with col3:
                st.metric("High Risk Customers", f"{stats['high_risk_count']:,}")
            
            # Model distribution
            if stats["model_distribution"]:
                st.subheader("Model Usage Distribution")
                
                # Prepare data for the pie chart
                models = list(stats["model_distribution"].keys())
                counts = list(stats["model_distribution"].values())
                
                # Create a pie chart
                fig = px.pie(
                    names=models,
                    values=counts,
                    title="Predictions by Model Type",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction trend over time
            if stats["prediction_trend"]:
                st.subheader("Prediction Trend")
                
                # Convert to DataFrame for plotting
                trend_df = pd.DataFrame(stats["prediction_trend"])
                
                # Create a line chart
                fig = px.line(
                    trend_df, 
                    x="date", 
                    y=["count", "churn_count"],
                    title="Predictions Over Time",
                    labels={"value": "Count", "date": "Date", "variable": "Type"},
                    color_discrete_map={"count": "#636EFA", "churn_count": "#EF553B"}
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction history found. Make some predictions to see statistics here.")
    
    except Exception as e:
        st.error(f"Error retrieving prediction statistics: {str(e)}")
    
    # Recent predictions table
    st.subheader("Recent Predictions")
    
    try:
        predictions_df = get_user_predictions(st.session_state.user_email)
        
        if not predictions_df.empty:
            # Style predictions dataframe with colors
            def highlight_churn(val):
                if val == 1:
                    return 'background-color: rgba(255, 75, 75, 0.2)'
                return ''
            
            st.dataframe(
                predictions_df.style.applymap(
                    highlight_churn, 
                    subset=['prediction']
                ),
                use_container_width=True
            )
            
            # Export to CSV option
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                "Download Predictions as CSV",
                csv,
                "churn_predictions.csv",
                "text/csv",
                key='download-predictions-csv'
            )
        else:
            st.info("No predictions found. Train a model and make predictions to see results here.")
    
    except Exception as e:
        st.error(f"Error retrieving predictions: {str(e)}")
    
    # Model performance history
    st.subheader("Model Performance History")
    
    try:
        metrics_df = get_user_model_metrics(st.session_state.user_email)
        
        if not metrics_df.empty:
            # Display metrics as a table
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visualize metrics over time
            st.subheader("Model Performance Comparison")
            
            # Create a radar chart comparing models
            fig = go.Figure()
            
            for i, row in metrics_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['accuracy'], row['precision'], row['recall'], row['f1']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    fill='toself',
                    name=f"{row['model_type']} ({row['created_at']})"
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model metrics found. Train models to track performance here.")
    
    except Exception as e:
        st.error(f"Error retrieving model metrics: {str(e)}")
    
    # Database management section
    with st.expander("Database Management (Admin)"):
        st.warning("⚠️ These operations can result in permanent data loss. Use with caution.")
        
        # Export entire database
        if st.button("Export All Data"):
            try:
                # In a real app, this would export all tables
                st.success("Export functionality would be implemented here in production.")
                st.info("This would export all your prediction data to a downloadable file.")
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")
        
        # Clear prediction history
        if st.button("Clear My Prediction History"):
            # In production, this would clear the user's prediction history
            st.success("Clear functionality would be implemented here in production.")
            st.info("This would remove all your prediction history from the database.")
    
    # Help section
    with st.expander("Help & FAQ"):
        st.markdown("""
        ## Using the Prediction History
        
        This page displays all predictions you've made and tracks model performance over time.
        
        ### FAQ
        
        **How are predictions stored?**  
        Predictions are stored in an SQLite database and linked to your user account.
        
        **Can I export my prediction history?**  
        Yes, use the "Download Predictions as CSV" button to export your prediction data.
        
        **What is a "high risk" customer?**  
        Customers with a churn probability greater than 70% are considered high risk.
        
        **Why track model performance?**  
        Tracking model performance helps you understand which models work best for your data.
        """)