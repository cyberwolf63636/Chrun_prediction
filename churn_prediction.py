import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import streamlit as st

def predict_churn(data, features, target, model_type="RandomForest", test_size=0.2, random_state=42):
    """
    Train a model and make predictions on customer churn.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The preprocessed dataset
    features : list
        List of feature column names
    target : str
        Name of the target column
    model_type : str
        Type of model to train (RandomForest, LogisticRegression, GradientBoosting)
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    model, predictions, metrics
    """
    # Split data into features and target
    X = data[features]
    y = data[target]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features for logistic regression
    if model_type == "LogisticRegression":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Train model based on model_type
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == "LogisticRegression":
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Get feature importance if available
    predictions = {'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob}
    if hasattr(model, 'feature_importances_'):
        predictions['feature_importance'] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        predictions['feature_importance'] = np.abs(model.coef_[0])
    
    return model, predictions, metrics

def train_model(data, features, target, model_type="RandomForest", test_size=0.2, random_state=42, save_to_db=False, user_email=None):
    """
    Wrapper function for predict_churn that handles errors and displays progress.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The preprocessed dataset
    features : list
        List of feature column names
    target : str
        Name of the target column
    model_type : str
        Type of model to train
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
    save_to_db : bool
        Whether to save predictions and metrics to the database
    user_email : str
        Email of the user making the predictions (required if save_to_db=True)
    
    Returns:
    --------
    model, predictions, metrics
    """
    try:
        with st.spinner(f"Training {model_type} model..."):
            model, predictions, metrics = predict_churn(
                data, features, target, model_type, test_size, random_state
            )
            
            # Save to database if requested
            if save_to_db and user_email:
                try:
                    from database import save_predictions_to_db, save_model_metrics
                    
                    # Create a DataFrame for predictions
                    predictions_df = pd.DataFrame({
                        'Actual': predictions['y_test'],
                        'Predicted': predictions['y_pred'],
                        'Probability': [p[1] for p in predictions['y_prob']]
                    })
                    
                    # Save predictions to database
                    save_predictions_to_db(user_email, predictions_df, model_type)
                    
                    # Save model metrics to database
                    feature_importance = None
                    if 'feature_importance' in predictions:
                        feature_importance = list(zip(features, predictions['feature_importance'].tolist()))
                    
                    save_model_metrics(user_email, metrics, model_type, feature_importance)
                    
                    st.success("Predictions and metrics saved to database successfully!")
                except Exception as e:
                    st.warning(f"Could not save to database: {str(e)}")
        
        return model, predictions, metrics
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : trained model object
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics
