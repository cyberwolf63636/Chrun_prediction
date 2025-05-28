import os
import sqlite3
import pandas as pd
import streamlit as st

# Database paths
DEFAULT_DB_PATH = "churn_predictions.db"

def get_db_path():
    """Get the database path from environment or use default"""
    return os.environ.get("DB_PATH", DEFAULT_DB_PATH)

def init_database():
    """Initialize the SQLite database with required tables if they don't exist"""
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create predictions table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        customer_id TEXT,
        prediction INTEGER,
        probability REAL,
        model_type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_email) REFERENCES users(email)
    )
    ''')
    
    # Create model_metrics table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        model_type TEXT,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1 REAL,
        feature_importance TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_email) REFERENCES users(email)
    )
    ''')
    
    # Create customer_segments table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customer_segments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        customer_id TEXT,
        segment TEXT,
        segment_method TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_email) REFERENCES users(email)
    )
    ''')
    
    conn.commit()
    conn.close()

def save_predictions_to_db(user_email, predictions_df, model_type):
    """
    Save predictions to the database
    
    Parameters:
    -----------
    user_email : str
        The email of the user making the predictions
    predictions_df : pandas.DataFrame
        DataFrame containing predictions with columns: 'customer_id', 'prediction', 'probability'
    model_type : str
        The type of model used for predictions
    """
    conn = sqlite3.connect(get_db_path())
    
    # Prepare the data for insertion
    data_to_insert = []
    for _, row in predictions_df.iterrows():
        # Ensure customer_id is a string (could be index or actual ID column)
        customer_id = str(row.name) if 'customer_id' not in row else str(row['customer_id'])
        
        # Get the prediction and probability
        prediction = int(row['Predicted'])
        probability = float(row['Probability'])
        
        data_to_insert.append((user_email, customer_id, prediction, probability, model_type))
    
    # Insert data into the predictions table
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO predictions (user_email, customer_id, prediction, probability, model_type) VALUES (?, ?, ?, ?, ?)",
        data_to_insert
    )
    
    conn.commit()
    conn.close()

def save_model_metrics(user_email, metrics, model_type, feature_importance=None):
    """
    Save model metrics to the database
    
    Parameters:
    -----------
    user_email : str
        The email of the user who trained the model
    metrics : dict
        Dictionary containing model metrics
    model_type : str
        The type of model used
    feature_importance : list, optional
        Feature importance values (will be stored as JSON)
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    # Convert feature importance to string if provided
    feature_importance_str = None
    if feature_importance is not None:
        import json
        feature_importance_str = json.dumps(feature_importance)
    
    cursor.execute(
        """
        INSERT INTO model_metrics 
        (user_email, model_type, accuracy, precision, recall, f1, feature_importance) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_email, 
            model_type, 
            metrics['accuracy'], 
            metrics['precision'], 
            metrics['recall'], 
            metrics['f1'], 
            feature_importance_str
        )
    )
    
    conn.commit()
    conn.close()

def save_customer_segments(user_email, segments_df, segment_method):
    """
    Save customer segments to the database
    
    Parameters:
    -----------
    user_email : str
        The email of the user creating the segments
    segments_df : pandas.DataFrame
        DataFrame containing segments with columns: 'segment'
    segment_method : str
        The method used for segmentation (KMeans, RFM, Manual)
    """
    conn = sqlite3.connect(get_db_path())
    
    # Prepare the data for insertion
    data_to_insert = []
    for idx, row in segments_df.iterrows():
        # Ensure customer_id is a string
        customer_id = str(idx)
        
        # Get the segment value
        segment = str(row['segment']) if 'segment' in row else str(row['Segment'])
        
        data_to_insert.append((user_email, customer_id, segment, segment_method))
    
    # Insert data into the customer_segments table
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO customer_segments (user_email, customer_id, segment, segment_method) VALUES (?, ?, ?, ?)",
        data_to_insert
    )
    
    conn.commit()
    conn.close()

def get_user_predictions(user_email, limit=100):
    """
    Retrieve the predictions made by a user
    
    Parameters:
    -----------
    user_email : str
        The email of the user
    limit : int
        Maximum number of records to retrieve
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the predictions
    """
    conn = sqlite3.connect(get_db_path())
    
    query = f"""
    SELECT customer_id, prediction, probability, model_type, created_at
    FROM predictions
    WHERE user_email = ?
    ORDER BY created_at DESC
    LIMIT {limit}
    """
    
    df = pd.read_sql(query, conn, params=(user_email,))
    conn.close()
    
    return df

def get_user_model_metrics(user_email, limit=5):
    """
    Retrieve the model metrics for a user
    
    Parameters:
    -----------
    user_email : str
        The email of the user
    limit : int
        Maximum number of records to retrieve
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the model metrics
    """
    conn = sqlite3.connect(get_db_path())
    
    query = f"""
    SELECT model_type, accuracy, precision, recall, f1, created_at
    FROM model_metrics
    WHERE user_email = ?
    ORDER BY created_at DESC
    LIMIT {limit}
    """
    
    df = pd.read_sql(query, conn, params=(user_email,))
    conn.close()
    
    return df

def get_user_segments(user_email, limit=100):
    """
    Retrieve the customer segments created by a user
    
    Parameters:
    -----------
    user_email : str
        The email of the user
    limit : int
        Maximum number of records to retrieve
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the customer segments
    """
    conn = sqlite3.connect(get_db_path())
    
    query = f"""
    SELECT customer_id, segment, segment_method, created_at
    FROM customer_segments
    WHERE user_email = ?
    ORDER BY created_at DESC
    LIMIT {limit}
    """
    
    df = pd.read_sql(query, conn, params=(user_email,))
    conn.close()
    
    return df

def get_prediction_history_stats(user_email):
    """
    Get statistics about prediction history for a user
    
    Parameters:
    -----------
    user_email : str
        The email of the user
    
    Returns:
    --------
    dict
        Dictionary containing prediction history stats
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    # Get total prediction count
    cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_email = ?",
        (user_email,)
    )
    total_predictions = cursor.fetchone()[0]
    
    # Get churn percentage
    cursor.execute(
        """
        SELECT 
            SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as churn_percentage
        FROM predictions 
        WHERE user_email = ?
        """,
        (user_email,)
    )
    result = cursor.fetchone()
    churn_percentage = result[0] if result[0] is not None else 0
    
    # Get high risk count (probability > 0.7)
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM predictions 
        WHERE user_email = ? AND prediction = 1 AND probability > 0.7
        """,
        (user_email,)
    )
    high_risk_count = cursor.fetchone()[0]
    
    # Get model type distribution
    cursor.execute(
        """
        SELECT model_type, COUNT(*) as count
        FROM predictions 
        WHERE user_email = ?
        GROUP BY model_type
        """,
        (user_email,)
    )
    model_distribution = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Get prediction trend by date
    cursor.execute(
        """
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as count,
            SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as churn_count
        FROM predictions 
        WHERE user_email = ?
        GROUP BY DATE(created_at)
        ORDER BY date
        LIMIT 10
        """,
        (user_email,)
    )
    prediction_trend = [
        {"date": row[0], "count": row[1], "churn_count": row[2]} 
        for row in cursor.fetchall()
    ]
    
    conn.close()
    
    return {
        "total_predictions": total_predictions,
        "churn_percentage": churn_percentage,
        "high_risk_count": high_risk_count,
        "model_distribution": model_distribution,
        "prediction_trend": prediction_trend
    }