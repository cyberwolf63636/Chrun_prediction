import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_customer_id():
    return f"{random.randint(1000, 9999)}-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}"

def generate_tenure():
    return random.randint(1, 72)  # 1-72 months

def generate_monthly_charges(services):
    base = 20.0  # Base phone service
    if services['PhoneService'] == 'Yes':
        if services['MultipleLines'] == 'Yes':
            base += 10.0
    if services['InternetService'] == 'DSL':
        base += 30.0
    elif services['InternetService'] == 'Fiber optic':
        base += 50.0
    if services['OnlineSecurity'] == 'Yes':
        base += 10.0
    if services['OnlineBackup'] == 'Yes':
        base += 10.0
    if services['DeviceProtection'] == 'Yes':
        base += 10.0
    if services['TechSupport'] == 'Yes':
        base += 10.0
    if services['StreamingTV'] == 'Yes':
        base += 10.0
    if services['StreamingMovies'] == 'Yes':
        base += 10.0
    return round(base + random.uniform(-5, 5), 2)

def generate_total_charges(monthly_charges, tenure):
    return round(monthly_charges * tenure * random.uniform(0.9, 1.1), 2)

def generate_churn(services, tenure, monthly_charges):
    churn_prob = 0.0
    
    # Base churn probability
    churn_prob += 0.1
    
    # Contract type influence
    if services['Contract'] == 'Month-to-month':
        churn_prob += 0.2
    elif services['Contract'] == 'One year':
        churn_prob += 0.1
    
    # Tenure influence
    if tenure < 6:
        churn_prob += 0.2
    elif tenure < 12:
        churn_prob += 0.1
    
    # Service influence
    if services['InternetService'] == 'Fiber optic':
        churn_prob -= 0.1
    if services['OnlineSecurity'] == 'Yes':
        churn_prob -= 0.1
    if services['TechSupport'] == 'Yes':
        churn_prob -= 0.1
    
    # Price influence
    if monthly_charges > 100:
        churn_prob += 0.1
    
    return 'Yes' if random.random() < churn_prob else 'No'

def generate_customer():
    gender = random.choice(['Male', 'Female'])
    senior_citizen = random.choice([0, 1])
    partner = random.choice(['Yes', 'No'])
    dependents = random.choice(['Yes', 'No'])
    
    phone_service = random.choice(['Yes', 'No'])
    multiple_lines = random.choice(['Yes', 'No', 'No phone service']) if phone_service == 'Yes' else 'No phone service'
    
    internet_service = random.choice(['DSL', 'Fiber optic', 'No'])
    if internet_service == 'No':
        online_security = 'No internet service'
        online_backup = 'No internet service'
        device_protection = 'No internet service'
        tech_support = 'No internet service'
        streaming_tv = 'No internet service'
        streaming_movies = 'No internet service'
    else:
        online_security = random.choice(['Yes', 'No'])
        online_backup = random.choice(['Yes', 'No'])
        device_protection = random.choice(['Yes', 'No'])
        tech_support = random.choice(['Yes', 'No'])
        streaming_tv = random.choice(['Yes', 'No'])
        streaming_movies = random.choice(['Yes', 'No'])
    
    contract = random.choice(['Month-to-month', 'One year', 'Two year'])
    paperless_billing = random.choice(['Yes', 'No'])
    payment_method = random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    
    tenure = generate_tenure()
    
    services = {
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract
    }
    
    monthly_charges = generate_monthly_charges(services)
    total_charges = generate_total_charges(monthly_charges, tenure)
    churn = generate_churn(services, tenure, monthly_charges)
    
    return {
        'customerID': generate_customer_id(),
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    }

# Generate 1000 customers
customers = [generate_customer() for _ in range(1000)]

# Create DataFrame
df = pd.DataFrame(customers)

# Save to CSV
df.to_csv('sample_data/telecom_customer_churn.csv', index=False) 