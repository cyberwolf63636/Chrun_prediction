import os
import streamlit as st
import json
import hashlib
import requests

# Firebase configuration
def get_firebase_config():
    """Get Firebase configuration from environment variables or use defaults for development"""
    config = {
        "apiKey": "AIzaSyDCcA9AERaSNEQunvIqMUWjsDMwVTeCRPo",
        "authDomain": "deeksha-english-bot.firebaseapp.com",
        "databaseURL": "https://deeksha-english-bot-default-rtdb.asia-southeast1.firebasedatabase.app",
        "projectId": "deeksha-english-bot",
        "storageBucket": "deeksha-english-bot.firebasestorage.app",
        "messagingSenderId": "343535445396",
        "appId": "1:343535445396:web:acd2003d31291aa2e91b81"
    }
    return config

def initialize_firebase():
    """Initialize Firebase configuration"""
    return get_firebase_config()

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(email, password):
    """Authenticate a user with Firebase or simulated authentication"""
    if not email or not password:
        return False, "Email and password cannot be empty"
    
    # Input validation
    if '@' not in email or '.' not in email:
        return False, "Please enter a valid email address"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    # Fixed demo accounts for testing - these always work
    if email == "demo@example.com" and password == "password123":
        return True, {"email": email}
    elif email == "user@example.com" and password == "user123":
        return True, {"email": email}
    
    # Maximum retry attempts for network issues
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Get Firebase config
            config = get_firebase_config()
            
            # Firebase Auth REST API endpoint for email/password sign in
            auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={config['apiKey']}"
            
            # Prepare the request payload
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            # Make the authentication request with timeout
            response = requests.post(auth_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                # Authentication successful
                auth_data = response.json()
                # Store the authentication token in session state
                st.session_state.auth_token = auth_data.get("idToken")
                return True, {"email": email, "idToken": auth_data.get("idToken")}
            else:
                # Authentication failed - handle specific error cases
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "")
                
                if "EMAIL_NOT_FOUND" in error_message:
                    return False, "Account not found. Please check your email or sign up for a new account."
                elif "INVALID_PASSWORD" in error_message:
                    return False, "Incorrect password. Please try again or reset your password."
                elif "TOO_MANY_ATTEMPTS_TRY_LATER" in error_message:
                    return False, "Too many failed attempts. Please try again later."
                elif "USER_DISABLED" in error_message:
                    return False, "This account has been disabled. Please contact support."
                else:
                    return False, "Authentication failed. Please verify your credentials and try again."
                    
        except requests.exceptions.Timeout:
            if retry_count < max_retries - 1:
                retry_count += 1
                continue
            return False, "Authentication request timed out. Please check your internet connection and try again."
            
        except requests.exceptions.ConnectionError:
            if retry_count < max_retries - 1:
                retry_count += 1
                continue
            return False, "Unable to connect to authentication service. Please check your internet connection."
            
        except requests.exceptions.RequestException as e:
            # Log the error for debugging
            print(f"Authentication error: {str(e)}")
            
            # Fall back to session state users if Firebase is unavailable
            if 'users' not in st.session_state:
                st.session_state.users = {}
            
            hashed_password = hash_password(password)
            if email in st.session_state.users and st.session_state.users[email] == hashed_password:
                return True, {"email": email}
            
            return False, "An error occurred during authentication. Please try again."
        
        break  # Break the loop if request was successful
    
    return False, "Authentication service is currently unavailable. Please try again later."

def create_user(email, password):
    """Create a new user"""
    if not email or not password:
        return False, "Email and password cannot be empty"
    
    if '@' not in email or '.' not in email:
        return False, "Invalid email format"
    
    if len(password) < 6:
        return False, "Password should be at least 6 characters"
    
    # Initialize users dict if not exists
    if 'users' not in st.session_state:
        st.session_state.users = {}
    
    # Prevent overwriting demo accounts
    if email == "demo@example.com" or email == "user@example.com":
        return False, "This email is reserved. Please use a different email."
    
    # Check if user already exists
    if email in st.session_state.users:
        return False, "Email already exists. Please use a different email."
    
    # Store user (in a real app, this would use Firebase API)
    st.session_state.users[email] = hash_password(password)
    return True, {"email": email}

def login_form():
    """Display login form and handle login process"""
    st.markdown("""
        <style>
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('🔐 <h2 style="color: #2196F3;">Login</h2>', unsafe_allow_html=True)
    
    # Demo credentials info with icon
    st.info('🔑 Welcome To Cyber Wolf Team')
    
    email = st.text_input('📧 Email', key='login_email')
    password = st.text_input('🔒 Password', type='password', key='login_password')
    
    if st.button('🚀 Login'):
        if not email or not password:
            st.error("Please enter both email and password.")
            return
        
        with st.spinner("Logging in..."):
            success, result = authenticate_user(email, password)
        
        if success:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.success('🎉 Login successful!')
            st.rerun()
        else:
            st.error(result)

def signup_form():
    """Display signup form and handle signup process"""
    st.markdown("""
        <style>
        .signup-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="signup-container">', unsafe_allow_html=True)
    st.markdown('✨ <h2 style="color: #2196F3;">Create an Account</h2>', unsafe_allow_html=True)
    
    email = st.text_input('📧 Email', key='signup_email')
    password = st.text_input('🔒 Password', type='password', key='signup_password')
    confirm_password = st.text_input('🔐 Confirm Password', type='password', key='signup_confirm_password')
    
    if st.button('✅ Sign Up'):
        if not email or not password or not confirm_password:
            st.error("Please fill in all fields.")
            return
        
        if password != confirm_password:
            st.error("Passwords do not match.")
            return
        
        with st.spinner("Creating account..."):
            success, result = create_user(email, password)
        
        if success:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.success('🎉 Account created successfully!')
            st.rerun()
        else:
            st.error(result)

def logout():
    """Log out the current user"""
    st.session_state.authenticated = False
    st.session_state.user_email = None
    # Clear other session state variables
    for key in ['data', 'model', 'predictions', 'preprocessed_data', 'features', 'metrics', 
                'ollama_analysis', 'segments', 'speech_summary', 'demographic_data']:
        if key in st.session_state:
            st.session_state[key] = None
