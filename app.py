import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Titanic Survival Analysis",
    page_icon="🚢",
    layout="wide"
)

# Title and description
st.title("🚢 Titanic Survival Analysis")
st.markdown("""
This interactive dashboard analyzes the Titanic dataset to predict passenger survival using machine learning techniques.
Explore the data, view visualizations, and make predictions!
""")

# Load data
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

# Preprocess data
def preprocess_data(df):
    df = df.copy()
    
    # Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Create new features
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Convert categorical variables
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Title'] = df['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})
    
    return df

# Load and preprocess data
train_df, test_df = load_data()
train_processed = preprocess_data(train_df)

# Initialize model and scaler
X = train_processed[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize']]
y = train_df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Visualizations", "Model Analysis", "Make Predictions"])

if page == "Data Overview":
    st.header("Dataset Overview")
    
    # Show basic statistics
    st.subheader("Basic Statistics")
    st.write(train_df.describe())
    
    # Show missing values
    st.subheader("Missing Values")
    missing_data = pd.DataFrame({
        'Missing Values': train_df.isnull().sum(),
        'Percentage': (train_df.isnull().sum() / len(train_df)) * 100
    })
    st.write(missing_data)
    
    # Show raw data
    st.subheader("Raw Data")
    st.dataframe(train_df)

elif page == "Visualizations":
    st.header("Data Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Survival by Class", "Age Distribution", "Gender Analysis", "Correlation Heatmap"])
    
    with tab1:
        st.subheader("Survival Rate by Passenger Class")
        survival_by_class = train_df.groupby('Pclass')['Survived'].mean().reset_index()
        fig = px.bar(survival_by_class, x='Pclass', y='Survived',
                    title='Survival Rate by Passenger Class',
                    labels={'Pclass': 'Passenger Class', 'Survived': 'Survival Rate'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Age Distribution by Survival")
        fig = px.histogram(train_df, x='Age', color='Survived',
                          title='Age Distribution by Survival Status',
                          barmode='overlay',
                          labels={'Age': 'Age', 'Survived': 'Survival Status'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Survival Rate by Gender")
        survival_by_gender = train_df.groupby('Sex')['Survived'].mean().reset_index()
        fig = px.bar(survival_by_gender, x='Sex', y='Survived',
                    title='Survival Rate by Gender',
                    labels={'Sex': 'Gender', 'Survived': 'Survival Rate'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Correlation Heatmap")
        numeric_cols = train_processed.select_dtypes(include=[np.number]).columns
        corr = train_processed[numeric_cols].corr()
        fig = px.imshow(corr, 
                       title='Feature Correlation Heatmap',
                       labels=dict(x="Features", y="Features", color="Correlation"))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Analysis":
    st.header("Model Analysis")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Show metrics
    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(feature_importance, x='importance', y='feature',
                 title='Feature Importance',
                 orientation='h',
                 labels={'importance': 'Importance', 'feature': 'Feature'})
    st.plotly_chart(fig, use_container_width=True)

else:  # Make Predictions
    st.header("Make Predictions")
    
    # Create input form
    st.subheader("Enter Passenger Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Gender", ["male", "female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        fare = st.number_input("Fare", min_value=0.0, value=50.0)
    
    with col2:
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
        title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])
        family_size = st.number_input("Family Size", min_value=1, max_value=10, value=1)
    
    # Preprocess input
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [1 if sex == 'female' else 0],
        'Age': [age],
        'Fare': [fare],
        'Embarked': [0 if embarked == 'S' else 1 if embarked == 'C' else 2],
        'Title': [1 if title == 'Mr' else 2 if title == 'Miss' else 3 if title == 'Mrs' else 4 if title == 'Master' else 5],
        'FamilySize': [family_size]
    })
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    if st.button("Predict Survival"):
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("The passenger is predicted to survive!")
        else:
            st.error("The passenger is predicted to not survive.")
        
        st.write(f"Survival Probability: {probability[0][1]:.2%}") 