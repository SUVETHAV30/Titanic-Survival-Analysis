import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')



def load_data():
    """Load and combine training and test datasets."""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    gender_submission = pd.read_csv('gender_submission.csv')
    return train_df, test_df, gender_submission

def preprocess_data(df):
    """Preprocess the data by handling missing values and feature engineering."""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Fill missing age values with median age
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Fill missing embarked values with most common port
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Fill missing fare values with median fare
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Create new features
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Convert categorical variables to numeric
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Title'] = df['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})
    
    # Select features for model
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize']
    return df[features]

def create_visualizations(train_df):
    """Create and save visualizations for key insights."""
    # Create output directory for plots
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Survival by class
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Pclass', y='Survived', data=train_df)
    plt.title('Survival Rate by Passenger Class')
    plt.savefig('plots/survival_by_class.png')
    plt.close()
    
    # Survival by gender
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sex', y='Survived', data=train_df)
    plt.title('Survival Rate by Gender')
    plt.savefig('plots/survival_by_gender.png')
    plt.close()
    
    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_df, x='Age', hue='Survived', multiple="stack")
    plt.title('Age Distribution by Survival')
    plt.savefig('plots/age_distribution.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    sns.heatmap(train_df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nModel Performance Metrics:")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    return accuracy

def evaluate_gender_baseline(train_df):
    """Evaluate the gender-based baseline model."""
    # Create gender-based predictions (all females survive, all males die)
    gender_predictions = (train_df['Sex'] == 'female').astype(int)
    actual = train_df['Survived']
    
    # Calculate accuracy
    accuracy = accuracy_score(actual, gender_predictions)
    
    print("\nGender Baseline Model Performance:")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(actual, gender_predictions))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(actual, gender_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Gender Baseline Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/gender_baseline_confusion_matrix.png')
    plt.close()
    
    return accuracy

def main():
    # Load data
    train_df, test_df, gender_submission = load_data()
    
    # Create visualizations
    create_visualizations(train_df)
    
    # Evaluate gender baseline
    gender_baseline_accuracy = evaluate_gender_baseline(train_df)
    
    # Preprocess data
    X = preprocess_data(train_df)
    y = train_df['Survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate model
    model = train_model(X_train_scaled, y_train)
    rf_accuracy = evaluate_model(model, X_test_scaled, y_test)
    
    # Compare models
    print("\nModel Comparison:")
    print(f"Gender Baseline Accuracy: {gender_baseline_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Improvement: {(rf_accuracy - gender_baseline_accuracy)*100:.2f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    print("\nFeature Importance:")
    print(feature_importance)

if __name__ == "__main__":
    main() 