import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('../Data/tested_cleaned.csv')

@st.cache_resource
def load_model():
    return joblib.load('../models/logistic_regression_smote.pkl')

# Initialize
df = load_data()
model = load_model()

# Header
st.markdown('<h1 class="main-header">🚢 Titanic Survival Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Data Analysis", "Prediction", "Model Performance"])

if page == "Home":
    st.markdown('<h2 class="sub-header">Welcome to Titanic Survival Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Passengers", len(df))
    
    with col2:
        survival_rate = (df['Survived'].sum() / len(df)) * 100
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
    
    with col3:
        st.metric("Features", len(df.columns) - 1)
    
    st.markdown("""
    ### About This Project
    This interactive dashboard analyzes the Titanic disaster dataset and provides survival predictions using machine learning.
    
    ### Features:
    - 📊 **Data Analysis**: Explore passenger demographics and survival patterns
    - 🔮 **Prediction**: Predict survival probability for new passengers
    - 📈 **Model Performance**: View model metrics and visualizations
    
    ### Dataset Overview:
    - **Passenger Class**: 1st, 2nd, 3rd class
    - **Sex**: Male (0) or Female (1)
    - **Age**: Passenger age
    - **Siblings/Spouses**: Number aboard
    - **Parents/Children**: Number aboard
    - **Fare**: Ticket price
    - **Embarked**: Port of embarkation
    """)

elif page == "Data Analysis":
    st.markdown('<h2 class="sub-header">📊 Data Analysis & Visualizations</h2>', unsafe_allow_html=True)
    
    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Survival by Class**")
        survival_by_class = df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
        survival_by_class.columns = ['Total', 'Survived', 'Rate']
        st.dataframe(survival_by_class)
    
    with col2:
        st.write("**Age Statistics**")
        age_stats = df['Age'].describe()
        st.dataframe(age_stats)
    
    # Visualizations
    st.subheader("Visualizations")
    
    # 1. Survival by Class
    fig1 = px.histogram(df, x='Pclass', color='Survived', 
                        title='Survival by Passenger Class',
                        labels={'Pclass': 'Passenger Class', 'Survived': 'Survived'},
                        color_discrete_map={0: '#ff7f0e', 1: '#2ca02c'})
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Age Distribution
    fig2 = px.histogram(df, x='Age', color='Survived',
                        title='Age Distribution by Survival',
                        nbins=30,
                        color_discrete_map={0: '#ff7f0e', 1: '#2ca02c'})
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Fare Distribution
    fig3 = px.box(df, x='Survived', y='Fare',
                  title='Fare Distribution by Survival',
                  labels={'Survived': 'Survived', 'Fare': 'Fare'},
                  color='Survived',
                  color_discrete_map={0: '#ff7f0e', 1: '#2ca02c'})
    st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Correlation Heatmap
    st.subheader("Feature Correlation")
    corr_matrix = df.corr()
    fig4 = px.imshow(corr_matrix,
                     labels=dict(x="Features", y="Features", color="Correlation"),
                     x=corr_matrix.columns,
                     y=corr_matrix.columns,
                     color_continuous_scale='RdBu_r',
                     aspect="auto")
    fig4.update_layout(title='Feature Correlation Matrix')
    st.plotly_chart(fig4, use_container_width=True)

elif page == "Prediction":
    st.markdown('<h2 class="sub-header">🔮 Survival Prediction</h2>', unsafe_allow_html=True)
    
    st.write("Enter passenger details to predict survival probability:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: ['Male', 'Female'][x])
        age = st.slider("Age", 0, 80, 30)
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    
    with col2:
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 600.0, 32.0)
        embarked = st.selectbox("Port of Embarkation", [0, 1, 2], format_func=lambda x: ['Cherbourg', 'Queenstown', 'Southampton'][x])
    
    if st.button("Predict Survival", type="primary"):
        # Prepare input
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [embarked]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", "Survived" if prediction == 1 else "Not Survived")
        
        with col2:
            st.metric("Survival Probability", f"{probability[1]:.1%}")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Survival Probability"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 100], 'color': "lightgreen"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 50}}))
        
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)
    
    # Load test data for evaluation
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = accuracy_score(y, y_pred)
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y, y_pred)
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        recall = recall_score(y, y_pred)
        st.metric("Recall", f"{recall:.3f}")
    
    with col4:
        f1 = f1_score(y, y_pred)
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    
    fig_cm = px.imshow(cm,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Not Survived', 'Survived'],
                       y=['Not Survived', 'Survived'],
                       color_continuous_scale='Blues',
                       text_auto=True)
    fig_cm.update_layout(title='Confusion Matrix')
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Feature Importance
    st.subheader("Feature Importance")
    feature_names = X.columns
    coefficients = model.coef_[0]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', ascending=True)
    
    fig_importance = px.bar(importance_df, x='Coefficient', y='Feature',
                           orientation='h', title='Feature Importance (Logistic Regression Coefficients)')
    st.plotly_chart(fig_importance, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Created with ❤️ using Streamlit**")
