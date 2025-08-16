# 🚢 Titanic Survival Prediction

An interactive web application for predicting Titanic passenger survival using machine learning. Built with Python, Streamlit, and scikit-learn.

## 🎯 Project Overview

This project analyzes the historic Titanic disaster dataset to predict passenger survival based on various features like age, sex, passenger class, and more. The application provides:

- **Interactive Predictions**: Input passenger details to get survival probability
- **Data Visualization**: Comprehensive charts and analysis
- **Model Performance**: Detailed metrics and feature importance
- **Educational Tool**: Learn about machine learning concepts

## 🚀 Features

- **📊 Data Analysis Dashboard**: Explore passenger demographics and survival patterns
- **🔮 Survival Predictor**: Real-time predictions with probability scores
- **📈 Model Performance**: View accuracy, precision, recall, and F1-score
- **🎨 Interactive Visualizations**: Plotly charts and correlation matrices
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Logistic Regression with SMOTE
- **Data Processing**: Pandas, NumPy

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   └── test_titanic_app_fixed.py # Test file
│
├── Data/
│   ├── tested_cleaned.csv        # Cleaned dataset
│   └── Original/                 # Original dataset files
│ 
├── doc/
│   └── test_cases.md             # Test cases
│ 
├── models/
│   └── logistic_regression_smote.pkl  # Trained ML model
│
├── requirements/
│   └── requirements.txt          # Python dependencies
│
├── README.md                     # Project documentation
└── .gitignore                   # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements/requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app/streamlit_app.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`

## 📊 Dataset Information

The dataset contains information about Titanic passengers including:
- **Passenger Class** (1st, 2nd, 3rd)
- **Age** (0-80 years)
- **Siblings/Spouses Aboard**
- **Parents/Children Aboard**
- **Fare** (Ticket price)
- **Port of Embarkation** (Cherbourg, Queenstown, Southampton)

## 🤖 Model Details

- **Algorithm**: XGBoost Classifier (tree-based ensemble model)
- **Accuracy**: Displayed in the Model Performance section
- **Features**: 6 numerical features after preprocessing
- **Target**: Binary classification (Survived/Not Survived)
- **Feature Engineering**: Categorical variables encoded as numerical values

## 🎯 Usage Guide

### For Data Analysis
1. Navigate to "Data Analysis" tab
2. Explore passenger demographics
3. View survival patterns by class, age, and other features
4. Analyze correlation between features

### For Predictions
1. Go to "Prediction" tab
2. Enter passenger details:
   - Passenger class (1st, 2nd, 3rd)
   - Age
   - Family members aboard
   - Ticket fare
   - Port of embarkation
3. Click "Predict Survival"
4. View prediction and probability gauge

### For Model Evaluation
1. Visit "Model Performance" tab
2. View accuracy metrics
3. Analyze confusion matrix
4. Check feature importance

## 📈 Performance Metrics

The model performance is evaluated using:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 🔧 Development

### Adding New Features
1. Modify `streamlit_app.py` for UI changes
2. Update model training pipeline in notebooks
3. Add new visualizations in relevant sections

### Model Retraining
1. Use notebooks in the notebooks/ directory
2. Save new model to `models/` directory
3. Update model loading in Streamlit app

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## � Acknowledgments

- Titanic dataset from Kaggle
- Streamlit community for excellent documentation
- scikit-learn for machine learning algorithms
- Plotly for interactive visualizations

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.
