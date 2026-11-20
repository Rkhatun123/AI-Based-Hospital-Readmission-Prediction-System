# 🏥 AI-Based Hospital Readmission Prediction System

An end-to-end Machine Learning project that predicts the likelihood of hospital readmission using patient-related data. The system uses trained machine learning models and a Flask-based web application to generate predictions through a simple user interface.

## 📌 Project Overview

Hospital readmissions are an important challenge in healthcare. Identifying patients who may require readmission can help healthcare professionals make better-informed decisions and plan appropriate follow-up care.

This project uses machine learning techniques to analyze patient data and predict the likelihood of hospital readmission.

The project covers the complete machine learning workflow, from data understanding and preprocessing to model training, evaluation, and web application deployment.

## ✨ Key Features

- **Data Understanding:** Explore and understand the hospital patient dataset.
- **Data Preprocessing:** Handle missing values and prepare data for machine learning.
- **Exploratory Data Analysis:** Analyze patterns, distributions, and relationships in the dataset.
- **Feature Engineering:** Prepare and transform relevant features for model training.
- **Model Training:** Train and compare multiple machine learning algorithms.
- **Model Evaluation:** Evaluate model performance and select the final model.
- **Prediction Web App:** Generate readmission predictions through a Flask-based interface.
- **Trained Model Integration:** Load saved machine learning models and use them for predictions.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Backend | Flask |
| Frontend | HTML, CSS |
| Development Environment | Jupyter Notebook, VS Code |

## 🤖 Machine Learning Workflow

The project follows these steps:

1. Data Understanding
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation and Comparison
7. Final Model Selection
8. Model Saving and Integration
9. Flask Web Application Deployment

## 📂 Project Structure

```text
AI-Based-Hospital-Readmission-Prediction-System/
│
├── 01_Data_Understanding.ipynb
├── 02_Data Preprocessing.ipynb
├── 03_Exploratory_Data_Analysis.ipynb
├── 04_Feature_Engineering.ipynb
├── 05_Model_Training.ipynb.ipynb
├── 06 — Model Evaluation & Final Model Selection.ipynb
│
├── 07_Deployment/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── artifacts/
│   ├── final_readmission_model.pkl
│   ├── model_input_columns.pkl
│   └── model_X_train.pkl
│
├── diabetic_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Rkhatun123/AI-Based-Hospital-Readmission-Prediction-System.git
```

### 2. Navigate to the Project Directory

```bash
cd AI-Based-Hospital-Readmission-Prediction-System
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python 07_Deployment/app.py
```

Open the local URL displayed in your terminal in your browser.

## 📊 Model Artifacts

The following trained artifacts are used by the deployment application:

- `final_readmission_model.pkl` – Final trained machine learning model.
- `model_input_columns.pkl` – Input feature column information.
- `model_X_train.pkl` – Training feature data used by the application.

## 🎯 Project Objective

The main objective of this project is to apply machine learning techniques to healthcare data and develop a practical prediction application that demonstrates the complete ML development lifecycle.

## ⚠️ Disclaimer

This project is developed for educational and portfolio purposes. Its predictions are not intended to replace professional medical judgment or clinical decision-making.

## 👩‍💻 Author

**Rani Khatun**

- GitHub: [Rkhatun123](https://github.com/Rkhatun123)

---

⭐ If you find this project interesting, consider giving the repository a star!
