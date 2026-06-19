<div align="center">
  <h1>HR Employee Attrition Prediction 🕵️‍♂️</h1>
  <p>
    <i>Predict employee turnover using Logistic Regression & Machine Learning</i>
  </p>
  
  <p>
    <a href="#-project-overview">Overview</a> •
    <a href="#-features">Features</a> •
    <a href="#-tech-stack">Tech Stack</a> •
    <a href="#-getting-started">Getting Started</a> •
    <a href="#-model-details">Model Details</a>
  </p>
</div>

<hr />

## 🎯 Project Overview

Employee attrition is a critical challenge for modern HR departments. This project leverages **Logistic Regression** to predict the likelihood of an employee leaving the company based on various demographic, job-related, and satisfaction features. 

The project includes exploratory data analysis, model training, and an interactive **Streamlit web application** that allows users to input employee details and receive real-time attrition probability predictions.

<br />

## ✨ Features

<ul>
  <li><b>Interactive Dashboard:</b> A clean, user-friendly UI built with Streamlit.</li>
  <li><b>Real-Time Predictions:</b> Instantly calculates the probability of attrition based on user inputs.</li>
  <li><b>Comprehensive Input Features:</b> Considers age, income, job role, travel frequency, satisfaction levels, and more.</li>
  <li><b>Model Interpretability:</b> Displays clear metrics including Model Accuracy, Precision, Recall, and F1-Score.</li>
</ul>

<br />

## 🛠 Tech Stack

<table>
  <tr>
    <th>Category</th>
    <th>Technologies</th>
  </tr>
  <tr>
    <td><b>Language</b></td>
    <td>Python</td>
  </tr>
  <tr>
    <td><b>Web Framework</b></td>
    <td>Streamlit</td>
  </tr>
  <tr>
    <td><b>Machine Learning</b></td>
    <td>Scikit-Learn, Pandas, NumPy</td>
  </tr>
  <tr>
    <td><b>Data Visualization</b></td>
    <td>Matplotlib, Seaborn</td>
  </tr>
</table>

<br />

## 🚀 Getting Started

Follow these steps to run the project locally.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd LogisticRegression
```

### 2. Install dependencies
Make sure you have Python installed. Install the required packages using:
```bash
pip install -r requirements.txt
```

### 3. Run the application
Start the Streamlit server:
```bash
streamlit run app.py
```
The app will open automatically in your default web browser at `http://localhost:8501`.

<br />

## 🧠 Model Details

The core of this application is a **Logistic Regression** model trained on the `hremployees.csv` dataset. The data pipeline includes:
- **Data Preprocessing:** Handling missing values and encoding categorical variables using Label Encoding / One-Hot Encoding.
- **Feature Scaling:** Standardizing numerical features using `StandardScaler` to improve model convergence and performance.
- **Evaluation Metrics:** 
  - Accuracy: `~87%`
  - Precision: `~82%`
  - Recall: `~75%`
  - F1-Score: `~78%`

<br />

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit application
├── hremployees.csv             # HR dataset used for training
├── LogisticRegression.ipynb    # Jupyter Notebook for model training
├── HrAnalaytics.ipynb          # Jupyter Notebook for EDA
├── requirements.txt            # Project dependencies
└── *.pkl files                 # Serialized model, scaler, and encoders
```

<hr />

<div align="center">
  <p>Made with ❤️ for Data Science</p>
</div>
