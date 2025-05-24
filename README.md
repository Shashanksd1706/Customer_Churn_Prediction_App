# 📉 Customer Churn Prediction App

An end-to-end machine learning web application that predicts whether a customer will churn based on their demographic and financial data. Built using Streamlit, this app utilizes a trained Artificial Neural Network (ANN) and is equipped with pre-processing pipelines, a user-friendly frontend, and supporting notebooks for experimentation and reproducibility.
Website link -https://shashanksd1706-customer-churn-prediction-app-app-cyd1z9.streamlit.app/

---

## 📂 Directory Structure

```
Customer_Churn_Prediction_App/
|
├── app.py                        # Main Streamlit application script
├── requirements.txt             # List of required Python packages
├── Churn_Modelling.csv          # Dataset used for training/testing
|
├── model.h5                     # Trained ANN model saved in HDF5 format
├── label_encoder_gender.pkl     # Label encoder for 'Gender'
├── onehot_encoder.pkl           # One-hot encoder for 'Geography'
├── scaler.pkl                   # Standard scaler for feature normalization
|
├── experiments.ipynb            # Jupyter Notebook for EDA and model training
├── prediction.ipynb             # Notebook to test model predictions using serialized files
|
└── templates/
    └── index.html               # Frontend HTML interface rendered by Flask
```

---

## 🚀 Getting Started

Follow these instructions to set up and run the churn prediction app on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/Shashanksd1706/Customer_Churn_Prediction_App.git
cd Customer_Churn_Prediction_App
```

### 2. Create and Activate Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
env\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask App

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser to interact with the application.

---

## 🔢 Dataset Overview

The application uses the `Churn_Modelling.csv` dataset. It includes information on 10,000 bank customers with the following features:

* **CreditScore**: Credit score of the customer
* **Geography**: Country of residence (France, Germany, Spain)
* **Gender**: Male or Female
* **Age**: Age of the customer
* **Tenure**: Years of association with the bank
* **Balance**: Account balance
* **NumOfProducts**: Number of bank products used
* **HasCrCard**: Whether customer has a credit card
* **IsActiveMember**: Customer activity status
* **EstimatedSalary**: Annual salary estimate
* **Exited**: Target variable (1 = Churned, 0 = Retained)

---

## 🧠 Model Summary

* **Architecture**: Deep Artificial Neural Network (ANN) with multiple dense layers
* **Input Preprocessing**:

  * Gender encoding with `LabelEncoder`
  * Geography encoding with `OneHotEncoder`
  * Feature scaling with `StandardScaler`
* **Activation Functions**: ReLU (hidden layers), Sigmoid (output layer)
* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam
* **Output**: Probability of churn (converted to binary class)

---

## 🕹️ Web Application Workflow

1. User fills in customer details via the web form
2. Input is transformed using serialized encoders and scaler
3. Trained ANN model (`model.h5`) predicts churn probability
4. Result is rendered back on the browser ("Likely to Churn" / "Likely to Stay")

### Example Inputs:

```
Credit Score: 600
Geography: France
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Num of Products: 2
Has Credit Card: Yes
Is Active Member: No
Estimated Salary: 50000
```

### Output:

```
Prediction: This customer is likely to STAY.
```

---

## 📊 Notebooks for Analysis

### `experiments.ipynb`

* Data exploration and visualization
* Feature engineering and encoding
* ANN training using Keras
* Evaluation metrics (accuracy, confusion matrix, etc.)

### `prediction.ipynb`

* Demonstrates how to use saved encoders and model to make predictions
* Useful for testing outside the web app context

---

## 🛠️ Tools & Technologies

* **Backend**: Python, Flask
* **ML/DL**: TensorFlow, Keras, Scikit-learn
* **Data Handling**: Pandas, NumPy
* **Frontend**: HTML, CSS (Jinja2 templating)
* **Environment Management**: venv

---

## 📊 Performance & Results

* **Test Accuracy**: \~86%
* **Balanced Precision/Recall**: Ensures fairness in churn prediction
* **Scalable Preprocessing**: Uses saved .pkl files for production reliability

---

## 🚜 Future Enhancements

* Model interpretability with SHAP/LIME
* Feature to download results as CSV
* User registration/authentication
* Docker containerization and cloud deployment (Heroku, AWS EC2)
* Dynamic visual dashboards (Plotly, Dash, or Streamlit)

---

## 💼 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute.

---

## 🙋‍♂️ Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

If you find a bug or want to request a feature, open an issue [here](https://github.com/Shashanksd1706/Customer_Churn_Prediction_App/issues).

---

## 🚀 Author

**Shashank S D**
GitHub: [@Shashanksd1706](https://github.com/Shashanksd1706)
LinkedIn: *(www.linkedin.com/in/shashank-dwivedi-91508a245)*
