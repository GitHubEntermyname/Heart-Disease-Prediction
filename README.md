# ❤️ Heart Disease Prediction

### 📌 Overview
This project uses **machine learning** to predict the presence of heart disease based on clinical data.  
It applies **data preprocessing**, **feature scaling**, **model comparison**, and **Random Forest classification** to build a predictive model.  
Implemented in **Python** using **Google Colab**.

---

### ⚙️ Features
- Loads and processes heart disease dataset  
- Normalizes features using MinMaxScaler  
- Splits data into training and test sets  
- Compares multiple classifiers using LazyPredict  
- Trains a Random Forest model  
- Evaluates accuracy on training and test data  
- Saves and loads trained model using `pickle`  
- Accepts new input for prediction

---

### 🧪 Technologies Used
- Python (NumPy, Pandas, Scikit-learn)  
- LazyPredict for model benchmarking  
- RandomForestClassifier for final prediction  
- Google Colab for development and testing  
- Pickle for model serialization

---

### 📂 Files
- `Heart_Disease_Prediction.ipynb` → Main notebook  
- `heart_disease_data.csv` → Dataset (not included, add manually)  
- `.gitignore` → ignored files configuration  
- `.gitattributes` → repository attributes  
- `README.md` → project documentation  
- `Heart_Disease_Trained_Model_151` → saved model (optional)

---

### 🚀 How It Works
1. Load and inspect dataset  
2. Normalize selected features  
3. Split into training and test sets  
4. Compare classifiers using LazyClassifier  
5. Train Random Forest model  
6. Evaluate accuracy  
7. Save and reload model  
8. Predict heart disease for new input

---

### 📚 Required Libraries
```bash
pip install lazypredict
pip install scikit-learn==1.5.1
