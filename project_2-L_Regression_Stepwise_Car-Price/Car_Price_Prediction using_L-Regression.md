
<br>


# 🚗 Car Price Prediction using Linear Regression

This repository presents a complete **Linear Regression analysis** for predicting car prices based on multiple vehicle attributes. The project applies **stepwise variable selection**, statistical validation, and exploratory data analysis to build a robust and interpretable model.

<br><br>

## 📌 Project Overview

The objective of this project is to explain and predict **car prices** using a linear regression model, where price is influenced by technical and usage-related variables.  

The analysis focuses not only on model performance, but also on **statistical significance, correlation, and residual validation**, ensuring methodological rigor.

<br><br>


## 🧠 Use Case

- Predict car prices based on vehicle characteristics  
- Understand which variables most strongly influence price  
- Apply statistical tests to validate model assumptions  

<br><br>


## 📂 Dataset

The dataset is included in the repository and contains the following variables:

<br>

| Variable | Description |
|--------|------------|
| `Age` | Vehicle age (years) |
| `Engine_Size` | Engine displacement (liters) |
| `Electric` | Electric vehicle indicator (0 = No, 1 = Yes) |
| `Mileage` | Total kilometers driven |
| `Doors` | Number of doors |
| `Previous_Owners` | Number of previous owners |
| `Price` | **Target variable** |

The data is loaded directly from an Excel file within the notebook.

<br><br>

## 🛠️ Technologies & Libraries

- Python  
- Pandas  
- NumPy  
- Statsmodels  
- SciPy  
- Matplotlib  
- Seaborn  

<br><br>


## 🔍 Methodology

### 1. Data Preparation
- Load dataset from Excel
- Remove missing or invalid values
- Prepare independent and dependent variables

<br>

### 2. Stepwise Variable Selection
A **stepwise regression approach** is used to:
- Iteratively add statistically significant variables
- Remove variables that lose significance
- Optimize explanatory power while avoiding overfitting

<br>

### 3. Regression Model
- Ordinary Least Squares (OLS)
- Intercept included
- Final model based on selected predictors

<br><br>

## 📊 Model Evaluation

The following statistical metrics are analyzed:

- **Correlation Matrix**
- **Intercept**
- **Multiple R**
- **R-Squared**
- **Adjusted R-Squared**
- **ANOVA (F-Test)**

### Residual Diagnostics
- **Shapiro-Wilk Test** applied to verify normality of residuals

<br><br>

## 📈 Data Visualization

The project includes visual analysis to support interpretation:

- Scatter Plot: **Price vs. Engine Size**
- Scatter Plot: **Price vs. Mileage**
- Correlation Heatmap of all variables

These plots help identify trends, outliers, and relationships between predictors and price.

<br><br>

## ✅ Results & Findings

- The stepwise method successfully identified the most relevant predictors
- The model achieved a **strong explanatory fit**
- ANOVA confirmed overall statistical significance
- Residuals follow a normal distribution, supporting model validity

<br><br>

## 📌 Recommendations

- Use this model as a baseline for car price estimation
- Periodically retrain the model with updated data
- Extend analysis with nonlinear models or interaction terms if needed

<br><br>

## 📁 Project Structure

├── car_price_prediction.ipynb
├── car_prices.xlsx
├── README.md


<br><br>


## 👩‍💻 Author

**Fabiana Campanari**  
AI/ML Engineer • Data Scientist 🚀  

<br><br>

## 📄 License

This project is intended for educational and analytical purposes.

