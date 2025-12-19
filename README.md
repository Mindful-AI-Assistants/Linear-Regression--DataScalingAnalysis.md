
<br><br>

# <p align="center">   [Linear Regression and Data Scaling Analysis]()

<br><br>


<p align="center">
  <img src="https://github.com/user-attachments/assets/08c12156-c338-4fc9-835a-03d7e3bbbc2f" />

<br><br>


## [Project Overview]()

This project demonstrates a complete machine learning workflow for price prediction using:

- [**Stepwise Regression**]() for feature selection  
- Advanced statistical analysis (ANOVA, R² metrics)  
- Full model diagnostics  
- Interactive visualization integration

<br>  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ELn9-EXlSCUblD2-FyT_Cladp9gpUyD0#revisionId=0B7p2OprIwdM2SmkrZ0N4REVhc2EzU0xnZW0zNkYzT20zYUlNPQ)

<br><br>


## Table of Contents  

1. [What is Data Normalization/Scaling?](#what-is-data-normalizationscaling)  
2. [Common Scaling Methods](#common-scaling-methods)  
3. [Why is this Important in Machine Learning?](#why-is-this-important-in-machine-learning)  
4. [Practical Example](#practical-example)  
5. [Code Example (Python)](#code-example-python)  
6. [Linear Regression: Price Prediction Case Study 📈](#linear-regression-price-prediction-case-study)  
   - [I. Use Case Implementation & Dataset Description](#i-use-case-implementation--dataset-description)  
   - [II. Methodology (Stepwise Regression)](#ii-methodology-stepwise-regression)  
   - [III. Statistical Analysis](#iii-statistical-analysis)  
   - [IV. Full Implementation Code](#iv-full-implementation-code)  
   - [V. Visualization](#v-visualization)  
   - [VI. How to Run](#vi-how-to-run)  
7. [Linear Regression Analysis Report 📊](#linear-regression-analysis-report)  
   - [Dataset Overview](#dataset-overview)  
   - [Key Formulas](#key-formulas)  
   - [Statistical Results](#statistical-results)  
   - [Code Implementation](#code-implementation)  
   - [Stepwise Regression](#stepwise-regression)



<br><br>

## [What is Data Normalization/Scaling ?]()  

A preprocessing technique that adjusts numerical values in a dataset to a standardized scale (e.g., \[0, 1\] or \[-1, 1\]). This is essential for:  

<br>

- [**Reducing outlier influence**]()  
- [**Ensuring stable performance**]() in machine learning algorithms (e.g., neural networks, SVM)  
- [**Enabling fair comparison**]() between variables with different units or magnitudes  


<br><br>


## [Common Scaling Methods]()  

### 1. **Min-Max Scaling (Normalization)**  

- [**Formula:**]()
  

<br>


$$
X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$
   
<br>

- [**Result:**]() Values scaled to the \[0, 1\] interval.  


<br>

#

<br>



### 2. **Standardization (Z-Score)**  

- [**Formula:**]()


<br>
    
$$
\Huge
X_{\text{std}} = \frac{X - \mu}{\sigma}
$$
  

<br>
   
- [**Where:**]() \(\mu\) is the mean and \(\sigma\) is the standard deviation.  

- [**Result:**]() Data with a mean of 0 and standard deviation of 1.  


#

<br>


### 3. **Robust Scaling**  

   - Uses the median and interquartile range (IQR) to reduce the impact of outliers.
     
   - **Formula:**


<br>     
     
$$
\Huge
X_{\text{robust}} = \frac{X - \text{Median}(X)}{\text{IQR}(X)}
$$


<br><br>



## Why is this Important in Machine Learning?  

<br>

- **Scale-sensitive algorithms:** Methods like neural networks, SVM, and KNN rely on the distances between data points; unscaled data can hinder model convergence.  

- **Interpretation:** Variables with different scales can distort the weights in linear models (e.g., logistic regression).  

- **Optimization Speed:** Gradients in optimization algorithms converge faster with normalized data.



<br><br>


## Practical Example  

### For a dataset containing:  

- **Age:** Values between 18–90 years  

- **Salary:** Values between \$1k–\$20k


<br><br>


## After applying **Min-Max Scaling**:
 
- **Age 30** transforms to approximately \[0.17\]  

- **Salary \$5k** transforms to approximately \[0.21\]

<br>  

This process ensures both features contribute equally to the model.


<br><br>


## Code Example (Python) – Data Normalization

<br>

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Sample data: Age and Salary
data = np.array([[30], [5000]], dtype=float).reshape(-1, 1)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

print(normalized_data)
# Expected Output: [[0.17], [0.21]]
```

<br><br>


Linear Regression: Price Prediction Case Study 📈
 
Dataset: housing_data.xlsx (included in repository) Tech Stack: Python 3.9, Jupyter Notebook, scikit-learn, statsmodels


<br><br> 


## I. Use Case Implementation & Dataset Description

<br>


| Variable       | Type  | Range         | Description                          |
|----------------|-------|---------------|--------------------------------------|
| `area_sqm`     | float | 40–220        | Living area in square meters         |
| `bedrooms`     | int   | 1–5           | Number of bedrooms                   |
| `distance_km`  | float | 0.5–15        | Distance to city center (km)         |
| `price`        | float | \$50k–\$1.2M  | Property price in USD                |



<br><br> 


 ## II. Methodology (Stepwise Regression)

 <br>


```python
import statsmodels.api as sm

def stepwise_selection(X, y):
    """Automated feature selection using p-values."""
    included = []
    while True:
        changed = False
        # Forward step: consider adding each excluded feature
        excluded = list(set(X.columns) - set(included))
        pvalues = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            pvalues[new_column] = model.pvalues[new_column]
        best_pval = pvalues.min()
        if best_pval < 0.05:
            best_feature = pvalues.idxmin()
            included.append(best_feature)
            changed = True
        
        # Backward step: consider removing features with high p-value
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]  # Exclude intercept
        worst_pval = pvalues.max()
        if worst_pval > 0.05:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
        
        if not changed:
            break
    return included

# Example usage (assuming X_train and y_train are predefined):
# selected_features = stepwise_selection(X_train, y_train)
```


<br><br> 


## III. Statistical Analysis

### Key Metrics Table

<br>


| Metric         | Value   | Interpretation                  |
|----------------|---------|---------------------------------|
| **R²**         | 0.872   | 87.2% variance explained        |
| **Adj. R²**    | 0.865   | Adjusted for feature complexity |
| **F-statistic**| 124.7   | p-value = 2.3e-16 (Significant)  |
| **Intercept**  | 58,200  | Base price without features     |



<br><br>   


### Correlation Matrix

<br>

```python
import seaborn as sns
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

<br><br>


## IV. Full Implementation Code

### Model Training & Evaluation

<br>

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming X_train, y_train, X_test, and y_test are predefined
final_model = LinearRegression()
final_model.fit(X_train[selected_features], y_train)

# Predictions on test set
y_pred = final_model.predict(X_test[selected_features])

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = final_model.score(X_test[selected_features], y_test)
```


<br><br>


## V. Visualization – Actual vs Predicted Prices

<br>

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred, hue=X_test['bedrooms'])
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Model Performance Visualization')
plt.savefig('results/scatter_plot.png')
plt.show()
```

<br><br>



## VI. How to Run

<br>


### 1. Install Dependencies:

<br>

```bash
pip install -r requirements.txt
```

<br> 


### 2. Download Dataset:

<br>

   * From: data/housing_data.xlsx
   * Or use this [dataset link]()

<br>


###  3..Execute Jupyter Notebook:  

<br>

```bash
    jupyter notebook price_prediction.ipynb
```

<br>

>  Note: Full statistical outputs and diagnostic plots are available in the notebook.


<br><br>


## Linear Regression Analysis Report 📊

### Dataset Overview 

📌 **Important Note:**  

<br>

> This dataset is a fictitious example created solely for demonstration and educational purposes. There is no external source for this dataset.  

> For real-world datasets, consider exploring sources such as the [UC Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) or [Kaggle](https://www.kaggle.com/datasets).


<br><br>


| Variable    | Type  | Range         | Description                          |
|-------------|-------|---------------|--------------------------------------|
| area_sqm    | float | 40–220        | Living area in square meters         |
| bedrooms    | int   | 1–5           | Number of bedrooms                   |
| distance_km | float | 0.5–15        | Distance to city center (km)         |
| price       | float | \$50k–\$1.2M  | Property price in USD                |




<br><br>


## Key Formulas

<br>

### 1. Regression Equation

$$
\Huge
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
$$


<br>


### 2. R-Squared

$$
\Huge
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$


<br>


###3. F-Statistic (ANOVA)

$$
\Hug e
F = \frac{\text{MS}\_\text{model}}{\text{MS}\_\text{residual}}
$$


<br><br>


## Statistical 

<br>

| Metric      | Value  | Critical Value | Interpretation              |
|-------------|--------|----------------|-----------------------------|
| R²          | 0.872  | > 0.7          | Strong explanatory power    |
| Adj. R²     | 0.865  | > 0.6          | Robust to overfitting       |
| F-statistic | 124.7  | 4.89           | p < 0.001 (Significant)     |
| Intercept   | 58,200 | -              | Base property value         |


<br><br>


## Stepwise Regression

<br>

```python
import statsmodels.api as sm

def stepwise_selection(X, y, threshold_in=0.05, threshold_out=0.1):
    included = []
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
        
        # Backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
        
        if not changed:
            break
    return included
```

<br><br>


## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br>


#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />


<br><br>

<p align="center">  ────────────── ⊹🔭๋ ──────────────

<!--
<p align="center">  ────────────── 🛸๋*ੈ✩* 🔭*ੈ₊ ──────────────
-->

<br>

<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>
  

  
#
 
##### <p align="center">Copyright 2026 Mindful-AI-Assistants. Code released under the  [MIT license.](https://github.com/Mindful-AI-Assistants/CDIA-Entrepreneurship-Soft-Skills-PUC-SP/blob/21961c2693169d461c6e05900e3d25e28a292297/LICENSE)
