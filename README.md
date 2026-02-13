#  Sales & Demand Forecasting using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)



##  Project Overview

This project develops a Sales Forecasting system using historical retail data.  
The objective is to predict future sales trends and support business decision-making using Machine Learning.

Sales forecasting is a critical real-world ML application used in retail, e-commerce, and supply chain industries.



## Business Problem

Businesses need accurate forecasts to:

- Optimize inventory levels  
- Reduce overstock & shortages  
- Plan workforce requirements  
- Forecast revenue & cash flow  

This project demonstrates how Machine Learning can support strategic business planning.



## Data Preparation

- Converted date column to datetime format  
- Aggregated daily total sales  
- Sorted data chronologically  
- Created time-based feature (Time Index)  
- Split dataset into training and testing sets  



##  Model Implementation

**Algorithm Used:** Linear Regression  

The model was trained on historical sales trends and used to forecast future demand.



## Model Performance

| Metric | Value |
|--------|--------|
| MAE    | 157,340 |
| RMSE   | 180,406 |

The model captures the overall sales growth pattern while accounting for daily fluctuations.


## Visual Results

### 1️⃣ Forecast vs Actual Sales
Shows model accuracy on test data.

### 2️⃣ 30-Day Future Forecast
Predicts upcoming sales trend based on learned patterns.



## Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  


