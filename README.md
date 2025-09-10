# county-mobility-analysis
## **Overview**
Transportation agencies allocate billions in infrastructure funding without precise data on which counties generate the most long-distance travel demand. This project analyzes Bureau of Transportation Statistics (BTS) data to predict long-distance travel behavior across Washington state counties using machine learning regression algorithms.
The analysis enables data-driven decisions for highway capacity planning, regional transit optimization, and transportation budget allocation across Washington's 39 counties.
## **Key Findings**
- Model Performance: Random Forest achieved 98.3% R² accuracy, dramatically outperforming linear methods (23-32% accuracy)
- Geographic Patterns: Rural counties average 15.6% long-distance trips vs 6.2% for urban counties
- Infrastructure Priorities: 33% of counties need high-priority highway investments, 49% should focus on local transit
- Predictive Insights: Medium-distance trips (3-5 miles) are the strongest predictor of long-distance travel propensity
- Budget Framework: Clear county classification enables evidence-based transportation resource allocation
## **Dataset**
- Source: [Trips by Distance Dataset](https://catalog.data.gov/dataset/trips-by-distance)
- Scope: Washington state county-level data for 2023
- Size: 14,231 clean records from 6M+ original dataset
- Features: Trip counts across 10 distance categories, population metrics, temporal features
- Target: Long-distance trip percentage (>25 miles), ranging from 1.3% to 42.1%
## **Models Implemented**
- Linear Regression (with regularization)
- Support Vector Regression (Linear, RBF, Polynomial kernels)
- Decision Tree Regressor (with hyperparameter tuning)
- Random Forest Regressor
- Gradient Boosting Regressor
- Bagging Regressor
## **How to Run**
### Prerequisites
- Install required packages:
  - `pandas`
  - `numpy` 
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
### Setup
- Clone repository
- Sample dataset included for demonstration
- For full analysis, download complete dataset from [BTS Daily Mobility Statistics](https://catalog.data.gov/dataset/trips-by-distance)
### Execution
- Open `population_mobility.ipynb` in Jupyter Notebook
- Run all cells sequentially
