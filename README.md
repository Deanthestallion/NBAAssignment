# NBA Player Longevity Prediction – Feature Engineering

## Project Overview
This project focuses on analyzing an NBA player performance dataset to engineer features for a machine learning model. The objective is to forecast player longevity by classifying whether a player's career will last 5 years or more (`target_5yrs`). By performing data exploration, cleaning, scaling, and feature engineering, we convert raw sports statistics into a robust, ML-ready dataset.

## Dataset Source
The dataset contains historical NBA player performance metrics, including games played, minutes, points, shooting percentages (field goals, 3-pointers, free throws), rebounds, assists, steals, blocks, and turnovers. The target column `target_5yrs` is a binary variable indicating if the player's career lasted 5 or more years.

## Prediction Goal
The primary prediction goal is binary classification on the `target_5yrs` variable. By leveraging a player's performance statistics, the trained machine learning model will predict whether that player is likely to have a lasting career in the NBA (≥ 5 years). 

## Feature Engineering Workflow
1. **Data Exploration & Target Definition**: 
   - Defined `target_5yrs` as the dependent variable.
   - Dropped non-predictive columns (`Unnamed: 0` and `name`) to prevent data leakage and remove noise, as names and arbitrary indices do not generalize to new predictive models.
   - **Class Balance Check**: Plotted the value counts for `target_5yrs` to visually inspect class balance. Ensuring the target class is balanced helps us identify if oversampling/undersampling techniques will be needed during the modeling phase.

2. **Feature Transformation & Null Handling**:
   - Assessed all performance columns for null values.
   - Cleaned the dataset by robustly filling any missing values with 0. In basketball statistics, missing percentage values (like 3P%) typically occur because the player made 0 attempts, making 0 a logically sound imputation that maintains data integrity.
   - **Feature Scaling**: Implemented `StandardScaler` to standardize the continuous input features (ensuring a mean of 0 and a standard deviation of 1). Scaling prevents features with larger numerical ranges (like Minutes or Points) from disproportionately dominating distance-based algorithms or neural networks.

3. **Feature Selection & Extraction (Composite Features)**:
   - **Points Per Minute (`pts_per_min`)**: Engineered by dividing Points (`pts`) by Minutes (`min`). This normalizes scoring output and reveals a player's raw scoring efficiency relative to the time they spend on the court.
   - **Assist-to-Turnover Ratio (`ast_tov_ratio`)**: Engineered by dividing Assists (`ast`) by Turnovers (`tov`). This is a widely used basketball metric that captures a player's reliability and decision-making quality.

4. **Multicollinearity & Correlation Analysis**:
   - Computed a correlation matrix and visualized it using a Seaborn heatmap to identify multicollinearity.
   - Highly correlated metric groups (e.g., `fgm`, `fga`, and `pts`; or `oreb`, `dreb`, and `reb`) were identified.
   - Dropped the redundant raw counting stats (`fgm`, `fga`, `ftm`, `fta`, `3p_made`, `3pa`, `oreb`, `dreb`) while retaining the consolidated metrics (`pts`, `reb`, and the percentage columns). This reduces the dimensionality of the dataset and prevents over-weighting correlated attributes without losing critical performance information.

## Repository Contents
- `nba_feature_engineering.ipynb`: The main Jupyter Notebook containing all the executed code, visualizations (including class balance and correlation heatmap), and Markdown documentation.
- `README.md`: This file, summarizing the project logic, steps taken, and insights for stakeholders.
