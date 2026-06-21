# NBA Player Longevity Prediction – Feature Engineering

## Project Overview
This project focuses on analyzing an NBA player performance dataset to engineer features for a machine learning model. The objective is to forecast player longevity by classifying whether a player's career will last 5 years or more (`target_5yrs`). By performing data cleaning, correlation analysis, and feature engineering, we convert raw sports statistics into a robust, ML-ready dataset.

## Dataset Source
The dataset contains historical NBA player performance metrics, including games played, minutes, points, shooting percentages (field goals, 3-pointers, free throws), rebounds, assists, steals, blocks, and turnovers. The target column `target_5yrs` is a binary variable indicating if the player's career lasted 5 or more years.

## Prediction Goal
The primary prediction goal is binary classification on the `target_5yrs` variable. By leveraging a player's performance statistics, the trained machine learning model will predict whether that player is likely to have a lasting career in the NBA (≥ 5 years). 

## Feature Engineering Workflow
1. **Target Definition & Noise Reduction**: 
   - Defined `target_5yrs` as the dependent variable.
   - Dropped non-predictive columns (`Unnamed: 0` and `name`) to prevent data leakage and remove noise, as names and arbitrary indices do not generalize to new predictive models.

2. **Handling Missing Values**:
   - Assessed all performance columns for null values.
   - Cleaned the dataset by filling any missing values with 0. In basketball statistics, missing percentage values (like 3P%) typically occur because the player made 0 attempts, making 0 a logically sound imputation that maintains data integrity.

3. **Engineering Composite Features**:
   - **Points Per Minute (`pts_per_min`)**: Engineered by dividing Points (`pts`) by Minutes (`min`). This normalizes scoring output and reveals a player's raw scoring efficiency relative to the time they spend on the court.
   - **Assist-to-Turnover Ratio (`ast_tov_ratio`)**: Engineered by dividing Assists (`ast`) by Turnovers (`tov`). This is a widely used basketball metric that captures a player's reliability and decision-making quality.

4. **Correlation Analysis & Redundancy Reduction**:
   - Computed a correlation matrix and visualized it using a Seaborn heatmap to identify multicollinearity.
   - Highly correlated metric groups (e.g., `fgm`, `fga`, and `pts`; or `oreb`, `dreb`, and `reb`) were identified.
   - Dropped the redundant raw counting stats (`fgm`, `fga`, `ftm`, `fta`, `3p_made`, `3pa`, `oreb`, `dreb`) while retaining the consolidated metrics (`pts`, `reb`, and the percentage columns). This reduces the dimensionality of the dataset and prevents over-weighting correlated attributes without losing critical performance information.

## Repository Contents
- `nba_feature_engineering.ipynb`: The main Jupyter Notebook containing all the executed code, visualizations, and Markdown documentation.
- `README.md`: This file, explaining the workflow and objectives.
