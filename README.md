# WineQT
 Develop predictive models to estimate wine quality based on physicochemical properties, using regression and classification techniques.Objectives:
Model Wine Quality: Develop predictive models to estimate wine quality based on physicochemical properties, using regression and classification techniques.
Exploratory Data Analysis (EDA): Perform an exploratory data analysis to understand the distribution of the physicochemical properties, identify patterns, correlations, and potential outliers, and visualise the relationship between these properties and wine quality.
Data Pre-processing: Apply necessary data pre-processing steps, including handling missing values (if any), feature scaling, and normalization. Address issues such as class imbalance using techniques like oversampling, undersampling, or SMOTE to improve model performance.
Feature Importance Analysis: Conduct an analysis to identify which physicochemical properties most significantly impact wine quality, assisting in feature selection for improved model performance.
Application Development: Build a data mining application that allows winemakers or researchers to input physicochemical properties and predict the wine's quality score, aiding in decision-making.


 Red Wine Dataset:
Number of records (instances): 1,599
Number of attributes: 12 (11 physicochemical properties + 1 quality score)
Type of attributes: All attributes, except the quality score, are continuous.
Attributes:
Fixed acidity (continuous): Non-volatile acid in the wine.
Volatile acidity (continuous): Amount of acetic acid in the wine, which gives it vinegar-like taste.
Citric acid (continuous): Adds flavor and freshness to the wine.
Residual sugar (continuous): Amount of sugar remaining after fermentation.
Chlorides (continuous): Amount of salt in the wine.
Free sulfur dioxide (continuous): Prevents microbial growth and oxidation.
Total sulfur dioxide (continuous): Total presence of sulfur dioxide in the wine.
Density (continuous): The wine's density, related to the sugar and alcohol content.
pH (continuous): Measure of the acidity/basicity of the wine on a scale from 0-14.
Sulphates (continuous): Aids in wine preservation.
Alcohol (continuous): Percentage of alcohol in the wine.
Quality (ordinal, integer): A quality score of the wine between 0 & 10
                 All the attributes have size 4 bytes except target attribute  

 White Wine Dataset:
Number of records: 4,898 instances
Number of attributes: 12 (same as red wine dataset)
Type of attributes: same as red wine dataset

Dependent attribute : 
We are using regression and classification models for both the models the target attribute is “Quality(ordinal , integer)”.
