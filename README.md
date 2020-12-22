# Predict Area Tree Types

## Goal of the project is to build a model that predicts what types of trees grow in an area

Project overview:
1. Created a tool to predict what types of trees grow in an area
2. SVM Classifier and XGBoost classifiers are trained with GridSearchCV to get the best model
3. Created models  with Pickle which are ready for deployment

Workflow:

raw_data_set -> data-processing.ipynb -> prepared-data-file -> model-building.ipynb

## Data Cleaning

### 1. Check for missing values in every column

There are not missing values in dataset.

## EDA

### 1. Explore the distribution of classes in Cover Type

![Cover Type distribution plot](/images/cover_type_distribution.png)

### 2. Discover any presence of multicollinearity and its degree with a heatmap

Heatmap of all features:
![Collinearity heatmap](/images/all_correlations.png)

Detected 7 correlated features:
![Collinearity-correlated features](/images/correlated_features.png)

### 3. Visualiziation of variables with distribution, bar and box plots and
Elevation/Aspect/Slope:
![Elevation plot](/images/Elevation_distribution.png)
![Aspect plot](/images/Aspect_distribution.png)
![Slope plot](/images/Slope_distribution.png)
Horizontal/Vertical Distance:
![Horizontal Distance to Fire Points plot](/images/horizontal_distance_to_fire_points.png)
![Horizontal Distance to Hydrology plot](/images/horizontal_distance_to_hydrology_distribution.png)
![Horizontal Distance to Roadways plot](/images/horizontal_distance_to_roadways_distribution.png)
![Vertical Distance to HYdrology plot](/images/vertical_distrance_to_hydrology_distribution.png)
Hillshade:
![Hillshade_9am plot](/images/hillshade_9am.png)
![Hillshade_noon plot](/images/hillshade_noon.png)
![Hillshade_3pm plot](/images/hillshade_3pm.png)

### 4. Normalization for continuous variables

Continuous variables were normalized using different techniques and experimenting with data distributions retreived:
Gaussian aproximation


### 5. Randomized undersampling

There is a custom script to undersample dataset into train and test sets using the stratify strategy. Train size 80%, Validation set and Test set 10% each.

## Model Building

**Metrics** for evaluating models: 
1. This is Multiclass classification problem since we are predicting the probabilities of the cover type label which contains 7 different values.
2. F1-Score and Accuracy

### 2. Baseline model:  **SVM Classifier**

**SVM Classifier**

Initial run of SVM model with parameters: kernel=linear and C=1: Accuracy=0.79, F1-Score=0.78

SVM model after tuning with *GridSearchCV* : C, gamma, kernel and degree.
Best parameters: C=1, gamma=1, kernel=rbf, degree=1

![SVM ROC](/images/svm_rbf_roc.png)

SVM Accuracy=0.84

SVM F1-Score=0.84

### 3. Explore ensemble model: **XGBoost**

**XGBoost**

Initial XGB model

![XGB mean logloss plot](/images/xgb_mlogloss.png)
![XGB mean error plot](/images/xlb_merror.png)
![XGB feature importance plot](/images/xgb_feature_importance.png)

XGB model after tuning with *GridSearchCV* : max_depth, min_child_weight and reg_alpha

![mean logloss plot](/images/plot18.png)![mean error plot](/images/plot19.png)

![XGB ROC](/images/xlb_roc.png)

Our XGBoost model pays high attention on the Soil Type + Elevation variables. This could be due to the fact that there are only 44 customers with 'unknown' marital status, hence to reduce bias, our xgb model assigns more weight to 'unknown' feature.

XGBoost Accuracy: 0.8642745709828393
XGBoost F1-Score (Micro): 0.8642745709828393

Final XGBoost model is selected since it gives higher F1-score and accuracy. Performances can be evaluated with further tuning parameter values for the model.

## Model Deployment

Pickle file is attached for further deployment of the model into FlaskAPI for productionization.
