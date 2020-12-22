# Predict Area Tree Types

# Goal of the project is to build a model that predicts what types of trees grow in an area

Project overview:
1. Created a tool to predict what types of trees grow in an area
2. SVM Classifier and XGBoost classifiers are trained with GridSearchCV to get the best model
3. Created models made deployment ready with Pickle

Workflow:

data-processing.ipynb -> prepared-data-file -> model-building.ipynb

## Data Cleaning

### 1. Check for missing values in every column

### 2. Removed rows with no touchpoints value / nTouchpoints = 0

## EDA

### 1. Explore the distribution of classes in Cover Type

![Cover Type distribution plot](/images/cover_type_distribution.png)

### 2. Discover any presence of multicollinearity and its degree with a heatmap

![Collinearity heatmap](/images/all_correlations.png)
![Collinearity-correlated features](/images/correlated_features.png)

### 3. Visualizing distribution of variables with distribution, bar and box plots and

![Elevation plot](/images/Elevation_distribution.png)
![Aspect plot](/images/Aspect_distribution.png)
![Horizontal Distance to Fire Points plot](/images/horizontal_distance_to_fire_points.png)
![Horizontal Distance to Hydrology plot](/images/horizontal_distance_to_hydrology_distribution.png)
![Horizontal Distance to Roadways plot](/images/horizontal_distance_to_roadways_distribution.png)
![Slope plot](/images/Slope_distribution.png)
![Vertical Distance to HYdrology plot](/images/vertical_distrance_to_hydrology_distribution.png)

### 4. Normalization for continuous variables

Continuous variables were normalized using different techniques and experimenting with data distributions retreived:


## Model Building

**Metrics** for evaluating models: 
1. Multiclass classification since we are predicting the probabilities of the cover type, I want to find the average difference between all probability distributions.
2. F1-Score(Micro) since we have imbalanced classes of labels

### 1a. Standardize/normalize numerical data

![Age distribution plot](/images/plot13.png)
![Income distribution plot](/images/plot14.png)
![Average spending dist plot](/images/plot15.png)

### 1b. Randomized undersampling

There is a custom script to undersample dataset into train, validation and test sets using the stratify strategy. Train size 80%, Validation set and Test set 10% each.

### 2. Baseline model:  **SVM Classifier**

**SVM Classifier**

I picked RF Classifer simply because it runs fast and I am able to use GridSearchCV to iterate to the best model possible efficiently. 
After initializing and tuning my RandomForestClassifier model with GridSearchCV, I got a train accuracy of 1.0 and test 
accuracy of 0.77688 which shows overfitting.

![SVM ROC](/images/svm_rbf_roc.png)

SVM Accuracy: 0.84

SVM F1-Score (Micro): 0.84

### 3. Explore ensemble model: **XGBoost**

**XGBoost**

Initial XGB model

![XGB mean logloss plot](/images/xgb_mlogloss.png)
![XGB mean error plot](/images/xlb_merror.png)

XGB model after tuning with *GridSearchCV* : max_depth, min_child_weight and reg_alpha

![mean logloss plot](/images/plot18.png)![mean error plot](/images/plot19.png)

![FI](/images/plot21.png)

Our XGBoost model pays high attention to the 'unknown' marital status. This could be due to the fact that there are only 44 customers with 'unknown' marital status, hence to reduce bias, our xgb model assigns more weight to 'unknown' feature.

XGBoost Accuracy: 0.9678972712680578

XGBoost F1-Score (Micro): 0.9678972712680578

Final XGBoost model is selected since it gives significantly higher F1-score and accuracy. Overfitting can be handled by further tuning the reg_alpha value in the model.

## Model Deployment

Pickle file is attached for further deployment of the model into FlaskAPI for productionization.
