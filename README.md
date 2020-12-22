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

### 2. Removed rows with no touchpoints value / nTouchpoints = 0

## EDA

### 1. Explore the distribution of classes in Cover Type

![Cover Type distribution plot](/images/cover_type_distribution.png)

### 2. Discover any presence of multicollinearity and its degree with a heatmap

Detected 7 correlated features:
![Collinearity heatmap](/images/all_correlations.png)

Heatmap of all features:
![Collinearity-correlated features](/images/correlated_features.png)

### 3. Visualiziation of variables with distribution, bar and box plots and

![Elevation plot](/images/Elevation_distribution.png)
![Aspect plot](/images/Aspect_distribution.png)
![Slope plot](/images/Slope_distribution.png)

![Horizontal Distance to Fire Points plot](/images/horizontal_distance_to_fire_points.png)
![Horizontal Distance to Hydrology plot](/images/horizontal_distance_to_hydrology_distribution.png)
![Horizontal Distance to Roadways plot](/images/horizontal_distance_to_roadways_distribution.png)
![Vertical Distance to HYdrology plot](/images/vertical_distrance_to_hydrology_distribution.png)

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
2. F1-Score(Micro) since we have imbalanced classes of labels

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

![XGB ROC](/images/xlb_roc.png)

Our XGBoost model pays high attention to the 'unknown' marital status. This could be due to the fact that there are only 44 customers with 'unknown' marital status, hence to reduce bias, our xgb model assigns more weight to 'unknown' feature.

XGBoost Accuracy: 0.9678972712680578

XGBoost F1-Score (Micro): 0.9678972712680578

Final XGBoost model is selected since it gives higher F1-score and accuracy. Performances can be evaluated with further tuning the reg_alpha value for the model.

## Model Deployment

Pickle file is attached for further deployment of the model into FlaskAPI for productionization.
