# Predict Area Tree Types

## Goal of the project is to build a model that predicts what types of trees grow in an area

Project overview:
1. Created a model to predict what types of trees grow in an area
2. SVM Classifier and XGBoost classifiers are trained with GridSearchCV to get the best model
3. Created models  with Pickle which are ready for deployment

Workflow:

```raw_data_set -> data-processing.ipynb -> prepared-data-file -> model-building.ipynb```

## Data Cleaning

### 1. Check for missing values in every column

There are not missing values in dataset.

## EDA

### 1. Explore the distribution of classes in Cover Type

![Cover Type distribution plot](/images/cover_type_distribution.png)

### 2. Visualiziation of variables with distribution, bar and box plots
It is noticable that most of the variables are left or right skewed with present outliers.

- Continuous numerical variables before normalization:

<table border=0>
  <tr>
    <td valign="top"><img src="/images/Elevation_distribution.png"></td>
    <td valign="top"><img src="/images/Aspect_distribution.png"></td>
    <td valign="top"><img src="/images/Slope_distribution.png"></td>
  </tr>
  <tr>
    <td valign="top"><img src="/images/horizontal_distance_to_fire_points.png"></td>
    <td valign="top"><img src="/images/horizontal_distance_to_hydrology_distribution.png"></td>
    <td valign="top"><img src="/images/horizontal_distance_to_roadways_distribution.png"></td>
    <td valign="top"><img src="/images/vertical_distrance_to_hydrology_distribution.png"></td>
  </tr>
   <tr>
    <td valign="top"><img src="/images/hillshade_9am.png"></td>
    <td valign="top"><img src="/images/hillshade_noon.png"></td>
    <td valign="top"><img src="/images/hillshade_3pm.png"></td>
  </tr>
 </table>


- Horizontal/Vertical Distance:
![Horizontal Distance to Fire Points plot](/images/horizontal_distance_to_fire_points.png)
![Horizontal Distance to Hydrology plot](/images/horizontal_distance_to_hydrology_distribution.png)
![Horizontal Distance to Roadways plot](/images/horizontal_distance_to_roadways_distribution.png)
![Vertical Distance to HYdrology plot](/images/vertical_distrance_to_hydrology_distribution.png)
- Hillshade:
![Hillshade_9am plot](/images/hillshade_9am.png)
![Hillshade_noon plot](/images/hillshade_noon.png)
![Hillshade_3pm plot](/images/hillshade_3pm.png)

### 2. Discover any presence of multicollinearity and its degree with a heatmap
For this purpose Feature Selector tool is used: [Feature Selector](https://github.com/WillKoehrsen/feature-selector)

It contains wide range of functions for qualitative datasets analysis such as identification of:

- Missing values above particular threshold - set on 0.5
0 features
- Single unique values
0 features
- Colinear features above particular threshold - set on 0.5
7 features - ```'Vertical_Distance_To_Hydrology', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type29'```

Detected 7 correlated features:
![Collinearity-correlated features](/images/correlated_features.png)

Heatmap of all features:
![Collinearity heatmap](/images/all_correlations.png)

- Zero importance features - using Gradient Boosting Model
- Low importance features - 

In total 28 features were removed as not important for the classification model building.

### 4. Normalization for continuous variables

Continuous variables were normalized together with outliers removal using different techniques and experimenting with data distributions retreived:
- Gaussian aproximation
- Quantiles normalization
- Yeo-Johnson PowerTransformer method (available in Scikit-Learn >=0.20)

Latest showed the best performances and was used for data processing.

### 5. Randomized undersampling

Package [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/api.html) was used to undersample multi-class dataset into using the RandomUnderSampler strategy. 

## Model Building

This is Multiclass classification problem since we are predicting the probabilities of the cover type label which contains 7 different values.

**Metrics** for evaluating models: 
1. F1-Score and Accuracy
2. ROC and AUC curves
3. Confussion matrix

### 2. Model:  **SVM Classifier**

**SVM Classifier**

Initial run of SVM model with parameters: ```kernel=linear and C=1: Accuracy=0.79, F1-Score=0.78```

SVM model after tuning with *GridSearchCV* : C, gamma, kernel and degree.

Best parameters selected by *GridSearchCV*: ```C=1, gamma=1, kernel=rbf, degree=1```

![SVM ROC](/images/svm_rbf_roc.png)

```SVM Accuracy=0.84```

```SVM F1-Score=0.84```

### 3. Explore ensemble model: **XGBoost**

**XGBoost**

Initial XGB model parameters

```xgb.XGBClassifier(learning_rate=0.1,```
                    ```n_estimators=1000,```
                    ```max_depth=5,```
                    ```min_child_weight=1,```
                    ```gamma=0,```
                    ```subsample=0.8,```
                    ```colsample_bytree=0.8,```
                    ```objective='multi:softmax',```
                    ```nthread=4,```
                    ```num_class=7,```
                    ```seed=27)```
F1-Score: 0.84
![XGB mean logloss plot](/images/xgb_mlogloss.png)
![XGB mean error plot](/images/xlb_merror.png)
![XGB feature importance plot](/images/xgb_feature_importance.png)

XGB model after tuning with *GridSearchCV* : max_depth, min_child_weight and reg_alpha
Best parameters/mean selected by *GridSearchCV*: 

{'best_mean': 0.8651985002262362,
  'best_param': {'max_depth': 9, 'min_child_weight': 1}})
  
![XGB ROC](/images/xlb_roc.png)

Our XGBoost model pays high attention on the Soil Type + Elevation variables. This could be due to the fact that there are only 44 customers with 'unknown' marital status, hence to reduce bias, our xgb model assigns more weight to 'unknown' feature.

```XGBoost Accuracy: 0.8642745709828393```
```XGBoost F1-Score (Micro): 0.8642745709828393```

Final XGBoost model is selected since it gives higher F1-score and accuracy. Performances can be evaluated with further tuning parameter values for the model.

## Model Deployment

Pickle file is attached for further deployment of the model into FlaskAPI for productionization.
