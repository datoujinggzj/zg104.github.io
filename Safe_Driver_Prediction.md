---
layout: page
title: Porto Seguro’s Safe Driver Prediction
subtitle: Predict if a driver will file an insurance claim next year.
cover-img: image/bps.jpg
---

### How have I organized this page?
This page contains details about Kaggle projects that I have been working on. In order to help readers get the crux of my work with a single look, I have organized each project using a self-designed template:

- [Project Overview](#projectoverview)
- [Main Processes of the Project](#main)
- [Links to the Project](#link)
- [Other Materials](#other)

# <span id="projectoverview">Project Overview</span>

## Description

Nothing ruins the thrill of buying a brand new car more quickly than seeing your new insurance bill. The sting’s even more painful when you know you’re a good driver. It doesn’t seem fair that you have to pay so much if you’ve been cautious on the road for years.


[Porto Seguro](https://www.portoseguro.com.br/), one of Brazil’s largest auto and homeowner insurance companies, completely agrees. Inaccuracies in car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones.

In this competition, I was challenged to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. While Porto Seguro has used machine learning for the past 20 years, they’re looking to Kaggle’s machine learning community to explore new, more powerful methods. A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more drivers.

## Evaluation (Scoring Metric)

Submissions are evaluated using the [Normalized Gini Coefficient](https://en.wikipedia.org/wiki/Gini_coefficient).

During scoring, observations are sorted from the largest to the smallest predictions. Predictions are only used for ordering observations; therefore, the relative magnitude of the predictions are not used during scoring. The scoring algorithm then compares the cumulative proportion of positive class observations to a theoretical uniform proportion.

The Gini Coefficient ranges from approximately `0` for random guessing, to approximately `0.5` for a perfect score. The theoretical maximum for the discrete calculation is `(1 - frac_pos) / 2`

The Normalized Gini Coefficient adjusts the score by the theoretical maximum so that the maximum score is 1.

[Here](https://www.kaggle.com/cppttz/gini-coefficient-an-explanation-with-math/) is the math explanation of Gini coefficient.


# <span id="main">Main Processes of the Project</span>


## Data Preparation 

- [Basic Inspection of the Data](#jump1)
- [Metadata](#jump2)
- [Descriptive Statistics](#jump3)
- [Handling Imbalanced Classes](#jump4)
- [Data Quality](#jump5)
- [EDA](#jump6)
- [Feature Engineering](#jump7)
- [Feature Selection](#jump8)
- [Feature Scaling](#jump9)

## XGBoost

- [XGBoost](#XGBoost)
- [Model Set Up](#jump10)
- [Auxiliary Functions](#jump11)
- [Model Training & Evaluation](#jump12)
- [Parameter Tuning](#jump13)
 
### <span id="jump1">Basic Inspection of the Data</span>

Here is an excerpt of the the data description:

- Features that belong to __similar groupings are tagged__ as such in the feature names (e.g., ind, reg, car, calc).
- Feature names include the postfix __bin__ to indicate binary features and __cat__ to indicate categorical features.
- Features without these designations are either __continuous__ or __ordinal__.
- Values of __-1__ indicate that the feature was __missing__ from the observation.
- The target columns signifies whether or not a claim was filed for that policy holder.
- The test set does not contain the target column, therefore it is an unsupervised ML problem.

That's important information to get us started.

After importing the packages we might need for this challenge, I would like to check the basic information of the training set using `train.info()` and `train.describe()`. Then, we should check out the number of rows and columns in the training set using `train.shape`. So, we have 59 variables and 595212 observations in the training set. Then, we have 58 variables and 892816 observations in the test set. We miss one variable which is the target variable. It is totally fine.

### <span id="jump2">Metadata</span>

Basically, [metadata](https://en.wikipedia.org/wiki/Metadata) is the data of the data.

To facilitate the data management, we'll store meta-information about the variables in a DataFrame. This will be helpful when we want to select specific variables for analysis, visualization, modeling, ...

Concretely we will store:

- role: input, ID, target
- level: nominal, interval, ordinal, binary
- keep: True or False
- dtype: int, float, str

Using metadata, we can extract the columns we might want to use convinently and systematically.

| role | level | count |
| :-- | :-- | :-- |
| id | nominal | 1 |
| input | binary | 17 |
| input | interval | 10 |
| input | nominal | 14 |
| input | ordinal | 16 |
| target | binary | 1 |

Above the number of variables per role and level are displayed.

### <span id="jump3">Descriptive Statistics</span>

We can also apply the describe method on the dataframe. However, it doesn't make much sense to calculate the mean, std, ... on categorical variables and the id variable. We'll explore the categorical variables visually later.

Thanks to our meta file we can easily select the variables on which we want to compute the descriptive statistics. To keep things clear, we'll do this per data type.

After checking the description of different types of variables we might use, we have the folowing information attached.

#### Interval variables

__reg variables__

- only ps_reg_03 has missing values
- the range (min to max) differs between the variables. We could apply scaling (e.g. StandardScaler), but it depends on the classifier we will want to use.

__car variables__

- ps_car_12 and ps_car_15 have missing values
- again, the range differs and we could apply scaling.

__calc variables__

- no missing values
- this seems to be some kind of ratio as the maximum is 0.9
- all three _calc_ variables have very similar distributions

Overall, we can see that the range of the interval variables is rather small. Perhaps some transformation (e.g. log) is already applied in order to anonymize the data?

#### Ordinal variables

- Only one missing variable: ps_car_11
- We could apply scaling to deal with the different ranges

#### Binary variables

- A priori in the train data is 3.645%, which is strongly imbalanced.
- From the means we can conclude that for most variables the value is zero in most cases.

<p align="center">
  <img src="/image/before_undersample.png">
</p>

### <span id="jump4">Handling Imbalanced Classes</span>

As we mentioned above the proportion of records with target=1 is far less than target=0. This can lead to a model that has great accuracy but does have any added value in practice. Two possible strategies to deal with this problem are:

- oversampling records with target=1
- undersampling records with target=0

There are many more strategies of course and MachineLearningMastery.com gives a nice overview. As we have a rather large training set, we can go for __undersampling__.

After undersampling, the number of records with target = 0 after undersampling is 195246.

<p align="center">
  <img src="/image/after_undersample.png">
</p>

### <span id="jump5">Data Quality</span>

#### Checking missing values

```python
Variable ps_ind_02_cat has 216 records (0.04%) with missing values 
Variable ps_ind_04_cat has 83 records (0.01%) with missing values 
Variable ps_ind_05_cat has 5809 records (0.98%) with missing values 
Variable ps_reg_03 has 107772 records (18.11%) with missing values 
Variable ps_car_01_cat has 107 records (0.02%) with missing values 
Variable ps_car_02_cat has 5 records (0.00%) with missing values 
Variable ps_car_03_cat has 411231 records (69.09%) with missing values 
Variable ps_car_05_cat has 266551 records (44.78%) with missing values 
Variable ps_car_07_cat has 11489 records (1.93%) with missing values 
Variable ps_car_09_cat has 569 records (0.10%) with missing values 
Variable ps_car_11 has 5 records (0.00%) with missing values 
Variable ps_car_12 has 1 records (0.00%) with missing values 
Variable ps_car_14 has 42620 records (7.16%) with missing values 
In total, there are 13 variables with missing values
```

- __ps_car_03_cat__ and __ps_car_05_cat__ have a large proportion of records with missing values. Remove these variables.
- For the other categorical variables with missing values, we can leave the missing value -1 as such.
- ps_reg_03 (continuous) has missing values for 18% of all records. Replace by the mean.
- Others has missing values less than 10% of all records. Replace by the mean.

So, we are going to drop the variables with too many missing values and [impute](https://www.kaggle.com/dansbecker/handling-missing-values) other variables with the mean or mode. 

#### Checking the cardinality of the categorical variables

Cardinality refers to the number of different values in a variable. As we will create dummy variables from the categorical variables later on, we need to check whether there are variables with many distinct values. We should handle these variables differently as they would result in many dummy variables.

```python
Variable ps_ind_02_cat has 5 distinct values
Variable ps_ind_04_cat has 3 distinct values
Variable ps_ind_05_cat has 8 distinct values
Variable ps_car_01_cat has 13 distinct values
Variable ps_car_02_cat has 3 distinct values
Variable ps_car_04_cat has 10 distinct values
Variable ps_car_06_cat has 18 distinct values
Variable ps_car_07_cat has 3 distinct values
Variable ps_car_08_cat has 2 distinct values
Variable ps_car_09_cat has 6 distinct values
Variable ps_car_10_cat has 3 distinct values
Variable ps_car_11_cat has 104 distinct values
```

Only __ps_car_11_cat__ has many distinct values, although it is still reasonable.

```python
# Script by https://www.kaggle.com/ogrellier
# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
```

This technique of encoding enables __ps_car_11_cat__ to serve as a numerical feature along with the information unchanged as a categorical features even though it contains `104` distinct values! Amazing! Actually, I want to drop this column at first since the traditional method of encoding can be a massive increment to the dimension of the training set.

### <span id="jump6">Exploratory Data Analysis</span>

#### Categorical variables

Let's look into the categorical variables and the proportion of customers with target = 1. 

As we can see from the variables with missing values, it is a good idea to keep the missing values as a separate category value, instead of replacing them by the mode for instance. The customers with a missing value appear to have a much higher (in some cases much lower) probability to ask for an insurance claim.

#### Interval

Checking the correlations between interval variables. A heatmap is a good way to visualize the correlation between variables. 

<p align="center">
  <img src="/image/heatmap.png">
</p>

There are a strong correlations between the variables:

- ps_reg_02 and ps_reg_03 (`0.7`)
- ps_car_12 and ps_car13 (`0.67`)
- ps_car_12 and ps_car14 (`0.58`)
- ps_car_13 and ps_car15 (`0.67`)

Seaborn has some handy plots to visualize the (linear) relationship between variables. We could use a pairplot to visualize the relationship between the variables. But because the heatmap already showed the limited number of correlated variables, we'll look at each of the highly correlated variables separately.

Allright, so now what? How can we decide which of the correlated variables to keep? We could perform [Principal Component Analysis](https://www.kaggle.com/nirajvermafcb/principal-component-analysis-explained) (PCA) on the variables to reduce the dimensions. In the AllState Claims Severity Competition I made this kernel to do that. But as the number of correlated variables is rather low, we will let the model do the heavy-lifting.

<p align="center">
  <img src="/image/pca.png">
</p>

With __7__ components we already explain more than 90% of all variance in the features. So we could reduce the number of features to half of the original numerical features.

#### Checking the correlations between ordinal variables

<p align="center">
  <img src="/image/heatmap_1.png">
</p>

For the ordinal variables we do not see many correlations.

### <span id="jump7">Feature Engineering</span>


#### Creating [dummy variables](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/60425)

The values of the categorical variables do not represent any order or magnitude. For instance, category 2 is not twice the value of category 1. Therefore we can create dummy variables to deal with that. We drop the first dummy variable as this information can be derived from the other dummy variables generated for the categories of the original variable.

```python
Before dummification we have 57 variables in train
After dummification we have 109 variables in train
```
So, creating dummy variables adds 52 variables to the training set.

#### Creating [interaction variables](https://chrisalbon.com/machine_learning/linear_regression/adding_interaction_terms/)

Interaction effects can be account for by including a new feature comprising the product of corresponding values from the interacting features.

```python
Before creating interactions we have 109 variables in train
After creating interactions we have 164 variables in train
```
### <span id="jump8">Feature Selection</span>

#### Removing features with low or zeri variance

Personally, I prefer to let the classifier algorithm chose which features to keep. But there is one thing that we can do ourselves. That is removing features with no or a very low variance. Sklearn has a handy method to do that: VarianceThreshold. By default it removes features with zero variance. This will not be applicable for this competition as we saw there are no zero-variance variables in the previous steps. But if we would remove features with less than 1% variance, we would remove 31 variables.

```python
28 variables have too low variance.
These variables are ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_12', 'ps_car_14', 'ps_car_11_cat_te', 'ps_ind_05_cat_2', 'ps_ind_05_cat_5', 'ps_car_01_cat_1', 'ps_car_01_cat_2', 'ps_car_04_cat_3', 'ps_car_04_cat_4', 'ps_car_04_cat_5', 'ps_car_04_cat_6', 'ps_car_04_cat_7', 'ps_car_06_cat_2', 'ps_car_06_cat_5', 'ps_car_06_cat_8', 'ps_car_06_cat_12', 'ps_car_06_cat_16', 'ps_car_06_cat_17', 'ps_car_09_cat_4', 'ps_car_10_cat_1', 'ps_car_10_cat_2', 'ps_car_12^2', 'ps_car_12 ps_car_14', 'ps_car_14^2']
```
We would lose rather many variables if we would select based on variance. But because we do not have so many variables, we'll let the classifier chose. For data sets with many more variables this could reduce the processing time.

#### Selecting features with a Random Forest and SelectFromModel

Here we'll base feature selection on the feature importances of a random forest. With Sklearn's [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html) you can then specify how many variables you want to keep. You can set a threshold on the level of feature importance manually. But we'll simply select the top __50%__ best variables by setting the threshold to be `"median"`.

```python
Number of features before selection: 162
Number of features after selection: 81
```
Now, we halve the number of features selected as an input of our model! Exciting!

### <span id="jump9">Feature Scaling</span>

As mentioned before, we can apply standard scaling to the training data. Some classifiers perform better when this is done.

### <span id="XGBoost">XGBoost</span>

[XGBoost](XGBoost is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). XGBoost models dominate many Kaggle competitions.) is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). XGBoost models dominate many Kaggle competitions.


<p align="center">
  <img src="https://dzone.com/storage/temp/13069535-xgboost-features.png">
</p>


### <span id="jump10">Model Set Up</span>

```python
################
# Model Set Up #
################

# parameter set up
MAX_ROUNDS = 400
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 30 

# import packages we might need
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from numba import jit
import time
import gc
```

> Setting `MAX_ROUNDS` fairly high is recommended and using `OPTIMIZE_ROUNDS = True` to get an idea of the appropriate number of rounds (which, in my judgment, should be close to the maximum value of `best_ntree_limit` among all 5 folds, maybe even a bit higher if your model is adequately regularized...or alternatively, you could set `verbose=True` and look at the details to try to find a number of rounds that works well for all folds). Then I would turn off `OPTIMIZE_ROUNDS` and set `MAX_ROUNDS` to the appropraite number of total rounds.

> The problem with "__early stopping__" by choosing the best round for each fold is that it overfits to the validation data. It's therefore liable not to produce the optimal model for predicting test data, and if it's used to produce validation data for stacking/ensembling with other models, it would cause this one to have too much weight in the ensemble. Another possibility (and the default for XGBoost, it seems) is to use the round where the early stop actually happens (with the lag that verifies lack of improvement) rather than the best round. That solves the overfitting problem (provided the lag is long enough), but so far it doesn't seem to have helped. (I got a worse validation score with 20-round early stopping per fold than with a constant number of rounds for all folds, so the early stopping actually seemed to underfit.)


### <span id="jump11">Auxiliary Functions</span>

```python
#######################
# Auxiliary Functions #
#######################

# Compute normalized gini 

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)

# Set up classifier
model = XGBClassifier(    
                        n_estimators=MAX_ROUNDS,     # the number of trees (rounds)
                        max_depth=4,                 # The maximum depth of a tree     
                        objective="binary:logistic",
                        learning_rate=LEARNING_RATE, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                     )
```

Let me explain these parameters:

- `n_estimators`: __The number of trees or rounds__. Adding more trees will be at the risk of overfitting. The reason is in the way that the boosted tree model is constructed, sequentially where each new tree attempts to model and correct for the errors made by the sequence of previous trees. Quickly, the model reaches a point of diminishing returns.

- `max_depth`: __The maximum depth of a tree__. It is also used to control overfitting as higher depth will allow model to learn relations very specific to a particular sample. Typically, it should be chosen from `3` to `10` and tuned using CV.

- `objective`: __The loss function to be minimized__. `binary:logistic` is for binary classification, which will return predicted probability (NOT CLASS).

- `learning_rate`: __The convergence control parameter in gradient descent__. It is intuitive that XGB will not reach its minimum if both `n_estimaters` and `learning_rate` are very small.

- `subsample`: __The fraction of observations to be randomly chosen for each tree__. Lower values make the algorithm more conservative and prevents overfitting, but too small values might lead to underfitting. So, be careful to choose and the typical values are between `0.5` and `1`.

- `min_child_weight`: __The minimum sum of weights all observations required in  child__. It is the minimum weight (or number of samples if all samples have a weight of 1) required in order to create a new node in the tree. A smaller `min_child_weight` allows the algorithm to create children that correspond to fewer samples, thus allowing for more complex trees, but again, more likely to overfit.

- `colsample_bytree`: __The fraction of features to use__. By default it is set to 1 meaning that we will use all features. But in order to avoid the number of highly correlated trees is getting too big, we would like to use a sample of all the features for training to avoid overfitting.

- `scale_pos_weight`: __The parameter that controls the balance of positive and negative weights, useful for unbalanced classes__. This dataset is unbalanced as we have seen, so we should be careful to tune it. The typical value to consider: `sum(negative instances) / sum(positive instances)`. 

- `gamma`: __The minimum loss reduction required to make a split__. A node is split only when the resulting split gives a positive reduction in the loss function. The larger `gamma` is, the more conservative (overfitting) the algorithm will be. The values can vary depending on the loss function and should be tuned.

- `reg_alpha`: __L1 regularization term on weights__. Increasing this value will make model more conservative.

- `reg_lambda`: __L2 regularization term on weights__. Increasing this value will make model more conservative. Normalised to number of training examples.

### <span id="jump12">Model Training & Evaluation</span>

```python
# Run CV

for i, (train_index, test_index) in enumerate(kf.split(train_df)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    X_test = test_df.copy()
    print( "\nFold ", i)
    
    # Enocode data
    for f in f_cats:
        X_train[f + "_avg"], X_valid[f + "_avg"], X_test[f + "_avg"] = target_encode(
                                                        trn_series=X_train[f],
                                                        val_series=X_valid[f],
                                                        tst_series=X_test[f],
                                                        target=y_train,
                                                        min_samples_leaf=200,
                                                        smoothing=10,
                                                        noise_level=0
                                                        )
    # Run model for this fold
    if OPTIMIZE_ROUNDS:
        eval_set=[(X_valid,y_valid)]
        fit_model = model.fit( X_train, y_train, 
                               eval_set=eval_set,
                               eval_metric=gini_xgb,
                               early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                               verbose=False
                             )
        print( "  Best N trees = ", model.best_ntree_limit )
        print( "  Best gini = ", model.best_score )
    else:
        fit_model = model.fit( X_train, y_train )
        
    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  Gini = ", eval_gini(y_valid, pred) )
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    y_test_pred += fit_model.predict_proba(X_test)[:,1]
    
    del X_test, X_train, X_valid, y_train
    
y_test_pred /= K  # Average test set predictions

print( "\nGini for full training set:" )
eval_gini(y, y_valid_pred)
```

__Here is the evaluation result__:

```python
Fold  0
  Gini =  0.27653902080236903

Fold  1
  Gini =  0.29872856349009047

Fold  2
  Gini =  0.2912801518761168

Fold  3
  Gini =  0.27827947856447943

Fold  4
  Gini =  0.2776883922018696

Gini for full training set:
0.2843699295733353
```

As is known, the better, the closer normalized Gini is approaching `0.5`. Pretty good! We will continue to tune the hyperparameters to improve our model.

## <span id="jump13">Parameter Tuning</span>

# <span id="link">Links</span>

[Porto Seguro's Safe Driver Prediction EDA & XGBoost -- Version 1](https://colab.research.google.com/drive/1ZyBvbnQhL09dwoCaoi11tSE_1stC88T6#scrollTo=HUQg-1XPLpDe)

- Data Preparation
- EDA
- XGBoost (undersampling)

The normalized Gini Coefficient is around 0.233. Not bad! I assume that the number of feature selected for training is 81 which is a little bit large.

[Porto Seguro's Safe Driver Prediction EDA & XGBoost -- Version 2](https://colab.research.google.com/drive/1eP8kkBU3dUSEZZFBV7z0VuehQirUqNH9#scrollTo=UG9y6BjdZndW)

- Still undersampling
- Remove feature selection by importance using RF.

The normalized Gini Coefficient is around 0.284. Good! The number of features for training shrinks to 36. They are proven to be more significantly correlated feature for this model.

