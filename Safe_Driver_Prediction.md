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

## Evaluation

__Scoring Metric__

Submissions are evaluated using the [Normalized Gini Coefficient](https://en.wikipedia.org/wiki/Gini_coefficient).

During scoring, observations are sorted from the largest to the smallest predictions. Predictions are only used for ordering observations; therefore, the relative magnitude of the predictions are not used during scoring. The scoring algorithm then compares the cumulative proportion of positive class observations to a theoretical uniform proportion.

The Gini Coefficient ranges from approximately 0 for random guessing, to approximately 0.5 for a perfect score. The theoretical maximum for the discrete calculation is `(1 - frac_pos) / 2`

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
- [Model Training](#jump12)
- [Model Evaluation](#jump13)
 
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

### <span id="jump6">EDA</span>

#### Categorical variables

Let's look into the categorical variables and the proportion of customers with target = 1. 

As we can see from the variables with missing values, it is a good idea to keep the missing values as a separate category value, instead of replacing them by the mode for instance. The customers with a missing value appear to have a much higher (in some cases much lower) probability to ask for an insurance claim.

#### Interval

Checking the correlations between interval variables. A heatmap is a good way to visualize the correlation between variables. 

<p align="center">
  <img src="/image/heatmap.png">
</p>

There are a strong correlations between the variables:

- ps_reg_02 and ps_reg_03 (0.7)
- ps_car_12 and ps_car13 (0.67)
- ps_car_12 and ps_car14 (0.58)
- ps_car_13 and ps_car15 (0.67)

Seaborn has some handy plots to visualize the (linear) relationship between variables. We could use a pairplot to visualize the relationship between the variables. But because the heatmap already showed the limited number of correlated variables, we'll look at each of the highly correlated variables separately.

Allright, so now what? How can we decide which of the correlated variables to keep? We could perform Principal Component Analysis (PCA) on the variables to reduce the dimensions. In the AllState Claims Severity Competition I made this kernel to do that. But as the number of correlated variables is rather low, we will let the model do the heavy-lifting.

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

### <span id="jump9">Feature Scaling</span>

As mentioned before, we can apply standard scaling to the training data. Some classifiers, such as CNN, SVM  perform better when this is done.

### <span id="XGBoost">XGBoost</span>

[XGBoost](XGBoost is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). XGBoost models dominate many Kaggle competitions.) is the leading model for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). XGBoost models dominate many Kaggle competitions.


<p align="center">
  <img src="https://dzone.com/storage/temp/13069535-xgboost-features.png">
</p>


#-----------------------------

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

