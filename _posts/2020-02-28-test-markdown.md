---
layout: post
title: Mathematics or Statistics in basic logistic regression
subtitle: Which do you think is more important?
---

Let me take a brief introduction of myself!

I am currently a graduate student in Georgetown University, majoring in Matheatics & Statistics.
Don't think I am a boring person though lots of people of this major probably are. I love reclining, drinking a cup of my favorite coffee and let the time pass by quietly. That is why I have adequate space and time to free myself from study.

**Here, you might want to know how terrible math can be**

![](https://github.com/zg104/zg104.github.io/blob/master/image/l2-term.png)

Let's just take the most basic machine learning classification method - Logistic Regression for example. You maybe very familier with it if you are assigned to solve an easy binary classification problem. Given a dataset heading like this, what would you do?

| id | Height(cm) | Weight(kg) | Gender |
| :------ |:--- | :--- | :--- |
| 1 | 180 | 83 | Male |
| 2 | 160 | 45 | Female |
| 3 | 176 | 76 | Male |
| 4 | 162 | 54 | Female |
| 5 | 157 | 43 | Female |

Obviously, we classify people that are taller and heavier as "Male". So, we try to teach the machine to do this, either.

If you are a Python guy, you definitely will write:

```python
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)  # Suppose we have splited the data into traing, test set.
predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
```

If you are familier with R, you maybe write:

```R
glm.fit <- glm(Gender ~ Height + Weight, data = data, family = binomial)
glm.probs <- predict(glm.fit,type = "response")
glm.pred <- ifelse(glm.probs > 0.5, "Male", "Female")
attach(data)
table(glm.pred,Gender) # confusion matrix
mean(glm.pred == Gender) # classification accuracy
```

I seems that just 3 or 4 lines of code will give you want you want. You do not see any mathematical elements in each case, right? That's why we should thank the Python & R packages developers. 

Now, let us taste the version using plain Python without machine learning packages from [Kaggle](https://www.kaggle.com/jeppbautista/logistic-regression-from-scratch-python).

```python
# prepare the data
import numpy as np
X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))

# 1. Sigmoid function --> core of logistic regression
def sigmoid(X, weight):
    z = np.dot(X, weight)
    return 1 / (1 + np.exp(-z))
    
# 2. loss function --> here comes the mathematics!
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

# 3. gradient descent, you definitely heard about it!
def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]
def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient
    
# 4. Maximum Likelihood Estimation --> all mathematics
def log_likelihood(x, y, weights):
    z = np.dot(x, weights)
    ll = np.sum( y*z - np.log(1 + np.exp(z)) )
    return ll
def gradient_ascent(X, h, y):
    return np.dot(X.T, y - h)
def update_weight_mle(weight, learning_rate, gradient):
    return weight + learning_rate * gradient
# 5. Now, it is time to train our model!
num_iter = 100000

intercept = np.ones((X.shape[0], 1)) 
X = np.concatenate((intercept, X), axis=1)
theta = np.zeros(X.shape[1])

for i in range(num_iter):
    h = sigmoid(X, theta)
    gradient = gradient_descent(X, h, y)
    theta = update_weight_loss(theta, 0.1, gradient)

result = sigmoid(X, theta)

# 6. interpret the result
f = pd.DataFrame(np.around(result, decimals=6)).join(y)
f['pred'] = f[0].apply(lambda x : 0 if x < 0.5 else 1)
print("Accuracy (Loss minimization):")
f.loc[f['pred']==f['class']].shape[0] / f.shape[0] * 100
```

I am not saying we should go back to where we start. A lot of machine learning algorithms are stored in Sckit-learn within Python, which provides us great flexibility to solve ML problems!

However, we should understand each method to develop and utilize it into real application.

### Here, I have some questions for you.
{: .box-note}
**Questions:** 
1. Why we use logistic regression, not linear regression? Why are the disadvantages of linear regression for classification?
2. What type of datasets is most suited for logistic regression?
3. Can you explain or interpret the hypothesis output of logistic regression?
4. Why we define the sigmoid function, create a new version of cost function, and applied MLE to derive logistic regression?
5. How to deal with overfitting?
6. What are the disadvantage of logistic regression?

### Warning

{: .box-warning}
**My answers:**

1. Linear regression can give us the values which are not between 0 and 1. Also, linear regression is sensitive to the outliers.
However, the sigmoid function restrict the values between 0 and 1, which can be interpreted as the conditional probability of assigning the data to the particular class given the data parametrized by theta.

<p align="center">
    <img src="https://github.com/zg104/zg104.github.io/blob/master/image/loglin.png" width="500" height="350">
</p>

2. Logistic regression likes overlapping data, instead of well separated data. Linear Discriminent Analysis will perform better for well separated data since the decision boundary is linear.

3. We try to set a threshold to determine which class each data point should be assigned based on the conditional probability (I have clarified in Q1) derived from the sigmoid function. Typically, we set the threshold to be 0.5. However, it can be adjusted between 0 and 1 for personal specification, such as restriction on TPR (True Positive Rate). 

4. - Sigmoid function helps transform the linear esitimation into non-linear one. 
   - If we use the mean squared error as the cost function the same as linear regression, it is impossible to find the derivatives of the cost function with respect to theta, since the sigmoid function will make the cost function non-convex. So, we have to use gradient descent to minimize the cost function instead of computing the gradient by hand.
   - You might wonder why the cost function of logistic regression is like [this!](https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11) That is beacuse we applied the MLE to maximize the probability to make the model the most plausible for all data points. You always minimize the loss function, which is just the negative form of the loglikelihood after MLE.

5. It can be pretty easy for every machine learning method to be overfitting. It is not a big deal! <br/> A regularization term is added to the cost function where the first part is loss function, and the second is the penalty term. 

### Error

{: .box-error}
**Error:** This is an error box.
