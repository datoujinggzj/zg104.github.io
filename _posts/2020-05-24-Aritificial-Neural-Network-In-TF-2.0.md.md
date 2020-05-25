---
layout: post
title: Artificial Neural Network in TensorFlow 2.0
subtitle: ANN for Classification & Regression intro in TF 2.0
cover-img: image/cover14.jpg
tags: [books, test]
---

As Tom Golway said:

> What people call deep learnig is no more than finding answers to questions we know to ask. 
> Real deep learning is answering questions we haven't dreamed of yet.


[Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) is the most powerful branch of Machine Learning. It's a technique that instructs your computer to do what comes naturally to humans: learn by example. Deep learning is a critical technology behind driverless cars, enabling them to recognize a stop sign or to distinguish a pedestrian from a lamppost. It is the key to voice control in consumer devices like phones, tablets, TVs, and hands-free speakers. Deep learning is getting lots of attention lately and for good reason. It’s achieving results that were not possible before.

## Neural Networks versus traditional ML methods

As is known, deep learning is basically deep neural networks which contains multiple hidden layers composed of a great many hidden units. That is where "deep" comes from. So, what is the difference between Neural Networks and traditional machine learning methods?

| Algorithms | Features | Category | 
| :--------  | :---------  | :----  | 
| [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering) | Feature Learning | Unsupervised | 
| [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) | Sigmoid(Softmax) function | Supervised | 
| [k-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) | Voting Algorithm | Supervised | 
| [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine) | Max Margin; kernels | Supervised | 
| [Random Forest](https://en.wikipedia.org/wiki/Random_forest) | Feature bagging; Tree based | Supervised | 
| [XGBoost](https://en.wikipedia.org/wiki/XGBoost) | Gradient Boosting; Clever Penalization; shrinking | Supervised | 
| [Neural Networks](https://en.wikipedia.org/wiki/Neural_network) | Blackbox; Backpropogation; Hidden Layers | Supervised | 

## Types of Neural Networks 

![](https://www.digitalvidya.com/wp-content/uploads/2019/01/Image-1-2.png)

| Types of NN | Features | Fields |
| :--------  | :---------  | :------ |
| [Aritificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network) | One of the simplest neural networks | basic Classification & Regression
| [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) | Convolution; Paddling; Falttening; Pooling; Fully Connection | Image Identification; Computer Vision |
| [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network) | Long Short Term Memory (LSTM); Recursive | Natural Language Processing; Time Series Analysis |

So, let us start from the simpliest one to dive into -- __ANN__.

## Intro to Artificial Neural Network

An __Artificial Neural Network (ANN)__ is a computational model that is inspired by the way biological neural networks in the human brain process information. Artificial Neural Networks have generated a lot of excitement in Machine Learning research and industry, thanks to many breakthrough results in speech recognition, computer vision and text processing. In this blog post we will try to develop an understanding of ANN, and number identification in TensoFlow 2.0.

The reason why we call ANN a blackbox is that the backpropagation, which is the core of training a neural network, is pretty hard to interpret. Here is an example of backpropagation with just one hidden layer.

![](https://i.stack.imgur.com/7Ui1C.png) 

If we have more and more hidden layers to compose a complex deep neural network like this.

![](https://i.pinimg.com/originals/70/82/9d/70829d7a2aa5e1a1f562d890c90037ec.png)

A fully connected neural network like that will result in extremely large computation trouble, which is hard to interpret. A complex composition need to be differentiated with respect to the unknown parameters by chain rule. You might see the real mathematics behind that! 


![test image size](/image/back.png)

But, TensorFlow 2.0 get us out of this justing using several lines of codes. Amazing!

## Building an ANN for number identification in TF 2.0

### Recap of the steps

1. Load in the data
    - MNIST dataset
    - 10 digits (0 to 9)
    - Already included in Tensorflow
2. Build the model
    - Sequential dense layers ending with multiclass logistic regression
3. Train the model
    - Backpropagation using TensorFlow constructure
4. Evaluate the model
    - Confusion Matrix
    - Classification Report
5. Make predictions
    - Being able to see what the neural network is getting wrong will be insightful.
 
### Programming 

Let's get straight into the Python code!

```python
###################################
# Number identification in TF 2.0 #
###################################

# Commented out IPython magic to ensure Python compatibility.
# Install TensorFlow
# !pip install -q tensorflow-gpu==2.0.0-beta1

try:
#   %tensorflow_version 2.x  # Colab only.
except Exception:
  pass

import tensorflow as tf
print(tf.__version__)

# Load in the data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape:", x_train.shape)

# Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Plot loss per iteration
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()

# Evaluate the model
print(model.evaluate(x_test, y_test))

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))

from sklearn.metrcis import classification_report
print(classification_report(y_test, p_test))

# Do these results make sense?
# It's easy to confuse 9 <--> 4, 9 <--> 7, 2 <--> 7, etc.

# Show some misclassified examples
misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" % (y_test[i], p_test[i]));
```

Here is the evaluation results:

- Confusion Matrix

![](/image/cm.png)

- Misclassification case

![](/image/confuse.png)
 





## It is your turn!

{: .box-note}
**I have some questions for you:**
1.  Why we use logistic regression, not linear regression? What are the disadvantages of linear regression for classification? 
2.  What type of datasets is most suited for logistic regression? 
3.  Can you explain or interpret the hypothesis output of logistic regression?
4.  Why we define the sigmoid function, create a new version of cost function, and applied MLE to derive logistic regression? 
5.  How to deal with overfitting?  
6.  What are the disadvantage of logistic regression? 


{: .box-warning}
**My answers:**

1. - Linear regression can give us the values which are not between 0 and 1. 
   
   - Also, linear regression is sensitive to the outliers. However, the sigmoid function restrict the values between 0 and 1, which can be interpreted as the conditional probability of assigning the data to the particular class given the data parametrized by theta.

    <p align="center">
        <img src="https://static.javatpoint.com/tutorial/machine-learning/images/linear-regression-vs-logistic-regression.png" width="800" height="300">
    </p>



2. - Logistic regression likes overlapping data, instead of well separated data.  

   - Linear Discriminent Analysis will perform better for well separated data since the decision boundary is linear.

3. - We try to set a threshold to determine which class each data point should be assigned based on the conditional probability (I have clarified in Q1) derived from the sigmoid function. 
   
   - Typically, we set the threshold to be 0.5. However, it can be adjusted between 0 and 1 for personal specification, such as restriction on TPR (True Positive Rate). 

4. - Sigmoid function helps transform the linear esitimation into non-linear one. 
   
   - If we use the mean squared error as the cost function the same as linear regression, it is impossible to find the derivatives of the cost function with respect to theta, since the sigmoid function will make the cost function non-convex. So, we have to use gradient descent to minimize the cost function instead of computing the gradient by hand.
   
   - You might wonder why the cost function of logistic regression is like [this!](https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11) That is beacuse we applied the MLE to maximize the probability to make the model the most plausible for all data points. You always minimize the loss function, which is just the negative form of the loglikelihood after MLE.

5. - It can be pretty easy for every machine learning method to be overfitting. It is not a big deal!
   
   - A regularization term is added to the cost function where the first part is loss function, and the second is the penalty term.

    <p align="center">
        <img src="https://miro.medium.com/max/3232/1*vwhvjVQiEgLcssUPX6vxig.png" width="700" height="400">
    </p>


6. - You should use k-fold cross validation to determine the highest polynomial of the features if the decision boundary is non-linear. It can be easy for this to overfit.

   - Logistic regression is unstable when dealing with well separated datasets.
   
   - Logistic regression requires relatively large datasets for training.
   
   - Logistic regression is not that popular for multiclassification problems. Sigmoid function should be ungraded to Softmax function(You may hear about it if you know about Neural Networks).
   
## Conclusion 

__Mathematics and statistics are just like twins that Nobody is able to completly separate. <br/> So, let this be a reminder for us all to always remember that it is extremely important and necessary to truly understand the mathematical backgroud of every machine learning algorithm as much as possible.__

