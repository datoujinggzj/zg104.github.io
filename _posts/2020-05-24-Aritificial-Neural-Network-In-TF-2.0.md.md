---
layout: post
title: Digit Recognition in Artificial Neural Network Using TensorFlow 2.0
subtitle: Develop an ANN for MINST Digit Classification
cover-img: image/cover14.jpeg
tags: [blogs, insights]
---

# Digit Recognition in Deep Learning

To make machines more intelligent, the developers are diving into machine learning and deep learning techniques. A human learns to perform a task by practicing and repeating it again and again so that it memorizes how to perform the tasks. Then the neurons in his brain automatically trigger and they can quickly perform the task they have learned. 

Deep learning is also very similar to this. It uses different types of neural network architectures for different types of problems. For example – object recognition, image and sound classification, object detection, image segmentation, etc.


![](https://www.digitalvidya.com/wp-content/uploads/2019/01/Image-1-2.png)


As Tom Golway said:

> What people call deep learnig is no more than finding answers to questions we know to ask. 
> Real deep learning is answering questions we haven't dreamed of yet.


[Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) is the most powerful branch of Machine Learning. It's a technique that instructs your computer to do what comes naturally to humans: learn by example. Deep learning is a critical technology behind driverless cars, enabling them to recognize a stop sign or to distinguish a pedestrian from a lamppost. It is the key to voice control in consumer devices like phones, tablets, TVs, and hands-free speakers. Deep learning is getting lots of attention lately and for good reason. It’s achieving results that were not possible before.

## Neural Networks vs. Traditional ML Algorithms

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

<br/>

| Types of Neural Networks | Features | Fields |
| :--------  | :---------  | :------ |
| [Aritificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network) | One of the simplest neural networks | basic Classification & Regression
| [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) | Convolution; Paddling; Falttening; Pooling; Fully Connection | Image Identification; Computer Vision |
| [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network) | Long Short Term Memory (LSTM); Recursive | Natural Language Processing; Time Series Analysis |

So, let us start from the simplest one to dive into __ANN__.

## Intro to Artificial Neural Network (ANN) 

An __Artificial Neural Network (ANN)__ is a computational model that is inspired by the way biological neural networks in the human brain process information. Artificial Neural Networks have generated a lot of excitement in Machine Learning research and industry, thanks to many breakthrough results in speech recognition, computer vision and text processing. In this blog post we will try to develop an understanding of ANN, and number identification in TensoFlow 2.0.

The reason why we call ANN a blackbox is that the backpropagation, which is the core of training a neural network, is pretty hard to interpret. Here is an example of backpropagation with just one hidden layer.

![](https://i.stack.imgur.com/7Ui1C.png) 

If we have more and more hidden layers to compose a complex deep neural network like this.

![](https://i.pinimg.com/originals/70/82/9d/70829d7a2aa5e1a1f562d890c90037ec.png)

A fully connected neural network like that will result in extremely large computation trouble, which is hard to interpret. A complex composition need to be differentiated with respect to the unknown parameters by chain rule. You might see the real mathematics behind that! 


![test image size](/image/back.png)

But, TensorFlow 2.0 get us out of this justing using several lines of codes. Amazing!

## ANN for Digit Recognition in TF 2.0

### Steps

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

__Why not get straight into the Python code?__

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

__Here is the evaluation results__:

- Accuracy curve

  <p align="center">
    <img src="/image/accuracy.png">
  </p>

  The classification accuracy from both the training set and test set is reaching 99% after 10 epochs training. Awesome!

- Confusion Matrix

  <p align="center">
    <img src="/image/confusionm.png">
  </p>

  The confusion matrix is a classical metric for evalutaing the classification accuracy. It gives a more specific feedback about the misclassification distribution of each category.
  
- Classification Report 

  <p align="center">
    <img src="/image/classification_report.png" width="400" height="280">
  </p>
  
  The classification report is from the [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics). It provides more advanced evaluation indicators to describe the performance of the machine learning models.

- Misclassification case

  <p align="center">
    <img src="/image/confuse.png">
  </p>

   <p align="center">
     <img src="/image/confuse2.png">
   </p>

  <p align="center">
    <img src="/image/confuse3.png">
  </p>

  <p align="center">
    <img src="/image/confuse4.png">
  </p>

  These 4 images illustrate the misclassification cases. It is resonable to consider all of these misclassifcation make sense, isn't it? 


## It is your turn!

{: .box-note}
**I have some questions for you:**
1.  What are the disadvantages of neural networks?

2.  What are the steps for using a gradient descent algorithm?

3.  Which of the following techniques perform similar operations as dropout in a neural network?

    A. Bagging <br/>
    B. Boosting <br/>
    C. Stacking <br/>
    
4. Which of the following gives non-linearity to a neural network?

    A. Stochastic Gradient Descent <br/>
    B. Rectified Linear Unit <br/>
    C. Convolution function <br/>
    
5. In a neural network, which of the following techniques is used to deal with overfitting?

    A. Dropout <br/>
    B. Regularization <br/>
    C. Batch Normalization <br/>
    D. All of these <br/>
    
6. What is a dead unit in a neural network?

    A. A unit which doesn’t update during training by any of its neighbour <br/>
    B. A unit which does not respond completely to any of the training patterns <br/>
    C. The unit which produces the biggest sum-squared error <br/>
    
7. Which of the following statement is the best description of early stopping?

    A. Train the network until a local minimum in the error function is reached <br/>
    B. Simulate the network on a test dataset after every epoch of training. Stop training when the generalization error starts to increase <br/>
    C. Add a momentum term to the weight update in the Generalized Delta Rule, so that training converges more quickly <br/>
    D. A faster version of backpropagation, such as the Quickprop algorithm <br/>

{: .box-warning}
**My answers:**

1.  - Hard to feed massive amounts of available data into neural network for training to get a good result.  
   
    - Hard to interpret unexplained functioning of the network.

    - Hard to tune the hyperparameters (time-consuming, computational cost)

2.  - Initialize random weight and bias.

    - Pass an input through the network and get values from output layer.  
    
    - Calculate error between the actual value and the predicted value.
    
    - Go to each neurons which contributes to the error and change its respective values to reduce the error.
    
    - Reiterate until you find the best weights of network.

3.  A
    - Bagging is just bootstrap, which is a ensmble methods for estimating some statistical variables. The core is to average the training results using different training sets, which are randomly selected with replacement from the original data, to reduce the probability to overfit, and generalized the model.
    
    - Dropout layer is meant to reduce the probability to overfit, too. The nodes are randomly chosen to drop out of the network to change the construction for each epoch. 
    
    - Dropout is a regularization method that approximates training a large number of neural networks with different architectures in parallel. It has the effect of making the training process noisy, forcing nodes within a layer to probabilistically take on more or less responsibility for the inputs.
    
    <p align="center">
        <img src="https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/assets/tfdl_0408.png">
    </p>

4. B
   
   - Rectified Linear Unit (ReLU) is an activation function defined as the positive part of its argument. It is something just like our old friend -- Sigmoid. As is known, the most important and computational part of neural network is backpropagation, which requires to compute the gradient out of a pretty complex composition function. So, the weights and bias will be updated properly if the gradient is relatively easy to compute of each activation layer.
   
   - The derivatives of ReLU, which is a piecewise function, are both constant. It is computationally kind to use ReLU as a choice of activation function.
   
   - Also, it is able to give the data non-linearity since it is a piecewise function. The data was transformed over and over again after being fed into each activation layer.

5. D

   - Dropout is clarified above as a method to avoid overfitting. (included in the codes above)
   
   - L2 & L1 regularization update the general cost function by adding another term known as the regularization term.
   
     In L2, we have:
      
     <p align="center">
        <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/Screen-Shot-2018-04-04-at-1.59.54-AM.png">
     </p>
   
     In L1, we have:
   
     <p align="center">
        <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/Screen-Shot-2018-04-04-at-1.59.57-AM.png">
     </p>
   
     In TF 2.0, you can do this
   
     ```python
     from keras import regularizers
     model.add(Dense(128, input_dim=(28,28),
                kernel_regularizer=regularizers.l2(0.01) 
                # it is hyperparameter for tuning
     ```

   - Batch normalization allows each layer of a network to learn by itself a little bit more independently of other layers. 
   
     To increase the stability of a neural network, batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.
     
   - EXTRA: Data Augmentation
    
     The simplest way to reduce overfitting is to increase the size of the training data. In machine learning, we were not able to increase the size of training data as the labeled data was too costly. But, now let’s consider we are dealing with images. In this case, there are a few ways of increasing the size of the training data – rotating the image, flipping, scaling, shifting, etc.
      
     In TF 2.0, you can do this
   
     ```python
     from keras.preprocessing.image import ImageDataGenerator
     data_gen = ImageDataGenerator(horizontal flip=True)
     data_gen.fit(X_train)
     ```

6. A

   - When using ReLU, you are using a stepwise function that evaluates to 0 whenever the input is less than or equal to 0. 
   
   - Because of this piecewise nature, the gradient is 0 if the input <= 0, since the slope here is 0. However, if every training example causes a certain neuron to have a negative value (which then becomes 0 after ReLU is applied), then the neuron will never be adjusted, since no matter which training example is selected (or which batch) the gradient on the neuron will be 0. 

7. B

   - Early stopping is a kind of cross-validation strategy where we keep one part of the training set as the validation set. When we see that the performance on the validation set is getting worse, we immediately stop the training on the model.
  
     <p align="center">
        <img src="https://www.jeremyjordan.me/content/images/2017/07/Screen-Shot-2017-07-25-at-3.55.30-PM.png">
     </p>
     
     In TF 2.0, you can do this
     
     ```python
     # Set callback functions to early stop training and save the best model so far
     callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
     
     r = model.fit(x_train, 
                   y_train, 
                   validation_data=(x_test, y_test), 
                   epochs=10,
                   callbacks=callbacks, # Early stopping
                   verbose=0 # Print description after each epoch)
     ```
     
## Other Materials

- [Tensorflow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.17096&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

- [MNIST Tutorial in TensorFlow](https://www.tensorflow.org/quantum/tutorials/mnist)

- [MNIST Playground](http://www.ccom.ucsd.edu/~cdeotte/programs/MNIST.html)

   
## Conclusion 

__Artificial Neural Network (ANN) is one of the most simplest neural networks to start with. However, it is so powerful that a great many problems can be easily solved by ANN. Always remember the trade-off of building an ANN, comparison with other algorithms, and techniques for tuning parameters to avoid overfitting.__

**Enjoy Deep Learning!**

