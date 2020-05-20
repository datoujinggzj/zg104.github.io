---
layout: page
title: Classification 
subtitle: How am I classified?
---

### How have I organized this page?
This page contains details about projects that I have worked on. In order to help readers get the crux of my work with a single look, I have organized each project using a self-designed template:

- The Project Overview
- The Links to the Project
- Reason for the Project’s proximity to my Heart
- Additional Materials (If applicable)

### Glass Indentification in Criminal Analysis 

[Glass Indentification](https://github.com/zg104/Projects/blob/master/Statistical_learning/Zijing%20Gao%20642-project-final.pdf) by Zijing Gao.

__The project overview:__ Constructed different machine learning methods to identify the glass type in the crime scene to help crack criminals.

- Data preparation: I extracted the data from UCI, and preprocess it into the suitable version for R analysis. When it comes the data resource, I have to say there won't be too many stupid criminals who will escape through the window and left the broken glass in the crime scene, since the dataset is relatively small. But, it is still cool if correct classfication of glass type can help cracking a criminal, isn't it? If you are interested in it, please [visit!](https://www.crimemuseum.org/crime-library/forensic-investigation/glass-analysis/). I prepared and cleaned the data to detect the relationship among all the compositions and properties of glass. It can be unavoidable to conduct EDA and data visualization with Python. No one can imagine how excited I was after spending a lot of effort in preprocessing the data and be prepared to feed it into my model!

- Modeling: 

![](image/lstm.png)

Long Short Term Memory, usually just called "LSTMs" – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in solving time series problems. Financial data is typically time series data, so I assume that LSTM may be a better choice than the traditional method, such as ARIMA. I constrcuted LSTM networks based on TensorFlow in Python, and spent much effort on the transformation of procecessed data. 

- Evaluation: LSTM is also a kind of neural network, which can be hard to interpret. I utilized an attention-based LSTM neural network to predict the short term stock price trend, which gives me a relatively good result before parameter tuning. My proposed model is significantly better than the other machine learning models, with an adjusted R2 average of 0.95. As is known, parameter tuning is very time-counsuming when I use grid searching. There are many hyperparameters waiting for tuning, such as the number of epochs, the batch size, the number of neurons, and so on. 2000 years later, I finally imrpoved my model accuracy with a good fitting to my test set. However, I can imagine how unefficient I would be if I bump up into some real-life problems.

- Follow-on Work: Financial data is full of stochastical uncertainty, just like the wind that we never catch up with. What I try to do is keep track of the trend and improve the model accuracy at a liitle cost of efficiency. I consulted a bunch of materials and find something interesting when I open the gate of GCP (Google Cloud Platform). I dived into the learning of BigQuery, DataFlow, and so on to get in touch with more and more realistic problems and know exactly what problems the most customers really want to solve. Amazing! It really inspired to develop a data-driven system aiming at analyzing the streamed flow of financial data. Pretty cool if succeeded.


