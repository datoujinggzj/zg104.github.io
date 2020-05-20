---
layout: page
title: Projects
subtitle: That's where I squeeze time from socialization.
---

### How have I organized this page?
This page contains details about projects that I have worked on. In order to help readers get the crux of my work with a single look, I have organized each project using a self-designed template:

- The Project Overview
- The Links to the Project
- Reason for the Project’s proximity to my Heart
- Additional Materials (If applicable)

### Google Stock Price Prediction in Deep Learning

[Stock Price Prediction](https://github.com/zg104/Projects/blob/master/Deep%20Learning/RNN%20project.ipynb) by Zijing Gao.

__The project overview:__ Utilized an attention-based LSTM neural network to predict the Google stock price.

- Data preparation: It is not that hard to extract financial data from Tiingo. However, 80% of a machine learning project is all about
data preprocessing, right? Even though I was not a big guy in programming, I tried to use PostgreSQL for processing, Python for exploratory data analysis, and R for data analysis. Great effort was spent on such a time-consuming thing! I have to appreciate it for making me familier to what a data analyst should basicaly do.

- Modelling: 
![lstm](https://github.com/zg104/Projects/blob/master/Deep%20Learning/lstm.png) <br/>
Long Short Term Memory, usually just called "LSTMs" – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in solving time series problems. Financial data is typically time series data, so I assume that LSTM may be a better choice than the traditional method, such as ARIMA. I constrcuted LSTM networks based on TensorFlow in Python, and spent much effort on the transformation of procecessed data. 

- Evaluation: LSTM is also a kind of neural network, which can be hard to interpret. I utilized an attention-based LSTM neural network to predict the short term stock price trend, which gives me a relatively good result before parameter tuning. My proposed model is significantly better than the other machine learning models, with an adjusted R2 average of 0.95. As is known, parameter tuning is very time-counsuming when I use grid searching. There are many hyperparameters waiting for tuning, such as the number of epochs, the batch size, the number of neurons, and so on. 2000 years later, I finally imrpoved my model accuracy with a good fitting to my test set. However, I can imagine how unefficient I would be if I bump up into some real-life problems.

- Follow-on Work: Financial data is full of stochastical uncertainty, just like the wind that we never catch up with. What I try to do is keep track of the trend and improve the model accuracy at a liitle cost of efficiency. I consulted a bunch of materials and find something interesting when I open the gate of GCP (Google Cloud Platform). I dived into the learning of BigQuery, DataFlow, and so on to get in touch with more and more realistic problems and know exactly what problems the most customers really want to solve. Amazing! It really inspired to develop a data-driven system aiming at analyzing the streamed flow of financial data. Pretty cool if succeeded.
