---
layout: page
title: Porto Seguro’s Safe Driver Prediction
subtitle: Predict if a driver will file an insurance claim next year.
cover-img: image/bps.jpg
---

### How have I organized this page?
This page contains details about Kaggle projects that I have been working on. In order to help readers get the crux of my work with a single look, I have organized each project using a self-designed template:

- The Project Overview
- The Links to the Project
- Main Processes of the Project
- Coding (If applicable)

# Project Overview

## Description

Nothing ruins the thrill of buying a brand new car more quickly than seeing your new insurance bill. The sting’s even more painful when you know you’re a good driver. It doesn’t seem fair that you have to pay so much if you’ve been cautious on the road for years.

[Porto Seguro](https://www.portoseguro.com.br/), one of Brazil’s largest auto and homeowner insurance companies, completely agrees. Inaccuracies in car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones.

In this competition, I was challenged to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. While Porto Seguro has used machine learning for the past 20 years, they’re looking to Kaggle’s machine learning community to explore new, more powerful methods. A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more drivers.

## Evaluation

### Scoring Metric

Submissions are evaluated using the [Normalized Gini Coefficient](https://en.wikipedia.org/wiki/Gini_coefficient).

During scoring, observations are sorted from the largest to the smallest predictions. Predictions are only used for ordering observations; therefore, the relative magnitude of the predictions are not used during scoring. The scoring algorithm then compares the cumulative proportion of positive class observations to a theoretical uniform proportion.

The Gini Coefficient ranges from approximately 0 for random guessing, to approximately 0.5 for a perfect score. The theoretical maximum for the discrete calculation is `(1 - frac_pos) / 2`

The Normalized Gini Coefficient adjusts the score by the theoretical maximum so that the maximum score is 1.

[Here](https://www.kaggle.com/cppttz/gini-coefficient-an-explanation-with-math/) is the math explanation of Gini coefficient.


# Porto Seguro’s Safe Driver Prediction in XGBoost

## Exploratory Data Analysis 

[A](#A)
[B](#B)
 
## A
## B
## 3.标题三
