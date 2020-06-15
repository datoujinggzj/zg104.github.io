---
layout: page
title: COVID-19 Analysis, Visualization, and Prediction
subtitle: Don't panic and be safe!
cover-img: image/COVID19.jpg
---

<p align="center">
  <img src="https://techcrunch.com/wp-content/uploads/2020/02/coronavirus.jpg">
</p>

### How have I organized this page?
This page contains details about Kaggle projects that I have been working on. In order to help readers get the crux of my work with a single look, I have organized each project using a self-designed template:

- [Project Overview](#projectoverview)
- [Main Processes of the Project](#main)
- [Links to the Project](#link)
- [Other Materials](#other)

# <span id="projectoverview">Project Overview</span>

## Description

[Coronavirus](https://www.health.harvard.edu/diseases-and-conditions/covid-19-basics) is a family of viruses that can cause illness, which can vary from common cold and cough to sometimes more severe disease. Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV) were such severe cases with the world already has faced.

SARS-CoV-2 (n-coronavirus) is the new virus of the coronavirus family, which first discovered in 2019, which has not been identified in humans before. It is a contiguous virus which started from Wuhan in December 2019. Which later declared as Pandemic by WHO due to high rate spreads throughout the world. Currently (on the date 10 June 2020), this leads to a total of 411K+ Deaths across the globe, including 180K+ deaths alone in Europe.

Pandemic is spreading all over the world; it becomes more important to understand about this spread. This NoteBook is an effort to analyze the cumulative data of confirmed, deaths, and recovered cases over time. In this notebook, the main focus is to analyze the spread trend of this virus all over the world.

# <span id="main">Main Processes of the Project</span>


## Data Analysis

- [Basic Inspection of the Data](#jump1)
- [Data Quality](#jump2)
- [Correlation Analysis](#jump3)
- [Descriptive Statistics](#jump4)
- [EDA](#jump5)

## COVID-19 By Country

- [China(1st Epicentre)](#china)
- [World(excluding China)](#world)
- [Italy(2nd Epicentre)](#Italy)
- [US(3rd Epicentre)](#US)

### <span id="jump1">Basic Inspection of the Data</span>

First, we `import` the packages we might need for analysis.

```python
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
import numpy as np 
import pandas as pd
import os
```

After loading in the data, we observe that the data is of the shape `df.shape = (42264, 8)` by the data of `'2020-06-14'`.
The first 5 observations of the data is listed below:


| SNo | ObservationDate | Province/State | Country/Region | Last Update | Confirmed | Deaths | Recovered |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | 01/22/2020 | Anhui | Mainland China | 1/22/2020 17:00 | 1.0 | 0.0 | 0.0 |
| 2 | 01/22/2020 | Beijing | Mainland China | 1/22/2020 17:00 | 14.0 | 0.0 | 0.0 |
| 3 | 01/22/2020 | Chongqing | Mainland China | 1/22/2020 17:00 | 6.0 | 0.0 | 0.0 |
| 4 | 01/22/2020 | Fujian | Mainland China | 1/22/2020 17:00 | 1.0 | 0.0 | 0.0 |
| 5 | 01/22/2020 | Gansu | Mainland China | 1/22/2020 17:00 | 0.0 | 0.0 | 0.0 |


### <span id="jump2">Data Quality</span>

To check the data quality, we find that about `43%` of Province/State are missing from the data. After transforming the observation date into datetime and other numerical variables into integers, we create a separate dataframe for the most recent case representing __14th June 2020__.

Out of the `42264` records we are now left with only `729` and many of the records dont have a Province defined. These are mostly provinces that are not part of China. Wherever Province is null, we replace it with the Country name and we group Mainland China and China together in China

Then, we use `LabelEncoder` to encode the Country and Province.

```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_update['ProvinceID'] = le.fit_transform(df_update['Province/State'])
df_update['CountryID']=le.fit_transform(df_update['Country/Region'])
df_update.head()
```

### <span id="jump3">Correlation Analysis</span>

<p align="center">
  <img src="/image/corr.png">
</p>

There is no strong correlation between any of the variables except for Confirmed and Deaths variables


__Country wise Correlation__

<p align="center">
  <img src="/image/corr1.png">
</p>

__Continent Wise Correlation__

<p align="center">
  <img src="/image/corr2.png">
</p>

### <span id="jump4">Descriptive Statistics</span>

US leads with 27.18% of the confirmed cases all over the world. This scenario is in total contrast to the initial days when China accounted for nearly 99% of the cases. The growth rate for US has slowed down in the past few days. The numbers in Brazil and Russia have increased drastically placing them in 2nd and 3rd spot.India moved to 5th place with respect to the number of confirmed cases taking over Spain.

__Representation of confirmed cases per country__

<p align="center">
  <img src="/image/percountry.png">
</p>

__Top 10 countries which are affected by the COVID-19 the most__

<p align="center">
  <img src="/image/top10.png">
</p>

__Descriptive statistics grouping by country__

<p align="center">
  <img src="/image/gbcase.png">
</p>

__Plot country active cases, confirmed, recovered, and growth metrics__

```python

def smoother(inputdata,w,imax):
    data = 1.0*inputdata
    data = data.replace(np.nan,1)
    data = data.replace(np.inf,1)
    #print(data)
    smoothed = 1.0*data
    normalization = 1
    for i in range(-imax,imax+1):
        if i==0:
            continue
        smoothed += (w**abs(i))*data.shift(i,axis=0)
        normalization += w**abs(i)
    smoothed /= normalization
    return smoothed
# function to compute growth factor
def growth_factor(confirmed):
    confirmed_iminus1 = confirmed.shift(1, axis=0)
    confirmed_iminus2 = confirmed.shift(2, axis=0)
    return (confirmed-confirmed_iminus1)/(confirmed_iminus1-confirmed_iminus2)
#function to compute growth ratio
def growth_ratio(confirmed):
    confirmed_iminus1 = confirmed.shift(1, axis=0)
    return (confirmed/confirmed_iminus1)

# We don't need a function for growth rate since we can use the np.gradient() function.

# This is a function which plots (for in input country) the active, confirmed, and recovered cases, deaths, and the growth factor.
def plot_country_active_confirmed_recovered_growth_metrics(country):
    
    # Plots Active, Confirmed, and Recovered Cases. Also plots deaths.
    global_data = df.copy()
    country_data = global_data[global_data['Country/Region']==country]
    table = country_data.drop(['SNo','Province/State', 'Last Update'], axis=1)
    table['ActiveCases'] = table['Confirmed'] - table['Recovered'] - table['Deaths']
    table2 = pd.pivot_table(table, values=['ActiveCases','Confirmed', 'Recovered','Deaths'], index=['ObservationDate'], aggfunc=np.sum)
    table3 = table2.drop(['Deaths'], axis=1)
   
    # Growth Factor
    w = 0.5
    table2['GrowthFactor'] = growth_factor(table2['Confirmed'])
    table2['GrowthFactor'] = smoother(table2['GrowthFactor'],w,5)

    # 2nd Derivative
    table2['2nd_Derivative'] = np.gradient(np.gradient(table2['Confirmed'])) #2nd derivative
    table2['2nd_Derivative'] = smoother(table2['2nd_Derivative'],w,7)


    #Plot confirmed[i]/confirmed[i-1], this is called the growth ratio
    table2['GrowthRatio'] = growth_ratio(table2['Confirmed'])
    table2['GrowthRatio'] = smoother(table2['GrowthRatio'],w,5)
    
    #Plot the growth rate, we will define this as k in the logistic function presented at the beginning of this notebook.
    table2['GrowthRate']=np.gradient(np.log(table2['Confirmed']))
    table2['GrowthRate'] = smoother(table2['GrowthRate'],0.5,3)
    
    # horizontal line at growth rate 1.0 for reference
    x_coordinates = [1, 100]
    y_coordinates = [1, 1]
    #plots
    table2['Deaths'].plot(title='Deaths')
    plt.show()
    table3.plot() 
    plt.show()
    table2['GrowthFactor'].plot(title='Growth Factor')
    plt.plot(x_coordinates, y_coordinates) 
    plt.show()
    table2['2nd_Derivative'].plot(title='2nd_Derivative')
    plt.show()
    table2['GrowthRatio'].plot(title='Growth Ratio')
    plt.plot(x_coordinates, y_coordinates)
    plt.show()
    table2['GrowthRate'].plot(title='Growth Rate')
```

Here is the statistcis of `US`

<p align="center">
  <img src="/image/us1.png">
</p>

<p align="center">
  <img src="/image/us2.png">
</p>

<p align="center">
  <img src="/image/us3.png">
</p>

<p align="center">
  <img src="/image/us4.png">
</p>

<p align="center">
  <img src="/image/us5.png">
</p>


### <span id="jump5">EDA</span>

Since the COVID-19 is coming too fast and spread out through the world in such a short time with tremendous destruction. We want to perform some spread analysis.

__Number of countries affected over the time__

<p align="center">
  <img src="/image/spread.png">
</p>

Therefore, it is clear that the number of countries affected increased sharply between 20 Feb and 20 Mar. After efficient and instant control, the number became steady.

Now, let us visualize the confirmed, deaths, recovered and active caes trends over the world.

__COVID-19 global spread trends__

<p align="center">
  <img src="/image/globalspread.png">
</p>

__COVID-19 spread comparison of in different continents__

<p align="center">
  <img src="/image/continents1.png">
</p>

<p align="center">
  <img src="/image/continents2.png">
</p>

Now, let us further dive into the case of each epicentre.

### <span id="china">China(1st epicentre)</span>

__Plot country active cases, confirmed, recovered, and growth metrics in China__

<p align="center">
  <img src="/image/china1.png">
</p>

<p align="center">
  <img src="/image/china2.png">
</p>

<p align="center">
  <img src="/image/china3.png">
</p>

<p align="center">
  <img src="/image/china4.png">
</p>

<p align="center">
  <img src="/image/china5.png">
</p>

<p align="center">
  <img src="/image/china6.png">
</p>

Since China was the first epicentre we are checking for Provinces within China. Now, it seems the control of COVID-19 is relatively effcient in China. As a Chinese, I am pround of it! The growth rate and ratio is close to `0`.

__Province cases in China__

<p align="center">
  <img src="/image/procasechina.png">
</p>

As is known, public health officials and partners are working hard to identify the source of COVID-19. The first infections were linked to a live animal market in Wuhan, Hubei Province. Almost all the cases in China came from Hubei.

__Per day statistics for China__

<p align="center">
  <img src="/image/perdaychina.png">
</p>

Above is the per day statistics of China where the line graph shows flattening of curve for confirmed cases. However on `Feb 13` there was a sudden rise in Confirmed cases and on `Apr 17` there was a sudden rice in Deaths cases.




