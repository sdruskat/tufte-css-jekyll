This is a short post introducing Generalised Additive Models (GAMs) - not the nuts and bolts, but some things you can do with them. We will be follwoing this post: https://petolau.github.io/Analyzing-double-seasonal-time-series-with-GAM-in-R/ but we won't go so deep into the theory, all the data come from the github repository linked in the post.

GAMs are a very flexible modelling technique, but unfortunately there isn't a Python package as good as R's `mgcv` yet. It's something we're working on! In this post, I'll fit a simple GAM using `PyGAM` and in a later post I'll talk about some theory, and some extensions.

GAMs are smooth, semi-parametric models of the form:

$$ y = \sum_{i=0}^{n-1} \beta_i f_i\left(x_i\right) $$

where \\(y\\) is the dependent variable, \\(x_i\\) are the independent variables, \\(\beta\\) are the model coefficients, and \\(f_i\\) are the feature functions. We build the \\(f_i\\) using a type of function called a spline; splines allow us to automatically model non-linear relationships without having to manually try out many different transformations on each variable.

Nedt we'll load some data and fit a GAM!


```python
import feather
from pygam import LinearGAM
from pygam.utils import generate_X_grid
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
```


```python
def load_data(file):
    df = feather.read_dataframe(file)

    weekday_map = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
    df['weekday'] = df['week'].map(weekday_map)

    n_type = pd.unique(df['type'])
    n_date = pd.unique(df['date_time'])
    n_weekdays = pd.unique(df['weekday'])
    period = 48
    begin = "2012-02-27"
    end = "2012-03-12"
    mask = (df['date_time'] > begin) & (df['date_time'] <= end)
    data = df.loc[mask]

    data = data[df['type'] == n_type[0]]
    return data

data = load_data('DT_4_ind.dms')

data.plot(x='date_time', y='value')
```

    /Users/thomas.kealy/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:16: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      app.launch_new_instance()





    <matplotlib.axes._subplots.AxesSubplot at 0x11f0bbac8>




    
![png](2021-08-13-Time-Series-with-GAMS_files/2021-08-13-Time-Series-with-GAMS_2_2.png)
    


The above code loads some data, and does a little bit of preprocessing - makes weekday names more legible to humans, and just selects a few weeks of data about 'Commerical Properties'. You can see that the time series has a lot of structure - exhibiting daily, but also weekly periodicity. There are 48 measurements during the day and 7 days during the week so that will be our independent variables to model response variable - electricity load. Let's build it!


```python
period = 48
N = data.shape[0] # number of observations in the train set
window = N / period # number of days in the train set

weekly = data['weekday']
x = np.array(range(1, period+1))
daily = np.tile(x, int(window))

matrix_gam = pd.DataFrame(columns=['daily', 'weekly', 'load'])
matrix_gam['load'] = data['value']
matrix_gam['daily'] = daily
matrix_gam['weekly'] = weekly
```


```python
gam = LinearGAM(n_splines=10).gridsearch(matrix_gam[['daily', 'weekly']], matrix_gam['load'])
XX = generate_X_grid(gam)
```

    100% (11 of 11) |#########################| Elapsed Time: 0:00:00 Time: 0:00:00



```python
fig, axs = plt.subplots(1, 2)
fig.set_figheight(10)
fig.set_figwidth(15)
titles = ['daily', 'weekly']

for i, ax in enumerate(axs):
    pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
    confi = np.asarray(confi)
    confi = confi.squeeze()
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi, c='r', ls='--')
    ax.set_title(titles[i])
```


    
![png](2021-08-13-Time-Series-with-GAMS_files/2021-08-13-Time-Series-with-GAMS_6_0.png)
    


This is good! You can see that the electricity load follows an approximate `sin` pattern during the day, and that the electricity load falls off during the week! If we'd tried using a linear model to do this, we'd have had to build these features manually - the good thing about GAMs is that they do this for us. Let's visualise the fit.


```python
predictions = gam.predict(matrix_gam[['daily', 'weekly']])
```


```python
fig = plt.figure(figsize=(15, 10))
plt.plot(data['date_time'], matrix_gam['load'])
plt.plot(data['date_time'], predictions)
plt.xticks(rotation='vertical')
plt.legend(['True', 'Predicted'])
```




    <matplotlib.legend.Legend at 0x11eb87f28>




    
![png](2021-08-13-Time-Series-with-GAMS_files/2021-08-13-Time-Series-with-GAMS_9_1.png)
    


Alas, this isn't the best fit, but it'll do!


```python

```
