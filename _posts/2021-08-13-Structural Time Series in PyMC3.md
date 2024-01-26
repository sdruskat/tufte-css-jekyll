```python
%matplotlib inline
import matplotlib as mpl
from matplotlib import pylab as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sns.set_context("notebook", font_scale=1.)
sns.set_style("darkgrid")
```


```python
passengers = pd.read_csv('passengers.csv', header=0, sep=';')
passengers['Passengers'] = passengers['Passengers'].astype(float)
passengers['Month'] = pd.to_datetime(passengers['Month'])
passengers.set_index('Month', inplace=True)
passengers.plot(figsize=(12, 6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x134a79d50>




    
![png](2021-08-13-Structural%20Time%20Series%20in%20PyMC3_files/2021-08-13-Structural%20Time%20Series%20in%20PyMC3_1_1.png)
    



```python
passengers['Passengers'].values[0]
```




    112.0




```python
with pm.Model():
    delta = pm.GaussianRandomWalk('delta', mu=0, sd=1, shape=(144,))
    mu = pm.GaussianRandomWalk('mu', mu=delta, sd=1, shape=(143,), observed=passengers['Passengers'])
    trace = pm.sample(5000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [delta]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='24000' class='' max='24000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [24000/24000 00:11<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 20 seconds.



```python
az.plot_trace(trace)
```

    /Users/tomkealy/opt/anaconda3/lib/python3.7/site-packages/arviz/data/io_pymc3.py:91: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      FutureWarning,
    /Users/tomkealy/opt/anaconda3/lib/python3.7/site-packages/arviz/plots/traceplot.py:195: UserWarning: rcParams['plot.max_subplots'] (20) is smaller than the number of variables to plot (144), generating only 20 plots
      UserWarning,





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x132062690>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x13210eed0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x134d33350>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1321ceed0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x13595a9d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x132154a90>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x13229f650>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x132384b50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1323c2f90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1324bc7d0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1324fcb50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1325eaed0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x132639bd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x132727cd0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x132773850>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x132862d10>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1328a6fd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x13299d990>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1329e0c10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x132ad8610>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x132b1ad90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x132c08f50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x132c91a10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x132d57ad0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x132da6690>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x132e96b50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x132ed8f50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x132fcf7d0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x133014b50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x133101ed0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x13314dbd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x13323ccd0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1332d0850>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x133395d10>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1333d9fd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1334cf990>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x133511c10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x13360c610>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x13364cd90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x133739f50>]],
          dtype=object)




    
![png](2021-08-13-Structural%20Time%20Series%20in%20PyMC3_files/2021-08-13-Structural%20Time%20Series%20in%20PyMC3_4_2.png)
    



```python

```


```python
def plot_forecast(data_df,
                  col_name,
                  forecast_start,
                  forecast_mean, 
                  forecast_scale, 
                  forecast_samples,
                  title, 
                  x_locator=None, 
                  x_formatter=None):
    """Plot a forecast distribution against the 'true' time series."""
    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    y = data_df[col_name]
    x = data_df.index

    num_steps = data_df.shape[0]
    num_steps_forecast = forecast_mean.shape[-1]
    num_steps_train = num_steps - num_steps_forecast

    ax.plot(x, y, lw=2, color=c1, label='ground truth')

    forecast_steps = data_df.loc[forecast_start:].index

    ax.plot(forecast_steps, 
            forecast_samples.T, 
            lw=1, 
            color=c2, 
            alpha=0.1)

    ax.plot(forecast_steps, 
            forecast_mean, 
            lw=2, 
            ls='--', 
            color=c2,
            label='forecast')

    ax.fill_between(forecast_steps,
                   forecast_mean-2*forecast_scale,
                   forecast_mean+2*forecast_scale, 
                   color=c2, 
                   alpha=0.2)

    ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
    yrange = ymax-ymin
    ax.set_ylim([ymin - yrange*0.1, ymax + yrange*0.1])
    ax.legend()
    return fig, ax

fig, ax = plot_forecast(
    passengers,
    'Passengers',
    '1959-01-01',
    forecast_mean, 
    forecast_scale, 
    forecast_samples,
    title='Airplane Passenger Numbers')
ax.legend(loc="upper left")
ax.set_ylabel("Passenger Numbers")
ax.set_xlabel("Month")
fig.autofmt_xdate()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-4-d3135643ac66> in <module>
         54     'Passengers',
         55     '1959-01-01',
    ---> 56     forecast_mean,
         57     forecast_scale,
         58     forecast_samples,


    NameError: name 'forecast_mean' is not defined



```python

```
