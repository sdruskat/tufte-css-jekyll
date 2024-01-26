Often, we have data which has a group structure. For example, in the dataset we use in this post, radon measurements were taken in ~900 houses in 85 counties. It's unreasonable to expect that radon levels do not vary by state as well as house, and so we will integrate this into our analysis. 

Typically, in linear regression we assume that each data point is independent and regresses with a constant slope amongst each other:

$$ y = X^T\beta + \varepsilon $$

where

$$ \varepsilon \sim N\left(0, I\right) $$ 

and \\(X\\) are known as fixed effects coefficients. To define a mixed model, we include a term \\(Z\eta\\), which corresponds to *random* effects. The model is now:

$$ y = X^T\beta + Z^T\eta + \varepsilon $$

where

$$ \varepsilon \sim N\left(0, \sigma\right) $$ 

and 

$$ \eta \sim N\left(0, \sigma^2I\right) $$

We wish to infer \\(\beta, \eta, \sigma\\). Given the random effects have mean 0, the term \\(X^T\beta\\) captures the data's mean amd the term \\(Z^T\eta\\) captures variations in the data.


```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_context('notebook')

import pystan
import statsmodels.formula.api as smf
from patsy import dmatrices

# Import radon data
radon = pd.read_csv('radon.csv')
radon.columns = radon.columns.map(str.strip)
radon_mn = radon.assign(fips=radon.stfips*1000 + radon.cntyfips)[radon.state=='MN']
```


```python
radon_mn.county = radon_mn.county.str.strip()
n_county = radon_mn.county.unique()

county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
county = radon_mn['county_code'] = radon_mn.county.replace(county_lookup).values
radon = radon_mn.activity
log_radon = radon_mn['log_radon']
floor_measure = radon_mn.floor.values

u = np.log(radon_mn.Uppm)
```


```python
radon_mn['fips'] = radon_mn.stfips*1000 + radon_mn.cntyfips
```


```python
radon_mn.activity.apply(lambda x: np.log(x+0.1)).hist(bins=25, figsize=(20, 10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a229295c0>




    
![png](2021-01-01-Mixed-Models_files/2021-01-01-Mixed-Models_4_1.png)
    


The simplest qway to fit a mixed model is to use StatsModels: which is what we'll do!


```python
data = radon_mn[['county', 'log_radon', 'floor']]
```


```python
formula = "log_radon ~ floor + county"
```


```python
md  = smf.mixedlm(formula, data, groups=data["county"])
mdf = md.fit()
```

    /Users/thomas.kealy/anaconda3/lib/python3.7/site-packages/statsmodels/regression/mixed_linear_model.py:2066: ConvergenceWarning: The Hessian matrix at the estimated parameter values is not positive definite.
      warnings.warn(msg, ConvergenceWarning)



```python
print(mdf.summary())
```

                    Mixed Linear Model Regression Results
    ======================================================================
    Model:                 MixedLM      Dependent Variable:      log_radon
    No. Observations:      919          Method:                  REML     
    No. Groups:            85           Scale:                   0.5279   
    Min. group size:       1            Likelihood:              -994.6192
    Max. group size:       116          Converged:               Yes      
    Mean group size:       10.8                                           
    ----------------------------------------------------------------------
                                Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    ----------------------------------------------------------------------
    Intercept                    0.887    0.810  1.095 0.274 -0.701  2.475
    county[T.ANOKA]              0.043    1.088  0.040 0.968 -2.090  2.177
    county[T.BECKER]             0.662    1.167  0.568 0.570 -1.625  2.949
    county[T.BELTRAMI]           0.700    1.123  0.623 0.533 -1.501  2.901
    county[T.BENTON]             0.567    1.147  0.494 0.621 -1.682  2.816
    county[T.BIG STONE]          0.650    1.167  0.557 0.578 -1.637  2.936
    county[T.BLUE EARTH]         1.138    1.104  1.031 0.303 -1.026  3.301
    county[T.BROWN]              1.109    1.148  0.966 0.334 -1.140  3.358
    county[T.CARLTON]            0.158    1.113  0.142 0.887 -2.022  2.339
    county[T.CARVER]             0.679    1.128  0.602 0.547 -1.533  2.890
    county[T.CASS]               0.541    1.136  0.476 0.634 -1.685  2.767
    county[T.CHIPPEWA]           0.869    1.148  0.757 0.449 -1.381  3.118
    county[T.CHISAGO]            0.197    1.128  0.175 0.861 -2.013  2.408
    county[T.CLAY]               1.119    1.104  1.014 0.311 -1.044  3.282
    county[T.CLEARWATER]         0.481    1.148  0.419 0.675 -1.768  2.730
    county[T.COOK]              -0.170    1.204 -0.141 0.888 -2.529  2.189
    county[T.COTTONWOOD]         0.380    1.148  0.331 0.741 -1.870  2.630
    county[T.CROW WING]          0.273    1.108  0.246 0.805 -1.898  2.444
    county[T.DAKOTA]             0.485    1.091  0.444 0.657 -1.654  2.623
    county[T.DODGE]              0.930    1.167  0.797 0.426 -1.357  3.216
    county[T.DOUGLAS]            0.863    1.115  0.774 0.439 -1.322  3.049
    county[T.FARIBAULT]         -0.109    1.128 -0.096 0.923 -2.320  2.102
    county[T.FILLMORE]           0.532    1.204  0.442 0.658 -1.827  2.891
    county[T.FREEBORN]           1.226    1.114  1.100 0.271 -0.958  3.409
    county[T.GOODHUE]            1.078    1.104  0.976 0.329 -1.087  3.242
    county[T.HENNEPIN]           0.506    1.040  0.486 0.627 -1.533  2.544
    county[T.HOUSTON]            0.900    1.128  0.798 0.425 -1.311  3.111
    county[T.HUBBARD]            0.389    1.136  0.342 0.732 -1.838  2.616
    county[T.ISANTI]             0.204    1.167  0.175 0.861 -2.083  2.490
    county[T.ITASCA]             0.083    1.111  0.074 0.941 -2.095  2.260
    county[T.JACKSON]            1.149    1.136  1.012 0.312 -1.077  3.375
    county[T.KANABEC]            0.382    1.148  0.333 0.739 -1.868  2.631
    county[T.KANDIYOHI]          1.189    1.147  1.036 0.300 -1.060  3.437
    county[T.KITTSON]            0.730    1.167  0.625 0.532 -1.557  3.017
    county[T.KOOCHICHING]       -0.018    1.123 -0.016 0.987 -2.218  2.182
    county[T.LAC QUI PARLE]      2.064    1.204  1.714 0.086 -0.296  4.423
    county[T.LAKE]              -0.401    1.115 -0.359 0.719 -2.586  1.785
    county[T.LAKE OF THE WOODS]  0.989    1.148  0.862 0.389 -1.260  3.238
    county[T.LE SUEUR]           0.876    1.136  0.772 0.440 -1.349  3.102
    county[T.LINCOLN]            1.435    1.147  1.250 0.211 -0.814  3.684
    county[T.LYON]               1.092    1.119  0.975 0.329 -1.102  3.286
    county[T.MAHNOMEN]           0.499    1.309  0.381 0.703 -2.066  3.064
    county[T.MARSHALL]           0.751    1.115  0.674 0.500 -1.433  2.936
    county[T.MARTIN]             0.220    1.123  0.196 0.845 -1.981  2.420
    county[T.MCLEOD]             0.432    1.106  0.391 0.696 -1.735  2.599
    county[T.MEEKER]             0.359    1.136  0.316 0.752 -1.868  2.586
    county[T.MILLE LACS]         0.081    1.204  0.067 0.946 -2.278  2.440
    county[T.MORRISON]           0.294    1.115  0.263 0.792 -1.891  2.478
    county[T.MOWER]              0.840    1.105  0.760 0.447 -1.327  3.006
    county[T.MURRAY]             1.614    1.309  1.233 0.217 -0.951  4.179
    county[T.NICOLLET]           1.291    1.148  1.125 0.261 -0.959  3.540
    county[T.NOBLES]             1.055    1.167  0.905 0.366 -1.231  3.342
    county[T.NORMAN]             0.398    1.166  0.341 0.733 -1.889  2.684
    county[T.OLMSTED]            0.454    1.091  0.416 0.677 -1.685  2.592
    county[T.OTTER TAIL]         0.752    1.119  0.672 0.501 -1.441  2.945
    county[T.PENNINGTON]         0.303    1.167  0.260 0.795 -1.983  2.590
    county[T.PINE]              -0.074    1.128 -0.066 0.947 -2.285  2.137
    county[T.PIPESTONE]          0.991    1.147  0.863 0.388 -1.258  3.239
    county[T.POLK]               0.852    1.148  0.743 0.458 -1.397  3.101
    county[T.POPE]               0.420    1.204  0.349 0.727 -1.939  2.779
    county[T.RAMSEY]             0.309    1.077  0.287 0.774 -1.802  2.421
    county[T.REDWOOD]            1.111    1.136  0.978 0.328 -1.115  3.337
    county[T.RENVILLE]           0.800    1.166  0.686 0.493 -1.486  3.086
    county[T.RICE]               0.977    1.111  0.879 0.379 -1.201  3.155
    county[T.ROCK]               0.448    1.204  0.372 0.710 -1.911  2.807
    county[T.ROSEAU]             0.795    1.105  0.719 0.472 -1.371  2.961
    county[T.SCOTT]              0.935    1.106  0.846 0.398 -1.232  3.102
    county[T.SHERBURNE]          0.238    1.119  0.213 0.832 -1.956  2.431
    county[T.SIBLEY]             0.388    1.148  0.338 0.736 -1.862  2.637
    county[T.ST LOUIS]           0.036    0.857  0.042 0.967 -1.644  1.715
    county[T.STEARNS]            0.631    1.089  0.579 0.562 -1.503  2.765
    county[T.STEELE]             0.717    1.113  0.644 0.519 -1.464  2.897
    county[T.STEVENS]            0.922    1.204  0.766 0.444 -1.437  3.281
    county[T.SWIFT]              0.139    1.148  0.122 0.903 -2.110  2.389
    county[T.TODD]               0.852    1.166  0.730 0.465 -1.434  3.138
    county[T.TRAVERSE]           1.133    1.148  0.987 0.323 -1.116  3.382
    county[T.WABASHA]            0.955    1.122  0.851 0.395 -1.244  3.155
    county[T.WADENA]             0.434    1.136  0.382 0.702 -1.792  2.660
    county[T.WASECA]            -0.181    1.147 -0.158 0.875 -2.430  2.068
    county[T.WASHINGTON]         0.476    1.095  0.435 0.664 -1.670  2.622
    county[T.WATONWAN]           1.813    1.167  1.553 0.120 -0.475  4.100
    county[T.WILKIN]             1.353    1.309  1.034 0.301 -1.212  3.919
    county[T.WINONA]             0.769    1.105  0.696 0.487 -1.397  2.935
    county[T.WRIGHT]             0.779    1.106  0.705 0.481 -1.388  2.947
    county[T.YELLOW MEDICINE]    0.330    1.204  0.274 0.784 -2.030  2.689
    floor                       -0.689    0.071 -9.760 0.000 -0.828 -0.551
    Group Var                    0.528                                    
    ======================================================================
    


    /Users/thomas.kealy/anaconda3/lib/python3.7/site-packages/statsmodels/base/model.py:1092: RuntimeWarning: invalid value encountered in sqrt
      bse_ = np.sqrt(np.diag(self.cov_params()))



```python
fe_params = pd.DataFrame(mdf.fe_params, columns=['LMM'])
random_effects = pd.DataFrame(mdf.random_effects)
random_effects = random_effects.transpose()
random_effects = random_effects.rename(index=str, columns={'Group': 'LMM'})

#%% Generate Design Matrix for later use
Y, X   = dmatrices(formula, data=data, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices("log_radon ~ county", data=data, return_type='matrix')
X      = np.asarray(X) # fixed effect
Z      = np.asarray(Z) # mixed effect
Y      = np.asarray(Y).flatten()
nfixed = np.shape(X)
nrandm = np.shape(Z)
```


```python
#%% ploting function 
def plotfitted(fe_params=fe_params,random_effects=random_effects,X=X,Z=Z,Y=Y):
    plt.figure(figsize=(18,9))
    ax1 = plt.subplot2grid((2,2), (0, 0))
    ax2 = plt.subplot2grid((2,2), (0, 1))
    ax3 = plt.subplot2grid((2,2), (1, 0), colspan=2)
    
    fe_params.plot(ax=ax1)
    random_effects.plot(ax=ax2)
    
    ax3.plot(Y.flatten(), 'o', color='k', label='Observed', alpha=.25)
    for iname in fe_params.columns.get_values():
        fitted = np.dot(X, fe_params[iname]) + np.dot(Z, random_effects[iname]).flatten()
        print("The MSE of "+ iname + " is " + str(np.mean(np.square(Y.flatten()-fitted))))
        ax3.plot(fitted, lw=1, label=iname, alpha=.5)
    ax3.legend(loc=0)
    #plt.ylim([0,5])
    plt.show()

plotfitted(fe_params=fe_params, random_effects=random_effects, X=X, Z=Z, Y=Y)
```

    The MSE of LMM is 0.4784635589205236



    
![png](2021-01-01-Mixed-Models_files/2021-01-01-Mixed-Models_11_1.png)
    



```python
xbar = radon_mn.groupby('county')['floor'].mean().rename(county_lookup).values
x_mean = xbar[county]
```


```python
stan_code = """
data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[N] u;
  vector[N] x;
  vector[N] x_mean;
  vector[N] y;
} 
parameters {
  vector[J] a;
  vector[3] b;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + u[i]*b[1] + x[i]*b[2] + x_mean[i]*b[3];
}
model {
  mu_a ~ normal(0, 1);
  a ~ normal(mu_a, sigma_a);
  b ~ normal(0, 1);
  y ~ normal(y_hat, sigma_y);
}
"""
```


```python
stan_datadict = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1
                          'u': u,
                          'x_mean': x_mean,
                          'x': floor_measure,
                          'y': log_radon}

stan_datadict['prior_only'] = 0

sm = pystan.StanModel(model_code=stan_code)
fit = sm.sampling(data=stan_datadict, iter=1000)
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_b7928e9396770558600f70afccda0012 NOW.
    /Users/thomas.kealy/anaconda3/lib/python3.7/site-packages/pystan/misc.py:399: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      elif np.issubdtype(np.asarray(v).dtype, float):



```python
fit['b'].mean(0)
```




    array([ 0.68677081, -0.68513895,  0.3876573 ])




```python
with mpl.rc_context():
    mpl.rc('figure', figsize=(30, 10))
    fit.plot('b')
```


    
![png](2021-01-01-Mixed-Models_files/2021-01-01-Mixed-Models_16_0.png)
    



```python

```
