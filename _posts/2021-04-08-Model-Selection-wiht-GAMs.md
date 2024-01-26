The last part of moddeling with (univariate spline) GAMs is choosing the smoothing parameter \\( \lambda \\). This post will elaborate on this, using the `scikit-learn` `GriddSearchCV` functionality to do this. We'll use `pyGAM` to do the heavy lifting, and we'll use the same data as the last post


```python
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy as sp
import seaborn as sns
from statsmodels import api as sm

%matplotlib inline

df = pd.read_csv('mcycle.csv')
df = df.drop('Unnamed: 0', axis=1)

min_time = df.times.min()
max_time = df.times.max()

fig, ax = plt.subplots(figsize=(8, 6))
blue = sns.color_palette()[0]
ax.scatter(df.times, df.accel, c=blue, alpha=0.5)
ax.set_xlabel('time')
ax.set_ylabel('Acceleration')
```




    Text(0,0.5,'Acceleration')




    
![png](2021-04-08-Model-Selection-wiht-GAMs_files/2021-04-08-Model-Selection-wiht-GAMs_1_1.png)
    



```python
def splines(df):

    def R(x, z):
        return ((z - 0.5)**2 - 1 / 12) * ((x - 0.5)**2 - 1 / 12) / 4 - ((np.abs(x - z) - 0.5)**4 - 0.5 * (np.abs(x - z) - 0.5)**2 + 7 / 240) / 24

    R = np.frompyfunc(R, 2, 1)

    def R_(x):
        return R.outer(x, knots).astype(np.float64)

    y, X = patsy.dmatrices('accel ~ times + R_(times)', data=df)

    knots = df.times.quantile(np.linspace(0, 1, q))
    
def GAM(df, q=20, gamma=1.0):    

    S = np.zeros((q + 2, q + 2))
    S[2:, 2:] = R_(knots)

    B = np.zeros_like(S)
    B[2:, 2:] = np.real_if_close(sp.linalg.sqrtm(S[2:, 2:]), tol=10**8)

    def fit(y, X, B, lambda_=gamma):
        # build the augmented matrices
        y_ = np.vstack((y, np.zeros((q + 2, 1))))
        X_ = np.vstack((X, np.sqrt(lambda_) * B))
    
        return sm.OLS(y_, X_).fit()
    
    return fit(X, y, b, lambda_)
```


```python
min_time = df.times.min()
max_time = df.times.max()

plot_x = np.linspace(min_time, max_time, 100)
plot_X = patsy.dmatrix('times + R_(times)', {'times': plot_x})

results = GAM(df)

fig, ax = plt.subplots(figsize=(8, 6))
blue = sns.color_palette()[0]
ax.scatter(df.times, df.accel, c=blue, alpha=0.5)
ax.plot(plot_x, results.predict(plot_X))
ax.set_xlabel('time')
ax.set_ylabel('accel')
ax.set_title(r'$\lambda = {}$'.format(1.0))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~/anaconda3/lib/python3.6/site-packages/patsy/compat.py in call_and_wrap_exc(msg, origin, f, *args, **kwargs)
         35     try:
    ---> 36         return f(*args, **kwargs)
         37     except Exception as e:


    ~/anaconda3/lib/python3.6/site-packages/patsy/eval.py in eval(self, expr, source_name, inner_namespace)
        165         return eval(code, {}, VarLookupDict([inner_namespace]
    --> 166                                             + self._namespaces))
        167 


    <string> in <module>()


    NameError: name 'R_' is not defined

    
    The above exception was the direct cause of the following exception:


    PatsyError                                Traceback (most recent call last)

    <ipython-input-3-9eb1425fd693> in <module>()
          3 
          4 plot_x = np.linspace(min_time, max_time, 100)
    ----> 5 plot_X = patsy.dmatrix('times + R_(times)', {'times': plot_x})
          6 
          7 results = GAM(df)


    ~/anaconda3/lib/python3.6/site-packages/patsy/highlevel.py in dmatrix(formula_like, data, eval_env, NA_action, return_type)
        289     eval_env = EvalEnvironment.capture(eval_env, reference=1)
        290     (lhs, rhs) = _do_highlevel_design(formula_like, data, eval_env,
    --> 291                                       NA_action, return_type)
        292     if lhs.shape[1] != 0:
        293         raise PatsyError("encountered outcome variables for a model "


    ~/anaconda3/lib/python3.6/site-packages/patsy/highlevel.py in _do_highlevel_design(formula_like, data, eval_env, NA_action, return_type)
        163         return iter([data])
        164     design_infos = _try_incr_builders(formula_like, data_iter_maker, eval_env,
    --> 165                                       NA_action)
        166     if design_infos is not None:
        167         return build_design_matrices(design_infos, data,


    ~/anaconda3/lib/python3.6/site-packages/patsy/highlevel.py in _try_incr_builders(formula_like, data_iter_maker, eval_env, NA_action)
         68                                       data_iter_maker,
         69                                       eval_env,
    ---> 70                                       NA_action)
         71     else:
         72         return None


    ~/anaconda3/lib/python3.6/site-packages/patsy/build.py in design_matrix_builders(termlists, data_iter_maker, eval_env, NA_action)
        694                                                    factor_states,
        695                                                    data_iter_maker,
    --> 696                                                    NA_action)
        697     # Now we need the factor infos, which encapsulate the knowledge of
        698     # how to turn any given factor into a chunk of data:


    ~/anaconda3/lib/python3.6/site-packages/patsy/build.py in _examine_factor_types(factors, factor_states, data_iter_maker, NA_action)
        441     for data in data_iter_maker():
        442         for factor in list(examine_needed):
    --> 443             value = factor.eval(factor_states[factor], data)
        444             if factor in cat_sniffers or guess_categorical(value):
        445                 if factor not in cat_sniffers:


    ~/anaconda3/lib/python3.6/site-packages/patsy/eval.py in eval(self, memorize_state, data)
        564         return self._eval(memorize_state["eval_code"],
        565                           memorize_state,
    --> 566                           data)
        567 
        568     __getstate__ = no_pickling


    ~/anaconda3/lib/python3.6/site-packages/patsy/eval.py in _eval(self, code, memorize_state, data)
        549                                  memorize_state["eval_env"].eval,
        550                                  code,
    --> 551                                  inner_namespace=inner_namespace)
        552 
        553     def memorize_chunk(self, state, which_pass, data):


    ~/anaconda3/lib/python3.6/site-packages/patsy/compat.py in call_and_wrap_exc(msg, origin, f, *args, **kwargs)
         41                                  origin)
         42             # Use 'exec' to hide this syntax from the Python 2 parser:
    ---> 43             exec("raise new_exc from e")
         44         else:
         45             # In python 2, we just let the original exception escape -- better


    ~/anaconda3/lib/python3.6/site-packages/patsy/compat.py in <module>()


    PatsyError: Error evaluating factor: NameError: name 'R_' is not defined
        times + R_(times)
                ^^^^^^^^^



```python

```
