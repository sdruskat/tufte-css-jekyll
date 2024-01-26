In this post we'll discuss some ways of doing feature selection within a Bayesian framework

Let \\(y\\) be a set of real-valued observations. The basic linear regression model can be decribed as

$$ y_t = \beta^Tx + \varepsilon_t$$

Where \\( \varepsilon \sim N\left(0, \sigma^2\right) \\). This can be equivalently written:

$$ y \sim N\left(\beta^Tx, \sigma^2\right) $$.

We will first place a prior on \\(\beta\\), like:

$$ \beta_i \sim \left(\frac{\tau}{2}\right)^p \mathrm{exp}\left(-\tau \mid\beta\mid \right)$$

Then we will use the horseshoe prior:

$$ \beta_i \sim N\left(0, \lambda_i\right) $$

$$ \lambda_i \sim \mathrm{Cauchy}^+\left(0, \tau\right)$$ 

$$ \tau \sim \mathrm{Cauchy}^+\left(0, 1\right)$$


```python
import numpy as np
import pymc3 as pm
from sklearn.metrics import r2_score
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('ggplot')
%matplotlib inline

from pymc3_models.exc import PyMC3ModelsError
from pymc3_models.models import BayesianModel
```


```python
class BayesianLassoRegression(BayesianModel):
    """
    Linear Regression built using PyMC3.
    """

    def __init__(self):
        super(BayesianLassoRegression, self).__init__()
        self.ppc = None

    def create_model(self):
        """
        Creates and returns the PyMC3 model.
        Note: The size of the shared variables must match the size of the training data. Otherwise, setting the shared variables later will raise an error. See http://docs.pymc.io/advanced_theano.html
        Returns
        ----------
        the PyMC3 model
        """
        model_input = theano.shared(np.zeros([self.num_training_samples, self.num_pred]))

        model_output = theano.shared(np.zeros(self.num_training_samples))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
        }

        model = pm.Model()
        with model:
            beta = pm.Laplace('beta', mu = 0.0, b = 10.0, shape = (1, self.num_pred))
            sigma = pm.HalfNormal('sigma', sd=10.0)
            ll = pm.math.dot(model_input, beta.T)
            mu = T.sum(ll)
            y = pm.Normal('y', mu=mu, sd=sigma, observed=model_output)
        return model

    def fit(self, X, y, inference_type='nuts', minibatch_size=None, inference_args={'draws': 1000}):
        """
        Train the Linear Regression model
        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]
        y : numpy array, shape [n_samples, ]
        inference_type : string, specifies which inference method to call. Defaults to 'advi'. Currently, only 'advi' and 'nuts' are supported
        minibatch_size : number of samples to include in each minibatch for ADVI, defaults to None, so minibatch is not run by default
        inference_args : dict, arguments to be passed to the inference methods. Check the PyMC3 docs for permissable values. If no arguments are specified, default values will be set.
        """
        self.num_training_samples, self.num_pred = X.shape

        self.inference_type = inference_type

        if y.ndim != 1:
            y = np.squeeze(y)

        if not inference_args:
            inference_args = self._set_default_inference_args()

        if self.cached_model is None:
            self.cached_model = self.create_model()

        if minibatch_size:
            with self.cached_model:
                minibatches = {
                    self.shared_vars['model_input']: pm.Minibatch(X, batch_size=minibatch_size),
                    self.shared_vars['model_output']: pm.Minibatch(y, batch_size=minibatch_size),
                }

                inference_args['more_replacements'] = minibatches
        else:
            self._set_shared_vars({'model_input': X, 'model_output': y})

        self._inference(inference_type, inference_args)

        return self

    def predict(self, X, return_std=False):
        """
        Predicts values of new data with a trained Linear Regression model
        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]
        return_std : Boolean flag of whether to return standard deviations with mean values. Defaults to False.
        """

        if self.trace is None:
            raise PyMC3ModelsError('Run fit on the model before predict.')

        num_samples = X.shape[0]

        if self.cached_model is None:
            self.cached_model = self.create_model()

        self._set_shared_vars({'model_input': X, 'model_output': np.zeros(num_samples)})

        self.ppc = pm.sample_ppc(self.trace, model=self.cached_model, samples=2000)

        if return_std:
            return self.ppc['y'].mean(axis=0), ppc['y'].std(axis=0)
        else:
            return self.ppc['y'].mean(axis=0)

    def score(self, X, y):
        """
        Scores new data with a trained model.
        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]
        y : numpy array, shape [n_samples, ]
        """

        return r2_score(y, self.predict(X))

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(LinearRegression, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(LinearRegression, self).load(file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
```


```python
class HorseshoeRegression(BayesianModel):
    """
    Linear Regression built using PyMC3.
    """

    def __init__(self):
        super(HorseshoeRegression, self).__init__()
        self.ppc = None

    def create_model(self):
        """
        Creates and returns the PyMC3 model.
        Note: The size of the shared variables must match the size of the training data. Otherwise, setting the shared variables later will raise an error. See http://docs.pymc.io/advanced_theano.html
        Returns
        ----------
        the PyMC3 model
        """
        model_input = theano.shared(np.zeros([self.num_training_samples, self.num_pred]))

        model_output = theano.shared(np.zeros(self.num_training_samples))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
        }

        model = pm.Model()
        m = 10
        ss = 3
        dof = 25
        with model:
            sigma = pm.HalfNormal('sigma', 2)
            tau_0 = m / (self.num_training_samples - m) * sigma / T.sqrt(self.num_training_samples)

            tau = pm.HalfCauchy('tau', tau_0)
            c2  = pm.InverseGamma('c2', dof/2, dof/2 * ss**2)
            lam = pm.HalfCauchy('lam', 1)

            l1 = lam * T.sqrt(c2)
            l2 = T.sqrt(c2 + tau * tau * lam * lam)
            lam_d = l1 / l2

            beta = pm.Normal('beta', 0, tau * lam_d, shape=self.num_pred)
            y_hat = T.dot(model_input, beta)

            y = pm.Normal('y', mu=y_hat, observed=model_output)

        return model

    def fit(self, X, y, inference_type='nuts', minibatch_size=None, inference_args={'draws': 1000}):
        """
        Train the Linear Regression model
        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]
        y : numpy array, shape [n_samples, ]
        inference_type : string, specifies which inference method to call. Defaults to 'advi'. Currently, only 'advi' and 'nuts' are supported
        minibatch_size : number of samples to include in each minibatch for ADVI, defaults to None, so minibatch is not run by default
        inference_args : dict, arguments to be passed to the inference methods. Check the PyMC3 docs for permissable values. If no arguments are specified, default values will be set.
        """
        self.num_training_samples, self.num_pred = X.shape

        self.inference_type = inference_type

        if y.ndim != 1:
            y = np.squeeze(y)

        if not inference_args:
            inference_args = self._set_default_inference_args()

        if self.cached_model is None:
            self.cached_model = self.create_model()

        if minibatch_size:
            with self.cached_model:
                minibatches = {
                    self.shared_vars['model_input']: pm.Minibatch(X, batch_size=minibatch_size),
                    self.shared_vars['model_output']: pm.Minibatch(y, batch_size=minibatch_size),
                }

                inference_args['more_replacements'] = minibatches
        else:
            self._set_shared_vars({'model_input': X, 'model_output': y})

        self._inference(inference_type, inference_args)

        return self

    def predict(self, X, return_std=False):
        """
        Predicts values of new data with a trained Linear Regression model
        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]
        return_std : Boolean flag of whether to return standard deviations with mean values. Defaults to False.
        """

        if self.trace is None:
            raise PyMC3ModelsError('Run fit on the model before predict.')

        num_samples = X.shape[0]

        if self.cached_model is None:
            self.cached_model = self.create_model()

        self._set_shared_vars({'model_input': X, 'model_output': np.zeros(num_samples)})

        self.ppc = pm.sample_ppc(self.trace, model=self.cached_model, samples=2000)

        if return_std:
            return selfl.ppc['y'].mean(axis=0), ppc['y'].std(axis=0)
        else:
            return self.ppc['y'].mean(axis=0)

    def score(self, X, y):
        """
        Scores new data with a trained model.
        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]
        y : numpy array, shape [n_samples, ]
        """

        return r2_score(y, self.predict(X))

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(LinearRegression, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(LinearRegression, self).load(file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
```


```python
# Generate some sparse data to play with
np.random.seed(42)

n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0  # sparsify coef
y = np.dot(X, coef)

# add noise
y += 0.01 * np.random.normal(size=n_samples)

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]
```


```python
def plot_beta(b, b_hat, std=None):
    x = range(len(b))
    plt.plot(x, b, 'k', alpha=0.5)
    plt.plot(x, b_hat, alpha=0.5)

    if std is not None:
        plt.fill_between(x, b_hat + std, b_hat - std, alpha=0.3)
    plt.show()

def make_data(n, m):
    from scipy.stats import bernoulli

    alpha = 3
    sigma = 1
    sig_p = 0.05

    beta = np.zeros(m)
    f = np.zeros(m)
    for i in range(m):
        if bernoulli(sig_p).rvs():
            if bernoulli(0.5).rvs():
                beta[i] = np.random.normal(10, 1)
            else:
                beta[i] = np.random.normal(-10, 1)
            f[i] = 1

        else:
            beta[i] = np.random.normal(0, 0.25)

    X = np.random.normal(0, 1, (n, m))
    y = np.random.normal(X.dot(beta) + alpha, sigma)

    return X, y, beta, f

X, y, beta, f = make_data(100, 200)

with mpl.rc_context():
    mpl.rc('figure', figsize=(30, 10))
    plt.plot(beta, '--', color='navy', label='original coefficients')
```


    
![png](2021-03-01-bayesian-feature-selection_files/2021-03-01-bayesian-feature-selection_5_0.png)
    



```python
lasso = BayesianLassoRegression()
lasso.fit(X, y)
y_pred_lasso = lasso.predict(X)
r2_score_lasso = lasso.score(X, y)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)
```

    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sigma, beta]
    Sampling 2 chains: 100%|██████████| 3000/3000 [04:20<00:00, 10.81draws/s]
    The chain reached the maximum tree depth. Increase max_treedepth, increase target_accept or reparameterize.
    The chain reached the maximum tree depth. Increase max_treedepth, increase target_accept or reparameterize.
    The estimated number of effective samples is smaller than 200 for some parameters.
    100%|██████████| 2000/2000 [00:15<00:00, 125.78it/s]

    BayesianLassoRegression()
    r^2 on test data : -0.003733


    



```python
horse_lasso = HorseshoeRegression()
horse_lasso.fit(X, y)
y_pred_hlasso = horse_lasso.predict(X)
r2_score_hlasso = r2_score(y, y_pred_hlasso)
print("r^2 on test data : %f" % r2_score_hlasso)
```

    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [beta, lam, c2, tau, sigma]
    Sampling 2 chains: 100%|██████████| 3000/3000 [01:37<00:00, 30.67draws/s]
     47%|████▋     | 934/2000 [00:07<00:08, 120.36it/s]


```python
with mpl.rc_context():
    mpl.rc('figure', figsize=(30, 10))
    plt.plot(coef, '--', color='navy', label='original coefficients')
    plt.plot(lasso.trace.get_values('beta').mean(0).T, color='gold', label='original coefficients')
    plt.plot(horse_lasso.trace.get_values('beta').mean(0).T, color='lightgreen', label='original coefficients')
    plt.legend(loc='best')
```


    
![png](2021-03-01-bayesian-feature-selection_files/2021-03-01-bayesian-feature-selection_8_0.png)
    



```python
from pymc3 import summary, traceplot
```


```python

```
