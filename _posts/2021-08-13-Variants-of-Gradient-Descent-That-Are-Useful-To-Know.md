# Variants of Gradient Descent which are useful to know

Sometimes, pure gradient descent can be too slow, or for some other reason it's not what you need. This post dicusses some alternatives.

First, we'll make some classification data and run vanilla gradient descent to create a baseline for more exotic variants


```python
from sklearn.datasets import make_circles
import autograd.numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

np.random.seed(0)

NUM_FEATURES=6
NUM_CLASSES=2
NUM_SAMPLES=400

X, y = make_circles(n_samples=NUM_SAMPLES, factor=.3, noise=.05)

plt.figure()
reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
```




    <matplotlib.collections.PathCollection at 0x1148cce48>




    
![png](2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_files/2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_2_1.png)
    


This data is not linearly separable, so it'll be difficult to classify these points using the same method we used last time. No fear though! We can add features to X which will make the data linearly seperable - we'll transform X into a higher space. You can think of the current data set as points on a hill, and we're looking down at them. If the blue points are higher than the red, then a plane which slices the hill in half will separate the data. The next function creates the extra columns which define the new space.


```python
def quadratic_kernal(X):
    """
    Adds quadratic features. 
    This expansion allows your linear model to make non-linear separation.
    
    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
    
    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    X_expanded = np.zeros((X.shape[0], 6))
    
    # TODO:<your code here>
    
    X_0_squared = X[:,0] * X[:, 0]
    X_1_squared = X[:,1] * X[:, 1]
    X_10 = X[:, 0] * X[:, 1]
    
    X_expanded[:, 0] = X[:, 0]
    X_expanded[:, 1] = X[:, 1]
    X_expanded[:, 2] = X_0_squared
    X_expanded[:, 3] = X_1_squared
    X_expanded[:, 4] = X_10
    X_expanded[:, 5] = np.ones((X.shape[0],)).T
    return X_expanded
```


```python
X_kernal = quadratic_kernal(X)
```

When we classified points in the last post, we used the sigmoid function to create probabioilties.


```python
def sigmoid(x):
    prob = 1.0 / (1.0 + np.exp(-x))
    return prob

def predict_prob(w, x):
    return sigmoid(np.dot(x, w))

def predict(probs):
    return np.greater(probs, 0.5)
```

Let's try it out on some random data, and plot the predictions.


```python
weights = np.random.randn(NUM_FEATURES,)
y_probs = predict_prob(weights, X_kernal)
y_pred = predict(y_probs)


plt.figure()
reds = y_pred == 0
blues = y_pred == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
```




    <matplotlib.collections.PathCollection at 0x114a935c0>




    
![png](2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_files/2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_9_1.png)
    


As you can see, this gets everything wrong! Let's try a better strategy.


```python
def loss(weights, inputs, targets):
    num_samples = inputs.shape[0]
    y_pred = predict_prob(weights, inputs)
    label_probabilities = y_pred * targets + (1 - y_pred) * (1 - targets)
    return -np.sum(np.log(label_probabilities))
```


```python
loss(weights, X_kernal, y)
```




    357.50054021310473




```python
from autograd import grad
gradient = grad(loss)

def gradient_descent_auto(X, y, cost, num_classes=2, learning_rate=0.001, num_iters=500):
    from autograd import grad
    num_samples, num_features = X.shape
    weights = np.zeros((num_features,))
    gradient = grad(cost)
    yield weights, cost(weights, X, y)
    
    for i in range(num_iters):
        nabla = gradient(weights, X, y)
        weights = weights - learning_rate * nabla
        yield weights, cost(weights, X, y)
```


```python
weights = gradient_descent_auto(X_kernal, y, loss, learning_rate=0.0001, num_classes=NUM_CLASSES)
w = list(weights)
costs = [x[1] for x in w]
```


```python
plt.plot(costs)
```




    [<matplotlib.lines.Line2D at 0x1164a2780>]




    
![png](2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_files/2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_15_1.png)
    



```python
y_probs = predict_prob(w[-1][0], X_kernal)
y_pred = predict(y_probs)

plt.figure()
reds = y_pred == 0
blues = y_pred == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
```




    <matplotlib.collections.PathCollection at 0x11519f080>




    
![png](2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_files/2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_16_1.png)
    


Much better!

Gradient descent is taking a lot longer to converge in this setting. Let's try a different variant of Gradient descent - one with momentum. Momentum is a method that helps accelerate gradient descent in the relevant direction and dampens oscillations as can be seen in image below. It does this by adding a fraction $\alpha$ of the update vector of the past time step to the current update vector.
<br>
<br>

$$ \nu_t = \alpha \nu_{t-1} + \eta \nabla_w L(w_t, x_{i_j}, y_{i_j}) $$
$$ w_t = w_{t-1} - \nu_t$$

<br>



```python
def gradient_descent_with_momentum(X, y, cost, num_classes=2, learning_rate=0.001, alpha=0.9, num_iters=500):
    from autograd import grad
    num_samples, num_features = X.shape
    weights = np.zeros((num_features,))
    nu = np.zeros_like(weights)
    gradient = grad(cost)
    yield weights, cost(weights, X, y)
    
    for i in range(num_iters):
        nabla = gradient(weights, X, y)
        nu = alpha * nu + learning_rate * nabla
        weights = weights - nu
        yield weights, cost(weights, X, y)
```


```python
weights = gradient_descent_with_momentum(X_kernal, y, loss, learning_rate=0.0001, num_classes=NUM_CLASSES)
w = list(weights)
costs = [x[1] for x in w]
```


```python
plt.plot(costs)
```




    [<matplotlib.lines.Line2D at 0x1147e8860>]




    
![png](2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_files/2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_20_1.png)
    



```python
y_probs = predict_prob(w[-1][0], X_kernal)
y_pred = predict(y_probs)

plt.figure()
reds = y_pred == 0
blues = y_pred == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
```




    <matplotlib.collections.PathCollection at 0x116c33cf8>




    
![png](2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_files/2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_21_1.png)
    


As you can see, this algorithm is converging faster than vanilla gradient descent! A final variant of Gradient Descent is RMSProp which uses squared gradients to adjust learning rate:

$$ G_j^t = \alpha G_j^{t-1} + (1 - \alpha) g_{tj}^2 $$
$$ w_j^t = w_j^{t-1} - \dfrac{\eta}{\sqrt{G_j^t + \varepsilon}} g_{tj} $$


```python
def RMSProp(X, y, cost, num_classes=2, learning_rate=0.001, alpha=0.9, num_iters=500):
    from autograd import grad
    num_samples, num_features = X.shape
    weights = np.zeros((num_features,))
    g2 = np.zeros_like(weights)
    eps = 1e-8
    gradient = grad(cost)
    yield weights, cost(weights, X, y)
    
    for i in range(num_iters):
        nabla = gradient(weights, X, y)
        g2 = alpha * g2 + (1 - alpha) * nabla**2
        weights = weights - learning_rate * nabla / np.sqrt(g2 + eps)
        yield weights, cost(weights, X, y)
```


```python
weights = RMSProp(X_kernal, y, loss, learning_rate=0.0001, num_classes=NUM_CLASSES)
w = list(weights)
costs = [x[1] for x in w]
```


```python
plt.plot(costs)
```




    [<matplotlib.lines.Line2D at 0x116cd32b0>]




    
![png](2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_files/2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_25_1.png)
    



```python
y_probs = predict_prob(w[-1][0], X_kernal)
y_pred = predict(y_probs)

plt.figure()
reds = y_pred == 0
blues = y_pred == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
```




    <matplotlib.collections.PathCollection at 0x116c89cc0>




    
![png](2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_files/2021-08-13-Variants-of-Gradient-Descent-That-Are-Useful-To-Know_26_1.png)
    

