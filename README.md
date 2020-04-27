## The Logistic Regression Algorithm

Logistic regression is the appropriate regression analysis to conduct when the dependent variable is binary.  Like all regression analyses, the logistic regression is a predictive analysis.  Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. [statisticsolutions](https://www.statisticssolutions.com/what-is-logistic-regression/)

Logistic Regression is a method for classifying data into discrete outcomes. The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values.

<br>

![Logistic Regression Description](images/LR.png)

The Logistic regression algorithm was implemented in [logistic_regression.py](logistic_regression.py) and you can reference this [notebook](Logistic%20Regression%20Notebook.ipynb) for more practical details on how the linear regression algorithm works.

<br>

The class `LogisticRegression` which contains several variables & methods (public and private) to carry out the relationships modelled.

```python
class LogisticRegression:
    def __init__(self, x, y, alpha=0.01, num_iter=1000, verbose=False, lambd=0.0):
        pass
```


At initilization of the Logistic Regression model:
- `x` will be the input feature which should be a ($X<sup>m</sup>, n) matrix.
- `y` will be the target feature which should be a ($Y<sup>m</sup>, 1) matrix.

`alpha` is the learning rate and `num_iter` the number of iterations used in gradient descent. `verbose` if True will produce the detailed output of the cost function for diagnostic purposes, and `lambd` is the parameter used to perform regularization (L2 Regularization) on the model, it is of no effect if `lambd` is set to 0.0.

**Note** "m" is the number of training examples and "n" is the number of features.

The choice of numpy array was to perform vectorization on the data, thus avoiding the constant use of excessive for loops and thus optimizing the program.

The Hypothesis of a simple logistic regression is given as: sigmoid(h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x)

The sigmoid function being given as: 1 / (1 + e<sup>z</sup>)

where x is the input variable.

The cost function or **mean squared error** is used to measure the accuracy of our hypothesis. This takes the average difference of all the result of the with the inputs from and the actual output y's.

The mean of the cost function is halved as a convenience for the computation of gradient descent. 

```python
def fit(self, timeit=True, count_at=100):
    pass
```

So when we have our hypothesis function and we have a way of measuring how well it fits into the data. We then need to estimate the parameters in the hypothesis function and this is where **Gradient Descent** comes in and this process goes on for a period of time until the cost converge to a global minimum.

Follow me on **Twitter** [@GM_Olalekan](https://twitter.com/GM_Olalekan?s=09)
