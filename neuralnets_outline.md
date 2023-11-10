# Neural Nets Outline

## Recap of Relevant Topics

### (Supervised??) Classification and Regression Models

In the [Learning from Data](./13-Learning-From-Data.ipynb) lecture, we formulated the __Supervised Learning__ approach of learning from data.

You are given, or have to create, some data, which we'll think of abstractly as tuples
$\{(\mathbf{x}_i, y_i)\,|\,i = 1,\dots,N\}$.  

We usually call this the training dataset since we have the input data and the
expected output data, which we sometimes call _ground truth_.

> Show examples of images of cats and dogs as $x$ and the label as $y$.
> Also show examples where $x$ is number of bedrooms and zip codes and $y$ is
> the price of a house.

Your goal is to learn a rule (or a function or parameters) that allows you to predict some new $y_j$ when given some new
$\mathbf{x}_j$ that is not in the example data you were given.

$$ \mathbf{x}_j \mapsto y_j$$

Typically $\mathbf{x}_i$ is a vector and each component could be considered a
feature.

if $y$ is discrete valued (e.g. cat, doc), then the problem is called __classification__. If $y$ is continuous valued (e.g. price of a house), then the problem is call __regression__ for historical reasons.

We also saw that we can define an __objective function__ (or __loss function__, or __dissimilarity function__ )  that measures how well our rule is doing in predicting the output.

A popular one is the __squared error__ function, which when minimized we get the __least squares__ solution.

We saw a simple artificial example with weights that we have to optimize.

Some concepts seen so far.

* the supervised learning problem for both classification and regression
* defining and objective or loss function
* the notion of hyperparameters that must by set some how
* The problem of model overfitting
* separating data into training and test sets
    * one time random train/test split
    * k-fold cross validation
* Linear regression as a shallow network with no activation function
* Logistic Regression as shallow network with tanh activation function

In [Classification II](./15-Classification-II-kNN.ipynb) we talked about the Curse of Dimensionality.

## New Lectures

* We were introduce to supervised learning -- learning from data
* We talked about loss functions or objective functions as something we often want to minimize.
* We'll start in 1-D where it is easier to visualize and the math is simpler and then generalize to higher dimensions.

Start with a graph of a quadratic function for the time being, even though we know that our objective functions won't be simple convex functions with global minimum.

From a random starting point, we'll want to move towards the minimum.

How do we know which way to move?

We calculate the slope.

Definition of slope:
* as a limit
* derived analytically

Show python examples...

Basically go through Karpathy example with
* graphviz of the computation graph...
* intro to chain rule...
* forward pass
* backward pass
* training loop
* PyTorch equivalent

$$ \hat{y} $$

Show line fitting example from UDL for shallow networks, then deep networks...

Then extend to multidimensions
* Linear algebra based derivatives and chain rule...
* 2-D plot from UDL showing saddle points...

Learning rate and momentum from UDL? Different optimizers, LR schedules...

Extension to CNN.