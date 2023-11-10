# DS 701 Course Outline


## 9/5 MS Student Orientation


## 9/7 Introduction and Essential Tools
* [Intro to Python](./01-Intro-to-Python.ipynb)
* [GIT and Jupyter](./02A-Git-Jupyter.ipynb)


### Pandas
* [Pandas Review Notebook](./02B-Pandas-Review.ipynb)


### Prob and Stats Refresher
* [Prob and Stats Refresher Notebook](./03-Probability-and-Statistics-Refresher.ipynb)

__Definition: Fundamental Axions of Probability.__  Consider a set $\Omega$, referred to as the
_sample space._  A _probability
  measure_ on $\Omega$ is a function $P[\cdot]$ defined on all the subsets of $\Omega$ (the
  _events_) such that:
  
1. $P[\Omega] = 1$
2. For any event $A \subset \Omega$, $P[A] \geq 0.$
3. For any events $A, B \subset \Omega$ where $A \cap B =
    \emptyset$, $P[A \cup B] = P[A] + P[B]$.

__Definition.__ The _conditional probability_ of an event $A$ given that
event $B$ (having positive probability) is known to occur, is 

$$ P[A|B] = \frac{P[A \cap B]}{P[B]}  \text{ where } P[B] > 0 $$

__Definition.__ Two events $A$ and $B$ are __independent__ if $P[A\cap B] = P[A] \cdot P[B].$

This is exactly the same as saying that $P[A|B] = P[A].$  

__Definition.__ The variance of a random variable $X$ is

$$ \text{Var} (X) \equiv E[(X - E[X])^2]. $$

For example, given a discrete R.V. with $E[X] = \mu$ this would be:

$$ \text{Var} (X) = \sum_{x=-\infty}^{+\infty} (x-\mu)^2\; P[X=x]. $$

We use the symbol $\sigma^2$ to denote variance.

__Definition.__ For two random variables $X$ and $Y$, their _covariance_ is defined as:

$$\text{Cov}(X,Y) = E\left[(X-\mu_X)(Y-\mu_Y)\right]$$

If covariance is positive, this tells us that $X$ and $Y$ tend to be both above their means together, and both below their means together.

We will often denote $\text{Cov}(X,Y)$ as $\sigma_{XY}$

If we are interested in asking "how similar" are two random variables, we want to normalize covariance by the amount of variance shown by the random variables.

The tool for this purpose is __correlation__, ie, normalized covariance:

$$\rho(X,Y) = \frac{E\left[(X-\mu_X)(Y-\mu_Y)\right]}{\sigma_X \sigma_Y}$$

### Linear Algebra Refresher
* [Linear Algebra Refresher Notebook](./04-Linear-Algebra-Refresher.ipynb)

* Matrix and vector notation.
* Vector addition
* Matrix multiplication

__Definition.__ A matrix $A$ is called __invertible__ if there exists a matrix $C$ such that

$$ AC = I \text{  and  } CA = I. $$

In that case $C$ is called the _inverse_ of $A$.   Clearly, $C$ must also be square and the same size as $A$.

The inverse of $A$ is denoted $A^{-1}.$

* inner product
* vector norm
* $\ell_2$ norm

We start from the __law of cosines:__

#### Angle between Two Vectors

$$ c^2 = a^2 + b^2 - 2ab\cos\theta$$

Applying the law of cosines we get:
    
$$\Vert\mathbf{u}-\mathbf{v}\Vert^2 = \Vert\mathbf{u}\Vert^2 + \Vert\mathbf{v}\Vert^2 - 2\Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert\cos\theta$$

So 

$$ \mathbf{u}^T\mathbf{v} = \Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert\cos\theta$$

So 

$$ \frac{\mathbf{u}^T\mathbf{v}}{\Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert} = \cos\theta$$

* Least Squares
* Eigen decomposition

## 9/12 Spark Pitch Day


## 9/14 Distance and Similarity Functions, Timeseries 
* [Distance and Similarity Functions Notebook](./05-Distance-and-Similarity-Functions.ipynb)
* Assignment Due: Spark Project Choices

Topics

* Feature space representation
* one-hot encoding

It is helpful to constrain dissimilarities to be a _metric_.

The dissimilarity $d(x, y)$ between two objects $x$ and $y$ is a
__metric__ if

* $d(i, j) = 0 \leftrightarrow i == j\;\;\;\;\;\;\;\;$ (identity of indiscernables)
* $d(i, j) = d(j, i)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$ (symmetry)
* $d(i, j) \leq d(i, h)+d(h, j)\;\;$ (triangle inequality)

Norms

Assume some function $p(\mathbf{v})$ which measures the "size" of the vector $\mathbf{v}$.

$p()$ is called a __norm__ if:

* $p(a\mathbf{v}) = |a|\; p(\mathbf{v})\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$ (absolute scaling)
* $p(\mathbf{u} + \mathbf{v}) \leq p(\mathbf{u}) + p(\mathbf{v})\;\;\;\;\;\;\;\;\;\;$  (subadditivity)
* $p(\mathbf{v}) = 0 \leftrightarrow \mathbf{v}$ is the zero vector $\;\;\;\;$(separates points)

Norms are important for this reason, among others:
    
__Every norm defines a corresponding metric.__

That is, if $p()$ is a norm, then $d(\mathbf{x}, \mathbf{y}) = p(\mathbf{x}-\mathbf{y})$ is a metric.


## 9/19 Clustering I: k-means
[Clustering I: k-means NB](./06-Clustering-I-kmeans.ipynb)


* K-means clustering
* K-means++
* choosing k


## 9/21 Clustering II: In practice
* [Clustering II: In practice NB](./07-Clustering-II-In-Practice.ipynb)
* Assignment Due:  Homework 0

Topics

* example with scikit-learn

Specifically:

Let $a$ be the number of pairs that have the same label in $T$ and the same label in $C$. 

Let $b$ be: the number of pairs that have different labels in $T$ and different labels in $C$. 

Then the Rand Index is: 

$$ \text{RI}(T,C) = \frac{a+b}{n \choose 2} $$

__Definition of Adjusted Rand Index.__

To "calibrate" the Rand Index this way, we use the expected Rand Index of random labelings, denoted $E[\text{RI}]$.   

The Expected Rand Index considers $C$ to be a clustering that has the same cluster sizes as $T$, but labels are assigned at random.

Using that, we define the adjusted Rand index as a simple __rescaling__ of RI:

$$
\begin{equation}
\text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}
\end{equation}
$$

The computation of the $E[\text{RI}]$ and the $\max(\text{RI})$ are simple combinatorics (we'll omit the derivation).

* deciding number of clusters


## 9/26 Clustering III: Hierarchical Clustering
* [Clustering III: Hierarchical Clustering NB](./08-Clustering-III-Hierarchical-Clustering.ipynb)



## 9/28 Clustering IV: GMM and Expectation Maximization
* [Clustering IV: GMM and Expectation Maximization NB](./09-Clustering-IV-GMM-EM.ipynb)

From "hard" to "soft" clustering. Don't just assign each datapoint to one group or class, but rather somehow find the probability that they belong to each group or class. Also called __soft assignment__.

Another way to state it is that give some independent variable $x_j$, find

$$ P(C_i | x_j) = p_i $$

The conditional probabilities given that $x_j$ should sum to 1.

$$ \sum_i P(C_i | x_j) = 1 $$

### Mixture of Gaussians

A popular approach is to assume each cluster can be modeled as a gaussian with some mean and standard deviation.

$${\displaystyle f(x\;|\;\mu ,\sigma ^{2})={\frac {1}{\sqrt {2\sigma ^{2}\pi }}}\;exp({-{\frac {(x-\mu )^{2}}{2\sigma ^{2}}}}})$$

This can be a fair assumption for many naturally occuring phenomena.

For multiple independent variables, now each point is a vector in $n$ dimensions: $\mathbf{x} = (x_1, \dots, x_n)^T$.  

As a reminder, the multivariate Gaussian has a pdf (density) function of:

$$
f(x_1,\ldots ,x_n)=
\frac{1}{\sqrt{ (2\pi)^n |\boldsymbol  \Sigma }|}
    \exp \left(
            -\frac{1}{2} ({\mathbf x} - {\boldsymbol  \mu })^{\mathrm{T}}
            {\boldsymbol  \Sigma }^{-1}
            ({\mathbf  x}-{\boldsymbol  \mu})
        \right)
$$
where
$$
\Sigma_{i,j} = \mathrm{E}[(X_i - \mu_i)(X_j - \mu_j)] = \mathrm{Cov}[X_i, X_j]
$$
and
$$
| \boldsymbol  \Sigma | \equiv \mathrm{det } \boldsymbol \Sigma
$$
is the [determinant](https://en.wikipedia.org/wiki/Determinant) of $\boldsymbol \Sigma$.

Then a __Gaussian Mixture Model__ is defined by the means, variance and a 
weighting of the models, where $w_i$ is the prior probability (weight) of the 
$i^{th}$ Gaussian, such that
$$\sum_i w_i = 1,\;\;\; 0\leq w_i\leq 1.$$

Intuitively, $w_i$ tells us "what fraction of the data comes from Gaussian $i$."

Then the probability density at any point $x$ is given by:
    
$$ p(x) = \sum_i w_i \cdot \mathcal{N}(x\,|\, \mu_i, \Sigma_i) $$




## 10/3 Learning from Data
* [Learning from Data NB](./13-Learning-From-Data.ipynb)

The supervised learning problem in general is:
    
You are given some example data, which we'll think of abstractly as tuples
$\{(\mathbf{x}_i, y_i)\,|\,i = 1,\dots,N\}$.  

We usually call this the training dataset since we have the input data and the
expected out data, which we sometimes call _ground truth_.

Your goal is to learn a rule that allows you to predict $y_j$ for some
$\mathbf{x}_j$ that is not in the example data you were given.

$$ \mathbf{x}_j \mapsto y_j$$

Typically $\mathbf{x}_i$ is a vector and each component could be considered a
feature.

if $y$ is discrete valued, then the problem is called __classification__.

Illustrate concepts from data generated by
a known model plus additive noise.
    
$$ y = \sin(2\pi x) + n(x)$$

and the Gaussian random addition does not depend on $x$ so we cannot hope to predict it.

```python
N = 10  # 10 points
x = np.linspace(0, 1, N)

from numpy.random import default_rng

y = np.sin(2 * np.pi * x) + default_rng(2).normal(size = N, scale = 0.20)
``````

The class of models we will consider are polynomials.   They are of the form:

$$ 
y(x, \mathbf{w}) = 
    w_0 + w_1 x + w_2 x^2 + \dots + w_k x^k = 
    \sum_{j = 0}^k w_jx^j 
$$

where $k$ is the _order_ of the polynomial.

* Code examples on scikit-learn.
* Parameters and hyperparameters
* Holding out data
* Hold out strategies
    * k-fold cross validation



## 10/5 Classification I: Decision Trees
* [Classification I: Decision Trees NB](./14-Classification-I-Decision-Trees.ipynb)
* Assignment Due: Homework 1



## 10/10 <span style='color: red;'>NO CLASS; MONDAY SCHEDULE</span>



## 10/12 Classification II: k-Nearest Neighbors
* [Classification II: k-Nearest Neighbors NB](./15-Classification-II-kNN.ipynb)
* Assignment Due:  Project Deliverable 0

The Curse of Dimensionality


## 10/15 <span style='color: red;'>MIDTERM START



## 10/17 Ethical Analysis of Data Science Projects (Seth Villegas)


## 10/19 & 10/24 Classification III: Naive Bayes, Support Vector Machines
* [Classification III: Naive Bayes, Support Vector Machines NB](./16-Classification-III-NB-SVM.ipynb)




## 10/26 SVD I : Low Rank Approximation 
* [SVD I : Low Rank Approximation NB](./10-Low-Rank-and-SVD.ipynb)
* Assignment Due: MIDTERM END midnight




## 10/31 SVD II: Dimensionality Reduction
* [SVD II: Dimensionality Reduction NB](./11-Dimensionality-Reduction-SVD-II.ipynb)
* Assignment Due: Project Deliverable 1





## 11/2 Regression I: Linear Regression
* [Regression I: Linear Regression NB](./17-Regression-I-Linear.ipynb)
* Assignment Due: Project Ethics Audit




## 11/7 Regression II: Logistic Regression
* [Regression II: Logistic Regression NB](./18-Regression-II-Logistic.ipynb)
* Assignment Due: Project Deliverable 2



## 11/9 Regression III: In Practice
* [Regression III: In Practice NB](./19-Regression-III-More-Linear.ipynb)



## 11/14 Recommender Systems
* [Recommender Systems NB](./20-Recommender-Systems.ipynb)



## 11/16 Gradient Descent
* [The Original Gradient Descent NB](./23-Gradient-Descent.ipynb)
* [Revised Gradient Descent NB](./23-Gradient-Descent-Edits.ipynb)
* Assignment Due: Project Deliverable 3



## 11/17 <span style='color: red;'>Homework 2 Assigned</span>

## 11/21 Neural Networks I

## 11/23 <span style='color: red;'>NO CLASS; Thanksgiving Break</span>

## 11/28 Neural Networks II
* Assignment Due: Project Deliverable 4

## 11/30 Network Analysis I
* Assignment Due: Homework 3

## 12/5 Network Analysis II

## 12/7 <span style='color: red;'>Final Project Presentations</span>

## 12/12 <span style='color: red;'>Final Project Presentations</span>
* Assignment Due: Final Project Report

## 12/13 Demo Day Participation Required

