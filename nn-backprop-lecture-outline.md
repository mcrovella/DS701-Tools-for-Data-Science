# NN Lecture Outlines

## Lecture 1: Gradient Descent and Back Propagation

Aproximate outline:
1. The "Unreasonable" Effectiveness of Deep Neural Networks
    1. Examples from UDL book?
    2. CS231n javascript live example?
    3. Google NN playground?
    4. And of course Large Language Models
        1. Show the inflection point? From "Emergent Abilities..."??
    5. Application preceded Theory
        1. Prevented some people from pursuing
        2. But not uncommon in engineering. Cite avionics and other example from leCunn Princeton lecture
        3. So can be a bit ironic to teach as theory first
        4. We'll balance the 2 here and in the Spring Course
2. Restate Supervised Learning Problem Formulation
    1. Definition of Least Squares Loss and finding "a" minumum
    2. DNNs loss function don't have simple global minumum (non-convex)
        1. Show exampe plot picture from DS701 Gradient Lecture
3. Minimizing a simple 1-D convex function with Gradient Descent (for illustration and building intuition)
    1. Derivative refresher and illustrations
        1. LinR and LogR have convex loss and in fact have closed form solutions (verify)
            1. Show LinR and LogR loss functions, or refer to an "appendix" notebook
        2. DL models are non-linear and don't have simple closed form solutions. We have to search for them.
        3. Define a quadratic function and plot it.
        4. Define derivative as the limit and show on plot
        5. Remind about analytical derivatives (cite wikipedia?), show notation
            1. examples: d/dx(3), d/dx(3x), d/dx(3x^2), d/dx(e^3x), d/dx(3x e^3x), anything else for Gabor?
            2. show how you can verify your analytical solution with the limit theorem both analytically and numerically in python
    2. 1st derivative as slope and Gradient Descent
        1. So we have to evaluate the derivative(slope) at a particular location -- show on plot with slider
        2. Now we can use the slope (1st derivative) to take a step down the slope
        3. ask the class to how we step compared to slope (-slope)
        4. the amount we step, the optiimization step, is called the Learning Rate in the machine learning literature. The size of the step will impact how fast we get to a minimum -- the larger the step, the faster we move, but the more we may bounce around the minimum rather than get exactly to it. We could possible show an animation of different learning rates and how close it can get to the maximum.
2. Gabor example in 2D with local and global minima and saddle points (perhaps push to 2nd lecture on learning rate, optimizers and regularizers)
    1. Follow UDL book and lecture. Recreate the plots? Or reeuse.
    2. partial derivative w.r.t. each variable. Everything else is a constant
    3. show the cases of bouncing out of a local minimum, or bouncing around in a minimum.
3. Go to simple neuron example
    1. Show biological and algorithmic neuron
        1. Sidebar or Notes at End: single neuron was showed by Minsky and Papert to not be unviversal... and argued that it was a deadend. Caused the 1st AI Winter. What wasn't understood until later that deep arrangements did work as a universal approximator. Cite the Unsual Effectiveness of DL, and DL as Universal Approximators. Another intereating note is that application preceded the theory which turned alot of people off, but is not that unusual in engineering. Cite aviation... other...?
    2. Show single neuron example ReLU(3x+2) or ReLU(wx + b)
4. Build the Value Class
    1. Maybe for DS701, we'll just show the simple class, with a few steps of construcion
    2. We're going to follow the PyTorch style interface so this wil help you understand what is going on inside Pytorch
    3. Doing arithmetic with a new Class
    4. Capturing the operation and the child objects, let's you think about it as a _computation graph_ which will be crucial for scaling this up to any size neural network
    5. Show GraphViz visualization of the graph.
5. To take the derivative of this, it helps to represent each operation as a function
    1. Order of operation is multiplication, f(.), addition, g(.) then ReLU, h(.)
    2. So the operation could be expressed as h(g(f(.)))
    3. How do we take the derivative of nested functions? The Chain Rule
    4. Define the chain rule: d/dx g(f(x)) ... https://en.wikipedia.org/wiki/Chain_rule. show with both Lagrange and Leibniz' notation
    5. So we will care about partial derivative, which we see is localized at each node, evaluated at the current value at that node (see Leibniz's notation). This trick makes this scalable to large networks!
    6. Add gradient property to the Value object, and update graphviz to show.
    7. Now look at the graph in terms of the derivatives...
    8. We run the graph forward to calculate the value at each node, then use the chain rule to start at the end of the network and calculate the partial derivatives.
    9. Go through an example by hand.
    10. Then show the Value object implementation...
    11. Then show in a training loop.
    12. Then show PyTorch code...
4. Close with Google Torch Playground?


## Lecture 2: Multi-D Loss, Optimizers and Regularization


## Lecture 3: CNNs

