---
title: "Improving my first ANN"
date: 2018-06-23
tags: [machine learning, neural network]
header:
    image: "images/NON/smart_machine.jpg"
excerpt: "What is the right number of neurons?"
---


## Finding the optimal number of neurons?

While the weights are optimized by training the network, there are a few
hyperparameters which needs to be set at the beginning. Creating a neural network
is actually a problem of finding the right architecture.For example, what
is the optimal number of neurons?

In my previous post, I created an ANN with two hidden layers and each of the layers
had three neurons. Why three? Well, three is my lucky number. However, you could argue
that it may not be the lucky number of my neural network.

You could be very right with that. Therefore, it would be better, we reiterate
through a combination of network architectures with different amout of neurons
in hidden layer 2 and 3.

I continued to use the "ann.py" script I wrote in the previous post. This script
creates the basic architecture of my neural network. I only modify the execution
file "main.py" for this purpose.

This is the modified main.py

```python

import ann as a
import timeit
import numpy as np
import pandas as pd

start = timeit.default_timer()

#This is the input matrix X with the input features A and B
X = np.array([[0,0],
             [1,0],
             [0,1],
             [1,1],
             ])

#This is the output vector
y = np.array([[1],
             [0],
             [0],
             [1],
             ])

#empty list for recording training cost (error)
cost = []

#reiterating between 1 and 20 neurons for layer 2 and layer 3
for neu_2 in range(1,21,1):
    for neu_3 in range (1,21,1):
        ANN = a.ThreeLayerNet(X=X, y=y, epoch=1000,neurons_l2=neu_2, neurons_l3=neu_3 )
        ANN.train_NN()
        print ("Architecture: {}/{}\tCost: {}\t\tOutput: {}".
               format(neu_2,neu_3,np.round(ANN.cost,6),(np.round(ANN.y_est,3)).flatten()))
        cost.append([neu_2, neu_3, ANN.cost,(np.round(ANN.y_est,3))])

#creating pandas dataframe to record architecture combination and its training cost
test_results = pd.DataFrame(cost,columns=["Neurons Layer 2",
                                          "Neurons Layer 3",
                                          "Training Cost",
                                          "Output"])
test_sorted = test_results.sort_values(by=["Training Cost"])

pd.set_option('display.width', 1000)
print ("\nTop 10 combinations:\n",test_sorted.head(10))
print ("\nBottom 10 combinations:\n",test_sorted.tail(10))

#counting running time
end = timeit.default_timer()
time_elapsed = (end-start)
print ("\nRunning time of script in seconds: {}".format(round(time_elapsed,5)))

```

This time the main.py script included two for-loops which reiterated through
a number of neurons from 1 to 20 (21 is excluded in python ranges) for each
of the two hidden layers. I keep the amount of epoch at 1000. Each
network will be trained 1000 times. As we have 20 combinations for each of the
two layers and train each of it 2000 times, the computer is calculating the
neural network 400,000 times.

Let's run it...

<video width="630" height="270" controls="controls">
  <source src="/images/NON/main_py_non.mp4" type="video/mp4">
</video>

400,000 neural network with multliple layers and neurons, forward and backpropagation,
and all in just 30 seconds. I am still working only with the CPU. Still impressed
with the computational power of machines these days.

## Who is the winner?

As you can see in the video, I calculate for each combination an item I
call cost. In the previous post, I wrote a function loss_calculation for
ann.py and in this function I calculate "cost". The formula is a standard
way of measuring the loss of simple classification problems.

This was the python code.

```python
def loss_calculation(self):
        self.error = self.y - self.y_est
        self.cost = 0.5*np.sum((np.power(self.error,2)))
        self.cost_deriv = self.y_est - self.y
```

It takes the difference between the actual and estimated output. Then, it
calculates the power of 2 of this difference. Why power of 2? This is a simple
mathematical trick to make sure that the difference between the two numbers is
always positive. Finally, it sums it up across all outputs and multiplies it by
0.5. Now, you may ask why 0.5? Well, another mathematical trick. When taking
the derivative of this formula, bringing down the exponent 2 to the coefficient 0.5,
results in 1 (0.5*2) and the multiplication of this formula disappears.

The formula measures the distance between the actual and estimated output and
punishes exponentially large gaps.

For example,

<img src="{{ site.url }}{{ site.baseurl }}/images/NON/cost_examples.PNG"
alt="Examples of Training Cost">

You can see in the last row that an output of only ones resulted in a cost of
almost 1. The actual output supposed to be 1,0,1,0 and in two cases the loss
is at its maximum. Therefore, the cost are very high. The first line shows
network estimated outputs of 0.5 for all four observations. In this case, the
training cost are 0.5 as the distance to the actual output is smaller.














