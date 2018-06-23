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













