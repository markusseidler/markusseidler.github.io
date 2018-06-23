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


400,000 neural networks with multliple layers and neurons, forward and backpropagation,
and all in just 29 seconds. I am still working only with the CPU. Just impressed
with the computational power of machines these days.

Have you watched the video to the end? I always find it very relaxing watching a
computer running a script. It's like watching the rain... although there is a lot
of work in the background, it looks so peaceful and smooth.

## Who is the winner?

As you can see in the video, I calculate for each combination an item I
call "cost". In the previous post, I wrote a function loss_calculation for
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
always positive. Furthermore, it punishes large differences more.

Finally, it sums it up across all outputs and multiplies it by
0.5. Now, you may ask why 0.5? Well, another mathematical trick. When taking
the derivative of this formula, bringing down the exponent 2 to the coefficient 0.5,
it results in 1 (0.5*2) and the multiplication of this formula disappears.

The formula measures the distance between the actual and estimated output and
punishes exponentially large gaps.

For example,

<img src="{{ site.url }}{{ site.baseurl }}/images/NON/cost_examples.PNG"
alt="Examples of Training Cost">

You can see in the last row that an output of only ones resulted in a cost of
almost 1. The actual output supposed to be 1,0,1,0 and in two cases the loss
is at its maximum. Therefore, the cost are very high. The first line shows
network estimated outputs of 0.5 for all four observations. In this case, the
training cost are 0.5 as the exponential distance to the actual output is smaller.
If we would calculate the average of distance, then in both cases, the cost would
be 0.5. Due to the power of 2 in the formula, the first case produces higher cost.

Let's look at the top and bottom 10 results of our training.

<img src="{{ site.url }}{{ site.baseurl }}/images/NON/top10_bottom10.PNG"
alt="Top and Bottom 10 results">


You can see that the top 10 all achieved good results. In practice, we would
round the outputs to the closest integer and get 0 or 1. However, in this cost, I
kept in unrounded to see the difference in output compared to cost. The best
results could be achieved with 3 or 4 neurons in the first hidden layer. The
second layer could have more neurons but the first layer was rather small.

We can also see for this simple problem and the small amount of observations
(only 4), complex hidden layers with up to 20 neurons failed to produce meaningful
training results.

## Plot it. Please.

Finally, I thought it would be very interesting to plot the training results of
the various architectures in a 3D plot. By doing this, we could see where the results
started to improve and at what number of neurons the network failed to perform.

For this, I added some code to my "main.py" file that plots the training results on
the z-axis while the number of neurons in Layer 2 (the first hidden layer) are plotted on
the x-axis and of Layer 3 are on the y-axis.

```python

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

#setting the figure and adding 4 subplots in 2x2 grid.
fig = plt.figure()
ax1 = fig.add_subplot(221, projection = "3d")
ax2 = fig.add_subplot(222, projection = "3d")
ax3 = fig.add_subplot(223, projection = "3d")
ax4 = fig.add_subplot(224, projection = "3d")

#iterating through the four plots
for i in range (1,5):
    ax = eval("ax{}".format(i))
    ax.scatter (test_results["Neurons Layer 2"],
            test_results["Neurons Layer 3"],
            test_results["Training Cost"],
            s=60, c=test_results["Training Cost"],
            cmap=cm.viridis)

    ax.set_xlabel ("Neurons Layer 2")
    ax.set_ylabel ("Neurons Layer 3")
    ax.set_zlabel ("Training Cost")

#plotting the figure
plt.tight_layout()
plt.show()
```

I created four different subplots that I can show the 3D plot from various angles.


<img src="{{ site.url }}{{ site.baseurl }}/images/NON/3d plot cost.PNG"
alt="3D plot of training cost against number of neurons">

The yellow dots indicate the highest cost, values of up to 1 while green dots show
combinations with cost of around 0.5. Best combinations are the dark dots on the ground,
close to training cost of 0. You can see in the plots on the right side, that the training
cost are rather low for any amount of neurons in layer 3. However, the plots on the left side
show that once the amount of neurons in layer 2 exceeded 10 that the training cost rises.
Especially, a combination of neurons close to 20 resulted in a network failure.

## That's it? You can plot better than this.

Really? Maybe. What if let the plot automatically rotate along its axis?

Just for fun.

```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

#setting the figure and with one subplot
fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(221, projection = "3d")

#creating the plot
ax.scatter (test_results["Neurons Layer 2"],
            test_results["Neurons Layer 3"],
            test_results["Training Cost"],
            s=60, c=test_results["Training Cost"],
            cmap=cm.viridis)

ax.set_xlabel ("Layer 2")
ax.set_ylabel ("Layer 3")
ax.set_zlabel ("Training Cost")

#rotating the plot along its x and y axis
for j in range (5, 51,15):
    for i in range (0, 361, 2):
                    ax.view_init (elev=j, azim=i)
                    plt.draw()
                    plt.pause(0.001)
#plotting the figure
plt.show()
```

And now rotate, rotate, rotate...


<video width="630" height="270" controls="controls">
  <source src="/images/NON/3d plot rotation.mp4" type="video/mp4">
</video>















