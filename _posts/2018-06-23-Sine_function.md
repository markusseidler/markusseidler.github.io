---
title: "Let the machine learn the sine function"
date: 2018-06-23
tags: [machine learning, neural network]
header:
    image: "images/SINE/sine_screen.jpg"
excerpt: "Can an ANN learn the function of sine waves?"
---

## What is the sine function?

Sine is one the major functions in trigonometry. The sine function of a right triangle
is the ratio of the angle-opposite side compared to the hypotenuse. Sine waves can also
be observed in nature and two of the most common observations with sine patterns
are sound and light waves.

<img src="{{ site.url }}{{ site.baseurl }}/images/SINE/sine.jpg"
alt="Sine Formula">

If the triangle follows the circumference, then the ratio changes following
the pattern of a sine wave. For me, it just describes the perfectness and
harmony of a circle. Simple. Beautiful.

We can very quickly code sine waves with numpy and plot them with matplotlib.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10.5,10.5,0.1)
y = np.sin(x)

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10,6))
ax.plot(x, y, color="red")
ax.set_title("Sine Function of X")

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

plt.savefig("sine_wave.jpg")
```

This is the plot of the sine function. It is clearly a non-linear function and
therefore, a great playground for a neural network.

<img src="{{ site.url }}{{ site.baseurl }}/images/SINE/sine_wave.jpg"
alt="Sine Wave">

## Learn it. Fast and Furious.

I used my ThreeLayerNet saved in the script "ann.py" and improve it further.
I want to bring it to the next level by introducing two important concepts.
Bias and Stochastic Gradient Descent (SGD). With Bias I am adding to my ANN
an input vector of the number 1. By adding a bias, the ANN can tweak the
approximation of the sine function not only by finding the right slope but it
can also shift it from the left to right and shift away from a 0 center. The
concept is similiar to the concept of linear regression where a function of
y = aX + b is superior to a solution y = aX.

Stochastic Gradient Descent is an enhancement of the classic Gradient Descent.
The Gradient Descent I used in the previous posts calculates the Gradient
Descent across all input samples. This is inferior for two reasons. First, it is
what it is called "computionally wasteful" as the result of enough smaller sample calculations
is a good approximation of the total universe. Secondly, the risk of Gradient
Descent iteration is that the network optimization get stuck in local minimum.
The hiker heading downhills thinks he or she is already at the lowest point of
the valley as all the next steps in any directions are uphill. However, it could
be that he or she is just in front of a smaller hill that the hiker needs to cross
for the final descent. In order to avoid the trap of local minima, the technique of
Stochastic Gradient Descent does not calculate the gradient vector for the full dataset.
It takes samples, so-called batches, to calculate the weights updates. The size of a batch
is a hyperparameter and the optimal number can be tested with Grid Searches. However,
it is standard practice to use a size number to the power of 2 such as 4, 8, 16, 32.










