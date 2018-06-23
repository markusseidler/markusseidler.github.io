---
title: "Let the machine learn the sine function"
date: 2018-06-23
tags: [machine learning, neural network]
header:
    image: "images/SINE/sine_screen.jpg"
excerpt: "Can an ANN learn the function of sine waves?"
---

### What is the sine function?

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

### Learn it. Fast and Furious.

I use my ThreeLayerNet saved in the script "ann.py" and modify only the
execution script "main.py" for a first trial.







