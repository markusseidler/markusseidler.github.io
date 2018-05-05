---
title: "Image Classification of Fruits with Support Vector Machine and
Histogram of Oriented Gradients"
date: 2018-05-05
tags: [machine learning, support vector machine, image recognition]
header:
    image: "images/IRCHOGSVM/still.jpg"
excerpt: "Classifying fruits with SVM and HOG"
---


## What is next?

I have trained and tested a Random Forest Classifier in the previous
post. The machine learning algorithm could achieve an accuracy ratio
of 88% on the validation data set. However, it fails to perform if the
sample image shows a different background or style than in the test and
validation data set.

This time I try to see if I get better results if we change the machine
learning algorithm to a Support Vector Machine (SVM) and if we pre-process
the images with a technique called "Histogram of Oriented Gradients" (HOG).


## What? Histo ...what?

Well, this needs a bit background. Histogram of Oriented Gradients are
based on "Image Gradient" calculations. For every pixel, you look at the
changes in color intensity if you go up or down 1 pixel along the x axis
and if you go up or down 1 pixel along the y axis. Combining those changes
gives for each pixel a vector where the angle (degree) is a function of the intensity
change between x and y. The so-to-speak length of this vector is the
magnitude of the combined changes of x and y. In math language, an image
gradient is the vector of its partial derivatives. In other words, how
does the pixel changes if x and y change.

Histogram of Oriented Gradients is a methodology where for each subset of
the image, a small window of the total picture, a histogram is calculated.
This histogram has bins with the magnitude of the various image gradients.
The allocation of a gradient magnitude ("length of the vector") to a certain
bin is decided by its degree (angle).

Sounds complicated? It is not easy to understand and needs a bit reading
and research to imagine the methodolgy. More important is to understand
that this approach helps to identify edges in an image. It shows the intensity
of changes from one pixel to another pixel. Edge detection is important when
image objects like a car or a person in a picture needs to be identified.
Or fruits?

How does the result of image gradients look like? Let me demonstrate this
with a few lines of code. First, I create an object class called ImageArray
for this purpose. Then, I add methods to transform the picture. Finally, I
open an instance of this class.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as sig

class ImageArray:

    def __init__(self, image):
        self.image = image
        self.array = np.array(self.image, dtype=np.uint8)
        self.blue, self.green, self.red = cv2.split(self.image)

    def gray(self):
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return np.array(self.image_gray)

    def image_gradient(self):
        x_kernel = np.array([[-1], [0], [1]])
        y_kernel = np.array ([[-1, 0, 1]])
        self.transform_x = sig.convolve2d(self.gray(), x_kernel, mode="valid")
        self.transform_y = sig.convolve2d(self.gray(), y_kernel, mode="valid")
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,6))
        axes[1].imshow((self.transform_x+255)/2, cmap="gray")
        axes[1].set_xlabel("X-Axis transformed")
        axes[2].imshow((self.transform_y+255)/2, cmap="gray")
        axes[2].set_xlabel("Y-Axis transformed")
        axes[0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0].set_xlabel("Original image")
        fig.savefig("transform.jpg")


img = cv2.imread("../dataset/fruits-360/Validation/Banana/108_100.jpg")
banana_yellow = ImageArray(img)

banana_yellow.image_gradient()
```

The result are images with edges that are highlighted. A great approach
to detect objects.

<img src="{{ site.url }}{{ site.baseurl }}/images/IRCHOGSVM/transform.jpg"
alt="Edges transformed">




