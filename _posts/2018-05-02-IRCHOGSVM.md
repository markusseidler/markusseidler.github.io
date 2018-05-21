---
title: "Image Classification of Fruits with Support Vector Machine and
Histogram of Oriented Gradients"
date: 2018-05-05
tags: [machine learning, support vector machine, image classification]
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

We can also display the array with HOG values in a KDE plot. If color changes
from bright to dark, the values are higher and more concentrated around 0.2.
On the other hand, changes around bright colors result in values closer to 0.

Here is the code creating the HOG transformation and the plotting the KDE plot.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

t1 = (cv2.imread("t1.jpg"))
t2 = (cv2.imread("t2.jpg"))
t3 = (cv2.imread("t3.jpg"))
t4 = (cv2.imread("t4.jpg"))
t5 = (cv2.imread("t5.jpg"))


#HOG parameters

#how big is the image?
winsize = (100, 100)

#moving cells over the image, subset of image
cellsize = (10, 10)

#block to normalize vectors, to handle illumination
blocksize = (10, 10)

#overlap with neighboring cells, usually 50% of blocksize
blockstride = (5, 5)

#number of bins of histogram, usually 9
nbins = 9

#Should the bins go from 0 to 180 degree (unsigned) or 0 to 360 (signed)? Usually unsigned
signedGradients = False

#no idea what it is but everyone says keep it default, there are not important?
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64


hog = cv2.HOGDescriptor(winsize, cellsize, blocksize, blockstride, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)


image_list = [t1, t2, t3, t4, t5]


#designing basic framework of chart plot
fig, axes = plt.subplots(nrows=len(image_list),ncols=2, figsize = (10,12))

#reiterating through each image
for i, fruit in zip(range(len(image_list)), image_list):
    fruit = cv2.resize(fruit, (100,100))
    fruit = cv2.cvtColor(fruit, cv2.COLOR_BGR2RGB)

    #if right, show image, if left, show KDE plot
    for j in range(2):
        if j == 0:
            axes[i][j].imshow(fruit)
        if j == 1:
            fruit = cv2.cvtColor(fruit, cv2.COLOR_RGB2GRAY)
            fruit_hog = hog.compute(fruit)
            x = fruit_hog.flatten()

            sns.kdeplot(np.arange(len(x)),x, cmap="BuPu", shade=True, ax=axes[i][j])
            axes[i][j].set_xlim(-50, len(fruit_hog)+50)
            axes[i][j].set_ylim(0, 0.1)

#labelling plots and axes
axes[0][0].set_title("Original fruit image")
axes[len(image_list)-1][0].set_xlabel("Pixel")
axes[0][1].set_title("KDE plot of Image HOG")
axes[len(image_list)-1][1].set_xlabel("HOG array")

fig.savefig("KDE_HOG1.jpg")
```

How does it look like? Here we go...

<img src="{{ site.url }}{{ site.baseurl }}/images/IRCHOGSVM/KDE_HOG1.jpg"
alt="Edges transformed">

This is a very colorful way to visualize it but perhaps not the most intuitive one.
Image gradients consist of vectors and their length and direction are indication for
the magnitude of changes. Maybe, we should also try to show it as a series of vectors.
Similiar to the previous KDE plot, I am building up a matplotlib figure with
multiple axis.

## Dancing Fruits

I decided to enlarge the training dataset. Originally, I worked with 28,736
images of 60 different fruits. But how can I strengthen the dataset?
I want to make it independent from the angle how a picture was taken.
I take the dataset and let each picture rotate around its axis.

I make use of a library from Adrian Rosebrock. The libary is called imutils
and includes a method called "rotate_bound". However, I slighlty modified it.
In the original version it creates a black background. I changed it that
it maintains a white background.

This is my script which rotates a fruit 360 degrees and plots the new images.

```python
import numpy as np
import imutils
import cv2
import os
import glob
from timeit import default_timer as timer

start = timer()

for path in glob.glob("../DataSet/fruits-360/sub-set/test"):
    print ("\n\n",path)
    # print (path) gives the full path of the folder
    path_split = path.split("/") [-1]
    # print (path_split)  gives the name of the folder
    for pic in glob.glob (os.path.join(path,"*.jpg")):
        #taks out the picture name
        pic_name = (os.path.split(pic)[-1][:-4])
        # print (pic) gives the full path of the pic
        image = cv2.imread(pic)
        for angle in np.arange(0, 360, 1):
            rotated = imutils.rotate_bound(image, angle)
            cv2.imshow("Dancing Fruits", rotated)
            print ("Picture Name: ", pic_name, "Rotation Angle:  ", str(angle))
            cv2.waitKey(5)
            # new_file = pic_name+str(angle)+".jpg"
            # file_name =  (os.path.join(path, new_file))
            # cv2.imwrite(file_name, rotated)
cv2.destroyAllWindows()
end = timer()
duration = end - start
print ("\n\tRunning time of script in seconds: \n\t", round(duration,5))
```

Have you ever seen fruits dancing? Each picture creates 360 new pictures.


<video width="630" height="270" controls="controls">
  <source src="/images/IRCHOGSVM/dancing_fruits.mp4" type="video/mp4">
</video>


When I apply it to the training dataset I do not choose an incremental rotation
change of one degree but of ten degrees. This allows faster processing and
provides still a significantly enlarged dataset. In total, I increase my
training dataset from 28,736 pictures to over 1 million pictures. A dataset
with over 10 billion of numbers.



