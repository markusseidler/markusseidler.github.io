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
t6 = (cv2.imread("t6.jpg"))
t7 = (cv2.imread("t7.jpg"))
t8 = (cv2.imread("t8.jpg"))

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


image_list = [t1, t2, t3, t4, t5, t6, t7, t8]


#designing basic framework of chart plot
fig, axes = plt.subplots(nrows=len(image_list),ncols=2, figsize = (10,20))

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

fig.savefig("KDE_HOG.jpg")
```

How does it look like? Here we go...

<img src="{{ site.url }}{{ site.baseurl }}/images/IRCHOGSVM/KDE_HOG.jpg"
alt="KDE HOG">

This is a very colorful way to visualize it but perhaps not the most intuitive one.
Furthermore, looking at the various banana images, the background is a
significant attribute of the picture. Ideally, we can find something that
is less sensitive to the background color.

Bascially, image gradients consist of vectors and their length and direction
are indication for the magnitude of changes. Maybe, we should also try to
show it as a series of vectors. Similiar to the previous KDE plot, I am
building up a matplotlib figure with multiple axis.

```python
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import exposure

t1 = (cv2.imread("t1.jpg"))
t2 = (cv2.imread("t2.jpg"))
t3 = (cv2.imread("t3.jpg"))
t4 = (cv2.imread("t4.jpg"))
t5 = (cv2.imread("t5.jpg"))
t6 = (cv2.imread("t6.jpg"))
t7 = (cv2.imread("t7.jpg"))
t8 = (cv2.imread("t8.jpg"))


image_list = [t1, t2, t3, t4, t5, t6, t7, t8]

#designing basic framework of chart plot
fig, axes = plt.subplots(nrows=len(image_list),ncols=2, figsize = (10,20))

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
            fd, hog_image = hog (fruit,
                    orientations=9,
                    pixels_per_cell=(8,8),
                    cells_per_block=(3,3),
                    visualise=True,
                    block_norm="L2-Hys")

            #increasing the intensity values of the image
            hog_image_resc = exposure.rescale_intensity(hog_image, in_range=(0,3))

            axes[i][j].imshow(hog_image_resc, cmap=plt.cm.gray)

#labelling plots and axes
axes[0][0].set_title("Original fruit image")
axes[len(image_list)-1][0].set_xlabel("Pixel")
axes[0][1].set_title("Visualization of HOG")
axes[len(image_list)-1][1].set_xlabel("HOG array")

fig.savefig("Visualize_HOG.jpg")
```

This time I also use a library called skimage. With the help of this library,
I can calculate the HOG descriptor and also visualize it through the plot of
matplotlib.


<img src="{{ site.url }}{{ site.baseurl }}/images/IRCHOGSVM/Visualize_HOG.jpg"
alt="Visualize HOG">

We can clearly see the edges of the fruits. Furthermore, unlike in the
previous KDE plot, the background seems to be less significant. Important
is the change in color intensity from one pixel to another.




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
change of one degree but of 45 degrees. This allows faster processing and
provides still a significantly enlarged dataset. In total, I increase my
training dataset from 28,736 pictures to almost 260,000 pictures. A dataset
with over 2.5 billion of numbers of gray intensity.

## Finding C

One of the important hyperparameters when using Support Vector Machines
is the so-called "Penalty parameter of the error term" C. It manages
the tolerance of SVM in handling misclassification errors. Training this
dataset of 260,000 images with a SVM is computationally very expensive and
could take a few hours to one day of processing. Therefore, I will use a
smaller dataset first to find the optimal hyperparameter C before I let
the machine learning algorithm train the full dataset.

I chose a subset of my training set. I selected seven fruits out of those
60 fruits. Then I changed it to gray-scale and let those pictures
rotate in steps of 45 degrees. In total, I used a dataset of 30,762 images.
Subsequently, I also reduced the validation set to those seven fruits and
validated the training algorithm on a set of 1,143 images.

This is the full code including the HOG descriptor. This time I also saved
my models with joblib and the dictionary of fruits with pickle.

```python
import os
import glob
import cv2
import numpy as np
from timeit import default_timer as timer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pickle
import pandas as pd

training_folder = ("../DataSet/fruits-360/small_train/*")
validation_folder = ("../DataSet/fruits-360/small_val/*")

start = timer()

#creating empty lists, dictionaries and variables
X_fruit_arrays = []
y_fruit_ID = []

name_dictionary = {}

X_fruit_arrays_val = []
y_fruit_ID_val = []
name_dictionary_val = {}

c_range = [1,3,5,8,10,15,20,25,30,40,50,100]
accuracy = []
duration = []

count = 0
count2 = 0
count3 = 0
count4 = 0

#HOG parameters

#how big is the image?
winsize = (100, 100)

#moving cells over the image, subset of image
cellsize = (20, 20)

#block to normalize vectors, to handle illumination, usually 2 x cell size
blocksize = (20, 20)

#overlap with neighboring cells. Usually 50% of blocksize
blockstride = (10, 10)

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


for path in glob.glob(training_folder):
    path_split = path.split("/") [-1]
    name_dictionary[count] = path_split.split("\\")[-1]
    for pic in glob.glob (os.path.join(path,"*.jpg")):

        image = cv2.imread(pic,0)
        image = cv2.resize(image, (100,100))
        print ("Pic Name ", pic, " Count ", count3)
        count3 +=1
        hog_descriptor = hog.compute (image)
        X_fruit_arrays.append(hog_descriptor)
        y_fruit_ID.append(count)

    count += 1

X_fruit_arrays = np.array(X_fruit_arrays)
y_fruit_ID = np.array(y_fruit_ID)

x = set(y_fruit_ID)

X_fruit_arrays = X_fruit_arrays.reshape(-1,900)


for path in glob.glob(validation_folder):
    path_split = path.split("/") [-1]
    name_dictionary_val[count2] = path_split.split("\\")[-1]
    for pic in glob.glob (os.path.join(path,"*.jpg")):
        image = cv2.imread(pic,0)
        image = cv2.resize(image, (100,100))
        print ("Pic Name ", pic, " Count ", count4)
        count4 += 1
        hog_descriptor = hog.compute (image)
        X_fruit_arrays_val.append(hog_descriptor)
        y_fruit_ID_val.append(count2)

    count2 += 1

X_fruit_arrays_val = np.array(X_fruit_arrays_val)
y_fruit_ID_val = np.array(y_fruit_ID_val)
X_fruit_arrays_val = X_fruit_arrays_val.reshape(-1,900)


c_range = [1,3,5,8,10,15,20,25,30,40,50,100]
accuracy = []
duration = []

for c in c_range:
    start2 = timer()
    svm = SVC(C=c)
    svm.fit(X_fruit_arrays, y_fruit_ID)

    test_predictions_val = svm.predict(X_fruit_arrays_val)
    precision = accuracy_score(y_fruit_ID_val,test_predictions_val)*100
    accuracy.append(round(precision,5))

    filename = ("C_{}_number_of_fruits_{}_number_of_pics_{}.sav".format(c, count, count3))
    joblib.dump(svm, filename)
    print ("File ", filename," saved." )
    end2=timer()
    dur = round (end2-start2,2)
    print (precision)
    print (dur)
    duration.append(dur)

acc_df = pd.DataFrame()
acc_df ["C"] = c_range
acc_df ["Accuracy"] = accuracy
acc_df ["Duration"] = duration

print (acc_df)
#
with open ("name_dictionary_small.pickle", "wb") as file:
    pickle.dump(name_dictionary, file)
    print ("Pickle saved.")

end = timer()
duration = end - start
print ("\n\tRunning time of script in seconds: \n\t", round(duration,5))
```

I reiterated the Support Vector Machine Classifier (SVC) through a list
of different penalty parameter C. The result shows that the accuracy ratio
increases up to a C of around 20.

<img src="{{ site.url }}{{ site.baseurl }}/images/IRCHOGSVM/SVM C variation.PNG"
alt="SVM C variation">

Based on these results, I decided to use C of 20 for the test with the full
data set.

## SVM in full action

This time I used the full data set of 60 fruits. Including the rotations of
images I did earlier, I had a training set of 258,625 images. I validated the
algorithm on the original validation set of 9,673 images.

<img src="{{ site.url }}{{ site.baseurl }}/images/IRCHOGSVM/SVM full accuracy.PNG"
alt="SVM full accuracy">

The accuracy ratio was 82.8%. It was a computionally expensive training for
the machine learning algorithm. It took 21,665 seconds. Six hours of CPU
calculations to process the Support Vector Machine.

<img src="{{ site.url }}{{ site.baseurl }}/images/IRCHOGSVM/SVM full running time.PNG"
alt="SVM full running time">

Honestly, I was a bit disappointed when I saw the result after one night of
computer processing. I thought the HOG/SVM combination will deliver better
results. Better results than the Random Forest on a dataset with out rotations.
In comparison, the Random Forest Classifier achieved an accuracy ratio of 88%
with 55 Estimators and the calculation took 77 seconds.

<img src="{{ site.url }}{{ site.baseurl }}/images/IRCHOGSVM/disappointed.jpg"
alt="Disappointed">

Certainly, I was not too happy with the result and I wanted to understand better
where the algorithm failed. Luckily, I saved the model as joblib and could
retrieve it any time. Let's look first at the classification report. Maybe,
we see better the weakness of the model when we look at the results fruit by fruit.

<img src="{{ site.url }}{{ site.baseurl }}/images/IRCHOGSVM/name_dictionary.PNG"
alt="Name Dict">

<img src="{{ site.url }}{{ site.baseurl }}/images/IRCHOGSVM/classification_report.PNG"
alt="Class Rep">

Looking at the first half of the classification report shows that the SVM
classifier performed very well in class 17 (Cherry), 23 (Grape White),
24 (Grape White 2), 25 (Grapefruit Pink), 26 (Grapefruit White), and
28 (Huckleberry). However, it failed miserably in classes such as 14 (Banana Red),
18 (Clementine), and 11 (Avocado).

Interestingly, the Random Forest Classifier in the previous post was strong
in classifying Clementine (F1: 91%) and Avocado (91%) but also failed in
identifying correctly "Red Bananas".



