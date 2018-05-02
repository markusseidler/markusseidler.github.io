---
title: "Image Recognition of Fruits with Support Vector Machine and
Histogram of Oriented Gradients"
date: 2018-05-05
tags: [machine learning, support vector machine, image recognition]
header:
    image: "images/IRCHOGSVM/apples.jpg"
excerpt: "Recognizing fruits with SVM and HOG"
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


