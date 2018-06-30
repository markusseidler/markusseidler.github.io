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

Let me you show the code of the modified "ann.py":

```python
import numpy as np
import pandas as pd
import time

#Creating the Neural Network as Python Class
class ThreeLayerNet(object):

#Defining the relevant variables and external inputs.
    def __init__(self, X, y, neurons_l2=4, neurons_l3=16, weight_init="uniform",
                 epoch=1000, bias=True, batchsize=4):
        self.X = X
        self.y = y
        self.neurons_l2 = neurons_l2
        self.neurons_l3 = neurons_l3
        self.weight_init = weight_init
        self.bias = bias
        self.epoch = epoch
        self.batchsize = batchsize
        self.variable_list = ["self.X", "self.y",
                              "self.W1", "self.z2", "self.a2",
                              "self.W2", "self.z3", "self.a3",
                              "self.W3", "self.z4", "self.y_est",
                              "self.error", "self.cost", "self.cost_deriv",
                              "self.delta4", "self.dCdW3",
                              "self.delta3", "self.dCdW2",
                              "self.delta2", "self.dCdW1",
                              ]

        '''
        This is new. This creates a vector of "ones" in the length of X. After,
        that it combines this bias vector with the input vector X.
        '''

        if self.bias == True:
            bias_vector = np.ones(self.X.shape[0]).reshape((-1,1))
            self.X = np.hstack((bias_vector, self.X))

        else:
            self.X = self.X

        #Three ways to initiate the network weights.
        #Choice 1: Random with normal distributuon.
        if self.weight_init == "normal":
            self.W1 = np.random.randn(self.X.shape[1],self.neurons_l2)
            self.W2 = np.random.randn(self.neurons_l2, self.neurons_l3)
            self.W3 = np.random.randn(self.neurons_l3, self.y.shape[1])

        #Choice 2: Random with uniform distribution.
        elif self.weight_init == "uniform":
            self.W1 = np.random.uniform(0,1,self.X.shape[1]*self.neurons_l2).\
                reshape((self.X.shape[1],self.neurons_l2))
            self.W2 = np.random.uniform(0,1,self.neurons_l2*self.neurons_l3).\
                reshape((self.neurons_l2,self.neurons_l3))
            self.W3 = np.random.uniform(0,1,self.neurons_l3*self.y.shape[1]).\
                reshape((self.neurons_l3,self.y.shape[1]))

        #Choice 3: All weights are initially set at 1.
        else:
            self.W1 = np.ones((self.X.shape[1],self.neurons_l2))
            self.W2 = np.ones((self.neurons_l2,self.neurons_l3))
            self.W3 = np.ones((self.neurons_l3,self.y.shape[1]))

    '''
    This is also new. This is the class method that splits the input data in
    batches.
    '''
    def batch_split (self):
        for i in np.arange(0, self.X.shape[0], self.batchsize):
            yield (self.X[i:i+self.batchsize], self.y[i:i+self.batchsize])

    #Sigmoid function
    def sigmoid (self, z):
        return (1/(1+np.exp(-z)))

    #Derivative of Sigmoid function.
    def sigmoid_deriv (self, z):
        return (np.exp(-z)/(np.power((1+np.exp(-z)),2)))

    '''
    This function manages the so-called forward propagation. Calculating the output
    step by step.

    The dot product of the first layer (input layer X) and weights matrix W1 results
    in the second layer (z2). Then applying the activation function sigmoid to z2
    and receiving the first activation layer a2. The next step is connecting a2 with
    the second hidden layer by calculating the dot product of a2 and weight matrix W2.
    This gives z3 and passing it through the sigmoid function leads to a3. Dot product
    of a3 with W3 is z4 and applying the sigmoid function on the last layer gives us
    the estimated output y_est.
    '''

    def forward_propagation(self, X_batch):

        self.z2 = np.dot(X_batch,self.W1)
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2,self.W2)
        self.a3 = self.sigmoid(self.z3)

        self.z4 = np.dot(self.a3,self.W3)
        self.y_est = self.sigmoid(self.z4)

    '''
    This function calculates the error, the cost function, the difference between
    the estimated output y_est and the actual value y. It also calculates the
    derivative as a first input for backpropagation.
    '''

    def loss_calculation(self,y_batch):
        self.error = y_batch - self.y_est
        self.cost = 0.5*np.sum((np.power(self.error,2)))
        self.cost_deriv = self.y_est - y_batch

    '''
    This is the famous backpropagation. The major technique that allows an ANN
    to learn and improve its results. Basically, what it does is it takes the prediction
    error and tries to see which parts of the network, which weights are most important
    in creating an error.

    Mathematically, we take the partial derivatives layer by layer
    and end up with the calculated sensitivity in respect to the weight matrices.

    Delta is the sensitivity of the layer cost and dCdW puts this sensitivity in respect
    to a specific weight matrix. In other words, how much does the error (or cost C)
    change if we change the weights W.
    '''

    def back_propagation(self, X_batch):
        self.delta4 = np.multiply(self.cost_deriv,self.sigmoid_deriv(self.z4))
        self.dCdW3 = np.dot(self.a3.T,self.delta4)

        self.delta3 = np.multiply(np.dot(self.delta4,self.W3.T),self.sigmoid_deriv(self.z3))
        self.dCdW2 = np.dot(self.a2.T,self.delta3)

        self.delta2 = np.multiply(np.dot(self.delta3,self.W2.T),self.sigmoid_deriv(self.z2))
        self.dCdW1 = np.dot(X_batch.T,self.delta2)

    '''
    This function trains the network by applying for each epoch the forward propagation,
    the loss_calculation and the back_propagation. After that it updates the weight
    by the sensitivities and recalculates everything for the next round, the next epoch.
    '''

    #This time I also added a parameter that allows me tracking the result of each layer.
    def train_NN (self, record_layers=False):

        if record_layers == False:
            self.loss_per_training = []
            for i in np.arange(0,self.epoch+1,1):
                self.loss_per_epoch = []

                '''
                Here I apply the split of the input data by batches. Instead of
                calculating the gradient every time for the full set of input data,
                this time I calculate it only for a subgroup of data, for a batch.
                Batch by batch, the network learns and then averages the losses in
                a list.
                '''

                for X_batch, y_batch in self.batch_split():
                    self.forward_propagation(X_batch)
                    self.loss_calculation(y_batch)
                    self.back_propagation(X_batch)
                    self.loss_per_epoch.append(self.cost)
                    self.W1 -= self.dCdW1
                    self.W2 -= self.dCdW2
                    self.W3 -= self.dCdW3
                self.loss_per_training.append(np.average(self.loss_per_epoch))
                self.avg_loss_per_training = np.average(self.loss_per_training)

        elif record_layers == True:
            self.record_data = []
            for i in np.arange(0,self.epoch+1,1):
                self.loss_per_epoch = []
                for X_batch, y_batch in self.batch_split():
                    self.forward_propagation(X_batch)
                    self.loss_calculation(y_batch)
                    self.back_propagation(X_batch)
                    self.loss_per_epoch.append(self.cost)
                    self.W1 -= self.dCdW1
                    self.W2 -= self.dCdW2
                    self.W3 -= self.dCdW3
                print (np.average(self.loss_per_epoch))
                self.record_data.append([(self.X), (self.y), (self.W1),
                                        (self.z2), (self.a2), (self.W2),
                                        (self.z3), (self.a3), (self.W3),
                                        (self.z4), (self.y_est), (self.y),
                                        (self.error), (self.cost),
                                        (self.dCdW1), (self.dCdW2), (self.dCdW3),
                                        (self.delta2), (self.delta3), (self.delta4)
                                        ])

            test_results = pd.DataFrame(self.record_data,columns=
                                ["self.X", "self.y",
                                  "self.W1", "self.z2", "self.a2",
                                  "self.W2", "self.z3", "self.a3",
                                  "self.W3", "self.z4", "self.y_est", "self.y",
                                 "self.error", "self.cost", "self.dCdW1",
                                 "self.dCdW2", "self.dCdW3",
                                 "self.delta2", "self.delta3", "self.delta4"
                                 ])

            filename = ("excel/ThreeLayerNet test {}.xlsx".format(int(time.time())))
            writer = pd.ExcelWriter(filename, engine="xlsxwriter")
            test_results.to_excel(writer, sheet_name="data")
            writer.save()
            print ("{} saved.".format(filename))

    #Printing the shape of all matrices
    def print_shape(self):
        for var in self.variable_list:
            print ("\n {} \n {}".format(var, eval(var).shape))

    #Printing the values of all matrices
    def print_values (self):
        for var in self.variable_list:
            print ("\n {} \n {}".format(var, eval(var)))

```

I also added a parameter in the train_NN method that allowed me to turn on
the data recording of each layer. I did that to understand the dynamic within
the network. A multi-layer network approximating the sine function with sigmoid
activiation is at risk to produce saturated neurons. This means that the neurons
have reached such high levels that the "firing" of the neuron, the strength of
its activation is not significant any more. Consequently, the network cannot use
this neuron any more for further learning.

I used this time the same approach and split execution from the network file.
I also modified the "main" file. I did again a so-called "Grid Search", a reiteration
across various hyperparameters, to find the best architecture. I reiterated over
different combinations of neurons per layer.

```python
import ann3 as a
import timeit
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

start = timeit.default_timer()

#This is the input matrix X with the input features A and B
X_unscaled = np.linspace(0,20,240).reshape((-1,1))
y_unscaled = np.sin(X_unscaled).reshape((-1,1))

#I scale the input and output vector to get arrays from 0 to 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X_unscaled)
y = scaler.fit_transform(y_unscaled)

loss_recording = []

#Reinterate from 2 to 20 neurons per layer
for i in np.arange(2, 21, 2):
    for j in np.arange(2,21,2):
        ann = a.ThreeLayerNet(X,y, neurons_l2=i, neurons_l3=j, epoch=2000, batchsize=4)
        ann.train_NN(record_layers=False)
        loss_recording.append((i, j, ann.avg_loss_per_training))

loss_df = pd.DataFrame(loss_recording, columns=["Neurons in Layer 2",
                                                "Neurons in Layer 3",
                                                "Loss"
                                                ])

#Sorting the table and print the Top 15 and Bottom 15 results
loss_df.index.name = "Training"
loss_df = loss_df.sort_values("Loss")
print ("\nTop 15 Neuron Combinations:\n")
print (loss_df.head(15))
print ("\nBottom 15 Neuron Combinations:\n")
print (loss_df.tail(15))

#counting running time
end = timeit.default_timer()
time_elapsed = (end-start)
print ("\nRunning time of script in seconds: {}".format(round(time_elapsed,5)))

```

Unlike in the previous post where I approximated with my neural network
the XOR problem, this time the original input vector is not within 0 and 1.
It is important to scale and normalize input data and to preserve ranges
between 0 and 1. Computers think in 0 and 1 and its just benefical to translate
our world in their language. It reduces the risk of local minima, saturated neurons,
and increases the speed of processing.

I reiterated over 400 architecture combinations. Each of them calculated 2000 epochs.
The input vector had a length of 240 and the batch size was 4. This means that
every epoch adjusted the weights 60 times. In sum, we asked our friend, the computer,
to calculate 48 million times forward and 48 million backwards across the network.
Quite some work to do. I hope our friend was not too angry about it...

These are the results of the Grid Search:


<img src="{{ site.url }}{{ site.baseurl }}/images/SINE/top15.PNG"
alt="Top 15 Results of Grid Search">

<img src="{{ site.url }}{{ site.baseurl }}/images/SINE/bottom15.PNG"
alt="Bottom 15 Results of Grid Search">

It looks like a combination of six neurons in the first hidden layer (layer 2)
and 14 or 16 neurons in the second hidden layer (layer 3) produced the best results.
Interestingly, a network with fewer neurons in layer 3 delivered inferior results.

I decided to go ahead with 6 and 14 as combination for my further analysis.

### Just numbers. Can we again see some graphs?






