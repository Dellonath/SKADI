<p align="center">
  <img width = 300 src="https://user-images.githubusercontent.com/56659549/106782675-342d2c00-6629-11eb-96a7-ca0096278de9.png">
</p>

<br>

<h3 align="center">
  <b>SKADI</b> is a Deep Learning framework developed to be simple, intuitive and friendly with beginners in deep learning. 
</h3>

<br>

<p align="center">
  <img src="https://img.shields.io/badge/progress-60%25-important.svg?style=for-the-badge">
  <img src="https://img.shields.io/badge/version-1.0-orange.svg?color=greeb&style=for-the-badge">
  <a href="https://github.com/Dellonath/SKADI/blob/main/LICENSE">
    <img src="https://img.shields.io/apm/l/vim-mode?color=greeb&style=for-the-badge">
  </a>
  <a href="https://github.com/Dellonath/SKADI/stargazers">
    <img src="https://img.shields.io/github/stars/Dellonath/SKADI?color=greeb&style=for-the-badge">
  </a> 
  <a href="https://github.com/Dellonath/SKADI/network/members">
    <img src="https://img.shields.io/github/forks/Dellonath/SKADI?color=greeb&style=for-the-badge">
  </a>
</p>

<br>
<h3 align="center">Table of Contents</h3>

<p align="center">
 <a href="#Objective">
   <img src="https://img.shields.io/badge/Objective-grey?style=for-the-badge">
 </a>
 <a href="#Status">
   <img src="https://img.shields.io/badge/Status-grey?style=for-the-badge">
 </a>
 <a href="#Technologies">
   <img src="https://img.shields.io/badge/Technologies-grey?style=for-the-badge">
 </a>
 <a href="#Features">
   <img src="https://img.shields.io/badge/Features-grey?style=for-the-badge">
 </a>
 <a href="#To Use">
   <img src="https://img.shields.io/badge/To Use-grey?style=for-the-badge">
 </a>
 <a href="#Demonstration">
   <img src="https://img.shields.io/badge/Demonstration-grey?style=for-the-badge">
 </a>
 <a href="#Author">
   <img src="https://img.shields.io/badge/Author-grey?style=for-the-badge">
 </a> 
</p>

___________________________

<a name="Objective"/>
<h3 align="center">Objective</h3>
<p align="left">
  The objective is that users can use the framework and consult to better understand the functioning of a neural network. Maybe jobs with deep learning models     can be difficult to apply and understand, so this project was created to help the user to understand how a neural network works to create their own deep         learning model.
</p>

___________________________

<a name="Status"/>
<h3 align="center">Status</h3>
<p align="left">
  The framework is <b>ready to use</b>, you can apply it in your projects. The features to be implemented are to improve and increase the capacity of the neural   network, making it capable of solving the most varied types of problems in different ways, but them dont are impediments.
</p>

___________________________

<a name="Technologies"/>
<h3 align="center">Technologies</h3>
<p align="center">
  <a href="https://www.python.org/">
   <img src="https://img.shields.io/static/v1?label=Python&message=Language&color=107C10&style=for-the-badge&logo=Python"/>
  </a>
  <a href="https://numpy.org/">
   <img src="https://img.shields.io/static/v1?label=Numpy&message=Library&color=107C10&style=for-the-badge&logo=Numpy"/>
  </a>
</p>
<p align="left">
Python was choiced because it is a powerful language and the most used in Artificial Intelligence field. Is a easy tool to use, dynamic and user friendly.</p>
<p align="left">
Numpy it's an important and powerful math library that facilitates the implementation with your vectorization, statistics methods and another powerful tools.
</p>

___________________________

<a name="Features"/>
<h3 align="center">Features</h3>

- [x] learning rate
  - [x] time-based learning rate decay
  - [x] exponential learning rate decay
  - [x] staircase learning rate decay
- [x] cost functions
  - [x] mean squared error
  - [x] mean absolute error
  - [ ] root mean squared error
  - [ ] root mean squared logarithmic error
  - [x] binary cross-entropy
  - [x] sigmoid cross-entropy
  - [x] negative log-likelihood
  - [x] softmax negative log-likelihood
  - [ ] hinge
  - [ ] huber
  - [ ] kullback-leibler
- [x] momentum
- [x] layers
  - [x] common
  - [x] dropout
- [x] regularization
  - [x] l1
  - [x] l2
  - [ ] batch normalization
- [x] mini-batch gradient descent
- [x] early stopping
- [x] save/load
- [x] freezing
- [x] weights initialization 
  - [x] glorot normal (xavier normal)
  - [ ] glorot uniform
  - [ ] random normal
  - [ ] random uniform
  - [ ] zeros
  - [ ] ones
- [ ] gradient checking


___________________________
<a name="To Use"/>
<h3 align="center">To Use</h3>

<p align="left">
  To use this framework in your work, firstly you must install the python language, the git and numpy library in your computer, do the both last with this command lines
  
    apt install git
    pip install numpy
    
  Now go to your project folder and there execute the follow command line
     
    git clone https://github.com/Dellonath/SKADI.git
   
  Copy the SKADI/skadi folder beside your file.py where you will import the SKADI, like below
  

</p>

<img src="https://user-images.githubusercontent.com/56659549/106929595-512e3180-66f3-11eb-96e6-deef5d070770.png">


<p align="left">
  <br>
  Now, in your python file, you need to import the framework, you can do it with this lines:

  ```Python
  from skadi.model import Model            
  from skadi.layers import Layer, Dropout
  ```
  It is everything, now you are ready to work!!
</p>

<p align="left">
  <br>
  For certain types of problems, whether regression or classification, it is necessary to adopt different approaches and structures for the neural network to perform well. For this, the table below defines some structures for the network to be able to return good results.
</p>

|Problem|Cost Function|Last Layer Activation|Number of Neurons in last layer|
|----|----|----|----|
|Linear Regression|MSE, MAE|Linear|Number of values to predict|
|Binary Classification 1|Binary Cross-Entropy|Sigmoid|1|
|Binary Classification 2|Softmax + Neg. Log-Likelihood|Linear|2|
|Multiclass Classification|Softmax + Neg. Log-Likelihood|Linear|Number of categories|


___________________________

<a name="Demonstration"/>
<h3 align="center">Demonstration</h3>

<p align="left">
  Below we have some examples of use and performance of the framework, resulting in a comparative graph between loss of training and loss of validation on the left and accuracy on the right.
</p>

<h4 align="left">Multiclass Classification</h4>

```Python
from skadi.model import Model  # to import the framework base Model
from skadi.layers import Layer, Dropout  # to import the framework Layers
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_classification  

# creating a classification problem
X_class, Y_class = make_classification(n_samples=500, n_features=4, n_classes=4, n_clusters_per_class=1)
X_class = StandardScaler().fit_transform(X_class)
Y_class = OneHotEncoder(sparse=False).fit_transform(Y_class.reshape(-1, 1))

# creating the base of the model
nn = Model(
    cost_function = 'softmax_neg_log_likelihood', 
    learning_rate = 0.01, 
    momentum = .9, 
    batch = ('sequential', 50),
    lr_decay = ('staircase', .99, 5)
)

# adding different type layers with different hyperparameters
nn.add(Layer(4)) # input layer
nn.add(Layer(16, activation_function = 'relu', regularization = ('l1', 0.01)))
nn.add(Dropout(units = 16, activation_function = 'tanh', p = 0.3, regularization = ('l2', 0.1)))
nn.add(Layer(4, activation_function = 'linear')) # out layer

# fitting the model
nn.fit(X_class, 
       Y_class, 
       epochs = 3000, 
       verbose = 500, 
       validation_rate = 0.2 
)
Out:
  epoch:     0/3000  loss: 1.3509106021  val_acc: 0.0
  epoch:   500/3000  loss: 0.4634072936  val_acc: 0.87
  epoch:  1000/3000  loss: 0.4100287291  val_acc: 0.88
  epoch:  1500/3000  loss: 0.4496435452  val_acc: 0.87
  epoch:  2000/3000  loss: 0.4166560359  val_acc: 0.87
  epoch:  2500/3000  loss: 0.5221857908  val_acc: 0.87
  epoch:  3000/3000  loss: 0.4676211429  val_acc: 0.87
```

<img src = "https://user-images.githubusercontent.com/56659549/106908152-ce4eac00-66dd-11eb-981f-2c5b76e3fc87.png">

<h4 align="left">Binary Classification</h4>

```Python
from skadi.model import Model  # to import the framework base Model
from skadi.layers import Layer, Dropout  # to import the framework Layers
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification  

X_class, Y_class = make_classification(n_samples=500, n_features=4, n_classes=2, n_clusters_per_class=1)
X_class = StandardScaler().fit_transform(X_class)
Y_class = Y_class.reshape(-1, 1)

nn = Model(
  cost_function = 'binary_cross_entropy', 
  learning_rate = 1e-2, 
  momentum = .99, 
  lr_decay = ('exponential', 0.9, None)
)

nn.add(Layer(4))
nn.add(Layer(10, 'tanh'))
nn.add(Layer(1, 'sigmoid'))
nn.fit(
  X_class,
  Y_class, 
  epochs = 1000, 
  verbose = 500, 
  validation_rate = 0.3, 
  loss_stop = (500, 1e-4)
)

Out:
  epoch:     0/1000  loss: 0.7545087989  val_acc: 0.427
  epoch:   500/1000  loss: 0.0923987324  val_acc: 0.940
  loss_stop activated:   914/1000  loss: 0.0677572368  val_acc: 0.973
```
<img src = "https://user-images.githubusercontent.com/56659549/106910637-30101580-66e0-11eb-83ad-3df5822f9bd4.png">

```Python
from skadi.model import Model  # to import the framework base Model
from skadi.layers import Layer, Dropout  # to import the framework Layers
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles  

X_class, Y_class = make_circles(n_samples = 500, noise = .1, factor = .4)
Y_class = Y_class.reshape(-1, 1)
X_class = StandardScaler().fit_transform(X_class)

nn = Model('sigmoid_cross_entropy', learning_rate=0.5)
nn.add(Layer(2))
nn.add(Layer(10, 'tanh'))
nn.add(Dropout(10, 'tanh', .1))
nn.add(Layer(1, 'linear'))
nn.fit(X_class, Y_class, epochs = 3000, verbose = 500, validation_rate = 0.2)

Out:
  epoch:     0/3000  loss: 0.7131146386  val_acc: 0.700
  epoch:   500/3000  loss: 0.0123973415  val_acc: 1.000
  epoch:  1000/3000  loss: 0.0031569242  val_acc: 1.000
  epoch:  1500/3000  loss: 0.0026429120  val_acc: 1.000
  epoch:  2000/3000  loss: 0.0019530552  val_acc: 1.000
  epoch:  2500/3000  loss: 0.0011332451  val_acc: 1.000
  epoch:  3000/3000  loss: 0.0010627856  val_acc: 1.000
```
<img width = 550 src = "https://user-images.githubusercontent.com/56659549/106915703-17562e80-66e5-11eb-9d5d-d07831f98c37.png">

```Python
from skadi.model import Model  # to import the framework base Model
from skadi.layers import Layer, Dropout  # to import the framework Layers
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_moons

X_class, Y_class = make_moons(n_samples = 100, noise = .1)
Y_class = OneHotEncoder(sparse=False).fit_transform(Y_class.reshape(-1, 1))
X_class = StandardScaler().fit_transform(X_class)

nn = Model('softmax_neg_log_likelihood', learning_rate=1e-1)
nn.add(Layer(2))
nn.add(Layer(12, 'tanh', regularization = ('l2', 0.001)))
nn.add(Layer(12, 'tanh', regularization = ('l2', 0.001)))
nn.add(Layer(2, 'linear'))
nn.fit(X_class, Y_class, epochs = 5000, verbose = 1000, validation_rate = 0.2)

Out:
  epoch:     0/5000  loss: 1.1460166199  val_acc: 0.350
  epoch:  1000/5000  loss: 0.0084950357  val_acc: 1.000
  epoch:  2000/5000  loss: 0.0026057702  val_acc: 1.000
  epoch:  3000/5000  loss: 0.0015145193  val_acc: 1.000
  epoch:  4000/5000  loss: 0.0010876888  val_acc: 1.000
  epoch:  5000/5000  loss: 0.0008667776  val_acc: 1.000
```
<img width = 550 src = "https://user-images.githubusercontent.com/56659549/106918512-04912900-66e8-11eb-8372-e8c8fc7cc807.png">


<h4 align="left">Linear Regression</h4>

```Python
from skadi.model import Model  # to import the framework base Model
from skadi.layers import Layer, Dropout  # to import the framework Layers
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import make_regression  

X_linear, Y_linear = make_regression(n_samples = 100, n_features=1, n_targets = 1, shuffle = True, noise = 15)
Y_linear = Y_linear.reshape(len(X_linear), 1)
X_linear = StandardScaler().fit_transform(X_linear)
Y_linear = StandardScaler().fit_transform(Y_linear)

nn = Model('mae', learning_rate=1e-1)
nn.add(Layer(1))
nn.add(Layer(10))
nn.add(Layer(1))
nn.fit(X_linear, Y_linear, epochs = 300, verbose = 100, validation_rate = 0.3)

Out:
  epoch:     0/300  loss: 0.8991597990  val_acc: 24.668
  epoch:   100/300  loss: 0.2203735767  val_acc: 7.783
  epoch:   200/300  loss: 0.2200636261  val_acc: 7.789
  epoch:   300/300  loss: 0.2179173881  val_acc: 7.751
```
<img src = "https://user-images.githubusercontent.com/56659549/106913824-2a67ff00-66e3-11eb-8af7-48006c16fdb1.png">

```Python
from skadi.model import Model  # to import the framework base Model
from skadi.layers import Layer, Dropout  # to import the framework Layers
from sklearn.preprocessing import StandardScaler 

X_linear, Y_linear = make_exp(n_samples=100, x_min = 0, x_max = 5, noise = 10)
X_linear = StandardScaler().fit_transform(X_linear)
Y_linear = StandardScaler().fit_transform(Y_linear)

nn = Model('mae', learning_rate=1e-2)
nn.add(Layer(1))
nn.add(Layer(10, 'tanh'))
nn.add(Layer(10, 'tanh'))
nn.add(Layer(1, 'linear'))
nn.fit(X_linear, Y_linear, epochs = 10000, verbose = 2500, validation_rate = .05)

Out:
  epoch:     0/10000  loss: 1.0819791908  val_acc: 5.167
  epoch:  2500/10000  loss: 0.1388328752  val_acc: 1.023
  epoch:  5000/10000  loss: 0.1337075506  val_acc: 0.948
  epoch:  7500/10000  loss: 0.1323456180  val_acc: 0.916
  epoch: 10000/10000  loss: 0.1311388838  val_acc: 0.831
```
<img src = "https://user-images.githubusercontent.com/56659549/106914835-2a1c3380-66e4-11eb-90be-867a9bde28c1.png">


```Python
from skadi.model import Model  # to import the framework base Model
from skadi.layers import Layer, Dropout  # to import the framework Layers
from sklearn.preprocessing import StandardScaler 

X_linear, Y_linear = make_cubic(n_samples=100, x_min=-4, x_max=4, a=1, b=0, c=-10, d=0, noise=3)
X_linear = StandardScaler().fit_transform(X_linear)
Y_linear = StandardScaler().fit_transform(Y_linear)

nn = Model('mae', learning_rate=1e-1)
nn.add(Layer(1))
nn.add(Layer(16, 'relu'))
nn.add(Layer(8, 'tanh'))
nn.add(Layer(1, 'linear'))
nn.fit(X_linear, Y_linear, epochs = 3000, verbose = 600, validation_rate = .02, loss_stop=(100, 1e-3))

Out:
  epoch:     0/3000  loss: 1.3110190012  val_acc: 1.212
  epoch:   600/3000  loss: 0.2925245185  val_acc: 0.383
  epoch:  1200/3000  loss: 0.2856206497  val_acc: 0.312
  epoch:  1800/3000  loss: 0.2997886400  val_acc: 0.318
  epoch:  2400/3000  loss: 0.2594068682  val_acc: 0.325
  epoch:  3000/3000  loss: 0.2519903201  val_acc: 0.303
```
<img width = 350 src = "https://user-images.githubusercontent.com/56659549/106931690-9c494400-66f5-11eb-8d60-b08f83b91566.png">

___________________________

<a name="Author"/>

<h3 align="center">Author</h3>

<p align="center">
  <img width = 150 src="https://user-images.githubusercontent.com/56659549/106941005-03b8c100-6701-11eb-9d77-0a4c296bd615.png">
</p>

<p>
  I am a Computer Science student at the Universidade Federal de Itajubá, with a focus on statistics, data science, data analysis, machine learning, deep learning among others. I made this project as a personal challenge with the intention of applying several concepts that I learned in the excellent course <a href = "https://www.udemy.com/course/redes-neurais/">Manual Prático do Deep Learning - Redes Neurais Profundas</a> offered by <a href="https://www.linkedin.com/in/arnaldo-gualberto/">Arnaldo Gualberto</a>, in addition to the objective of incorporating this work into my portfolio. 
</p>
<p>
<space>After the implementation of this project, I had a significant improvement in the theoretical and technical part of Deep Learning, as in the functioning of a neural network, the feedforward and backpropagation algorithms, partial derivatives and how they are applied in the latter, which are activation functions and cost and how to implement them, what each hyperparameter means and how they affect the final result of the network, how to create an adequate structure for each type of problem and many other very important concepts for the development of a neural network. Another important learning was the fact of better understanding how I should prepare the data for a neural network, from its cleaning to the application of techniques such as Normalization, Standardization, One-Hot-Encoder and others.
</p>
  

  
<h4 align="center">Contact</h4>
<p align="center">
  <a href="mailto:dellonath@gmail.com?
    subject=MessageTitle&amp;
    body=Message Content">
    <img src="https://img.shields.io/static/v1?label=Gmail&message=dellonath@gmail.com&color=EA4335&style=for-the-badge&logo=Gmail">
  </a>
  <a href="https://www.linkedin.com/in/douglas-oliveira-5b36201b2/">
    <img src="https://img.shields.io/static/v1?label=LinkedIn&message=Douglas%20Oliveira&color=0077B5&style=for-the-badge&logo=LinkedIn">
  </a>
  <a href="https://www.facebook.com/dellonath/">
    <img src="https://img.shields.io/static/v1?label=Facebook&message=Douglas%20Oliveira&color=1877F2&style=for-the-badge&logo=Facebook">
  </a>
</p>
