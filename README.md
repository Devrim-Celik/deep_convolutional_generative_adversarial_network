# Deep Convolutional Generative Adverserial Network
This is an implementation of a Deep Convolutional [Generative Adverserial Network](https://en.wikipedia.org/wiki/Generative_adversarial_networks) trained using [Adam Optimizers](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) and [Tensorflow](https://en.wikipedia.org/wiki/TensorFlow) as our final task for our university course "Implementing Artificial Neural Networks with Tensorflow".

## Demonstration
![Dataset](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/preprocessedImgs_5x5.png)
![Training Progress](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase.gif)


## Folders and Sripts
- Folders
    + ```/Images-Gifs```: Images and Gifs for Git-Hub
    + ```/data```: For storing the training images and the pickle file for training
    + ```/docs```: Documentation
    + ```/function```: Functions used for preprocessing, visualisation ...
    + ```/save```: For saving pictures which were produces during training & storing weights & plots of loss ...

- Scripts
    + ```dc_gan.py```: main script
    + ```preprocessing_pickle.py```: Applies preprocessing to images stored in ```/data/unprocessed_images``` and saves a pickle containing them in ```/save/pickle```.
    + ```celeba_input.py```: Loads pickle and supplies ```dc_gan.py``` with batches.
    + ```auxiliary_functions.py```: Contains visualisation tools and argument handling
    + ```gifMaker.py```: creates Gifs

## Installation and Prerequisites
Cloning the Repository via git:
```
git clone https://github.com/D3vvy/iannwtf_DCGAN.git
```

Make sure, you have the following installed:
- Matplotlib: ```pip install matplotlib```
- Numpy: ```pip install numpy```
- PIL ```pip install PIL```
- OpenCV2 ```pip install opencv``` or ```conda install opencv```
- [ImageIO](https://pypi.python.org/pypi/imageio)

Dataset:
- In this Deep Convolutional Generative Adversarial Network used the cropped and alligned version of the CelebA Datset, available [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
- If you just want to to train using a pickle, you can download a preprocessed version of the dataset here TODO.

Note: Written for **Python 3**

## Instructions: Apply Preprocessing
In case you want to use your own images, store them in ```/data/unprocessed_images```. Next of, execute the ```/functions/preprocessing_pickle.py```/ script. It will preprocess those images and store them in a pickle-file in ```/data/pickle```.

## Instructions: Training
It is necessary to provide a pickle-file in  ```/data/pickle```. Either create it yourself or download it from the above link.

To simply train the network, either just execute the file or provide the argument ```-train``` when doing so the file.
```
python3 dc_gan.py
python3 dc_gan.py -train
```
By supplying an integer after ```-nbatch``` , you can change the number of batches used to train (by default **1000**).
```
python3 dc_gan.py -nbatch 5000
```
In case you want to reload model-paramters and train on them (by default they have to be provided in ```/save/model```), provide the ```-load``` option. 
```
python3 dc_gan.py -train -load
```
If you also want the images (generated during training) and the loss and the number of updates of the Networks (in recent iterations) saved (in ```/save/figs```) , provide the ```-vis``` option additionally.
```
python3 dc_gan.py -train -vis
```

###### Note: ```python3 dc_gan.py``` is equivalent to ```python3 dc_gan.py -train -vis```

## Instructions: Testing
In case you want to generate pictures on a trained version (py providing weights in ```/save/model```) use the ``` -test n``` to generate a ```n*n``` picture (by default **5x5**, between 1 and 10). The generated pictures will be saved in ```/my_imgs```.
```
python3 dc_gan.py -test 10
```
If you want to see the influence. TODO via **z-interpolation**, provide the ```-z``` tag, instead of ```n```tag.
```
python3 dc_gan.py -test -z
```

###### Note: In case you supply both ```-train``` and ```-test``` the Network woll first train and then test.
```
python3 dc_gan.py -train -test ...
```

## Implementation details
### SAMPLE TODO!
Neural networks are very costly both in memory and time. This module uses the backpropagation algorithm to efficiently calculate the partial derivatives. Whenever possible, for-loops are avoided and vector calculations using `numpy` are used instead. When you set `n_h = 0` (set the number of hidden layers to 0), your algorithm will be equivalent to a [multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) which is typically a good enough model for most industrial applications and serves as a great baseline statistical model. Additionally, the penalty function is convex which guarantees that you will obtain a unique and global solution (and fast). When you want to learn more complex hypothesis, which is the whole point of this module, you will set the number of hidden layers to 1 or more. In this case, the penalty function is not convex and can have global minimums. Additionally, learning requires more time when you have the hidden neurons. In this module, we use [mini-batch gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) to approximate the partial derivatives quicker and we use the [momentum method](https://en.wikipedia.org/wiki/Gradient_descent#The_momentum_method) for gradient descent to try to overcome local minimums and try to approach the global minimum even on non-convex surfaces. Both of these parameters can be tuned. 


## Contact
In case you have questions or feedback, feel free to contact us via **random@mail.de**. Feel free to fork this repository.

