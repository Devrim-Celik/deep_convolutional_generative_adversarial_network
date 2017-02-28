# Deep Convolutional Generative Adverserial Network
This is an implementation of a Deep Convolutional [Generative Adverserial Network](https://en.wikipedia.org/wiki/Generative_adversarial_networks) trained using [Adam Optimizers](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam). It was build using [Tensorflow](https://en.wikipedia.org/wiki/TensorFlow) and ist supposed to be our final task for our university course "Implementing Artificial Neural Networks with Tensorflow".

## Demonstration
![alt text](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase_gif.gif "Training Progress")

## Folders and Sripts
- Folders
    + ```/images-gifs```: Images and Gifs for displaying on Git-Hub
    + ```/data```: For storing data
        * ```/data/pickle```: Stores the pickle data [Necessary]
        * ```/data```: Stores unprocessed raw images [Optional]
    + ```/docs```: Documentation
    + ```/functions```: Functions used for preprocessing, visualisation ...
    + ```/save```: For saving pictures which were produces during training & storing weights & plots of loss ...
        * ```/save/figs``` stores all generated images
        * ```/save/models``` stores all generated models

- Scripts
    + ```dc_gan.py```: Main script
    + ```preprocessing_pickle.py```: Applies preprocessing to images stored in ```/data/unprocessed_images``` and saves a pickle containing them in ```/save/pickle```.
    + ```celeba_input.py```: Loads pickle (per default from ```/data/pickle```) and supplies ```dc_gan.py``` with batches.
    + ```auxiliary_functions.py```: Contains visualisation tools, argument handling and a function which returns the name of the most advance model in ```/figs/models```
    + ```gifMaker.py```: creates Gifs

## Installation and Prerequisites
Cloning the Repository via git:
```
git clone https://github.com/D3vvy/iannwtf_DCGAN.git
```

**Make sure, you have the following modules installed**:
- [Tensorflow](https://www.tensorflow.org/versions/r0.12/get_started/os_setup)
- Matplotlib: ```pip install matplotlib```
- Numpy: ```pip install numpy```
- [Scipy](https://www.scipy.org/scipylib/download.html)
- PIL ```pip install PIL``` (optional: preprocessing and visualisation)
- OpenCV2 ```pip install opencv``` or ```conda install opencv``` (optional: preprocessing)
- [ImageIO](https://pypi.python.org/pypi/imageio) (optional: creating gifs)

**Dataset**:
- In this Deep Convolutional Generative Adversarial Network we used the cropped and alligned version of the CelebA Datset, available [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
- If you just want to to train using a pickle file, you can download a preprocessed version of the dataset [here](https://www.dropbox.com/sh/2ls6zf1qvdhfind/AAAYrGDpPKQ25YmrqEf-rR3_a?dl=0).

Note: Written for **Python 3**

## Instructions: Applying Preprocessing
In case you want to use your own images, store them in ```/data/unprocessed_images```. Next of, execute the ```/functions/preprocessing_pickle.py``` script. It will preprocess those images and store them in a pickle file in ```/data/pickle``` (by default, please name it ```celeba_input.dat```).

## Instructions: Training
It is necessary to provide a pickle-file in  ```/data/pickle```. Either create it yourself or download it from the above link and name it ```celeba_input.dat```, so the built-in mechanisms apply.

To simply train the network, just execute the file and provide the argument ```-train``` when doing so. At the end of any training process, the information plots will be stored in ```/save/figs```.
```
python3 dc_gan.py -train
```
By supplying ```-nbatch``` and an integer after that, you can change the number of batches used to train (by default **2000**).
```
python3 dc_gan.py -nbatch 5000
```
In case you want to reload model-parameters and train on them (by default they have to be provided in ```/save/model```), provide the ```-load``` option. Models are saved in such a way that their are labeled with number of batches it took to create them. In the default case the model with the most batches will be taken. In case you want to start from a certain point in your trainings-process, just look up the number of the model and provide it as an additional argument after ```-load```.
```
python3 dc_gan.py -train -load
python3 dc_gan.py -train -load 6436
```
If you also want images which were created during training to be saved (in ```/save/figs```), provide the ```-vis``` option additionally.
```
python3 dc_gan.py -train -vis
```

###### Note: ```python3 dc_gan.py``` is equivalent to ```python3 dc_gan.py -train -vis```

## Instructions: Testing
All of the following will need a model to test on. Just provide one (or multiple) models in the ```/save/model``` folder with the original name (the one they were created with). By default the most advanced one will be taken.

In case you want to generate pictures on a trained version (py providing weights in ```/save/model```) use the ``` -test n``` flag to generate a ```n*n``` picture (by default **5x5**, has to be between 1 and 8). The generated pictures will be saved in ```/save/figs```.
```
python3 dc_gan.py -test
python3 dc_gan.py -test 10
```
If you want to see the results of the **Z-Interpolation**, provide the ```-z_int``` tag. You won't be needing to provide an integer, since only one will be generated. The image will be saved in ```/save/figs```.
```
python3 dc_gan.py -test -z_int
```
Furthermore, if you are interested in seeing the results of the **Z-Parameter Change**, provide the ```-z_change``` tag (see Documentation for information). As you can see in our documentary it enables you to iterate over one of the 100 parameter values. If you want to specify which parameter to change provide an integer between 1 and 100 as an additional argument. In case you do not a random values between 1 and 100 will be chosen. You won't be needing to provide an integer, since only one will be generated. The image will be saved in ```/save/figs```.
```
python3 dc_gan.py -test -z_change
python3 dc_gan.py -test -z_change 42
```

###### Note: In case you supply both ```-train``` and ```-test``` the Network will first train and then test.
```
python3 dc_gan.py -train -test ...
```

#### In case you need help: Just add the ```-- help``` tag!

## Contact
In case you have questions or feedback, feel free to contact us via **random@mail.de**. Feel free to fork this repository.
