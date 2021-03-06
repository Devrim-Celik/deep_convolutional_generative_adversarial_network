# Deep Convolutional Generative Adversarial Network - Documentation

## Task Description
One of the tasks we did at our university required us to build a simple DCGAN, trained on [NMIST](http://yann.lecun.com/exdb/mnist/). Although impressed, we asked ourselves how far we can come with a more challenging dataset. At the end we went for the [CelebA-Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), since we, as humans, have a high standard for faces.
Next to that we also were interested how the **Z** vector is influencing the resulting image. The idea that the Generator is supposed to map this vector into a "space of faces" was quite astonishing and so we decided also to implement evaluation tools such as a **Z-interpolation**.
Finally we hope that this repository serves as guidance for students, such as we are, to better understand this fascinating model.

## Introduction: What is a DCGAN
Generative Adversarial Models are fairly new branch of unsupervised learning. Their goal is to generate samples (images, text, music, videos) which resembles their trainings-set. The remarkable thing is: It does **not reproduce** them; i.e. it generates images which were never there before.

To dive further into the topic and our implementation, let's cover the basics first.

#### Basics 1: What is a Generative Model
Like it was said before, generative models are able to randomly generate observable data values. Let's assume a generative Model **g** is trained on a training data **X** sampled from a true distribution **D**. If, after the training is complete, we provide **g** with some standard random distribution **Z** it will produce a distribution **D'** which is supposed to (more or less) closely resemble **D**.
The convenient way to determine **g** involves a maximum likelihood estimation and further complex calculations.

#### Basics 2: What is a Generative Adversarial Model
By using an adversarial training process we can have an elegant way of avoiding these convoluted calculations. The idea is that instead of having only a generating Model **g** called the **Generator**, we also bring a discriminating Model **d** in the playing field which is called the **Discriminator**.

Assuming our training data **X** is sampled from a Distribution **D<sup>d</sup>**, after training **g** is able map random input **D<sup>n</sup> ==> D<sup>d</sup>**. It will try to make those samples as "good" as realistic as possible.

The discriminator **d**'s task is to map **D<sup>d</sup> ==> {0,1}**. It will get real samples from **X** and "fake" samples **X<sub>fake</sub>** generated by **g** and try to discriminate them by ideally labeling all samples **X** with 1 and all samples **X<sub>fake</sub>** with 0.

The reason for having these 2 models is to put them up against each other, i.e. **d** will try to "filter out" all generated sample **X<sub>fake</sub>** and recognize all real samples **X**. On the other hand **g** will try to create samples which closely resemble **X** so that **d** is not able to categorize them as fakes.

##### Analogy for the conceptual idea behind GANs
Imagine a criminal (**Generator**) who is trying to counterfeit famous paintings to later sell them to a museum in his town. The museum on the other hand, has employed an famous art expert (**Discriminator**) whose salary depends on deciding whether a painting is real or a fake. This is a *zero-sum game* setup, i.e. the better the criminal does, the less money the art expert gets; and the better the art expert does in distinguishing between real and fake portrayals, the less money the criminal earns. By having this kind of competition, we can ensure both the criminal and the art expert will try their hardest. Even if the expert has found a solid way to discriminate fake and genuine pictures, the criminal will change his style until they again pass, at which point the expert will again try to improve himself and so on ...
The only real difference to the real training process is that both Discriminator and Generator are untrained at the start.

#### Basics 3: What is a Deep Convolutional Generative Adversarial Model
Traditional Generator and Discriminator models, as they were described in papers such as [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661), were multilayer perceptron. A different angle of generating and discriminating images is via convolution.

##### Convolution
![alt text](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase_convolution.gif "Convolution")

Images are discriminating by using traditional convolution, while they are generated by using [transposed convolutions](https://arxiv.org/abs/1603.07285).

As you can see in the [original paper](https://arxiv.org/abs/1511.06434v2) of this idea, one of the many advantages coming with a deep convolutional adversarial pair is that it is able to "learn a hierarchy of representation from object parts to scenes in both the generator and discriminator".

## Preprocessing
It is important to mention that our used dataset is already preprocessed, in sense that we used a cropped and aligned version of it.
The first thing we did is to crop it into a quadratic form. Next off, we resize it into 64x64 and grayscale it; this is done due to the fact that we have limited computational power. We also map the grey values of the images from **0 - 255** to **-1 - 1**, due to the fact that the input unit (**Z** vector) is taken from a uniform distribution between these values. Finally, we add a third "empty dimension" for compatibility reasons.

![alt text](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase_processed.jpg "Preprocessed Image")

The limitation in terms of resizing and grayscale is an important step since, even with it, we have 202000\*64<sup>2</sup>\*256 (**~200 billion**) values to work with.

## Theoretical Basis of the Training
Again, when we talk about a generator **g** and a discriminator **d** we are talking about two Neural Networks; one generates images and the other discriminates them. Question is, how does it work?

First off **g** is provided with a vector **z**; the product of this input are *fake* images **g(z)**. This process can be done either via a the conventional way of a multilayer percepton or, as described above, with transposed convolution mapping this vector into the right format.

On the other hand **d** has to be able to discriminate, i.e. we need to supply it with *real* and *fake* images. The fake images are the output of the generator and the real images are our trainings data **X**; usually the same amount of them is fed into **d**.

**d(X)** represents the probability of a sample **X** being a *real* image, while **d(g(z))** describes the probability of **g(z)** being a *real* image. This means it optimal for the discriminator to map **d(X)** as close to one and **d(g(z))** to zero as possible. One the other side, the generator optimally is able to generate image in such a way that **d(g(z))** is as close to 1 as possible (the discriminator is unable to differentiate between *real* and *fake*). This results in the following descriptive functions:

**g** tries to minimize: log(1-d(g(z))) while **d** tries to maximize: log(d(X)) + log(1-d(g(z))).

To showcase the idea of the zero-sum game, we can also say, that **g** tries to minimize the latter function, while **d** tries to maximize it.


## Network Structure and Design Choices

![alt text](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase_architecture.gif "GAN Architecture")

Our network architecture is a scaled down version inspired by [Radford et al. (2015)](https://arxiv.org/abs/1511.06434). We do not have such a powerful architecture since our data has lower dimensionality (64x64, grayscale not rgb).

The generator **g** first expands the **z** vector of size **100** to a vector of length **512x4x4** and reshapes this vector into a **512x4x4** matrix. It then expands the **4x4** feature maps via four transposed convolutions with stride **2**, a **5x5** kernel size and decreasing amount of feature maps to a single **64x64** matrix of depth **1** (grayscale), i.e. the generated fake image. Except for the final layer, we use batch-normalization after each convolution.

The discriminator **d** mirrors this architecture by convoluting its input four times, reshaping to a 1d-layer and mapping to a single output unit. We are using strides of **2** and **5x5** kernels again. To regulate the capacity with respect to the generator's training progress, we apply dropout after the second and fourth convolution. No batch-normalization is applied in the discriminator.

##### Dropout
The term "Dropout" describes a regularization technique used to avoid overfitting in neural networks.
The basic idea is that we leave out certain units ('drop') for every training iteration. The chance of which a unit is left out is random; in the simplest case we have a static probability **p** which describes the chance that a unit is dropped.

By "thinning" a network out like this we have an interesting effect: Every training iteration we have a different model we train with. This prevents weights from converging to identical positions. Note that the backpropagation, on the other hand, is done with all units "turned on".

## Training Procedure
During training, we update the discriminator with mini-batches consisting of 64 randomly picked real images and 64 **Z** vectors. The generator is updated using the same z_vector batch as well as an additional 64 **Z** vectors. We use the Adam optimizer with initial learning rates of 0.0002 for the discriminator and 0.0002 for the generator.

![alt text](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase_interval.png "Interval-Illustration")

The two learning rates are adjusted given the discriminator's accuracy on the fakes  as well as the real data. We set two thresholds to determine the learning rate adjustment: If either of the discriminator's accuracies is below the first (lower) threshold the discriminator's learning rate is increased slightly and the generator's decreased since the discriminator's classification is considered less informative. If the discriminator's accuracies reach above the second (higher) threshold, we do the opposite, since the discriminator should not get too powerful and the generator gets very informative feedback. In between the thresholds we increase both learning rates. This enables us to control the training process which again yields better results.

The key to success in training a DCGAN is tuning the interplay between generator and discriminator, because one of them gets "ahead". If the discriminator is too powerful, it will classify all the generators image as fake. In the opposite case, the discriminator is fooled every time and hence classifies everything as true. For the generator both cases result in the same situation: *It is left with no useful feedback*. Both architecture and learning parameters have to be tuned to achieve good results.

The game between discriminator and generator was hard to balance. We were not able to fully eliminate the chance of the network getting "stuck". Our first approach to avoid oscillations was to [leave either the discriminator-update or the generator-update out](http://torch.ch/blog/2015/11/13/gan.html) for one training batch if the accuracies behaved as described above. The interval needs very precise tuning: setting the lower threshold too low results in the discriminator failing and setting the higher threshold too high makes it too powerful. During training, the network eventually "recovers", but a lot of mini-batches are wasted. To still be able to use all batches for both components of our GAN, we instead decided to increase/decrease the learning rate to this update schedule. Limiting the learning rate within a range was required, again to avoid oscillations. The procedure yielded more stable and consistent results and visually clear transitions between sample images (lower learning rate at times).

## Evaluation

### Z-Parameter Change
![alt text](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase_Z-Change1.png "Z_Change1")
![alt text](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase_Z-Change2.png "Z_Change2")

One question which intrigued us while building the Network was how the different parameters of the Z-vector influence the resulting images. To get some insight into it, we created a Z-vector and iterated over different values for only one of its 100 parameters while keeping the other 99 parameters. At the end we have 12 of those images (of cause it is possible to change in this code). In the visualization above you can see in total 8/12 of those images (iteration is done in the rows while columns are indicating different values for one parameter).

What we can see here is firstly, that the value of every parameter influences the background if it is big enough, which makes sense.

In the first row of the first sample image it is also recognizable that this parameter influences the gender of the generated face, the higher this value gets (more to the right) the more masculine the face gets (although the hair still is feminine).

In the fourth row of the first sample we can see that this parameter also influence the gesture, since the person on the left is clearly laughing (teeth visible) while the highest value (most right picture) is just smiling.

### Z-Interpolation
![alt text](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase_Z-interpolation.png "Z_Interpolation")
![alt text](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase_Z-interpolation-gif.gif "Z_Interpolation_Gif")

The idea for Z-interpolation was provided by [Generating Faces with Deconvolution Networks](https://zo7.github.io/blog/2016/09/25/generating-faces.html). What we do is to take two random Z-vectors and over the course, of a total of 64 images, interpolate between them. The pictures and the give above are exactly those 64 vectors, after the Generator mapped them onto faces.

Why do we do this and how does it help use evaluate our model: A good model maps the given Z-Vectors into a space (in our case in a 64<sup>2</sup> dimensional), lets call it "face-space". A bad model may just try to reproduce the images and not generate new ones. By taking little steps, we can check that the transitions are smooth and thus, a good "face-space" was found.

In terms of **evaluation** we can say with a clear conscience that our model does a good job. This is due to the fact that the interpolating values from the one image (top-left of the left image) to another one (bottom-right of the left image) are faces themselves and the transitions are really smooth. This shows that the mapping into a "face-space" was successful and images are genuinely generated and not duplicated.

### Evaluation

Interestingly, most of the faces (~75%), have long hair while the faces are quite neutral. Although we have some hypothesis to why this is the case (One of them being the fact, that long hair covers more area and thus also covers more background.  The background was quite versatile on the original images, thus "eliminating it" was beneficial for the learning process of the GAN.), the most probable reason seems to be a uneven distribution of men and women in the dataset.

The training process itself was quite interesting when considering which features of a human face the network learned first:
1. **Head Shape**: In the early iteration (up until 300 batches) the network tries to fit the the head form. Most of them result in having a "bright" face with a dark background. As you can see in the Demonstration on the README there are also some faces which are the other way around, i.e. the faces are almost black while the background is white. This is probably due to the fact that most of images of the dataset feature a bright background while a minority does have a bright background.
2. **Hair**: The hair color on the other side does not seem to be related neither to the background nor to the color of the head.
3. **Eyes and Mouths**: Next off eyes and mouths arise. The eyes possess the complementary color of the head, which makes perfect sense, since they would not be visible otherwise. The evolution of the mouths is quite interesting: For almost every face the shape of the mouth oscillates between a more neutral/friendly smile and a big smile, where teeth are observable.
4. **Complex Details**: Some of the pictures develop even more sophisticated details. We could observe generations where some of the persons wore glasses. In the demonstration on the README, one person (4<sup>th</sup> row, 1<sup>st</sup> column) seems to exhibit bangs toward the end of the animation. Sadly these animations are rare.

#### Plots
![alt text](https://github.com/D3vvy/iannwtf_DCGAN/blob/master/images-gifs/showcase_stats.png "Stats")

Lets look at the 3<sup>rd</sup> plot of the ones above. It describes the change of the learning rate; as mentioned before, in our case the learning rates depend on how well the networks do: the better one model does, the lower its learning rate while we increase the learning rate of the other network.

In the first part of the training (0-50) one can clearly see that the discriminator is prioritized in terms of learning. This is due to the facts that at this point, it has not the closest idea what characteristics a real face exhibits. After it got to an acceptable level, the generator is prioritized (50-120); it's the Generators turn to learn what makes a face a face. Lastly, when both of them have reached a considerable level, them "converge" into equilibrium.

Note that this is not always the case. Often time it takes some tuning during the training process, since the non-linear nature of GANs make it quite easy to converge into bad values.

### Conclusion
Finally, one can say that Generative Adversarial Networks implemented via Deep Convolution are a powerful and versatile tool. Although in our case we only generated images, there is no real limit to what it can do. Be it [Video](http://web.mit.edu/vondrick/tinyvideo/), Sound or [Text](https://arxiv.org/abs/1605.05396); after adjustments in the architecture they all can be done.
Although our Network surely has room for improvement, the results are already convincing. There may be some samples which do not really look like a face, what indicates the Network mapped them out of the "face space". On the other hand, most of them do not only resemble a face, but a hard to discriminate even for the human eye. That fact alone makes this project a success in our book.
