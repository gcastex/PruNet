<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/gcastex/PruNet">
    <img src="logo.png" alt="Logo" width="800" height="276">
  </a>

  <h3 align="center">Weight loss for machine learning</h3>

  <p align="center">
  A study on pruning and the Lottery Ticket Hypothesis.
  </p>
</p>




<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [What is pruning?](#what-is-pruning)
  * [Why use pruning](#why-use-pruning)
  * [How does it work](#how-does-it-work)
  * [The Lottery Ticket Hypothesis](#the-lottery-ticket-hypothesis)
* [Facial Recognition](#facial-recognition)
  * [Dataset](#dataset)
  * [Pipeline](#pipeline)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project
 The goal of this project is to understand the concept of pruning for neural networks. 
 Two networks are being pruned in this study: an image classifier built with a fully connected network trained on MNIST, and a facial recognition model trained on a custom dataset.
The reinitialization method presented in the Lottery Ticket Hypothesis paper is tested on both models.


### Built With

The project was coded in Pytorch.


## What is pruning?
Pruning is a technique that allows to reduce the size of a neural network by removing some of its branches.

## Why use pruning?
Let's first have a look at why pruning is used.

Running a ML model can be computationally expensive in terms of:
-Disk Space
-Memory usage
-CPU/GPU runtime

There is a strong incentive in making the machine learning models as small as possible to reduce their costs.

There are several techniques to reduce the size and the computing time of a model: quantization/binarization, knowledge distillation, low-rank factorization, parameters pruning... Several of these techniques can be used together to optimally compress a model.
We will study pruning in this project.

## How does it work?

## The Lottery Ticket Hypothesis

The original paper can be found here:
[https://arxiv.org/abs/1803.03635](https://arxiv.org/abs/1803.03635)


## Facial Recognition

## Dataset

The dataset contains facial images of seven characters from the TV series Game of Thrones.
The characters are:
1. Daenerys
2. Jon Snow
3. Varys
4. Tyrion 
5. Podrick
6. Grey Worm
7. Bronn

Images were collected from Google Images and each class was manually cleaned.
About 100 images per character were collected.

## Pipeline

Preprocessing: Faces in the images are detected with the MTCNN algorithm, cropped and saved as new images. Cropped face images are then resized to 144x144 and converted to grayscale.

Grayscale images are used as input for the LightCNN facial recognition model.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Guillaume Castex - g.castex117@gmail.com

Project Link: [https://github.com/gcastex/PruNet](https://github.com/gcastex/PruNet)

## Acknowledgements
