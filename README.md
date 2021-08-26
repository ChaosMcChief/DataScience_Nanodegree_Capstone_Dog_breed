
# DataScience_Nanodegree_Capstone_Dog_breed

## Intro
This repository refers to the final capstone-project of the Udacity Data Science Nanodegree. 
In this case we develop a model which can detect dogs and humans in images. 
If a dog is detected, the model predicts the breed. If a human is detected, the model predicts a dog breed which has the most resemblence to the provided human face.

The algorithm works best with close up pictures of humans an dogs, so make sure to crop accordingly, if necessary.

## What's in here?
The repo basically consists of two main parts.

- A jupyter notebook (also available as a HTML-export in wich the modeling and experiments are done. It goes through different approaches (CNNs from scratch, transfer learning, etc.) as well as the other parts of the algorithm, such as a human face detector using openCV. 
- A web app, which lets you upload an image and then classifies it. If the algorithm detects a dog, it predicts the dog breed and outputs it. If it detects a human face, it outputs the dog breed, which most resembles the human in the photo. If neither of these categories are detected, it gives you a message accordingly.

## Install
The model runs on Keras and Tensorflow. It can be a bit of a hustle to install all the necessary packages. 
To get going easily, just create a Conda-Environment and install the necessary packages with
```
conda create --name dogbreed python=3.8.10
conda activate dogbreed
pip install requirements.txt
```
If you want to use a CUDA-enabled GPU, make sure to have the CUDA-Toolkit 11.3 and cuDNN 8.2 installed.

## Instructions
### Data
The dataset and the bottleneck features are not provided due to their size. If you want to follow along all of the jupyter notebook, you have to download them yourself.
Bottleneck features:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

Dataset
- [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
- [human_dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

The dataset has to be stored a `data`-folder in the project-direcotry, i.e. 
- `data/dog_images/...` for the dog-images
-  `data/lfw/...` for the images of human faces.

Also...in `images/test_images`are a few test images you can try.

### Models
Modelwise the repo comes with the necessary models to run the web app. The other models can be optained by running the notebook.

### Web app
To start the web app, silmpy open a console and navigate in the `app`-folder. There execute the command
`python run.py`
As soon as the server is up and running just go to the localhost-url: [localhost:3001](localhost:3001)
There you can upload an image and let the algorithm classify it. 
Enjoy :)

## ToDos / What's next
If a human was detected in the image and the model predicts the most resembling dog breed, it would be nice to find the most simalar image of the predicted dog breed to the provided image and show it next to the human image.
I think one possible way to achieve this could be to run all the training images of the respective dog breed through the ResNet50-model to compute the bottleneck features. Then calculate a metric of difference (e.g. cosine similarity) with the bottleneck features of the provided image, sort them and pick the dog image with the highest similarity. Then show it next to the oroginal photo. In this way you would not search for the most similar image (like pixelwise similarity), but more for the most similar features (i.e. why the model predicted the specific dog breed). 

What do you think? Is there a better way to do this? Let me know, thanks!


