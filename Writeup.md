# Data Scientist Capstone - Dog breed classification

![The web app in action](https://github.com/ChaosMcChief/DataScience_Nanodegree_Capstone_Dog_breed/blob/main/images/write_up/Webapp.PNG?raw=true)

## Tabel of contents
1. [Project Definition](#project-definition) 
2. [Analysis](#analysis)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Conclusion](#conclusion)

 

## Project Definition
### Project Overview
In this project we tackle a task in the computer vision domain, specifically a image classification tast. 

We try to build a model that can detect dogs and human faces in images. Once a dog is detected, it predicts the dog's breed. If, on the other hand, a human face is detected, it predicts a dog breed, which most resembles the given picture. 
The project is the final capstone project of the Udacity DataScience Nanodegree. 
All project files with all the implementations can be found in my [github-repo](https://github.com/ChaosMcChief/DataScience_Nanodegree_Capstone_Dog_breed). 

### Problem Statement
Our algorithm has to have the following features / capabilites:

- It can detect dogs in a given image
- It can detec human faces in a given image
- If neither of the above is detected, the algorithm has to give a according message
- If an image of a dog is provided, it has to state, that a it is a picture of a dog and predict the underlying dog breed
- If an image of a human is provided, it has to state, that it is a picture of a human and predict the most resembling dog breed.
- The dog- and human-detector both have to have an accuracy of >90%.
- The dog-breed-predictor has to have an accuracy of >60%

### Metrics
The main metric used in this project is accuracy. Even though it has it's drawbacks, especially with inbalanced data, it is also a very intuitive metric and easy to use, even for a non expert user. Since it would be great if this project would create interest in novice data scientists and overall interested people, i opted for a easy-to-understand metric.

## Analysis
The dataset consists overal of 8.351 dog-images (6.680 trian, 835 validation and 836 test) of 133 different dog breeds and 13.233 images of human faces. The human images were just used for validation of the face detector, since a pretrained model was used.
The images are mostly dog photos from different angles of one dog of the respective breed. However, sometimes there are more than one dogs in an image (even from different breeds) or there are humans next to the dog. 

<img src="https://github.com/ChaosMcChief/DataScience_Nanodegree_Capstone_Dog_breed/blob/main/images/write_up/American_foxhound_00474.jpg?raw=true" width="300" height="auto"/>
To note is also, that the resolution is varying between the images, making preprocessing and resmapling the image a necessary step in the data preparation.

The distribution of the training data (i.e. how many images per category are in the training data) is as follows:
<img src="https://github.com/ChaosMcChief/DataScience_Nanodegree_Capstone_Dog_breed/blob/main/images/write_up/Breed_histogram.png?raw=true" width="480" height="auto"/>

||Value counts
|-----|-----|
|count| 133.0
|mean| 50.2
|std |11.9
|min |26.0
|25% |42.0
|50% |50.0
|75% |61.0
|max |77.0

## Methodology
### Data Preprocessing
The main data preprocessing happens in the following function:
```
def path_to_tensor(img_path):
    
    # loads RGB image as PIL.Image.Image type and resamples it to 224 x 224px
    
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    
    return np.expand_dims(x, axis=0)

```

The returned tensor can then be fed to the neural network. 
In case of training data augmentation is also performed via the `keras.preprocessing.image.ImageDataGenerator`-class. The following augmentations are randomly applied:

- rotation up to 20°
- shift in width and height by up to 20%
- horizontal flip

## Implementation and refinement
In order to achieve the goals stated above, the algorithm consists of four main parts:

- A dog detector: A pretrained ResNet50, wich checks, if the predicted class is of a dog-class. This model is a downloaded model, trained on the ImageNet-dataset
- A human face detector: A openCV-model-CascadeClassifier with pretrained weights (`haarcascade_frontalface_alt.xml`)
- A dog breed predicotr: A feed forward neural network on top of a pretrained ResNet50-model without the top-layer. The output-layer has softmax activation function and 133 output-nodes, one for each dog breed in the training set.
- The web app in which you can upload an image and let the model classify it.

The most testing and optimizing went in the dog breed predictor. 

The first approach was a convolutional neural net from scratch with the following architecture:
<img src="https://github.com/ChaosMcChief/DataScience_Nanodegree_Capstone_Dog_breed/blob/main/images/sample_cnn.png?raw=true" width="640" height="auto"/>
The results were pretty poor with a test-accuracy of 3.38%. This is probably due to the low image count per class. Therefore, transfer learning was the next logical step to take.

I tested three of the four suggested network-architectures. I left the Xception architecture out because the bottleneck features were too big and the performance of the other architectures were also on a high level. I tested three base-networks with identical on-top-architecture to see which architecture is the most promising. Here are the results:

||VGG19 |ResNet50 |InceptionV3 |
|---|---|---|---|
|Test-accuracy [%] |68.06 |__81.58__ |69.86 |

So the ResNet50-Model seemed the most promising model-architecture and I continued optimizing it by adding dropout to the network. I also tried introducing some complexity to the top end of the network by adding layer and/or nodes. To find the best model I wrote a helper-function to perform a simple grid search and save the best model for later use. As you can see, I was able to optimize the performance of the model a bit -- although not by much to be honest. Interestingly, additional model complexity made the performance worse. At least some Dropout helped a bit. But with a accuracy of >82% I'm really pleased with the result and think the model is suitable for this application and can be used in the web app.

layer|dropout|accuracy
---|---|---
__[]__|__0.1__|__82.18__
[]|0.2|80.26
[]|0.3|80.26
[1024]|0.1|74.64
[1024]|0.2|79.55
[1024]|0.3|75.72
[2048, 1024]|0.1|76.80
[2048, 1024]|0.2|73.68
[2048, 1024]|0.3|80.26

## Results
In addition to the raw accuracy I tested the model with various pictures of dogs, humans and other things to see its performance. Here are a few examples:
<img src="https://github.com/ChaosMcChief/DataScience_Nanodegree_Capstone_Dog_breed/blob/main/images/write_up/Test_1.PNG?raw=true" width="640" height="auto"/>

<img src="https://github.com/ChaosMcChief/DataScience_Nanodegree_Capstone_Dog_breed/blob/main/images/write_up/Test_3.PNG?raw=true" width="640" height="auto"/>

<img src="https://github.com/ChaosMcChief/DataScience_Nanodegree_Capstone_Dog_breed/blob/main/images/write_up/Test_2.PNG?raw=true" width="640" height="auto"/>

<img src="https://github.com/ChaosMcChief/DataScience_Nanodegree_Capstone_Dog_breed/blob/main/images/write_up/Test_4.PNG?raw=true" width="640" height="auto"/>

As you can see, the dogs were detected correctly by their breed. The algorithm detected correctly neither a dog nor a human and gave a massage accordingly. Also was Heidi Klum detected as a human. I let you be the jugde if she really looks like a silky terrier ;)

## Conclusion
### Reflection
Overall this was a great project to implement the new learned skills from this Nanodegree. Starting with the raw images of dogs and humans to built a complete algorithm which can pretty consistently predict dog breeds and is also capable of detecting humans in pictures was pretty exciting. 
One of the hardest part was the software engineering part developing the web app. I had to figure out how to upload images with flask and handle the requests. But although it wasn't easy, I learned a lot while doing it and I'm prette happy how it turned out in the end.

### Improvement
If a human was detected in the image and the model predicts the most resembling dog breed, it would be nice to find the most simalar image of the predicted dog breed to the provided image and show it next to the human image. I think one possible way to achieve this could be to run all the training images of the predicted dog breed through the ResNet50-model to compute the bottleneck features. Then calculate a metric of difference (e.g. cosine similarity) with the bottleneck features of the provided image, sort them and pick the dog image with the highest similarity. Then show it next to the oroginal photo. In this way you would not search for the most similar image (like pixelwise similarity), but more for the most similar features (i.e. why the model predicted the specific dog breed). This would enhance the user experience quite a bit I think.