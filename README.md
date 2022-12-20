<div align="center">
	<h1>ASL Learner</h1>
</div>

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px" src = "https://imgur.com/UlCYr54.png" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </div>
</div>
<br/>


## Description
ASL Learner is a machine learning-based website. We have created this application for people who are interested in learning American Sign language. Through this app, a user can learn how to spell each letter in American Sign language. This app is very useful for people with speaking disabilities and hearing disabilities. This website will teach sign language to people with hearing disability, which will allow them to communication with others. This app will also help people who want to pursue any career to help the hearing-disabled person. Our website is very user-friendly and eye-catching. It is very easy to interact with. We have pages on the website but in the beginning, a user will be greeted with the home page. Then, they can move to the next page where they will find many words on a different levels. The user can choose any word from those levels and it will take them to the page. On that page, the user will try to mimic the sign language of each letter of that chosen word. When the user is done, the website will bring them back to the word choice page. There is also a sign in feature which saves a user’s progress which they can access through the info page. The different pages of our website is shown below:

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px" src = "https://imgur.com/6c8fb1J.png" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </div>
</div>
<br/>

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px" src = "https://imgur.com/5nMo6JW.png" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </div>
</div>
<br/>

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px" src = "https://imgur.com/7oTwy6L.png" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </div>
</div>
<br/>

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px" src = "https://imgur.com/7QcTnYb.png" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </div>
</div>
<br/>

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px" src = "https://imgur.com/ZY2JUl1.png" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </div>
</div>
<br/>

## Design and Experiments 

### Software Design

### ML Model Design
The ML model used for gesture classification is a modified ResNet152. The output layer of the model was modified to produce 26 classes which correspond to 26 letters in the alphabet. The model was trained for one epoch with ResNet152 layers frozen and then for an additional epoch with all layers unfrozen. 

#### Hyperparameters
```python
epochs = 1
max_lr = 1e-4
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```

To improve the model training each image has a resize to 200x200, 25-pixel padding added to each side, a random horizontal flip, a random rotation of 10 degrees, and a random perspective.

#### Train Data Transforms
```python
train_tfms = tt.Compose([
                         tt.Resize((200, 200)),
                         tt.RandomCrop(200, padding=25, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.RandomRotation(10),
                         tt.RandomPerspective(distortion_scale=0.2),
                         tt.ToTensor()
                        ])                       
```
The images used for testing the model were first preprocessed using a [tool](https://github.com/danielgatis/rembg) to remove the background of an image. This [background removal model](https://github.com/danielgatis/rembg) is also used in the backend to preprocess images before the evaluation using ResNet152. That allows us to ignore the background of the gestures confusing the model which increases the accuracy of classification.

An example of image preprocessing using this tool:

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px" src = "https://imgur.com/zqFGk1u.png" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <img style="margin-left: 35px" src = "https://imgur.com/1cVVKhi.png" />
    </div>
</div>
<br/>

The initial model for the website was a simple convolutional neural network built from scratch. There were many different variations of the built-from-scratch model, however, the best overall was a one-dimensional image classification model trained on images 75x75, which had 4 ReLU dense layers, and 3 ReLU convolutional layers.

#### Units per Layer
```python
self.dense_layer_1 = nn.Linear(1470, 800)
self.dense_layer_2 = nn.Linear(800, 600)
self.dense_layer_3 = nn.Linear(600, 500)
self.dense_layer_4 = nn.Linear(500, 300)
self.output_layer = nn.Linear(300, 26)
```
#### Convolutional Layer Parameters
```python
kernel_size1=5, stride1=2
kernel_size2=4, stride2=1
kernel_size3=3, stride3=1
```
#### Hyperparameters
```python
epochs = 4
criterion = torch.nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
```
However, that initial approach was incorrect, since image classification is much more complex and such a simple model would not be able to produce good results. Even though in combination with background removal the initial model allowed us to achieve high testing and training accuracy, it failed at the evaluation of other datasets that were not used for training. The best accuracy on the non-training dataset was 17%, which is more than a probability of random guessing (1/26 = 3.8%), but still not enough for good classification of gestures.

So for the reason described above, a pre-trained image classification model was used. The first version was ResNet34. It immediately was able to produce a much higher accuracy on non-training datasets (60% to 70% depending on the dataset used for testing), unlike the built-from-scratch model. It was clear that the overfitting didn't happen anymore and usage of a pre-trained model was the right path. Different experiments were performed to make the accuracy even higher, but the best solution again was to use another more complex model, which is the ResNet152. A model similar to ResNet152, Regnet_y_32gf, was also tested, but the accuracy results were similar, so the decision was to stick with ResNet152. The parameters used to train pre-trained models are identical to the parameters used for the final model. Testing data was also first preprocessed using [background removal model](https://github.com/danielgatis/rembg).

### ML Dataset Description
#### Synthetic
Dataset used for training the built-from-scratch model. The idea was to make our model distinguish between hand and background since the synthetic dataset contains a lot of images with simulated backgrounds. It didn't work, so the background removal model was used after.

https://www.kaggle.com/datasets/allexmendes/asl-alphabet-synthetic

#### Londhe
Was used first to test the built-from-scratch model trained with a Synthetic dataset. Since the accuracy was 0% this dataset was also used to train the built-from-scratch model.
https://www.kaggle.com/datasets/kapillondhe/american-sign-language

### ML Model Experimental Performance

### User Interface Design

## Code Organization
* ASL-Learner
  - public: this folder contains all the images for website logos
  - src: this folder contains all the .js, .css files. This folder also contains components, asl_letters, images folders
  	- asl_letters: this folder contains all

## How to Run

## How to Train

## Challenges Faced
### ML side
The model ResNet152 used for the project is not ideal Some gestures (particularly gestures for pairs of letters E&S, K&V, and G&H) are often misclassified because of the similarity of hand gestures themselves. See the example for letters E and S below:

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px" src = "https://imgur.com/SjzD6Gz.png" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <img style="margin-left: 35px" src = "https://imgur.com/RWperuv.png" />
    </div>
</div>
<br/>

We have tried to implement two possible solutions to it. One of them was a multiple frame classification and another one was to use another additional binary models that would classify one of the letters in those pairs.

However, it didn’t seem to improve the user experience and we’ve decided to simply implement custom thresholds in the backend for those letters since the purpose of our project is to teach a user how to do the fingerspelling. The human mind is much more advanced than our model. If a user places his thumb finger just a little bit differently from what our model could understand, another person would be able to identify the gesture from the context correctly.


## Future Work

## The Team
This project is the combined effort of ...
