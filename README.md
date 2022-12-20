<div align="center">
	<h1>Name</h1>
</div>

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img src = "link to image" />
    </div>
</div>
<br/>


## Description

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
The images used for testing the model were first preprocessed using a [tool](https://github.com/danielgatis/rembg) to remove the background of an image. This [background removal model](https://github.com/danielgatis/rembg) is also used in the backend to preprocess images before the evaluation using ResNet152.

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

### ML Model Experimental Performance

### User Interface Design

## Code Organization

## How to Run

## How to Train

## Challenges Faced

## Future Work

## The Team
This project is the combined effort of ...
