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
The ML model used for gesture classification is a modified ResNet152. The output layer of the model was modified to produce 26 classes which correspond to 26 letters in the alphabet. The model is trained for one epoch with ResNet152 layers frozen and then for an additional epoch with all layers unfrozen. 

#### Hyperparameters
```
epochs = 1
max_lr = 1e-4
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```

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
