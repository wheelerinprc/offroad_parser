# Off Road Parsing with UNet
## Introduction
In this project, I replicate traditional [U-Net](https://arxiv.org/pdf/1505.04597.pdf) model and train it to realise 
off-road parsing inference task.
## Dataset
[RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)
## Inference Sample Result
#### Image input
![Image input](https://github.com/wheelerinprc/offroad_parser/blob/master/proj/original_image.jpg?raw=true)
#### Inference result
![Inference result](https://github.com/wheelerinprc/offroad_parser/blob/master/proj/inference_result.png?raw=true)
#### Ground truth
![Ground truth](https://github.com/wheelerinprc/offroad_parser/blob/master/proj/ground_truth.png?raw=true)
#### Indicator
See [indication.json](https://github.com/wheelerinprc/offroad_parser/blob/master/proj/indication.json)
## Scripts
#### dataset.py
Load image and label from given dataset directory. 
- Use ToTensor to transfer image to normalized (0-1.0) tensor
- Use PILToTensor to transfer label to 0-255 tensor
#### unet.py
Replicate U-Net. Model's layer, width and input/output channel can be set when initialization.
#### diceloss.py
Calculate dice loss for training and evaluation.
#### eval.py
Use validation dataset to evaluate the model.
#### train.py
Build the training pipeline for U-Net.
- Automatic mixed precision (AMP) is implemented.
- Dynamic learning rate is implemented.
- Cross entropy loss is used. Dice loss is imported and tried.
- Call eval.py to evaluate the model on validation dataset every epoch.
- Save the model parameter checkpoint
#### process.py
1. Read and parse configuration json file
2. Load dataset
3. Initialize Tensorboard logger
4. Construct U-Net
5. Train the model
#### infer.py
Read an image, load the trained model and do inference on it. Export inference visualization result and indicator.