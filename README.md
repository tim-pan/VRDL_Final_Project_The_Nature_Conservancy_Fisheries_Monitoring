# VRDL_Final_Project_The_Nature_Conservancy_Fisheries_Monitoring
The given training dataset includes 3777 images, each image contains a certain species of fish, our goal is to train a model which can classify images into the following 8 classes.

## Coding Environment
- Google Colab

## Reproducing Submission
To reproduct the testing prediction, please follow the steps below:
1. [Running environment](#environment)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Testing](#testing)

## Environment
- Google Colab

## Dataset
- Because of using pytorch to predict the image, imagefolder could be used to generate training and testing data.
- Imagefolder needs the image be arranged in the folder according to their labels, so we need to classify the image with folders.
1. Download the dataset from https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring.
2. Set the download dataset's root according to your file path.
3. Run the "split_train_val.py" file and we can get the "train_split" and "val_split" folder.


## Training
Upload the "Main.ipynb"、"inference.ipynb"、"split_train_val.py"、"train_split"、"val_split" folder 、 "utils" folder which contains "dataset.py"、"function.py"、"split_train_val.py"、"ensemble" folder which contains "ensemble_experiment.ipynb" on google drive.
Run the "Main.ipynb" file will start to train the model.

The training parameters are:

Model | learning rate | Training epochs | Batch size
----------------------------- | ------------------------- | ------------------------- | -------------------------
efficientnet_b7 | 0.0001 | 40 | 8

Model | learning rate | Training epochs | Batch size
----------------------------- | ------------------------- | ------------------------- | -------------------------
inceptionv3 | 0.0001 | 25 | 8

Model | learning rate | Training epochs | Batch size
----------------------------- | ------------------------- | ------------------------- | -------------------------
regnet_y_32_gf | 0.0001 | 40 | 8

Model | learning rate | Training epochs | Batch size
----------------------------- | ------------------------- | ------------------------- | -------------------------
resnet152 | 0.0001 | 40 | 8

### Pretrained models
Pretrained model "efficientnet_b7" which is provided by torchvision.
Pretrained model "inceptionv3" which is provided by keras.
Pretrained model "regnet_y_32_gf" which is provided by torchvision.
Pretrained model "resnet152" which is provided by torchvision.

### Link of my trained model
- The model which training with The_Nature_Conservancy_Fisheries_Monitoring dataset：https://drive.google.com/file/d/1DBKRTMACJHhJ8zhiXia6wutzxSNV_Lh0/view?usp=sharing

### Inference
Load the trained model parameters without retraining again.

“model.pth” need to be download to your own device and run “inference.ipynb” you will get the submission file.
