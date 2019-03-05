# Image-Classifier-using-Pytorch

## Description
This project involves the implementation of an image classification application using a deep learning model on a dataset of images.The trained model will then be used to classify new images .GPU processing is recommended for this project.

Part 1 of the project involves implementing an image classifier that is trained on a flower data set . There are 102 different types of flowers, where there are approximately 20 images per flower to train upon

Part2 involves converting the notebook into a python application that can be run from the command line to train a neural network and use it to make predictions

## Dataset Used
If flower dataset is used, download/extract/place the image dataset from [link](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) in the following folder structure:
        
        project
          |-- codes
          |-- data
               |--- train
                    |-- class 1
                        |-- images of class 1
                    |-- class 2
                        |-- images of class 2
                    |-- etc
               |-- valid
                    |-- class 1
                        |-- images of class 1
                    |-- class 2
                        |-- images of class 2
                    |-- etc
               |-- test
                    |-- class 1
                        |-- images of class 1
                    |-- class 2
                        |-- images of class 2
                    |-- etc
                    
                    
If other images are used, place them in the same folder structure as above.

## Files in the repo
**Image Classifier Project.ipynb** It is used to build the model using jupyter notebook .It can used indpendently to see how the model works.

**cat_to_name.json** It is used to map flower number to flower names

**train.py** It is used to train a new network on a dataset and save the model as checkpoint.

**predict.py** It uses the trained network to predict the class for an input image.

## How to run the commandline application
- Train a new network on a data set with train.py
  - Basic usage: python train.py data_directory
  - Prints out training loss, validation loss, and validation accuracy as the network trains
  - Options:
       - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
       - Choose architecture: python train.py data_dir --arch "vgg13"
       - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
       - Use GPU for training: python train.py data_dir --gpu
- Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
  - Basic usage: python predict.py /path/to/image checkpoint
  - Options:
      - Return top KK most likely classes: python predict.py input checkpoint --top_k 3
      - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
      - Use GPU for inference: python predict.py input checkpoint --gpu
           
           
## Acknowledgements
Credit to Udacity for creating such a wonderful project
