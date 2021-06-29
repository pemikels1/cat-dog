# Cat-dog
This project is classifying images as either 'cat' or 'dog.'

## Data
Data is from the Kaggle Competition "Dogs vs. Cats" found [here.](https://www.kaggle.com/c/dogs-vs-cats/overview)

The data has 12,500 images of dogs and 12,500 images of cats in the labeled train set. There are 12,500 images of dogs and cats in the unlabeled test set. 

For this exercise, I split the train set into a train and a dev set, with 20,000 images in the train set and 5,000 images in the dev set.

Each image is resized to size 3 x 300 x 400 and normalized with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).

I created a custom Convolutional Neural Network consisting of seven convolutional layers and two fully connected layers. Max pooling, Dropout, and ReLU are used as well.

## Results
After 9 epochs, the average loss of the dev set is 0.0129, and the classification accuracy of the dev set is 93.2% (4658 / 5000). The following graph shows the accuracy and average loss of the model at each epoch:

![image](https://user-images.githubusercontent.com/26016287/123530289-22265580-d6be-11eb-8fab-7a364a3fe5dd.png)

