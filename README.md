# Flipkart Object Localization
This is my attempt on the Object Localization problem from the Flipkart Grid challenge.<br>

<h2>Problem Statement</h2>
The problem is an agnostic object detection problem in which only bounding boxes must be predicted without any class labels. Also any pre-trained weights are not allowed.
<br>The training data was provided by Flipkart and consisted of 14k images.<br>
https://dare2compete.com/o/Flipkart-GRiD-Teach-The-Machines-2019-74928 

<h2>Solution</h2>
The solution that was propose to solve the object localization problem consists of a CNN having 4 neurons in the output layer where each corresponds to the 4 output values i.e the bounding boxes coordedinates to be determined. 
Therefore it will be solved as a regression problem.

![Given Image](readme/normal.png?raw=true "Given image")
Given Image
![Bounded Box](readme/bounded.png?raw=true "Bounded Box")
Bounded Box

<h2>Architecture</h2>
The architecture of the CNN is based on the popular VGG-16 architecture commonly used for image classification problems.
The final layer has been modified to output four bounding box coordinates and the activation has changed from softmax to relu.

<h2>Results</h2>
The CNN model was trained on Google Colab for 10 epochs.
It achieved <b>79% accuracy</b> when tested on the test set.<br>
Due to the inavailability of images for training such a model the accuracy is a bit less.<br>
Also perhaps a bit more data augmentation would have helped.
