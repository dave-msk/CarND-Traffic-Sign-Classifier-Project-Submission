#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/visualization.png "Visualization"
[image2]: ./writeup/exploration.jpg "Exploration"
[image3]: ./writeup/1.jpg "Traffic Sign 1"
[image4]: ./writeup/12.jpeg "Traffic Sign 2"
[image5]: ./writeup/13.jpeg "Traffic Sign 3"
[image6]: ./writeup/18.jpeg "Traffic Sign 4"
[image7]: ./writeup/22.jpeg "Traffic Sign 5"
[image8]: ./writeup/40.jpg "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed. 

![alt text][image1]

Ten images for each class are also displayed for illustration purpose.

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

I decided to scale the pixels values of images to [0.0, 1.0] to enhance numerical stability of the trained weights, as large input values would result in small weights which is susceptible to floating point underflow error.

As the color of the sign contains clues about its type (such as blue for "mandatory"), I avoided converting the images to grayscale to preserve such information.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The training, validation and testing data are loaded directly from the provided pickled dataset files.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5 		| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Max pooling 2x2		| 2x2 stride,  outputs 16x16x16 				|
| Convolution 5x5 		| 1x1 stride, same padding, outputs 16x16x64	|
| RELU					|												|
| Max pooling 2x2		| 2x2 stride,  outputs 8x8x64					|
| Convolution 5x5 		| 1x1 stride, same padding, outputs 8x8x128		|
| RELU					|												|
| Max pooling 2x2		| 2x2 stride,  outputs 4x4x128					|
| Flatten 				| inputs 4x4x64,  outputs 2048					|
| Fully connected 		| inputs 2048,  outputs 512						|
| Activation 			| tanh 											|
| Dropout 				| keep probability = 0.5 						|
| Fully connected 		| inputs 512,  outputs 128 						|
| Activation 			| tanh 											|
| Dropout 				| keep probability = 0.5 						|
| Fully connected 		| inputs 128,  outputs 43 						|
| Softmax				| output layer									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh to tenth cells of the ipython notebook. 

To train the model, I used the AdamOptimizer with learning rate 0.0001. The batch size is set to 128, and the training is to be run for 400 epochs. The validation accuracy is evaluate for each epoch. The early-stopping approach is employed to get the best generalizing model until the recent epoch according to the measure given by the validation accuracy.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 95.9%
* test set accuracy of 95.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The architecture is a modification of LeNet. It was pretty much the same through out all the trials, the only difference was the use of dropout layers and hyperparameter configurations. It was chosen as it has been shown to have a satisfying performance in object recognition, and my personal experience suggests the same.

* What were some problems with the initial architecture?
As the architecture consists of a huge amount (in the order of million) of trainable parameters, it was prone to overfitting in the initial architecture as training progresses.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
The initial architecture contains no dropout layers. As a noticeable overfitting behaviour was observed, two dropout layers were added to the architecture.

* Which parameters were tuned? How were they adjusted and why?
The weight-keeping probabilities were set to (0.7, 0.5) initially. The validation performance was improved to have accuracy reaching 0.94. As the performance was still far from satisfying, a more aggressive weight-keeping probability configuration (0.5, 0.5) was introduced. The validation accuracy raised to about 96%.
The number of epochs was also tuned according to validation accuracies. Typically the model can reach 85% in the first ten epochs of training. The accuracy would generally raise to about 92% to 94% in the next 100 epochs. Then, the next 200 epochs would introduce an extra 1%-2% to the model. After then, no significant improvement was observed. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Firstly, the convolutional layer provides translation invariance of features, which is a natural requirement for object recognition. For example, a corner is still a corner no matter where it is place in the image.
Next, the dropout layers are crucial to mitigate overfitting as overfitting itself is almost guaranteed to appear in neural networks due to its super high dimensionalty.
For the activation functions, due to its simplicity, ReLU gives a relatively good balance between performance and computational complexity. However, the exploration power provided by ReLU is limited due to its partial-linearity. In the situations where non-linearity is required to express more complex internal properties of the input, ReLU may not be a good choice. One good choice for fulfilling such non-linearity requirement would be the hyperbolic tangent function, which has been shown to have great performance and is used intensively in a wide range of architectures (including CNN and RNN).
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8]

The first image might be difficult to classify due to limited resolution of characters in the middle after resizing to 32x32.

The color of the outer square is not quite the same as those in the training data, which might cause some confusions to the network.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 12th and 13th cells of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution 		| General Caution								| 
| Yield 				| Yield 										|
| 30 km/h				| 30km/h										|
| Bumpy road			| Bumpy Road					 				|
| Roundabout mandatory	| Roundabout mandatory							|
| Priority road			| Priority road									|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.4%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is almost sure that this is a general caution sign (probability of 1.00), and the image does contain a general caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| General Caution 								| 
| 0.00     				| Pedestrians 									|
| 0.00					| Traffic signals								|
| 0.00	      			| 70 km/h					 					|
| 0.00				    | Go straight or left							|


For the second image, the model is almost sure that this is a yield sign (probability of 1.00), and the image does contain a yield sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield 										| 
| 0.00     				| 100 km/h 										|
| 0.00					| Ahead only									|
| 0.00	      			| No vehicles					 				|
| 0.00				    | 50km/h										|


For the third image, the model is almost sure that this is a speed limit (30 km/h) sign (probability of 1.00), and the image does contain a speed limit (30 km/h). The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 30 km/h 										| 
| 0.00     				| End of speed limit (80km/h) 					|
| 0.00					| 20 km/h										|
| 0.00	      			| 50 km/h					 					|
| 0.00				    | 80 km/h										|


For the fourth image, the model is almost sure that this is a speed limit (30 km/h) sign (probability of 1.00), and the image does contain a speed limit (30 km/h). The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Bumpy road 									| 
| 0.00     				| Bicycles crossing 							|
| 0.00					| Children crossing								|
| 0.00	      			| Dangerous curve to the right					|
| 0.00				    | 120 km/h										|


For the fifth image, the model is relatively sure that this is a roundabout mandatory sign (probability of 0.65), and the image does contain a roundabout mandatory sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.65         			| Roundabout mandatory 							| 
| 0.35     				| Keep right 									|
| 0.00					| End of speed limit (80km/h)					|
| 0.00	      			| Go straight or left							|
| 0.00				    | 20 km/h										|


For the sixth image, the model is almost sure that this is a priority road sign (probability of 1.00), and the image does contain a priority road sign. The top five soft max probabilities were
| Probability         	|     Prediction	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| 1.00         			| Priority road 										| 
| 0.00     				| Traffic signals 										|
| 0.00					| 100 km/h												|
| 0.00	      			| End of no passing by vehicles over 3.5 metric tons	|
| 0.00				    | Right-of-way at the next intersection					|