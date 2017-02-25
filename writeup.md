#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./reports/vis0.png "An example sign image"
[image2]: ./reports/vis1.png "Histogram of types of signs in training data"
[image3]: ./reports/vis2.png "Histogram of types of signs in validation data"
[image4]: ./reports/vis3.png "Histogram of types of signs in test data"
[image5]: ./reports/prep0.png "Apply gray and normalization"
[image6]: ./reports/image0.png "Beware of ice/snow"
[image7]: ./reports/image1.png "Speed limit (60km/h)"
[image8]: ./reports/image2.png "Wild animals crossing"
[image9]: ./reports/image3.png "Speed limit (50km/h)"
[image10]: ./reports/image4.png "No passing"

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

The following is an example sign image, title is sign id and meaning read from csv file.

![An example sign image][image1]

I also plotted a histogram of signs among train, validation and test data. The signs do not appear evenly in the datasets. For example, the speed limits/passing related signs appear more frequently than warning signs. 
![training][image2]
![validation][image3]
![test][image4]

It is also noticeable that train/validation/test seem to have a similar distribution of types of signs.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because grayscale should sufficiently capture images features and make the network easier to train. I also added a normalization step to map pixels from [min, max] to [0, 1], making pixel distribution more uniform across the entire datasets.

Here is an example of a traffic sign image before, after grayscale, after both grayscale and norm.
![image preprocessing][image5]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I directly used the train and validation data from 'traffic-signs-data' directory.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Concat                | concat two pooling, outputs 5x5x32+14x14x10=2760|
| Fully connected		| 2760->300, dropout =0.5						|
| RELU                  |                                               |
| Fully connected		| 300->43						|
| Softmax				|        									|
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an Adam optimizer which comes with learning rate decay. The loss function that I used is soft max cross entropy.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

* training set accuracy of 99.9%
* validation set accuracy of 94.6%
* test set accuracy of 93.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

Initially I used the LeNet with some modifications because it already worked well on digit classification.

* What were some problems with the initial architecture?

Validation accuracy was low, mainly because the image features are more complicated, more types of classes (43 compared to 10).

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

There are many things I tested. 

1. Modify model architecture. Influenced by [Sermanet and LeCun's paper] (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I concatenated pooling results from first and second convolution layer to generate a new feature vector for the purpose of capturing features of different granularity.

2. Add dropout to fully connected layers, this further improved validation accuracy and made the network more robust.

3. Tune hyper parameters (to be discussed in next question)

* Which parameters were tuned? How were they adjusted and why?

Batch size, learning rate, epochs. Batch size was tuned based on validation accuracy (finally set at 30). Similarly learning rate (0.0003). Epochs were tuned based on when validation accuracy gets stuck (20).

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Use two level convolution layers to capture images features of various scale and concatenate pooling output from those. Dropout helps in preventing overfit and significantly improve validation accuracy and robustness of the model.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The first image seems hard to classify because of low resolution: I only see a blob in the middle and in this case we need the network to learn and shine on high level large scale featues.

The third image is also hard because of seemingly graffiti on top of the "5" figure.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Beware of ice/snow      		| Beware of ice/snow   									| 
| Speed limit (60km/h)     			| Speed limit (60km/h) 										|
| Wild animals crossing					| Wild animals crossing											|
| Speed limit (50km/h)	      		| Speed limit (50km/h)					 				|
| No passing			|  No passing      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .989         			| Beware of ice/snow   									| 
| .999     				| Speed limit (60km/h)  										|
| 1.000					| Wild animals crossing											|
| 1.000	      			| Speed limit (50km/h)					 				|
| 1.000				    | No passing      							|


For the second image ... 