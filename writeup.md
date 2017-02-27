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
[image2]: ./reports/vis1.png "Histogram of types of signs in training/validation data"
[image3]: ./reports/vis2.png "Histogram of types of signs in training/testing data"
[image4]: ./reports/prep0.png "Apply gray and normalization"
[image5]: ./reports/prep1.png "Apply translation ([10,-10]), rotate (5 degrees) and scaling (x1.1)"
[image6]: ./reports/image0.png "Beware of ice/snow"
[image7]: ./reports/image1.png "Speed limit (60km/h)"
[image8]: ./reports/image2.png "Wild animals crossing"
[image9]: ./reports/image3.png "Speed limit (50km/h)"
[image10]: ./reports/image4.png "No passing"
[image11]: ./reports/test0.png "Speed limit (70km/h)"
[image12]: ./reports/test1.png "Dangerous curve to the right"
[image13]: ./reports/test2.png "Speed limit (120km/h)"
[image14]: ./reports/test3.png "Traffic signals"
[image15]: ./reports/test4.png "Beware of ice/snow"
[image16]: ./reports/softmax0.png "Softmax on downloaded image 0"
[image17]: ./reports/softmax1.png "Softmax on failed test image 0"

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

I also plotted a histogram of signs among train vs validation and train vs test data. The signs do not appear evenly in the datasets. For example, the speed limits/passing related signs appear more frequently than warning signs. 
![training/validation][image2]
![training/testing][image3]

It is also noticeable that train/validation/test seem to have a similar distribution of types of signs.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because grayscale should sufficiently capture images features and make the network easier to train. I also added a normalization step: contrast-limited adaptive histogram equalization to make the pixel value distibution more uniform while not overamplifying contrast. 

Here is an example of a traffic sign image before, after grayscale, after both grayscale and clahe.
![grayscale and norm][image4]

I also add a random translation (up to +/-10%), rotation (up to +/- 5 degrees) and scaling (0.9 to 1.1). Here is a before and after comparison of [+3, -3] pixel translation, followed by 5 degree rotation and a scaling of 1.1.
![translation/rotation/scaling][image5]


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

* training set accuracy of 99.7%
* validation set accuracy of 95.9%
* test set accuracy of 94.8%

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

Batch size, learning rate, epochs. Batch size was tuned based on validation accuracy and limiting spread between training and validation accuracy (finally set at 64). Similarly learning rate (0.0005). Epochs were tuned based on when validation accuracy gets stuck (30).

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

Here are the results of the prediction on the 5 downloaded images:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Beware of ice/snow      		| Beware of ice/snow   									| 
| Speed limit (60km/h)     			| Speed limit (60km/h) 										|
| Wild animals crossing					| Wild animals crossing											|
| Speed limit (50km/h)	      		| Speed limit (50km/h)					 				|
| No passing			|  No passing      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

I went ahead to pick five failed cases from the test set:

![alt text][image11] ![alt text][image12] ![alt text][image13] 
![alt text][image14] ![alt text][image15]

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (120km/h)   									| 
| Dangerous curve to the right    			| Beware of ice/snow 										|
|Speed limit (120km/h)					| Speed limit (80km/h)											|
| Traffic signals	      		| Pedestrians					 				|
| Beware of ice/snow			|  Slippery road      							|

These images looks challenging: the first one was heavily shaded in a stripe pattern.  The second sign (dangerous curve) looks similar to beware of ice/snow sign, especially when tilted. The third sign was rotated 45 degrees. The fourth sign (traffic light) is close to pedestrian sign when grayscaled (in this case, inputting all three channels to the network might help). The last sign is of very poor visual quality.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the five downloaded images, the model achieves softmax on the predicted sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000         			| Beware of ice/snow   									| 
| 1.00000     				| Speed limit (60km/h)  										|
| 1.00000					| Wild animals crossing											|
| 1.00000	      			| Speed limit (50km/h)					 				|
| 1.00000				    | No passing      							|

![alt text][image16]

```
Predict confidence logits 41.04348, softmax 1.00000 - sign 30: Beware of ice/snow

Predict confidence logits 21.30332, softmax 0.00000 - sign 11: Right-of-way at the next intersection

Predict confidence logits 19.42945, softmax 0.00000 - sign 29: Bicycles crossing

Predict confidence logits 11.77894, softmax 0.00000 - sign 23: Slippery road

Predict confidence logits 11.05733, softmax 0.00000 - sign 28: Children crossing



Predict confidence logits 58.46395, softmax 1.00000 - sign 3: Speed limit (60km/h)

Predict confidence logits 44.31151, softmax 0.00000 - sign 5: Speed limit (80km/h)

Predict confidence logits 33.52377, softmax 0.00000 - sign 2: Speed limit (50km/h)

Predict confidence logits 7.01293, softmax 0.00000 - sign 1: Speed limit (30km/h)

Predict confidence logits 0.50415, softmax 0.00000 - sign 10: No passing for vehicles over 3.5 metric tons



Predict confidence logits 28.40719, softmax 1.00000 - sign 31: Wild animals crossing

Predict confidence logits 8.40440, softmax 0.00000 - sign 21: Double curve

Predict confidence logits 8.12611, softmax 0.00000 - sign 29: Bicycles crossing

Predict confidence logits 7.33135, softmax 0.00000 - sign 23: Slippery road

Predict confidence logits -0.39528, softmax 0.00000 - sign 5: Speed limit (80km/h)



Predict confidence logits 62.63970, softmax 1.00000 - sign 2: Speed limit (50km/h)

Predict confidence logits 34.13874, softmax 0.00000 - sign 5: Speed limit (80km/h)

Predict confidence logits 33.97448, softmax 0.00000 - sign 3: Speed limit (60km/h)

Predict confidence logits 25.59088, softmax 0.00000 - sign 1: Speed limit (30km/h)

Predict confidence logits 10.76384, softmax 0.00000 - sign 4: Speed limit (70km/h)



Predict confidence logits 51.35054, softmax 1.00000 - sign 9: No passing

Predict confidence logits 32.86082, softmax 0.00000 - sign 10: No passing for vehicles over 3.5 metric tons

Predict confidence logits 14.30334, softmax 0.00000 - sign 16: Vehicles over 3.5 metric tons prohibited

Predict confidence logits 4.12077, softmax 0.00000 - sign 15: No vehicles

Predict confidence logits -0.32799, softmax 0.00000 - sign 3: Speed limit (60km/h)
```

For the five failed images, the model seems to be sure about its predictions although they seem to be wrong. It clearly overfits. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000         			| Speed limit (120km/h)   									| 
| 1.00000     				| Beware of ice/snow  										|
| 1.00000					| Speed limit (80km/h)											|
| 1.00000	      			| Pedestrians					 				|
| 1.00000				    | Slippery road      							|

![alt text][image17]

```
Predict confidence logits 15.86540, softmax 0.95740 - sign 8: Speed limit (120km/h)

Predict confidence logits 12.71894, softmax 0.04117 - sign 4: Speed limit (70km/h)

Predict confidence logits 8.65446, softmax 0.00071 - sign 40: Roundabout mandatory

Predict confidence logits 8.03615, softmax 0.00038 - sign 1: Speed limit (30km/h)

Predict confidence logits 7.91193, softmax 0.00034 - sign 14: Stop



Predict confidence logits 15.47153, softmax 0.44538 - sign 30: Beware of ice/snow

Predict confidence logits 15.24166, softmax 0.35392 - sign 28: Children crossing

Predict confidence logits 14.52717, softmax 0.17322 - sign 20: Dangerous curve to the right

Predict confidence logits 12.62212, softmax 0.02578 - sign 25: Road work

Predict confidence logits 9.90527, softmax 0.00170 - sign 29: Bicycles crossing



Predict confidence logits 20.39185, softmax 0.96184 - sign 5: Speed limit (80km/h)

Predict confidence logits 16.72698, softmax 0.02463 - sign 3: Speed limit (60km/h)

Predict confidence logits 15.73135, softmax 0.00910 - sign 8: Speed limit (120km/h)

Predict confidence logits 15.01068, softmax 0.00443 - sign 7: Speed limit (100km/h)

Predict confidence logits 1.01626, softmax 0.00000 - sign 2: Speed limit (50km/h)



Predict confidence logits 13.80245, softmax 0.60577 - sign 27: Pedestrians

Predict confidence logits 13.37121, softmax 0.39357 - sign 26: Traffic signals

Predict confidence logits 6.63034, softmax 0.00047 - sign 24: Road narrows on the right

Predict confidence logits 5.69351, softmax 0.00018 - sign 28: Children crossing

Predict confidence logits 2.39223, softmax 0.00001 - sign 11: Right-of-way at the next intersection



Predict confidence logits 9.59632, softmax 0.99831 - sign 23: Slippery road

Predict confidence logits 2.45944, softmax 0.00079 - sign 10: No passing for vehicles over 3.5 metric tons

Predict confidence logits 1.93290, softmax 0.00047 - sign 30: Beware of ice/snow

Predict confidence logits 1.40592, softmax 0.00028 - sign 31: Wild animals crossing

Predict confidence logits 0.76817, softmax 0.00015 - sign 19: Dangerous curve to the left
``` 