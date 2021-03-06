# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


You're reading it! and here is a link to my [project code](https://github.com/DongzheWu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1..

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. 

Here is an exploratory visualization of the data set. 
The pictures are randomly selected from the train examples.
The bar chart shows the number of train examples for each class.

![image](./images/bar_chart1.jpg)

![image](./images/train_example.jpg)

### Design and Test a Model Architecture

#### 1. 

As a first step, I decided to generate additional data because I realized that the number of traffic examples for some classes is not enough. I added 1000 examples to the classes which have no more than examples. 
I used rotation and blur to generate train examples. Rotation can simulate the pictures took from different angles to add the diversity of train data. I also found that there are some pictures unclear in the train examples, so I think using blur can add more unclear pictures to train the neural network better.


![image](./images/Xtrain1136.jpg)
![image](./images/blur.jpg)
![image](./images/rotation.jpg)


Then I converted all images to grayscale, because I got this idea from the paper Traffic Sign Recognition with Multi-Scale Convolutional Networks.
After implementing grayscale images, the network was trained better.
Here is an example of a traffic sign image before and after grayscaling.


![image](./images/Xtrain1136.jpg)
![image](./images/gray.jpg)


#### 2. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	(x1)    | 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling (x2)	    | 2x2 stride,  outputs 5x5x64 					|
| flatten(x1)			| outputs 6272 									|
| flatten(x2)			| outputs 1600 									|
| concat(x1 + x2)		| outputs 7872 									|
| dropout				| Keep probability 0.8							|
| Fully connected		| outputs 300  									|
| RELU					|												|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 43  									|
 


#### 3. 

To train the model, I set the number of epochs to 30, batch size to 128, learning rate to 0.001 and I used Adam optimizer.

#### 4. 
My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 95.1%
* test set accuracy of 92.5%

For the first architecture, I just copied LeNet from last exercise. The validation accuracy was only 89%. Then I converted images to grayscale, the validation accuracy was improved to 90%. And I changed the number of filters of first layer from 6 to 108 and the number of filters of second layer from 16 to 108. The validation accuracy was 92%. After that, I modified LeNet to multi-scale CNN, concatenated the output of first layer and second layer. However, the validation accuracy was very low and it seemed it was overfitting because too many parameters. Therefore, I reduced the number of filters of first layer and second layer to 32 and 64. I also added dropout to prevent overfitting and generated more images by rotation and blur to increase diversity of training data. Finally, the final model worked well. The validation accuracy was 95.1%.


 
### Test a Model on New Images

#### 1. 

Here are five German traffic signs that I found on the web:

![image](./test_img/testimages.jpg) 

The sixth, seventh and eighth images might be difficult to classify because sixth and eighth images are took by oblique perspective. The seventh image is not very clear inside.

#### 2. 

Here are the results of the prediction:

| Image			              |     Prediction	        				| 
|:---------------------:|:--------------------------:| 
| Speed limit(50km/h) 	 | Speed limit(50km/h)  						| 
| Road work    	      		| Road work 							        		|
| Priority road		       | Priority road					     				|
| Yield	      		       	| Yield					 						          |
| Keep right		  	       | Keep right      		   						|
| Stop				             	| Stop      							        		|
| Slippery Road		      	| Speed limit(30km/h)     			|
| Children crossing	   	| Priority road      						  |

The model was able to correctly predict 6 of the 8 traffic signs, which gives an accuracy of 75%. The accuracy is lower than the accuracy on the test set of 92.5%. However, there are only 8 images. The Children crossing sign wasn't detected correctly by the model, because the picture wasn't took by front view. I think if I warp images by using what I learned in the previous project to generate more training data, the model will work better. Moreover, Slippery Road was detected as Speed limit(30km/h). It seems that the model is not good at recognize the signs which have blur pattern inside. I think I can try to change the number of layers and the number of filters, maybe I can find another solution.  

#### 3. 
The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.
---
First image

Speed limit (50km/h):

Prediction and Probability

Speed limit (50km/h) 100.00%

Speed limit (30km/h) 0.00%

Speed limit (60km/h) 0.00%

---
Second image

Road work:

Prediction and Probability

Road work 100.00%

Speed limit (20km/h) 0.00%

Speed limit (30km/h) 0.00%

---
Third image

Priority road:

Prediction and Probability

Priority road 100.00%

Keep right 0.00%

Roundabout mandatory 0.00%

---
Fourth image 

Yield:

Prediction and Probability

Yield 100.00%

Speed limit (20km/h) 0.00%

Speed limit (30km/h) 0.00%

---
Fifth image

Keep right:

Prediction and Probability

Keep right 100.00%

Speed limit (20km/h) 0.00%

Speed limit (30km/h) 0.00%

---
Sixth image

Stop:

Prediction and Probability

Stop 100.00%

Keep right 0.00%

Turn left ahead 0.00%

---
Seventh image

Slippery road:

Prediction and Probability

Speed limit (30km/h) 99.99%

Road work 0.01%

Speed limit (50km/h) 0.00%

---
Eighth image

Children crossing:

Prediction and Probability

Priority road 66.59%

Bicycles crossing 29.24%

Keep right 2.37%



