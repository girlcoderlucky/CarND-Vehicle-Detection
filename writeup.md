---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar_example.png
[image2]: ./output_images/hog_example.png
[image3]: ./output_images/raw_normalized_features_example.png
[image4]: ./output_images/test6_sliding_window.png
[image5]: ./output_images/test6_heatmap.png
[image6]: ./output_images/test6_final_output.png
[video1]: ./project_video.mp4

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook (Vehicle_Detection.ipynb)
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Example : "./output_images/car_notcar_example.png"

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Example : "./output_images/hog_example.png"

####2. Explain how you settled on your final choice of HOG parameters.

I tried training with HOG features from single channel and also using HOG features from all the channels. hog_channel = 'ALL' gave best accuracy so stuck with it.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using using all the `vehicle` and `non-vehicle` images. You can find the function "train_model()" in Vehicle_Detection.ipynb.
The function single_img_features() extracts features from a single image. The features extracted are HOG, spatial bins and color histogram.
The function extract_features() will extract all the features from the vehicle/ non vehicle data. Then trained the SVC using all the features extracted and save the model so that I do not have train it again and again. 
The accuracy was better when I used HOG features from all the channel around 98%

50.91 Seconds to train SVC...
Test Accuracy of SVC =  0.9786
My SVC predicts    :  [ 0.  1.  1.  1.  0.  1.  0.  0.  0.  1.]
For these 10 labels:  [ 0.  1.  1.  1.  0.  1.  0.  0.  0.  1.]
0.04799 Seconds to predict 10 labels with SVC

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

See the function slide_window(), Sliding Window function that takes an image, start and stop positions in both x and y, window size (x and y dimensions), and overlap fraction (for both x and y)
I am using windows of multiple sizes 64x64, 96x96 and 128x128 windows. Also y_start_stop=[400, 720]. I wanted use window size of 32x32 for better car detection but I stopped using it to save computation time.

Example : "./output_images/test6_sliding_window.png"

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using ALL HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  
Trained the SVC using all the features extracted and save the model so that I do not have train it again and again. I wanted use window size of 32x32 for better car detection but I stopped using it to save computation time.

Here are example images:
"./output_images/test6_sliding_window.png"
"./output_images/test6_heatmap.png"
"./output_images/test6_final_output.png"
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a link to my video result(./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded 2 that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
Initially to test the pipeline and extracted the frames from the test_video, ran my pipeline on them using heatmap to detect false positives and combine multiple detections. 
The results can be seen in folder "./output_images/heatmap_frames/frames*"

### Here is an example of heatmap: "./output_images/test6_heatmap.png"

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Heatmap reduced the false positives but still I see very few false positive in the video output. I would like improve the pipeline to remove false positives 100%.
My pipeline is not detecting small cars in the frame because I am not using 32x32 in my multi-scale window sizes due to computation limitation.
Performance of of my current pipeline is ~2secs/frame on my laptop.

Improvements : 
First, I want to improve the accuracy and performance of the pipeline. 
Second, want to try deep learning network to classify images instead of SVM. 
Third, capablity to detect other objects like pedestrians, trafic light.
 