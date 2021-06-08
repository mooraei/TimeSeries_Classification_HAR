# Time Series Classification for HAR

### Abestract
you can download data here 
[Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

If you want use machine learning models such ensemble and decision tree, you need to strong feature
enginnering and need expertise in the field. 

But recent research shown that you can use deep learning method and don't ever need those feature
enginnering. 

Here i want implement a recurrent neural networks (LSTMs). Instead of feature enginnering, i using feature
learning on raw data. Approximate accuracy is about 90%.

##

### Data 
You can see data in detail on uci page but in brief:

The six activity performed:

* Walking
* Walking Upstairs
* Walking Downstairs
* Sitting
* Standing
* Laying

The movement data was the x, y, z accelerometer data and gyroscopic data.

Raw data is not available. Instead, a pre-processed version of dataset was made available.
The result was a 561 element vector of features.

