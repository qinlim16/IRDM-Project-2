# IRDM-Project-2
IRDM Project for Group 26 

Requirments
* numpy
* math
* Tensorflow 0.12

Python Files - All files are run on folder 1 
* preprocessing.py - This file takes in the specified train.txt/test.txt/vali.txt file and generates 2 arrays containing the true labels and the feature vector which will be used in the logistic regression file. 
* logistic_regression.py - This file contains our implementation of the logistic regression classifier. It will save an array of the true relevance scores(truelabel.npy) and predicted relevance scores(predlabel.npy) which will be used to calculate NDCG@10 and ERR@10. The best model after the specfied epoch run is saved.
* logistic_regression_hiddenlayer.py - This file contains our implementation of the logistic regression classifier. It will save an array of the true relevance scores(truelabel_hidden.npy) and predicted relevance scores(predlabel_hidden.npy) which will be used to calculate NDCG@10 and ERR@10. The best model after the specfied epoch run is saved.
* ERR.py - This file calculates the ERR@10 scores using the array of true labels and predicted labels calculated in the earlier functions. Calculates the ERR@10 per query id and outputs the average over all the query ids. 
* NDCG.py - This file calculates the NDCG@10 scores using the array of true labels and predicted labels calculated in the earlier functions. Calculates the NDCG@10 per query id and outputs the average over all the query ids. 
