# IRDM-Project-2
IRDM Project for Group 26 

**Requirments**
* numpy
* math
* Tensorflow 0.12

**Python Files - All files are run on folder 1 **
* preprocessing.py - This file takes in the specified train.txt/test.txt/vali.txt file and generates 2 arrays containing the true labels and the feature vector which will be used in the logistic regression file. 
* logistic_regression.py - This file contains our implementation of the logistic regression classifier. It will save an array of the true relevance scores(truelabel.npy) and predicted relevance scores(predlabel.npy) which will be used to calculate NDCG@10 and ERR@10. The best model after the specfied epoch run is saved.
* logistic_regression_hiddenlayer.py - This file contains our implementation of the logistic regression classifier. It will save an array of the true relevance scores(truelabel_hidden.npy) and predicted relevance scores(predlabel_hidden.npy) which will be used to calculate NDCG@10 and ERR@10. The best model after the specfied epoch run is saved.
* ERR.py - This file calculates the ERR@10 scores using the array of true labels and predicted labels calculated in the earlier functions. Calculates the ERR@10 per query id and outputs the average over all the query ids. 
* NDCG.py - This file calculates the NDCG@10 scores using the array of true labels and predicted labels calculated in the earlier functions. Calculates the NDCG@10 per query id and outputs the average over all the query ids. 

**Ranklib**
* Using the Ranklib Library from the Lemur Project, the RankNet, LamdaMART and AdaRank models are run on our data. 
* An example of the command used is 

java -jar RankLib-2.8.jar -train MSLR-WEB10K/Fold1/train.txt -test MSLR-WEB10K/Fold1/test.txt -validate MSLR-WEB10K/Fold1/vali.txt -ranker 1 -metric2t ERR@10 -metric2T ERR@10 -lr 0.03 -layer 1 -node 20 -epoch 50 -save RankNet_Fold1_ERR.txt 

