# IRDM-Project-2
IRDM Project for Group 26 

## Getting Started

### Requirments
* Python 2.7
* numpy
* math
* Tensorflow 0.12

### Dataset
* Download MSLR-WEB10K dataset from https://www.microsoft.com/en-us/research/project/mslr/ 
* Copy the datatest into the 5 folders and run preprocessing.py on each of the 5 folders by changing the _folder_ directiory in the script

## Usage
### Python Files : All files are run on folder 1
* subsampling.py - This file takes in _train.txt , test.txt, vali.txt file_ and performs subsampling to extract 10% of the original dataset. This subsampled data will be used for training RankNet which is to computationally expensive to tune on the full dataset.
* preprocessing.py - This file takes in _train.txt , test.txt, vali.txt file_ and generates 2 arrays containing the true labels and the feature vector for each file which will be used in the logistic regression file. 
* logistic_regression.py - This file contains our implementation of the logistic regression classifier. It will save an array of the true relevance scores(truelabel.npy) and predicted relevance scores(predlabel.npy) which will be used to calculate NDCG@10 and ERR@10. The best model after the specfied epoch run is saved.
* logistic_regression_hiddenlayer.py - This file contains our implementation of the logistic regression classifier. It will save an array of the true relevance scores(truelabel_hidden.npy) and predicted relevance scores(predlabel_hidden.npy) which will be used to calculate NDCG@10 and ERR@10. The best model after the specfied epoch run is saved.
* ERR.py - This file calculates the ERR@10 scores using the array of true labels and predicted labels calculated in the earlier functions. Calculates the ERR@10 per query id and outputs the average over all the query ids. 
* NDCG.py - This file calculates the NDCG@10 scores using the array of true labels and predicted labels calculated in the earlier functions. Calculates the NDCG@10 per query id and outputs the average over all the query ids. 
* MAP.py - This file calculates the MAP scores. Implemented but not used for comparison because only suitable for binary relevance and our dataset contains 5 relevance labels. 

### Ranklib 
* Using the Ranklib Library from the Lemur Project, the RankNet, LamdaMART and AdaRank models are run on our data. 
* An example of the command used is 

java -jar RankLib-2.8.jar -train MSLR-WEB10K/Fold1/train.txt -test MSLR-WEB10K/Fold1/test.txt -validate MSLR-WEB10K/Fold1/vali.txt -ranker 1 -metric2t ERR@10 -metric2T ERR@10 -lr 0.03 -layer 1 -node 20 -epoch 50 -save RankNet_Fold1_ERR.txt 

## Authors 
* **Lim Qin Zhi ucabqzl@ucl.ac.uk**
* **ZhaoFeng Jin zhaofeng.jin.16@ucl.ac.uk**
* **Xin Wei ucakxwe@ucl.ac.uk**
* **Jinhang Zhang ucakjjz@ucl.ac.uk**
