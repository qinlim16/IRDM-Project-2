from __future__ import division  
import numpy as np


def readSplit(input_path_DIR):
	input = open(input_path_DIR, 'r')
	lines = input.readlines()
	input.close()
	z = []
	for line in lines:
		step1 = line.split(" ")
		if (step1[0] == '\n'):
			break
		else:
			z.append(float(step1[1].split(":")[1]))
	return z


folder = 'MSLR-WEB10K/Fold5/'

input_path_DIR = folder+"test.txt"
z = readSplit(input_path_DIR)
trueR = np.load(folder+"truelabel.npy").tolist()
predR = np.load(folder+"predlabel.npy").tolist()

set = set(z)
result = list(set)

total=[]
pre_query=[]
index=-1

probability={0:0, 1:1/16,2:3/16,3:7/16,4:15/16}

for num in z:
	index+=1
	total.append((num,trueR[index],predR[index]))

err=0
for num in result:
	pre=[]
	pre_query=[]
	for (i,t,p) in total:
		if i==num:
			pre_query.append((i,t,p))
	pre_query = sorted(pre_query, key=lambda x:x[1], reverse=True)

	p_stop=1	
	if(len(pre_query)>9):
		for i in range(10):
			prob=probability[pre_query[i][2]]
			p_stop2=prob*p_stop
			p_stop=p_stop*(1-prob)
			err += p_stop2/(i+1)
	else:
		for i in range(len(pre_query)):
			prob=probability[pre_query[i][2]]
			p_stop2=prob*p_stop
			p_stop=p_stop*(1-prob)
			err += p_stop2/(i+1)
print ("Final ERR")
print err/len(result)

