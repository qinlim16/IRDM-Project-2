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
	# input = open(input_path_DIR, 'r')
	# lines = input.readlines()
	# input.close()
	# z = []
	# lines=lines[1:]
	# for line in lines:
	# 	step1 = line.split(" ")
	# 	z.append(float(step1[1].split(":")[1]))
	# return z

	


input_path_DIR = "MSLR-WEB10K-2/Fold1/test.txt"
z = readSplit(input_path_DIR)
trueR = np.load("truelabel.npy").tolist()
predR = np.load("predlabel.npy").tolist()

set = set(z)
result = list(set)

total=[]
pre_query=[]
pre_res=[]
index=-1
print len(trueR)
print len(predR)
print len(z)
for num in z:
	index+=1
	total.append((num,trueR[index],predR[index]))

for num in result:
	pre=[]
	pre_query=[]
	for (i,t,p) in total:
		if i==num:
			pre_query.append((i,t,p))
	# pre_query.sort(key = lambda x:x[1])
	pre_query = sorted(pre_query, key=lambda x:x[1], reverse=True)

	length=0
	precision=0
	for (i,t,p) in pre_query:
		length+=1
		if t==p:
			precision+=1
			pre.append(precision/length)
			# print(precision/length)
	if len(pre)!=0:
		pre_res.append(sum(pre)/len(pre))
		# print sum(pre)/len(pre)
	else:
		pre_res.append(0)

print sum(pre_res)/len(pre_res)

