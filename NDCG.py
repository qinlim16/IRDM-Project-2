import numpy as np
import math
np.set_printoptions(threshold = np.nan)

folder = 'MSLR-WEB10K/Fold1/'

ltrue = np.load(folder + "truelabel_hidden.npy")
lpred = np.load(folder + "predlabel_hidden.npy")
def getQid(testfile):
	input = open(testfile, 'r')
	lines = input.readlines()
	input.close()
	qid = []
	for line in lines:
		step1 = line.split(" ")
		for obj in step1:
			step2 = obj.split(":")
			if step2[0] == 'qid':
				qid.append(int(step2[1]))
			if (step2[0] == '\r\n'):
				break
	unique_qid, count = np.unique(qid, return_counts = True)
	return unique_qid, count

def DCG(pred_list, true_list, k):
	order_pred = np.argsort(pred_list)[::-1]
	order_true_list = np.take(true_list, order_pred[:k])

	gains = np.asarray(order_true_list[:k])
	gains = 2 ** gains - 1
	discounts = np.log2(np.arange(len(order_true_list[:k])) + 2)	
	# print np.sum(gains/discounts)
	return np.sum(gains/discounts)

def IDCG(true_list, k):
	order = np.argsort(true_list)[::-1]
	order_true_list = np.take(true_list, order[:k])

	gains = np.asarray(order_true_list[:k])
	gains = 2 ** gains - 1
	discounts = np.log2(np.arange(len(true_list[:k])) + 2)	
	# print np.sum(gains/discounts)
	return np.sum(gains/discounts)

def NDCG(pred_list, true_list, k):
	dcg = DCG(pred_list, true_list, k)
	idcg = IDCG(true_list, k)
	result = dcg/idcg
	if math.isnan(result):
		result = 0
	#print result
	return result

if __name__ == '__main__':
	testfile = folder+"test.txt"
	unique_qid, count = getQid(testfile)
	pair = []
	for ind in range(0, len(unique_qid)):
		pair.append([unique_qid[ind], count[ind]])

	i = 0
	ndcg_result = []
	for obj in pair:
		unique = obj[0]
		count = obj[1]
		if count > 10:
			k = 10
		else:
			k = count
		pred_list = lpred[i:(i+count)]
		true_list = ltrue[i:(i+count)]
		ndcg_result.append(NDCG(pred_list, true_list, k))
		i = i + count
	# print ndcg_result
	final_avg = sum(ndcg_result)/len(ndcg_result)
	print ("Overall NDCG")
	print final_avg
	# print NDCG([3,2,3,0,1,2], 6)
