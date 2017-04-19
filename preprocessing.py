import numpy as np

def readSplit(input_path_DIR):
	input = open(input_path_DIR, 'r')
	lines = input.readlines()
	input.close()
	x = []
	y = []
	for line in lines:
		step1 = line.split(" ")
		y.append(float(step1[0]))
		index = 0
		for obj in step1:
			if index < 2:
				index = index + 1
				continue
			step2 = obj.split(":")
			if step2[0] == "\r\n":
				continue
			step2[1] = float(step2[1])
			x.append(step2[1])
			index = index + 1
	return np.array(x), np.array(y)

def convertToOne(y):
	init = np.zeros([y.size, 5])
	counter = 0
	for ind in np.nditer(y):
		ind = int(ind)
		init[counter, ind] = 1
		counter = counter + 1	
	return init

folder = 'MSLR-WEB10K/Fold5/'

input_path_DIR = folder + "train.txt"
[x, y] = readSplit(input_path_DIR)
new_y = convertToOne(y)
output_x_path_DIR = folder + "trainx.npy"
output_y_path_DIR = folder + "trainy.npy"
np.save(output_x_path_DIR, np.reshape(x,[-1,136]))
np.save(output_y_path_DIR, new_y)

input_path_DIR2 = folder + "train.txt"
[x2, y2] = readSplit(input_path_DIR2)
new_y2 = convertToOne(y2)
output_x_path_DIR2 = folder + "trainx.npy"
output_y_path_DIR2 = folder + "trainy.npy"
np.save(output_x_path_DIR2, np.reshape(x2,[-1,136]))
np.save(output_y_path_DIR2, new_y2)

input_path_DIR3 = folder + "train.txt"
[x3, y3] = readSplit(input_path_DIR3)
new_y3 = convertToOne(y3)
output_x_path_DIR3 = folder + "trainx.npy"
output_y_path_DIR3 = folder + "trainy.npy"
np.save(output_x_path_DIR3, np.reshape(x3,[-1,136]))
np.save(output_y_path_DIR3, new_y3)

# a = np.load("trainx.npy")
# print a.shape

# b = np.load("trainy.npy")
# print b.shape
