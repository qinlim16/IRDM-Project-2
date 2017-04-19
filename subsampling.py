def readAndWrite(input_path_DIR, output_path_DIR):
	input = open(input_path_DIR, 'r')
	lines = input.readlines()
	input.close()

	output = open(output_path_DIR, 'w')
	index = 1
	for line in lines:
		if not line:
			break
		if index == 1:
			output.write(line)
		elif (index%10 == 0):
			output.write(line)
		index = index + 1
	output.close()

input_path_DIR = "Fold1/train.txt"
output_path_DIR = "Fold1/restructure_train.txt"
readAndWrite(input_path_DIR, output_path_DIR)