import os

os.chdir('path/to/testset')

testing_file = open('testing.txt','r')
for line in testing_file:
	line_list = line.split(' ')
	LabelFile = open('imageLabel.txt','a')
	if line_list[11] == '1' and line_list[13] == '1':
		LabelFile.write('Class 0 (Male with glasses)'+'\n')
	elif line_list[11] == '1' and line_list[13] == '2':
		LabelFile.write('Class 3 (Male without glasses)'+'\n')
	elif line_list[11] == '2' and line_list[13] == '1':
		LabelFile.write('Class 2 (Female with glasses)'+'\n')
	elif line_list[11] == '2' and line_list[13] == '2':
		LabelFile.write('Class 1 (Female without glasses)'+'\n')


LabelFile.close()