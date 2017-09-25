import linecache
output_file = open('output.txt','r')
acc_file = open('accuracy.txt','a')
i = 1
for line in output_file:
	if '()' in line:
		acc_file.write('Image: '+str(linecache.getline('testImagePath.txt', i)))
		i = i + 1
	if '(1)' in line:
		line_list = line.split(' ')
		acc_file.write('Class 1 (Female without glasses): '+line_list[8])
	elif '(2)' in line:
		line_list = line.split(' ')
		acc_file.write('Class 2 (Female with glasses): '+line_list[8])

	elif '(3)' in line:
		line_list = line.split(' ')
		acc_file.write('Class 3 (Male without glasses): '+line_list[8])

	elif '(0)' in line:
		line_list = line.split(' ')
		acc_file.write('Class 0 (Male with glasses): '+line_list[8])