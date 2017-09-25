import os

os.chdir('path/to/testset')

testing_file = open('testing.txt','r')
for line in testing_file:
	line_list = line.split(' ')
	pathFile = open('imagePath.txt','a') 
	pathFile.write(line_list[0]+'\n')


pathFile.close()