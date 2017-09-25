import linecache
from collections import Counter
label_file = open('testImageLabel.txt','r')

total_images = 1
correct_classified = 0
highest_acc = 2
failed = []
for line in label_file:
	label_line_split = line.split(' ') 
	acc_line = linecache.getline('accuracy.txt', highest_acc)
	acc_line_split = acc_line.split(' ')
	predicted_class = acc_line_split[1]
	labeled_class = label_line_split[1]
	
	if predicted_class == labeled_class:
		correct_classified = correct_classified + 1
	else:
		failed.append(predicted_class)
	
	total_images = total_images + 1
	highest_acc = highest_acc + 5

finalAcc = float(correct_classified)/float(total_images)

print 'Test images: '+str(total_images)
print 'Correctly classified: '+str(correct_classified)
print 'Final accuracy: '+str(finalAcc*100)+' %' 	

newList = Counter(failed)
print newList
print 'Failed from class 0 (Male with glasses): '+ str(float(newList['0'])/1143)
print 'Failed class 1 (Female without glasses): '+ str(float(newList['1'])/3928)
print 'Failed class 2 (Female with glasses): '+ str(float(newList['2'])/267)
print 'Failed class 3 (Male without glasses): '+ str(float(newList['2'])/4666)