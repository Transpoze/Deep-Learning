file="/path/to/testset/imagePath.txt"
while IFS= read -r line
do
	bazel-bin/tensorflow/examples/label_image/label_image --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --output_layer=final_result --input_layer=Mul \
	--image=/Users/christianchamoun/Tensorflow-git/tensorflow/testset/$line 2>> output.txt
	echo 'Computing accuracy on image...'
done <"$file"
