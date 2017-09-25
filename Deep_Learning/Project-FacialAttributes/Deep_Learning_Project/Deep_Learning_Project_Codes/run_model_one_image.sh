#Specify image to test from testset folder as input
read -p "Enter file name with extension: " -e input
bazel-bin/tensorflow/examples/label_image/label_image --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --output_layer=final_result --input_layer=Mul \
--image=/Users/christianchamoun/Tensorflow-git/tensorflow/testset/AFLW/$input &> output.txt
