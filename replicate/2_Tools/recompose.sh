#!/bin/bash
set -e
set -x

output_folder=../3_IntermediarySteps/5_vcc_Input
input_folder=../3_IntermediarySteps/4_Post_Sally

mkdir -p $output_folder
#


echo '===============Recompose================'
#training_
cat $input_folder/training_neg_reg.libsvm $input_folder/training_pos.libsvm > $output_folder/training_reg.libsvm

#tr_n1000
cat $input_folder/training_neg_1000.libsvm $input_folder/training_pos.libsvm > $output_folder/training_1000.libsvm

#tr_n5000
cat $input_folder/training_neg_5000.libsvm $input_folder/training_pos.libsvm > $output_folder/training_5000.libsvm

#tr_n10000
cat $input_folder/training_neg_10000.libsvm $input_folder/training_pos.libsvm > $output_folder/training_10000.libsvm

#test
cat $input_folder/test_neg.libsvm $input_folder/test_pos.libsvm > $output_folder/test.libsvm


#type_training_
cat $input_folder/type_train_neg.libsvm $input_folder/type_train_pos.libsvm > $output_folder/type_training_reg.libsvm

#type_test
cat $input_folder/type_test_neg.libsvm $input_folder/type_test_pos.libsvm > $output_folder/type_test.libsvm


#unlabeled to do
##unlabeled_train
cp $input_folder/unlabeled_train.libsvm $output_folder/

##unlabeled_test
cp $input_folder/unlabeled_test.libsvm $output_folder/


#mega_train
cat $output_folder/unlabeled_train.libsvm  $output_folder/training_reg.libsvm > $output_folder/mega_training.libsvm
#mega_test
cat $output_folder/unlabeled_test.libsvm  $output_folder/test.libsvm > $output_folder/mega_test.libsvm

