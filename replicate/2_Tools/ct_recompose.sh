#!/bin/bash
set -e
set -x

output_folder=../4_Results/1_Data/2_ct_input
input_folder=../3_IntermediarySteps/ct_4_Post_Sally

#mkdir -p ct_post_sally
#mkdir -p ct_post_sally/ct_test
#mkdir -p ct_post_sally/ct_train
#mkdir -p ct_post_sally/ct_tr1000
#mkdir -p ct_post_sally/ct_tr5000
#mkdir -p ct_post_sally/ct_tr10000
#mkdir -p ct_post_sally/unlabeled_train
#mkdir -p ct_post_sally/unlabeled_test

mkdir -p $output_folder
mkdir -p $output_folder/ct_test
mkdir -p $output_folder/ct_train
mkdir -p $output_folder/ct_tr1000
mkdir -p $output_folder/ct_tr5000
mkdir -p $output_folder/ct_tr10000
mkdir -p $output_folder/unlabeled_train
mkdir -p $output_folder/unlabeled_test


echo '===============Recompose================'
#ct_training_
cat $input_folder/ct_train_neg_cm.libsvm $input_folder/ct_train_pos_cm.libsvm > $output_folder/ct_train/ct_train_cm.libsvm
cat $input_folder/ct_train_neg_md.libsvm $input_folder/ct_train_pos_md.libsvm > $output_folder/ct_train/ct_train_md.libsvm
#ct_tr_n1000
cat $input_folder/ct_tr_n1000_cm.libsvm $input_folder/ct_train_pos_cm.libsvm > $output_folder/ct_tr1000/ct_tr1000_cm.libsvm
cat $input_folder/ct_tr_n1000_md.libsvm $input_folder/ct_train_pos_md.libsvm > $output_folder/ct_tr1000/ct_tr1000_md.libsvm
#ct_tr_n5000
cat $input_folder/ct_tr_n5000_cm.libsvm $input_folder/ct_train_pos_cm.libsvm > $output_folder/ct_tr5000/ct_tr5000_cm.libsvm
cat $input_folder/ct_tr_n5000_md.libsvm $input_folder/ct_train_pos_md.libsvm > $output_folder/ct_tr5000/ct_tr5000_md.libsvm
#ct_tr_n10000
cat $input_folder/ct_tr_n10000_cm.libsvm $input_folder/ct_train_pos_cm.libsvm > $output_folder/ct_tr10000/ct_tr10000_cm.libsvm
cat $input_folder/ct_tr_n10000_md.libsvm $input_folder/ct_train_pos_md.libsvm > $output_folder/ct_tr10000/ct_tr10000_md.libsvm
#ct_tr_
cat $input_folder/ct_test_neg_cm.libsvm $input_folder/ct_test_pos_cm.libsvm > $output_folder/ct_test/ct_test_cm.libsvm
cat $input_folder/ct_test_neg_md.libsvm $input_folder/ct_test_pos_md.libsvm > $output_folder/ct_test/ct_test_md.libsvm

#unlabeled to do
##unlabeled_train
cp $input_folder/unlabeled_train_cm.libsvm $output_folder/unlabeled_train/ 
cp $input_folder/unlabeled_train_md.libsvm $output_folder/unlabeled_train/
##unlabeled_test
cp $input_folder/unlabeled_test_cm.libsvm $output_folder/unlabeled_test/
cp $input_folder/unlabeled_test_md.libsvm $output_folder/unlabeled_test/

echo 'All cotraining_data generated in ' $output_folder

