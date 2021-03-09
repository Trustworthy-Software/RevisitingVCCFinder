#!/bin/bash
#Fetch data and rename
#
#
set -e
set -x

function die()
{
	print ERROR in $0: $1
	exit 2
}

# By construction both the _code_metrics_matrix.csv and the _commit_msg_matrix.csv SHOULD have ids in the same order
# i.e., line x of both files will have the same Commit_id (first col), and hence talk about the same commit
# Bad things will happen if this doesn't hold true -> Let's ensure this is true
function check_id_order()
{
    set +x # no logging inside func
    local base_file=$1
    local cm_file=${base_file}_code_metrics_matrix.csv
    local md_file=${base_file}_commit_msg_matrix.csv
    ids_cm=$(cut -d',' -f1 ${cm_file})
    ids_md=$(cut -d',' -f1 ${md_file})
    [[ zz"${ids_cm}" == zz"${ids_md}" ]] || die "Files ${cm_file} and ${md_file} are not in the same order. Aborting..."
	# Ids lists are different, Ids are in NOT the same order
	# Note: this could also mean that both files have different numbers of lines
}

function merge()
{
    local base_file=$1
    local cm_file=${base_file}_code_metrics_matrix.csv
    local md_file=${base_file}_commit_msg_matrix.csv
    check_id_order ${base_file} || die "check_id_order: files have different IDs lists, aborting" # Better safe than sorry
    # Paste outputs on one single line:  line x from file A    AND    line x from file B
    # (What cat does vertically, paste does horizontally)
    # Here, we remove the id column of the second CSV, as we only want one ID
    cut -d',' -f2- ${cm_file} | paste -d',' ${md_file} -
}




csv_dir="../exp_new_features"
[[ -d "./${csv_dir}" ]] || die "csv_dir path does NOT exist"

libsvm_dir="./3_IntermediarySteps/new_libsvm/"
[[ -d ./${libsvm_dir} ]] || mkdir -p ./${libsvm_dir}

# "Fused" files, i.e., containg both negative and positive samples
fused_libsvm_dir="./3_IntermediarySteps/fused_feature_files"
[[ -d ./${fused_libsvm_dir} ]] || mkdir -p ./${fused_libsvm_dir}



#test_
merge "${csv_dir}/test_neg" | ./csv_to_libsvm.awk -v LABEL="0" > ${libsvm_dir}/test_neg.libsvm
merge "${csv_dir}/test_pos" | ./csv_to_libsvm.awk -v LABEL="1" > ${libsvm_dir}/test_pos.libsvm
cat ${libsvm_dir}/test_neg.libsvm ${libsvm_dir}/test_pos.libsvm > ${fused_libsvm_dir}/test.libsvm

#train
merge "${csv_dir}/training_neg" | ./csv_to_libsvm.awk -v LABEL="0" > ${libsvm_dir}/training_neg.libsvm
merge "${csv_dir}/training_pos" | ./csv_to_libsvm.awk -v LABEL="1" > ${libsvm_dir}/training_pos.libsvm
cat ${libsvm_dir}/training_neg.libsvm ${libsvm_dir}/training_pos.libsvm > ${fused_libsvm_dir}/training.libsvm

#train_*
merge "${csv_dir}/train_data_1000" | ./csv_to_libsvm.awk -v LABEL="0" > ${libsvm_dir}/training_neg_1000.libsvm
cat ${libsvm_dir}/training_pos.libsvm ${libsvm_dir}/training_neg_1000.libsvm > ${fused_libsvm_dir}/training_1000.libsvm
merge "${csv_dir}/train_data_5000" | ./csv_to_libsvm.awk -v LABEL="0" > ${libsvm_dir}/training_neg_5000.libsvm
cat ${libsvm_dir}/training_pos.libsvm ${libsvm_dir}/training_neg_5000.libsvm > ${fused_libsvm_dir}/training_5000.libsvm
merge "${csv_dir}/train_data_10000" | ./csv_to_libsvm.awk -v LABEL="0" > ${libsvm_dir}/training_neg_10000.libsvm
cat ${libsvm_dir}/training_pos.libsvm ${libsvm_dir}/training_neg_10000.libsvm > ${fused_libsvm_dir}/training_10000.libsvm

#unlabeled
merge "${csv_dir}/unlab_test" | ./csv_to_libsvm.awk -v LABEL="0" > ${libsvm_dir}/unlab_test.libsvm

cat ${fused_libsvm_dir}/test.libsvm ${libsvm_dir}/unlab_test.libsvm > ${fused_libsvm_dir}/mega_test.libsvm
