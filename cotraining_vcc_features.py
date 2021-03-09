#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import csv
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.svm import LinearSVC
from scipy.sparse import vstack, hstack
from CoTrainingClassifier import CoTrainingClassifier


def cotrainingfunction(
    train_code_metrics,
    train_code_metrics_Y,
    train_commit_msg,
    train_commit_msg_Y,
    other_code_metrics,  # don't need Y for that one 
    other_commit_msg,  # don't need Y for that one 
    #other_commit_msg_Y,
    test_code_metrics,
    test_code_metrics_Y,
    test_commit_msg,
    test_commit_msg_Y,
    perc
):
    trainset1 = vstack([train_code_metrics, other_code_metrics[0:perc, :]])
    assert (train_code_metrics.shape[0] + perc == trainset1.shape[0])
    trainset2 = vstack([train_commit_msg, other_commit_msg[0:perc, :]])
    assert (train_commit_msg.shape[0] + perc == trainset2.shape[0])

    label_train = train_code_metrics_Y
    # CoTraining implementation expects a label of -1 for "unlabelled"
    tmp = np.repeat(-1, perc)
    label_train = np.hstack([label_train, tmp])
    assert (label_train.shape[0] == trainset1.shape[0])
    assert (label_train.shape[0] == trainset2.shape[0])

    label_test = test_code_metrics_Y

    print('SVM CoTraining')
    svm_co_clf = CoTrainingClassifier(clf=LinearSVC(), clf2=LinearSVC())
    svm_co_clf.fit(trainset1, trainset2, label_train)
    y_pred1 = svm_co_clf.predict(test_code_metrics, test_commit_msg)
    assert (y_pred1.shape[0] == test_code_metrics.shape[0])
    y_code_metrics, y_commit_msg = svm_co_clf.decision_function(
        test_code_metrics, test_commit_msg
    )
    assert (y_code_metrics.shape[0] == y_commit_msg.shape[0])
    assert (y_code_metrics.shape[0] == y_pred1.shape[0])

    return label_test, y_pred1, y_code_metrics, y_commit_msg, svm_co_clf


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception(
            "This script MUST be provided a path as parameter, AND a value for PERC"
        )
    experiment_path = sys.argv[1]
    perc = int(sys.argv[2])

    train_code_metrics, train_code_metrics_Y = load_svmlight_file(
        os.path.join(experiment_path, "./ct_train/ct_train_cm.libsvm"),
        dtype=bool
    )
    train_commit_msg, train_commit_msg_Y = load_svmlight_file(
        os.path.join(experiment_path, "./ct_train/ct_train_md.libsvm"),
        dtype=bool
    )

    other_code_metrics, other_code_metrics_Y = load_svmlight_file(
        os.path.join(
            experiment_path, "./unlabeled_train/unlabeled_train_cm.libsvm"
        ),
        dtype=bool
    )
    other_commit_msg, other_commit_msg_Y = load_svmlight_file(
        os.path.join(
            experiment_path, "./unlabeled_train/unlabeled_train_md.libsvm"
        ),
        dtype=bool
    )

    test_unlabelled_code_metrics, test_unlabelled_code_metrics_Y = load_svmlight_file(
        os.path.join(
            experiment_path, "./unlabeled_test/unlabeled_test_cm.libsvm"
        ),
        dtype=bool
    )
    test_unlabelled_commit_msg, test_unlabelled_commit_msg_Y = load_svmlight_file(
        os.path.join(
            experiment_path, "./unlabeled_test/unlabeled_test_md.libsvm"
        ),
        dtype=bool
    )

    test_code_metrics, test_code_metrics_Y = load_svmlight_file(
        os.path.join(experiment_path, "./ct_test/ct_test_cm.libsvm"),
        dtype=bool
    )
    test_commit_msg, test_commit_msg_Y = load_svmlight_file(
        os.path.join(experiment_path, "./ct_test/ct_test_md.libsvm"),
        dtype=bool
    )

    # Those are sparse matrix -> need to ensure they have compatible shapes
    # for *__code_metrics
    max_features_code_metrics = max(
        train_code_metrics.shape[1], test_code_metrics.shape[1],
        other_code_metrics.shape[1], test_unlabelled_code_metrics.shape[1]
    )
    train_code_metrics = csr_matrix(
        train_code_metrics,
        shape=(train_code_metrics.shape[0], max_features_code_metrics)
    )
    test_code_metrics = csr_matrix(
        test_code_metrics,
        shape=(test_code_metrics.shape[0], max_features_code_metrics)
    )

    test_unlabelled_code_metrics = csr_matrix(
        test_unlabelled_code_metrics,
        shape=(
            test_unlabelled_code_metrics.shape[0], max_features_code_metrics
        )
    )

    other_code_metrics = csr_matrix(
        other_code_metrics,
        shape=(other_code_metrics.shape[0], max_features_code_metrics)
    )

    max_features_commit_msg = max(
        train_commit_msg.shape[1], test_commit_msg.shape[1],
        other_commit_msg.shape[1], test_unlabelled_commit_msg.shape[1]
    )
    train_commit_msg = csr_matrix(
        train_commit_msg,
        shape=(train_commit_msg.shape[0], max_features_commit_msg)
    )
    test_commit_msg = csr_matrix(
        test_commit_msg,
        shape=(test_commit_msg.shape[0], max_features_commit_msg)
    )
    test_unlabelled_commit_msg = csr_matrix(
        test_unlabelled_commit_msg,
        shape=(test_unlabelled_commit_msg.shape[0], max_features_commit_msg)
    )
    other_commit_msg = csr_matrix(
        other_commit_msg,
        shape=(other_commit_msg.shape[0], max_features_commit_msg)
    )

    lab1, pred1, prob1, prob2, trained_clf = cotrainingfunction(
        train_code_metrics,
        train_code_metrics_Y,
        train_commit_msg,
        train_commit_msg_Y,
        other_code_metrics,  # don't need Y for that one 
        other_commit_msg,  # don't need Y for that one 
        #other_commit_msg_Y,
        test_code_metrics,
        test_code_metrics_Y,
        test_commit_msg,
        test_commit_msg_Y,
        perc
    )

    predictions = []
    for i in range(len(prob1)):
        predictions.append((prob1[i] + prob2[i]) / 2)
    file_out = open(
        os.path.join(
            experiment_path, f"test_predictions_cotraining_vcc_features_{perc}.txt"
        ), "w"
    )
    csv_out = csv.writer(file_out, csv.unix_dialect, quoting=csv.QUOTE_NONE)
    # write a header
    csv_out.writerow(("Prediction", "actual_class"))
    for (prediction, actual_label) in zip(predictions, lab1):
        csv_out.writerow((prediction, actual_label))
    file_out.close()

    # now, let's predict the unlabelled (i.e. without ground Truth) set
    # first: load the matrices: ALREADY DONE : test_unlabelled_code_metrics and test_unlabelled_commit_msg
    # the actual prediction:
    unlabelled_pred1, unlabelled_pred2   = trained_clf.decision_function(
        test_unlabelled_code_metrics, test_unlabelled_commit_msg
    )
    unlab_predictions = []
    for i in range(len(unlabelled_pred1)):
        unlab_predictions.append((unlabelled_pred1[i] + unlabelled_pred2[i]) / 2)

    # write prediction file
    # we consider the class/label to be 0 for all unclassified samples:
    labels_unlabelled = [0] * len(unlab_predictions)
    file_out = open(
        os.path.join(
            experiment_path,
            f"unlabelled_predictions_cotraining_vcc_features_{perc}.txt"
        ), "w"
    )
    csv_out = csv.writer(file_out, csv.unix_dialect, quoting=csv.QUOTE_NONE)
    # write a header
    csv_out.writerow(("Prediction", "actual_class"))
    for (prediction, actual_label) in zip(unlab_predictions, labels_unlabelled):
        csv_out.writerow((prediction, actual_label))
    file_out.close()
