#!/usr/bin/env python3
# coding: utf-8

from sklearn.svm import LinearSVC
import sys
import os
import numpy as np
import pandas as pd
from CoTrainingClassifier import CoTrainingClassifier
import csv


# What does  'perc' stand for?
# FIXME/ find better parameter name for perc
def cotraining(
    train_blame_code_metrics, train_blame_commit_msg, train_fix_code_metrics,
    train_fix_commit_msg, train_other_code_metrics, train_other_commit_msg,
    test_fix_code_metrics, test_fix_commit_msg, test_blame_code_metrics,
    test_blame_commit_msg, perc
):
    """
    trains a cotraining classifier, and predicts the test set
    """
    # FIXME: we need better variable names
    lefix = min(len(test_fix_code_metrics), len(test_fix_commit_msg))
    leblame = min(len(test_blame_code_metrics), len(test_blame_commit_msg))

    # FIXME: we need better variable names
    levul = min(len(train_blame_code_metrics), len(train_blame_commit_msg))
    lebf = min(len(train_fix_code_metrics), len(train_fix_commit_msg))
    #letes = min(len(train_other11), len(train_other22))

    trainset1 = []
    label_train = []
    for i in range(levul):
        trainset1.append(train_blame_code_metrics[i])
        label_train.append(1)
    for i in range(lebf):
        trainset1.append(train_fix_code_metrics[i])
        label_train.append(0)
    for i in range(perc):
        trainset1.append(train_other_code_metrics[i])
        label_train.append(-1)

    trainset2 = []
    for i in range(levul):
        trainset2.append(train_blame_commit_msg[i])
    for i in range(lebf):
        trainset2.append(train_fix_commit_msg[i])
    for i in range(perc):
        trainset2.append(train_other_commit_msg[i])

    # FIXME: we need better variable names
    labbb = []
    predict1 = []
    for i in range(lefix):
        predict1.append(test_fix_code_metrics[i])
        labbb.append(0)
    for i in range(leblame):
        predict1.append(test_blame_code_metrics[i])
        labbb.append(1)

    predict2 = []
    for i in range(lefix):
        predict2.append(test_fix_commit_msg[i])
    for i in range(leblame):
        predict2.append(test_blame_commit_msg[i])

    X1 = np.array(trainset1)
    X2 = np.array(trainset2)
    label_train1 = np.array(label_train)
    Xt1 = np.array(predict1)
    Xt2 = np.array(predict2)

    print('SVM CoTraining')
    svm_co_clf = CoTrainingClassifier(clf=LinearSVC(), clf2=LinearSVC())
    svm_co_clf.fit(X1, X2, label_train1)
    y_pred = svm_co_clf.predict(Xt1, Xt2)
    y_test1, y_test2 = svm_co_clf.decision_function(Xt1, Xt2)

    return labbb, y_pred, y_test1, y_test2, svm_co_clf


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception(
            "This script MUST be provided a path as parameter, AND a value for PERC"
        )
    experiment_path = sys.argv[1]
    perc = int(sys.argv[2])

    train_fix_code_metrics_df = pd.read_csv(
        os.path.join(experiment_path, "training_neg_code_metrics_matrix.csv"),
        index_col=0,
        header=None
    )
    train_fix_commit_msg_df = pd.read_csv(
        os.path.join(experiment_path, "training_neg_commit_msg_matrix.csv"),
        index_col=0,
        header=None
    )

    train_blame_code_metrics_df = pd.read_csv(
        os.path.join(experiment_path, "training_pos_code_metrics_matrix.csv"),
        index_col=0,
        header=None
    )
    train_blame_commit_msg_df = pd.read_csv(
        os.path.join(experiment_path, "training_pos_commit_msg_matrix.csv"),
        index_col=0,
        header=None
    )

    test_fix_code_metrics_df = pd.read_csv(
        os.path.join(experiment_path, "test_neg_code_metrics_matrix.csv"),
        index_col=0,
        header=None
    )
    test_fix_commit_msg_df = pd.read_csv(
        os.path.join(experiment_path, "test_neg_commit_msg_matrix.csv"),
        index_col=0,
        header=None
    )

    test_blame_code_metrics_df = pd.read_csv(
        os.path.join(experiment_path, "test_pos_code_metrics_matrix.csv"),
        index_col=0,
        header=None
    )
    test_blame_commit_msg_df = pd.read_csv(
        os.path.join(experiment_path, "test_pos_commit_msg_matrix.csv"),
        index_col=0,
        header=None
    )

    train_other_code_metrics_df = pd.read_csv(
        os.path.join(
            experiment_path, f"train_data_{perc}_code_metrics_matrix.csv"
        ),
        index_col=0,
        header=None
    )
    train_other_commit_msg_df = pd.read_csv(
        os.path.join(
            experiment_path, f"train_data_{perc}_commit_msg_matrix.csv"
        ),
        index_col=0,
        header=None
    )

    # IDs, Values
    id_train_other_code_metrics, train_other_code_metrics = np.asarray(
        train_other_code_metrics_df.index
    ), np.asarray(train_other_code_metrics_df)
    id_train_other_commit_msg, train_other_commit_msg = np.asarray(
        train_other_commit_msg_df.index
    ), np.asarray(train_other_commit_msg_df)

    id_train_fix_commit_msg, train_fix_commit_msg = np.asarray(
        train_fix_commit_msg_df.index
    ), np.asarray(train_fix_commit_msg_df)
    id_train_blame_code_metrics, train_blame_code_metrics = np.asarray(
        train_blame_code_metrics_df.index
    ), np.asarray(train_blame_code_metrics_df)

    id_test_fix_commit_msg, test_fix_commit_msg = np.asarray(
        test_fix_commit_msg_df.index
    ), np.asarray(test_fix_commit_msg_df)
    id_test_blame_commit_msg, test_blame_commit_msg = np.asarray(
        test_blame_commit_msg_df.index
    ), np.asarray(test_blame_commit_msg_df)

    id_test_fix_code_metrics, test_fix_code_metrics = np.asarray(
        test_fix_code_metrics_df.index
    ), np.asarray(test_fix_code_metrics_df)
    id_test_blame_code_metrics, test_blame_code_metrics = np.asarray(
        test_blame_code_metrics_df.index
    ), np.asarray(test_blame_code_metrics_df)

    id_train_fix_code_metrics, train_fix_code_metrics = np.asarray(
        train_fix_code_metrics_df.index
    ), np.asarray(train_fix_code_metrics_df)
    id_train_blame_commit_msg, train_blame_commit_msg = np.asarray(
        train_blame_commit_msg_df.index
    ), np.asarray(train_blame_commit_msg_df)

    # FIXME: find better variable names
    lab1, pred1, prob1, prob2, trained_clf = cotraining(
        train_blame_code_metrics, train_blame_commit_msg,
        train_fix_code_metrics, train_fix_commit_msg, train_other_code_metrics,
        train_other_commit_msg, test_fix_code_metrics, test_fix_commit_msg,
        test_blame_code_metrics, test_blame_commit_msg, perc
    )
    # Let's save the predictions obtained on the test set (the part with a ground truth)
    test_predictions = []
    for i in range(len(prob1)):
        test_predictions.append((prob1[i] + prob2[i]) / 2)

    file_out = open(
        os.path.join(
            experiment_path,
            f"test_predictions_cotraining_new_features_{perc}.txt"
        ), "w"
    )
    csv_out = csv.writer(file_out, csv.unix_dialect, quoting=csv.QUOTE_NONE)
    # write a header
    csv_out.writerow(("Prediction", "actual_class"))
    for (prediction, actual_label) in zip(test_predictions, lab1):
        csv_out.writerow((prediction, actual_label))
    file_out.close()

    # now, let's predict the unlabelled (i.e. without ground Truth) set
    # first: load the matrices:
    unlabelled_code_metrics_df = pd.read_csv(
        os.path.join(experiment_path, f"unlab_test_code_metrics_matrix.csv"),
        index_col=0,
        header=None
    )
    unlabelled_commit_msg_df = pd.read_csv(
        os.path.join(experiment_path, f"unlab_test_commit_msg_matrix.csv"),
        index_col=0,
        header=None
    )
    # to np.array
    id_unlabelled_code_metrics, unlabelled_code_metrics = np.asarray(
        unlabelled_code_metrics_df.index
    ), np.asarray(unlabelled_code_metrics_df)
    id_unlabelled_commit_msg, unlabelled_commit_msg = np.asarray(
        unlabelled_commit_msg_df.index
    ), np.asarray(unlabelled_commit_msg_df)

    # the actual prediction:
    unlabelled_pred1, unlabelled_pred2  = trained_clf.decision_function(
        unlabelled_code_metrics, unlabelled_commit_msg
    )
    unlab_predictions = []
    for i in range(len(unlabelled_pred1)):
        unlab_predictions.append((unlabelled_pred1[i] + unlabelled_pred2[i]) / 2)

    # write prediction file
    # we consider the class/label to be 0 for all unclassified samples:
    labels_unlabelled = [0] * len(id_unlabelled_commit_msg)
    file_out = open(
        os.path.join(
            experiment_path,
            f"unlabelled_predictions_cotraining_new_features_{perc}.txt"
        ), "w"
    )
    csv_out = csv.writer(file_out, csv.unix_dialect, quoting=csv.QUOTE_NONE)
    # write a header
    csv_out.writerow(("Prediction", "actual_class"))
    for (prediction, actual_label) in zip(unlab_predictions, labels_unlabelled):
        csv_out.writerow((prediction, actual_label))
    file_out.close()
