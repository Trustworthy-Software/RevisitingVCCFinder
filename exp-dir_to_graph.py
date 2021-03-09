#!/usr/bin/env python3
# coding: utf-8

import sys
import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_curve
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('pdf')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception(
            "This script MUST be provided as parameters: 1) the path of the experiment to plot, "
            "2) either 'new_features' or 'vcc_features', and 3) a value for plot 'title'"
        )
    experiment_path = sys.argv[1]
    features_name = sys.argv[2]
    assert (features_name == 'new_features' or features_name == 'vcc_features')
    plot_title = sys.argv[3]

    df1000 = pd.read_csv(
        os.path.join(
            experiment_path, f'test_predictions_cotraining_{features_name}_1000.txt'
        ),
        index_col=None,
        header=0
    )
    df1000_unlab = pd.read_csv(
        os.path.join(
            experiment_path, f'unlabelled_predictions_cotraining_{features_name}_1000.txt'
        ),
        index_col=None,
        header=0
    )
    df1000_unlab = pd.concat([df1000, df1000_unlab], ignore_index=True, copy=False)
    df5000 = pd.read_csv(
        os.path.join(
            experiment_path, f'test_predictions_cotraining_{features_name}_5000.txt'
        ),
        index_col=None,
        header=0
    )
    df5000_unlab = pd.read_csv(
        os.path.join(
            experiment_path, f'unlabelled_predictions_cotraining_{features_name}_5000.txt'
        ),
        index_col=None,
        header=0
    )
    df5000_unlab = pd.concat([df5000, df5000_unlab], ignore_index=True, copy=False)
    df10000 = pd.read_csv(
        os.path.join(
            experiment_path,
            f'test_predictions_cotraining_{features_name}_10000.txt'
        ),
        index_col=None,
        header=0
    )
    df10000_unlab = pd.read_csv(
        os.path.join(
            experiment_path,
            f'unlabelled_predictions_cotraining_{features_name}_10000.txt'
        ),
        index_col=None,
        header=0
    )
    df10000_unlab = pd.concat([df10000, df10000_unlab], ignore_index=True, copy=False)

    plt.clf()
    plt.rcParams.update({'font.size': 27})
    plt.figure(figsize=(17, 17))

    plt.ylim([-0.01, 1.01])
    plt.xlim([-0.01, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    for (df, name) in [
        (df1000, f"CoTraining {features_name}, 1000 unlab, Test w/o Unlab"), #(df5000, "df5000"), 
        (df10000, f"CoTraining {features_name}, 10 000 unlab, Test w/o Unlab"), 
        (df1000_unlab, f"CoTraining {features_name}, 1000 unlab, Test w Unlab"), #(df5000_unlab, "df5000_unlab"),  
        (df10000_unlab, f"CoTraining {features_name}, 10 000 unlab, Test w Unlab")
    ]:
        precision_test, recall_test, threshold_test = precision_recall_curve(
            df["actual_class"], df["Prediction"]
        )
        plt.plot(
            recall_test,
            precision_test,
            label=name,
            marker='o',
            linestyle='',
            linewidth=2,
            markersize=4,
            rasterized=True
        )
    plt.legend(loc="best", markerscale=5, fontsize=23)
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x_iso = np.linspace(0.01, 1, num=50)
        y_iso = (f_score * x_iso) / (2 * x_iso - f_score)
        plt.plot(
            x_iso[y_iso > 0],
            y_iso[y_iso > 0],
            color='gray',
            alpha=0.3,
            linewidth=3
        )
    plt.grid(color='gray', linestyle=':', linewidth=1)
    plt.title(label=plot_title)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, 'recall_precision.pdf'), dpi=400)
