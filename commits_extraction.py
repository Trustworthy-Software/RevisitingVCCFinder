#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import csv
import vcc_utils
from sklearn.feature_extraction.text import TfidfVectorizer


def commit_patch_to_code_metrics_vector(commit_id, commit_patch):
    """
    Code_metrics
    returns ONE feature vector from ONE commit_patch
    EXPECTS: only one commit at a time
    """
    # init vars
    # bad var names, since those are not addition/removal
    nb_added_if, nb_removed_if = 0, 0
    nb_added_loop, nb_removed_loop = 0, 0
    nb_added_file, nb_removed_file = 0, 0
    nb_added_function, nb_removed_function = 0, 0
    nb_added_paren, nb_removed_paren = 0, 0
    nb_added_bool, nb_removed_bool = 0, 0
    nb_added_assignement, nb_removed_assignement = 0, 0
    nb_added_break, nb_removed_break = 0, 0
    nb_added_sizeof, nb_removed_sizeof = 0, 0
    nb_added_return, nb_removed_return = 0, 0
    nb_added_continue, nb_removed_continue = 0, 0
    nb_added_INTMAX, nb_removed_INTMAX = 0, 0
    nb_added_goto, nb_removed_goto = 0, 0
    nb_added_define, nb_removed_define = 0, 0
    nb_added_struct, nb_removed_struct = 0, 0
    nb_added_void, nb_removed_void = 0, 0
    nb_added_offset, nb_removed_offset = 0, 0
    nb_added_line, nb_removed_line = 0, 0
    # FIXME:
    kbp = 0
    kbn = 0

    for line in commit_patch:
        if line.startswith("+"):
            nb_added_line += 1
            if "if (" in line:
                nb_added_if += 1
            if ("for (" in line) or ("while (" in line):
                nb_added_loop += 1
            if line.startswith('+++'):
                # a file is modified pay this patch
                nb_added_file += 1
            if any(
                i in line for i in
                ["int", "static", "void", "float", "char", "char*", "string"]
            ):
                if "(" in line and ")" in line:
                    nb_added_function += 1

            if ("(" in line) and (")" in line):
                # parenthesis expr detection
                nb_added_paren += 1
            if any(i in line for i in ["||", "&&", "!"]):
                # Boolean operator
                if "!=" in line:
                    # FIXME: What does kbp mean ???
                    kbp += 1
                nb_added_bool += 1
            nb_added_assignement += sum([1 for char in line if char == '='])
            if "sizeof" in line:
                nb_added_sizeof += 1
            if "break" in line:
                nb_added_break += 1
            if "return" in line:
                nb_added_return += 1
            if "continue" in line:
                nb_added_continue += 1
            if "int max" in line:
                nb_added_INTMAX += 1
            if "goto" in line:
                nb_added_goto += 1
            if "#define" in line:
                nb_added_define += 1
            if "struct" in line:
                nb_added_struct += 1
            if "void" in line:
                nb_added_void += 1
            if ("offset =" in line) or ("offset=" in line):
                nb_added_offset += 1
        # same thing, but for removal
            nb_removed_line += 1
            if line.startswith("-"):
                if "if (" in line:
                    nb_removed_if += 1
                if "sizeof" in line:
                    nb_removed_sizeof += 1
                if "break" in line:
                    nb_removed_break += 1
                if "return" in line:
                    nb_removed_return += 1
                if "continue" in line:
                    nb_removed_continue += 1
                if "int max" in line:
                    nb_removed_INTMAX += 1
                if "goto" in line:
                    nb_removed_goto += 1
                if "#define" in line:
                    nb_removed_define += 1
                if "struct" in line:
                    nb_removed_struct += 1
                if "void" in line:
                    nb_removed_void += 1
                if "offset =" in line:
                    nb_removed_offset += 1
                if ("for (" in line) or ("while (" in line):
                    nb_removed_loop += 1
                if line.startswith('---'):
                    nb_removed_file += 1
                if any(
                    i in line for i in [
                        "int", "static", "void", "float", "char", "char*",
                        "string"
                    ]
                ):
                    if "(" in line and ")" in line:
                        nb_removed_function += 1
                if ("(" in line) and (")" in line):
                    nb_removed_paren += 1
                if any(i in line for i in ["||", "&&", "!"]):
                    if "!=" in line:
                        # FIXME: What does kbn mean ???
                        kbn = kbn + 1
                    nb_removed_bool += 1
                nb_removed_assignement += sum(
                    [1 for char in line if char == '=']
                )
    f1 = nb_added_if - nb_removed_if
    f2 = nb_added_loop - nb_removed_loop
    f3 = nb_added_line - nb_removed_line
    f4 = nb_added_file - nb_removed_file
    f5 = nb_added_function - nb_removed_function
    f6 = nb_added_paren - nb_removed_paren
    f7 = nb_added_bool - nb_removed_bool
    f8 = nb_added_assignement - nb_removed_assignement
    f17 = nb_added_struct - nb_removed_struct
    f19 = nb_added_void - nb_removed_void
    f21 = nb_added_offset - nb_removed_offset

    f9 = nb_added_if + nb_removed_if
    f10 = nb_added_loop + nb_removed_loop
    f11 = nb_added_line + nb_removed_line
    f12 = nb_added_file + nb_removed_file
    f13 = nb_added_function + nb_removed_function
    f14 = nb_added_paren + nb_removed_paren
    f15 = nb_added_bool + nb_removed_bool
    f16 = nb_added_assignement + nb_removed_assignement
    f18 = nb_added_struct + nb_removed_struct
    f20 = nb_added_void + nb_removed_void
    f22 = nb_added_offset + nb_removed_offset

    row = [
        commit_id,
        nb_added_line,
        nb_removed_line,
        nb_added_if,
        nb_removed_if,
        nb_added_loop,
        nb_removed_loop,
        nb_added_file,
        nb_removed_file,
        nb_added_function,
        nb_removed_function,
        nb_added_paren,
        nb_removed_paren,
        nb_added_bool,
        nb_removed_bool,  # FIXME: see FIXME: kbn and kbpS
        nb_added_assignement,
        nb_removed_assignement,
        f1,
        f2,
        f3,
        f4,
        f5,
        f6,
        f7,
        f8,
        f9,
        f10,
        f11,
        f12,
        f13,
        f14,
        f15,
        f16,
        nb_added_offset,
        nb_removed_offset,
        nb_added_return,
        nb_removed_return,
        nb_added_break,
        nb_removed_break,
        nb_added_continue,
        nb_removed_continue,
        nb_added_INTMAX,
        nb_removed_INTMAX,
        nb_added_define,
        nb_removed_define,
        nb_added_struct,
        nb_removed_struct,
        nb_added_void,
        nb_removed_void,
        nb_added_offset,
        nb_removed_offset,
        f17,
        f18,
        f19,
        f20,
        f21,
        f22
    ]
    return row


def commits_to_code_metrics_csv(commit_flow, output_filename):
    f_out = open(output_filename, "w", newline='')
    csv_out = csv.writer(f_out, csv.unix_dialect, quoting=csv.QUOTE_NONE)
    commit_flow = vcc_utils.filter_non_code(commit_flow)
    for (commit_id, commit_message, commit_patch) in commit_flow:
        vector = commit_patch_to_code_metrics_vector(commit_id, commit_patch)
        csv_out.writerow(vector)
    f_out.close()


### Other feature set : Commit_msg
def commits_to_commit_msg_csv(commit_flow, output_filename):
    # Commit_flow is a gen of (commit_id, commit_message, commit_patch)
    # We need [commit_id] and [commit_message]
    commit_flow = list(commit_flow)
    commit_ids = [x[0] for x in commit_flow]
    commit_messages = [x[1] for x in commit_flow]
    commit_flow = None
    vectorizer = TfidfVectorizer(
        encoding="utf-8",
        min_df=0.0,
        analyzer="word",
        tokenizer=None,
        lowercase="True",
        preprocessor=None,
        stop_words="english",
        max_features=10,
        use_idf="True"
    )
    tdidf_vectors = vectorizer.fit_transform(commit_messages)
    matrix = tdidf_vectors.todense()
    csv_out = csv.writer(
        open(output_filename, "w", newline=''),
        csv.unix_dialect,
        quoting=csv.QUOTE_NONE
    )
    assert(len(matrix) == len(commit_ids))
    for idx in range(len(matrix)):
        csv_out.writerow([int(commit_ids[idx])] + matrix[idx].tolist()[0])




if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            "This script MUST be provided one path as parameter: experiment path"
        )
    experiment_path = sys.argv[1]

    # First batch of lists of commit ids is in replicate/1_Data/commit_refs/
    commit_ids_files = [
        'test_neg', 'test_pos', 
        'training_neg', 'training_pos', 
        'unlabeled_test',# 'unlabeled_train'
        ]
    for my_file in commit_ids_files:
        file_path = os.path.join('replicate/1_Data/commit_refs/', my_file)
        f_in = open(file_path, 'r')
        commit_ids = [ int(x.rstrip()) for x in  f_in.readlines()]
        # random name change
        if 'unlabeled' in my_file:
            my_file = my_file.replace('unlabeled', 'unlab')
        # Extract feature set 1: code_metrics
        output_filename = os.path.join(experiment_path, my_file + '_code_metrics_matrix.csv')
        commit_flow = vcc_utils.get_commits_from_db(commit_ids)
        commits_to_code_metrics_csv(commit_flow, output_filename)
        print(f"DONE writing {output_filename}")
        # Extract feature set 2: Commit_msg
        # generator has been consumed... Need to redo
        output_filename = os.path.join(experiment_path, my_file + '_commit_msg_matrix.csv')
        commit_flow = vcc_utils.get_commits_from_db(commit_ids)
        commits_to_commit_msg_csv(commit_flow, output_filename)
        print(f"DONE writing {output_filename}")


    # Second batch of lists fo commit ids is in replicate/1_Data/unlab_train_extended/
    commit_ids_files = ['1000', '5000', '10000']
    for my_file in commit_ids_files:
        file_path = os.path.join('replicate/1_Data/unlab_train_extended/', "unlab_train_" + my_file + '.txt')
        f_in = open(file_path, 'r')
        commit_ids = [ int(x.rstrip()) for x in  f_in.readlines()]
        # Extract feature set 1: code_metrics
        output_filename = os.path.join(experiment_path, 'train_data_' + my_file + '_code_metrics_matrix.csv')
        commit_flow = vcc_utils.get_commits_from_db(commit_ids)
        commits_to_code_metrics_csv(commit_flow, output_filename)
        print(f"DONE writing {output_filename}")
        # generator has been consumed... Need to redo
        output_filename = os.path.join(experiment_path, 'train_data_' + my_file + '_commit_msg_matrix.csv')
        commit_flow = vcc_utils.get_commits_from_db(commit_ids)
        commits_to_commit_msg_csv(commit_flow, output_filename)
        print(f"DONE writing {output_filename}")

