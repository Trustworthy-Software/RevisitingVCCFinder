#####################################################################
#
#	The aim of this file is to set the value of the vectors in the positive case to 1
#		to mean 'it is a Vulnerability Introducing Commit'
#
#
#
###################################################################

import re
from os import listdir
from os.path import join, isfile

##Input folder
#folder="post_sally"   ##formerly, kept for ref
folder = "../3_IntermediarySteps/2_Post_sally"

##List folders for setting value to one as positives cases
folder_list = list()
folder_list.append("test_pos.libsvm")
folder_list.append("training_pos.libsvm")

for folder_n in folder_list:
    path_to_file = join(folder, folder_n)
    #all_elts=listdir(folder_path)
    #all_files=list()
    #print("-",folder_n)
    #for elt in all_elts:
    #    print ("--",elt)
    #if isfile(elt):
    #    all_files.append(elt)
    #print ("--..",all_files)
    #for file_elt in all_files:
    #print ("---", path_to_file)
    #path_to_file=join(file,file_elt)
    new_lines = list()
    with open(path_to_file, 'r') as f:
        file_lines = f.readlines()
    for line in file_lines:
        new_lines.append((re.sub('^\S+\s', "1 ", line)).replace("\n", ""))
        #print ("----",line)
    opened_file = open(path_to_file, 'w')
    for line in new_lines:
        #print ("----",line)
        print(line, file=opened_file)
    opened_file.close()

#The point here is to do the same as earlier but on the data related to the generation of input
#for the co-training. So we go fetch in ../4_Results and change concerned files
#
#folder="post_sally_ct_input"
outFolder = "../4_Results/1_Data/ct_input"

subfolders = list()
#subfolders.append("test_pos")
#subfolders.append("test_neg")
#subfolders.append("tr_n1000")
#subfolders.append("tr_n5000")
#subfolders.append("tr_n10000")
#subfolders.append("train_neg")
#subfolders.append("train_pos")
#subfolders.append("unlabeled_train")
#subfolders.append("unlabeled_test")

f_kinds = list()
f_kinds.append("cm")
f_kinds.append("md")

################################################################# WHY ???????? subfolders is empty !!!!!!
for sf in subfolders:
    if re.match('unlab', sf) is None:
        fold = "ct_" + sf
    else:
        fold = sf
    #print (join("ct_input",fold))
    all_elt = listdir(join(outFolders, fold))
    all_files = list()
    #print (all_elt)
    #for elt in all_elt:
    #if isfile(elt):
    all_subfiles = listdir(join("ct_input", join(fold, all_elt[0])))
    #            print (all_subfiles)
    for sfile in all_subfiles:
        file_id = re.findall('^[0-9]+', sfile)[0]
        #        print (file_id)
        all_files.append(file_id)

    if not (fold == sf):

        #Access in post_sally_ct bidule and swap the comment for the appropriate label
        for k in f_kinds:

            path = join(folder, join(sf, sf + "_" + k + ".libsvm"))
            with open(path, 'r') as f:
                all_file_lines = f.readlines()
            #print (len(all_files),"||",len(all_file_lines))
            #print (all_files[-1],"-",all_file_lines[-1])
            new_lines = list()
            for line_nb in range(0, len(all_file_lines) - 1):
                #print ("------------------------")
                #print ( all_file_lines[line_nb])
                new_lines.append(
                    re.sub(
                        '\#[^\n]+', "#" + all_files[line_nb],
                        all_file_lines[line_nb]
                    )
                )
                #print ( re.match('pos',sf),sf )
                if not (re.search('pos', sf) is None):
                    new_lines[-1] = (re.sub('^\S+\s', "1 ",
                                            new_lines[-1])).replace('\n', '')
                else:
                    new_lines[-1] = (re.sub('^\S+\s', "0 ",
                                            new_lines[-1])).replace('\n', '')
                #print (new_lines[-1])
            #opened_file= open(path,'w')
            #for line in new_lines:
            #    print(line.replace('\n',''),f=opened_file)
            #opened_file.close()
