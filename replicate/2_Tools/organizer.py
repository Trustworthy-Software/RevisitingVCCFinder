#################################################################################
#                    The point of this file is to look at generated files
#			and reveal a potential mistake in the generation
#		plus it will reorganize the generated files in a compliant way for Sally
#################################################################################
import shutil
from os import listdir, makedirs, rmdir
from os.path import isfile, join
import re

#ct_path="ct_input"
ct_path = "../3_IntermediarySteps/ct_1_Generated"

list_folders = listdir(ct_path)

for folder1 in list_folders:
    #create list of files that are not folders
    path = join(ct_path, folder1)
    all_elts = listdir(path)
    all_files = list()
    for elt in all_elts:
        if isfile(join(ct_path, join(folder1, elt))):
            all_files.append(elt)
    #print("Size of ", join(folder1, elt), ":", len(all_files))
    #if needed create cm and md folders or reset them
    if not ("cm" in all_elts):

        #print("--cm not present")
        makedirs(join(path, "cm"))
    if not ("md" in all_elts):
        #print("--md not present")
        makedirs(join(path, "md"))
    #cp files cm for including code, md if including meta
    #print("Unlabeled treatement: ", folder1, "||")
    if (not (re.match('unlab', folder1) is None)):
        #print ("------",len(all_files))
        type_folders = listdir(path)
        #print ("-------",all_top_folders )
        #cm or md
        for type in type_folders:
            type_pathes = join(path, type)
            all_digit_groups = listdir(type_pathes)
            for digit_group in all_digit_groups:
                dg_path = join(type_pathes, digit_group)
                unlab_data_files = listdir(dg_path)
                for data_file in unlab_data_files:
                    shutil.move(join(dg_path, data_file), type_pathes)
                rmdir(dg_path)
    for file_data in all_files:
        #print (path,file_data,"|" ,re.search('code',file_data))
        #if (not (re.match('unlab',folder1) is None)) :
        #print ("------",join(path,file_data))
        #all_top_folders=listdir(join(path,file_data))
        #print ("-------",len(all_top_folders))

        if not re.search('code', file_data) is None:
            #print (file_data,"is a ",folder1,"/cm")
            shutil.move(join(path, file_data), join(path, "cm"))
        if not re.search('meta', file_data) is None:
            #print (file_data,"is a ",folder1,"/md")
            shutil.move(join(path, file_data), join(path, "md"))
        #uncopy files from ct_input/unlab_train/cm/X or md/Y one file up if
        #print ("||",folder1,"||",re.search('unlab',folder1))
        #if (not (re.match('unlab',folder1) is None)) is True :
        #print ("------",join(path,file_data))
        #all_top_folders=listdir(join(path,file_data))
        #print ("-------",len(all_top_folders))
    #print (folder1, type(folder1))
