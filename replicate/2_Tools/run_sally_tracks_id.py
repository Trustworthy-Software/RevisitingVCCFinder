#!/usr/bin/env python3
####################################################################
#
#		Launches sally file by file to keep track of the commit id
#		From 3_Intermediary_Steps/1_Extraction to  3_Intermediary_Steps/4_Post_sally
#		Needs post processing recomposing step done in ./recompose.sh before feeding ML
#
######################################################################

import shutil
import re
import time
from os import system, listdir, makedirs, rmdir, rename, remove
from os.path import join, isfile, exists


print ("######Run SALLY Tracking Commit ID######")

root_dir = "../3_IntermediarySteps"
##################################################
#
#	define input Path and stop if does not exists
#
#	folder hierarchy in input is either: 
#		1_Extraction/test_data/test_neg/commit_files
#	or
#		1_Extraction/unlabeled_(train|test)/commit_files
#	referred as
#		1_Extraction/folder_lvl1/folder_lvl2(/folder_lvl3)
##################################################
input_path = join(root_dir, "1_Extraction")
if not exists(input_path):
    puts("ERROR ---------- No Input File")
    sys.exit(404)

##Create a temporary work folder for oneLine SALLY processing before rebuilding files in 4_Post_Sally
work_dir = "3_OneLine"
work_path = join(root_dir, work_dir)
if exists(work_path):
    shutil.rmtree(work_path)
makedirs(work_path)

#Create_temp end_folder for already processed lines per file
pre_sally_path = join(root_dir, "2_Pre-Sally")
if exists(pre_sally_path):
    shutil.rmtree(pre_sally_path)
makedirs(pre_sally_path)

#output folder
out_folder = "../3_IntermediarySteps/4_Post_Sally"
if exists(out_folder):
    shutil.rmtree(out_folder)
makedirs(out_folder)

last_time = time.time()


counter = 0
th_counter = 0

all_lvl1_folders = listdir(input_path)
##########################
#	Starting use of Sally
##########################
for lvl1_folder in all_lvl1_folders:
    #In data level folder
    print ('Lvl1 Folder name: ',lvl1_folder)
    ##lvl2_elements_list holds all the files 
    lvl1_path = join(input_path, lvl1_folder)
    lvl2_elements_list = listdir(lvl1_path)
    

    files_parent_folder_name = lvl1_folder
    out_file_path = ""
    out_file = ""
    #if the file does not exist, make a dull one #USELESS?
    if not (isfile(lvl2_elements_list[0])):
        out_file_path = join(out_folder, lvl1_folder + ".libsvm")
        out_file = open(out_file_path, 'w')
        out_file.close()

    for lvl2_element in lvl2_elements_list:
        # In contains folder if pos of neg, contains files if unlabeled
        loop_on_file = list()
        lvl2_element_path = join(lvl1_path, lvl2_element)
        if isfile(lvl2_element_path):
            #case_unlabeled
            loop_on_file.append(lvl2_element)
        else:
            #case VCC or Fix
            out_file_path = join(out_folder, lvl2_element + ".libsvm")
            out_file = open(out_file_path, 'w')
            out_file.close()
            for unit in listdir(lvl2_element_path):
                unit_path = join(lvl2_element_path, unit)
                files_parent_folder_name = lvl2_element
                loop_on_file.append(join(lvl2_element, unit))
																#DEBUG
        #loop_on_files contains files
        #will only hold one element in unlabeled case but several in test/training case
        for file in loop_on_file:
            ##cp file to ../3_IxxSxx/2_Pre-Sally to be individually sally-ed 
            path_to_file = join(lvl1_path, file)
            shutil.copy(path_to_file, pre_sally_path)
            ##extract id
            id = (re.findall('[0-9]{6,10}', file))[0]
            ##sally-it  to ../3_IxxSxx/3_OneLine aka workpath  ###  find workdir/sally_input/ -type f -exec cat {} + |
            cmd = "sally -i lines -o libsvm --vect_embed bin -d' ' -g tokens " + pre_sally_path + "/* " + work_path + "/one_line.libsvm"
            system(cmd)
            ##read sally output in file
            with open(work_path + '/one_line.libsvm', 'r') as f:
                post_sally_lines = f.readlines()
            ##correct what shall be corrected: label and id
            if not (len(post_sally_lines) == 1):
                print(
                    '------------------ERROR --------------\n',
                    post_sally_lines,
                    "--------------Not one line exactly------------------------"
                )
                sys.exit(0)
            ##add_ commit database identifier
            new_line = (re.sub('#[^\n]+', "#" + id,
                               post_sally_lines[0])).replace('\n', '')
            ##check folder name for pos or neg
            label = 0
            if re.search('pos', lvl1_folder):
                #pos
                label = 1
            else:
                #neg or unlabeled
                label = 0
            new_line = re.sub('^\S+\s', str(label) + " ", new_line)
            
            #Why twice ?
            #check top_input for pos or neg
            label = 0
            if re.search('pos', lvl2_element):
                #pos
                label = 1
            else:
                #neg or unlabeled
                label = 0
            new_line = re.sub('^\S+\s', str(label) + " ", new_line)
            
            ###copy-it to the bigger folder
            #print (out_file_path)
            output_file = open(out_file_path, 'a+')
            print(new_line, file=output_file)
            output_file.close()
            file_to_close = listdir(pre_sally_path)[0]
            remove(join(pre_sally_path, file_to_close))
            #    print ("-",end=' ')
            if counter == 9000:
                th_counter += 1
                this_time = time.time()
                #print (str(th_counter*9)+"000 treated\t"+str((this_time-last_time))[0:4]+" sec")					#DEBUG
                last_time = this_time
                counter = 0
            counter += 1
        ##move the bigger folder accordingly
    #shutil.copy(out_file_path,temp_folder)
