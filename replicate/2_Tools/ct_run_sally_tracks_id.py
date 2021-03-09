###############################################################
#
#	  Aim launch sally file by file to keep track of the id
#
#	From ct_input after being organized to post_sally_ct_input
#		....do not forget to recompose
#
#		The output is ../3_IntermediarySteps/ct_4_Post_Sally
################################################################

import shutil
import time
import re
from os import system, listdir, makedirs, rmdir, rename, remove
from os.path import join, isfile, exists

#Supposedly everything is set in ct_input/X/cm or /md
start_time = time.time()

##Create a temp directory
work_dir = "../3_IntermediarySteps"
#if exists(work_dir):
#    shutil.rmtree(work_dir)
#makedirs(work_dir)

#Use it ? Now yes :-)
input_path = join(work_dir, "ct_2_Pre_Sally")
if exists(input_path):
    shutil.rmtree(input_path)
makedirs(input_path)

#One_line aid repo
oneLine_repo = join(work_dir, "ct_3_oneLine")
if exists(oneLine_repo):
    shutil.rmtree(oneLine_repo)
makedirs(oneLine_repo)

#create_temp end_folder
temp_folder = join(work_dir, "ct_4_Post_Sally")
if exists(temp_folder):
    shutil.rmtree(temp_folder)
makedirs(temp_folder)

#input_top_folder="ct_input" ###### OLD FOLDER ARCHITECTURE
input_top_folder = "../3_IntermediarySteps/ct_1_Generated"
all_inputs = listdir(input_top_folder)
##for each ct_input/X/Y fill up a file with each subfile from ct_input/X/Y where
##X
for top_input in all_inputs:
    type_path = join(input_top_folder, top_input)
    type_folders = listdir(type_path)
    #print(top_input, "|| Positive case? ", re.search("pos", top_input))
    ##Y
    for type in type_folders:
        file_path = join(type_path, type)
        file_list = listdir(file_path)

        type_file_path = join(temp_folder, top_input + "_" + type + ".libsvm")
        type_file = open(type_file_path, 'w')
        type_file.close()
        for file in file_list:
            ###cp file  #shutil.copy(file,dest)
            path_to_file = join(file_path, file)
            shutil.copy(path_to_file, input_path)
            new_name = join(input_path, "input_file")
            rename(join(input_path, file), new_name)
            ###extract id
            id = (re.findall('^[0-9]+', file))[0]
            ###sally-it   ###  find workdir/sally_input/ -type f -exec cat {} + |
            #system('pwd')
            #system('ls workdir/sally_input')
            #sally_command="sally -i lines -o libsvm --vect_embed bin -d' ' -g tokens workdir/sally_input/* workdir/one_line.libsvm"  #Old File Architecture
            sally_command = "sally -i lines -o libsvm --vect_embed bin -d' ' -g tokens " + input_path + "/* " + oneLine_repo + "/one_line.libsvm"
            system(sally_command)
            ###read sally output in file
            with open(join(oneLine_repo, 'one_line.libsvm'), 'r') as f:
                post_sally_lines = f.readlines()
            ###correct what shall be corrected: label and id
            if not (len(post_sally_lines) == 1):
                print(
                    '------------------ERROR --------------\n',
                    post_sally_lines,
                    "--------------Not one line exactly------------------------"
                )
            ####add_ id
            new_line = (re.sub('#[^\n]+', "#" + id,
                               post_sally_lines[0])).replace('\n', '')
            ####check top_input for pos or neg
            label = 0
            if re.search('pos', top_input):
                #pos
                label = 1
            else:
                #neg or unlabeled
                label = 0
            new_line = re.sub('^\S+\s', str(label) + " ", new_line)
            ###copy-it to the bigger folder
            temp_type_file = open(type_file_path, 'a+')
            print(new_line, file=temp_type_file)
            temp_type_file.close()
            remove(new_name)
        ##move the bigger folder accordingly  ## No NEED in new file archiecture as files are indexed to follow procedure
        #shutil.copy(type_file_path,temp_folder)
        #system("ls -l workdir/temp_file")
    ##Do the junctions between pos and negs from work_dir
    all_temps = listdir(temp_folder)
    ##train: ct_tr_n1000 and cd tr_pos for cm and md
    ##cm
    #file=

    ##md

    #tr1000
    ##md
    ##md
    #tr5000
    #tr10000
    #test

#Ending time
end_time = time.time()
print("----%s minutes----" % ((end_time - start_time) / 60))
