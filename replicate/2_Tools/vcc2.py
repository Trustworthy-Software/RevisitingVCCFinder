#!/usr/bin/env python3
########################################################################
#
#	This program will interact with the postgresql database
#	to determine,1st ,the IDs of commit belonging to each category
#	training pos, training_neg, test_pos, test_neg.
#		->These are saved in 1_Data/commit_refs
#	And then will extract the commits and their features from the DB
#	so to populate accordingly-to-the-category named files in
#		-> 3_Intermediary_Steps/1_Extraction
########################################################################
import psycopg2
import psycopg2.extras
import sys
import os
import numpy as np
import sklearn.preprocessing
from functools import partial
import shutil
import re
import copy

DBNAME = "vccfinder"
USER = None
HOST = "/var/run/postgresql"  # hostname, IP or unix socket dir
PASSWORD = None




#####################################################
#
#	Will clean a folder removing 
#	all subdirs first and then the desired folder
#
#####################################################
def clean_folder(folder_to_clean):
    if os.path.exists(folder_to_clean):
        for name in os.listdir(folder_to_clean):
            os.remove(os.path.join(folder_to_clean, name))
        os.rmdir(folder_to_clean)

#Set up Error Directory
OUTPUT_DIR = '../3_IntermediarySteps/0_Error_Save_Dir/'
clean_folder(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print ("-Connect to database-")
PG = psycopg2.connect(f"dbname={DBNAME} host={HOST}")
# We have to let psycopg2 that hstore should be casted to dict (not enabled by default)
psycopg2.extras.register_hstore(PG, globally=True)



CODE_FILE_EXTENSIONS = ['.c', '.cpp', '.cc', '.h', '.java']
def is_code_file(extension_list):
    for i in extension_list:
        if i in CODE_FILE_EXTENSIONS:
            return True
    return False
EXTENSIONS_FINDER = re.compile('(\.\w+)')
########################################################
#
#		filter from patch files 
#	that does not contain interesting code:
#	Files for which there is no modification in 
#		c, cpp,  .h file or java
#
#
#########################################################
def filter_commit(commit):
    patch_commit = commit['patch']
    patch_array = patch_commit.split('\n')
    new_commit_patch = list()
    keep_skipping = False
    for line in patch_array:
        if ("diff --git" in line):
            # So... this is a change to a file
            # collect everything that looks like a file extension
            matches = EXTENSIONS_FINDER.findall(line)
            # Does this line mention a filetype we are interested in ?
            if is_code_file(matches):
                keep_skipping = False
                new_commit_patch.append(line)
                continue
            else:
                keep_skipping = True
                continue
        if keep_skipping:
            continue
        else:
            new_commit_patch.append(line)
            continue
    commit['patch'] = "\n".join(new_commit_patch)
    return commit

def get_all_cve():
    dict_cur = PG.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    dict_cur.execute("SELECT * FROM export.cves")
    # CHECK: will that return one dict or an iterable of dict ?
    res = dict()
    for cve in dict_cur.fetchall():
        res[cve['id']] = cve
    dict_cur.close()
    return res

def get_all_repo():
    dict_cur = PG.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    dict_cur.execute("SELECT * FROM export.repositories")
    # CHECK: will that return one dict or an iterable of dict ?
    res = dict()
    for repo in dict_cur.fetchall():
        res[repo['id']] = repo
    dict_cur.close()
    return res

def get_all_commit():
    dict_cur = PG.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    dict_cur.execute("SELECT * FROM export.commits")
    # CHECK: will that return one dict or an iterable of dict ?
    res = dict()
    for commit in dict_cur.fetchall():
        treated_commit = filter_commit(commit)
        res[commit['id']] = treated_commit  #commit
    dict_cur.close()
    return res


# load all cves, repositories and commits into RAM
print ("-Loading CVEs in RAM-")
cves = get_all_cve()
print ("-Loading repositories in RAM-")
repos = get_all_repo()
print ("-Loading commits in RAM-")
commits = get_all_commit()

#for the purpose of categorising commits per feature
#    we will record a range of values for each feature
#dictionnaries are per feature: feature, max
maxes = dict()
mins = dict()
means = dict()
std_devs = dict()

#code_metrics and meta_data are dictionnaries that list features:
# Keys are str: feature_name
# Values are: function that takes a commit_id and return the relevant value
code_metrics = dict()
code_metrics['additions'] = lambda x: commits[x].get('additions', None)
code_metrics['deletions'] = lambda x: commits[x].get('deletions', None)
code_metrics['hunk_count'] = lambda x: commits[x].get('hunk_count', None)

############################################
#
#	populate global list of keywords in patch for features
#
# 	 patch_keywords needs special handling
#############################################
all_kws = dict()
for my_id in commits:
    kws = commits[my_id].get('patch_keywords', None)
    if kws:
        for kw in kws:
            if kw not in all_kws:
                all_kws[kw] = dict()


############################################
#
#	For a specific commit 
#	get the value associated 
#	with one specific keyword
#
############################################
def get_kw(commit_id, kw):
    commit = commits[commit_id]
    if not commit:
        return None
    kws = commit.get('patch_keywords', None)
    if not kws:
        return None
    if kw not in kws:
        return None
    return int(kws[kw])


for kw in all_kws:
    feat_name = 'kw_' + kw
    code_metrics[feat_name] = partial(get_kw, kw=kw)

meta = dict()
meta['number_commits'] = lambda x: repos[commits[x]['repository_id']
                                         ].get('commits_count', None)
meta['number_unique_contributors'] = lambda x: repos[commits[x][
    'repository_id']].get('distinct_authors_count', None)
meta['contributions_in_project'] = lambda x: commits[
    x].get('author_contributions_percent', None)
meta['past_changes'] = lambda x: commits[x].get('past_changes', None)
meta['future_changes'] = lambda x: commits[x].get('future_changes', None)
meta['past_different_authors'] = lambda x: commits[
    x].get('past_different_authors', None)
meta['future_different_authors'] = lambda x: commits[
    x].get('future_different_authors', None)


def get_year(date_datetime):
    return int(str(date_datetime)[0:4])

##########################################
#
#	 yields value of all given commits
#	for specific feature
#
##########################################
# iterator over values for a given feature
def gen_feat(feat, commit_ids):
    if feat in code_metrics:
        mapping = code_metrics
    else:
        mapping = meta
    for my_id in commit_ids:
        if mapping[feat](my_id):
            yield mapping[feat](my_id)


##########################################
#
#	Build 10 uniform scalers for each feature
#	to be populated by commits according to 
#	the value associated with this feature 
#	of this commit.
#	Does it for code metrics and meta-data
#	
##########################################
normalisers = dict()
def build_sklearn_scalers():
    global normalisers
    for feat in code_metrics:
        scaler = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=10, output_distribution='uniform'
        )
        scaler.fit(
            np.array(list(gen_feat(feat, commits.keys()))).reshape(-1, 1)
        )
        normalisers[feat] = lambda x, scaler=scaler: int(
            scaler.transform(np.array(x).reshape(-1, 1)) * 100
        )
    for feat in meta:
        scaler = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=10, output_distribution='uniform'
        )
        scaler.fit(
            np.array(list(gen_feat(feat, commits.keys()))).reshape(-1, 1)
        )
        normalisers[feat] = lambda x, scaler=scaler: int(
            scaler.transform(np.array(x).reshape(-1, 1)) * 100
        )

print ("-Building scalers-")
build_sklearn_scalers()

def get_first_digit(x):
    """
    Value -> bin value
    10 possible bins
    x should be < 100
    """
    x = int(x)
    if x < 0:
        return 0
    x = str(x)
    if len(x) == 1:  # less than 10 ?
        return 0
    else:
        return int(x[0])

########################
#
#    For a specific commit given by id
#    Will associated in a String all features
#    with the bag the feature value enabled to scale
#    the commit in.
#    in the form '{feature_name1}:{associated_bag1} {feature_name2}:{associated_bin2} ...'
#    with patch and commit message appended
#
########################
def commit_to_txt(commit_id: int):
    commit = commits[commit_id]
    res = list()
    # TODO: Compute numerical features, one per line or separated by a space
    for feat in code_metrics:
        val = code_metrics[feat](commit_id)
        if val:
            normalised = normalisers[feat](val)
            my_bin = get_first_digit(normalised)
            res.append(f"{feat}::{my_bin}")
    for feat in meta:
        val = meta[feat](commit_id)
        if val:
            normalised = normalisers[feat](val)
            my_bin = get_first_digit(normalised)
            res.append(f"{feat}::{my_bin}")
    # Extract text from patch
    patch = commit.get('patch', None)
    if patch:
        res.extend(patch.split('\n'))
    msg = commit.get('message', None)
    if msg:
        res.extend(msg.split('\n'))
    return ' '.join(res)

###############################
#
#	Accesses to individual commit as text
#	so get save it in a file aside
#
###############################
def commit_to_file(commit_id: int, dir_location=OUTPUT_DIR):
    string = commit_to_txt(commit_id)
    f_out = open(os.path.join(dir_location, str(commit_id)), 'w')
    print(string, file=f_out)
    f_out.close()

##################################
#
#	From the database, accesses to commit
#	that are blamed by another commit
#	for contributing to a vulnerability
#	it is trying to patch. Base of VCC.
#
###################################
def get_all_blamed_commits():
    dict_cur = PG.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    # Sure you need a cartesian product ?
    dict_cur.execute(
        "SELECT C2.* FROM export.commits C, export.commits C2 WHERE C2.id=C.blamed_commit_id "
    )
    res = dict()
    for cve in dict_cur.fetchall():
        res[cve['id']] = cve
    dict_cur.close()
    return res

####################################
#
#	Making sure this commit is accessible
#	calls function to get year out of string
#	with date, month and year
#
####################################
def get_year_from_commitID(commit_id):
    for my_id in commits:
        if commits[my_id].get('id', None) == commit_id:
            break
    year = get_year(commits[my_id].get('author_when'))
    return year

########################################
#
#	Will copy into 1_Data/commits_ref 
#	all the ids for each category according to temporality: 
#	( train/test wrt. year 2011 )
#	and according to if Contributing or Fixing a Vulnerability 
#	( Positive and Negative class resp. )
#
########################################
def generate_commit_id_lists():
    id_dir = "../1_Data/commit_refs"
    if os.path.exists(id_dir):
        for name in os.listdir(id_dir):
            os.remove(os.path.join(id_dir, name))
        os.rmdir(id_dir)
    os.makedirs(id_dir)
    ###############training_pos aka VCC/blamed before 2011
    Res = []
    all_blamed = get_all_blamed_commits()
    for commit_iter in all_blamed:
        if get_year(commits[commit_iter].get('author_when', None)) < 2011:  #1
            Res.append(str(commits[commit_iter].get('id', None)) + '\n')
    string = ''.join(Res)
    string = string[:-1]
    f_out = open(os.path.join(id_dir, "training_pos"), 'w')
    print(string, file=f_out)
    f_out.close()
    ###############training_neg aka Fixing commits before 2011
    Res = []
    for commit_iter in commits:
        if commits[commit_iter].get(
            'blamed_commit_id', None
        ) != None and commits[commit_iter].get(
            'author_when', None
        ) != None and int(
            str(commits[commit_iter].get('author_when', None))[0:4]
        ) < 2011:
            Res.append(str(commits[commit_iter].get('id', None)) + '\n')
    string = ''.join(Res)[:-1]
    f_out = open(os.path.join(id_dir, "training_neg"), 'w')
    print(string, file=f_out)
    f_out.close()
   ###############test_pos aka VCC/blamed after 2011 but before 2015 to be closer to VCCFinder original paper
    Res = []
    for commit_iter in all_blamed:
        if commits[commit_iter].get('author_when', None) != None and int(
            str(commits[commit_iter].get('author_when', None))[0:4]
        ) >= 2011:  #3
            Res.append(str(commits[commit_iter].get('id', None)) + '\n')
    string = ''.join(Res)[:-1]
    f_out = open(os.path.join(id_dir, "test_pos"), 'w')
    print(string, file=f_out)
    f_out.close()
    ###############test_neg aka Fixing Commits after 2011 but before 2015 to be closer to VCCFinder original paper
    Res = []
    for commit_iter in commits:
        if commits[commit_iter].get(
            'blamed_commit_id', None
        ) != None and commits[commit_iter].get(
            'author_when', None
        ) != None and int(
            str(commits[commit_iter].get('author_when', None))[0:4]
        ) >= 2011 and get_year(
            commits[commit_iter].get('author_when', None)
        ) < 2015:  		#4
            Res.append(str(commits[commit_iter].get('id', None)) + '\n')
    string = ''.join(Res)[:-1]
    f_out = open(os.path.join(id_dir, "test_neg"), 'w')
    print(string, file=f_out)
    f_out.close()
    #Those ids are of the commit left and recuperated aside
    shutil.copyfile(
        '../1_Data/backup/unlabeled_train',
        '../1_Data/commit_refs/unlabeled_train'
    )
    shutil.copyfile(
        '../1_Data/backup/unlabeled_test',
        '../1_Data/commit_refs/unlabeled_test'
    )


############################
#
#	This function will enable verifications on the data
#		set in RAM for commits  
#	so to access the IDs in an alternate way 
#		based on the blaming-blamed relationship
#	This step helps understand the data in the dataset	
#		and the complementarity between contributing
#		and fixing commits in the DB. 
#		(e.g. some fix have no blamed_commit field 
#		which can help evaluate the dataset)
############################
def db_link_level_check():
    ########train_pos
    Res=list()
    for commit_iter in commits:
        #print(commits[commit_iter].get('blamed_commit_id',0))
        if commits[commit_iter].get(
            'type', None
        ) == 'blamed_commit' and commits[commit_iter].get(
            'author_when', None
        ) != None and int(
            str(commits[commit_iter].get('author_when', None))[0:4]
        ) < 2011:
            Res.append(str(commits[commit_iter].get('id', None)) + '\n')
    print ("Number of blamed commit before 2011 ",len(Res))
    string = ''.join(Res)[:-1]
    error_dir = "../3_IntermediarySteps/0_Error_Save_Dir"
    error_file = open(
                    os.path.join(error_dir,"train_pos"),
                    'w'
                )
    print(string , file=error_file)
    error_file.close()

    #######train_neg
    Res=list()  
    for commit_iter in commits:
        #print(commits[commit_iter].get('blamed_commit_id',0))
        if commits[commit_iter].get(
            'type', None
        ) == 'fixing_commit' and commits[commit_iter].get(
            'author_when', None
        ) != None and int(
            str(commits[commit_iter].get('author_when', None))[0:4]
        ) < 2011:
            Res.append(str(commits[commit_iter].get('id', None)) + '\n')
    print ("Number of fixing commit before 2011: ",len(Res))
    string = ''.join(Res)[:-1]

    error_file = open(
                    os.path.join(error_dir,"train_neg"),
                    'w'
                )
    print(string , file=error_file)
    error_file.close()

    #######test_pos
    Res=list()
    for commit_iter in commits:
        #print(commits[commit_iter].get('blamed_commit_id',0))
        if commits[commit_iter].get(
            'type', None
        ) == 'blamed_commit' and commits[commit_iter].get(
            'author_when', None
        ) != None and int(  
            str(commits[commit_iter].get('author_when', None))[0:4]
        ) > 2011 and int(
            str(commits[commit_iter].get('author_when', None))[0:4]
        ) < 2015:
            Res.append(str(commits[commit_iter].get('id', None)) + '\n')
    print ("Number of blamed commit after 2011: ",len(Res))
    string = ''.join(Res)[:-1]
    error_file = open( 
                    os.path.join(error_dir,"test_pos"),
                    'w'
                )
    print(string , file=error_file)
    error_file.close()

    #######test_neg
    Res=list()
    for commit_iter in commits:
        #print(commits[commit_iter].get('blamed_commit_id',0))
        if commits[commit_iter].get( 
            'type', None
        ) == 'fixing_commit' and commits[commit_iter].get(
            'author_when', None
        ) != None and int(
            str(commits[commit_iter].get('author_when', None))[0:4]
        ) > 2011 and int(
            str(commits[commit_iter].get('author_when', None))[0:4]
        ) < 2015:
            Res.append(str(commits[commit_iter].get('id', None)) + '\n')
    print ("Number of fixing commit after 2011: ",len(Res))
    string = ''.join(Res)[:-1]
    error_file = open(
                    os.path.join(error_dir,"test_neg"),
                    'w'
                )
    print(string , file=error_file)
    error_file.close()



#####################################################################################
#Get value on how much the db is populating in directly linked in fixing and contributing commits 
db_link_level_check()

#Populate 1_Data/commit_refs with files holding the IDs
generate_commit_id_lists()


##prepare for coTraining
##split features in meta/code
##save the result in csv format
def strip_list(list_id):
    newList = list()
    for elt in list_id:
        #print('|',elt,'|')
        newList.append(int(elt.rstrip()))
        #print('|',elt,'|')
    return newList

#########################################################
#
#	Use ID lists to extract commits from DB
#	and split them into files related to code etrics 
#	and files related to meta-data 
#
#
#########################################################
def make_ct_lists():
    ######open and clean dir
    ct_dir = "../3_IntermediarySteps/ct_1_Generated"
    ct_tr_p = "ct_train_pos"
    ct_tr_n = "ct_train_neg"
    ct_tr_n1 = "ct_tr_n1000"
    ct_tr_n5 = "ct_tr_n5000"
    ct_tr_n10 = "ct_tr_n10000"
    ct_te_p = "ct_test_pos"
    ct_te_n = "ct_test_neg"
    unl_dir = "unlabeled_test"
    unl_train = "unlabeled_train"
    if os.path.exists(ct_dir):
        shutil.rmtree(ct_dir)
    os.makedirs(ct_dir)
    clean_folder(os.path.join(unl_train, "cm"))
    clean_folder(os.path.join(unl_train, "md"))
    clean_folder(unl_train)
    clean_folder(os.path.join(unl_dir, "cm"))
    clean_folder(os.path.join(unl_dir, "md"))
    clean_folder(unl_dir)
    clean_folder(ct_tr_p)
    clean_folder(ct_tr_n)
    clean_folder(ct_tr_n1)
    clean_folder(ct_tr_n5)
    clean_folder(ct_tr_n10)
    clean_folder(ct_te_p)
    clean_folder(ct_te_n)
    clean_folder(ct_dir)
    os.makedirs(ct_dir)
    os.makedirs(os.path.join(ct_dir, unl_train))
    os.makedirs(os.path.join(ct_dir, os.path.join(unl_train, "cm")))
    os.makedirs(os.path.join(ct_dir, os.path.join(unl_train, "md")))
    os.makedirs(os.path.join(ct_dir, unl_dir))
    os.makedirs(os.path.join(ct_dir, os.path.join(unl_dir, "cm")))
    os.makedirs(os.path.join(ct_dir, os.path.join(unl_dir, "md")))
    os.makedirs(os.path.join(ct_dir, ct_tr_p))
    os.makedirs(os.path.join(ct_dir, ct_tr_n))
    os.makedirs(os.path.join(ct_dir, ct_tr_n1))
    os.makedirs(os.path.join(ct_dir, ct_tr_n5))
    os.makedirs(os.path.join(ct_dir, ct_tr_n10))
    os.makedirs(os.path.join(ct_dir, ct_te_p))
    os.makedirs(os.path.join(ct_dir, ct_te_n))
    ######access commits ids
    #tr_pos
    with open("../1_Data/commit_refs/training_pos") as f:
        train_pos_list = f.readlines()
        train_pos_list = strip_list(train_pos_list)
    #tr_neg
    with open("../1_Data/commit_refs/training_neg") as f:
        train_neg_list = strip_list(f.readlines())
    #te_pos
    with open("../1_Data/commit_refs/test_pos") as f:
        test_pos_list = strip_list(f.readlines())
    #te_neg
    with open("../1_Data/commit_refs/test_neg") as f:
        test_neg_list = strip_list(f.readlines())
    #tr_neg with 1000 unlabeled to see effect of introducing diversity in this set     
    with open("../1_Data/unlab_train_extended/unlab_train_1000.txt") as f:
        train_neg_1000 = strip_list(f.readlines())
        train_neg_1000.extend(train_neg_list)
    #tr_neg with 5000 unlabeled to see effect of introducing diversity in this set    
    with open("../1_Data/unlab_train_extended/unlab_train_5000.txt") as f:
        train_neg_5000 = strip_list(f.readlines())
        train_neg_5000.extend(train_neg_list)
    #tr_neg with 10000 unlabeled to see effect of introducing diversity in this set    
    with open("../1_Data/unlab_train_extended/unlab_train_10000.txt") as f:
        train_neg_10000 = strip_list(f.readlines())
        train_neg_10000.extend(train_neg_list)
    ######fetch data from data in RAM
    unlab_tr_counter = 0
    unlab_te_counter = 0

    for commit_id in commits:
        ##code_metrics and text
        c_m = list()
        c_m.append(str(commit_id))
        for feat in code_metrics:
            associated_value = str(code_metrics[feat](commit_id))
            c_m.append(f"{feat}::{associated_value}")
        patch = (commits[commit_id].get('patch', None)).split('\n')
        ##meta_data and commit message
        m_d = list()
        m_d.append(str(commit_id))
        for feat in meta:
            associated_value = str(meta[feat](commit_id))
            m_d.append(f"{feat}::{associated_value}")
        msg = (commits[commit_id].get('message', None)).split('\n')

        ##save accordingly
        #edit save_path
        save_path = ''
        if commit_id in train_pos_list:
            save_path = os.path.join(ct_dir, ct_tr_p)
        elif commit_id in train_neg_list:
            save_path = os.path.join(ct_dir, ct_tr_n)
        elif commit_id in test_pos_list:
            save_path = os.path.join(ct_dir, ct_te_p)
        elif commit_id in test_neg_list:
            save_path = os.path.join(ct_dir, ct_te_n)
        elif commit_id in train_neg_1000:
            save_path = os.path.join(ct_dir, ct_tr_n1)
        elif commit_id in train_neg_5000:
            save_path = os.path.join(ct_dir, ct_tr_n5)
        elif commit_id in train_neg_10000:
            save_path = os.path.join(ct_dir, ct_tr_n10)
        unlab_train_savepathes = list()
        unlab_train_savepathes.append(os.path.join(ct_dir, ct_tr_n1))
        unlab_train_savepathes.append(os.path.join(ct_dir, ct_tr_n5))
        unlab_train_savepathes.append(os.path.join(ct_dir, ct_tr_n10))
        #save in files
        #c_m
        if not (save_path == ''):
            cm_file = open(
                os.path.join(save_path, (str(commit_id)) + "_code_metrics"),
                'w'
            )
            c_m.extend(patch)
            print(' '.join(c_m), file=cm_file)
            cm_file.close()
            #m_d
            md_file = open(
                os.path.join(save_path, (str(commit_id)) + "_meta_data"), 'w'
            )
            m_d.extend(msg)
            print(' '.join((m_d)), file=md_file)
            md_file.close()
            #train_neg, then add as well in the 3 unlabeled_train
            if commit_id in train_neg_list:
                for u_t_sp in unlab_train_savepathes:
                    #print ('--',u_t_sp)
                    save_path = u_t_sp
                    cm_file = open(
                        os.path.join(
                            save_path, (str(commit_id)) + "_code_metrics"
                        ), 'w'
                    )
                    print(' '.join(c_m), file=cm_file)
                    cm_file.close()
                    #m_d
                    md_file = open(
                        os.path.join(
                            save_path, (str(commit_id)) + "_meta_data"
                        ), 'w'
                    )
                    print(' '.join((m_d)), file=md_file)
                    md_file.close()
            if commit_id in train_neg_5000:
                save_path = os.path.join(ct_dir, ct_tr_n5)
                cm_file = open(
                    os.path.join(
                        save_path, (str(commit_id)) + "_code_metrics"
                    ), 'w'
                )
                print(' '.join(c_m), file=cm_file)
                cm_file.close()
                md_file = open(
                    os.path.join(save_path, (str(commit_id)) + "_meta_data"),
                    'w'
                )
                print(' '.join((m_d)), file=md_file)
                md_file.close()
            if commit_id in train_neg_10000:
                save_path = os.path.join(ct_dir, ct_tr_n10)
                cm_file = open(
                    os.path.join(
                        save_path, (str(commit_id)) + "_code_metrics"
                    ), 'w'
                )
                print(' '.join(c_m), file=cm_file)
                cm_file.close()
                md_file = open(
                    os.path.join(save_path, (str(commit_id)) + "_meta_data"),
                    'w'
                )
                print(' '.join((m_d)), file=md_file)
                md_file.close()

        #unlabeled
        else:
            #unlabeled test
            if get_year(
                commits[commit_id].get('author_when', None)
            ) > 2011 and get_year(
                commits[commit_id].get('author_when', None)
            ) < 2015:  # get_year(commits[commit_iter].get('author_when',None))<2011
                group = str(commit_id)[0:4]
                path1 = os.path.join(
                    ct_dir, os.path.join(unl_dir, os.path.join("cm", group))
                )
                path2 = os.path.join(
                    ct_dir, os.path.join(unl_dir, os.path.join("md", group))
                )
                if not os.path.exists(path1):
                    os.makedirs(path1)
                if not os.path.exists(path2):
                    os.makedirs(path2)
                unlabeled_test_path = os.path.join(
                    path1,
                    str(commit_id) + "_unlabeled_test_cm"
                )
                unl_file = open(
                    os.path.join(path1,
                                 str(commit_id) + "_unlabeled_test_cm"), 'w'
                )
                c_m.extend(patch)
                print(' '.join(c_m), file=unl_file)
                unl_file.close()

                unl_file = open(
                    os.path.join(path2,
                                 str(commit_id) + "_unlabeled_test_md"), 'w'
                )
                m_d.extend(msg)
                print(' '.join(m_d), file=unl_file)
                unl_file.close()
                unlab_te_counter += 1
            #unlabeled train    
            elif get_year(commits[commit_id].get('author_when', None)) <= 2011:
                group = str(commit_id)[0:4]
                path_cm = os.path.join(
                    ct_dir, os.path.join(unl_train, os.path.join("cm", group))
                )
                path_md = os.path.join(
                    ct_dir, os.path.join(unl_train, os.path.join("md", group))
                )
                if not os.path.exists(path_cm):
                    os.makedirs(path_cm)
                if not os.path.exists(path_md):
                    os.makedirs(path_md)

                unl_file = open(
                    os.path.join(
                        path_cm,
                        str(commit_id) + "_unlabeled_train_cm"
                    ), 'w'
                )
                c_m.extend(patch)
                print(' '.join(c_m), file=unl_file)
                unl_file.close()
                unl_file = open(
                    os.path.join(
                        path_md,
                        str(commit_id) + "_unlabeled_train_md"
                    ), 'w'
                )
                m_d.extend(msg)
                print(' '.join(m_d), file=unl_file)
                unl_file.close()
                unlab_tr_counter += 1



################################################
#
#	Begin extraction of features from DB based on
#	commit IDs present in 1_Data/commit_refs
#	extraction to 3_Intermediary_Steps/1_Extraction
#
#################################################
print ("#################################")
print ("#Extraction of Features from DB#")
print ("#################################")


train_dir = "../3_IntermediarySteps/1_Extraction/training_data"
train_pos_dir = "../3_IntermediarySteps/1_Extraction/training_data/training_pos"
train_neg_dir = "../3_IntermediarySteps/1_Extraction/training_data/training_neg_reg"
train_neg_1000 = "../3_IntermediarySteps/1_Extraction/training_data/training_neg_1000"
train_neg_5000 = "../3_IntermediarySteps/1_Extraction/training_data/training_neg_5000"
train_neg_10000 = "../3_IntermediarySteps/1_Extraction/training_data/training_neg_10000"
test_dir = "../3_IntermediarySteps/1_Extraction/test_data"
test_pos_dir = "../3_IntermediarySteps/1_Extraction/test_data/test_pos"
test_neg_dir = "../3_IntermediarySteps/1_Extraction/test_data/test_neg"
unlabeled_test_data = "../3_IntermediarySteps/1_Extraction/unlabeled_test"
unlabeled_train_dir = "../3_IntermediarySteps/1_Extraction/unlabeled_train"
type_test_data="../3_IntermediarySteps/1_Extraction/type_test_data"
type_train_data="../3_IntermediarySteps/1_Extraction/type_training_data"
type_train_pos=os.path.join(type_train_data,"type_train_pos")
type_train_neg=os.path.join(type_train_data,"type_train_neg")
type_test_pos=os.path.join(type_test_data,"type_test_pos")
type_test_neg=os.path.join(type_test_data,"type_test_neg")
#clean from previous use
clean_folder(train_pos_dir)
clean_folder(train_neg_dir)
clean_folder(train_neg_1000)
clean_folder(train_neg_5000)
clean_folder(train_neg_10000)
clean_folder(train_dir)
clean_folder(test_pos_dir)
clean_folder(test_neg_dir)
clean_folder(test_dir)
clean_folder(unlabeled_test_data)
clean_folder(unlabeled_train_dir)
clean_folder(type_train_pos)
clean_folder(type_train_neg)
clean_folder(type_test_pos)
clean_folder(type_test_neg)
clean_folder(type_train_data)
clean_folder(type_test_data)
#generate folder trees for later use
os.makedirs(train_dir)
os.makedirs(train_pos_dir)
os.makedirs(train_neg_dir)
os.makedirs(train_neg_1000)
os.makedirs(train_neg_5000)
os.makedirs(train_neg_10000)
os.makedirs(test_dir)
os.makedirs(test_neg_dir)
os.makedirs(test_pos_dir)
os.makedirs(unlabeled_test_data)
os.makedirs(unlabeled_train_dir)
os.makedirs(type_train_data)
os.makedirs(type_test_data)
os.makedirs(type_train_pos)
os.makedirs(type_train_neg)
os.makedirs(type_test_pos)
os.makedirs(type_test_neg)



#fetch ids for training and test
with open("../1_Data/commit_refs/training_pos") as f:
    train_pos_list = f.readlines()
with open("../1_Data/commit_refs/training_neg") as f:
    train_neg_list = f.readlines()
with open("../1_Data/commit_refs/test_neg") as f:
    test_neg_list = f.readlines()
with open("../1_Data/commit_refs/test_pos") as f:
    test_pos_list = f.readlines()
with open("../1_Data/commit_refs/unlabeled_test") as f:
    unlabeled_test_list = f.readlines()
with open("../1_Data/commit_refs/unlabeled_train") as f:
    unlabeled_train_list = f.readlines()
with open("../1_Data/unlab_train_extended/unlab_train_1000.txt") as f:
    unlab_ext_1000 = f.readlines()
with open("../1_Data/unlab_train_extended/unlab_train_5000.txt") as f:
    unlab_ext_5000 = f.readlines()
with open("../1_Data/unlab_train_extended/unlab_train_10000.txt") as f:
    unlab_ext_10000 = f.readlines()

#get list of commits recuperated by alternate way for db check and db extraction check
with open("../3_IntermediarySteps/0_Error_Save_Dir/train_neg") as f:
     type_train_neg_list = f.readlines()
with open("../3_IntermediarySteps/0_Error_Save_Dir/train_pos") as f:
     type_train_pos_list = f.readlines()
with open("../3_IntermediarySteps/0_Error_Save_Dir/test_neg") as f:
     type_test_neg_list = f.readlines()
with open("../3_IntermediarySteps/0_Error_Save_Dir/test_pos") as f:
     type_test_pos_list = f.readlines()




##get list of all commits related to training
#train_pos
for my_id in train_pos_list:
    my_id = int(my_id.rstrip())
    commit_to_file(my_id, train_pos_dir)
#train_neg and extension to test diversity introduction
for my_id in train_neg_list:
    my_id = int(my_id.rstrip())
    commit_to_file(my_id, train_neg_dir)
    commit_to_file(my_id, train_neg_1000)
    commit_to_file(my_id, train_neg_5000)
    commit_to_file(my_id, train_neg_10000)
#unlabeled_test
for my_id in unlabeled_test_list:
    my_id = int(my_id.strip())
    commit_to_file(my_id, unlabeled_test_data)
#Open the right list
for my_id in unlab_ext_1000:
    my_id = int(my_id.strip())
    commit_to_file(my_id, train_neg_1000)
for my_id in unlab_ext_5000:
    my_id = int(my_id.strip())
    commit_to_file(my_id, train_neg_5000)
for my_id in unlab_ext_10000:
    my_id = int(my_id.strip())
    commit_to_file(my_id, train_neg_10000)

#Unlabeled_train
for my_id in unlabeled_train_list:
    my_id = int(my_id.strip())
    commit_to_file(my_id, unlabeled_train_dir)
#Test_neg
for my_id in test_neg_list:
    commit_to_file(int(my_id.rstrip()), test_neg_dir)
#Test_pos    
for my_id in test_pos_list:
    commit_to_file(int(my_id.rstrip()), test_pos_dir)

#Type
for my_id in type_test_pos_list:
    commit_to_file(int(my_id.rstrip()), type_test_pos)
for my_id in type_test_neg_list:
    commit_to_file(int(my_id.rstrip()), type_test_neg)
for my_id in type_train_pos_list:
    commit_to_file(int(my_id.rstrip()), type_train_pos)
for my_id in type_train_neg_list:
    commit_to_file(int(my_id.rstrip()), type_train_neg)



print("-All files generated for replication-")

print("###################Make co training list#######################")
make_ct_lists()
