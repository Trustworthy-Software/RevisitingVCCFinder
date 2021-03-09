#!/usr/bin/env python3
############################################################################
#
#	This script will deal with the prediction on treated data
#	it will produce Figure 2, Figure 3 and Figure 4
#	(resp. replication, Exploration of C parameter influence, 
#	and Exploration over the classifier algorithm)
#
#	It also produces the representation for the exploration over the weight
#	that is a point discussed in the article.
#
#
#	In this form the Flags = "#MARKER_XP_LIST" indicate
#	commented lines to see the influence of increasing the training set
#	with unlabeled progressively. Point considered irrelevant in the paper
#	but let here for further exploration, and as data is available.
#
#	See #REPLICATION for beginning of operations
#	See #C_INFLUENCE
#	See #CLASSWEIGHT_INFLUENCE
#	See #ALGORITHM_INFLUENCE
#	See #EXPLORE_UNLABELED For applying GrountTruth to Unlabeled Test
###############################################################################3
import numpy as np
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_curve
from os import system, listdir, makedirs, rmdir, rename, remove
from os.path import join, isfile, exists, isdir
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import re

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

############################################################################
names = ["regular", "1000", "5000", "10000"]

################################################################
#	save precision,recall,threshold and F1 score  
#		in file 4_Results/1_Data/Prec_Rec_Thres.csv
#################################################################
def save_p_r_thresholds(precision_test, recall_test, threshold_test):
    #print ("---Saving Prec|Rec|Thres|F1---")

    output_folder_path = "../4_Results/1_Data"
    nameResultDir = "1_Results"
    output_path = join(output_folder_path, nameResultDir)
    if not(exists(output_folder_path)):
        makedirs(output_folder_path)
    all_directories = listdir(output_folder_path)
    if not (nameResultDir in all_directories):
        makedirs(output_path)
    output_file_path = join(output_path, 'Prec_Rec_Thres.csv')
    
    with open(output_file_path, 'w') as file_out:
        header='Precision,Recall,Threshold,F1'
        print (header,file=file_out)

        for i in range(0,len(threshold_test)):
            line_to_write_list=list()
            line_to_write_list.append( str(threshold_test[i]) )
            line_to_write_list.append( str(precision_test[i]) )
            line_to_write_list.append( str(recall_test[i]) )
            line_to_write_list.append( str(( 2*precision_test[i]*recall_test[i] )/( recall_test[i]+precision_test[i] )) )

            print(
                    ','.join(line_to_write_list),
                    file=file_out
                )
    #print ('--->saved in ',output_file_path)

###############################################
#	Saving 10 commits computed to be the more
#	likely to be VCCs in 4_Results/1_Data/Top_10
#	and distance to hyperplan
##############################################
def save_top_10(score_list, unitary_position_list, iteration):
    folder_path = "../4_Results/1_Data"
    top10 = "Top_10"
    path_to_file = join(folder_path, top10)
    all_dirs = listdir(folder_path)
    if not (top10 in all_dirs):
        makedirs(path_to_file)
    file_path = join(path_to_file, names[iteration])
    with open(file_path, 'w') as f_out:

        for id in range(0, 10):
            all_idstr = " "
            #build_string to write in file
            print(
                str(unitary_position_list[id] + "|" + str(score_list[id])),
                file=f_out
            )



####################################
#    from unlabeled_test, will extract the commit id
#    and return a list of commits id
#
####################################
def get_commit_id_list():
    commit_id_list = list()
    with open(
        '../3_IntermediarySteps/5_vcc_Input/unlabeled_test.libsvm', 'r'
    ) as input_file:
        all_lines = input_file.readlines()
    for line in all_lines:
        dirty_id = (line.split('#'))[1].replace('\n', '')
        clean_id = re.findall('[0-9]+', dirty_id)[0]
        commit_id_list.append(clean_id)
    return commit_id_list


#############################################
##  For one particular experiment with unlabeled set (cf MARKER_XP_LIST)
##     will  determine the 10 commits to be the more likely to be VCC
##     
#############################################
def get_top_ten(dist_list, iteration):
    #init
    position_of_commit_id = get_commit_id_list()
    score_list = list()
    position_list = list()
    score_list.append(dist_list[0])
    position_list.append(position_of_commit_id[0])
    for indix in range(1, len(dist_list)):
        ten_list_indix = 0
        while ten_list_indix < len(score_list) and dist_list[
            indix] < score_list[ten_list_indix]:
            ten_list_indix += 1
        if ten_list_indix < len(score_list) and dist_list[indix] > score_list[
            ten_list_indix]:
            score_list.insert(ten_list_indix, dist_list[indix])
            position_list.insert(ten_list_indix, position_of_commit_id[indix])
        if ten_list_indix < len(score_list) and len(score_list) > 10:
            score_list.pop()
            position_list.pop()
    save_top_10(score_list, position_list, iteration)

################################################
#
#	Depending on which experiment are uncommented 
#	at MARKER_XP_LIST, the ten commits computed to
#	be the more likely to be VCC are those for which
#	the sum of distances to the hyper-plan is the highest
#	Not saving anywhere as not proven that distance are
#	meaningful in comparison with one another in each experiment 
#
################################################
def get_top_ten_over_all(dist_list):
    pos_to_commit_id = get_commit_id_list()
    score_list = list()
    position_list = list()
    score_list.append(dist_list[0])
    position_list.append(0)

    for indix in range(1, len(dist_list)):
        ten_list_indix = 0
        while ten_list_indix < len(score_list) and dist_list[
            indix] < score_list[ten_list_indix]:
            ten_list_indix += 1
        if ten_list_indix < len(score_list) and dist_list[indix] > score_list[
            ten_list_indix]:
            score_list.insert(ten_list_indix, dist_list[indix])
            position_list.insert(ten_list_indix, pos_to_commit_id[indix])
        if ten_list_indix < len(score_list) and len(score_list) > 10:
            score_list.pop()
            position_list.pop()
#    print("--Global--")
#    print("--", score_list)
#    print("--", position_list)
#    print("______________________")



##########################################
#    Saves in /4_Results/1_Data/1_Results/regular
#    probas(distance to hyper-plan), predicted_class, ground_truth_class, commit_id
#    
#
##########################################
def save_prediction_results(probas_test,preds_test,Yt ):
    #init
    pos_to_commit_id = get_commit_id_list()

    output_folder = "../4_Results/1_Data"
    if not (exists(output_folder)):
        makedirs(output_folder)
    nameResultDir = "1_Results"
    output_path = join(output_folder, nameResultDir)
    all_dirs = listdir(output_folder)
    if not (nameResultDir in all_dirs):
        makedirs(output_path)
    file_path = join(output_path, names[0])
    match_on_test="matched_tested"
    match_path=join(output_folder,match_on_test)
    true_pos_counter=0
    false_pos_counter=0
    false_neg_counter=0
    with open(file_path, 'w') as f_out:
        header="proba,predicted,realClass,commit_db_id"
        print (header,file=f_out)
        with open(match_path,'w') as match_out:
            print (header,file=match_out)
            for id in range(0,len(probas_test)):
                line = str(  str(probas_test[id])+","+str(preds_test[id])+","+str(Yt[id])+","+pos_to_commit_id[id]  )
                #input(line)
                #build_string to print 
                print(
                    line,
                    file=f_out
                )
                if '1.0' in str(preds_test[id]) and '1.0' in str(Yt[id]):
                    print (
                        line,
                        file=match_out
                    ) 
                    #print (line)
                    true_pos_counter+=1
                elif '0.0' in str(preds_test[id]) and '1.0' in str(Yt[id]):
                    false_neg_counter+=1
                elif '1.0' in str(preds_test[id]) and '0.0' in str(Yt[id]):
                    false_pos_counter+=1
    print ("True positives: "+str(true_pos_counter))
    print ("False negatives: "+str(false_neg_counter))
    print ('False positives: '+str(false_pos_counter))

#REPLICATION
############################################################################
print ("====Sklearn_drawNt10====")
# sklearn.datasets.load_svmlight_file(f, n_features=None, dtype=<class ‘numpy.float64’>,
# multilabel=False, zero_based=’auto’, query_id=False, offset=0, length=-1)[
#####################################################################################
#################	Replication and generating Figure 2 	#####################
########################################################################Loading  Data
#nb_features = 4194304 usually

#Xgt_train for groung Truth
(Xgt_train, Ygt_train) = load_svmlight_file(
    '../3_IntermediarySteps/5_vcc_Input/training_reg.libsvm',
    dtype=bool
)
#MARKER_XP_LIST
#Adding diversity in training_neg with 1,000 unlabeled neg
#(X1, Y1) = load_svmlight_file(
#    '../3_IntermediarySteps/5_vcc_Input/training_1000.libsvm',
#    dtype=bool
#)
#Adding diversity in training_neg with 5,000 unlabeled neg
#(X5, Y5) = load_svmlight_file(
#    '../3_IntermediarySteps/5_vcc_Input/training_5000.libsvm',
#    dtype=bool
#)
#Adding diversity in training_neg with 10,000 unlabeled neg
#(X10, Y10) = load_svmlight_file(
#    '../3_IntermediarySteps/5_vcc_Input/training_10000.libsvm',
#    dtype=bool
#)

#Ground Truth Test Data
(Xgt_test, Ygt_test) = load_svmlight_file(
    '../3_IntermediarySteps/5_vcc_Input/test.libsvm',
    dtype=bool
)
#Unlabeled Test Data
(Xu,Yu) = load_svmlight_file(
    '../3_IntermediarySteps/5_vcc_Input/unlabeled_test.libsvm',
    dtype=bool
)

#Training with Unlabeled Data
(MXtr, MYtr) = load_svmlight_file(
    '../3_IntermediarySteps/5_vcc_Input/mega_training.libsvm',
    dtype=bool
)

#Test with Unlabeled
(MXte, MYte) = load_svmlight_file(
    '../3_IntermediarySteps/5_vcc_Input/mega_test.libsvm',
    dtype=bool
)

####################################################################reshape matrix to max sizes

max_features_all_files = max( Xgt_train.shape[1], Xgt_test.shape[1], Xu.shape[1], MXtr.shape[1], MXte.shape[1] )

Xgt_train = csr_matrix( Xgt_train, shape=(Xgt_train.shape[0], max_features_all_files ) )
Xgt_test = csr_matrix( Xgt_test, shape = (Xgt_test.shape[0], max_features_all_files ) )
Xu = csr_matrix( Xu, shape = (Xu.shape[0], max_features_all_files ) )
MXtr = csr_matrix( MXtr, shape = (MXtr.shape[0], max_features_all_files ) )
MXte = csr_matrix( MXte, shape = (MXte.shape[0], max_features_all_files ) )


#####################################################################classifiers initialisation

# class sklearn.svm.LinearSVC(penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001,
# C=1.0, multi_class=’ovr’, fit_intercept=True,
# intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)[
params_weighted={"max_iter":200000,"class_weight":{0: 1,1: 100}}
classif = LinearSVC()
Mclassif = LinearSVC()
classif.set_params(**params_weighted)
Mclassif.set_params(**params_weighted)
#MARKER_XP_LIST
#classif1 = LinearSVC(max_iter=200000)
#classif5 = LinearSVC(max_iter=200000)
#classif10 = LinearSVC(max_iter=200000)

########################################################################Training of classifiers
classif.fit(Xgt_train, Ygt_train)
Mclassif.fit(MXtr,MYtr)

#MARKER_XP_LIST
#print("...")
#classif1.fit(X1, Y1)			#Attempt to artificially increase training set
#print("...")
#classif5.fit(X5, Y5)			#Attempt to artificially increase training set
#print("...")
#classif10.fit(X10, Y10)		#Attempt to artificially increase training set




########################################################################Prediction of classifiers

lines = []
#Ground Truth Replication
preds_test = classif.predict(Xgt_test)
probas_test = classif.decision_function(Xgt_test)
precision_test, recall_test, threshold_test = precision_recall_curve(
    Ygt_test, probas_test
)

save_p_r_thresholds(precision_test, recall_test, threshold_test)
#Unlabeled_replication
Wpreds_Mtest=classif.predict(MXte)
Wprobas_Mtest=classif.decision_function(MXte)
Wprecision_Mtest, Wrecall_Mtest, Wthreshold_Mtest = precision_recall_curve(
    MYte, Wprobas_Mtest
)
#Unlabeled Trained Replication
Wpreds_Mtrain=Mclassif.predict(MXte)
Wprobas_Mtrain=Mclassif.decision_function(MXte)
Wprecision_Mtrain, Wrecall_Mtrain, Wthreshold_Mtrain = precision_recall_curve(
    MYte, Wprobas_Mtrain
) 





plt.clf()
plt.rcParams.update({'font.size': 38})
plt.figure(figsize=(18.5, 18.5))

#MARKER_XP_LIST
plt.ylim([-0.02, 1.02])
plt.xlim([-0.02, 1.02])
plt.xlabel('Recall')
plt.ylabel('Precision')
lines.append(
    plt.plot(
       recall_test,
        precision_test,
        label="Ground Truth",
        marker='o',
        linestyle='',
        linewidth=2,
        markersize=5,
        color='green'
    )
)
lines.append(
    plt.plot(
        Wrecall_Mtest,   
        Wprecision_Mtest,  
        label="Unlabeled Replication",
        marker='+', 
        linestyle='',
        linewidth=2,
        markersize=6, 
        color='red'
    )
)

lines.append(
    plt.plot(
        Wrecall_Mtrain,
        Wprecision_Mtrain,
        label="Unlabeled Trained Replication",
        marker='X', 
        linestyle='',
        linewidth=2,
        markersize=6,
        color='blue'
    )
)


#MARKER_XP_LIST
#train1000  #Attempt to artificially increase unlabeled set size
#preds_1000 = classif1.predict(Xgt_test)
#probas_1000 = classif1.decision_function(Xgt_test)
#precision_1000, recall_1000, threshold_1000 = precision_recall_curve(
#    Ygt_test, probas_1000
#)
#lines.append(
#    plt.plot(
#        recall_1000,
#        precision_1000,
#        label="train_1000",
#        marker='o',
#        linestyle='',
#        linewidth=2,
#        markersize=4,
#        color='yellow'
#    )
#)
#print('...')
#train5000    #Attempt to artificially increase unlabeled set size
#preds_5000 = classif5.predict(Xgt_test)
#probas_5000 = classif5.decision_function(Xgt_test)
#precision_5000, recall_5000, threshold_5000 = precision_recall_curve(
#    Ygt_test, probas_5000
#)
#lines.append(
#    plt.plot(
#        recall_5000,
#        precision_5000,
#        label="train_5000",
#        marker='o',
#        linestyle='',
#        linewidth=2,
#        markersize=4,
#        color='orange'
#    )
#)
#print("...")
#train10000   #Attempt to artificially increase unlabeled set size
#preds_10000 = classif10.predict(Xgt_test)
#probas_10000 = classif10.decision_function(Xgt_test)
#precision_10000, recall_10000, threshold_10000 = precision_recall_curve(
#    Ygt_test, probas_10000
#)
#lines.append(
#    plt.plot(
#        recall_10000,
#        precision_10000,
#        label="train_10000",
#        marker='o',
#        linestyle='',
#        linewidth=2,
#        markersize=4,
#        color='red'
#    )
#)

###iso-F1
f_scores = np.linspace(0.2, 0.8, num=4)
for f_score in f_scores:
    x_iso = np.linspace(0.01, 1, num=50)
    y_iso = (f_score * x_iso) / (2 * x_iso - f_score)
    plt.plot(
        x_iso[y_iso > 0],
        y_iso[y_iso > 0],
        color='gray',
        alpha=0.3,
        linewidth=4
    )
plt.tight_layout()
plt.grid(color='gray', linestyle=':', linewidth=1)
#lines.append(l)

###Legend
plt.legend(loc="upper right", markerscale=5)


###plt.show()
save_folder = "../4_Results/2_Plots/"
if not (exists(save_folder)):
    makedirs(save_folder)
plt.savefig('../4_Results/2_Plots/Figure2.pdf', dpi=720)


#save_prediction_results
save_prediction_results(probas_test,preds_test,Ygt_test )

########################################################################################################C_INFLUENCE
print ("----C_Values----")
plt.clf()
plt.rcParams.update({'font.size': 27})
plt.figure(figsize=(16.5, 16.5))

plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.xlabel('Recall test')
plt.ylabel('Precision test')

lines = []

index=0
c_values = [0.000001, 0.00001, 0.00003,0.00005, 0.00007, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
colours = ['red','tomato','coral', 'orange','gold', 'yellow', 'lime', 'green','seagreen','aquamarine', 'cyan', 'blue', 'purple', 'magenta']
for c_val in c_values:

    params = {"max_iter":200000,"class_weight":{0:1, 1:100},"C":c_val}
    c_classif = LinearSVC(**params)
    #training
    c_classif.fit(Xgt_train, Ygt_train)
    #prediction
    c_preds_test = c_classif.predict(Xgt_test)
    c_probas_test = c_classif.decision_function(Xgt_test)
    c_precision_test, c_recall_test, c_threshold_test = precision_recall_curve(
     Ygt_test, c_probas_test
    )
    
    lines.append(
        plt.plot(
            c_recall_test,
            c_precision_test,
            label="C:"+str(c_val),
            marker='o',
            linestyle='',
            linewidth=2,
            markersize=4,
            color=colours[index]
        )
    )
    index+=1

f_scores = np.linspace(0.2, 0.8, num=4)
for f_score in f_scores:
        x_iso = np.linspace(0.01, 1, num=50)
        y_iso = (f_score * x_iso) / (2 * x_iso - f_score)
        plt.plot(x_iso[y_iso > 0], y_iso[y_iso > 0], color='gray', alpha=0.3)

#Legend
plt.legend(loc="lower left",fontsize = 26)
save_folder = "../4_Results/2_Plots/"
if not (exists(save_folder)):
    makedirs(save_folder)
plt.savefig(
    '../4_Results/2_Plots/Figure3.pdf',
    dpi=720
)

###########################################################################CLASSWEIGHT_INFLUENCE
print ("----Class Weight----")
#for different class Weight
plt.clf()
plt.rcParams.update({'font.size': 27})
plt.figure(figsize=(16.5, 16.5))

plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])  
plt.xlabel('Recall test')   
plt.ylabel('Precision test')

lines = []

index=0
class_weights = [ 0.1, 1, 2, 5, 10, 25, 50, 100]

for class_w in class_weights:

    params = {"max_iter":200000,"class_weight":{0:1, 1:class_w},"C":1}
    c_classif = LinearSVC(**params)
    #training
    c_classif.fit(Xgt_train, Ygt_train)
    #predicting
    c_preds_test = c_classif.predict(Xgt_test)
    c_probas_test = c_classif.decision_function(Xgt_test)
    c_precision_test, c_recall_test, c_threshold_test = precision_recall_curve(
     Ygt_test, c_probas_test
    )

    lines.append(
        plt.plot(
            c_recall_test,
            c_precision_test,
            label="W1:"+str(class_w),
            marker='o',
            linestyle='',
            linewidth=2,
            markersize=4,
            color=colours[index]
        )
    )
    index+=1

f_scores = np.linspace(0.2, 0.8, num=4)
for f_score in f_scores:
        x_iso = np.linspace(0.01, 1, num=50)
        y_iso = (f_score * x_iso) / (2 * x_iso - f_score)
        plt.plot(x_iso[y_iso > 0], y_iso[y_iso > 0], color='gray', alpha=0.3)

#Legend
plt.legend(loc="lower left",fontsize = 22)
plt.savefig(
    '../4_Results/2_Plots/Explore_Weight_Influence.pdf',
    dpi=720
)

###############################################################################ALGORITHM_INFLUENCE

plt.clf()
plt.rcParams.update({'font.size': 27})
plt.figure(figsize=(16.5, 16.5))
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.xlabel('Recall test')
plt.ylabel('Precision test')
lines = []



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree",
         "Random Forest_depth5__10",
         "Random ForestF_depth2_0_200",
         "RF_6_0_200",
         "Neural Net",
         "AdaBoost"
         ]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025,probability=True),
    SVC(gamma=2, C=1,probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    RandomForestClassifier(max_depth=2,random_state=0,n_estimators=200),
    RandomForestClassifier(max_depth=6,random_state=0,n_estimators=200),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier()
    ]

index=0

lines.append(   
    plt.plot(
        recall_test,
        precision_test,
        label="VCCFinder_replication",
        marker='o',
        linestyle='',
        linewidth=2,
        markersize=4,
        color='blue'
    )
)


#list of classifiers
for name, clf, colour in zip(names, classifiers,colours):
    print ("----"+name)
    if name == "Gaussian Process" or name == "Decision Tree" or name == "Naive Bayes":
        Xtrain = Xgt_train.toarray()
        Xtest = Xgt_test.toarray()
    else:
        Xtrain = Xgt_train
        Xtest = Xgt_test
    clf.fit(Xtrain, Ygt_train)

    if hasattr(clf, "decision_function"):
        clf_score_test = clf.decision_function(Xtest)
    else:
        clf_score_test = clf.predict_proba(Xtest)[:, 1]
    
    clf_preds_test = clf.predict(Xtest)
    clf_probas_test = clf.predict_proba(Xtest)
    clf_precision_test, clf_recall_test, clf_threshold_test = precision_recall_curve(
        Ygt_test, clf_score_test
    )

    lines.append(   
        plt.plot(
            clf_recall_test,   
            clf_precision_test,   
            label=name,
            marker='o',
            linestyle='',
            linewidth=2,
            markersize=3,
            color=colour
        )
    )

#iso-F1
f_scores = np.linspace(0.2, 0.8, num=4)
for f_score in f_scores:
        x_iso = np.linspace(0.01, 1, num=50)
        y_iso = (f_score * x_iso) / (2 * x_iso - f_score)
        plt.plot(x_iso[y_iso > 0], y_iso[y_iso > 0], color='gray', alpha=0.3)

#Legend
plt.legend(loc="lower left",fontsize = 22,markerscale = 5)
plt.savefig(
    '../4_Results/2_Plots/Figure4.pdf',
    dpi=720
)

#EXPLORE_UNLABELED
################################################################################Top10 on unlabeled post 2011 exclusively
preds_u_test = classif.predict(Xu)
probas_u_test = classif.decision_function(Xu)
#MARKER_XP_LIST
#preds_u_1000 = classif1.predict(Xu)
#probas_u_1000 = classif1.decision_function(Xu)
#preds_u_5000 = classif5.predict(Xu)
#probas_u_5000 = classif5.decision_function(Xu)
#preds_u_10000 = classif10.predict(Xu)
#probas_u_10000 = classif10.decision_function(Xu)

##The point is, out of probas1000,probas5000,probas10000 to get the ids of the most likely commit
###Tests on proba computation
#MARKER_XP_LIST
proba_list = list()
proba_list.append(probas_u_test)
#proba_list.append(probas_u_1000)
#proba_list.append(probas_u_5000)
#proba_list.append(probas_u_10000)

iteration = 0

dist_sum = [0] * len(probas_u_test)
for probas in proba_list:
    max = np.amax(probas)
    position = np.where(probas == np.amax(probas))

    get_top_ten(probas, iteration)
    iteration += 1

    for indix in range(0, len(probas)):
        dist_sum[indix] += probas[indix]
get_top_ten_over_all(dist_sum)
