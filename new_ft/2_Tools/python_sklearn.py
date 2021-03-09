#!/usr/bin/env python3

############################################################################
#
#	This script will deal with the prediction on treated data with New_Features
#	it will produce Figure 5
#
#	See #NEWFEATURESNOCO-TRAINING
###############################################################################
import re
import numpy as np
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import os



################################################################
#	save precision,recall,threshold and F1 score  
#		in file 4_Results/1_Data/Prec_Rec_Thres.csv
#################################################################
def save_p_r_thresholds(precision_test, recall_test, threshold_test):
    folder = "../4_Results/1_Data"
    nameResultDir = "1_Results"
    output_path = os.path.join(folder, nameResultDir)
    all_dirs = os.listdir(folder)
    if not (nameResultDir in all_dirs):
        os.makedirs(output_path)
    file_path = os.path.join(output_path, 'Prec_Rec_Thres.csv')
    
    with open(file_path, 'w') as f_out:
        header='Precision,Recall,Threshold,F1'
        #header="proba,predicted,realClass,commit_db_id"
        print (header,file=f_out)

        for i in range(0,len(threshold_test)):
            line=list()
            line.append( str(threshold_test[i]) )
            line.append( str(precision_test[i]) )
            line.append( str(recall_test[i]) )
            line.append( str(( 2*precision_test[i]*recall_test[i] )/( recall_test[i]+precision_test[i] )) )

            print(
                    ','.join(line),
                    file=f_out
                )


####################################
#    from unlabeled_test, will extract the commit id
#    and return a list of commits id
#
####################################

COMMIT_ID_FINDER = re.compile('[0-9]+')
def get_commit_id_list():
    commit_id_list = list()
    with open(
        '../3_IntermediarySteps/fused_feature_files/test.libsvm', 'r'
    ) as f_in:
        all_lines = f_in.readlines()
    for line in all_lines:
        dirty_id = (line.split('#'))[1].replace('\n', '')
        clean_id = COMMIT_ID_FINDER.findall(dirty_id)[0]
        commit_id_list.append(clean_id)
    return commit_id_list


##########################################
#    Saves in /4_Results/1_Data/1_Results/regular
#    probas(distance to hyper-plan), predicted_class, ground_truth_class, commit_id
#    
#
##########################################

def save_prediction_results(probas_test,preds_test,Yt,name ):
    print ("---Saving Results---")
    #init
    pos_to_commit_id = get_commit_id_list()

    folder = "../4_Results/1_Data"
    nameResultDir = "1_Results"
    output_path = os.path.join(folder, nameResultDir)
    all_dirs = os.listdir(folder)
    if not (nameResultDir in all_dirs):
        os.makedirs(output_path)
    file_path = os.path.join(output_path, name)
    match_on_test="matched_tested"
    match_path = os.path.join(folder,match_on_test)
    #parsed_position_list=str(position_list).split(",),")
    true_pos_counter = 0
    false_pos_counter = 0
    false_neg_counter = 0
    true_neg_counter = 0
    with open(file_path, 'w') as f_out:

        header='Prediction,actual_class'
        #header="proba,predicted,realClass,commit_db_id"
        print (header,file=f_out)
        with open(match_path,'w') as match_out:
            print(header,file=match_out)
            for id in range(0, len(probas_test)):
                line=str( str(probas_test[id])+','+str(Yt[id])  )
                print(
                    line,
                    file=f_out
                )
                if '1.0' in str(preds_test[id]) and '1.0' in str(Yt[id]):
                    print (
                        line,
                        file=match_out
                    )
                    true_pos_counter+=1
                elif '0.0' in str(preds_test[id]) and '1.0' in str(Yt[id]): 
                    false_neg_counter+=1
                elif '1.0' in str(preds_test[id]) and '0.0' in str(Yt[id]):
                    false_pos_counter+=1
                else:
                    true_neg_counter+=1
    print ("True positives: "+str(true_pos_counter))
    print ("False negatives: "+str(false_neg_counter))
    print ('False positives: '+str(false_pos_counter))
    print ('True negatives: '+str(true_neg_counter))
    print ('Prec'+str(true_pos_counter/(true_pos_counter+false_pos_counter)))
    print ('Rec'+str(true_pos_counter/(true_pos_counter+false_neg_counter)))
    print ('-->saved in ',name,'\n')



#######################################################################################
##########################  New_Features No Co-Training	###############################
#######################################################################################
#NEWFEATURESNOCO-TRAINING

#Ground Truth Training Data
(Xreg, Yreg) = load_svmlight_file('../3_IntermediarySteps/fused_feature_files/training.libsvm',n_features=66, dtype=float)
#Ground Truth Test
(Xt,Yt) = load_svmlight_file('../3_IntermediarySteps/fused_feature_files/test.libsvm',n_features=66, dtype=float)
#Ground Truth and Unlabeled Test Data (M for Mega)
(MXte, MYte) = load_svmlight_file('../3_IntermediarySteps/fused_feature_files/mega_test.libsvm',n_features=66, dtype=float)

####################################################################reshape matrix to max sizes


#####################################################################classifiers initialisation
params_weighted={"max_iter":200000,"class_weight":{0: 1,1: 100}}
Wclassif=LinearSVC()
Wclassif.set_params(**params_weighted)

########################################################################Training of classifiers
Wclassif.fit(Xreg, Yreg)

########################################################################Prediction of classifiers
lines=[]
#Ground Truth Test
Wpreds_test=Wclassif.predict(Xt)
Wprobas_test = Wclassif.decision_function(Xt)
#print ('==DEBUG== ','Probas|',probas_test,'|Yt|',Yt.shape,'|')
Wprecision_test, Wrecall_test, Wthreshold_test = precision_recall_curve(Yt,Wprobas_test)

#Unlabeled and Ground Truth Test (M for Mega)
Wpreds_Mtest=Wclassif.predict(MXte)
Wprobas_Mtest = Wclassif.decision_function(MXte)
#print ('==DEBUG== ','Probas|',probas_test,'|Yt|',Yt.shape,'|')
Wprecision_Mtest, Wrecall_Mtest, Wthreshold_Mtest = precision_recall_curve(MYte,Wprobas_Mtest)

#############################################################################Ploting Results

plt.clf()
plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(16.5,16.5))
#Figure Settings
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.xlabel('Recall')   
plt.ylabel('Precision')
#Lines to be plotted
lines.append( plt.plot(Wrecall_test, Wprecision_test, label="Ground Truth New Features", marker='o', linestyle='', linewidth=2, markersize=4, color='green') )
lines.append( plt.plot(Wrecall_Mtest, Wprecision_Mtest, label="Unlabeled New Features", marker='+', linestyle='', linewidth=2, markersize=5, color='red') )

#Plot iso-F1 lines in grey
f_scores = np.linspace(0.2, 0.8, num=4)
for f_score in f_scores:
    x_iso=np.linspace(0.01, 1, num = 50)
    y_iso=(f_score*x_iso)/(2*x_iso-f_score)
    plt.plot(x_iso[y_iso>0],y_iso[y_iso>0], color='gray', alpha=0.3 ,linewidth = 4)

#################################################################################Saving Results
if not os.path.isdir('../4_Results/1_Data'):
        os.makedirs('../4_Results/1_Data')

save_prediction_results(Wprobas_test,Wpreds_test,Yt, "Ground_Truth_New_Feats")
save_prediction_results(Wprobas_Mtest,Wpreds_Mtest,MYte, "Unlabeled_New_Feats")

print ("-->results saved in ../4_Results/1_Data/1_Results/")
if not os.path.isdir('../4_Results/2_Plots'):
        os.makedirs('../4_Results/2_Plots')
#Legend
plt.legend(loc="upper right",markerscale = 6)
plt.tight_layout()
plt.grid(color='gray', linestyle=':', linewidth=1)
#plt.show()
plt.savefig('../4_Results/2_Plots/New_Features_No_Co-Training.pdf',dpi=720)
print ("-->saved to 4_Results/2_Plots/New_Features_No_Co-Training.pdf")
