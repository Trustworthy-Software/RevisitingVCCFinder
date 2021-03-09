import csv
from os.path import join

in_folder="../4_Results/1_Data/1_Results"
all_files=list()
#all_files.append('Mega-Train','Mega-Test','Replicate')
all_files.append(join(in_folder,'Mega-Train'))
all_files.append(join(in_folder,'Mega-Test'))
all_files.append(join(in_folder,'Replicate'))


for res_file in all_files:
    result_list=list()
    with open(res_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        
        for row in spamreader:
#            print (row)
            temp_list=list(row)
            result_list.append(temp_list)
    
    #min computation
    min_thres=0.0
    for row_i in range(1,len(result_list)):
        if float(result_list[row_i][0])< min_thres:
            min_thres=float(result_list[row_i][0])
    index=1
    tp_count=0
    fp_count=0
    fn_count=0
    recall=0
    precision=0
    former_r=0
    former_p=0
    #print (result_list[1])
    #all_thresholds=(result_list, key = lambda x: x[0])
    #threshold=min(result_list, key = lambda x: x[0])  #all_thresholds)
    threshold=min_thres
    out_var=0
    #print ("=====Starting threshold at ",threshold )
    while out_var==0:
        while index <len(result_list) :
            l_thres=float(result_list[index][0]) 
            if l_thres>threshold and '1.0' in result_list[index][1]:
                tp_count+=1
                recall=(tp_count/253)
                precision=(tp_count/(tp_count+fp_count))
                #print('--',recall)
            elif l_thres>threshold and '0.0' in result_list[index][1]:
                fp_count+=1
                precision=(tp_count/(tp_count+fp_count))
            elif l_thres<threshold and '1.0' in result_list[index][1]:
                fn_count+=1

            index+=1
        
        if recall>0.24:
            #print (threshold,'++',recall,'__',index,'/',len(result_list))
            former_t=threshold
            threshold+=0.01
            index=1
            tp_count=0
            fp_count=0
            former_r=recall
            former_p=precision
            recall=0
            precision=0
           
        if index>=len(result_list):
          out_var=1
            #print ("----",threshold,'	|	',recall)
    print (res_file)
    print ('Precision	|Recall	|Threshold')
    print (former_p,"	|",former_r,"	|",threshold)
    #print (precision,"   |",recall,"   |",former_t)
    #input ('#Next')
