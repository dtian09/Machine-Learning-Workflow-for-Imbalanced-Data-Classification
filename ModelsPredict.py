"""
###Program to predict PTB of patients from best readings selected using filter2
for each iteration i in {0,...,99} {
     select best readings for the patients of testset_i using the filter2 of iteration i
     Predict the PTB using model_i from the selected best readings
}
Output: the testset AUC to screen and save the results of prediction to logfile.txt
inputs: filter2 (random forest rf9)
        PTB classifiers (random forests) 
        100 test sets
        ids of all the 438 patients
paths setting: weka path
               filter2 path
               PTB classifiers path
               test sets path
               results path
"""
import pandas as pd
#import workflows as wf
import preprocess as prep
import postprocess as post
import classifiers as cl
import os, sys
import numpy as np
import SelectData as sd
#import dataframetoarff
from joblib import Parallel, delayed, load
#import multiprocessing
import random
#import subprocess
import operator
import utilities
#from keras.models import load_model
#from workflow_nn import nn_predict
#from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score#, confusion_matrix

class ModelsPredict:
    'class to predict PTB using weka models, sklearn models or keras models'
    def __init__(self,
                 select_readings_parallel=True,
                 eisfilter_path=None,
                 eisfilter_type='rf',
                 eisfilter_software='weka',
                 all_eis_readings_of_ids=None,
                 model_software='weka',
                 modeltype='rf', # or 'log_reg'
                 testset_ids_path=None,
                 results_path=None,
                 logfile=None,
                 weka_path=None,
                 java_memory='4g'
                 ):
        self.select_readings_parallel=select_readings_parallel
        self.data='EIS'
        #print(data)
        #filter2_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filter2 from sharc\\selected_unselected_eis_readings\\"
        #filter2_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\selected_unselected_438_V1_28inputs_by_amp1\\"
        self.filter2_path=eisfilter_path
        self.filter2type=eisfilter_type
        self.model_software=model_software
        self.filter2_software=eisfilter_software
        self.testset_ids_path=testset_ids_path
        self.modeltype=modeltype 
        self.no_of_models=100
        self.allreadings_with_ids=all_eis_readings_of_ids
        self.results_path=results_path
        self.logfile=logfile #results of PTB prediction
        self.weka_path=weka_path
        self.java_memory=java_memory
        ###EIS
        #model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_no_treatment_noise_added_10_to_15features\\"
        #model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_no_treatment\\"#path of PTB classifiers
        #model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_no_treatment2_added_noise\\" #EIS data
        #model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\"
        ###EIS+metabolite
        self.asymp_22wks_438_V1_9inputs_with_ids="U:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_9inputs_with_ids.csv"
        self.cst_with_ids="d:\\EIS preterm prediction\\metabolite\\cst_asymp_8inputs_filtered_data_28inputs_with_ids2.csv" #the categorical CST feature transformed to continuous using 1-hot-encoding
        #model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_9_inputs_438_V1_28inputs_by_amp1_no_missing\\" #EIS+metabolite data
        #model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_9_inputs_438_V1_28inputs_by_amp1_no_missing_10_to_15features\\" #EIS+metabolite data    
        #model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_filtered_data_28inputs_10_to_20features\\"
        self.model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_filtered_data_28inputs_no_treatment_5_to_15features_2\\"
        #model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_filtered_data_28inputs_no_treatment_10_to_15features\\"
        #model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_high_risk_9_inputs_438_V1_28inputs_by_amp1_10_to_15features\\" #EIS+metabolite data    
        #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\15dec_filtered_data_28inputs\\"#location of testsets consisting of the best readings selected by filter2s
        #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_9_inputs_438_V1_28inputs_by_amp1_no_missing_10_to_15features\\"#location of testsets consisting of the best readings selected by filter2s
        #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_high_risk_9_inputs_438_V1_28inputs_by_amp1_10_to_15features\\"#location of testsets consisting of the best readings selected by filter2s
        #results_path2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_9_inputs_438_V1_28inputs_by_amp1_no_missing_10_to_15features\\"#location of testsets consisting of the best readings and metabolite
        #results_path2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_high_risk_9_inputs_438_V1_28inputs_by_amp1_10_to_15features\\"#location of testsets consisting of the best readings and metabolite
        #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_filtered_data_28inputs_10_to_20features\\"#location of testsets consisting of the best readings
        #results_path2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_filtered_data_28inputs_10_to_20features\\"#location of testsets consisting of the best readings and metabolite
        #results_path2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_filtered_data_28inputs_no_treatment_10_to_15features\\"#location of testsets consisting of the best readings and metabolite
        #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_filtered_data_28inputs_no_treatment_5_to_15features_2\\"#location of testsets consisting of the best readings
        self.ModelNumber=0
        self.X_scaler=None
        self.trainset_i=None
        self.template_filter2_option=None
        self.ordinal_encode=False #encode discrete intervals of test set as integers before prediction
        self.hash_best_readings={} #key=id, value=best reading of the id
        self.scaler_file=None
        self.youden_threshold=None
    
    def set_modeltype(self,modeltype2):
        self.modeltype=modeltype2
        
    def set_model_number(self,ModelNumber2):
        self.ModelNumber=ModelNumber2
        
    def set_ordinal_encode(self,ordinal_encode2):
        self.ordinal_encode=ordinal_encode2
        
    def set_dataset(self,data2):
        self.data=data2
    
    def set_trainset_i(self,trainset_i2):
        self.trainset_i=trainset_i2
    
    def set_template_filter2_option(self,template_filter2_option2):
        self.template_filter2_option=template_filter2_option2
        
    def set_filter2(self,filter2type2,filter2_software2,filter2_path2):
        self.filter2type = filter2type2
        self.filter2_software = filter2_software2
        self.filter2_path = filter2_path2
        
    def set_model(self,modeltype2,model_software2,model_path2):
        self.modeltype = modeltype2
        self.model_software = model_software2
        self.model_path = model_path2
    
    def set_scaler(self,X_scaler2):
        self.X_scaler=X_scaler2
    
    def set_scaler_file(self,scaler_file2):
        self.scaler_file=scaler_file2
        
    def set_testset_ids_path(self,testset_ids_path2):
        self.testset_ids_path = testset_ids_path2
    
    def set_all_readings_with_ids_file(self,allreadings_with_ids2):
        self.allreadings_with_ids = allreadings_with_ids2
    
    def set_metabolite_data_with_ids_file(self,asymp_22wks_438_V1_9inputs_with_ids2):
        self.asymp_22wks_438_V1_9inputs_with_ids = asymp_22wks_438_V1_9inputs_with_ids2
    
    def set_results_path(self,results_path2):
        self.results_path = results_path2
    
    def set_logfile(self,logfile2):
        self.logfile = logfile2
    
    def set_weka_path(self,weka_path2):
        self.weka_path = weka_path2
    
    def set_java_memory(self,java_memory2):
        self.java_memory = java_memory2
    
    def filter_readings_using_weka_classifier(self,filter2,filter2_inputs_output_csv,filter2_discrete_cuts_file,readings_csv,results_path,weka_path,java_memory):
        df=pd.read_csv(readings_csv)
        (_,c)=df.shape
        target=df.iat[0,c-1]
        if target != '?':#known targets
            (_,predL,_,_,_,_,_,_)=self.predict_using_weka_model('prediction list',readings_csv,filter2_discrete_cuts_file,filter2,self.filter2type,filter2_inputs_output_csv,results_path,weka_path,java_memory)
        else:
            predL=self.predict_using_weka_model('prediction list',readings_csv,filter2_discrete_cuts_file,filter2,self.filter2type,filter2_inputs_output_csv,results_path,weka_path,java_memory)
        #print('====scores of all readings of the id===\n')
        for i in range(len(predL)):
            (inst,score,_,_,_)=predL[i]
            #print('score of reading '+str(inst)+': '+str(score))
        (inst,best_score,_,_,_)=predL[0]#best reading
        #print('\nreading '+str(inst.strip())+' is selected (score: '+str(best_score)+')\n')
        df=pd.read_csv(readings_csv)
        (_,c)=df.shape
        best_reading=df.iloc[int(inst)-1,:]
        return (best_reading,best_score)
    
    def select_best_reading_of_id(self,j,ids,cols3,data,filter2,filter2_inputs_output_csv,filter2_discrete_cuts_file):
        num=random.randint(0,2**32-1)
        readings_of_id_csv=self.results_path+"readings_of_id"+str(num)+".csv"
        pid=ids.iat[j,0]
        #print(pid)
        if self.hash_best_readings.get(pid)==None:
            (_,c)=data.shape#data has 30 columns including a column of ids
            cols=list(data.columns)
            data2=data[data[cols[0]]==pid]#get all the readings of an id
            readings=data2.iloc[:,1:c]#remove the ids column
            preterm_label=data2.iat[0,c-1]#get the PTB label at column index (c-1) i.e. column index 29
            #print('before: '+str(readings))
            readings=pd.DataFrame(readings,columns=cols3)#change the 'before37weeks' targets column to 'selected_reading' targets column 
            (r3,c3)=readings.shape
            readings=readings.astype(object)
            for i in range(r3):
                 readings.iat[i,c3-1]='?'#replace PTB label with 'unknown selection' label (?)
            #print('after: '+str(readings))
            readings.to_csv(readings_of_id_csv,index=False)
            if self.filter2_software == 'sklearn' or self.filter2_software=='keras':
                (reading_of_highest_score,score)=self.filter_readings_using_sklearn_or_keras_classifier(filter2,filter2_inputs_output_csv,readings_of_id_csv,self.results_path)
            elif self.filter2_software == 'weka':
                (reading_of_highest_score,score)=self.filter_readings_using_weka_classifier(filter2,filter2_inputs_output_csv,filter2_discrete_cuts_file,readings_of_id_csv,self.results_path,self.weka_path,self.java_memory)
            reading_of_highest_score=list(reading_of_highest_score)
            reading_of_highest_score[-1]=preterm_label#replace the 'unknown selection' (?) label with the PTB label of this id
            reading_of_highest_score.insert(0,pid)#insert id to front of list shifting elements to right
            self.hash_best_readings[pid]=(reading_of_highest_score,score)
        else:
            print('best reading of id '+pid+' has been selected already')
            best_reading=self.hash_best_readings[pid]
            reading_of_highest_score=best_reading[0]
            score=best_reading[1]
        utilities.delete_files([readings_of_id_csv]) 
        return (reading_of_highest_score,score) #format of reading_of_highest_score: patient_id,amp1,amp2,...,amp14,phase1,phase2,...,phase14,1  (class variable PTB)
        
    def select_readings_using_filtering(self,filter2,filter2_inputs_output_csv,filter2_discrete_cuts_file=None,readings_of_id=None,testset_ids=None):
        if testset_ids != None:#select best readings of the ids in testset_ids file from dataset    
            data=pd.read_csv(self.allreadings_with_ids)
            (r,c)=data.shape
            cols=list(data.columns)
            ids=pd.read_csv(testset_ids)
            (r2,_)=ids.shape    
            cols2=[]
            cols3=[]
            i=0#keep ids column in cols2
            while i < c-1:
                cols2.append(cols[i])
                i+=1
            cols2.append("before37weeksCell")
            i=1#skip ids column in cols3
            while i < c-1:
                cols3.append(cols[i])      
                i+=1
            cols3.append("selected_reading")
            if self.select_readings_parallel:        
                ####parallel version####
                #select best readings for N ids in parallel using N cpus. (select a best reading for an id using 1 cpu.)
                bestreadings_scoresL=[]    
                #try:
                if self.filter2_software == 'sklearn' or self.filter2_software == 'keras':
                    bestreadings_scoresL=Parallel(n_jobs=-1,verbose=20,batch_size=10)(delayed(self.select_best_reading_of_id)(j,ids,cols3,data,filter2,filter2_inputs_output_csv,filter2_discrete_cuts_file) for j in range(r2))
                elif self.filter2_software == 'weka':
                    bestreadings_scoresL=Parallel(n_jobs=-1)(delayed(self.select_best_reading_of_id)(j,ids,cols3,data,filter2,filter2_inputs_output_csv,filter2_discrete_cuts_file) for j in range(r2))
                    #bestreadings_scoresL=Parallel(n_jobs=-1,verbose=20,batch_size=5)(delayed(self.select_best_reading_of_id)(j,ids,cols3,data,filter2,filter2_inputs_output_csv,filter2_discrete_cuts_file) for j in range(r2))
                    #bestreadings_scoresL=Parallel(n_jobs=-1,verbose=20,batch_size=20)(delayed(self.select_best_reading_of_id)(j,ids,cols3,data,filter2,filter2_inputs_output_csv,filter2_discrete_cuts_file) for j in range(r2))    
                    #bestreadings_scoresL=Parallel(n_jobs=-1,verbose=20,batch_size=40)(delayed(self.select_best_reading_of_id)(j,ids,cols3,data,filter2,filter2_inputs_output_csv,filter2_discrete_cuts_file) for j in range(r2))       
                #except multiprocessing.TimeoutError:
                #      print('timeout')
                if len(bestreadings_scoresL) == r2:
                    print('finished select best readings of ids')
                else:
                    sys.exit(str(r2-len(bestreadings_scoresL))+' best readings were not selected: '+str(r2))
                bestreadingsL=[]
                scoresL=[]
                for bestreading_score in bestreadings_scoresL:
                    bestreading = bestreading_score[0] #format of bestreading: patient_id,amp1,amp2,...,amp14,phase1,phase2,...,phase14,1  (class variable PTB)
                    score = bestreading_score[1]
                    bestreadingsL.append(bestreading)
                    scoresL.append(score)
                bestreadings=pd.DataFrame(bestreadingsL,columns=cols2)
                ####parallel version end####
            elif self.select_readings_parallel==False:
                ###sequential version###
                bestreadingsL=[]
                scoresL=[]
                for i in range(r2):#select a best reading for each id in testset
                    pid=ids.iat[i,0]
                    data2=data[data[cols[0]]==pid]#get all the readings of an id
                    preterm_label=data2.iat[0,c-1]
                    readings=data2.iloc[:,1:c]#remove the ids column
                    readings=pd.DataFrame(readings,columns=cols3)
                    (r3,c3)=readings.shape
                    readings=readings.astype(object)
                    for j in range(r3):
                        readings.iat[j,c3-1]='?'
                    print(readings)
                    readings.to_csv(self.results_path+"readings_of_id.csv",index=False)
                    if self.filter2_software == 'weka':
                        (bestreading,score)=self.filter_readings_using_weka_classifier(filter2,filter2_inputs_output_csv,filter2_discrete_cuts_file,self.results_path+"readings_of_id.csv",self.results_path,self.weka_path,self.java_memory)
                    elif self.filter2_software == 'sklearn' or self.filter2_software == 'keras':
                        (bestreading,score)=self.filter_readings_using_sklearn_or_keras_classifier(filter2,filter2_inputs_output_csv,self.results_path+"readings_of_id.csv",self.results_path)           
                    bestreading=list(bestreading)
                    bestreading[-1]=preterm_label
                    bestreadingsL.append(bestreading)
                    scoresL.append(score)
                bestreadings=pd.DataFrame(bestreadingsL,columns=cols2)
                cols=list(bestreadings.columns)
                (_,c)=bestreadings.shape
                bestreadings[cols[c-1]]=bestreadings[cols[c-1]].astype(int)#convert targets column from float (0.0, 1.0) to int (0, 1)
                utilities.delete_files([self.results_path+"readings_of_id.csv"])
                ###sequential version end####
            return (bestreadings,scoresL)
        elif readings_of_id != None:#all readings of an id
             if self.filter2_software == 'weka':
                    (bestreading,score)=self.filter_readings_using_weka_classifier(filter2,filter2_inputs_output_csv,self.filter2type,filter2_discrete_cuts_file,readings_of_id,self.results_path,self.weka_path,self.java_memory)
             elif self.filter2_software == 'sklearn' or self.filter2_software == 'keras':
                    (bestreading,score)=self.filter_readings_using_sklearn_or_keras_classifier(filter2,filter2_inputs_output_csv,readings_of_id,self.results_path)           
             df=pd.read_csv(readings_of_id)
             cols=list(df.columns)
             bestreading=list(bestreading)
             bestreading[-1]='?'
             bestreading=[bestreading]
             cols[len(cols)-1]="before37weeksCell"
             bestreading=pd.DataFrame(bestreading,columns=cols)
             return (bestreading,score)
             
    def select_best_readings_of_patients(self,modelNumber,best_readings_testset_csv,testset_ids_csv='none',readings_of_id_csv='none'):
        ###Use the ith filter to select a best reading for each patient id of a test set
        #                   or
        # Use the ith filter to select a best reading for an id
        #output file: best_readings_testset_csv
        if self.filter2type == 'template match':
            if testset_ids_csv != 'none':#select best readings for the patients of a testset
                (bestreadings,scoresL)=utilities.template_match_filter(self.trainset_i,allreadings_with_ids,testset_ids_csv=testset_ids_csv,filter_option=self.template_filter2_option)
                bestreadings.to_csv(best_readings_testset_csv,index=False)
                mean=np.mean(scoresL)
                mini=np.min(scoresL)
                maxi=np.max(scoresL)
                return (mean,mini,maxi)
            elif readings_of_id_csv != 'none':#select a best reading for a patient id
                (bestreading,score)=utilities.template_match_filter(self.trainset_i,allreadings_with_ids,readings_of_id_csv=readings_of_id_csv,filter_option=self.template_filter2_option)
                bestreading.to_csv(best_readings_testset_csv,index=False)
                return score
        elif self.filter2type!=None:
            filter2=self.filter2_path+self.filter2type+str(modelNumber)+".model"
            filter2_inputs_output_csv=self.filter2_path+self.filter2type+modelNumber+".model_inputs_output.csv"
            filter2_discrete_cuts_file=self.filter2_path+self.filter2type+modelNumber+".discrete_cuts"
            if os.path.isfile(filter2):
                if testset_ids_csv != 'none':#select best readings for the patients of a testset
                    print('Select best readings for patients using filter2')
                    print('filter2: '+filter2)
                    print('filter2 type: '+self.filter2type)
                    print('filter2 software: '+self.filter2_software)           
                    (bestreadings,scoresL)=self.select_readings_using_filtering(filter2,filter2_inputs_output_csv,self.results_path,self.weka_path,self.java_memory,filter2_discrete_cuts_file=filter2_discrete_cuts_file,testset_ids=testset_ids_csv)
                    bestreadings.to_csv(best_readings_testset_csv,index=False)
                    mean=np.mean(scoresL)
                    mini=np.min(scoresL)
                    maxi=np.max(scoresL)
                    return (mean,mini,maxi)
                elif readings_of_id_csv != 'none':#select a best reading for a patient id
                    print('filter2: '+self.filter2type)
                    (bestreading,score)=self.select_readings_using_filtering(filter2,filter2_inputs_output_csv,self.results_path,self.weka_path,self.java_memory,filter2_discrete_cuts_file=filter2_discrete_cuts_file,readings_of_id=readings_of_id_csv)
                    bestreading.to_csv(best_readings_testset_csv,index=False)
                    return score
            else:
                print(filter2+' does not exist in select_best_readings_of_patients')
                return 'no filter2'
        else:
            print(filter2+' does not exist in select_best_readings_of_patients')
            return 'no filter2'
    
    def create_testset_of_best_readings_and_metabolite(self,modelNumber,testset_best_readings_csv,asymp_22wks_438_V1_9inputs_with_ids,testset_best_readings_and_metabolite_csv):
         #Merge the selected readings of ids of a testset with their metabolite data
         #output file: testset_best_readingsI.csv I=0,1,2,...,no_of_models-1
        bestreadings=pd.read_csv(testset_best_readings_csv)
        metabolite=pd.read_csv(asymp_22wks_438_V1_9inputs_with_ids)
        metabolite=prep.fill_missing_values('median','df',metabolite,None,has_targets_column=False)
        (r,c)=bestreadings.shape
        cols0=list(metabolite.columns)
        ids_col0=cols0[0]#name of the ids column of metabolite data
        cols0=cols0[1:len(cols0)]#skip the 1st column of ids
        cols=list(bestreadings.columns)
        ids_col=cols[0]#name of the ids column of best readings dataframe
        cols=cols[1:len(cols)]#skip the 1st column of ids
        cols2=cols0+cols
        testset_with_ids=prep.join_data(metabolite,bestreadings,ids_col0,ids_col)
        (_,c)=testset_with_ids.shape
        testset=testset_with_ids.drop(columns=[ids_col0])#drop the ids column
        testset_with_ids.iloc[:,1:c-1]=testset
        testset[cols2[len(cols2)-1]]=testset[cols2[len(cols2)-1]].astype(int)#convert targets column to int
        testset.to_csv(testset_best_readings_and_metabolite_csv,index=False)
        return testset_with_ids
    '''
    def select_best_readings_and_metabolite_for_patients(self,modelNumber,testset_ids_csv,best_readings_testset_csv,best_readings_and_metabolite_testset_csv,file):    
            #select best readings for those patients who have both metabolite data and EIS data        
            output=self.select_best_readings_of_patients(modelNumber,testset_ids_csv=testset_ids_csv,best_readings_testset_csv=best_readings_testset_csv)#use filter2 to select best readings
            if output != 'no filter2':
                 mean_score=output[0]
                 mini_score=output[1]
                 max_score=output[2]
                 print('==testset'+str(modelNumber)+'==\n')
                 print('mean score of all selected readings: '+str(mean_score))
                 print('mini score of all selected readings: '+str(mini_score))
                 print('max score of all selected readings: '+str(max_score))    
                 file.write('==testset'+str(modelNumber)+'==\n')
                 file.write('mean score of all selected readings: '+str(mean_score)+'\n')
                 file.write('mini score of all selected readings: '+str(mini_score)+'\n')
                 file.write('max score of all selected readings: '+str(max_score)+'\n')
                 ids_df=pd.read_csv(testset_ids_csv)
                 df=pd.read_csv(best_readings_testset_csv)
                 df2=ids_df.join(df)#add ids to the selected readings
                 best_readings_testset_with_ids=self.results_path+"testset_best_readings"+str(modelNumber)+"_with_ids.csv"
                 df2.to_csv(best_readings_testset_with_ids,index=False)#add ids to the best readings
                 return self.create_testset_of_best_readings_and_metabolite(modelNumber,best_readings_testset_with_ids,self.asymp_22wks_438_V1_9inputs_with_ids,best_readings_and_metabolite_testset_csv)
            else:
                 print('in select_best_readings_and_metabolite_for_patients: no filter2\n')
    '''                
    def predict_ptb_of_testsets(self,file):
        #predict PTB of 100 testsets using 100 models
        print('Predict PTB of the test sets consisting of the best readings')
        testset_aucL=[]
        testset_tprL=[]
        testset_tnrL=[]
        testset_fprL=[]
        testset_fnrL=[] 
        if self.data == 'EIS':
            for i in range(no_of_models):
                print(i)
                filter2=self.filter2_path+self.filter2type+str(i)+".model"
                if os.path.isfile(filter2):
                    testset=self.results_path+'testset_best_readings'+str(i)+'.csv'
                    (testset_aucL,testset_tprL,testset_tnrL,testset_fprL,testset_fnrL)=self.predict_using_a_model(str(i),testset=testset,testset_aucL=testset_aucL,testset_tprL=testset_tprL,testset_tnrL=testset_tnrL,testset_fprL=testset_fprL,testset_fnrL=testset_fnrL,file=file)
        elif self.data == 'EIS+metabolite':
            for i in range(no_of_models):
                print(i)
                filter2=self.filter2_path+self.filter2type+str(i)+".model"
                if os.path.isfile(filter2):
                    testset=self.results_path+"testset_best_readings_and_metabolite"+str(i)+".csv"
                    (testset_aucL,testset_tprL,testset_tnrL,testset_fprL,testset_fnrL)=self.predict_using_a_model(str(i),testset=testset,testset_aucL=testset_aucL,testset_tprL=testset_tprL,testset_tnrL=testset_tnrL,testset_fprL=testset_fprL,testset_fnrL=testset_fnrL,file=file)
        print('mean test AUC: '+str(np.mean(testset_aucL)))
        print('min test AUC: '+str(np.min(testset_aucL)))
        print('max test AUC: '+str(np.max(testset_aucL)))
        file.write('mean test AUC: '+str(np.mean(testset_aucL))+'\n')
        file.write('min test AUC: '+str(np.min(testset_aucL))+'\n')
        file.write('max test AUC: '+str(np.max(testset_aucL))+'\n')
        print('mean test TPR: '+str(np.mean(testset_tprL)))
        print('min test TPR: '+str(np.min(testset_tprL)))
        print('max test TPR: '+str(np.max(testset_tprL)))
        file.write('mean test TPR: '+str(np.mean(testset_tprL))+'\n')
        file.write('min test TPR: '+str(np.min(testset_tprL))+'\n')
        file.write('max test TPR: '+str(np.max(testset_tprL))+'\n')
        print('mean test TNR: '+str(np.mean(testset_tnrL)))
        print('min test TNR: '+str(np.min(testset_tnrL)))
        print('max test TNR: '+str(np.max(testset_tnrL)))
        file.write('mean test TNR: '+str(np.mean(testset_tnrL))+'\n')
        file.write('min test TNR: '+str(np.min(testset_tnrL))+'\n')
        file.write('max test TNR: '+str(np.max(testset_tnrL))+'\n')
        print('mean test FPR: '+str(np.mean(testset_fprL)))
        print('min test FPR: '+str(np.min(testset_fprL)))
        print('max test FPR: '+str(np.max(testset_fprL)))
        file.write('mean test FPR: '+str(np.mean(testset_fprL))+'\n')
        file.write('min test FPR: '+str(np.min(testset_fprL))+'\n')
        file.write('max test FPR: '+str(np.max(testset_fprL))+'\n')
        print('mean test FNR: '+str(np.mean(testset_fnrL)))
        print('min test FNR: '+str(np.min(testset_fnrL)))
        print('max test FNR: '+str(np.max(testset_fnrL)))
        file.write('mean test FNR: '+str(np.mean(testset_fnrL))+'\n')
        file.write('min test FNR: '+str(np.min(testset_fnrL))+'\n')
        file.write('max test FNR: '+str(np.max(testset_fnrL))+'\n')
        print('The results of PTB prediction are saved to '+self.logfile)
        
    def select_best_readings_and_create_testsets(self,file):
        #input: 100 csv testsets (without ids) 
        #
        #output: 100 testset_best_readings_csv (for EIS data)
        #        100 testset_best_readings_and_metabolite_data (for EIS + metabolite data)
        #for each testset {
        #  Get the ids of the testset and save the ids to testset_ids
        #  Get all the readings of the ids 
        #  Use a filter2 to select a best reading for each id
        #  Save the selected readings of these ids to testset_best_readings_csv
        #  if data is EIS+metabolite
        #  then merged the selected readings with metabolite data to create a testset
        #  else the testset is the set of the selected readings
        #}
        s = sd.SelectData()
        if self.data == 'EIS':
            for i in range(no_of_models):#get the ids in each test set
                print('testset'+str(i))
                testset=model_path+'testset'+str(i)+'.csv' #input csv testset (without ids)
                idsL=s.get_patients_ids2(testset,allreadings_with_ids)    
                testset_ids=model_path+"testset"+str(i)+"_ids.csv"        
                df=pd.DataFrame(idsL,columns=['hospital_ids'])
                df.to_csv(testset_ids,index=False)
                best_readings_testset_csv=self.results_path+"testset_best_readings"+str(i)+".csv"
                score=self.select_best_readings_of_patients(str(i),testset_ids_csv=testset_ids,best_readings_testset_csv=best_readings_testset_csv)#use filter2 to select best readings of patients      
                if score != 'no filter2':
                    mean_score=score[0]
                    mini_score=score[1]
                    max_score=score[2]
                    print('mean score of all selected readings: '+str(mean_score))
                    print('mini score of all selected readings: '+str(mini_score))
                    print('max score of all selected readings: '+str(max_score))    
                    file.write('mean score of all selected readings: '+str(mean_score)+'\n')
                    file.write('mini score of all selected readings: '+str(mini_score)+'\n')
                    file.write('max score of all selected readings: '+str(max_score)+'\n')
                else:
                    print(str(i)+': no filter2\n')
        else:
            sys.exit('invalid data in select_best_readings_and_create_testsets: '+self.data)
        '''
        elif self.data == 'EIS+metabolite':
            for i in range(no_of_models):#get the ids in each test set
                print('testset'+str(i))
                testset=model_path+"testset"+str(i)+".csv" #input csv testset (without ids)
                idsL=s.get_patients_ids2(testset,allreadings_with_ids)    
                testset_ids=model_path+"testset"+str(i)+"_ids.csv"        
                df=pd.DataFrame(idsL,columns=['hospital_ids'])
                df.to_csv(testset_ids,index=False)
                best_readings_testset_csv=self.results_path+"testset_best_readings"+str(i)+".csv"
                score=self.select_best_readings_of_patients(str(i),testset_ids=testset_ids,best_readings_testset_csv=best_readings_testset_csv)#use filter2 to select best readings of patients      
                if score != 'no filter2':
                    mean_score=score[0]
                    mini_score=score[1]
                    max_score=score[2]
                    print('mean score of all selected readings: '+str(mean_score))
                    print('mini score of all selected readings: '+str(mini_score))
                    print('max score of all selected readings: '+str(max_score))    
                    file.write('mean score of all selected readings: '+str(mean_score)+'\n')
                    file.write('mini score of all selected readings: '+str(mini_score)+'\n')
                    file.write('max score of all selected readings: '+str(max_score)+'\n')
                    ids_df=pd.read_csv(testset_ids)
                    df=pd.read_csv(best_readings_testset_csv)
                    df2=ids_df.join(df)#add ids to the selected readings
                    best_readings_testset_with_ids=self.results_path+"testset_best_readings"+str(i)+"_with_ids.csv"
                    testset_best_readings_and_metabolite_data=self.results_path+"testset_best_readings_and_metabolite"+str(i)+".csv"
                    df2.to_csv(best_readings_testset_with_ids,index=False)#add ids to the best readings
                    self.create_testset_of_best_readings_and_metabolite(i,best_readings_testset_with_ids,self.asymp_22wks_438_V1_9inputs_with_ids,testset_best_readings_and_metabolite_data)
                else:
                    print(str(i)+': no filter2\n')
        elif self.data == 'EIS+metabolite+CST':
            for i in range(no_of_models):#get the ids in each test set
                print('testset'+str(i))
                testset=model_path+"testset"+str(i)+".csv" #input csv testset (without ids)
                idsL=s.get_patients_ids2(testset,allreadings_with_ids)    
                testset_ids=model_path+"testset"+str(i)+"_ids.csv"        
                df=pd.DataFrame(idsL,columns=['hospital_ids'])
                df.to_csv(testset_ids,index=False)
                best_readings_testset_csv=self.results_path+"testset_best_readings"+str(i)+".csv"
                score=self.select_best_readings_of_patients(str(i),testset_ids=testset_ids,best_readings_testset_csv=best_readings_testset_csv)#use filter2 to select best readings of patients      
                if score != 'no filter2':
                    mean_score=score[0]
                    mini_score=score[1]
                    max_score=score[2]
                    print('mean score of all selected readings: '+str(mean_score))
                    print('mini score of all selected readings: '+str(mini_score))
                    print('max score of all selected readings: '+str(max_score))    
                    file.write('mean score of all selected readings: '+str(mean_score)+'\n')
                    file.write('mini score of all selected readings: '+str(mini_score)+'\n')
                    file.write('max score of all selected readings: '+str(max_score)+'\n')
                    ids_df=pd.read_csv(testset_ids)
                    df=pd.read_csv(best_readings_testset_csv)
                    df2=ids_df.join(df)#add ids to the selected readings
                    best_readings_testset_with_ids=self.results_path+"testset_best_readings"+str(i)+"_with_ids.csv"
                    testset_best_readings_and_metabolite_data=self.results_path+"testset_best_readings_and_metabolite"+str(i)+".csv"
                    df2.to_csv(best_readings_testset_with_ids,index=False)#add ids to the best readings
                    best_readings_and_metabolite=self.create_testset_of_best_readings_and_metabolite(i,best_readings_testset_with_ids,self.asymp_22wks_438_V1_9inputs_with_ids,testset_best_readings_and_metabolite_data)
                    cst_df=pd.read_csv(self.cst_with_ids)
                    df=cst_df[['ID','CSTI','CSTII','CSTIII','CSTIII/V','CSTV','Other']]
                    cols0=list(best_readings_and_metabolite.columns)
                    ids_col0=cols0[0]
                    testset_with_ids=prep.join_data(best_readings_and_metabolite,df,ids_col0,'ID')
                    testset=testset_with_ids.drop(columns=[ids_col0])
                    testset_best_readings_metabolite_and_cst=self.results_path+"testset_best_readings_metabolite_and_cst"+str(i)+".csv"
                    testset.to_cst(testset_best_readings_metabolite_and_cst,index=False)
                else:
                    print(str(i)+': no filter2\n')        
        else:
                sys.exit('data option is invalid: '+self.data+'\n The data option must be EIS or EIS+metabolite')
        '''
        
    def select_best_readings_and_create_a_testset(self,i,file,best_readings_testset_i=None,testset_i_ids=None):
        ####Select best readings of the ids of the ith testset and create a testset consisting of the selected best readings
        #input: testset_ids
        #       testset_best_readings_csv
        #score: testset_best_readings_csv (for EIS data)
        #        testset_best_readings_and_metabolite_data (for EIS + metabolite data)   
        file.write('Select best readings using '+str(i)+'th filter2 and create a testset'+'\n')
        file.write('filter2type: '+self.filter2type+'\n')
        file.write('filter2path: '+self.filter2_path+'\n')
        print('testset'+str(i))
        if best_readings_testset_i!=None:
            if testset_i_ids!=None:
               self.select_best_readings_of_patients(str(i),testset_ids_csv=testset_i_ids,best_readings_testset_csv=best_readings_testset_i)#use filter2 to select best readings of patients        
            else:
               sys.exit('testset_i_ids == None in select_best_readings_and_create_a_testset')
        else:
            sys.exit('best_readings_testset_i == None in select_best_readings_and_create_a_testset')
    
    def performance_using_threshold_parallel_step(self,preterm_probL,i,targetsL):
        threshold=preterm_probL[i]
        if threshold <= 0.9 and threshold >= 0.01:#probabilities of > 0.9 or < 0.01 are not suitable as thresholds. If a threshold > 0.9, instances tend to be classified as class0. If a threshold < 0.01, instances tend to be classified as class 1. 
                (tpr,tnr,fpr,fnr)=self.performance_using_threshold(threshold,preterm_probL,targetsL)
                return threshold,tpr,tnr,fpr,fnr

    def compute_tpr_tnr_fpr_fnr_of_thresholds(self,preterm_probL,targetsL):
        #use probabilities of preterm as thresholds to compute tpr, tnr, fpr and fnr of each the threshold
        thresholds=[]
        tprs=[]
        tnrs=[]
        fprs=[]
        fnrs=[]    
        scoreL=Parallel(n_jobs=-1,batch_size=100)(delayed(self.performance_using_threshold_parallel_step)(preterm_probL,i,targetsL) for i in range(len(preterm_probL)))
        scoreL=list(filter(lambda item: item is not None, scoreL))#delete any None from scoreL
        for threshold,tpr,tnr,fpr,fnr in scoreL:
            thresholds.append(threshold)
            tprs.append(tpr)
            tnrs.append(tnr)
            fprs.append(fpr)
            fnrs.append(fnr)
        #include 0.5 threshold to all the thresholds from preterm_probL 
        (tpr,tnr,fpr,fnr)=self.performance_using_threshold(0.5,preterm_probL,targetsL)
        tprs.append(tpr)
        tnrs.append(tnr)
        fprs.append(fpr)
        fnrs.append(fnr)
        thresholds.append(0.5)        
        return thresholds,tprs,tnrs,fprs,fnrs
    
    def performance_using_optimal_threshold_maximizing_tpr_tnr(self,preterm_probL,targetsL,ppv_npv_pos_lr_neg_lr=False,display_info=False,score='my_score',reward_scale=1/4,max_diff=0.3):
         ###find the optimal threshold maximizing tpr and tnr from preterm_probL (training set)     
         thresholds,tprs,tnrs,fprs,fnrs = self.compute_tpr_tnr_fpr_fnr_of_thresholds(preterm_probL,targetsL)
         L=[]
         for i in range(len(thresholds)):
             if score == 'my_score':
                 s = (tprs[i]+tnrs[i])/2 #my unbiased classification measure
             elif score == 'my_score2':
                 s = (tprs[i]+tnrs[i])/2+(tprs[i]-tnrs[i])*reward_scale #my score2= (tpr+tnr)/2+(tpr-tnr)*reward_scale where (tpr-tnr) is a reward if tpr > tnr and is a penalty (-ve reward) if tpr < tnr
                 #print('my_score2, reward_scale='+str(reward_scale))
             elif score=='my_score3':
                 s=(tprs[i]+tnrs[i])/2 + reward_scale * (tprs[i]-tnrs[i])/np.abs(tprs[i]-tnrs[i]) * np.exp(-(tprs[i]-tnrs[i]-max_diff))
                 #print('my_score2, reward_scale='+str(reward_scale)+', max_difference='+str(max_diff))
             elif score == 'G-mean':
                 s = np.sqrt(tprs[i]*tnrs[i]) #G-mean=sqrt(tpr*tnr) is an unbiased classification measure
             elif score == 'youden':
                 s = tprs[i] + tnrs[i] - 1    #youden's index = sensitivity + specificity - 1
             L.append((thresholds[i],s,tprs[i],tnrs[i],fprs[i],fnrs[i]))
         L.sort(key=operator.itemgetter(1),reverse=True)#sort thresholds in descending order of score    
         for i in range(len(L)):
             performance=L[i]
             if display_info:
                 print('threshold: '+str(performance[0])+', tpr='+str(performance[2])+', tnr='+str(performance[3])+', fpr='+str(performance[4])+', fnr='+str(performance[5]))
         optimal_threshold=L[0]
         #print('optimal threshold: '+str(optimal_threshold[0])+', tpr='+str(optimal_threshold[2])+', tnr='+str(optimal_threshold[3])+', fpr='+str(optimal_threshold[4])+', fnr='+str(optimal_threshold[5]))
         if ppv_npv_pos_lr_neg_lr:#PPV, NPV, Positive Likelihood Ratio, Negative Likelihood Ratio
             (tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)=self.performance_using_threshold(optimal_threshold[0],preterm_probL,targetsL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr)
             return (optimal_threshold[0],tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)
         else:
             return (optimal_threshold[0],optimal_threshold[2],optimal_threshold[3],optimal_threshold[4],optimal_threshold[5])
     
    def performance_using_threshold_maximizing_youden_index(self,preterm_probL,targetsL,ppv_npv_pos_lr_neg_lr=False,k=0,display_info=False):
         ###find the kth threshold maximizing youden index from preterm_probL (training set)              
         thresholds,tprs,tnrs,fprs,fnrs = self.compute_tpr_tnr_fpr_fnr_of_thresholds(preterm_probL,targetsL)
         L=[]
         for i in range(len(thresholds)):
             score = tprs[i] + tnrs[i] - 1 #youden's index = sensitivity + specificity - 1
             L.append((thresholds[i],score,tprs[i],tnrs[i],fprs[i],fnrs[i]))
             if display_info:
                 print('threshold: '+str(thresholds[i])+', tpr='+str(tprs[i])+', tnr='+str(tnrs[i])+', fpr='+str(fprs[i])+', fnr='+str(fnrs[i]))            
         L.sort(key=operator.itemgetter(1),reverse=True)#sort thresholds in descending order of score    
         optimal_threshold=L[k]
         #print('optimal threshold: '+str(optimal_threshold[0])+', tpr='+str(optimal_threshold[2])+', tnr='+str(optimal_threshold[3])+', fpr='+str(optimal_threshold[4])+', fnr='+str(optimal_threshold[5]))
         if ppv_npv_pos_lr_neg_lr:#PPV, NPV, Positive Likelihood Ratio, Negative Likelihood Ratio
             (tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)=self.performance_using_threshold(optimal_threshold[0],preterm_probL,targetsL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr)
             return (optimal_threshold[0],tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)
         else:
             return (optimal_threshold[0],optimal_threshold[2],optimal_threshold[3],optimal_threshold[4],optimal_threshold[5])
     
    def performance_using_threshold(self,threshold,preterm_probL,targetsL,ppv_npv_pos_lr_neg_lr=False):
        #use a threshold to measure prediciton performance
        #print('preterm_probL='+str(preterm_probL))
        predL = (np.array(preterm_probL) >= threshold).astype(int)
        tp=0#TP
        tn=0#TN
        fp=0#false positive
        fn=0#false negative
        p=0 #total no. of postive (preterm)
        n=0#total no. of negative (onterm)
        tpr=-999
        tnr=-999
        fpr=-999
        fnr=-999
        #print('targetsL:',targetsL)
        #print('predL:',predL)
        for i in range(len(predL)):
            if targetsL[i]==1:
                p+=1
                if predL[i]==1:
                    tp+=1
                else:
                    fn+=1
            else: #targetsL[i]==0
                n+=1            
                if predL[i]==0:
                    tn+=1
                else:
                    fp+=1
        if p>0 and n>0:
            tpr=tp/p
            tnr=tn/n
            fpr=fp/(fp+tn)
            fnr=fn/(fn+tp)
            if ppv_npv_pos_lr_neg_lr:#PPV, NPV, Positive Likelihood Ratio, Negative Likelihood Ratio
                if tp+fp==0:
                    ppv=float('inf')
                else:
                    ppv=tp/(tp+fp)
                if tn+fn==0:
                    npv=float('inf')
                else:
                    npv=tn/(tn+fn)
                if tnr < 1:
                    pos_lr=tpr/(1-tnr) 
                else:
                    pos_lr=float('inf')
                if tnr > 0:
                    neg_lr=(1-tpr)/tnr
                else:
                    neg_lr=float('inf')
                return (tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)
            else:                
                return (tpr,tnr,fpr,fnr)        
        else:
            sys.exit('invalid values of p and n in performance_using_threshold, p='+str(p)+', n='+str(n))
    
    def predict_using_all_spectra_of_each_patient(self,model,modeltype,model_inputs_output_csv,testset_csv,results_path,youden_index=False,k_youden_threshold=0,threshold=0.5,software='weka',final_prob='average of all probs'):
        #testset_csv, dataset with ids
        #software='weka' or 'sklearn' or 'keras'
        #for each SYMP patient with c1, c2 and c3 spectra of visit 1:
        #   c1, c2 and c3 spectra -> model -> prob1 of PTB, prob2 of PTB and prob3 of PTB -> mean -> prob of PTB
        #for each AHR patient with c1, c2 and c3 spectra of visit 1 and c1, c2 and c3 spectra of visit 2:
        #   1) c1, c2 and c3 spectra of visit1 -> model -> prob1 of PTB, prob2 of PTB and prob3 of PTB -> mean ->prob1 of PTB
        #   2) c1, c2 and c3 spectra of visit2 -> model -> prob1 of PTB, prob2 of PTB and prob3 of PTB -> mean ->prob2 of PTB
        #   3) prob1 of PTB x w1, prob2 of PTB x w2 -> mean -> prob of PTB 
        #input: testset_arff containing all spectra of ids in the testset
        #output: (predL,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)            
        #print('for each SYMP patient with c1, c2 and c3 spectra of a visit:')
        #print('   1) c1, c2 and c3 spectra -> model -> prob1 of PTB, prob2 of PTB and prob3 of PTB')
        #print('   2) Prob of PTB = (prob1 of PTB + prob2 of PTB + prob3 of PTB)/3')
        #print('for each AHR patient with c1, c2 and c3 spectra of visit 1 and c1, c2 and c3 spectra of visit 2:')
        #print('   1) c1, c2 and c3 spectra of visit1 -> model -> prob1 of PTB, prob2 of PTB and prob3 of PTB')
        #print('   2) Prob1 of PTB = (prob1 of PTB + prob2 of PTB + prob3 of PTB)/3')    
        #print('   3) c1, c2 and c3 spectra of visit2 -> model -> prob1 of PTB, prob2 of PTB and prob3 of PTB')
        #print('   4) Prob2 of PTB = (prob1 of PTB + prob2 of PTB + prob3 of PTB)/3')
        #print('   5) Prob of PTB = (Prob1 of PTB + Prob2 of PTB)/2')        
        #data=utilities.arff_to_dataframe(testset_arff)
        #data.info()
        data=pd.read_csv(testset_csv)
        (_,c)=data.shape#data has 30 columns including a column of ids
        pids=list(set(data.iloc[:,0]))#remove duplicate ids
        probL=Parallel(n_jobs=-1,verbose=20,batch_size=10)(delayed(self.predict_ptb_of_id_using_all_spectra_of_id)(j,pids,data,model,modeltype,model_inputs_output_csv,results_path,self.weka_path,self.java_memory,software=software,final_prob=final_prob) for j in range(len(pids)))
        #probL is a list of (preterm_prob, actual_class)
        #print(probL)
        prob_df=pd.DataFrame(probL,columns=['preterm_prob','actual_class'])
        preterm_probs=prob_df.iloc[:,0]
        actual_classes=prob_df.iloc[:,1]
        #print('preterm probs: '+str(preterm_probs))
        #print('actual_classes: '+str(actual_classes))
        auc=roc_auc_score(actual_classes,preterm_probs)
        #preterm_probL=list(preterm_probs)
        #actual_classL=list(actual_classes)
        if youden_index:#use the optimal threshold maximizing youden's index (sensitivity+specificity-1) 
           (youden_threshold,tpr,tnr,fpr,fnr)=self.performance_using_threshold_maximizing_youden_index(preterm_probs,actual_classes,k=k_youden_threshold)                
           predicted_classes = (preterm_probs >= youden_threshold).astype(int)
           self.youden_threshold=youden_threshold
        else:#use a specified threshold to evaluate performance
           (tpr,tnr,fpr,fnr)=self.performance_using_threshold(threshold,preterm_probs,actual_classes)
           predicted_classes = (preterm_probs >= threshold).astype(int)
        #predL=list of (pid,float(preterm_prob),predicted_class,actual_class,error)
        predL=[]
        for i in range(len(pids)):
            if predicted_classes.iloc[i] != actual_classes.iloc[i]:
                error='+'
            else:
                error=' '
            predL.append((pids[i],preterm_probs.iloc[i],predicted_classes.iloc[i],actual_classes.iloc[i],error))
        return (predL,auc,tpr,tnr,fpr,fnr)
    
    def predict_ptb_of_id_using_all_spectra_of_id(self,j,pids,data,model,modeltype,model_inputs_output_csv,results_path,weka_path,java_memory,software='weka',final_prob='average_of_majority_probs'):
        pid=pids[j]        
        cols=list(data.columns)
        data2=data[data[cols[0]]==pid]#get all the readings of an id
        (r,c)=data2.shape
        readings=data2.iloc[:,1:c]#skip the ids column
        actual_label=int(data2.iat[0,c-1])
        readings=readings.astype(object)
        (r,c)=readings.shape
        for i in range(r):
            readings.iat[i,c-1]='?'#replace PTB label with unknown label '?' because no need to compute the auc of all the spectra of this id
        num=random.randint(0,2**32-1)
        readings_of_id_csv=results_path+"readings_of_id"+str(num)+".csv"
        readings.to_csv(readings_of_id_csv,index=False)
        if software=='weka':
            discrete_cuts_file=results_path+modeltype+str(self.ModelNumber)+'.discrete_cuts'
            predL=self.predict_using_weka_model('prediction list',readings_of_id_csv,discrete_cuts_file,model,modeltype,model_inputs_output_csv,results_path,weka_path,java_memory)
        elif software=='sklearn' or software=='keras':
            predL=self.predict_using_sklearn_or_keras_model('prediction list',num,readings_of_id_csv,model,model_inputs_output_csv,results_path,software=software,linear_reg=False)
        #predL=list of (inst,float(preterm_prob),output_class,actual_class,error)
        utilities.delete_files([readings_of_id_csv])
        final_preterm_prob=None
        if final_prob == 'average of majority probs':
            print('average of the majority probabilities which are >= 0.5 or < 0.5')
            if len(predL)<=2:
                s=0
                for i in range(len(predL)):
                    pred=predL[i]
                    preterm_prob=pred[1]
                    s+=preterm_prob
                final_preterm_prob=s/len(predL)
            else:#average of the majority probabilities which are >= 0.5 or < 0.5 while ignoring the minority predictions which are >= 0.5 or < 0.5 
                k=0 #no. of probs of >= 0.5
                for i in range(len(predL)):
                    pred=predL[i]
                    preterm_prob=pred[1]
                    if preterm_prob >= 0.5:
                           k+=1
                if k/len(predL) >= 0.5:#most probs >= 0.5, final prob = average of the probs >= 0.5
                    s=0
                    for i in range(len(predL)):
                        pred=predL[i]
                        preterm_prob=pred[1]
                        if preterm_prob >= 0.5:
                            s+=preterm_prob
                    final_preterm_prob=s/k
                else: #most probs < 0.5, final prob = average of the probs < 0.5
                    s=0
                    for i in range(len(predL)):
                        pred=predL[i]
                        preterm_prob=pred[1]
                        if preterm_prob < 0.5:
                            s+=preterm_prob
                    final_preterm_prob=s/(len(predL)-k)
        elif final_prob =='average of all probs':
            ###final prob of PTB = average of all probs of all visits
            s=0
            for j in range(len(predL)):
                pred=predL[j]
                preterm_prob=pred[1]
                s+=preterm_prob
            final_preterm_prob=s/len(predL)
        else:
            sys.exit('invalid final_prob in predict_ptb_of_id_using_all_spectra_of_id: '+final_prob)
        return (final_preterm_prob,actual_label)
       
    def predict_using_weka_model(self,prediction_list_option,testset,discrete_cuts_file,classifier,classifiertype,model_inputs_output_csv,results_path,weka_path,java_memory,ppv_npv_pos_lr_neg_lr=False,k_youden_threshold=0,threshold=0.5):
        ###Predict prob of PTB of a testset using a weka model, then classify the testset using the optimal threshold found from the testset or the user-specified threshold
        #input: testset, a .csv test set file consisting of original features or polynomial features
        #       model_path, path to the model
        #       modeltype, 'log_reg' or 'rf' (random forest)
        #       testset_arff
        #       results_path, path to results files and temporary files 
        #Remove any training instances from the test set and predict the unseen test instances
        #print('###model_inputs_output_csv: '+model_inputs_output_csv)
        num=random.randint(0,2**32-1)
        file=open(testset,'r')
        l=(file.readline()).strip()
        l=l.split(',')
        testset_class=l[-1]
        testset_features=set(l)
        file.close()
        file2=open(model_inputs_output_csv,'r')
        l2=(file2.readline()).strip()
        l2=l2.split(',')
        model_class=l2[-1]
        model_features=set(l2)        
        file2.close()
        if testset_class != model_class:
            sys.exit('class variable of test_csv: \''+testset_class+'\' is different to class variable of model: \''+model_class+'\'')
        elif testset_features == model_features:
            #print('inputs of model are all the features')
            df=pd.read_csv(testset)
        elif model_features.issubset(testset_features):#the inputs of model are original features and the testset consists of original features
            #print('inputs of model is a feature subset')
            reduced_testset=results_path+'testset_reduced'+str(num)+'.csv'
            prep.reduce_data2('csv',testset,model_inputs_output_csv,reduced_testset)
            df=pd.read_csv(reduced_testset)
        else:#the inputs of model are polynomial features, but the testset consists of original features. Then, construct the same polynomial features of the model for the testset
            #print('inputs of model are polynomial features')
            reduced_testset=results_path+'testset_reduced'+str(num)+'.csv'
            utilities.construct_poly_features_of_another_dataset('original_features',testset,model_inputs_output_csv,reduced_testset,'none')
            df=pd.read_csv(reduced_testset)
        tempfilesL=[]
        tempfilesL.append(reduced_testset)
        #print(df)
        df=prep.fill_missing_values('median','df',df,outfile=None,has_targets_column=True)
        if self.scaler_file!=None:
           df=utilities.transform_inputs_df_using_scaler(df,self.scaler_file)
        reduced_testset_arff=results_path+'testset_reduced'+str(num)+'.arff'
        tempfilesL.append(reduced_testset_arff)    
        utilities.dataframe_to_arff(df,reduced_testset_arff)
        #If the arff file contains unknown classes, replace the missing target (?) or string target to nominal target e.g. @attribute before37weeksCell {0,1}
        (_,c)=df.shape
        target=df.iat[0,c-1]   
        if target == '?':#unknown class
           utilities.replace_missing_target_or_string_target_with_nominal_target(model_inputs_output_csv,reduced_testset_arff)
        if discrete_cuts_file!=None and os.path.isfile(discrete_cuts_file):#the inputs of weka model are discrete features
                #print('discretize inputs')
                reduced_testset_discrete_arff=results_path+'testset_reduced_discrete'+str(num)+'.arff'
                prep.discretize_using_cuts(reduced_testset_arff,discrete_cuts_file,reduced_testset_discrete_arff,'.',java_memory)
                if self.ordinal_encode:
                    prep.ordinal_encode(results_path+'testset_reduced_discrete_integer'+str(num)+'.arff',arff_discrete_train=discrete_cuts_file,arff_discrete_test=reduced_testset_discrete_arff)
                    testset_arff=results_path+'testset_reduced_discrete_integer'+str(num)+'.arff'          
                    tempfilesL.append(reduced_testset_discrete_arff)
                else:
                    testset_arff=reduced_testset_discrete_arff
                #print('discretize inputs done')
        else:#the inputs of weka model are continuous features
                testset_arff=results_path+'testset_reduced'+str(num)+'.arff'
        tempfilesL.append(testset_arff)
        tempfilesL.append(reduced_testset_arff)
        c=cl.Classifier('',testset,weka_path,java_memory)
        results=results_path+'testresults'+str(num)+'.txt'
        results2=results_path+'testsetresults2'+str(num)+'.txt'
        tempfilesL.append(results)
        tempfilesL.append(results2)
        if classifiertype=='log_reg':
            if target != '?':
                c.log_reg_predict(classifier,testset_arff,results)
                auc=post.get_auc(results)
            c.log_reg_predict2(classifier,testset_arff,results2)#list of predictions of instances
        elif classifiertype=='rf':
            if target != '?':
                c.random_forest_predict(classifier,testset_arff,results)
                auc=post.get_auc(results)
            c.random_forest_predict2(classifier,testset_arff,results2)
        elif classifiertype=='rbf_network':
            if target != '?':
                c.rbf_network_predict(classifier,testset_arff,results)
                auc=post.get_auc(results)
            c.rbf_network_predict2(classifier,testset_arff,results2)
        elif classifiertype=='rbf_classifier':
            if target != '?':
                c.rbf_classifier_predict(classifier,testset_arff,results)
                auc=post.get_auc(results)
            c.rbf_classifier_predict2(classifier,testset_arff,results2)
        elif classifiertype=='poly_svm':
            if target != '?':
                c.poly_svm_predict(classifier,testset_arff,results)
                auc=post.get_auc(results)
            c.poly_svm_predict2(classifier,testset_arff,results2)
        elif classifiertype=='poly_libsvm':
            if target != '?':
                c.poly_libsvm_predict(classifier,testset_arff,results)
                auc=post.get_auc(results)
            c.poly_libsvm_predict2(classifier,testset_arff,results2)
        elif classifiertype=='rbf_svm':
            if target != '?':
                c.rbf_svm_predict(classifier,testset_arff,results)
                auc=post.get_auc(results)
            c.rbf_svm_predict2(classifier,testset_arff,results2)
        else:
            sys.exit('modeltype is invalid: '+classifiertype+'\n')
        if target == '?':
            (predL,_,_,_,_)=post.model_output(results2)
            #print('predL'+str(predL))
        else:
                (predL,_,_,_,_)=post.model_output(results2)#tpr is computed using 0.5 threshold
                #predL=list of (inst,float(preterm_prob),output_class,actual_class,error)
                preterm_probL=[]
                actual_classL=[]
                for i in range(len(predL)):
                    pred=predL[i]
                    preterm_prob=pred[1]
                    actual_class=pred[3]
                    preterm_probL.append(float(preterm_prob))
                    actual_classL.append(float(actual_class))
                if threshold=='optimal threshold':#use the optimal threshold maximizing tpr and tnr of the input data to classify the input data
                    if ppv_npv_pos_lr_neg_lr:
                        (_,tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)=self.performance_using_optimal_threshold_maximizing_tpr_tnr(preterm_probL,actual_classL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr)
                    else:
                        (_,tpr,tnr,fpr,fnr)=self.performance_using_optimal_threshold_maximizing_tpr_tnr(preterm_probL,actual_classL)                       
                elif threshold=='youden index':#use the threshold maximizing youden's index=sensitivity+specificity-1) of the input data to classify the input data                    
                    if ppv_npv_pos_lr_neg_lr:
                        (optimalthreshold,tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)=self.performance_using_threshold_maximizing_youden_index(preterm_probL,actual_classL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr,k=k_youden_threshold)
                    else:
                        (optimalthreshold,tpr,tnr,fpr,fnr)=self.performance_using_threshold_maximizing_youden_index(preterm_probL,actual_classL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr,k=k_youden_threshold)                    
                    self.youden_threshold=optimalthreshold
                    print(self.youden_threshold)
                else:#use a specified threshold e.g. 0.5 (default) to evalate performance
                    if ppv_npv_pos_lr_neg_lr:
                        (tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)=self.performance_using_threshold(threshold,preterm_probL,actual_classL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr)
                    else:
                        (tpr,tnr,fpr,fnr)=self.performance_using_threshold(threshold,preterm_probL,actual_classL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr)     
        utilities.delete_files(tempfilesL)
        if target != '?':
           if prediction_list_option=='prediction list':
              if ppv_npv_pos_lr_neg_lr:
                   return (auc,predL,tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)
              else:
                   return (auc,predL,tpr,tnr,fpr,fnr)                    
           else:
                   if ppv_npv_pos_lr_neg_lr:
                       return (auc,tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)
                   else:
                       return (auc,tpr,tnr,fpr,fnr)
        else:#unknown targets
            return predL
        print('===prediction finished===')  
            
    def filter_readings_using_sklearn_or_keras_classifier(self,filter2,filter2_inputs_output_csv,readings_csv,results_path):
        num=random.randint(0,2**32-1)
        df=pd.read_csv(readings_csv)
        (_,c)=df.shape
        target=df.iat[0,c-1]
        if target != '?':#known targets
            (_,predL,_,_,_,_)=self.predict_using_sklearn_or_keras_model('prediction list',num,readings_csv,filter2,filter2_inputs_output_csv,results_path,software=self.filter2_software)
        else:   
            predL=self.predict_using_sklearn_or_keras_model('prediction list',num,readings_csv,filter2,filter2_inputs_output_csv,results_path,software=self.filter2_software)
        #print('====scores of all the readings of the id===\n')
        for i in range(len(predL)):
            (indx,score,_)=predL[i]
            #print('score of reading '+str(int(indx+1))+': '+str(score))
        (indx,best_score,_)=predL[0]
        #print('\nreading '+str(int(indx)+1)+' is selected (score: '+str(best_score)+')\n')
        df=pd.read_csv(readings_csv)
        (_,c)=df.shape
        best_reading=df.iloc[int(indx),:]#keep the last column containing the labels 'selected_reading'
        return (best_reading,best_score)
        
    def predict_using_sklearn_or_keras_model(self,prediction_list_option,randNum,testset_csv,model,model_inputs_output_csv,results_path,software='sklearn',ppv_npv_pos_lr_neg_lr=False,linear_reg=False,k_youden_threshold=0,threshold=0.5):
        #predict PTB of a testset using a sklearn model or keras model e.g. a neural network  
        #testset_csv, dataset with no ids
        file=open(testset_csv,'r')
        l=(file.readline()).strip()
        l=l.split(',')
        testset_class=l[-1]
        testset_features=set(l)
        file.close()
        file2=open(model_inputs_output_csv,'r')
        l2=(file2.readline()).strip()
        l2=l2.split(',')
        model_class=l2[-1]
        model_features=set(l2)        
        file2.close()
        if testset_class != model_class:
            sys.exit('class variable of test_csv: \''+testset_class+'\' is different to class variable of model: \''+model_class+'\'')
        elif testset_features == model_features:
            #print('inputs of model are all the features')
            df=pd.read_csv(testset_csv)
        elif model_features.issubset(testset_features):#the inputs of model is a subset of original features
            #print('inputs of model is a feature subset')
            reduced_testset=results_path+'testset_reduced'+str(randNum)+'.csv'
            prep.reduce_data2('csv',testset_csv,model_inputs_output_csv,reduced_testset)
            df=pd.read_csv(reduced_testset)
            utilities.delete_files([reduced_testset])
        else:#the inputs of model are polynomial features
            #print('inputs of model are polynomial features')
            reduced_testset=results_path+'testset_reduced'+str(randNum)+'.csv'
            utilities.construct_poly_features_of_another_dataset('original_features',testset_csv,model_inputs_output_csv,reduced_testset,'none')
            df=pd.read_csv(reduced_testset)
            utilities.delete_files([reduced_testset])
        df=prep.fill_missing_values('median','df',df,None)
        #print('features of testset df: ',str(df.columns))
        if software=='keras':
            from keras.models import load_model
            network=load_model(model)
            if self.X_scaler!=None:
                X_scaler2=load(self.X_scaler)
            else:
                sys.exit('X_scaler is None')
        elif software=='sklearn':
            model=load(model)
        else:
            sys.exit('invalid software: '+software)
        (r,c)=df.shape
        target=df.iat[0,c-1]
        if software == 'sklearn':
            if target == '?':#unknown targets, predict the classes of the testset without computing AUC etc
                if linear_reg==False:
                    prob=model.predict_proba(df.iloc[:,0:c-1])
                else:#linear regression
                    prob=model.predict(df.iloc[:,0:c-1])
                    for i in range(r):
                        if prob[i] > 1:
                            prob[i]=1
                        elif prob[i] < 0:
                            prob[i]=0
                predL=[]
                for i in range(r):
                    class1_prob=prob[i,1]
                    class0_prob=1-class1_prob
                    if class1_prob >= class0_prob: #classify instance using 0.5 threshold
                        y=1
                    else:
                        y=0
                    predL.append((i,float(class1_prob),y))
                predL=sorted(predL,key=operator.itemgetter(1),reverse=True)#sort patients in descending order of preterm probability
                #print('predL: '+str(predL))
                return predL
            else:#known targets, compute AUC, TPR etc.      
                (auc,predL,_,_,_,_)=utilities.predict_testset(True,model,df,linear_reg=linear_reg)
                #predL=list of (inst,float(preterm_prob),output_class,actual_class,error)
                preterm_probL=[]
                actual_classL=[]
                for i in range(len(predL)):
                    pred=predL[i]
                    preterm_prob=pred[1]
                    actual_class=pred[3]
                    preterm_probL.append(float(preterm_prob))
                    actual_classL.append(float(actual_class))
                if threshold=='optimal threshold':#use the optimal threshold maximizing tpr and tnr of the input data to classify the input data
                    if ppv_npv_pos_lr_neg_lr:
                        (_,tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)=self.performance_using_optimal_threshold_maximizing_tpr_tnr(preterm_probL,actual_classL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr)
                    else:
                        (_,tpr,tnr,fpr,fnr)=self.performance_using_optimal_threshold_maximizing_tpr_tnr(preterm_probL,actual_classL)                       
                elif threshold=='youden index':#use the threshold maximizing youden's index=sensitivity+specificity-1) of the input data to classify the input data                    
                    if ppv_npv_pos_lr_neg_lr:
                        (optimalthreshold,tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)=self.performance_using_threshold_maximizing_youden_index(preterm_probL,actual_classL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr,k=k_youden_threshold)
                    else:
                        (optimalthreshold,tpr,tnr,fpr,fnr)=self.performance_using_threshold_maximizing_youden_index(preterm_probL,actual_classL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr,k=k_youden_threshold)                    
                    self.youden_threshold=optimalthreshold
                else:#use a specified threshold e.g. 0.5 (default) to evalate performance
                    if ppv_npv_pos_lr_neg_lr:
                        (tpr,tnr,fpr,fnr,ppv,npv,pos_lr,neg_lr)=self.performance_using_threshold(threshold,preterm_probL,actual_classL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr)
                    else:
                        (tpr,tnr,fpr,fnr)=self.performance_using_threshold(threshold,preterm_probL,actual_classL,ppv_npv_pos_lr_neg_lr=ppv_npv_pos_lr_neg_lr)                 
                if prediction_list_option=='no prediction list':
                    return (auc,tpr,tnr,fpr,fnr)
                elif prediction_list_option=='prediction list':
                    return (auc,predL,tpr,tnr,fpr,fnr)
        elif software == 'keras':#use neural network to predict testset
            if target == '?':#unknown targets, predict the classes of the testset without computing AUC etc
                from workflow_nn import nn_predict
                return nn_predict(network,df,X_scaler2,testset_labels='unknown')            
            else:#known targets
                if prediction_list_option=='no prediction list':
                    (auc,tpr,tnr,fpr,fnr)=nn_predict(network,df,X_scaler2)
                    return (auc,tpr,tnr,fpr,fnr)
                elif prediction_list_option=='prediction list':
                    (predL,auc,tpr,tnr,fpr,fnr)=nn_predict(network,df,X_scaler2,prediction_list=True)
                    return (predL,auc,tpr,tnr,fpr,fnr)

    def predict_using_a_model(self,modelNumber,testset,testset_aucL='',testset_tprL='',testset_tnrL='',testset_fprL='',testset_fnrL='',file='none'):
            #Predict PTB of a testset using a model
            print('modelNumber=',modelNumber)
            print('modeltype=',self.modeltype)
            print('modelpath=',self.model_path)
            model=self.model_path+self.modeltype+str(modelNumber)+'.model'
            df=pd.read_csv(testset)
            (_,c)=df.shape
            target=df.iat[0,c-1]
            if os.path.isfile(model) and os.path.isfile(testset):
                model_inputs_output_csv=self.model_path+self.modeltype+str(modelNumber)+'.model_inputs_output.csv'
                discrete_cuts_file=self.model_path+self.modeltype+str(modelNumber)+'.discrete_cuts'
                if self.model_software=='weka' and target != '?':
                        (test_auc,tpr,tnr,fpr,fnr)=self.predict_using_weka_model('no prediction list',testset,discrete_cuts_file,model,self.modeltype,model_inputs_output_csv,self.results_path,self.weka_path,self.java_memory)        
                elif self.model_software=='sklearn' or self.model_software=='keras' and target != '?':
                    num=random.randint(0,2**32-1)
                    (test_auc,tpr,tnr,fpr,fnr)=self.predict_using_sklearn_or_keras_model('no prediction list',num,testset,model,model_inputs_output_csv,self.results_path,option=self.model_software)#use sklearn models to predict testset
                elif self.model_software=='weka' and target == '?':
                    predL=self.predict_using_weka_model('prediction list',testset,discrete_cuts_file,model,self.modeltype,model_inputs_output_csv,self.results_path,self.weka_path,self.java_memory)        
                elif self.model_software=='sklearn' or self.model_software=='keras' and target == '?':
                    num=random.randint(0,2**32-1)
                    predL=self.predict_using_sklearn_or_keras_model('prediction list',num,testset,model,model_inputs_output_csv,self.results_path,option=self.model_software)#use sklearn models to predict testset
                if target != '?':
                    testset_aucL.append(float(test_auc))
                    testset_tprL.append(float(tpr))
                    testset_tnrL.append(float(tnr))
                    testset_fprL.append(float(fpr))
                    testset_fnrL.append(float(fnr))
                    print('model'+str(modelNumber)+'='+model)
                    print('testset AUC='+str(test_auc))
                    if file != 'none':
                        file.write('model'+str(modelNumber)+'='+model+'\n')
                        file.write('testset AUC='+str(test_auc)+'\n')
                    return (testset_aucL,testset_tprL,testset_tnrL,testset_fprL,testset_fnrL)
                elif target == '?':
                    return predL
            else:
                sys.exit(model+' does not exist or '+testset+' does not exist.' )
    
    def predict_ptb_of_testset(self,file='none',testset_i=None,i=''):
        #preditc PTB of ith testset using ith model
        if file != 'none' and self.filter2_software!=None:
            file.write('filter2type: '+self.filter2type+'\n')
            file.write('filter2path: '+self.filter2_path+'\n')
            file.write('modeltype='+self.modeltype+'\n')
            file.write('model path='+self.model_path+'\n')
            file.write('results path='+self.results_path+'\n')
            file.write('weka path='+self.weka_path+'\n')
            file.write('java memory='+self.java_memory+'\n')
        #predict PTB of the ith testset using the ith models
        #print('Predict PTB of '+str(i)+' test set consisting of the best readings')
        testset_aucL=[]
        testset_tprL=[]
        testset_tnrL=[]
        testset_fprL=[]
        testset_fnrL=[]
        #testset1_aucL=[]#testset1_i at U:\EIS preterm prediction\trainsets1trainsets2\asymp_22wks_filtered_data_28inputs_no_treatment
        #testset1_tprL=[]
        #testset1_tnrL=[]
        #testset1_fprL=[]
        #testset1_fnrL=[]
        target=''
        #print('testset'+str(i))
        if testset_i != None:#predict testset_i
           testset=testset_i
           df=pd.read_csv(testset)
           (_,c)=df.shape
           target=df.iat[0,c-1]
           (testset_aucL,testset_tprL,testset_tnrL,testset_fprL,testset_fnrL)=self.predict_using_a_model(str(i),testset,testset_aucL=testset_aucL,testset_tprL=testset_tprL,testset_tnrL=testset_tnrL,testset_fprL=testset_fprL,testset_fnrL=testset_fnrL,file=file)
        '''
        elif self.data == 'EIS':
            testset=self.results_path+'testset_best_readings'+str(i)+'.csv'
            df=pd.read_csv(testset)
            (_,c)=df.shape
            target=df.iat[0,c-1]
            (testset_aucL,testset_tprL,testset_tnrL,testset_fprL,testset_fnrL)=self.predict_using_a_model(str(i),testset,testset_aucL=testset_aucL,testset_tprL=testset_tprL,testset_tnrL=testset_tnrL,testset_fprL=testset_fprL,testset_fnrL=testset_fnrL,file=file)
             #testset1=testset_ids_path+'testset1_'+str(i)+'.csv'
             #(testset1_aucL,testset1_tprL,testset1_tnrL,testset1_fprL,testset1_fnrL)=self.predict_using_a_model(str(i),testset1,testset_aucL=testset1_aucL,testset_tprL=testset_tprL,testset_tnrL=testset_tnrL,testset_fprL=testset1_fprL,testset_fnrL=testset1_fnrL,file=file)            
        elif self.data == 'EIS+metabolite':
             file.write(str(i)+'\n')
             testset=self.results_path+"testset_best_readings_and_metabolite"+str(i)+".csv"
             df=pd.read_csv(testset)
             (_,c)=df.shape
             target=df.iat[0,c-1]
             (testset_aucL,testset_tprL,testset_tnrL,testset_fprL,testset_fnrL)=self.predict_using_a_model(str(i),testset,testset_aucL=testset_aucL,testset_tprL=testset_tprL,testset_tnrL=testset_tnrL,testset_fprL=testset_fprL,testset_fnrL=testset_fnrL,file=file)
             #testset1=testset_ids_path+'testset1_'+str(i)+'.csv'
             #(testset1_aucL,testset1_tprL,testset1_tnrL,testset1_fprL,testset1_fnrL)=self.predict_using_a_model(str(i),testset1,testset_aucL=testset1_aucL,testset_tprL=testset1_tprL,testset_tnrL=testset1_tnrL,testset_fprL=testset1_fprL,testset_fnrL=testset1_fnrL,file=file)            
        elif self.data == 'EIS+metabolite+CST':
             file.write(str(i)+'\n')
             testset=self.results_path+"testset_best_readings_metabolite_and_cst"+str(i)+".csv"
             df=pd.read_csv(testset)
             (_,c)=df.shape
             target=df.iat[0,c-1]
             (testset_aucL,testset_tprL,testset_tnrL,testset_fprL,testset_fnrL)=self.predict_using_a_model(str(i),testset,testset_aucL=testset_aucL,testset_tprL=testset_tprL,testset_tnrL=testset_tnrL,testset_fprL=testset_fprL,testset_fnrL=testset_fnrL,file=file)   
        elif self.data == 'readings of an id' or self.data == 'metabolite':#testset_i contains all the readings of an id or metabolite data. Use model i to predict testset_i
            df=pd.read_csv(testset_i)
            (_,c)=df.shape
            target=df.iat[0,c-1]
            if target != '?': 
                (testset_aucL,testset_tprL,testset_tnrL,testset_fprL,testset_fnrL)=self.predict_using_a_model(str(i),testset_i,testset_aucL=testset_aucL,testset_tprL=testset_tprL,testset_tnrL=testset_tnrL,testset_fprL=testset_fprL,testset_fnrL=testset_fnrL)      
            elif target == '?':
                predL=self.predict_using_a_model(str(i),testset_i)
        '''
        if target != '?':
            if self.modeltype == 'rf':
                print('random forest: \n')
                if file != 'none':
                    file.write('random forest: \n')
            elif self.modeltype == 'log_reg':
                print('logistic regression: \n')
                if file != 'none':            
                    file.write('logistic regression: \n')
            print('testset AUC: '+str(testset_aucL[0]))
            if file != 'none':  
                file.write('testset AUC: '+str(testset_aucL[0])+'\n')
                file.write('testset TPR: '+str(testset_tprL[0])+'\n')
                file.write('testset TNR: '+str(testset_tnrL[0])+'\n')
                file.write('testset FPR: '+str(testset_fprL[0])+'\n')
                file.write('testset FNR: '+str(testset_fnrL[0])+'\n') 
            print('testset TPR: '+str(testset_tprL[0]))
            print('testset TNR: '+str(testset_tnrL[0]))
            print('testset FPR: '+str(testset_fprL[0]))
            print('testset FNR: '+str(testset_fnrL[0]))
            performance=(testset_aucL[0],testset_tprL[0],testset_tnrL[0],testset_fprL[0],testset_fnrL[0])
            return performance
        #else: #unknown targets
        #    return predL
    
    def ptb_of_id(self,readings_of_id,filter2type2,filter2_num,filter2_software2,filter2_path2,modeltype2,model_num,model_software2,model_path2):
        ###Select a best reading from all the readings of an id and predict the PTB from the selected reading
        #
        #all EIS readings of a patient id (input) -> filter2 -> a best reading (highest score (output1)) -> PTB classifier -> PTB prediction (output2)
        #unknown targets are represented using '?'
        #csv format of readings_of_id: "U:\EIS preterm prediction\ptb demo\onterm_id_BJ4146.csv":
        #27_EIS_Amplitude1,27_EIS_Amplitude2,27_EIS_Amplitude3,27_EIS_Amplitude4,27_EIS_Amplitude5,27_EIS_Amplitude6,27_EIS_Amplitude7,27_EIS_Amplitude8,27_EIS_Amplitude9,27_EIS_Amplitude10,27_EIS_Amplitude11,27_EIS_Amplitude12,27_EIS_Amplitude13,27_EIS_Amplitude14,27_EIS_Phase1,27_EIS_Phase2,27_EIS_Phase3,27_EIS_Phase4,27_EIS_Phase5,27_EIS_Phase6,27_EIS_Phase7,27_EIS_Phase8,27_EIS_Phase9,27_EIS_Phase10,27_EIS_Phase11,27_EIS_Phase12,27_EIS_Phase13,27_EIS_Phase14,before37weeksCell
        #11.3964,10.7567,10.1114,9.2925,8.212,6.924,5.6166,4.5635,3.8327,3.3183,2.9289,2.551,2.1447,1.7724,-5.2964,-7.729,-10.3953,-14.0677,-18.3854,-22.4323,-24.1443,-23.2932,-21.2737,-20.0566,-20.2443,-22.0213,-23.3156,-28.9139,?
        #45.0189,42.8677,40.0287,35.8297,31.0037,23.9971,17.4818,12.0418,7.9376,5.3243,3.8748,3.0018,2.3639,1.8306,-6.3001,-10.095,-14.7924,-21.0442,-28.4712,-35.6903,-42.0415,-46.5337,-47.6852,-44.4301,-38.7129,-34.5318,-33.085,-31.8822,?
        #12.648,12.3,11.9657,11.467,10.7139,9.4876,8.0939,6.7357,5.5102,4.4928,3.7199,3.0801,2.4869,1.9592,-2.6951,-4.385,-6.4467,-9.5517,-13.5198,-17.7232,-21.126,-23.5089,-25.0861,-25.5042,-26.0601,-26.5101,-27.6583,-29.0257,?
    
        self.set_dataset('readings of an id')
        results_path2='U:\\EIS preterm prediction\\ptb demo\\'
        self.set_results_path(results_path2)
        self.set_filter2(filter2type2,filter2_software2,filter2_path2)
        self.set_model(modeltype2,model_software2,model_path2)
        df=pd.read_csv(readings_of_id)
        (r,c)=df.shape
        cols=list(df.columns)
        cols[len(cols)-1]='selected_reading'#change the targets column name for filter2
        df=pd.DataFrame(df,columns=cols)
        df=df.astype(object)
        for i in range(r):
            df.iat[i,c-1]='?'#set unknown selection for each reading
        df.to_csv(self.results_path+'readings_of_id.csv',index=False)
        print('1. Select a best reading from all the readings of a patient.\n')
        print('readings file: '+readings_of_id+'\n')
        print('filter2: '+filter2_path2+filter2type2+filter2_num+'.model\n')
        best_score=self.select_best_readings_of_patients(modelNumber=str(filter2_num),readings_of_id=self.results_path+'readings_of_id.csv',best_readings_testset_csv=self.results_path+'best_reading_of_id.csv')#use filter2 to select best readings of patients      
        if best_score != 'no filter2':
            if float(best_score) < 0.2:#poor quality reading, recommend to get more readings
                print('Warning: The selected reading has a poor score of < 0.2. The preterm birth prediction using the selected reading could be inaccurate!!')
                print('Recommendation: Input more readings for this patient.\n')
            print('2. Predict preterm birth of the patient using the selected reading.\n')
            print('classifier: '+model_path2+modeltype2+model_num+'.model\n')
            predL=self.predict_ptb_of_testset(i=str(model_num),testset_i=self.results_path+'best_reading_of_id.csv')
            #print('score of selected reading: '+str(best_score))
            pred=predL[0]
            if str(pred[2]) == '1':#predicted class is 1
                y='preterm'
            else:
                y='onterm'
            print('Prediction: probability of preterm='+str(pred[1])+', class='+y+' (threshold 0.5)')
        else:
            sys.exit('no filter2')
            
    def main(self,
        ###predict a test set
        ###  or
        ###select best readings for a testset1_i using a filter2, then, predict PTB of testset1_i using the selected readings
        ### or
        ###Use 2 models to predict PTB of patients:
        #for each patient p in a testset{
        # p -> model1 (e.g. demographics+clinical history features based model)
        # if model1 ouputs 1 
        # then prediction=1
        # else p -> model2 (EIS based model)
        #      if model2 outputs 1 
        #      then prediction=1
        #      else prediction=0      
        #}
        i=0, #ith testset or ith model
        data2='EIS', #type of dataset
        ordinal_encode2=False,
        filter2_path2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filter2_asymp_22wks_filtered_data_28inputs_no_treatment\\",
        filter2type2='rf',
        filter2_software2='weka',
        model_path2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_filtered_data_28inputs_no_treatment_5_to_15features_2\\",
        modeltype2='random forest and log regression',
        model_software2='weka',
        libsvm2=True,
        scaler_file2=None,
        testset_ids_path2="U:\\EIS preterm prediction\\trainsets1trainsets2\\asymp_22wks_filtered_data_28inputs_no_treatment\\",
        allreadings_with_ids2="438_V1_4_eis_readings_28inputs_with_ids.csv",#ids of 438 patients with all EIS readings
        asymp_22wks_438_V1_9inputs_with_ids2="d:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_9inputs_with_ids.csv",
        create_dataset_only=None,#create a dataset of best readings of ids
        best_readings_testset_i2=None,#testset of ith iteration
        best_readings_hash=None,#key=id, value=best reading of id
        testset_i_ids2=None,#ids of testset of ith iteration
        testset_i2=None,#ith testset (used when filtering is not used)
        trainset_i2=None,#trainset of ith iteration (for template match filtering)
        template_filter2_option2=None,#filter1 or filter2 of template filtering
        #results_path2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_filtered_data_28inputs_no_treatment_5_to_15features_2\\",#location of testsets consisting of the best readings
        results_path2=None,
        logfile0_2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_filtered_data_28inputs_no_treatment_5_to_15features_2\\readings_scores_logfile_new11.txt", #scores of the best readings selected by filter2s
        logfile2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\asymp_22wks_filtered_data_28inputs_no_treatment_5_to_15features_2\\logfile_rf11.txt", #results of PTB prediction
        logfile2_option='a',
        weka_path2="c:\\Program Files\\Weka-3-9-4\\weka.jar",
        java_memory2='4g',
        X_scaler2=None, #file containing a inputs scaler for keras neural networks
        final_prob=None
        ):
        if data2!=None:
            self.set_dataset(data2)
        if filter2type2 == 'template match':
            file=open(self.logfile,logfile2_option)
            file.write('template match\n')
            if trainset_i2!=None:#use template match filtering to select best readings
                self.set_trainset_i(trainset_i2)
            else:
                sys.exit('trainset_i2 is not set for template match filtering')
            if template_filter2_option2!=None:
                self.set_template_filter2_option(template_filter2_option2)
                print('template filtering: ',template_filter2_option2)
                file.write('template filtering: '+template_filter2_option2+'\n')
                file.close()
            else:
                sys.exit('template_filter2_option2 is not set for template match filtering')
        self.set_filter2(filter2type2,filter2_software2,filter2_path2)
        self.set_testset_ids_path(testset_ids_path2)
        self.set_all_readings_with_ids_file(allreadings_with_ids2)
        self.set_metabolite_data_with_ids_file(asymp_22wks_438_V1_9inputs_with_ids2)
        self.set_results_path(results_path2)
        print('results_path: ',self.results_path)
        self.set_logfile(logfile2)
        self.set_weka_path(weka_path2)
        self.set_java_memory(java_memory2)
        self.set_ordinal_encode(ordinal_encode2)
        self.set_scaler_file(scaler_file2)
        
        if X_scaler2!=None:
            self.set_scaler(X_scaler2)
        file=open(self.logfile,logfile2_option)      
        if create_dataset_only:#select a best reading for each id of ith testset using the ith filter
            self.select_best_readings_and_create_a_testset(i=i,file=file,best_readings_testset_i=best_readings_testset_i2,testset_i_ids=testset_i_ids2)
            file.close()
        elif filter2_software2==None:#predict the testset without using filtering which selects a best reading for each id 
            if modeltype2 == 'polynomial kernel svm and rbf kernel svm':
                if libsvm2:
                    self.set_model('poly_libsvm','weka',model_path2)
                else:
                    self.set_model('poly_svm','weka',model_path2)
                poly_svm_performance=self.predict_ptb_of_testset(i=i,file=file,testset_i=testset_i2)
                self.set_model('rbf_svm','weka',model_path2)
                rbf_svm_performance=self.predict_ptb_of_testset(i=i,file=file,testset_i=testset_i2)
                file.close()
                return (poly_svm_performance,rbf_svm_performance)        
            elif modeltype2 == 'random forest and log regression':
                self.set_model('log_reg','weka',model_path2)
                log_reg_performance=self.predict_ptb_of_testset(i=i,file=file,testset_i=testset_i2)#predict using log regression
                self.set_model('rf','weka',model_path2)
                rf_performance=self.predict_ptb_of_testset(i=i,file=file,testset_i=testset_i2)#predict using random forest
                file.close()
                return (rf_performance,log_reg_performance)        
            elif modeltype2 == 'random forest':
                self.set_model('rf','weka',model_path2)
                rf_performance=self.predict_ptb_of_testset(i=i,file=file,testset_i=testset_i2)#predict using random forest
                file.close()
                return rf_performance
            elif modeltype2 == 'log regression':
                self.set_model('log_reg','weka',model_path2)
                log_reg_performance=self.predict_ptb_of_testset(i=i,file=file,testset_i=testset_i2)#predict using log regression
                file.close()
                return log_reg_performance
            else:
                sys.exit('invalid modeltype: '+modeltype2)
        '''
        elif filter2type2 == 'template match' or filter2_software2!=None:#use a filter to select a best reading for each id, then, predict the selected readings using a PTB predictor
            if best_readings_testset_i2!=None and testset_i_ids2!=None:
                self.select_best_readings_and_create_a_testset(i=i,file=file,best_readings_testset_i=best_readings_testset_i2,testset_i_ids=testset_i_ids2)        
            else:
                self.select_best_readings_and_create_a_testset(i=i,file=file)
            #predict the PTB of these patients using the ith PTB model
            if modeltype2 == 'random forest and log regression':
                self.set_model('log_reg','weka',model_path2)
                log_reg_performance=self.predict_ptb_of_testset(i=i,file=file)#predict using log regression
                self.set_model('rf','weka',model_path2)
                rf_performance=self.predict_ptb_of_testset(i=i,file=file)#predict using random forest
                file.close()
                return (rf_performance,log_reg_performance)
            elif modeltype2 == 'random forest':
                self.set_model('rf','weka',model_path2)
                rf_performance=self.predict_ptb_of_testset(i=i,file=file)#predict using random forest
                file.close()
                return rf_performance
            elif modeltype2 == 'log regression':
                self.set_model('log_reg','weka',model_path2)
                log_reg_performance=self.predict_ptb_of_testset(i=i,file=file)#predict using log regression
                file.close()
                return log_reg_performance
            else:
                sys.exit('invalid modeltype: '+modeltype2)
        '''
        
if __name__ == "__main__":
    ###Use 100 filter2 to select best readings for 100 testsets (without ids); then, use 100 ptb classifiers to predict PTB from the testsets
    #
    java_memory='13g'  
    modeltype='log_reg'
    no_of_models=100    
    set_dataset('EIS')    
    #model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_no_treatment\\" #path of PTB classifiers
    model_path="h:\\data\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_no_treatment\\" #path of PTB classifiers
    filter2type2='rf'
    filter2_software2='weka'
    #filter2_path2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filter2 from sharc\\selected_unselected_eis_readings_no_treatment\\" #path of filter2
    filter2_path2="h:\\data\\EIS preterm prediction\\results\\workflow1\\filter2 from sharc\\selected_unselected_eis_readings_no_treatment\\" #path of filter2
    set_filter2(filter2type2,filter2_software2,filter2_path2)
    self.set_model(modeltype,'weka',model_path)
    #results_path2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_no_treatment\\validate filter2\\"#location of testsets consisting of the best readings
    results_path2="h:\\data\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_no_treatment\\validate filter2\\"#location of testsets consisting of the best readings
    set_results_path(results_path2)
    allreadings_with_ids="d:\\EIS preterm prediction\\438_V1_4_eis_readings_28inputs_with_ids.csv"#ids of 438 patients with all EIS readings
    set_all_readings_with_ids_file(allreadings_with_ids)
    #asymp_22wks_438_V1_9inputs_with_ids="U:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_9inputs_with_ids.csv"
    #set_metabolite_data_with_ids_file(asymp_22wks_438_V1_9inputs_with_ids)
    #logfile2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_no_treatment\\validate filter2\\logfile.txt" #results of PTB prediction
    logfile2=results_path2+"logfile_log_reg.txt" #results of PTB prediction
    utilities.create_folder_if_not_exist(results_path2)
    file=open(logfile2,'w')
    #file=open(logfile2,'a')
    print('===Select readings for 100 testsets===\n')
    select_best_readings_and_create_testsets(file)
    print('===Predict ptb of 100 testsets using the selected readings===\n')
    predict_ptb_of_testsets(file)
    file.close()
