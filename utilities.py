import pandas as pd
import arff
import re
import os,sys
from sklearn.metrics import roc_auc_score, confusion_matrix
from preprocess import join_data, remove_duplicates, fill_missing_values, reduce_data2, poly_features2, convert_to_windows_path, dataframes_diff, split_train_test_sets
import subprocess
from shutil import copyfile
import numpy as np
import operator
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
import random
import SelectData as sd
from joblib import Parallel, delayed, load
import ModelsPredict as mp

def divide_features_by_frequency_squared(data_csv,outdata_csv):
    #input: dataset_csv (with or without ids column)
    
    freqL=[21, #16 frequencies of MIS data
           42,
           58,
           72,
           86,
           100,
           202,
           302,
           402,
           502,
           604,
           704,
           804,
           904,
           1013,
           1013]
    '''
    freqL=[402,#highest 8 frequencies of MIS data
           502,
           604,
           704,
           804,
           904,
           1013,
           1013]
    '''
    df=pd.read_csv(data_csv)
    (_,c)=df.shape
    if df.columns[0] == 'Identifier' or df.columns[0] == 'ID' or df.columns[0] == 'hospital_id':    
        data=df.iloc[:,1:c-1] #skip ids column and class column
        for j in range(len(freqL)):
            data.iloc[:,j]=data.iloc[:,j] / freqL[j]**2
            data.iloc[:,j+len(freqL)]=data.iloc[:,j+len(freqL)] / freqL[j]**2            
        df.iloc[:,1:c-1]=data
    else:#without ids column
        data=df.iloc[:,0:c-1] #skip the class column
        for j in range(len(freqL)):
            data.iloc[:,j]=data.iloc[:,j] / freqL[j]**2
            data.iloc[:,j+len(freqL)]=data.iloc[:,j+len(freqL)] / freqL[j]**2
        df.iloc[:,0:c-1]=data
    df.to_csv(outdata_csv,index=False) 
    
def mean_of_spectra_of_each_id(eis_readings_csv,mean_of_eis_readings_csv,data='eis'):
    #read all spectra of each id into a hashtable
    df=pd.read_csv(eis_readings_csv)
    hash_all_spectra={}#key=id, value=[[spectrum1,spectrum2,...],class]
    (r,c)=df.shape
    for i in range(r):
        pid=df.iat[i,0]
        spectrum=df.iloc[i,1:c-1]
        ptb_label=df.iloc[i,c-1]
        if hash_all_spectra.get(pid)==None:
            hash_all_spectra[pid]=[[spectrum],ptb_label]
        else:
            l=hash_all_spectra[pid]
            spectrumL=l[0]
            spectrumL.append(spectrum)
            hash_all_spectra[pid]=[spectrumL,ptb_label]
    #print(hash_all_spectra)
    f=open(mean_of_eis_readings_csv,'w')
    #cols=list(df.columns)
    if data=='eis':#EIS spectra have 14 frequencies
        f.write('hospital_id,Meanreal3_76,Meanreal3_153,Meanreal3_305,Meanreal3_610,Meanreal3_1221,Meanreal3_2441,Meanreal3_4883,Meanreal3_9766,Meanreal3_19531,Meanreal3_39063,Meanreal3_78125,Meanreal3_156250,Meanreal3_312500,Meanreal3_625000,Meanimg3_76,Meanimg3_153,Meanimg3_305,Meanimg3_610,Meanimg3_1221,Meanimg3_2441,Meanimg3_4883,Meanimg3_9766,Meanimg3_19531,Meanimg3_39063,Meanimg3_78125,Meanimg3_156250,Meanimg3_312500,Meanimg3_625000,before37weeks\n')
    elif data=='mis':#MIS spectra have 16 frequencies
        f.write('Identifier,MeanReal1,MeanReal2,MeanReal3,MeanReal4,MeanReal5,MeanReal6,MeanReal7,MeanReal8,MeanReal9,MeanReal10,MeanReal11,MeanReal12,MeanReal13,MeanReal14,MeanReal15,MeanReal16,MeanIm1,MeanIm2,MeanIm3,MeanIm4,MeanIm5,MeanIm6,MeanIm7,MeanIm8,MeanIm9,MeanIm10,MeanIm11,MeanIm12,MeanIm13,MeanIm14,MeanIm15,MeanIm16,PTB\n')
    else:
        sys.exit('invalid option in function mean_of_spectra_of_each_id: ',data)
        #m=1
    #while m < range(len(cols)-1):
    #    f.write('Mean'+cols[m]+',')
    #f.write(cols[-1]+'\n')
    #compute mean of all spectra of each id and write to a csv file
    for pid,spectraL in hash_all_spectra.items():
        spectra=spectraL[0]
        ptb=spectraL[-1]
        spectra_sum=spectra[0]
        #print('spectrum 0: '+str(spectra_sum))
        i=1
        while i < len(spectra):
            #print('spectrum '+str(i)+': '+str(spectra[i]))           
            spectrum=spectra[i]
            spectra_sum+=spectrum
            i+=1
        #print('sum of spectra of id: '+str(spectra_sum))
        mean=spectra_sum/len(spectra)
        #print('mean: '+str(mean))
        l=mean.size
        f.write(str(pid)+',')
        for k in range(l):
            f.write(str(mean[k])+',')
        f.write(str(ptb)+'\n')
    f.close()
    
def change_ids_to_dummy_ids(eis_readings_csv,eis_readings_with_dummy_ids):
    #change hospital_ids to dummy ids e.g. 1, 2, 3, ..., 438
    df=pd.read_csv(eis_readings_csv)
    ids=df.iloc[:,0]
    ids2=list(set(ids))#remove duplicate ids
    hash_ids={} #key=hospital id, value=dummy id
    j=1
    #j=366 #dummy id starts from j
    for id2 in ids2:
        if hash_ids.get(id2)==None:
            hash_ids[id2]=j
            j=j+1
    (r,_)=df.shape
    for i in range(r):
        df.iat[i,0]=hash_ids.get(df.iat[i,0])
    #df.columns=['hospital_id','real3_76','real3_153','real3_305','real3_610','real3_1221','real3_2441','real3_4883','real3_9766','real3_19531','real3_39063','real3_78125','real3_156250','real3_312500','real3_625000','img3_76','img3_153','img3_305','img3_610','img3_1221','img3_2441','img3_4883','img3_9766','img3_19531','img3_39063','img3_78125','img3_156250','img3_312500','img3_625000','before37weeks']
    #df.columns=['hospital_id','Amp1','Amp2','Amp3','Amp4','Amp5','Amp6','Amp7','Amp8','Amp9','Amp10','Amp11','Amp12','Amp13','Amp14','Phase1','Phase2','Phase3','Phase4','Phase5','Phase6','Phase7','Phase8','Phase9','Phase10','Phase11','Phase12','Phase13','Phase14','before37weeks']
    df.columns=['hospital_id','have_PTB_history','Amp1','Amp2','Amp3','Amp4','Amp5','Amp6','Amp7','Amp8','Amp9','Amp10','Amp11','Amp12','Amp13','Amp14','Phase1','Phase2','Phase3','Phase4','Phase5','Phase6','Phase7','Phase8','Phase9','Phase10','Phase11','Phase12','Phase13','Phase14','before37weeks']
    df.to_csv(eis_readings_with_dummy_ids,index=False)
      
def merge_features(metabolite_data_csv,new_data_csv,merge_operation='sum'):
    df=pd.read_csv(metabolite_data_csv)
    if merge_operation=='sum':
        #new feature: lactate+glx+formate+BCAA+alanine (the features correlated with Alanine (the most relevant feature) are merged with Alanine)
        new_feature=pd.DataFrame(data=df['ALANINE']+df['LACTATE']+df['GLX']+df['FORMATE']+df['BCAA'],columns=['ALANINE+LACTATE+GLX+FORMATE+BCAA'])#,index=df.index)
        #new feature: acetate+succinate (these features are correlated with each other and independent with the other features)
        new_feature2=pd.DataFrame(data=df['ACETATE']+df['SUCCINATE'],columns=['ACETATE+SUCCINATE'])#,index=df.index)
        df=df.drop(columns=['ALANINE','LACTATE','GLX','FORMATE','BCAA','ACETATE','SUCCINATE']) 
    elif merge_operation=='product':
        #new feature: lactate*glx*formate*BCAA*alanine (the features correlated with Alanine (the most relevant feature) are merged with Alanine)
        new_feature=pd.DataFrame(data=df['ALANINE']*df['LACTATE']*df['GLX']*df['FORMATE']*df['BCAA'],columns=['ALANINE*LACTATE*GLX*FORMATE*BCAA'])#,index=df.index)
        #new feature: acetate+succinate (these features are correlated with each other and independent with the other features)
        new_feature2=pd.DataFrame(data=df['ACETATE']*df['SUCCINATE'],columns=['ACETATE*SUCCINATE'])#,index=df.index)
        df=df.drop(columns=['ALANINE','LACTATE','GLX','FORMATE','BCAA','ACETATE','SUCCINATE'])        
    else:
        sys.exit('invalid merge_operation: '+merge_operation)
    df2=new_feature.join([new_feature2,df])
    df2.to_csv(new_data_csv,index=False)
    
def log10_transform(dataset_csv,log10dataset_csv):
    #input: dataset_csv (with or without ids column)
    df=pd.read_csv(dataset_csv)
    (_,c)=df.shape
    if df.columns[0] == 'Identifier' or df.columns[0] == 'ID' or df.columns[0] == 'hospital_id':    
        data=df.iloc[:,1:c-1] #skip ids column and class column
        data=np.log10(data)
        df.iloc[:,1:c-1]=data
    else:#without ids column
        data=df.iloc[:,0:c-1] #skip the class column
        data=np.log10(data)
        df.iloc[:,0:c-1]=data
    df.to_csv(log10dataset_csv,index=False)

def log_transform(dataset_csv,logdataset_csv):
    #compute the natural logarithm (ln(x)) of features of a dataset
    #input: dataset_csv (with or without ids column)
    df=pd.read_csv(dataset_csv)
    (_,c)=df.shape
    if df.columns[0] == 'Identifier' or df.columns[0] == 'ID' or df.columns[0] == 'hospital_id':    
        data=df.iloc[:,1:c-1] #skip ids column and class column
        data=np.log(data)
        df.iloc[:,1:c-1]=data
    else:#without ids column
        data=df.iloc[:,0:c-1] #skip the class column
        data=np.log(data)
        df.iloc[:,0:c-1]=data
    df.to_csv(logdataset_csv,index=False)

def multiply_features_by_number_transform(dataset,dataset2_csv,number,datatype='csv'):
    #multiply each feature by a number e.g. 100 to enlarge or reduce the features values
    #input: dataset_csv (with ids column)
    if datatype=='csv':
        df=pd.read_csv(dataset)
    elif datatype=='df':
        df=dataset
    (_,c)=df.shape
    if df.columns[0] == 'Identifier' or df.columns[0] == 'ID' or df.columns[0] == 'hospital_id':    
        data=df.iloc[:,1:c-1] #skip ids column and class column
        data=data*number
        df.iloc[:,1:c-1]=data
    else:
        data=df.iloc[:,0:c-1] #skip the class column
        data=data*number
        df.iloc[:,0:c-1]=data
    df.to_csv(dataset2_csv,index=False)
    
def remove_ids_column_from_dataset(dataset,results_path):
    data=pd.read_csv(dataset)
    (_,c)=data.shape
    if data.columns[0] == 'Identifier' or data.columns[0] == 'hospital_id':
        data=data.iloc[:,1:c] #skip ids column
        data.to_csv(results_path+'data.csv',index=False)
        dataset=results_path+'data.csv'
        print('The dataset without ids column is saved as: ',dataset)
    else:
        print('The dataset has no ids column and unchanged.')
    return dataset

def get_no_of_inputs_of_models(results_path,classifier='rf'):
    #get the no. of inputs of each model from each model_inputs_output file
    #classifier=='rf', 'log_reg' or 'rf_and_log_reg'
    if classifier=='log_reg' or classifier=='rf_and_log_reg':
        inputs_log_regL=[]
        for i in range(100):
            if os.path.isfile(results_path+'log_reg'+str(i)+'.model_inputs_output.csv'):
                f=open(results_path+'log_reg'+str(i)+'.model_inputs_output.csv','r')
                l=f.readline()
                fs=l.split(',')
                inputs=len(fs)-1
                inputs_log_regL.append(inputs)
                f.close()
                print('log reg',i,': inputs=',inputs)
        min_inputs=np.min(inputs_log_regL)
        max_inputs=np.max(inputs_log_regL)
        inputs_log_regL.sort()
        print('sorted list of inputs: ',inputs_log_regL)
        print('no. of inputs of logistic regression models are between :',min_inputs,' and ',max_inputs)
    if classifier=='rf' or classifier=='rf_and_log_reg':
        inputs_rfL=[]
        for i in range(100):
            if os.path.isfile(results_path+'rf'+str(i)+'.model_inputs_output.csv'):
                f=open(results_path+'rf'+str(i)+'.model_inputs_output.csv','r')
                l=f.readline()
                fs=l.split(',')
                inputs=len(fs)-1
                inputs_rfL.append(inputs)
                f.close()
                print('random forest',i,': inputs=',inputs)
        min_inputs2=np.min(inputs_rfL)
        max_inputs2=np.max(inputs_rfL)
        inputs_rfL.sort()
        print('sorted list of inputs: ',inputs_rfL)
        print('no. of inputs of random forest models are between :',min_inputs2,' and ',max_inputs2)
            
def features_stats(data_csv,normalize=False):
    #compute the mean, min, max and variance of each feature of a dataset
    df=pd.read_csv(data_csv)
    (_,c)=df.shape
    cols=list(df.columns)
    if cols[0] == 'Identifier' or cols[0] == 'hospital_id' or cols[0] == 'id' or cols[0] == 'Id' or cols[0] == 'ID':
       (_,c)=df.shape
       df=df.iloc[:,1:c] #removed ids column from whole dataset
    print(data_csv)
    (_,c)=df.shape
    print('no. of features: ',c-1)
    if normalize:#normalize features before computing stats
        (df,_,_)=zscore_normalize_inputs(df.iloc[:,0:c-1],df.iloc[:,0:c-1])
        df=pd.DataFrame(data=df,columns=cols[0:c-1])
        print('===compute stats of zscore-normalized features===')
    statsL=[]
    for i in range(c-1):
        f=df.iloc[:,i]
        mean=f.mean()
        mini=f.min()
        maxi=f.max()
        variance=f.var()
        print('feature ',i,', mean: ',np.round(mean,3),', min: ',np.round(mini,3),', max:',np.round(maxi,3),', variance: ',np.round(variance,3))
        statsL.append((i,mean,mini,maxi,variance))
    statsL.sort(key=operator.itemgetter(4),reverse=True)#sort features in descending order of variance
    print('===Ranking of features in descending order of variance===')
    for i in range(c-1):
        f=statsL[i]
        print('feature: ',f[0],', mean: ',f[1],', min: ',f[2],', max: ',f[3],', variance: ',f[4])
    
def get_number_of_inputs_of_models(resultspath,model='log_reg'):
    inputsL=[]
    for i in range(100):
        model_inputs_output=resultspath+model+str(i)+".model_inputs_output.csv"
        if os.path.isfile(model_inputs_output):
            f=open(model_inputs_output,'r')
            line=f.readline()
            L=line.split(',')
            inputs=len(L)-1
            inputsL.append(inputs)
    print('model: '+resultspath+model+', inputs: '+str(np.min(inputsL))+' to '+str(np.max(inputsL)))
    
def get_ids_of_patients():
    s = sd.SelectData()
    for i in [0]:
    #for i in range(100):
        csvfile="H:\\data\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\testset"+str(i)+'.csv'    
        idsfile="d:\\EIS preterm prediction\\438_V1_4_eis_readings_28inputs_with_ids.csv"
        idsL=s.get_patients_ids2(csvfile,idsfile)
        outfile="H:\\data\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\testset"+str(i)+'_ids.csv'
        df=pd.DataFrame(idsL,columns=['hospital_ids'])
        df.to_csv(outfile,index=False)
        
def split_train_testsets(dataset,testset_size,iterations,results_path,seeds_file=None):
    ###split data into training sets and test sets
    create_folder_if_not_exist(results_path)
    f=open(results_path+'seeds.txt','w')
    #seedsL=[i for i in range(iterations)]
    if seeds_file!=None:
        seedsL = [line.strip() for line in open(seeds_file)]
    i=0
    for iteration in range(iterations):
        if seeds_file!=None:
            seed0 = int(seedsL[i])
            i+=1
        else:
            #seed0=seedsL[iteration]
            #seed0=random.randint(0,5**9) #ith seed for data split 
            #seed0=random.randint(0,999999) #ith seed for data split
            seed0=random.randint(0,2**32-1)        
        print('seed of train/test split: '+str(seed0))
        f.write(str(seed0)+'\n')
        data=pd.read_csv(dataset)
        cols=list(data.columns)
        (_,c)=data.shape
        (train_set,test_set)=split_train_test_sets(data,testset_size,seed0,cols[c-1])
        train_set.to_csv(results_path+'trainset'+str(iteration)+'.csv',index=False)
        test_set.to_csv(results_path+'testset'+str(iteration)+'.csv',index=False)
    f.close()
    
def split_train_valid_testsets_by_ids(filtered_eis_readings_with_ids=None,#filtered EIS readings with ids and class variable: before37weeks
                                      selected_unselected_eis_readings_with_ids=None,#all EIS readings with ids and class variable: selected_reading
                                      results_path=None,
                                      trainset_size=0.66,
                                      validset_size=0,
                                      testset_size=0.34,
                                      iterations=100,
                                      trainsets1_only=False,
                                      trainsets2_only=False,
                                      trainsets1_and_trainsets2=False):
    #Split filtered_eis_readings_with_ids and all_eis_readings_with_ids into training sets, validation sets (if validset_size > 0) and testsets by ids
    #output: trainsets1_i (ith training set of filtered_eis_readings)
    #        trainsets2_i
    #        testsets1_i
    #        testsets2_i
    #        validsets1_i (if validset_size > 0)
    #        validsets2_i (if validset_size > 0)
    #        trainsets1_ids_i
    #        testsets1_ids_i
    #        validset1_ids_i
    #        trainsets2_ids_i
    #        testsets2_ids_i
    #        validset2_ids_i
    #        seeds.txt (random seeds of data splits)
    #        trainsets1_indx.txt
    #        trainsets2_indx.txt
    create_folder_if_not_exist(results_path)    
    seedsfile=results_path+"seeds.txt"
    if trainsets1_and_trainsets2 or trainsets2_only:
        trainsets1_indx_file=results_path+"trainsets1_indx.txt" #rows indices of training instances in the filtered_eis_readings_with_ids
        trainsets2_indx_file=results_path+"trainsets2_indx.txt" #rows indices of training instances in the data2    
    elif trainsets1_only:
        trainsets1_indx_file=results_path+"trainsets1_indx.txt" #rows indices of training instances in the filtered_eis_readings_with_ids
    elif trainsets2_only:
        trainsets2_indx_file=results_path+"trainsets2_indx.txt" #rows indices of training instances in the data2
    df1=pd.read_csv(filtered_eis_readings_with_ids,low_memory=False)#target variable: before37weeks
    (_,c)=df1.shape
    cols1=list(df1.columns)
    file=open(seedsfile,'w')
    file.write('trainset size: '+str(trainset_size)+'\n')
    file.write('validation set size: '+str(validset_size)+'\n')
    file.write('testset size: '+str(testset_size)+'\n')
    if trainsets1_only:
        file2=open(trainsets1_indx_file,'w')
    elif trainsets2_only:
        file3=open(trainsets2_indx_file,'w')
        df2=pd.read_csv(selected_unselected_eis_readings_with_ids,low_memory=False)#target variable: selected_reading
        (_,c2)=df2.shape
        cols2=list(df2.columns)        
    elif trainsets1_and_trainsets2:
        file2=open(trainsets1_indx_file,'w')
        df2=pd.read_csv(selected_unselected_eis_readings_with_ids,low_memory=False)#target variable: selected_reading
        (_,c2)=df2.shape
        cols2=list(df2.columns)
        file3=open(trainsets2_indx_file,'w')
    if trainset_size+testset_size+validset_size != 1:
        sys.exit('trainset_size + testset_size + validset_size must be 1.')
    elif validset_size > 0:
            testset_validset_size=testset_size+validset_size    
            testset_size2=testset_size/(testset_size+validset_size)    
    for i in range(iterations):             
        print(i)
        seed=random.randint(0,5**9) #ith seed for data split 
        #seed=random.randint(0,999999)
        print('seed of data split at iteration '+str(i)+': '+str(seed))
        file.write('seed of data split at iteration '+str(i)+': '+str(seed)+'\n')
        if validset_size > 0:
            (trainset,validset_testset)=split_train_test_sets(df1,testset_validset_size,seed,cols1[c-1])
            validset_testset.to_csv(results_path+'validset_testset.csv',index=False)
            validset_testset=pd.read_csv(results_path+'validset_testset.csv')
            (validset,testset)=split_train_test_sets(validset_testset,testset_size2,seed,cols1[c-1])
            if trainsets1_only:
                trainset_indx=list(trainset.index)
                for indx in trainset_indx[0:len(trainset_indx)-1]:
                    file2.write(str(indx)+',')
                file2.write(str(trainset_indx[len(trainset_indx)-1])+'\n')
                trainset1=trainset
                validset1=validset
                testset1=testset
                trainset1.to_csv(results_path+'trainset1_'+str(i)+'.csv',index=False)
                validset1.to_csv(results_path+'validset1_'+str(i)+'.csv',index=False)
                testset1.to_csv(results_path+'testset1_'+str(i)+'.csv',index=False)
                ids=trainset[cols1[0]]
                idsL=list(ids)
                ids=pd.DataFrame(ids,columns=[cols1[0]])#convert a series (a column) to a dataframe
                ids.to_csv(results_path+'trainset1_ids_'+str(i)+'.csv',index=False)            
                print('trainset: ',len(idsL))
            elif trainsets2_only:
                ids=trainset[cols1[0]]
                idsL=list(ids)
                ids=pd.DataFrame(ids,columns=[cols1[0]])#convert a series (a column) to a dataframe
                ids.to_csv(results_path+'trainset2_ids_'+str(i)+'.csv',index=False)            
                print('trainset: ',len(idsL))
                ###get all the readings of each id of trainset1 and save them as trainset2                            
                trainset2=df2.loc[lambda arg: np.isin(df2[cols2[0]],idsL)]
                trainset2_indx=list(trainset2.index)
                for indx in trainset2_indx[0:len(trainset2_indx)-1]:
                    file3.write(str(indx)+',')
                file3.write(str(trainset2_indx[len(trainset2_indx)-1])+'\n')
                trainset2=trainset2.iloc[:,1:c2]#skip the first column of ids
                trainset2.to_csv(results_path+'trainset2_'+str(i)+'.csv',index=False)
                ids=validset[cols1[0]]
                idsL=list(ids)
                print('validset: ',len(idsL))
                ids=pd.DataFrame(ids,columns=[cols1[0]])
                ids.to_csv(results_path+'validset2_ids_'+str(i)+'.csv',index=False)
                ###get all the readings of each id of validset1 and save them as validset2
                validset2=df2.loc[lambda arg: np.isin(df2[cols2[0]],idsL)]
                validset2=validset2.iloc[:,1:c2]#skip the first column of ids
                validset2.to_csv(results_path+'validset2_'+str(i)+'.csv',index=False)
                ###get all the readings of each id of testset1 and save them as testset2
                ids=testset[cols1[0]]
                idsL=list(ids)
                print('testset: ',len(idsL))
                ids=pd.DataFrame(ids,columns=[cols1[0]])
                ids.to_csv(results_path+'testset2_ids_'+str(i)+'.csv',index=False)
                testset2=df2.loc[lambda arg: np.isin(df2[cols2[0]],idsL)]
                testset2=testset2.iloc[:,1:c2]#skip the first column of ids
                testset2.to_csv(results_path+'testset2_'+str(i)+'.csv',index=False)
            elif trainsets1_and_trainsets2:
                trainset1=trainset.iloc[:,1:c]#skip the first column of ids
                validset1=validset.iloc[:,1:c]#skip the first column of ids
                testset1=testset.iloc[:,1:c]#skip the first column of ids
                trainset1.to_csv(results_path+'trainset1_'+str(i)+'.csv',index=False)
                validset1.to_csv(results_path+'validset1_'+str(i)+'.csv',index=False)
                testset1.to_csv(results_path+'testset1_'+str(i)+'.csv',index=False)
                ids=trainset[cols1[0]]
                idsL=list(ids)
                ids=pd.DataFrame(ids,columns=[cols1[0]])#convert a series (a column) to a dataframe
                ids.to_csv(results_path+'trainset1_ids_'+str(i)+'.csv',index=False)            
                print('trainset: ',len(idsL))
                ###get all the readings of each id of trainset1 and save them as trainset2                            
                trainset2=df2.loc[lambda arg: np.isin(df2[cols2[0]],idsL)]
                trainset2_indx=list(trainset2.index)
                for indx in trainset2_indx[0:len(trainset2_indx)-1]:
                    file3.write(str(indx)+',')
                file3.write(str(trainset2_indx[len(trainset2_indx)-1])+'\n')
                trainset2=trainset2.iloc[:,1:c2]#skip the first column of ids
                trainset2.to_csv(results_path+'trainset2_'+str(i)+'.csv',index=False)
                ids=validset[cols1[0]]
                idsL=list(ids)
                print('validset: ',len(idsL))
                ids=pd.DataFrame(ids,columns=[cols1[0]])
                ids.to_csv(results_path+'validset1_ids_'+str(i)+'.csv',index=False)
                ###get all the readings of each id of validset1 and save them as validset2
                validset2=df2.loc[lambda arg: np.isin(df2[cols2[0]],idsL)]
                validset2=validset2.iloc[:,1:c2]#skip the first column of ids
                validset2.to_csv(results_path+'validset2_'+str(i)+'.csv',index=False)
                ###get all the readings of each id of testset1 and save them as testset2
                ids=testset[cols1[0]]
                idsL=list(ids)
                print('testset: ',len(idsL))
                ids=pd.DataFrame(ids,columns=[cols1[0]])
                ids.to_csv(results_path+'testset1_ids_'+str(i)+'.csv',index=False)
                testset2=df2.loc[lambda arg: np.isin(df2[cols2[0]],idsL)]
                testset2=testset2.iloc[:,1:c2]#skip the first column of ids
                testset2.to_csv(results_path+'testset2_'+str(i)+'.csv',index=False)            
        elif validset_size == 0:
            print('validset size == 0')
            (trainset,testset)=split_train_test_sets(df1,testset_size,seed,cols1[c-1])
            trainset_indx=list(trainset.index)
            for indx in trainset_indx[0:len(trainset_indx)-1]:
                file2.write(str(indx)+',')
            file2.write(str(trainset_indx[len(trainset_indx)-1])+'\n')
            if trainsets1_and_trainsets2:#create trainset1 and transet2
                trainset1=trainset.iloc[:,1:c]#skip the first column of ids
                testset1=testset.iloc[:,1:c]#skip the first column of ids
            else:
                trainset1=trainset
                testset1=testset
            trainset1.to_csv(results_path+'trainset1_'+str(i)+'.csv',index=False)
            testset1.to_csv(results_path+'testset1_'+str(i)+'.csv',index=False)        
            if trainsets1_and_trainsets2:
                ###get all the readings of each id of trainset1 and save them as trainset2                            
                ids=trainset[cols1[0]]
                idsL=list(ids)
                print('trainset: ',len(idsL))
                ids=pd.DataFrame(ids,columns=[cols1[0]])
                ids.to_csv(results_path+'trainset1_ids_'+str(i)+'.csv',index=False)            
                trainset2=df2.loc[lambda arg: np.isin(df2[cols2[0]],idsL)]
                trainset2_indx=list(trainset2.index)
                for indx in trainset2_indx[0:len(trainset2_indx)-1]:
                    file3.write(str(indx)+',')
                file3.write(str(trainset2_indx[len(trainset2_indx)-1])+'\n')
                trainset2=trainset2.iloc[:,1:c2]#skip the first column of ids
                trainset2.to_csv(results_path+'trainset2_'+str(i)+'.csv',index=False)
                ###get all the readings of each id of testset1 and save them as testset2
                ids=testset[cols1[0]]
                idsL=list(ids)
                print('testset: ',len(idsL))            
                ids=pd.DataFrame(ids,columns=[cols1[0]])
                ids.to_csv(results_path+'testset1_ids_'+str(i)+'.csv',index=False)
                testset2=df2.loc[lambda arg: np.isin(df2[cols2[0]],idsL)]
                testset2=testset2.iloc[:,1:c2]#skip the first column of ids
                testset2.to_csv(results_path+'testset2_'+str(i)+'.csv',index=False)    
    file.close()
    if trainsets1_only:
        file2.close()
    elif trainsets2_only:
        file3.close()
    elif trainsets1_and_trainsets2:
        file2.close()
        file3.close()
    
def merge_parous_features(demo_obst_csv,merged_csv):
    #Merge 'Parous with 1 preterm delivery' with 'Parous with > 1 preterm delivery' to a feature 'Parous with 1 or more preterm delivery'
    #Merge 'Parous with 1 term delivery' with 'Parous with > 1 term delivery' to a feature 'Parous with 1 or more term delivery'
    df=pd.read_csv(demo_obst_csv)
    df['parous_with_1_or_more_preterm_delivery']=np.logical_or(df['parous_with_1_preterm_delivery']==1,df['parous_with_2_or_more_preterm_delivery']==1)
    df['parous_with_1_or_more_term_delivery']=np.logical_or(df['parous_with_1_term_delivery']==1,df['parous_with_2_or_more_term_delivery']==1)
    df['parous_with_1_or_more_preterm_delivery']=df['parous_with_1_or_more_preterm_delivery'].replace(True,1)
    df['parous_with_1_or_more_preterm_delivery']=df['parous_with_1_or_more_preterm_delivery'].replace(False,0)
    df['parous_with_1_or_more_term_delivery']=df['parous_with_1_or_more_term_delivery'].replace(True,1)
    df['parous_with_1_or_more_term_delivery']=df['parous_with_1_or_more_term_delivery'].replace(False,0)
    df=df.drop(columns=['parous_with_1_preterm_delivery','parous_with_2_or_more_preterm_delivery','parous_with_1_term_delivery','parous_with_2_or_more_term_delivery'])
    df.to_csv(merged_csv,index=False)

def merge_parous_features_into_boolean_feature_and_merge_nulliparous_features_into_boolean_feature(data_csv,merged_csv=None):
    #merge parous_with_1_or_more_preterm_delivery, parous_with_1_or_more_term_delivery and parous_with_previous_miscarriage into a feature 'parous'
    #merge nulliparous_with_no_pregnancy	nulliparous_with_previous_miscarriage	nulliparous_due_to_other_causes into a feature 'nulliparous'
    df=pd.read_csv(data_csv)
    df['parous']=np.logical_or(np.logical_or(df['parous_with_1_or_more_preterm_delivery']==1,df['parous_with_1_or_more_term_delivery']==1),df['parous_with_previous_miscarriage']==1)
    df['parous']=df['parous'].replace(True,1)
    df['parous']=df['parous'].replace(False,0)    
    df['nulliparous']=np.logical_or(np.logical_or(df['nulliparous_with_no_pregnancy']==1,df['nulliparous_with_previous_miscarriage']==1),df['nulliparous_due_to_other_causes']==1)
    df['nulliparous']=df['nulliparous'].replace(True,1)
    df['nulliparous']=df['nulliparous'].replace(False,0)
    df=df.drop(columns=['parous_with_1_or_more_preterm_delivery', 'parous_with_1_or_more_term_delivery', 'parous_with_previous_miscarriage','nulliparous_with_no_pregnancy','nulliparous_with_previous_miscarriage','nulliparous_due_to_other_causes'])
    if merged_csv!=None:
       df.to_csv(merged_csv,index=False)
    return df

def transform_numeric_obstetric_features_to_boolean_features(df,out_data_csv):
    #No. of preterm births (numeric) -> PTB history
    #No. of term births (numeric) -> term history
    #No. of previous pregnancies (numeric) -> previous_pregnancy
    #No. of previous early miscarriages (numeric) -> previous early miscarriage
    #No. of previous abortions (numeric) -> previous abortion
    #input: df, dataframe
    #output: transformed data    
    df['PTB_history']=np.logical_and(df['no_preterm_birthsCell'] > 0, np.isnan(df['no_preterm_birthsCell'])==False)
    df['term_history']=np.logical_and(df['no_term_birthsCell'] > 0, np.isnan(df['no_term_birthsCell'])==False)    
    df['previous_pregnancy']=np.logical_and(df['Number_of_previous_pregnancies'] > 0, np.isnan(df['Number_of_previous_pregnancies'])==False)    
    df['previous_early_miscarriage']=np.logical_and(df['number_previous_early_miscarriages'] > 0, np.isnan(df['number_previous_early_miscarriages'])==False)    
    df['previous_abortion']=np.logical_and(df['number_previous_TOPs'] > 0, np.isnan(df['number_previous_TOPs'])==False)    
    df['mid-trimester_pregnancy_loss']=np.logical_and(df['no_MTLCell'] > 0, np.isnan(df['no_MTLCell'])==False)
    df['PTB_history']=df['PTB_history'].replace(True,1)   
    df['PTB_history']=df['PTB_history'].replace(False,0)
    df['term_history']=df['term_history'].replace(True,1)   
    df['term_history']=df['term_history'].replace(False,0)
    df['previous_pregnancy']=df['previous_pregnancy'].replace(True,1)   
    df['previous_pregnancy']=df['previous_pregnancy'].replace(False,0)
    df['previous_early_miscarriage']= df['previous_early_miscarriage'].replace(True,1)
    df['previous_early_miscarriage']= df['previous_early_miscarriage'].replace(False,0)
    df['previous_abortion']=df['previous_abortion'].replace(True,1)
    df['previous_abortion']=df['previous_abortion'].replace(False,0)
    df['mid-trimester_pregnancy_loss']=df['mid-trimester_pregnancy_loss'].replace(True,1)
    df['mid-trimester_pregnancy_loss']=df['mid-trimester_pregnancy_loss'].replace(False,0)
    df=df.drop(columns=['no_preterm_birthsCell','no_term_birthsCell','Number_of_previous_pregnancies','number_previous_early_miscarriages','number_previous_TOPs','no_MTLCell'])
    df.to_csv(out_data_csv,index=False)
    
def merge_demographics_and_obstetric_history(demo_csv,races_csv,obst_csv,merged_csv):
    demo_df=pd.read_csv(demo_csv)
    demo_df=demo_df.drop(columns=['Ethnicity'])#delete this duplicate column
    races_df=pd.read_csv(races_csv)
    races_df=races_df.drop(columns=['before37weeksCell'])#delete this duplicate column
    obst_df=pd.read_csv(obst_csv)
    obst_df=obst_df.drop(columns=['before37weeksCell'])#delete this duplicate column
    df1=join_data(demo_df,races_df,'hospital_id','hospital_id',datatype='df')
    df2=join_data(df1,obst_df,'hospital_id','hospital_id',datatype='df')
    df2.to_csv(merged_csv,index=False)
    
def merge_races(data_csv,races_csv):
    #merge the following races into a category: African, Arab, Asian, Black, Caribbea, Mixed, Other (p
    #input: 438_V1_previous_history_and_demographics.csv
    #output: csv file with 1 feature 'Ethnicity'
    data=pd.read_csv(data_csv)
    non_white_df=data[data['Ethnicity']!='White']
    print(non_white_df)
    (r,_)=non_white_df.shape
    non_white_df['Ethnicity']=pd.Series(data=[0 for i in range(r)],index=non_white_df.index,name='Ethnicity')#replace non-white ethnicity with 0
    non_white_df=non_white_df[['hospital_id','Ethnicity','before37weeksCell']]
    white_df=data[data['Ethnicity']=='White']
    white_df=white_df[['hospital_id','Ethnicity','before37weeksCell']]
    (r,_)=white_df.shape
    white_df['Ethnicity']=pd.Series(data=[1 for i in range(r)],index=white_df.index,name='Ethnicity')#replace White ethnicity with 1
    data=pd.concat([white_df,non_white_df])
    data.to_csv(races_csv,index=False)
    
def extract_obstetric_features(data_csv,obst_csv):
    #extract the following obstretic features from 438_V1_previous_history_and_demographics.csv:
    #1. nulliparous due to no pregnancy (112)
    #2. nulliparous due to miscarriage (30)
    #3. Nulliparous due to other causes (23)
    #4. Parous with previous miscarriage (102)
    #5. Parous with 1 preterm delivery
    #6. Parous with > 1 preterm delivery
    #7. Parous with 1 term delivery
    #8. Parous with > 1 term delivery
    #input: 438_V1_previous_history_and_demographics.csv
    #output: the obstretic features with ids and class variable
    #1. Split the dataset into nulliparous and parous subsets
    #2. Split the nulliparous subset into 'no pregnancy', 'miscarriage' and 'unknwon cause' subsets and get their ids
    #3. Split the parous subset into 'miscarriage' and 'no miscarriage' subsets and get their ids
    #4. Create the obstetric features.
    #nulliparous patients with missing miscarriage information have a blank value in the feature 'Nulliparous due to miscarriage'
    #Parous patients with missing miscarriage information have blank values in the features 'Parous with miscarriage' and 'Parous with no miscarriage'
    data=pd.read_csv(data_csv)
    patient=data[data['hospital_id']=='JC6755']
    data.at[patient.index,'number_previous_early_miscarriages']=0#change the wrong miscarriage 1 to 0
    nulliparous=data[np.logical_and(data['no_preterm_birthsCell']==0,data['no_term_birthsCell']==0)]
    parous=data[np.logical_or(data['no_preterm_birthsCell']>0,data['no_term_birthsCell']>0)]
    (r,_)=nulliparous.shape
    print('nulliparous: ',r)
    (r,_)=parous.shape
    print('parous:',r)
    nulliparous.to_csv('nulliparous.csv',index=False)
    parous.to_csv('parous.csv',index=False)
    nulliparous_no_preg=nulliparous[np.logical_and(nulliparous['Number_of_previous_pregnancies']==0,np.isnan(nulliparous['number_previous_early_miscarriages'])==False)]
    nulliparous_miscarriage=nulliparous[np.logical_and(nulliparous['number_previous_early_miscarriages']>0,np.isnan(nulliparous['number_previous_early_miscarriages'])==False)]
    nulliparous_other_causes=nulliparous[np.logical_and(nulliparous['Number_of_previous_pregnancies']>0,nulliparous['number_previous_early_miscarriages']==0)]
    nulliparous_missing_miscarriage=nulliparous[np.isnan(nulliparous['number_previous_early_miscarriages'])==True]
    parous_miscarriage=parous[parous['number_previous_early_miscarriages']>0]
    parous_no_miscarriage=parous[parous['number_previous_early_miscarriages']==0]    
    parous_missing_miscarriage=parous[np.isnan(parous['number_previous_early_miscarriages'])==True]    
    parous_with_1_preterm_delivery=parous[parous['no_preterm_birthsCell']==1]
    parous_with_2_or_more_preterm_delivery=parous[parous['no_preterm_birthsCell']>1]
    parous_with_1_term_delivery=parous[parous['no_term_birthsCell']==1]
    parous_with_2_or_more_term_delivery=parous[parous['no_term_birthsCell']>1]
    #nulliparous_no_preg.info()
    #create features
    (r,_)=nulliparous_no_preg.shape
    print('nulliparous_no_preg',r)
    df1=pd.DataFrame(data=np.zeros((r,6)),index=nulliparous_no_preg.index,columns=['hospital_id','nulliparous_with_no_pregnancy','nulliparous_with_previous_miscarriage','nulliparous_due_to_other_causes','parous_with_previous_miscarriage','before37weeksCell'])
    df1=df1.astype(int)
    df1=df1.astype({'hospital_id':object})
    df1['hospital_id']=nulliparous_no_preg['hospital_id']
    df1['nulliparous_with_no_pregnancy']=pd.Series(data=[1 for i in range(r)],index=nulliparous_no_preg.index,name='nulliparous_with_no_pregnancy')
    df1['before37weeksCell']=nulliparous_no_preg['before37weeksCell']
    df1=df1.astype({'before37weeksCell':object})
    #print(df1)
    (r2,_)=nulliparous_miscarriage.shape
    print('nulliparous_miscarrriage',r2)
    df2=pd.DataFrame(data=np.zeros((r2,6)),index=nulliparous_miscarriage.index,columns=['hospital_id','nulliparous_with_no_pregnancy','nulliparous_with_previous_miscarriage','nulliparous_due_to_other_causes','parous_with_previous_miscarriage','before37weeksCell'])
    df2=df2.astype(int)
    df2=df2.astype({'hospital_id':object})
    df2=df2.astype({'before37weeksCell':object})
    df2['hospital_id']=nulliparous_miscarriage['hospital_id']
    df2['nulliparous_with_previous_miscarriage']=pd.Series(data=[1 for i in range(r2)],index=nulliparous_miscarriage.index,name='nulliparous_with_previous_miscarriage')
    df2['before37weeksCell']=nulliparous_miscarriage['before37weeksCell']
    df2.to_csv('nulliparous_miscarrriage.csv',index=False)
    (r3,_)=nulliparous_other_causes.shape
    print('nulliparous_other_causes',r3)
    df3=pd.DataFrame(data=np.zeros((r3,6)),index=nulliparous_other_causes.index,columns=['hospital_id','nulliparous_with_no_pregnancy','nulliparous_with_previous_miscarriage','nulliparous_due_to_other_causes','parous_with_previous_miscarriage','before37weeksCell'])
    df3=df3.astype(int)
    df3=df3.astype({'hospital_id':object})
    df3=df3.astype({'before37weeksCell':object})
    df3['hospital_id']=nulliparous_other_causes['hospital_id']
    df3['nulliparous_due_to_other_causes']=pd.Series(data=[1 for i in range(r3)],index=nulliparous_other_causes.index,name='nulliparous_due_to_other_causes')
    df3['before37weeksCell']=nulliparous_other_causes['before37weeksCell']
    (r4,_)=parous_miscarriage.shape
    print('parous_miscarriage',r4)
    df4=pd.DataFrame(data=np.zeros((r4,6)),index=parous_miscarriage.index,columns=['hospital_id','nulliparous_with_no_pregnancy','nulliparous_with_previous_miscarriage','nulliparous_due_to_other_causes','parous_with_previous_miscarriage','before37weeksCell'])
    df4=df4.astype(int)
    df4=df4.astype({'hospital_id':object})
    df4=df4.astype({'before37weeksCell':object})
    df4['hospital_id']=parous_miscarriage['hospital_id']
    df4['parous_with_previous_miscarriage']=pd.Series(data=[1 for i in range(r4)],index=parous_miscarriage.index,name='parous_with_previous_miscarriage')
    df4['before37weeksCell']=parous_miscarriage['before37weeksCell']
    (r5,_)=parous_no_miscarriage.shape
    print('parous with no miscarriage',r5)
    df5=pd.DataFrame(data=np.zeros((r5,6)),index=parous_no_miscarriage.index,columns=['hospital_id','nulliparous_with_no_pregnancy','nulliparous_with_previous_miscarriage','nulliparous_due_to_other_causes','parous_with_previous_miscarriage','before37weeksCell'])
    df5=df5.astype(int)
    df5=df5.astype({'hospital_id':object})
    df5=df5.astype({'before37weeksCell':object})
    df5['hospital_id']=parous_no_miscarriage['hospital_id']
    df5['before37weeksCell']=parous_no_miscarriage['before37weeksCell']    
    (r6,_)=nulliparous_missing_miscarriage.shape
    print('nulliparous_missing_miscarriage',r6)
    df6=pd.DataFrame(data=np.zeros((r6,6)),index=nulliparous_missing_miscarriage.index,columns=['hospital_id','nulliparous_with_no_pregnancy','nulliparous_with_previous_miscarriage','nulliparous_due_to_other_causes','parous_with_previous_miscarriage','before37weeksCell'])
    df6=df6.astype(int)
    df6=df6.astype({'hospital_id':object})
    df6=df6.astype({'before37weeksCell':object})
    df6['hospital_id']=nulliparous_missing_miscarriage['hospital_id']
    df6['nulliparous_with_previous_miscarriage']=pd.Series(data=[np.nan for i in range(r6)],index=nulliparous_missing_miscarriage.index,name='nulliparous_with_previous_miscarriage')
    df6['before37weeksCell']=nulliparous_missing_miscarriage['before37weeksCell']
    (r7,_)=parous_missing_miscarriage.shape
    print('parous_missing_miscarriage',r7)
    df7=pd.DataFrame(data=np.zeros((r7,6)),index=parous_missing_miscarriage.index,columns=['hospital_id','nulliparous_with_no_pregnancy','nulliparous_with_previous_miscarriage','nulliparous_due_to_other_causes','parous_with_previous_miscarriage','before37weeksCell'])
    df7=df7.astype(int)
    df7=df7.astype({'hospital_id':object})
    df7=df7.astype({'before37weeksCell':object})
    df7['hospital_id']=parous_missing_miscarriage['hospital_id']
    df7['parous_with_previous_miscarriage']=pd.Series(data=[np.nan for i in range(r7)],index=parous_missing_miscarriage.index,name='parous_with_previous_miscarriage')
    df7['before37weeksCell']=parous_missing_miscarriage['before37weeksCell']
    df_obst=pd.concat([df1,df2,df3,df4,df5,df6,df7])
    #add 4 more features
    (r,_)=parous_with_1_preterm_delivery.shape
    print('parous_with_1_preterm_delivery: ',r)
    (r,_)=parous_with_2_or_more_preterm_delivery.shape
    print('parous_with_2_or_more_preterm_delivery: ',r)
    (r,_)=parous_with_1_term_delivery.shape
    print('parous_with_1_term_delivery: ', r)
    (r,_)=parous_with_2_or_more_term_delivery.shape
    print('parous_with_2_or_more_term_delivery: ',r)
    (r,_)=df_obst.shape
    df8=pd.DataFrame(data=np.zeros((r,5)),index=df_obst.index,columns=['hospital_id','parous_with_1_preterm_delivery','parous_with_2_or_more_preterm_delivery','parous_with_1_term_delivery','parous_with_2_or_more_term_delivery'])
    df8=df8.astype(int)
    df8.astype({'hospital_id':object})
    df8['hospital_id']=df_obst['hospital_id']
    row_indxL=list(parous_with_1_preterm_delivery.index)
    for i in range(len(row_indxL)):
        df8.at[row_indxL[i],'parous_with_1_preterm_delivery']=1
    row_indxL2=list(parous_with_2_or_more_preterm_delivery.index)
    for i in range(len(row_indxL2)):
        df8.at[row_indxL2[i],'parous_with_2_or_more_preterm_delivery']=1
    row_indxL3=list(parous_with_1_term_delivery.index)
    for i in range(len(row_indxL3)):
        df8.at[row_indxL3[i],'parous_with_1_term_delivery']=1
    row_indxL4=list(parous_with_2_or_more_term_delivery.index)
    for i in range(len(row_indxL4)):
        df8.at[row_indxL4[i],'parous_with_2_or_more_term_delivery']=1
    df_obst=join_data(df_obst,df8,'hospital_id','hospital_id',datatype='df')
    df_obst.to_csv(obst_csv,index=False)
    remove_duplicates(obst_csv,obst_csv)
    return df_obst

#def merge_ethnicity_categories(data_csv,eth_csv):
    #merge the following ethnic categories into 1 category 'African or Arab or Asian or Black or Caribbean or Mixed or Other':
    #African (11), Arab (1), Asian (11), Black (2), Caribbean (2), Mixed (5) and Other (1)
    
def predict_trainset_and_testset_using_weka_model_and_optimal_threshold(trainset_csv,testset_csv,model,model_inputs_output_csv,results_path,model_discrete_cuts_file=None,weka_modeltype=None,display_info=False):
    import ModelsPredict as mp
    import os, sys
    weka_path='c:\\Program Files\\Weka-3-7-10\\weka.jar'   
    java_memory='4g'
    #predict the probabilities of PTB of training instances as all possible thresholds
    MP=mp.ModelsPredict(
                 results_path=results_path,
                 weka_path=weka_path,
                 java_memory=java_memory
                 )
    (auc,predL,_,_,_,_)=MP.predict_using_weka_model('prediction list',trainset_csv,model_discrete_cuts_file,model,weka_modeltype,model_inputs_output_csv,results_path,weka_path,java_memory)
    #predL=list of (inst,float(preterm_prob),output_class,actual_class,error)
    #find the optimal threshold maximizing tpr and tnr of the training set from all the thresholds
    preterm_probL=[]
    actual_classL=[]
    for i in range(len(predL)):
        pred=predL[i]
        preterm_prob=pred[1]
        actual_class=pred[3]
        preterm_probL.append(float(preterm_prob))
        actual_classL.append(float(actual_class))
    (optimal_threshold,tpr,tnr,fpr,fnr)=MP.performance_using_optimal_threshold_maximizing_tpr_tnr(preterm_probL,actual_classL,display_info=display_info)                       
    print('optimal threshold: ',str(np.round(optimal_threshold,2)))
    print('performance on training set using optimal threshold: ')
    print('TPR=',np.round(tpr,2))
    print('TNR=',np.round(tnr,2))
    print('FPR=',np.round(fpr,2))
    print('FNR=',np.round(fnr,2))
    print('AUC=',np.round(auc,2))
    f=open(results_path+'results_of_predicting_trainset_and_testset.txt','w')
    f.write('model='+str(model)+'\n')
    f.write('training set='+str(trainset_csv)+'\n')
    f.write('test set='+str(testset_csv)+'\n')
    f.write('optimal threshold: '+str(np.round(optimal_threshold,2))+'\n')
    f.write('performance on training set using optimal threshold: \n')
    f.write('TPR='+str(np.round(tpr,2))+'\n')
    f.write('TNR='+str(np.round(tnr,2))+'\n')
    f.write('FPR='+str(np.round(fpr,2))+'\n')
    f.write('FNR='+str(np.round(fnr,2))+'\n')
    f.write('AUC='+str(np.round(auc,2))+'\n')
    #predict probabilites of PTB of test instances, then, use the optimal threshold to classify the test instances
    if os.path.isfile(testset_csv):
        (auc2,tpr2,tnr2,fpr2,fnr2)=MP.predict_using_weka_model('no prediction list',testset_csv,model_discrete_cuts_file,model,weka_modeltype,model_inputs_output_csv,results_path,weka_path,java_memory,threshold=optimal_threshold)
        print('performance on test set using optimal threshold: ')
        print('TPR=',np.round(tpr2,2))
        print('TNR=',np.round(tnr2,2))
        print('FPR=',np.round(fpr2,2))
        print('FNR=',np.round(fnr2,2))
        print('AUC=',np.round(auc2,2))
        f.write('performance on test set using optimal threshold: \n')
        f.write('TPR='+str(np.round(tpr2,2))+'\n')
        f.write('TNR='+str(np.round(tnr2,2))+'\n')
        f.write('FPR='+str(np.round(fpr2,2))+'\n')
        f.write('FNR='+str(np.round(fnr2,2))+'\n')
        f.write('AUC='+str(np.round(auc2,2))+'\n')
        f.close()
    else:
        sys.exit(testset_csv+' does not exist')

def predict_testset_using_weka_model_and_threshold(testset_csv,model,model_inputs_output_csv,results_path,model_discrete_cuts_file=None,weka_modeltype=None,threshold=0.5):
    import ModelsPredict as mp
    import os, sys
    weka_path='c:\\Program Files\\Weka-3-7-10\\weka.jar'   
    java_memory='4g'
    MP=mp.ModelsPredict(
                 results_path=results_path,
                 weka_path=weka_path,
                 java_memory=java_memory
                 )
    if os.path.isfile(testset_csv):
        (auc2,tpr2,tnr2,fpr2,fnr2)=MP.predict_using_weka_model('no prediction list',testset_csv,model_discrete_cuts_file,model,weka_modeltype,model_inputs_output_csv,results_path,weka_path,java_memory,threshold=threshold)
        print('performance on test set using threshold:',threshold)
        print('TPR=',np.round(tpr2,2))
        print('TNR=',np.round(tnr2,2))
        print('FPR=',np.round(fpr2,2))
        print('FNR=',np.round(fnr2,2))
        print('AUC=',np.round(auc2,2))
        f=open(results_path+'results_of_predicting_testset.txt','w')
        f.write('model='+str(model)+'\n')
        f.write('test set='+str(testset_csv)+'\n')
        f.write('threshold: '+str(np.round(threshold,2))+'\n')
        f.write('performance on test set using threshold: \n')
        f.write('TPR='+str(np.round(tpr2,2))+'\n')
        f.write('TNR='+str(np.round(tnr2,2))+'\n')
        f.write('FPR='+str(np.round(fpr2,2))+'\n')
        f.write('FNR='+str(np.round(fnr2,2))+'\n')
        f.write('AUC='+str(np.round(auc2,2))+'\n')
        f.close()
    else:
        sys.exit(testset_csv+' does not exist')

def predict_prob_of_ptb_using_ensemble_with_hard_voting(traindf,model,optimal_thresholds_of_base_models=None,optimal_threshold_of_ensemble=None,base_models_take_columnTransformer_selected_features=False,score='my_score',mini_no_of_class1_votes='optimize',reward_scale=1/4,max_diff=0.3,logfile=None):
    #input: training data, traindf
    #       ensemble model using hard voting
    #output: probabilities of preterm birth of the instances of df output by hard voting (majority voting)
    (r,c)=traindf.shape
    modelsL=model.estimators_
    preterm_probL=[]
    optimal_thresholdsL=[]
    if optimal_thresholds_of_base_models==None:#find the optimal threshold of each base model by predicting the training data
        i=1
        for model in modelsL:
            print('===base model'+str(i)+'===')
            optimal_threshold,_,_,_,_,_,_,_,_,_,_ = predict_trainset_and_testset_using_sklearn_and_optimal_threshold(traindf,traindf,model,score=score,reward_scale=reward_scale,max_diff=max_diff)
            optimal_thresholdsL.append(np.round(optimal_threshold,2))
            i+=1
        if logfile!=None:
            f=open(logfile,'a')
            f.write('optimal thresholds of base models='+str(optimal_thresholdsL)+'\n')
            f.close()
            print('optimal_thresholdsL='+str(optimal_thresholdsL))
    else:#the optinal thrsholds of based models have been found previously
        optimal_thresholdsL=optimal_thresholds_of_base_models
    #predict test set using hard voting
    print('===predict traininset set or test set using hard voting===')
    for i in range(r):#use the optimal threshold of each model to classify each instance
      labels=[]
      j=0
      for model in modelsL:
          if base_models_take_columnTransformer_selected_features:
              instance=traindf.iloc[i,:].to_numpy()
              instance=instance.reshape(1,-1)#reshape a 1-d array to 2-d array of 1 row
              probs=model.predict_proba(pd.DataFrame(instance,columns=traindf.columns))          
          else:
              instance=traindf.iloc[i,0:c-1].to_numpy()
              instance=instance.reshape(1,-1)#reshape a 1-d array to 2-d array of 1 row
              probs=model.predict_proba(instance)
          prob_PTB = probs[:,1]
          if prob_PTB >= optimal_thresholdsL[j]:
              label=1
          else:
              label=0  
          labels.append(label)
          j+=1
      prob=sum(labels)/len(modelsL) #probability of an instance belonging to class1 (no. of class1 votes/total no. of votes)
      preterm_probL.append(prob)
      #optimal_threshold = np.mean(optimal_thresholdsL)
      #optimal_threshold = np.round(optimal_threshold,2)
      #print('optimal threshold of hard voting: ',str(np.round(optimal_threshold,2)))
    MP = mp.ModelsPredict()
    (_,c)=traindf.shape
    targetsL=list(traindf.iloc[:,c-1])
    if optimal_threshold_of_ensemble!=None:
       (tpr,tnr,fpr,fnr)=MP.performance_using_threshold(optimal_threshold_of_ensemble,preterm_probL,targetsL)
       return preterm_probL,optimal_threshold_of_ensemble,optimal_thresholdsL,tpr,tnr,fpr,fnr      
    else:
       if isinstance(mini_no_of_class1_votes,int):#a user specified mini_no_of_class1_votes for predicting a instance as class1
          optimal_threshold_of_ensemble=mini_no_of_class1_votes/len(modelsL)
          (tpr,tnr,fpr,fnr)=MP.performance_using_threshold(optimal_threshold_of_ensemble,preterm_probL,targetsL) 
       elif mini_no_of_class1_votes=='optimize':    
              best_score=0
              optimal_threshold_of_ensemble=-1
              for  i in range(len(modelsL)):
                  mini_no_of_class1_votes=i+1
                  threshold=mini_no_of_class1_votes/len(modelsL)
                  #print('threshold: ',threshold)
                  if threshold <= 0.5:#threshold 0.5 implements the majority vote i.e. if prob of class1 >= prob of class0 then predict class1 else predict class0. 
                      (tpr,tnr,fpr,fnr)=MP.performance_using_threshold(threshold,preterm_probL,targetsL)
                      if score=='my_score':
                          score2=(tpr+tnr)/2
                          if score2 > best_score:
                              optimal_threshold_of_ensemble=threshold
                              best_score=score2
                      else:
                          sys.exit('invalid score in predict_prob_of_ptb_using_ensemble_with_hard_voting: '+score+'\n')
                  else:
                      break
       elif mini_no_of_class1_votes=='majority':
           (tpr,tnr,fpr,fnr)=MP.performance_using_threshold(0.5,preterm_probL,targetsL)
           optimal_threshold_of_ensemble=0.5
       else:
              sys.exit('invalid mini_no_of_class1_votes: '+mini_no_of_class1_votes)
       return preterm_probL,optimal_threshold_of_ensemble,optimal_thresholdsL,tpr,tnr,fpr,fnr

def predict_trainset_and_testset_using_sklearn_and_optimal_threshold(traindf,testdf,model,display_info=False,ensemble_with_hard_voting=False,ensemble_with_cv_auc_voting=False,mini_no_of_class1_votes='optimize',base_models_take_columnTransformer_selected_features=False,score='my_score',reward_scale=1/4,max_diff=0.3,results_path=None,trainset_csv=None,testset_csv=None,logfile=None):
    #predict the probabilities of PTB of training instances as all possible thresholds
    #score: my_score, my_score2, G-mean or youden
    #my score2= (tpr+tnr)/2+(tpr-tnr)*reward_scale where (tpr-tnr) is a reward if tpr > tnr and is a penalty (-ve reward) if tpr < tnr
    import ModelsPredict as mp
    (r,c)=traindf.shape
    targetsL=list(traindf.iloc[:,c-1])
    if ensemble_with_hard_voting:#compute optimal threshold of each base model and predict prob of PTB using hard voting (majority vote)
        print('===hard voting===')
        preterm_probL,optimal_threshold,optimal_thresholds_of_base_models,tpr,tnr,fpr,fnr=predict_prob_of_ptb_using_ensemble_with_hard_voting(traindf,model,mini_no_of_class1_votes=mini_no_of_class1_votes,base_models_take_columnTransformer_selected_features=base_models_take_columnTransformer_selected_features,score=score,reward_scale=reward_scale,max_diff=max_diff,logfile=logfile)
    else:
        if ensemble_with_cv_auc_voting:
            prob=model.predict_proba(traindf.iloc[:,0:c-1])*len(model.estimators_) #prob of PTB = prob1 of model1 x cv_auc1/total_cv_auc + prob2 of model2 x cv_auc2/total_cv_auc+...+prob_k of model_k x cv_auc_k/total_cv_auc where weight1 is weight of model1, prob1 is prob of PTB of model1 etc. 
            for i in range(r):
                if prob[i,0] > 1: #if prob of onterm > 1, change it to 1
                    prob[i,0]=1
                if prob[i,1] > 1: #if prob of PTB > 1, change it to 1
                    prob[i,1]=1
        else:    
            prob=model.predict_proba(traindf.iloc[:,0:c-1])
        preterm_probL=list(prob[:,1])
        #print('preterm_probL',preterm_probL)
        MP = mp.ModelsPredict()
        (optimal_threshold,tpr,tnr,fpr,fnr)=MP.performance_using_optimal_threshold_maximizing_tpr_tnr(preterm_probL,targetsL,display_info=display_info,score=score,reward_scale=reward_scale,max_diff=max_diff)                       
    auc=roc_auc_score(targetsL,preterm_probL)
    print('optimal threshold: ',str(np.round(optimal_threshold,2)))
    print('performance on training set using optimal threshold: ')
    print('TPR=',np.round(tpr,2))
    print('TNR=',np.round(tnr,2))
    print('FPR=',np.round(fpr,2))
    print('FNR=',np.round(fnr,2))
    print('AUC=',np.round(auc,2))
    if logfile!=None:
            f=open(logfile,'a')
            f.write('optimal threshold: '+str(np.round(optimal_threshold,2))+'\n')
            f.write('performance on the training set using optimal threshold: \n')
            f.write('TPR='+str(np.round(tpr,2))+'\n')
            f.write('TNR='+str(np.round(tnr,2))+'\n')
            f.write('FPR='+str(np.round(fpr,2))+'\n')
            f.write('FNR='+str(np.round(fnr,2))+'\n')
            f.write('AUC='+str(np.round(auc,2))+'\n')
            f.close()
    if results_path!=None:
            f=open(results_path+'training_testing_using_optimal_threshold_results.txt','w')
            f.write('training set='+trainset_csv+'\n')
            f.write('testset ='+testset_csv+'\n')
            f.write('optimal threshold: '+str(np.round(optimal_threshold,2))+'\n')
            f.write('performance on the training set using optimal threshold: \n')
            f.write('TPR='+str(np.round(tpr,2))+'\n')
            f.write('TNR='+str(np.round(tnr,2))+'\n')
            f.write('FPR='+str(np.round(fpr,2))+'\n')
            f.write('FNR='+str(np.round(fnr,2))+'\n')
            f.write('AUC='+str(np.round(auc,2))+'\n')
            f.close()
    #predict probabilites of PTB of test instances, then, use the optimal threshold to classify the test instances
    ##if testset is not the same data as training set, predict testset.
    r,_=traindf.shape
    r2,_=testdf.shape
    if list(traindf.iloc[0,:])!=list(testdf.iloc[0,:]) and list(traindf.iloc[r-1,:])!=list(testdf.iloc[r2-1,:]) and list(traindf.iloc[2,:])!=list(testdf.iloc[2,:]):
        if ensemble_with_hard_voting:
            auc2,tpr2,tnr2,fpr2,fnr2 = predict_testset_using_sklearn_model_and_threshold(testdf,model,results_path,threshold=optimal_threshold,ensemble_with_hard_voting=ensemble_with_hard_voting,optimal_thresholds_of_base_models=optimal_thresholds_of_base_models,optimal_threshold_of_ensemble=optimal_threshold)
        else:
            auc2,tpr2,tnr2,fpr2,fnr2 = predict_testset_using_sklearn_model_and_threshold(testdf,model,results_path,ensemble_with_cv_auc_voting=ensemble_with_cv_auc_voting,threshold=optimal_threshold)        
        return optimal_threshold,auc,tpr,tnr,fpr,fnr,auc2,tpr2,tnr2,fpr2,fnr2
    else:
        return optimal_threshold,auc,tpr,tnr,fpr,fnr,999,999,999,999,999
        
def predict_testset_using_sklearn_model_and_threshold(testdf,model,results_path,threshold=0.5,ensemble_with_hard_voting=False,ensemble_with_cv_auc_voting=False,optimal_thresholds_of_base_models=None,mini_no_of_class1_votes=None,optimal_threshold_of_ensemble=None,testset_csv=None):
    #predict probabilites of PTB of test instances, then, use the optimal threshold to classify the test instances
    (r,c)=testdf.shape
    if ensemble_with_hard_voting:
        preterm_probL,_,_,_,_,_,_=predict_prob_of_ptb_using_ensemble_with_hard_voting(testdf,model,optimal_thresholds_of_base_models=optimal_thresholds_of_base_models,optimal_threshold_of_ensemble=optimal_threshold_of_ensemble,mini_no_of_class1_votes=mini_no_of_class1_votes)
    else: 
        if ensemble_with_cv_auc_voting:
            prob=model.predict_proba(testdf.iloc[:,0:c-1])*len(model.estimators_) #prob = prob of model1 x cv_auc1/total_cv_auc + prob of model2 x cv_auc2/total_cv_auc+...+prob of model_k x cv_auc_k/total_cv_auc 
            for i in range(r):
                if prob[i,0] > 1: #if prob > 1, change it to 1
                    prob[i,0]=1
                if prob[i,1] > 1:
                    prob[i,1]=1
        else:       
            prob=model.predict_proba(testdf.iloc[:,0:c-1])
        preterm_probL=list(prob[:,1])
    targetsL=list(testdf.iloc[:,c-1])
    #print('targetsL2: '+str(targetsL2))
    #print('preterm_probL2: '+str(preterm_probL2))
    auc=roc_auc_score(targetsL,preterm_probL)
    MP = mp.ModelsPredict()
    (tpr,tnr,fpr,fnr)=MP.performance_using_threshold(threshold,preterm_probL,targetsL)
    print('threshold='+str(np.round(threshold,2)))
    print('performance on test set using threshold='+str(np.round(threshold,2))+':')
    print('TPR=',np.round(tpr,2))
    print('TNR=',np.round(tnr,2))
    print('FPR=',np.round(fpr,2))
    print('FNR=',np.round(fnr,2))
    print('AUC=',np.round(auc,2))
    if results_path!=None and testset_csv!=None:
        f=open(results_path+'testing_using_threshold_results.txt','w')
        f.write('testset='+testset_csv+'\n')
        f.write('threshold='+str(threshold)+'\n')
        f.write('performance on the test set using the threshold:\n')
        f.write('TPR='+str(np.round(tpr,2))+'\n')
        f.write('TNR='+str(np.round(tnr,2))+'\n')
        f.write('FPR='+str(np.round(fpr,2))+'\n')
        f.write('FNR='+str(np.round(fnr,2))+'\n')
        f.write('AUC='+str(np.round(auc,2))+'\n')
        f.close()
    return auc,tpr,tnr,fpr,fnr

def predict_testset(option,model,testset,linear_reg=False):
    #use a sklearn model to predict classes of a testset using 0.5 threshold
    #testset, a dataframe
    (m,c)=testset.shape
    #s=model.score(testset.iloc[:,0:c-1],testset.iloc[:,c-1])
    #print('OOB score='+str(model.oob_score_))
    if linear_reg==False:
        y=model.predict(testset.iloc[:,0:c-1])#predict the classes of instances using 0.5 threshold
        #print('targets='+str(testset.iloc[:,c-1]))
        prob=model.predict_proba(testset.iloc[:,0:c-1])
        #print(prob)
        class1_prob=prob[:,1]
        y_testset=testset.iloc[:,c-1]
        auc=roc_auc_score(y_testset,class1_prob)
        #recall=recall_score(y_testset,y)
        #accuracy=accuracy_score(y_testset,y)
        tn, fp, fn, tp = confusion_matrix(y_testset,y).ravel()
        tnr=tn/(tn+fp)
        tpr=tp/(tp+fn)
        fpr=1-tnr
        fnr=1-tpr
        #print('accuracy='+str(accuracy))
        #y_true=np.array([2,1,2,2,2,2,1])#actual classes
        #y_score=np.array([1-0.76,1-0.76,0.962,0.962,0.962,0.962,1-0.947])#these are the scores of label 2 (the larger label) which are output by the classifier. 
                                                                         #these are the scores of label 1 when the labels are 0 and 1. 
        #roc_auc_score(y_true,y_score)
        if option:
          predL=[]
          for i in range(m):
              preterm_prob=float(class1_prob[i])
              #if y[i] != y_testset.iloc[i,0]:
              if y[i] != y_testset.iloc[i]:
                 error='+'
              else:
                 error=' '
              predL.append((i,float(preterm_prob),y[i],y_testset.iloc[i],error))
          return (auc,predL,tpr,tnr,fpr,fnr) 
        else:        
          return (auc,tpr,tnr,fpr,fnr)
    else:#use a linear regression to output prob of preterm birth
        preterm_probs=model.predict(testset.iloc[:,0:c-1])#prob of preterm birth (class 1)
        y_testset=testset.iloc[:,c-1]        
        #convert prob to between 0 and 1 if it is > 1 or < 0.
        predL=[]
        y=[]
        for i in range(m):
            if preterm_probs[i] > 1:
                preterm_probs[i]=1
            elif preterm_probs[i] < 0:
                preterm_probs[i]=0
            if preterm_probs[i] >= 1-preterm_probs[i]:#classification threshold 0.5
                predicted_class=1
            else:
                predicted_class=0
            y.append(predicted_class)
            if predicted_class != y_testset.iloc[i]:
                error='+'
            else:
                error=' '
            predL.append((i,preterm_probs[i],predicted_class,y_testset.iloc[i],error))
        auc=roc_auc_score(y_testset,preterm_probs)
        tn, fp, fn, tp = confusion_matrix(y_testset,y).ravel()
        tnr=tn/(tn+fp)
        tpr=tp/(tp+fn)
        fpr=1-tnr
        fnr=1-tpr
        if option:
          return (predL,auc,tpr,tnr,fpr,fnr) 
        else:        
          return (auc,tpr,tnr,fpr,fnr)  

def predict_PTB_using_PTB_history(testdatafile,threshold=0.5):
    #predict PTB using the rules:
    #1) If patient has PTB history then PTB prediction is “1”
    #2) If patient has no PTB history the PTB prediction is “0”
    #input: have_ptb_history_no_treatment_with_ids.csv (test data)
    #output: AUC of test data
    if os.path.isfile(testdatafile):
            data=pd.read_csv(testdatafile)
            cols=list(data.columns)
            (_,c)=data.shape
            targetsL=list(data.iloc[:,c-1])
            if cols[0] == 'Identifier' or cols[0] == 'hospital_id' or cols[0] == 'id' or cols[0] == 'Id' or cols[0] == 'ID':   
                preterm_probL=data.iloc[:,1]#'have PTB history' at 1st column
                data=data.iloc[:,1:c] #removed ids column from whole dataset
            else:
                preterm_probL=data.iloc[:,0]#'have PTB history' at 0th column    
    else:
            sys.exit('dataset does not exist: '+testdatafile)
    #print('preterm_probL',list(preterm_probL))
    #print('targetsL',targetsL)
    auc=roc_auc_score(targetsL,preterm_probL)
    print('AUC=',np.round(auc,2))
    MP = mp.ModelsPredict()
    (tpr,tnr,fpr,fnr) = MP.performance_using_threshold(threshold,preterm_probL,targetsL)
    print('threshold:',threshold)
    print('TPR=',np.round(tpr,2))
    print('TNR=',np.round(tnr,2))
    print('FPR=',np.round(fpr,2))
    print('FNR=',np.round(fnr,2))
    #information gain of PTB history feature (index 0) of dataset
    from gain import gain_of_feature
    (mi,_,col)=gain_of_feature(data,0)
    print('mutal information of '+col+':'+str(np.round(mi,2))+'\n')
    
def select_all_readings_of_ids(idsfile,allreadings_with_ids,outfile):
    data=pd.read_csv(allreadings_with_ids)
    cols=list(data.columns)
    ids=pd.read_csv(idsfile)
    (r2,_)=ids.shape
    readings_of_ids=pd.DataFrame(columns=cols)    
    for i in range(r2):
        pid=ids.iat[i,0]
        readings=data[data[cols[0]]==pid]
        readings=pd.DataFrame(readings,columns=cols)
        readings_of_ids=pd.concat([readings_of_ids,readings])
    readings_of_ids.to_csv(outfile,index=False)
    
def reduce_data(data_csv,model_inputs_output_csv,reduced_data_csv):
    #if model_inputs_output_csv contains original features, reduce data to the original features 
    #if model_inputs_output_csv contains polynomial features, transform the features of data to the polynomial features
    if os.path.isfile(data_csv):
        file=open(data_csv,'r')
        features=set(file.readline())
        file.close()
    else:
        features=set(list(data_csv.columns))
    file2=open(model_inputs_output_csv,'r')
    model_features=set(file2.readline())
    file2.close()
    if model_features.issubset(features):#The features of data includes the inputs of the model, select features
       reduce_data2('csv',data_csv,model_inputs_output_csv,reduced_data_csv)
    else:#the inputs of model are polynomial features, but, data consists of original features, polynomial feature transformation
       construct_poly_features_of_another_dataset('original_features',data_csv,model_inputs_output_csv,reduced_data_csv,'none')
    df=pd.read_csv(reduced_data_csv)
    df=fill_missing_values('median','df',df)
    if reduced_data_csv!=None:
        df.to_csv(reduced_data_csv,index=False)
    return df

def discrete_features_to_binary_features(discrete_data_arff,binary_data_arff,last_col,weka_path,java_memory):
    import os
    cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.filters.unsupervised.attribute.NominalToBinary -R first-"+last_col+" -i \""+discrete_data_arff+"\" -o \""+binary_data_arff+"\""
    os.system(cmd)
    print(cmd)
    
def get_p_value_auc_ci(resultsfile,resultsfiletype='log reg training testing'):
    #get P-value and AUC confidence intervals output by a R script
    #input: results file of a R script e.g. logreg.R
    #results file format of log reg training testing: p-value
    #                                                 c(lower bound, auc, upper bound)                       
    #results file format of random forest training testing: mean p-value, min p-value, max p-value
    #                                                       c(lower bound, auc, upper bound)
    #results file format of testing: c(lower bound, auc, upper bound)
    #                           e.g. c(0.455287598319069, 0.531794871794872, 0.608302145270675) 
    f=open(resultsfile,'r')
    line=f.readline()
    line=line.rstrip()#remove \n
    if resultsfiletype=='log reg training testing':
        m=re.match('^([-\d\.e]+)$',line)
        if m:
            pvalue=m.group(1)
        else:
            sys.exit('pvalue not matched')
        line=f.readline()
        line=line.rstrip()#remove \n
        m=re.match('^c\(([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\)$',line)
        if m:
            auc=m.group(2)
            ci_lower=m.group(1)
            ci_upper=m.group(3)
        else:
            sys.exit('auc ci not matched')
        return (float(pvalue),float(auc),float(ci_lower),float(ci_upper))
    elif resultsfiletype == 'rf training testing':
        m=re.match('^([-\d\.e]+),\s*([-\d\.e]+),\s*([-\d\.e]+)$',line)
        if m:
            mean_pval=m.group(1)
            min_pval=m.group(2)
            max_pval=m.group(3)
            print(mean_pval)
            print(min_pval)
            print(max_pval)
        else:
            sys.exit('pvalue not matched')
        line=f.readline()
        line=line.rstrip()#remove \n
        m=re.match('^c\(([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\)$',line)
        if m:
            auc=m.group(2)
            ci_lower=m.group(1)
            ci_upper=m.group(3)
        else:
            sys.exit('auc ci not matched')    
        return (float(mean_pval),float(min_pval),float(max_pval),float(auc),float(ci_lower),float(ci_upper))
    elif resultsfiletype=='testing':
        m=re.match('^c\(([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\)$',line)
        if m:
            auc=m.group(2)
            ci_lower=m.group(1)
            ci_upper=m.group(3)
        elif line=='c(NA, NA, NA)':#c(NA, NA, NA)
            return 'c(NA, NA, NA)'
        else:    
            sys.exit('auc ci not matched')
        return (float(auc),(float(ci_lower),float(ci_upper)))              
    
def convert_neg_phase_to_pos_phase(eis_data,outdata):
    #if a phase is -A, new phase is B=360-A
    data=pd.read_csv(eis_data)
    (r,_)=data.shape
    phase_indx=list(set([i for i in range(29)])-set([j for j in range(14)])-set([28]))
    for i in range(r):
        for j in phase_indx:
            if data.iat[i,j] < 0:#negative phase
                data.iat[i,j]=360+data.iat[i,j]#convert to positive phase
    data.to_csv(outdata,index=False)
    
def split_train_valid_and_test_sets_furthest_to_mean(data,train_fraction=0.66,valid_fraction=0.17,test_fraction=0.17,datatype='csv',trainfile=None,validfile=None,testfile=None):
    #Motivation: the patterns of a class furthest to the mean of the class are easy to distinguish from the ones of the other class furthest to the mean of that class
    #Strategy:
    #Select a train_size fraction of data as a training set so that the selected training patterns are furthest to the mean of each class.
    #Randomly and stratifiedly split the remaining data into validation and test sets.
    if train_fraction+valid_fraction+test_fraction!=1:
        sys.exit('train_fraction+valid_fraction+test_fraction!=1')
    if datatype == 'csv':
        data=pd.read_csv(data)
    (mean0L,mean1L)=compute_mean(data)
    cols=list(data.columns)
    df1=data[data[cols[-1]]==1]
    df0=data[data[cols[-1]]==0]
    #select class1 training patterns closest to mean of class1 of whole dataset
    L1=[]
    n=len(mean1L)
    (r1,_)=df1.shape
    for i in range(r1):
        eucdist1=0#Euclidean distance to class1 mean
        for k in range(n):
            fval=df1.iloc[i,k]
            eucdist1+=np.abs(mean1L[k]-fval)**2            
        eucdist1=np.sqrt(eucdist1)
        L1.append((i,eucdist1))
    L0=[]
    (r0,_)=df0.shape
    for i in range(r0):
        eucdist0=0#Euclidean distance to class0 mean
        for k in range(n):
            fval=df0.iloc[i,k]
            eucdist0+=np.abs(mean0L[k]-fval)**2            
        eucdist0=np.sqrt(eucdist0)
        L0.append((i,eucdist0))
    train_size1=int(np.round(train_fraction*r1))
    indx1L=[]
    L1.sort(key=operator.itemgetter(1),reverse=True)#sort class1 patterns in descending order of eucdist    
    for i in range(train_size1):
        (inst,_)=L1[i]
        indx1L.append(inst)
    train_size0=int(np.round(train_fraction*r0))
    indx0L=[]
    L0.sort(key=operator.itemgetter(1),reverse=True)
    for i in range(train_size0):
        (inst,_)=L0[i]
        indx0L.append(inst)
    df1=data.iloc[indx1L,:]
    df0=data.iloc[indx0L,:]
    trainset=pd.concat([df1,df0])
    ##plot mean of class1 and mean of class0
    import matplotlib.pyplot as plt
    xaxis=[i for i in range(len(mean0L))]
    plt.figure(1)
    plt.plot(xaxis,mean0L,'-bo',xaxis,mean1L,'-rs')
    (m0L,m1L)=compute_mean(trainset)
    plt.figure(2)
    plt.plot(xaxis,m0L,'-bo',xaxis,m1L,'-rs')
    plt.show()
    ###remove training set from whole data and split the remaining data into validation set and test set close to mean of class1 and mean of class0 of whole data
    df2=dataframes_diff(data,trainset)
    df2.to_csv('df2.csv',index=False)
    df2=pd.read_csv('df2.csv')
    if valid_fraction > 0 and test_fraction > 0:
       test_fraction2=test_fraction/(test_fraction+valid_fraction)
       seed=random.randint(0,5**9)
       (validset,testset)=split_train_test_sets(df2,test_fraction2,seed,cols[-1])
       if trainfile!=None:
           trainset.to_csv(trainfile,index=False)
       if testfile!=None:
           testset.to_csv(testfile,index=False)
       if validfile!=None:
           validset.to_csv(validfile,index=False)
       return (trainset,validset,testset)
    elif valid_fraction==0:
       testset=df2
       if trainfile!=None:
           trainset.to_csv(trainfile,index=False)
       if testfile!=None:
           testset.to_csv(testfile,index=False)
       return (trainset,testset)

def myplot(mean0L,mean1L,fignum):
    import matplotlib.pyplot as plt
    xaxis=[i for i in range(len(mean0L))]
    plt.figure(fignum)
    plt.plot(xaxis,mean0L,'-bo',xaxis,mean1L,'-rs')
    plt.show()
    
def compute_mean(data,datatype='df'):
    #compute the means of the features of class1 and means of features of class0
    mean0L=[]
    mean1L=[]
    if datatype=='df':
        (_,c)=data.shape
    elif datatype=='csv':
        data=pd.read_csv(data)
        (_,c)=data.shape
    else:
        sys.exit('invalid datatype: '+datatype)
    for j in range(c-1):#compute the mean of each feature of preterm class and onterm class excluding the class variable
        (mean0j,mean1j)=mean_of_feature(j,data,'df')
        mean0L.append(mean0j)
        mean1L.append(mean1j)
    return (mean0L,mean1L)

def template_match_filter(trainset,allreadings_with_ids,filter_option='filter1',testset_ids_csv=None,good_readings_csv=None,readings_of_id_csv=None):
    #select best-matched readings of testset ids to the preterm and onterm templates (vectors of means of training set)
    trainset=pd.read_csv(trainset)
    (mean0,mean1)=mean_of_feature(0,trainset,'df') #mean of 1st feature i.e. amplitude of lowest frequency
    (mean0_2,mean1_2)=mean_of_feature(1,trainset,'df') #mean of 2nd feature    
    (mean0_3,mean1_3)=mean_of_feature(2,trainset,'df') #mean of 3rd feature    
    (mean0_4,mean1_4)=mean_of_feature(3,trainset,'df') #mean of 4 feature    
    (mean0_5,mean1_5)=mean_of_feature(4,trainset,'df') #mean of 5 feature    
    (mean0_6,mean1_6)=mean_of_feature(5,trainset,'df') #mean of 6 feature    
    (mean0_7,mean1_7)=mean_of_feature(6,trainset,'df') #mean of 7 feature    
    (mean0_8,mean1_8)=mean_of_feature(7,trainset,'df') #mean of 8 feature    
    (mean0_9,mean1_9)=mean_of_feature(8,trainset,'df') #mean of 8 feature    
    (mean0_10,mean1_10)=mean_of_feature(9,trainset,'df') #mean of 10 feature
    (mean0_21,mean1_21)=mean_of_feature(20,trainset,'df') #mean of 21th feature    
    (mean0_22,mean1_22)=mean_of_feature(21,trainset,'df') #mean of 22th feature    
    (mean0_23,mean1_23)=mean_of_feature(22,trainset,'df') #mean of 23th feature    
    (mean0_24,mean1_24)=mean_of_feature(23,trainset,'df') #mean of 24th feature    
    
    mean0L=[]
    mean1L=[]
    (_,c)=trainset.shape
    for j in range(c-1):#compute the mean of each feature of preterm class and onterm class excluding the class variable
        (mean0j,mean1j)=mean_of_feature(j,trainset,'df')
        mean0L.append(mean0j)
        mean1L.append(mean1j)    
    data=pd.read_csv(allreadings_with_ids)
    (r,c)=data.shape
    cols=list(data.columns)
    if testset_ids_csv!=None:
        ids=pd.read_csv(testset_ids_csv)
        (r2,_)=ids.shape
        good_readings=pd.DataFrame(columns=cols[1:c-1])
        scoresL=[]
        for j in range(r2):    
            pid=ids.iat[j,0]
            #print(pid)
            data2=data[data[cols[0]]==pid]#get all the readings of an id
            readings_of_id=data2.iloc[:,1:c]#remove the ids column
            if filter_option=='filter1':
                (goodreading,score)=reading_filter1(mean0,mean1,readings_of_id)
            elif filter_option=='filter2':
                (goodreading,score)=reading_filter2((mean0,mean1),(mean0_2,mean1_2),(mean0_3,mean1_3),
                             (mean0_4,mean1_4),(mean0_5,mean1_5),(mean0_6,mean1_6),
                             (mean0_7,mean1_7),(mean0_8,mean1_8),(mean0_9,mean1_9),
                             (mean0_10,mean1_10),(mean0_21,mean1_21),
                             (mean0_22,mean1_22),(mean0_23,mean1_23),(mean0_24,mean1_24),readings_of_id)
            elif filter_option=='filter3':
                (goodreading,score)=reading_filter3(trainset,(mean0,mean1),(mean0_2,mean1_2),(mean0_3,mean1_3),
                 (mean0_4,mean1_4),(mean0_5,mean1_5),(mean0_6,mean1_6),
                 (mean0_7,mean1_7),(mean0_8,mean1_8),(mean0_9,mean1_9),
                 (mean0_10,mean1_10),(mean0_21,mean1_21),
                 (mean0_22,mean1_22),(mean0_23,mean1_23),(mean0_24,mean1_24),readings_of_id)
            elif filter_option=='filter4':#closest Euclidean distance between a reading and preterm template or onterm template
                (goodreading,score)=reading_filter4(mean0L,mean1L,readings_of_id)
            elif filter_option=='filter5':#closest Euclidean distance between a reading and preterm template or onterm template
                (goodreading,score)=reading_filter5(mean0L,mean1L,readings_of_id)
            elif filter_option=='filter6':#closest Euclidean distance between a reading and preterm template or onterm template
                (goodreading,score)=reading_filter6(mean0L,mean1L,readings_of_id)
            else:
                sys.exit('invalid filter_option in template_match_filter: ',filter_option)
            good_readings=good_readings.append(goodreading)
            scoresL.append(score)
        if good_readings_csv!=None:
            good_readings.to_csv(good_readings_csv,index=False)
        return (mean0L,mean1L,good_readings,scoresL)
    elif readings_of_id_csv !=None:
        readings_of_id=pd.read_csv(readings_of_id_csv)
        if filter_option=='filter1':
           (goodreading,score)=reading_filter1(mean0,mean1,readings_of_id)
        elif filter_option=='filter2':
           (good_reading,score)=reading_filter2((mean0,mean1),(mean0_2,mean1_2),readings_of_id)
        elif filter_option=='filter3':
           (goodreading,score)=reading_filter3(mean0,mean1,readings_of_id)
        elif filter_option=='filter4':#closest Euclidean distance between a reading and preterm template or onterm template
           (goodreading,score)=reading_filter4(mean0L,mean1L,readings_of_id)
        else:
           sys.exit('invalid filter_option in template_match_filter: ',filter_option)
        if good_readings_csv!=None:
           good_reading.to_csv(good_readings_csv,index=False)
        return (mean0L,mean1L,good_reading,score)

def change_between_features(valuesL):
    #Given a list of features f1, f2,...,f28, compute f1-f2,f2-f3,f3-f4,...,f27-f28
    diffL=[]
    for i in range(len(valuesL)-1):
        diff=valuesL[i]-valuesL[i+1]
        diffL.append(diff)
    return diffL

def change_between_features_transform(readingsdata,outfile):
    #transform an eis dataset (e.g. my_selected_unselected_eis_readings_28_inputs.csv) with 28 features (amplitudes and phases) to new features, the change between features 
    readingsdf=pd.read_csv(readingsdata)
    (r,c)=readingsdf.shape
    cols=list(readingsdf.columns)
    cols2=[]
    for i in range(c-2):
        col2=cols[i]+'-'+cols[i+1]
        cols2.append(col2)
    cols2.append(cols[-1])
    L=[]
    for i in range(r):
        reading=list(readingsdf.iloc[i,0:c-1])#exclude the last column (targets)
        diffL=change_between_features(reading)    
        diffL.append(readingsdf.iloc[i,c-1])
        L.append(diffL)
    df=pd.DataFrame(L,columns=cols2) 
    df.to_csv(outfile,index=False)
    return df

def reading_filter6(mean0L,mean1L,readings_of_id):
    d01=mean0L[0]-mean0L[8]#change in amplitude between freq 1 (max point) and freq 9 of onterm template
    d02=mean0L[14]-mean0L[22]#change in phase between freq 15 and freq 25 (mini point) of onterm template
    d03=mean0L[24]-mean0L[27]#change in phase between freq 25 and freq 28 of onterm template
    d11=mean1L[0]-mean1L[8]#change in amplitude between freq 1 and freq 9 of preterm template
    d12=mean1L[14]-mean1L[22]#change in phase between freq 15 and freq 23 (mini point) of preterm template
    d13=mean1L[22]-mean1L[27]#change in phase between freq 23 and freq 28 of preterm template
    (r,c)=readings_of_id.shape
    L=[]
    for i in range(r):
        reading=list(readings_of_id.iloc[i,0:c-1])#exclude the last column (targets)
        diff01=reading[0]-reading[8]
        diff02=reading[14]-reading[22]
        diff03=reading[24]-reading[27]
        diff12=reading[14]-reading[22]
        diff13=reading[22]-reading[27]
        diff022=np.abs(reading[22]-mean0L[22])
        diff122=np.abs(reading[22]-mean1L[22])
        dist0=np.abs(diff01-d01)+np.abs(diff02-d02)*np.abs(diff03-d03)
        dist1=np.abs(diff01-d11)+np.abs(diff12-d12)*np.abs(diff13-d13)
        #dist0=np.abs(diff02-d02)+np.abs(diff03-d03)
        #dist1=np.abs(diff12-d12)+np.abs(diff13-d13)
        if dist1 <= dist0:
            L.append((i,dist1))
        else:
            L.append((i,dist0))
    L.sort(key=operator.itemgetter(1),reverse=False)#sort readings in ascending order of eucdist
    (best_reading_indx,best_eucdist)=L[0]
    score=0.2**best_eucdist
    return (readings_of_id.iloc[best_reading_indx,:],score)

def reading_filter5(mean0L,mean1L,readings_of_id):
    #use change between features as features to find closest matched reading to preterm or onterm template 
    diffL0=change_between_features(mean0L)
    diffL1=change_between_features(mean1L)
    (r,c)=readings_of_id.shape
    L=[]
    for i in range(r):
        reading=list(readings_of_id.iloc[i,0:c-1])#exclude the last column (targets)
        diffL=change_between_features(reading)
        eucdist1=0#Euclidean distance to onterm template
        eucdist0=0#Euclidean distance to onterm template
        for k in range(len(diffL0)):
            eucdist1+=np.abs(diffL1[k]-diffL[k])**2
            eucdist0+=np.abs(diffL0[k]-diffL[k])**2
        eucdist1=np.sqrt(eucdist1)
        eucdist0=np.sqrt(eucdist0)        
        if eucdist0 < eucdist1:
            L.append((i,eucdist0))
        else:
            L.append((i,eucdist1))
    L.sort(key=operator.itemgetter(1),reverse=False)#sort readings in ascending order of eucdist
    (best_reading_indx,best_eucdist)=L[0]
    score=0.2**best_eucdist
    return (readings_of_id.iloc[best_reading_indx,:],score)

def reading_filter4(mean0L,mean1L,readings_of_id):
    #select the reading that is the closest to the mean vector of preterm or mean vector of onterm at each frequency 
    #readings_of_id: all the readings of an id in a test set   
    #mean0L: list of mean of each feature of onterm class of training set
    #mean1L: list of mean of each feature of preterm class of training set    
    (r,c)=readings_of_id.shape
    #L0=[]#list of readings closest to onterm template
    #L1=[]#list of readings closest to preterm template
    L=[]
    #w1=8 #weight for 1st feature 
    #w2=8 #weight for 23rd feature
    n=len(mean0L)
    for i in range(r):
        eucdist1=0#Euclidean distance to onterm template
        eucdist0=0#Euclidean distance to onterm template
        for k in range(n):
        #for k in [0,21,22]:#features 1, 22, 23
            fval=readings_of_id.iloc[i,k]
            eucdist1+=np.abs(mean1L[k]-fval)**2
            eucdist0+=np.abs(mean0L[k]-fval)**2
        eucdist1=np.sqrt(eucdist1)
        eucdist0=np.sqrt(eucdist0)        
        if eucdist0 < eucdist1:
            L.append((i,eucdist0))
        else:
            L.append((i,eucdist1))
    L.sort(key=operator.itemgetter(1),reverse=False)#sort readings in ascending order of dist
    (best_reading_indx,best_eucdist)=L[0]
    #if there are numerous readings closest to preterm template, select the closest one
    #else if there is one reading closest to preterm template, select it
    #else select the reading closest to onterm template
    #if len(L1)>=1: 
    #    L1.sort(key=operator.itemgetter(1),reverse=False)
    #    (best_reading_indx,best_eucdist)=L1[0]
    #    print('selected reading ',best_reading_indx,' is closest to preterm. eucdist=',best_eucdist)
    #else:
    #    L0.sort(key=operator.itemgetter(1),reverse=False)
    #    (best_reading_indx,best_eucdist)=L0[0]
    #    print('selected reading ',best_reading_indx,' is closest to onterm. eucdist=',best_eucdist)
    ###min-max rescale based score###
    '''
    eucdistL=[]
    for (_,eucdist) in L:
        eucdistL.append(eucdist)
    X_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    scaler=X_scaler.fit(np.array(eucdistL).reshape(-1,1))
    best_eucdist=scaler.transform(np.array([best_eucdist]).reshape(-1,1))
    best_eucdist=best_eucdist[0][0]
    best_score=1-best_eucdist
    '''
    ####score=0.2^dist###
    #base=0.8
    #best_score=base**best_eucdist
    #print('reading ',best_reading_indx,' is selected. score=',best_score)
    #print('===distances of readings of id===')
    #print('L1')
    #for i in range(len(L1)):
    #    (reading_indx,eucdist)=L1[i]
    #    print('reading ',reading_indx,'eucdist=',eucdist)
    #print('L0')
    #for i in range(len(L0)):
    #    (reading_indx,eucdist)=L0[i]
    #    print('reading ',reading_indx,'eucdist=',eucdist)
    #    score=base**eucdist
        ###min-max rescale based score###
        #eucdist=scaler.transform(np.array([eucdist]).reshape(-1,1))
        #eucdist=eucdist[0][0]
        #score=1-eucdist
    #    print('reading ',reading_indx,'score=',score)
    best_score=0.2**best_eucdist
    return (readings_of_id.iloc[best_reading_indx,:],best_score)

def reading_filter1(mean0,mean1,readings_of_id):
    #select the reading that is the closest to either mean of preterm at freq1 or mean of onterm at freq1 
    #readings_of_id: all the readings of an id in a test set   
    #mean0: mean of amplitude of frequency 1 of onterm class of training set
    #mean1: mean of amplitude of frequency 1 of preterm class of training set    
    (r,_)=readings_of_id.shape
    L=[]
    for i in range(r):
        amp1_of_reading=readings_of_id.iat[i,0]#amplitude of frequency 1
        dist0=np.abs(mean0-amp1_of_reading)
        dist1=np.abs(mean1-amp1_of_reading)
        if dist0 < dist1:
            L.append((i,dist0))
            #print('reading ',i,' closest to onterm')
        else:
            L.append((i,dist1))
            #print('reading ',i,' closest to preterm')
    L.sort(key=operator.itemgetter(1),reverse=False)#sort readings in ascending order of dist
    (best_reading_indx,dist)=L[0]
    if dist > 1:
        score=1-np.log10(dist)
    else:
        score=1-dist
    #print('reading ',best_reading_indx,' is selected. distance=',dist)
    #print('===distances of all readings of id===')
    #for i in range(len(L)):
    #    (reading_indx,dist)=L[i]
    #    print('reading ',reading_indx,'distance=',dist)
    return (readings_of_id.iloc[best_reading_indx,:],score)

def reading_filter2(mean0_and_mean1,mean0_2_and_mean1_2,mean0_3_and_mean1_3,
                    mean0_4_and_mean1_4,mean0_5_and_mean1_5,mean0_6_and_mean1_6,
                    mean0_7_and_mean1_7,mean0_8_and_mean1_8,mean0_9_and_mean1_9,
                    mean0_10_and_mean1_10,mean0_21_and_mean1_21,
                    mean0_22_and_mean1_22,mean0_23_and_mean1_23,mean0_24_and_mean1_24,readings_of_id):
#def reading_filter2(mean0_and_mean1,mean0_2_and_mean1_2,mean0_3_and_mean1_3,readings_of_id):
    #select the reading that is the closest to preterm template at mean of amplitude at freq1 and mean of phase at freq9 or onterm template 
    #readings_of_id: all the readings of an id in a test set   
    #mean0_and_mean1=(mean0,mean1)
    #mean0_2_and_mean1_2=(mean0_2,mean1_2)
    mean0=mean0_and_mean1[0]#mean0: mean of amplitude of frequency 1 of onterm class of training set
    mean1=mean0_and_mean1[1] #mean1: mean of amplitude of frequency 1 of preterm class of training set
    mean0_2=mean0_2_and_mean1_2[0] #mean0_2: mean of frequency 2 of preterm class of training set
    mean1_2=mean0_2_and_mean1_2[1] #mean1_2: mean of frequency 2 of onterm class of training set
    mean0_3=mean0_3_and_mean1_3[0] #mean0_3: mean of frequency 3 of preterm class of training set
    mean1_3=mean0_3_and_mean1_3[1] #mean1_3: mean of frequency 3 of onterm class of training set
    mean0_4=mean0_4_and_mean1_4[0] 
    mean1_4=mean0_4_and_mean1_4[1]
    mean0_5=mean0_5_and_mean1_5[0] 
    mean1_5=mean0_5_and_mean1_5[1]
    mean0_6=mean0_6_and_mean1_6[0] 
    mean1_6=mean0_6_and_mean1_6[1]
    mean0_7=mean0_7_and_mean1_7[0] 
    mean1_7=mean0_7_and_mean1_7[1]
    mean0_8=mean0_8_and_mean1_8[0] 
    mean1_8=mean0_8_and_mean1_8[1]
    mean0_9=mean0_9_and_mean1_9[0] 
    mean1_9=mean0_9_and_mean1_9[1]
    mean0_10=mean0_10_and_mean1_10[0] 
    mean1_10=mean0_10_and_mean1_10[1]
    mean0_21=mean0_21_and_mean1_21[0] 
    mean1_21=mean0_21_and_mean1_21[1] 
    mean0_22=mean0_22_and_mean1_22[0] 
    mean1_22=mean0_22_and_mean1_22[1] 
    mean0_23=mean0_23_and_mean1_23[0] 
    mean1_23=mean0_23_and_mean1_23[1] 
    mean0_24=mean0_24_and_mean1_24[0] 
    mean1_24=mean0_24_and_mean1_24[1]
    (r,_)=readings_of_id.shape
    L=[]
    for i in range(r):
        amp1_of_reading=readings_of_id.iloc[i,0]#amplitude of frequency 1
        amp2_of_reading=readings_of_id.iloc[i,1]#amplitude of frequency 2
        amp3_of_reading=readings_of_id.iloc[i,2]#amplitude of frequency 3
        amp4_of_reading=readings_of_id.iloc[i,3]#amplitude of frequency 3
        amp5_of_reading=readings_of_id.iloc[i,4]#amplitude of frequency 3
        amp6_of_reading=readings_of_id.iloc[i,5]#amplitude of frequency 3
        amp7_of_reading=readings_of_id.iloc[i,6]#amplitude of frequency 3
        amp8_of_reading=readings_of_id.iloc[i,7]#amplitude of frequency 3
        amp9_of_reading=readings_of_id.iloc[i,8]#amplitude of frequency 3
        amp10_of_reading=readings_of_id.iloc[i,9]#amplitude of frequency 3
        f21_of_reading=readings_of_id.iat[i,20]#feature 21        
        f22_of_reading=readings_of_id.iat[i,21]#feature 22
        f23_of_reading=readings_of_id.iat[i,22]#feature 20
        f24_of_reading=readings_of_id.iat[i,23]#feature 21
        
        dist0=np.round(np.abs(mean0-amp1_of_reading),4)
        dist1=np.round(np.abs(mean1-amp1_of_reading),4)
        dist0_2=np.round(np.abs(mean0_2-amp2_of_reading),4)
        dist1_2=np.round(np.abs(mean1_2-amp2_of_reading),4)
        dist0_3=np.abs(mean0_3-amp3_of_reading)
        dist1_3=np.abs(mean1_3-amp3_of_reading)
        dist0_4=np.abs(mean0_4-amp4_of_reading)
        dist1_4=np.abs(mean1_4-amp4_of_reading)
        dist0_5=np.abs(mean0_5-amp5_of_reading)
        dist1_5=np.abs(mean1_5-amp5_of_reading)
        dist0_6=np.abs(mean0_6-amp6_of_reading)
        dist1_6=np.abs(mean1_6-amp6_of_reading)
        dist0_7=np.abs(mean0_7-amp7_of_reading)
        dist1_7=np.abs(mean1_7-amp7_of_reading)
        dist0_8=np.abs(mean0_8-amp8_of_reading)
        dist1_8=np.abs(mean1_8-amp8_of_reading)
        dist0_9=np.abs(mean0_9-amp9_of_reading)
        dist1_9=np.abs(mean1_9-amp9_of_reading)
        dist0_10=np.abs(mean0_10-amp10_of_reading)
        dist1_10=np.abs(mean1_10-amp10_of_reading)
        dist0_21=np.round(np.abs(mean0_21-f21_of_reading),4)
        dist1_21=np.round(np.abs(mean1_21-f21_of_reading),4)
        dist0_22=np.round(np.abs(mean0_22-f22_of_reading),4)
        dist1_22=np.round(np.abs(mean1_22-f22_of_reading),4)
        dist0_23=np.abs(mean0_23-f23_of_reading)
        dist1_23=np.abs(mean1_23-f23_of_reading)
        dist0_24=np.abs(mean0_24-f24_of_reading)
        dist1_24=np.abs(mean1_24-f24_of_reading)
        
        if dist0 < dist1:
            s1=dist0
        else:
            s1=dist1
            
        if dist0_2 < dist1_2:
            s2=dist0_2
        else:
            s2=dist1_2
           
        if dist0_3 < dist1_3:
            s3=dist0_3
        else:
            s3=dist1_3
        
        if dist0_4 < dist1_4:
            s4=dist0_4
        else:
            s4=dist1_4
        
        if dist0_5 < dist1_5:
            s5=dist0_5
        else:
            s5=dist1_5
        if dist0_6 < dist1_6:
            s6=dist0_6
        else:
            s6=dist1_6
        if dist0_7 < dist1_7:
            s7=dist0_7
        else:
            s7=dist1_7
        if dist0_8 < dist1_8:
            s8=dist0_8
        else:
            s8=dist1_8
        if dist0_9 < dist1_9:
            s9=dist0_9
        else:
            s9=dist1_9
        if dist0_10 < dist1_10:
            s10=dist0_10
        else:
            s10=dist1_10
        
        if dist0_21 < dist1_21:
            s21=dist0_21
        else:
            s21=dist1_21
        
        if dist0_22 < dist1_22:
            s22=dist0_22
        else:
            s22=dist1_22
        
        if dist0_23 < dist1_23:
            s23=dist0_23
        else:
            s23=dist1_23
        if dist0_24 < dist1_24:
            s24=dist0_24
        else:
            s24=dist1_24
        '''
        if dist0 <= dist1 or dist0_2 <= dist1_2:#if a reading is closer to onterm template than to preterm template, get its distance to onterm template at other frequencies
            s1=dist0     #else get its distance to the preterm template at these frequencies
            s2=dist0_2
            s21=dist0_21
            s22=dist0_22
        elif dist1 <= dist0 or dist1_2 <= dist0_2: 
            s1=dist1
            s2=dist1_2
            s21=dist1_21
            s22=dist1_22
        '''    
        #total=s1+math.log(s2,10)+1.6*s21*s22 #total distance
        #total=s1+math.log(s2,2)+1.6*s21*s22 #total distance
        #total=s1+math.log(s2,2)+s22 #total distance
        total=s1**2+s21*s22
        
        L.append((i,total))
    L.sort(key=operator.itemgetter(1),reverse=False)#sort readings in ascending order of dist
    (best_reading_indx,dist)=L[0]
    
    if dist >= 1:
        score=1-np.log10(dist)
    else:
        score=1-dist
    #print('reading ',best_reading_indx,' is selected. distance=',dist)        
    #print('===distances of all readings of id===')
    #for i in range(len(L)):
    #    (reading_indx,dist)=L[i]
    #    print('reading ',reading_indx,'distance=',dist)
    return (readings_of_id.iloc[best_reading_indx,:],score)

def reading_filter3(trainset,mean0_and_mean1,mean0_2_and_mean1_2,mean0_3_and_mean1_3,
                    mean0_4_and_mean1_4,mean0_5_and_mean1_5,mean0_6_and_mean1_6,
                    mean0_7_and_mean1_7,mean0_8_and_mean1_8,mean0_9_and_mean1_9,
                    mean0_10_and_mean1_10,mean0_21_and_mean1_21,
                    mean0_22_and_mean1_22,mean0_23_and_mean1_23,mean0_24_and_mean1_24,                    
                    readings_of_id):
    #readings_of_id: all the readings of an id in a test set   
    mean0=mean0_and_mean1[0]#mean0: mean of amplitude of frequency 1 of onterm class of training set
    mean1=mean0_and_mean1[1] #mean1: mean of amplitude of frequency 1 of preterm class of training set
    mean0_2=mean0_2_and_mean1_2[0] #mean0_2: mean of frequency 2 of preterm class of training set
    mean1_2=mean0_2_and_mean1_2[1] #mean1_2: mean of frequency 2 of onterm class of training set
    mean0_3=mean0_3_and_mean1_3[0] #mean0_3: mean of frequency 3 of preterm class of training set
    mean1_3=mean0_3_and_mean1_3[1] #mean1_3: mean of frequency 3 of onterm class of training set
    (std0,std1)=std_of_feature(0,trainset,'df')
    (std0_1,std1_1)=std_of_feature(1,trainset,'df')
    (std0_2,std1_2)=std_of_feature(2,trainset,'df')
    
    (r,_)=readings_of_id.shape
    #cols=list(readings_of_id.columns)
    L=[]
    for i in range(r):
        amp1=readings_of_id.iloc[i,0]#amplitude of frequency 1
        amp2=readings_of_id.iloc[i,1]#amplitude of frequency 2
        amp3=readings_of_id.iloc[i,2]#amplitude of frequency 3
        s=amp1+amp2+amp3
        L.append((i,s))     
    L.sort(key=operator.itemgetter(1),reverse=True)#sort readings in descending order of amp1
    (max_reading_indx,s_max)=L[0]
    if len(L) > 1:
        (min_reading_indx,s_min)=L[-1]
    else:
        score=get_score(mean0,mean1,s_max)
        return (readings_of_id.iloc[max_reading_indx,:],score)

    if s_max >= (mean1+mean1_2+mean1_3) and s_max <= (mean1+mean1_2+mean1_3+np.sqrt(np.sqrt(std1))):#+std1_1+std1_2):#max reading is above both mean1 and mean0 (mean1 > mean0)
        score=get_score(mean0,mean1,s_max)
        return (readings_of_id.iloc[max_reading_indx,:],score)#return max reading
    if s_min <= (mean0+mean0_2+mean0_3) and s_min >= (mean0+mean0_2+mean0_3+std0):#+std0_1):#+std0_2) :#max reading is below both mean1 and mean0 (mean1 > mean0)
        score=get_score(mean0,mean1,s_min)
        return (readings_of_id.iloc[min_reading_indx,:],score)#return mini reading
    #elif s_max < mean1+mean1_2+mean1_3 and s_max > mean0+mean0_2+mean0_3 and s_min < mean0+mean0_2+mean0_3:#max reading is between mean1 and mean0 and min reading is below mean0
    #    score=get_score(mean0,mean1,s_min)
    #    return (readings_of_id.iloc[min_reading_indx,:],score)#return mini reading
    else:#max reading and min reading are between preterm template and onterm template
        return reading_filter2(mean0_and_mean1,mean0_2_and_mean1_2,mean0_3_and_mean1_3,
                               mean0_4_and_mean1_4,mean0_5_and_mean1_5,mean0_6_and_mean1_6,
                               mean0_7_and_mean1_7,mean0_8_and_mean1_8,mean0_9_and_mean1_9,
                               mean0_10_and_mean1_10,mean0_21_and_mean1_21,
                               mean0_22_and_mean1_22,mean0_23_and_mean1_23,mean0_24_and_mean1_24,
                               readings_of_id)#return the reading closest to preterm template or onterm template

def get_score(mean0,mean1,amp1_max):
    dist0=np.abs(mean0-amp1_max)
    dist1=np.abs(mean1-amp1_max)
    if dist0 < dist1:
       if dist0 >= 1:
          score=1-np.log10(dist0)
       else:
          score=1-dist0
    else:
       if dist1 >= 1:
          score=1-np.log10(dist1)
       else:
          score=1-dist1
    return score

def mean_of_feature(featureIndx,data,datatype='csv'):
    if datatype=='csv':
        data=pd.read_csv(data)
    cols=list(data.columns)
    preterm=data[data[cols[-1]]==1]
    onterm=data[data[cols[-1]]==0]    
    m1=preterm[cols[featureIndx]].mean()#amplitude of lowest frequency
    m0=onterm[cols[featureIndx]].mean()
    return (m0,m1)

def std_of_feature(featureIndx,data,datatype='csv'):
    if datatype=='csv':
        data=pd.read_csv(data)
    cols=list(data.columns)
    preterm=data[data[cols[-1]]==1]
    onterm=data[data[cols[-1]]==0]    
    std1=preterm[cols[featureIndx]].std()#amplitude of lowest frequency
    std0=onterm[cols[featureIndx]].std()
    return (std0,std1)

def create_folder_if_not_exist(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        #print("Directory " , dirName ,  " Created ")
    #else:    
        #print("Directory " , dirName ,  " already exists")
        
def split_list(option,l,n,rseed=123456):
    #split a list l into n equal-sized sublists of random items or sequential-ordered items
    #s=np.ceil(len(l)/n)
    #s=np.round(len(l)/n)
    s=np.floor(len(l)/n)
    if option=='random':
        indxL=[i for i in range(len(l))]
        sys.setrecursionlimit(5*n) #to prevent recursion depth error in Python
        #print('recursion limit:',sys.getrecursionlimit())
        return mysplit(1,n,indxL,l,s,rseed,[])
    elif option=='ordered':
        return mysplit2(1,n,l,s,[])
    else:
        sys.exit('wrong option in split_list: ',option)

def mysplit(k,n,indxL,l,s,rseed,sublists):
    #Recursively split a list of length L into a random sublist of size s and the remaining sublist of size L-s until the size of remaining sublist is < s for n times
    #k, the kth sublist of size s
    #output: list of sublists of size s
    if k < n:
       #print('k:',k)
       random.seed(rseed)
       #l2=random.sample(l,int(s))
       if int(s) <= len(indxL):
           indxL2=random.sample(indxL,int(s))
           l2=[]
           for indx in indxL2:
               l2.append(l[indx])
           sublists.append(l2)
           indxL3=list(set(indxL)-set(indxL2))
           return mysplit(k+1,n,indxL3,l,s,rseed,sublists)           
       else:
           l2=[]
           for indx in indxL:
               l2.append(l[indx])
           if len(l2)>0:
               sublists.append(l2)
           return sublists
    else:
       l2=[]
       for indx in indxL:
           l2.append(l[indx])
       sublists.append(l2)
       return sublists

def mysplit2(k,n,l,s,sublists):
    #Recursively split a list of length L into a sublist of size s with sequential-ordered items and the remaining sublist of size L-s for n times
    if k < n:
        l2=l[0:int(s)]
        sublists.append(l2)
        j=0
        while j < s:
            del l[0]
            j+=1
        #print(sublists)
        return mysplit2(k+1,n,l,s,sublists)
    else:
        sublists.append(l)
        return sublists

def compute_auc_ci(targets_csv,pred_csv,outfile_csv):
    #compute AUC CI from actual targets and the predicted probabilites of a classifier    
    import os
    cmd='Rscript ci.R \"'+targets_csv+'\" \"'+pred_csv+'\" \"'+outfile_csv+'\"'
    os.system(cmd)
    print(cmd)
    
def log_reg_training_testing_p_value_and_auc_ci(trainset_csv,testset_csv,resultsfile,modelfile,classvariable):
    #p value and auc confidence interval of logistic regression
    import os
    cmd='Rscript logreg.R training_testing \"'+trainset_csv+'\" \"'+testset_csv+'\" \"'+resultsfile+'\" \"'+modelfile+'\" \"'+classvariable+'\"'
    os.system(cmd)
    print(cmd)

def log_reg_testing_auc_ci(modelfile,testset_csv,resultsfile):
    #auc confidence interval of logistic regression
    import os
    cmd='Rscript logreg.R testing \"'+modelfile+'\" \"'+testset_csv+'\" \"'+resultsfile+'\"'
    os.system(cmd)
    print(cmd) 

def random_forest_training_testing_p_value_and_auc_ci(seed,features,trees,trainset_csv,testset_csv,resultsfile,modelfile,classvariable):
    #p values and auc confidence interval of random forest
    import os
    cmd='Rscript rf.R training_testing '+str(seed)+' '+str(features)+' '+str(trees)+' \"'+trainset_csv+'\" \"'+testset_csv+'\" \"'+resultsfile+'\" \"'+modelfile+'\" \"'+classvariable+'\"'
    os.system(cmd)
    print(cmd)

def random_forest_testing_auc_ci(modelfile,testset_csv,resultsfile):
    #auc confidence interval of random forest
    import os
    cmd='Rscript rf.R testing \"'+modelfile+'\" \"'+testset_csv+'\" \"'+resultsfile+'\"'
    os.system(cmd)
    print(cmd) 

def delete_files(filesL):
    for file in filesL:
        #file=convert_to_windows_path(file)
        if file!=None:            
            if os.path.isfile(file):
                if file=='none':
                    #current_dir=os.getcwd()    
                    #cmd='del \"'+current_dir+'\\none\"'
                    cmd='del none'
                else:
                    cmd='del \"'+file+'\"'
                #print(cmd)
                try:
                    subprocess.check_call(cmd,shell=True)
                except subprocess.CalledProcessError:
                    cmd='rm \"'+file+'\"'
                    subprocess.call(cmd,shell=True)

def transform_inputs_df_using_scaler(testset,X_scaler_file):
    #input: testset dataframe, X_scaler_file (.joblib)
    scaler=load(X_scaler_file)
    (_,c)=testset.shape
    X_test=testset.iloc[:,:c-1]
    X_test=scaler.transform(X_test)
    testset.iloc[:,:c-1]=X_test
    return testset

def zscore_normalize_inputs(X_train,X_test):
    #scale inputs to zero mean and unit variance (if inputs are non-Gaussian, z-score normalization is not suitable)
    X_scaler = preprocessing.StandardScaler()
    X_scaler = X_scaler.fit(X_train)
    X_train2 = X_scaler.transform(X_train)
    X_test2 = X_scaler.transform(X_test)
    return (X_train2,X_test2,X_scaler)

def zscore_normalize_inputs_df(trainset,testset):
    #input: trainset dataframe, testset dataframe
    (_,c)=trainset.shape
    X_train=trainset.iloc[:,:c-1]
    X_test=testset.iloc[:,:c-1]
    (X_train2,X_test2,X_scaler)=zscore_normalize_inputs(X_train,X_test)
    trainset2=trainset
    testset2=testset
    trainset2.iloc[:,:c-1]=X_train2
    testset2.iloc[:,:c-1]=X_test2
    return (trainset2,testset2,X_scaler)

def normalize_training_and_test_sets(data_path,
                                     normalized_data_path,
                                     iterations=100,
                                     train_set='trainset',
                                     test_set='testset',
                                     normalize_method='minmax',
                                     mini=0,
                                     maxi=1):
    create_folder_if_not_exist(normalized_data_path)
    for i in range(iterations):       
        trainset=pd.read_csv(data_path+train_set+str(i)+'.csv')
        testset=pd.read_csv(data_path+test_set+str(i)+'.csv')       
        if normalize_method=='minmax':
            print('minimum maximum normalize')
            (trainset,testset,_)=min_max_normalize_inputs_df(trainset,testset,mini=mini,maxi=maxi)
        elif normalize_method=='zscore':
            print('zscore normalize')
            (trainset,testset,_)=zscore_normalize_inputs_df(trainset,testset)
        else:
            sys.exit('In normalize_training_and_test_sets, invalid normalize method: ',normalize_method)
        trainset.to_csv(normalized_data_path+train_set+str(i)+'.csv',index=False)
        testset.to_csv(normalized_data_path+test_set+str(i)+'.csv',index=False)
        
def min_max_normalize_inputs(X_train,X_test,mini=0,maxi=1):
    #scale inputs to range [mini,max]
    print('min-max normalization: mini='+str(mini)+', max='+str(maxi))
    X_scaler = preprocessing.MinMaxScaler(feature_range=(int(mini),int(maxi)))
    X_scaler = X_scaler.fit(X_train)
    X_train2 = X_scaler.transform(X_train)
    X_test2 = X_scaler.transform(X_test)
    return (X_train2,X_test2,X_scaler)

def min_max_normalize_inputs_df(trainset,testset,mini=0,maxi=1):
    trainset2=trainset
    testset2=testset
    cols=list(trainset.columns)
    if cols[0] == 'Identifier' or cols[0] == 'hospital_id' or cols[0] == 'id' or cols[0] == 'Id' or cols[0] == 'ID':
        (_,c)=trainset.shape
        trainset=trainset.iloc[:,1:c] #remove ids column
        testset=testset.iloc[:,1:c] #remove ids column
    (_,c)=trainset.shape
    X_train=trainset.iloc[:,:c-1]
    X_test=testset.iloc[:,:c-1]
    (X_train2,X_test2,X_scaler)=min_max_normalize_inputs(X_train,X_test,mini=mini,maxi=maxi)
    (_,c2)=trainset2.shape
    if cols[0] == 'Identifier' or cols[0] == 'hospital_id' or cols[0] == 'id' or cols[0] == 'Id' or cols[0] == 'ID':
        trainset2.iloc[:,1:c2-1]=X_train2
        testset2.iloc[:,1:c2-1]=X_test2
    else:
        trainset2.iloc[:,:c2-1]=X_train2
        testset2.iloc[:,:c2-1]=X_test2
    return (trainset2,testset2,X_scaler)

def norm_normalize_inputs(X_data):
    #normalize inputs to unit l1 or l2 norm (default)
    t = Normalizer().fit(X_data)
    X_data2=t.transform(X_data)
    return (X_data2,t)

def norm_normalize_inputs_df(data):
    (_,c)=data.shape
    X=data.iloc[:,:c-1]
    X2=norm_normalize_inputs(X)
    data2=data
    data2.iloc[:,:c-1]=X2
    return data2

def power_transform_normalize_inputs(X_data,strategy):
    #input: X_data dataframe (inputs of a dataset)
    if strategy == 'yb':
        pt = PowerTransformer(method='yeo-johnson')
    elif strategy == 'bc':
        pt = PowerTransformer(method='box-cox')
    else:
        print('invalide power transform: '+strategy)
    pt = pt.fit(X_data)
    X_data = pt.transform(X_data)
    return (X_data,pt)

def mycopyfile(source,target):
    #copy content of source to target
    source=convert_to_windows_path(source)
    if os.path.isfile(source):
        try:
            copyfile(source, target)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)
        #print("copy file done.")
    #else:
        #print(source,' does not exist. mycopyfile failed.')
        
def construct_poly_features_of_another_dataset(featurestype,testset_csv,inputs_output_csv,testset_reduced_csv,testset_reduced_arff):
        #Transform the features of testset_csv to the polynomial features of inputs_output_csv
        #input: filetype, 'original_features' or 'poly_features'
        #       testset_csv (at results_path)
        #       input_output_csv
        #       testset_reduced_csv
        #       results_path, path where testset_csv is and outputs are saved
        #output: testdf 
        #        testset_reduced_csv (if not none)
        #        testset_reduced_arff (if not none)
        testdf=poly_features2(testset_csv,featurestype,inputs_output_csv,testset_reduced_csv)
        if testset_reduced_arff!='none' and testset_reduced_arff!=None:
            dataframe_to_arff(testdf,testset_reduced_arff)#write 0, 1 targets to arff file
            #Check whether the arff file contains unknown classes (?), replace the ?s or string target to nominal target e.g. @attribute before37weeksCell {0,1}
            f=open(testset_csv,'r')
            f.readline()
            line=f.readline().rstrip()
            l=line.split(',')
            targetVal=l[len(l)-1]
            f.close()
            if targetVal=='?': #unknown class and convert mssing targets in arff file to nominal targets
                replace_missing_target_or_string_target_with_nominal_target(inputs_output_csv,testset_reduced_arff)
        return testdf
    
def balanced_trainset_indx(iteration,option,trainsets_indx_file,balanced_trainset_size,dataset,results_path,seed,weka_path,java_memory):
   if os.path.isfile(trainsets_indx_file):
       indx_file=open(trainsets_indx_file,'r')
   else:
       sys.exit(trainsets_indx_file+' does not exist')
   lines=indx_file.readlines()
   train_indx=lines[iteration]
   train_indx=train_indx.split(',')
   indx_file.close()
   for i in range(len(train_indx)):
       train_indx[i]=int(train_indx[i])
   data=pd.read_csv(dataset)
   (_,c)=data.shape
   cols=list(data.columns)
   labels=data.iloc[train_indx,c-1]#get the class labels
   labels=list(labels)
   trainL=[]  
   for i in range(len(train_indx)):
       indx_class=[train_indx[i],labels[i]]
       trainL.append(indx_class)
   traindf=pd.DataFrame(trainL,columns=['indx',cols[c-1]])
   dataframe_to_arff(traindf,'trainset'+str(iteration)+'_indx.arff')
   (r,c)=traindf.shape
   z=str(int(balanced_trainset_size)/r*100)#size of the balanced training set as a percentage of the size of the original training set    
   prep.resample('trainset'+str(iteration)+'_indx.arff','trainset'+str(iteration)+'_indx_balanced.arff',str(seed),str(z),weka_path,java_memory)
   prep.convert_arff_to_csv('trainset'+str(iteration)+'_indx_balanced.arff','trainset'+str(iteration)+'_indx_balanced.csv',weka_path,java_memory)
   prep.remove_duplicates('trainset'+str(iteration)+'_indx_balanced.csv','trainset'+str(iteration)+'_indx_balanced_unique.csv')
   train_balanced_unique=pd.read_csv('trainset'+str(iteration)+'_indx_balanced_unique.csv',low_memory=False)
   train_balanced=pd.read_csv('trainset'+str(iteration)+'_indx_balanced.csv')
   #(r,_)=train_balanced.shape
   #if r < int(balanced_trainset_size):
   #    d=int(balanced_trainset_size) - r
   #    toadd=random.sample(list(train_balanced.index),d)
   #    train_balanced=pd.concat([train_balanced,train_balanced.iloc[toadd,:]])
   labels=prep.get_labels(traindf)
   if option == 1:
       train_balanced=add_remaining_samples_to_balanced_trainset(3,traindf,train_balanced_unique,train_balanced,results_path,labels,seed,weka_path,java_memory)                
   elif option == 2:       
       train_balanced=add_remaining_samples_to_balanced_trainset2(3,traindf,train_balanced_unique,train_balanced,results_path,labels,seed,weka_path,java_memory)
   elif option == 3:
       train_balanced=add_remaining_samples_to_balanced_trainset3(3,traindf,train_balanced_unique,train_balanced,results_path,labels,seed,weka_path,java_memory)
   else:
       print('invalid option: ',option)
       return -1
   (r2,c)=train_balanced.shape
   cols=list(train_balanced.columns)
   print('size of train balanced: '+str(r2))
   class0=train_balanced[train_balanced[cols[c-1]]==0]
   class1=train_balanced[train_balanced[cols[c-1]]==1]
   class0_row_labels=[]
   for row in class0.itertuples():
       row_label=row[1]
       class0_row_labels.append(row_label)
   class1_row_labels=[]
   for row in class1.itertuples():
       row_label=row[1]
       class1_row_labels.append(row_label)
   for i in range(len(class0_row_labels)):
       class0_row_labels[i]=int(class0_row_labels[i])
   for i in range(len(class1_row_labels)):
       class1_row_labels[i]=int(class1_row_labels[i])
   return (class0_row_labels,class1_row_labels)
   '''
   if r2 >= balanced_trainset_size:
       half0=int((balanced_trainset_size)/2)
       half1=balanced_trainset_size-half0
       class0=train_balanced[train_balanced[cols[c-1]]==0]
       class1=train_balanced[train_balanced[cols[c-1]]==1]
       (r0,_)=class0.shape
       (r1,_)=class1.shape
       ###remove or add instances from class0 or to class0
       indx0=set()
       class0_row_labels=[]
       for row in class0.itertuples():
               row_label=row[1]
               class0_row_labels.append(row_label)
       class0_row_labels2=class0_row_labels
       d0=np.abs(r0-half0)
       for row in class0.itertuples():
           row_label=row[1]
           indx0.add(row_label)
           if len(indx0)==d0:
               break
       if r0 > half0:
           #remove d0 instances from class0
           for row_label in indx0:
               for row_label2 in class0_row_labels:
                   if row_label == row_label2:
                       class0_row_labels2.remove(row_label)
                       break
       elif r0 < half0:
           #randomly add d0 instances to class0 with replacement
          toadd0=random.sample(class0_row_labels,d0)
          class0_row_labels2=class0_row_labels2+toadd0
       ###remove or add instances from class1 or to class1
       indx1=set()
       class1_row_labels=[]
       for row in class1.itertuples():
           row_label=row[1]
           class1_row_labels.append(row_label)
       class1_row_labels2=class1_row_labels
       d1=np.abs(r1-half1)
       for row in class1.itertuples():
           row_label=row[1]
           indx1.add(row_label)
           if len(indx1)==d1:
               break
       if r1 > half1:
           #remove d1 instances from class1           
           for row_label in indx1:
               for row_label2 in class1_row_labels:
                   if row_label == row_label2:
                       class1_row_labels2.remove(row_label)
                       break
       elif r1 < half1:
           #randomly add d1 instances to class1 with replacement
           #print('class1_row_labels: '+str(len(class1_row_labels)))
           #print('d1: '+str(d1))
           toadd1=random.sample(class1_row_labels,d1)
           
           class1_row_labels2=class1_row_labels2+toadd1
   else:
       sys.exit('size of train balanced < balanced_trainset_size')
   for i in range(len(class0_row_labels2)):
       class0_row_labels2[i]=int(class0_row_labels2[i])
   for i in range(len(class1_row_labels2)):
       class1_row_labels2[i]=int(class1_row_labels2[i])
   print('class0: '+str(len(class0_row_labels2)))
   print('class1: '+str(len(class1_row_labels2)))
   return (class0_row_labels2,class1_row_labels2)
   '''

def get_model_inputs_output(datatype,data,model_inputs_output_csv):
    #create a csv file containing the input variables, the output variable and the classes of a model. This file is used to reduce a new test set in order to test the model using the testset. 
    #inputs: datatype, 'df', 'csv' or 'arff'
    #        data, df, csv file or arff file
    #output: model_inputs_output_csv
    if datatype=='csv':
        df=pd.read_csv(data)
        inputs_output=list(df.columns)        
    elif datatype=='arff':
        df=arff_to_dataframe(data)
        inputs_output=list(df.columns)
    elif datatype=='df':#data is a dataframe
        df=data
        inputs_output=list(df.columns)
    else:
        sys.exit('datatype is invalid')
    cols=list(df.columns)
    target_col=cols[len(cols)-1]
    array_classes=df[target_col].unique()#get the classes
    f=open(model_inputs_output_csv,'w+')
    for i in range(len(inputs_output)-1):#write features
        f.write(inputs_output[i]+',')
    f.write(inputs_output[len(inputs_output)-1]+'\n')
    for i in range(array_classes.size):#write the classes
        f.write(str(array_classes[i])+'\n')
    f.close()
    
def replace_missing_target_or_string_target_with_nominal_target(inputs_output_csv,outfile_arff):
        #replace missing target (?) or string target of arff_data to nominal target e.g. @attribute before37weeksCell {0,1}
        f=open(inputs_output_csv,'r')
        lines=f.readlines()
        features=lines[0].rstrip()
        l=features.split(',')
        targetVar=l[len(l)-1]
        targets=set()
        for i in [1,2]:#get the classes
            target=lines[i].rstrip()
            if target == '1' or target == '1.0':
                targets.add(1)
            elif target == '0' or target == '0.0':
                targets.add(0)
        targets=list(targets)
        targets.sort()
        targets=set(targets)
        f.close()
        data=''
        f2=open(outfile_arff,'r')
        lines=f2.readlines()
        for line in lines:
            line=line.rstrip()
            m=re.match('^(@attribute\s+\'{0,1}'+targetVar+'\'{0,1}\s+)\{{0,1}\'{0,1}(\?|string)\'{0,1}\}{0,1}$',line)#line is the target variable with values ?, '?', string or 'string'
            if m:
                data+=m.group(1)+str(targets)+'\n'
            else:
                m2=re.match('^([^\']+)\'{0,1}(\?)\'{0,1}$',line)#line is an instance with unknown class ? or '?', remove any single quotes around ?
                if m2:
                    data+=m2.group(1)+m2.group(2)+'\n'
                else:#line is an attribute definition or an instance with known class
                    data+=line+'\n'
        f2.close()
        f2=open(outfile_arff,'w')#overwrite the arff file
        f2.write(data)
        f2.close()
        
def arff_to_dataframe(arff_file):
    f=open(arff_file,'r')
    cols=[]
    data=[]
    lines=f.readlines()
    for line in lines:
        line=line.rstrip()
        m=re.match('^@attribute\s+(\'{0,1}[_/\.\-\w\^\s\']+\'{0,1})\s+.+$',line)#line is an attribute e.g. 27_EIS_Amplitude1 or x2x0^2x5
        if m:
           cols.append(m.group(1))
           #print(m.group(1))
        else:
           #m2=re.match('^([^\'@relationd'']+)$',line)#line is an instance
           m2=re.match('^([^@]+)$',line)
           if m2:
                #print(m2.group(1))
                inst=m2.group(1)
                inst=inst.split(',')
                data.append(inst)            
    f.close()
    df=pd.DataFrame(data,columns=cols)
    df=df.replace('?',np.nan) #replace any ? (missing values) with nan
    return df
    '''    
    (_,c)=df.shape
    labels=df[cols[c-1]].unique()
    labels=set(labels)
    if np.nan in labels:#some instances have unknown labels
        return df
    else:
        df=df.astype({cols[c-1]:int})#convert targets column to int
        return df
    '''
def convert_integer_type_to_discrete_type(arff_file,ordinal_encoder):
    #convert the integer type of the features of an arff file to discrete type using an ordinal encoder (list of arrays of categories)
    #input: arff_file
    #       ordinal_encoder
    #output: arff_file with the discrete types
    ranges=[]#list of no. of discrete values of each feature
    categoriesL=ordinal_encoder.categories_
    for i in range(len(categoriesL)):
        k=categoriesL[i].size
        ranges.append(k)
    f=open(arff_file,'r')
    data=[]
    lines=f.readlines()
    i=0#ith feature
    for line in lines:
        line=line.rstrip()
        m=re.match('^(@attribute\s+\'{0,1}[_\.\w\^\s\']+\'{0,1}\s+)[integrumcal]+$',line)#line is an attribute of integer type e.g. 27_EIS_Amplitude1 or x2x0^2x5
        if m:
           valsL=[j for j in range(ranges[i])]
           vals=''
           for val in valsL:
               vals+=str(val)+','
           vals=vals.rstrip(',')
           discrete_range='{'+vals+'}'
           line2=m.group(1)+discrete_range+'\n'
           data.append(line2)
           i+=1
        else:
           data.append(line+'\n')
    f.close()
    f=open(arff_file,'w')#overwrite the arff file with the discrete features
    for line in data:
        f.write(line)
    f.close()
    
def dataframe_to_arff(data,arff_file,ordinal_encoder=None,class_labels=[0,1]):
    #input: data (data frame)
    #output: arff_file
    (_,c)=data.shape
    features=list(data.columns)
    features=add_quotes(features)
    if data.iat[0,c-1]!='?':
        data=data.astype({features[c-1]:int})
    else:
        data=data.astype({features[c-1]:object})
    targets=list(set(data.iloc[:,c-1]))
    if len(targets)==1:#all instances have the same class label
        targets=class_labels #use default labels: 0, 1
    #(_,targets)=convert_targets2(data,c-1)
    data=data.values.tolist()#data becomes a list of lists (instances) of numbers
    #convert the targets from float to int e.g. 0.0 to 0 and 1.0 to 1
    i=0
    #targets=set()
    for instance in data:#collect class labels of data excluding missing labels: nan or '?'
        target=instance[len(instance)-1]
        if target == '?':
            print()
        #elif isinstance(target,str):
        #    targets.add(target)
        elif np.isnan(target)==False:
            if isinstance(target,float):
                instance[len(instance)-1]=int(target)#convert float target to int target e.g. 0.0 to 0 and 1.0 to 1
                data[i]=instance
                #targets.add(int(target))            
        elif np.isnan(target):
            instance[len(instance)-1]='?'
            data[i]=instance
        #else:#add a target of other types: int, object 
        #    targets.add(target)
        i+=1    
    #if len(targets)==0:#all instances have missing labels, use default class labels
    #    targets=class_labels
    #else:
        #targets=list(targets)
    #target=targets[0]
    #if isinstance(target,int):
    if targets==['?']:#all instances have missing labels, use default class labels
        targets=class_labels
    else:
        targets.sort()#sort classes in ascending order
    #print('targets: ',targets)
    arff.dump(arff_file,data,relation="arff_data",names=features)
    #if os.path.isfile(arff_file):
        #print(arff_file,' exists')
    #else:
    #    print(arff_file,' does not exist')
    convert_string_or_integer_targets_to_nominal_targets(arff_file,features[-1],targets,arff_file)
    #if os.path.isfile(arff_file):
    #    print(arff_file,' exists2')
    #else:
    #    print(arff_file,' does not exist2')
    if ordinal_encoder!=None:#convert each feature of integer type to discrete type using the ordinal encoder of a training set
        convert_integer_type_to_discrete_type(arff_file,ordinal_encoder)         
        #if os.path.isfile(arff_file):
        #    print(arff_file,'exists3')
        #else:
        #    print(arff_file,' does not exist3')
'''
def dataframe_to_arff2(data,arff_data):
    #create weka header
    header=[]
    header.append('@relation arff_data')
    #get the type of each feature
    (r,c)=data.shape
    for i in range(c-1):#exclude the class attribute
        if data[data.columns[i]].dtype==float:
            attr='@attribute '+add_quotes2(data.columns[i])+' numeric'
        else:#nominal attribute
            values=str(set(data[data.columns[i]].unique()))          
            attr='@attribute '+add_quotes2(data.columns[i])+' '+values
        header.append(attr)
    #convert float targets to int targets
    targets=set()
    for i in range(r):
            target=data.iloc[i,c-1]
            if isinstance(target,float):
                data.iloc[i,c-1]=int(target)#convert float target to int target
                targets.add(int(target))
            else:
                targets.add(target)
    file=open(arff_data,'w+')
    for i in range(c-1):
        file.write(header[i]+'\n')
    file.write('@attribute '+add_quotes2(data.columns[c-1])+' '+str(targets)+'\n')#write the class attribute
    file.write('@data\n')
    #write data to arff file
    for i in range(r):
        for j in range(c-1):
            file.writelines([str(data.iloc[i,j]),','])
        file.writelines([str(data.iloc[i,c-1]),'\n'])
    file.close()
    
def add_quotes2(f):
    #add single quotes to a feature f if it contains 1 or more spaces
    import re
    #m=re.match('^([^\s\']+[\w\s]+[^\s\']+)$',f)
    m=re.match('^([x\d\^\s]+)$',f)#polynomial features e.g. x1 x5 x3 x6, x7 x9 or x13^2 x15 x26
    if m:
        f='\''+m.group(1)+'\''
    return f
'''
def add_quotes(featuresL):
    #add single quotes to features containing spaces so that x0 x^2 x3 becomes 'x0 x^2 x3'
    import re
    fs=[]
    for f in featuresL:
        #m=re.match('^([^\s\']+[\w\s]+[^\s\']+)$',f)
        m=re.match('^([x\d\^\s]+)$',f)#polynomial features e.g. x1 x5 x3 x6, x7 x9 or x13^2 x15 x26
        if m:
            f='\''+m.group(1)+'\''
            fs.append(f)
        else:
            fs.append(f)
    return fs

def add_noise(trainset,datatype='df',mini=0,maxi=0.001):
    #for each instance of the original training set:
    #     draw a random noise z between mini and maxi
    #     create 2 noisy instances as follows: 
    #     noisy instance1 <- instance + z
    #     noisy instance2 <- instance - z
    if datatype=='csv':
        trainset=pd.read_csv(trainset)
    elif datatype!='df':
        sys.exit('invalid datatype in add_noise: '+datatype)
    print('add random noise between ',mini,', ',maxi)
    (r,c)=trainset.shape
    fs=c-1
    z=random.uniform(mini,maxi)
    trainset1=trainset
    trainset2=trainset
    for i in range(r):
        for j in range(fs):
            trainset1.iloc[i,j]=trainset.iloc[i,j]+z
            trainset2.iloc[i,j]=trainset.iloc[i,j]-z
    trainset3=pd.concat([trainset1,trainset2])
    return trainset3
    
def add_noise_based_on_mean_of_each_class(trainset,datatype='df',noisytrainsetfile=None,percent=10):
    #for each feature j of training set{
    #   compute mean m1 of class1
    #   compute mean m0 of class0
    #   if m1>m0 
    #   then add noise to the feature j of each class1 instance and the feature j of each class0 instance as follows:
    #        draw a random noise z from N(0,1)    
    #        feature j of class1_instance <- feature j of class1_instance + percent * z                        
    #        feature j of class0_instance <- feature j of class0_instance - percent * z
    #   else if m0>m1
    #   then add noise to the feature j of each class1 instance and the feature j of each class0 instance as follows:
    #            draw a random noise z drawn from N(0,1)    
    #            feature j of class1_instance <- feature j of class1_instance - percent * z                        
    #            feature j of class0_instance <- feature j of class0_instance + percent * z
    #   else add noise to the feature j of each class1 instance and the feature j of each class0 instance as follows:
    #            draw a random noise z is drawn from N(0,1)
    #            feature j of class1_instance <- feature j of class1_instance + percent * z                        
    #            feature j of class0_instance <- feature j of class0_instance + percent * z
    #}
    print('===add noise to training set based on mean of each class===')
    print('noise percent: ',percent,'%')
    if datatype=='csv':
        trainset=pd.read_csv(trainset)
    elif datatype=='df':
        cols=list(trainset.columns)
    else:
        sys.exit('invalid datatype in add_noise_based_on_mean_of_each_class: '+datatype)
    (mean0L,mean1L)=compute_mean(trainset)
    (_,c)=trainset.shape
    class1=trainset[trainset[cols[c-1]]==1]
    class0=trainset[trainset[cols[c-1]]==0]
    fs=len(mean0L)#no. of features
    (r1,c)=class1.shape
    (r0,_)=class0.shape
    noisy_class1=class1
    noisy_class0=class0
    for i in range(fs):
        if mean1L[i] > mean0L[i]:
               z=np.abs(np.random.normal()/5)#generate a +ve random noise between 0 and 1
               z1=z*pd.Series([1 for k in range(r1)],name=cols[i],index=class1.index)#create a column vector of noise z1
               z0=z*pd.Series([1 for k in range(r0)],name=cols[i],index=class0.index)#create a column vector of noise z0
               noisy_class1.iloc[:,i]=noisy_class1.iloc[:,i]+z1*percent/100
               noisy_class0.iloc[:,i]=noisy_class0.iloc[:,i]-z0*percent/100
        elif mean0L[i] > mean1L[i]:
               z=np.abs(np.random.normal()/5)#generate a +ve random noise between 0 and 1
               z1=z*pd.Series([1 for k in range(r1)],name=cols[i],index=class1.index)
               z0=z*pd.Series([1 for k in range(r0)],name=cols[i],index=class0.index)
               noisy_class1.iloc[:,i]=noisy_class1.iloc[:,i]-z1*percent/100
               noisy_class0.iloc[:,i]=noisy_class0.iloc[:,i]+z0*percent/100
        else:
               z=np.abs(np.random.normal()/5)#generate a +ve random noise between 0 and 1
               z1=z*pd.Series([1 for k in range(r1)],name=cols[i],index=class1.index)
               z0=z*pd.Series([1 for k in range(r0)],name=cols[i],index=class0.index)
               noisy_class1.iloc[:,i]=noisy_class1.iloc[:,i]+z1*percent/100
               noisy_class0.iloc[:,i]=noisy_class0.iloc[:,i]+z0*percent/100
    noisy_trainset=pd.concat([noisy_class1,noisy_class0])
    #rescale the noisy training set to [0,1] as follows:
    #       if a noisy value > 1, then replace it with 1
    #       if a noisy value < 0, then replace it with 0
    (r,c)=noisy_trainset.shape
    for i in range(r):
        for j in range(c-1):
            if noisy_trainset.iloc[i,j]>1:
                noisy_trainset.iloc[i,j]=1
            elif noisy_trainset.iloc[i,j]<0:
                noisy_trainset.iloc[i,j]=0                
    if noisytrainsetfile!=None:
        noisy_trainset.to_csv(noisytrainsetfile,index=False)
    return noisy_trainset
    
def add_noise_to_metabolite(trainset,percent=10):
    #decrease the feature value of each preterm instance and increase the feature value of each onterm instance
    #input: a training set dataframe
    #output: a noisy training set of same size as the original training set
    print('===add noise to metabolite data===')
    print('noise level (%): ',percent)
    (_,c)=trainset.shape
    levels=[i for i in range(c-1)]#noise levels (as percentage) of features
    for i in range(c-1):
        levels[i]=percent
    cols=list(trainset.columns)
    class1=trainset[trainset[cols[c-1]]==1]    
    class0=trainset[trainset[cols[c-1]]==0]
    for j in range(c-1):
        level = levels[j]
        class1.iloc[:,j] = class1.iloc[:,j]*(1-level/100)#means of features of preterm < means of features of onterm
        class0.iloc[:,j] = class0.iloc[:,j]*(1+level/100)
    trainset=pd.concat([class0,class1])
    return trainset
    
def add_noise_to_amp_phase(trainset):
    amp_noise=[i for i in range(14)]#noise levels for amplitude features at 14 frequencies
    amp_noise[0]=10 #frequency 1 (largest distance between preterm and onterm)
    amp_noise[1]=8
    amp_noise[2]=6
    amp_noise[3]=4
    amp_noise[4]=1
    amp_noise[5]=1
    amp_noise[6]=1
    amp_noise[7]=1
    amp_noise[8]=1
    amp_noise[9]=1
    amp_noise[10]=1
    amp_noise[11]=1
    amp_noise[12]=1
    amp_noise[13]=0
    phase_noise=[i for i in range(14)]#noise levels for phase features at 14 frequencies
    phase_noise[0]=0
    phase_noise[1]=1
    phase_noise[2]=2
    phase_noise[3]=4
    phase_noise[4]=6
    phase_noise[5]=6  
    phase_noise[6]=6
    phase_noise[7]=6
    phase_noise[8]=10 #frequency 9 (largest distance between preterm and onterm)
    phase_noise[9]=6 
    phase_noise[10]=6
    phase_noise[11]=6
    phase_noise[12]=3
    phase_noise[13]=3
    (_,c)=trainset.shape
    cols=list(trainset.columns)
    class1=trainset[trainset[cols[c-1]]==1]    
    class0=trainset[trainset[cols[c-1]]==0]
    for j in range(14):#amplitude (+ve)
        level = amp_noise[j]
        class1.iloc[:,j] = class1.iloc[:,j]*(1+level/100)#amplitude of preterm > amplitude of onterm for 14 frequencies
        class0.iloc[:,j] = class0.iloc[:,j]*(1-level/100)
    i=0
    for k in range(14,28):#phase (-ve or +ve)
        level = phase_noise[i]
        class1.iloc[:,k] = class1.iloc[:,k]*(1-level/100)#phase of preterm < phase of onterm for 14 frequencies
        class0.iloc[:,k] = class0.iloc[:,k]*(1+level/100)        
        i+=1
    trainset=pd.concat([class0,class1])
    return trainset

def convert_string_or_integer_targets_to_nominal_targets(arff_file,targetVariable,targets,arff_file2):
    #set missing targets (?) or integer targets to nominal targets in an arff file
    import re
    f=open(arff_file,'r')
    found_target_variable=False
    attrs=[]
    while found_target_variable==False:
        line=f.readline().rstrip()
        m=re.match('^(@attribute\s+'+targetVariable+'\s+)[\{\}alintegrs\d\,]+$',line)
        m2=re.match('^(@attribute\s+'+targetVariable+'\s+)\{(0|1)\}$',line) #@attribute class {0} or attribute class {1}
        if m or m2:
            found_target_variable=True
            targetsStr=''
            for target in targets[:-1]:
                targetsStr+=str(target)+','
            targetsStr+=str(targets[-1])
            target_line=m.group(1)+'{'+targetsStr+'}\n'
        else:
            m3=re.match('^(@attribute.+)$',line)
            if m3:
                if m3.group(1) not in attrs:
                   attrs.append(m3.group(1))
            else:
                m4=re.match('^@relation.+$',line)
                if m4:
                    line0=line+'\n'
    data=[]
    data.append(line0)#add @relation line at top
    for attr in attrs:#add @attribute lines
        data.append(attr+'\n')
    data.append(target_line)#add target variable line
    line=f.readline().rstrip()
    while line!='':
       m=re.match('^(.+)\'\?\'$',line)#'?' at end of line
       if m:
           line=m.group(1)+'?\n'#replace '?' with ?
       data.append(line+'\n') #add instance line
       line=f.readline().rstrip()
    f.close()
    f2=open(arff_file,'w')#overwrite the arff file with the nominmal targets
    for line in data:
        f2.write(line)
    f2.close()

def all_models(models_sorted,model1L,model2L,logfile,ppv_npv_pos_lr_neg_lr_p_value_auc_ci=False,model1='logistic regression',model2='random forest',cross_validation=None):
        #write all models in descending order of performance    
        #return, model1L and model2L sorted in descending order of train_test_auc (overall performance)
        if models_sorted==False:
            model1L.sort(key=operator.itemgetter(1),reverse=True)#sort logistic regression models in descending order of train_test_auc (overall performance)
            model2L.sort(key=operator.itemgetter(1),reverse=True)#sort random forest models in descending order of train_test_auc (overall performance)
        if os.path.isfile(logfile)==False:#logfile does not exist, create a new one
            file=open(logfile,'w+')
        else:
            file=open(logfile,'a')
        if cross_validation!=None:
                file.write('\t\t\t          \t\t\t          \tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                print('\t\t\t          \t\t\t          \tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                for i in range(len(model2L)):
                    mod2=model2L[i]
                    iteration=mod2[0]
                    train_auc=mod2[2]
                    train_tpr=mod2[3]
                    train_tnr=mod2[4]
                    train_fpr=mod2[5]
                    train_fnr=mod2[6]
                    xval_auc=mod2[7]
                    if iteration!='none':
                        file.write('\t\t\titeration: '+str(iteration)+'\t'+model2+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')
                        print('\t\t\titeration: '+str(iteration)+'\t'+model2+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3)))            
                    else:#no iterations
                        file.write('\t\t\t\t\t'+model2+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')                
                        print('\t\t\t\t\t'+model2+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3)))                
                    file.write('\t\t\t\t\t'+model2+' ('+str(cross_validation)+'-fold CV)'+'\t'+str(np.round(xval_auc,3))+'\n')
                    print('\t\t\t\t\t'+model2+' ('+str(cross_validation)+'-fold CV)'+'\t'+str(np.round(xval_auc,3)))           
                for i in range(len(model1L)):    
                    mod1=model1L[i]
                    iteration=mod1[0]            
                    train_auc=mod1[2]
                    train_tpr=mod1[3]
                    train_tnr=mod1[4]
                    train_fpr=mod1[5]
                    train_fnr=mod1[6]
                    xval_auc=mod1[7]
                    if iteration!='none':
                        file.write('\t\t\titeration: '+str(iteration)+'\t'+model1+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')
                        print('\t\t\titeration: '+str(iteration)+'\t'+model1+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3)))
                    else:#no iterations
                        file.write('\t\t\t\t\t'+model1+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')
                        print('\t\t\t\t\t'+model1+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3)))
                    file.write('\t\t\t\t\t'+model1+' ('+str(cross_validation)+'-fold CV)'+' '+str(np.round(xval_auc,3))+'\n')
                    print('\t\t\t\t\t'+model1+' ('+str(cross_validation)+'-fold CV)'+' '+str(np.round(xval_auc,3)))
        else:#training and testing
                #(i,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)
                #(0,1             ,2        ,3        ,4        ,5        ,6        ,7       ,8       ,9       ,10      ,11      )
                file.write('\t\t\t          \t\t\t          \tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                print('\t\t\t          \t\t\t          \tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                for i in range(len(model2L)):
                    mod2=model2L[i]
                    iteration=mod2[0]
                    train_auc=mod2[2]
                    train_tpr=mod2[3]
                    train_tnr=mod2[4]
                    train_fpr=mod2[5]
                    train_fnr=mod2[6]
                    test_auc=mod2[7]
                    test_tpr=mod2[8]
                    test_tnr=mod2[9]
                    test_fpr=mod2[10]
                    test_fnr=mod2[11]
                    if iteration!='none':
                        file.write('\t\t\titeration: '+str(iteration)+'\t'+model2+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')
                        print('\t\t\titeration: '+str(iteration)+'\t'+model2+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3)))            
                    else:#no iterations
                        file.write('\t\t\t\t\t'+model2+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')                
                        print('\t\t\t\t\t'+model2+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3)))                
                    file.write('\t\t\t\t\t'+model2+' (testing)'+'\t\t'+str(np.round(test_auc,3))+'\t'+str(np.round(test_tpr,3))+'\t\t\t'+str(np.round(test_tnr,3))+'\t\t\t'+str(np.round(test_fpr,3))+'\t'+str(np.round(test_fnr,3))+'\n')
                    print('\t\t\t\t\t'+model2+' (testing)'+'\t\t'+str(np.round(test_auc,3))+'\t'+str(np.round(test_tpr,3))+'\t\t\t'+str(np.round(test_tnr,3))+'\t\t\t'+str(np.round(test_fpr,3))+'\t'+str(np.round(test_fnr,3)))           
                file.write('\t\t\t          \t\t\t          \tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                print('\t\t\t          \t\t\t          \tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                for i in range(len(model1L)):    
                    mod1=model1L[i]
                    iteration=mod1[0]            
                    train_auc=mod1[2]
                    train_tpr=mod1[3]
                    train_tnr=mod1[4]
                    train_fpr=mod1[5]
                    train_fnr=mod1[6]
                    test_auc=mod1[7]
                    test_tpr=mod1[8]
                    test_tnr=mod1[9]
                    test_fpr=mod1[10]
                    test_fnr=mod1[11]
                    if iteration!='none':
                        file.write('\t\t\titeration: '+str(iteration)+'\t'+model1+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')
                        print('\t\t\titeration: '+str(iteration)+'\t'+model1+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3)))
                    else:#no iterations
                        file.write('\t\t\t\t\t'+model1+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')
                        print('\t\t\t\t\t'+model1+' (training)'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3)))
                    file.write('\t\t\t\t\t'+model1+' (testing)'+'\t'+str(np.round(test_auc,3))+'\t'+str(np.round(test_tpr,3))+'\t\t\t'+str(np.round(test_tnr,3))+'\t\t\t'+str(np.round(test_fpr,3))+'\t'+str(np.round(test_fnr,3))+'\n')
                    print('\t\t\t\t\t'+model1+' (testing)'+'\t'+str(np.round(test_auc,3))+'\t'+str(np.round(test_tpr,3))+'\t\t\t'+str(np.round(test_tnr,3))+'\t\t\t'+str(np.round(test_fpr,3))+'\t'+str(np.round(test_fnr,3)))
        file.close()
        return (model1L,model2L)

def all_models_of_nn(models_sorted,L,logfile,outputs):
         #all models of neural network
         #return, L sorted in descending order of train_test_auc
        if models_sorted==False:
            L.sort(key=operator.itemgetter(1),reverse=True)#sort models in descending order of train_test_auc
        if os.path.isfile(logfile)==False:#logfile does not exist, create a new one
            file=open(logfile,'w+')
        else:
            file=open(logfile,'a')
        file.write('\t\t\t====the Models of All the Iterations Finished So Far====\n')
        print('\t\t\t====the Models of All the Iterations Finished So Far====\n')
        if outputs == 1 or outputs == 2:
            #(i,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,validset_auc     ,topology, weights)
            #(0,1             ,2        ,3        ,4        ,5        ,6        ,7       ,8       ,9       ,10      ,11      , 12              ,13      ,14             )
            file.write('\t\t\t          \t\tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR \ttopology \tweights\n')
            print('\t\t\t            \t\tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR \ttopology \tweights\n')
            for i in range(len(L)):
                model=L[i]
                iteration=model[0]
                train_auc=model[2]
                train_tpr=model[3]
                train_tnr=model[4]
                train_fpr=model[5]
                train_fnr=model[6]
                test_auc=model[7]
                test_tpr=model[8]
                test_tnr=model[9]
                test_fpr=model[10]
                test_fnr=model[11]
                validset_auc=model[12]
                topology=model[13]
                weights=model[14]
                if iteration!='none':
                    file.write('\t\t\titeration: '+str(iteration)+'\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t'+str(np.round(train_tnr,3))+'\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\t'+str(topology)+'\t'+str(weights)+'\n')
                    file.write('\t\t\t \t\tvalidation auc: '+str(np.round(validset_auc,3))+'\n')
                    print('\t\t\titeration: '+str(iteration)+'\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t'+str(np.round(train_tnr,3))+'\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\t'+str(topology)+'\t'+str(weights))            
                    print('\t\t\t \t\tvalidation auc: '+str(np.round(validset_auc,3)))                
                else:#no iterations
                    file.write('\t\t\t\t\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t'+str(np.round(train_tnr,3))+'\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\t'+str(validset_auc)+' (va\t'+str(topology)+' (topology) \t'+str(weights)+' (weights)\n')                
                    file.write('\t\t\t \t\tvalidation auc: '+str(np.round(validset_auc,3))+'\n')
                    print('\t\t\t\t\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t'+str(np.round(train_tnr,3))+'\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\t'+str(topology)+' (topology) \t'+str(weights)+' (weights)')                
                    print('\t\t\t \t\tvalidation auc: '+str(np.round(validset_auc,3)))            
                file.write('\t\t\t\t\ttesting'+'\t\t'+str(np.round(test_auc,3))+'\t'+str(np.round(test_tpr,3))+'\t'+str(np.round(test_tnr,3))+'\t'+str(np.round(test_fpr,3))+'\t'+str(np.round(test_fnr,3))+'\n')
                print('\t\t\t\t\ttesting'+'\t\t'+str(np.round(test_auc,3))+'\t'+str(np.round(test_tpr,3))+'\t'+str(np.round(test_tnr,3))+'\t'+str(np.round(test_fpr,3))+'\t'+str(np.round(test_fnr,3)))           
            file.close()
        else:#multi class
            #L=list of (i,train_test_recall,train_p,test_p,validset_p,nn_topology,nn_weights)
            #           0,                1,      2,     3,         4,          5,        6
            file.write('\t\t\t          \t\ttraining recall\ttesting recall \tvalidation recall \ttopology \tweights\n')
            print('\t\t\t            \t\ttraining recall\ttesting recall \tvalidation recall \ttopology \tweights\n')
            for i in range(len(L)):
               model=L[i]
               iteration=model[0]
               train_p=model[2]
               avg=train_p['macro avg']
               train_recall=avg['recall']
               test_p=model[3]
               avg=test_p['macro avg']
               test_recall=avg['recall']
               validset_p=model[4]
               avg=validset_p['macro avg']
               validset_recall=avg['recall']
               topology=model[5]
               weights=model[6]
               if iteration!='none':
                   file.write('\t\t\titeration: '+str(iteration)+'\t'+str(np.round(train_recall,3))+'\t'+str(np.round(test_recall,3))+'\t'+str(np.round(validset_recall,3))+'\t'+str(topology)+'\t'+str(weights)+'\n')
                   print('\t\t\titeration: '+str(iteration)+'\t'+str(np.round(train_recall,3))+'\t'+str(np.round(test_recall,3))+'\t'+str(np.round(validset_recall,3))+'\t'+str(topology)+'\t'+str(weights)+'\n')
               else:
                   file.write('\t\t\t\t'+str(np.round(train_recall,3))+'\t'+str(np.round(test_recall,3))+'\t'+str(np.round(validset_recall,3))+'\t'+str(topology)+'\t'+str(weights)+'\n')
                   print('\t\t\t\t'+str(np.round(train_recall,3))+'\t'+str(np.round(test_recall,3))+'\t'+str(np.round(validset_recall,3))+'\t'+str(topology)+'\t'+str(weights)+'\n')
            file.close()                    
        return L
        
def summarize_results(logfile,model1_performance,model2_performance,model1L,model1_train_aucL,model1_test_aucL,model2_L,model2_train_aucL,model2_test_aucL,model1='logistic regression',model2='random forest', cross_validation=None):    
    model1L.append(model1_performance)
    model1_train_aucL.append(float(model1_performance[2]))
    model1_test_aucL.append(float(model1_performance[7]))
    model2_L.append(model2_performance)
    model2_train_aucL.append(float(model2_performance[2]))  
    model2_test_aucL.append(float(model2_performance[7]))
    iteration=int(model1_performance[0])
    if cross_validation!=None:#training and CV performance
        #model1_performance=(iteration,train_xval_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,xval_auc,-999,-999,-999,-999)
        #model2_performance=(iteration,train_xval_auc2,train_auc2,train_tpr2,train_tnr2,train_fpr2,train_fnr2,xval_auc2,-999,-999,-999,-999)                 
        if iteration==0:
            (model1L,model2_L)=all_models(False,model1L,model2_L,logfile,model1=model1,model2=model2,cross_validation=cross_validation)#write all models so far to logfile
        else:
                    (model1L,model2_L)=all_models(False,model1L,model2_L,logfile,model1=model1,model2=model2,cross_validation=cross_validation)#write all models so far to logfile
                    #best_model(True,model1L,model2_L,logfile,model1=model1,model2=model2,cross_validation=cross_validation)#write best model to logfile    
                    print('\t\t\t========='+model1+'=========')
                    print('\t\t\tmean training AUC: '+str(np.mean(model1_train_aucL)))
                    print('\t\t\tmini training AUC: '+str(np.min(model1_train_aucL)))
                    print('\t\t\tmax training AUC: '+str(np.max(model1_train_aucL)))
                    print('\t\t\tmean '+str(cross_validation)+'-fold CV AUC: '+str(np.mean(model1_test_aucL)))
                    print('\t\t\tmini '+str(cross_validation)+'-fold CV AUC: '+str(np.min(model1_test_aucL)))
                    print('\t\t\tmax '+str(cross_validation)+'-fold CV AUC: '+str(np.max(model1_test_aucL)))
                    print('\t\t\t========='+model2+'=========')
                    print('\t\t\tmean training AUC: '+str(np.mean(model2_train_aucL)))
                    print('\t\t\tmini training AUC: '+str(np.min(model2_train_aucL)))
                    print('\t\t\tmax training AUC: '+str(np.max(model2_train_aucL)))
                    print('\t\t\tmean '+str(cross_validation)+'-fold CV AUC: '+str(np.mean(model2_test_aucL)))
                    print('\t\t\tmini '+str(cross_validation)+'-fold CV AUC: '+str(np.min(model2_test_aucL)))
                    print('\t\t\tmax '+str(cross_validation)+'-fold CV AUC: '+str(np.max(model2_test_aucL)))
                    file=open(logfile,'a')
                    file.write('\n\t\t\t========='+model1+'=========\n')
                    file.write('\t\t\tmean training AUC: '+str(np.mean(model1_train_aucL))+'\n')
                    file.write('\t\t\tmini training AUC: '+str(np.min(model1_train_aucL))+'\n')
                    file.write('\t\t\tmax training AUC: '+str(np.max(model1_train_aucL))+'\n')
                    file.write('\t\t\tmean '+str(cross_validation)+'-fold CV AUC: '+str(np.mean(model1_test_aucL))+'\n')
                    file.write('\t\t\tmini '+str(cross_validation)+'-fold CV AUC: '+str(np.min(model1_test_aucL))+'\n')
                    file.write('\t\t\tmax '+str(cross_validation)+'-fold CV AUC: '+str(np.max(model1_test_aucL))+'\n')
                    file.write('\t\t\t========='+model2+'=========\n')
                    file.write('\t\t\tmean training AUC: '+str(np.mean(model2_train_aucL))+'\n')
                    file.write('\t\t\tmini training AUC: '+str(np.min(model2_train_aucL))+'\n')
                    file.write('\t\t\tmax training AUC: '+str(np.max(model2_train_aucL))+'\n')
                    file.write('\t\t\tmean '+str(cross_validation)+'-fold CV AUC: '+str(np.mean(model2_test_aucL))+'\n')
                    file.write('\t\t\tmini '+str(cross_validation)+'-fold CV AUC: '+str(np.min(model2_test_aucL))+'\n')
                    file.write('\t\t\tmax '+str(cross_validation)+'-fold CV AUC: '+str(np.max(model2_test_aucL))+'\n')
                    file.close()
    else:#training and testing performance
        #model1_performance=(i,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)
        #model1_performance=(0,1             ,2        ,3        ,4        ,5        ,6        ,7       ,8       ,9       ,10      ,11      )
        if iteration==0:
            (model1L,model2_L)=all_models(False,model1L,model2_L,logfile,model1=model1,model2=model2)#write all models so far to logfile
            best_model(True,model1L,model2_L,logfile,model1=model1,model2=model2)#write best model to logfile    
        else:#more than 1 iterations done, get the best model so far
                    (model1L,model2_L)=all_models(False,model1L,model2_L,logfile,model1=model1,model2=model2)#write all models so far to logfile
                    best_model(True,model1L,model2_L,logfile,model1=model1,model2=model2)#write best model to logfile    
                    print('\t\t\t========='+model1+'=========')
                    print('\t\t\tmean training AUC: '+str(np.mean(model1_train_aucL)))
                    print('\t\t\tmini training AUC: '+str(np.min(model1_train_aucL)))
                    print('\t\t\tmax training AUC: '+str(np.max(model1_train_aucL)))
                    print('\t\t\tmean test AUC: '+str(np.mean(model1_test_aucL)))
                    print('\t\t\tmini test AUC: '+str(np.min(model1_test_aucL)))
                    print('\t\t\tmax test AUC: '+str(np.max(model1_test_aucL)))
                    print('\t\t\t========='+model2+'=========')
                    print('\t\t\tmean training AUC: '+str(np.mean(model2_train_aucL)))
                    print('\t\t\tmini training AUC: '+str(np.min(model2_train_aucL)))
                    print('\t\t\tmax training AUC: '+str(np.max(model2_train_aucL)))
                    print('\t\t\tmean test AUC: '+str(np.mean(model2_test_aucL)))
                    print('\t\t\tmini test AUC: '+str(np.min(model2_test_aucL)))
                    print('\t\t\tmax test AUC: '+str(np.max(model2_test_aucL)))
                    file=open(logfile,'a')
                    file.write('\t\t\t========='+model1+'=========\n')
                    file.write('\t\t\tmean training AUC: '+str(np.mean(model1_train_aucL))+'\n')
                    file.write('\t\t\tmini training AUC: '+str(np.min(model1_train_aucL))+'\n')
                    file.write('\t\t\tmax training AUC: '+str(np.max(model1_train_aucL))+'\n')
                    file.write('\t\t\tmean test AUC: '+str(np.mean(model1_test_aucL))+'\n')
                    file.write('\t\t\tmini test AUC: '+str(np.min(model1_test_aucL))+'\n')
                    file.write('\t\t\tmax test AUC: '+str(np.max(model1_test_aucL))+'\n')
                    file.write('\t\t\t========='+model2+'=========\n')
                    file.write('\t\t\tmean training AUC: '+str(np.mean(model2_train_aucL))+'\n')
                    file.write('\t\t\tmini training AUC: '+str(np.min(model2_train_aucL))+'\n')
                    file.write('\t\t\tmax training AUC: '+str(np.max(model2_train_aucL))+'\n')
                    file.write('\t\t\tmean test AUC: '+str(np.mean(model2_test_aucL))+'\n')
                    file.write('\t\t\tmini test AUC: '+str(np.min(model2_test_aucL))+'\n')
                    file.write('\t\t\tmax test AUC: '+str(np.max(model2_test_aucL))+'\n')                  
                    file.close()
    return (model1L,model1_train_aucL,model1_test_aucL,model2_L,model2_train_aucL,model2_test_aucL)

def summarize_results_of_2_classifiers(iteration,logfile,classifiers,classifier1_performance,classifier2_performance,classifier1L,classifier1_train_auc,classifier1_test_auc,classifier2L,classifier2_train_auc,classifier2_test_auc,cross_validation=None):    
    #summarize results of any 2 classifiers
    classifier1=classifiers[0]#name of classifier1
    classifier2=classifiers[1]#name of classifer2
    if cross_validation!=None:
        train_xval_auc=classifier1_performance[1]
        train_auc=classifier1_performance[2]
        train_tpr=classifier1_performance[3]
        train_tnr=classifier1_performance[4]
        train_fpr=classifier1_performance[5]
        train_fnr=classifier1_performance[6]
        xval_auc=classifier1_performance[7]
        classifier1L.append((int(iteration),train_xval_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,xval_auc))
        classifier1_train_auc.append(float(train_auc))
        classifier1_test_auc.append(float(xval_auc))
        train_xval_auc2=classifier2_performance[1]
        train_auc2=classifier2_performance[2]
        train_tpr2=classifier2_performance[3]
        train_tnr2=classifier2_performance[4]
        train_fpr2=classifier2_performance[5]
        train_fnr2=classifier2_performance[6]
        xval_auc2=classifier2_performance[7]
        classifier2L.append((int(iteration),train_xval_auc2,train_auc2,train_tpr2,train_tnr2,train_fpr2,train_fnr2,xval_auc2))
        classifier2_train_auc.append(float(train_auc2))  
        classifier2_test_auc.append(float(xval_auc2))
        (classifier1L,classifier2L)=all_models(False,classifier1L,classifier2L,logfile,cross_validation=cross_validation)#write all models so far to logfile
        if iteration>0:
                    print('\t\t\t========='+classifier1+'=========')
                    print('\t\t\tmean training AUC: '+str(np.mean(classifier1_train_auc)))
                    print('\t\t\tmini training AUC: '+str(np.min(classifier1_train_auc)))
                    print('\t\t\tmax training AUC: '+str(np.max(classifier1_train_auc)))
                    print('\t\t\tmean '+str(cross_validation)+'-fold CV AUC: '+str(np.mean(classifier1_test_auc)))
                    print('\t\t\tmini '+str(cross_validation)+'-fold CV AUC: '+str(np.min(classifier1_test_auc)))
                    print('\t\t\tmax '+str(cross_validation)+'-fold CV AUC: '+str(np.max(classifier1_test_auc)))
                    print('\t\t\t========='+classifier2+'=========')
                    print('\t\t\tmean training AUC: '+str(np.mean(classifier2_train_auc)))
                    print('\t\t\tmini training AUC: '+str(np.min(classifier2_train_auc)))
                    print('\t\t\tmax training AUC: '+str(np.max(classifier2_train_auc)))
                    print('\t\t\tmean '+str(cross_validation)+'-fold CV AUC: '+str(np.mean(classifier2_test_auc)))
                    print('\t\t\tmini '+str(cross_validation)+'-fold CV AUC: '+str(np.min(classifier2_test_auc)))
                    print('\t\t\tmax '+str(cross_validation)+'-fold CV AUC: '+str(np.max(classifier2_test_auc)))
                    file=open(logfile,'a')
                    file.write('\n\t\t\t========='+classifier1+'=========\n')
                    file.write('\t\t\tmean training AUC: '+str(np.mean(classifier1_train_auc))+'\n')
                    file.write('\t\t\tmini training AUC: '+str(np.min(classifier1_train_auc))+'\n')
                    file.write('\t\t\tmax training AUC: '+str(np.max(classifier1_train_auc))+'\n')
                    file.write('\t\t\tmean '+str(cross_validation)+'-fold CV AUC: '+str(np.mean(classifier1_test_auc))+'\n')
                    file.write('\t\t\tmini '+str(cross_validation)+'-fold CV AUC: '+str(np.min(classifier1_test_auc))+'\n')
                    file.write('\t\t\tmax '+str(cross_validation)+'-fold CV AUC: '+str(np.max(classifier1_test_auc))+'\n')
                    file.write('\t\t\t========='+classifier2+'=========\n')
                    file.write('\t\t\tmean training AUC: '+str(np.mean(classifier2_train_auc))+'\n')
                    file.write('\t\t\tmini training AUC: '+str(np.min(classifier2_train_auc))+'\n')
                    file.write('\t\t\tmax training AUC: '+str(np.max(classifier2_train_auc))+'\n')
                    file.write('\t\t\tmean '+str(cross_validation)+'-fold CV AUC: '+str(np.mean(classifier2_test_auc))+'\n')
                    file.write('\t\t\tmini '+str(cross_validation)+'-fold CV AUC: '+str(np.min(classifier2_test_auc))+'\n')
                    file.write('\t\t\tmax '+str(cross_validation)+'-fold CV AUC: '+str(np.max(classifier2_test_auc))+'\n')
                    file.close()
                    return
    else:
        #(i,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)
        #(0,1             ,2        ,3        ,4        ,5        ,6        ,7       ,8       ,9       ,10      ,11      )
        #(_,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)=log_reg_performance
        train_test_auc=classifier1_performance[1]
        train_auc=classifier1_performance[2]
        train_tpr=classifier1_performance[3]
        train_tnr=classifier1_performance[4]
        train_fpr=classifier1_performance[5]
        train_fnr=classifier1_performance[6]
        test_auc=classifier1_performance[7]
        test_tpr=classifier1_performance[8]
        test_tnr=classifier1_performance[9]
        test_fpr=classifier1_performance[10]
        test_fnr=classifier1_performance[11]
        classifier1L.append((int(iteration),train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr))
        classifier1_train_auc.append(float(train_auc))
        classifier1_test_auc.append(float(test_auc))
        #(_,train_test_auc2,train_auc2,train_tpr2,train_tnr2,train_fpr2,train_fnr2,test_auc2,test_tpr2,test_tnr2,test_fpr2,test_fnr2)=classifier2_performance
        train_test_auc2=classifier2_performance[1]
        train_auc2=classifier2_performance[2]
        train_tpr2=classifier2_performance[3]
        train_tnr2=classifier2_performance[4]
        train_fpr2=classifier2_performance[5]
        train_fnr2=classifier2_performance[6]
        test_auc2=classifier2_performance[7]
        test_tpr2=classifier2_performance[8]
        test_tnr2=classifier2_performance[9]
        test_fpr2=classifier2_performance[10]
        test_fnr2=classifier2_performance[11]
        classifier2L.append((int(iteration),train_test_auc2,train_auc2,train_tpr2,train_tnr2,train_fpr2,train_fnr2,test_auc2,test_tpr2,test_tnr2,test_fpr2,test_fnr2))
        classifier2_train_auc.append(float(train_auc2))
        classifier2_test_auc.append(float(test_auc2))                
        if iteration>0:#more than 1 iterations done, get the best model so far
                        (classifier1L,classifier2L)=all_models(False,classifier1L,classifier2L,logfile)#write all models so far to logfile
                        best_model(True,classifier1L,classifier2L,logfile)#write best model to logfile    
                        print('\t\t\t========='+classifier1+'=========')
                        print('\t\t\tmean training AUC: '+str(np.mean(classifier1_train_auc)))
                        print('\t\t\tmini training AUC: '+str(np.min(classifier1_train_auc)))
                        print('\t\t\tmax training AUC: '+str(np.max(classifier1_train_auc)))
                        print('\t\t\tmean test AUC: '+str(np.mean(classifier1_test_auc)))
                        print('\t\t\tmini test AUC: '+str(np.min(classifier1_test_auc)))
                        print('\t\t\tmax test AUC: '+str(np.max(classifier1_test_auc)))
                        print('\t\t\t========='+classifier2+'=========')
                        print('\t\t\tmean training AUC: '+str(np.mean(classifier2_train_auc)))
                        print('\t\t\tmini training AUC: '+str(np.min(classifier2_train_auc)))
                        print('\t\t\tmax training AUC: '+str(np.max(classifier2_train_auc)))
                        print('\t\t\tmean test AUC: '+str(np.mean(classifier2_test_auc)))
                        print('\t\t\tmini test AUC: '+str(np.min(classifier2_test_auc)))
                        print('\t\t\tmax test AUC: '+str(np.max(classifier2_test_auc)))
                        file=open(logfile,'a')
                        file.write('\t\t\t========='+classifier1+'=========\n')
                        file.write('\t\t\tmean training AUC: '+str(np.mean(classifier1_train_auc))+'\n')
                        file.write('\t\t\tmini training AUC: '+str(np.min(classifier1_train_auc))+'\n')
                        file.write('\t\t\tmax training AUC: '+str(np.max(classifier1_train_auc))+'\n')
                        file.write('\t\t\tmean test AUC: '+str(np.mean(classifier1_test_auc))+'\n')
                        file.write('\t\t\tmini test AUC: '+str(np.min(classifier1_test_auc))+'\n')
                        file.write('\t\t\tmax test AUC: '+str(np.max(classifier1_test_auc))+'\n')
                        file.write('\t\t\t========='+classifier2+'=========\n')
                        file.write('\t\t\tmean training AUC: '+str(np.mean(classifier2_train_auc))+'\n')
                        file.write('\t\t\tmini training AUC: '+str(np.min(classifier2_train_auc))+'\n')
                        file.write('\t\t\tmax training AUC: '+str(np.max(classifier2_train_auc))+'\n')
                        file.write('\t\t\tmean test AUC: '+str(np.mean(classifier2_test_auc))+'\n')
                        file.write('\t\t\tmini test AUC: '+str(np.min(classifier2_test_auc))+'\n')
                        file.write('\t\t\tmax test AUC: '+str(np.max(classifier2_test_auc))+'\n')                  
                        file.close()
        else:
                       (classifier1L,classifier2L)=all_models(False,classifier1L,classifier2L,logfile)#write all models so far to logfile
        return (classifier1L,classifier1_train_auc,classifier1_test_auc,classifier2L,classifier2_train_auc,classifier2_test_auc)

def summarize_results_nn(logfile,performance,L,train_performanceL,test_performanceL,outputs):    
    if outputs ==1 or outputs == 2:
        #performance=(i,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,validset_auc,nn_topology,nn_weights)
        #            (0,             1,        2,        3,        4,        5,        6,        7,       8,      9,      10,      11,          12,         13,        14)
        iteration=performance[0]
        train_auc=performance[2]
        test_auc=performance[7]
        L.append(performance)
        train_aucL=train_performanceL
        test_aucL=test_performanceL
        train_aucL.append(float(train_auc))
        test_aucL.append(float(test_auc))     
        if iteration>0:#more than 1 iterations done, get the best model so far
                    L=all_models_of_nn(False,L,logfile,outputs)#write all models so far to logfile
                    best_model_of_nn(True,L,logfile,outputs)#write best model so far to logfile    
                    print('\t\t\t=========Neural Networks=========')
                    print('\t\t\tmean training AUC: '+str(np.mean(train_aucL)))
                    print('\t\t\tmini training AUC: '+str(np.min(train_aucL)))
                    print('\t\t\tmax training AUC: '+str(np.max(train_aucL)))
                    print('\t\t\tmean test AUC: '+str(np.mean(test_aucL)))
                    print('\t\t\tmini test AUC: '+str(np.min(test_aucL)))
                    print('\t\t\tmax test AUC: '+str(np.max(test_aucL)))
                    file=open(logfile,'a')
                    file.write('\t\t\t=========Neural Networks=========\n')
                    file.write('\t\t\tmean training AUC: '+str(np.mean(train_aucL))+'\n')
                    file.write('\t\t\tmini training AUC: '+str(np.min(train_aucL))+'\n')
                    file.write('\t\t\tmax training AUC: '+str(np.max(train_aucL))+'\n')
                    file.write('\t\t\tmean test AUC: '+str(np.mean(test_aucL))+'\n')
                    file.write('\t\t\tmini test AUC: '+str(np.min(test_aucL))+'\n')
                    file.write('\t\t\tmax test AUC: '+str(np.max(test_aucL))+'\n')
                    file.close()
        else:
                    L=all_models_of_nn(False,L,logfile,outputs)#write all models so far to logfile
                    best_model_of_nn(True,L,logfile,outputs)#write best model so far to logfile  
    else:
        #performance=(i,train_test_recall,train_recall,test_recall,validset_recall,nn_topology,nn_weights)
        #             0,                1,           2,          3,              4,          5,        6
        iteration=performance[0]
        train_recall=performance[2]
        test_recall=performance[3]
        L.append(performance)
        train_recallL=train_performanceL
        test_recallL=test_performanceL
        train_recallL.append(float(train_recall))
        test_recallL.append(float(test_recall)) 
        if iteration>0:#more than 1 iterations done, get the best model so far
                    L=all_models_of_nn(False,L,logfile,outputs)#write all models so far to logfile
                    best_model_of_nn(True,L,logfile,outputs)#write best model so far to logfile    
                    print('\t\t\t=========Neural Networks=========')
                    print('\t\t\tmean training recall: '+str(np.mean(train_recallL)))
                    print('\t\t\tmini training recall: '+str(np.min(train_recallL)))
                    print('\t\t\tmax training recall: '+str(np.max(train_recallL)))
                    print('\t\t\tmean test recall: '+str(np.mean(test_recallL)))
                    print('\t\t\tmini test recall: '+str(np.min(test_recallL)))
                    print('\t\t\tmax test recall: '+str(np.max(test_recallL)))
                    file=open(logfile,'a')
                    file.write('\t\t\t=========Neural Networks=========\n')
                    file.write('\t\t\tmean training recall: '+str(np.mean(train_recallL))+'\n')
                    file.write('\t\t\tmini training recall: '+str(np.min(train_recallL))+'\n')
                    file.write('\t\t\tmax training recall: '+str(np.max(train_recallL))+'\n')
                    file.write('\t\t\tmean test recall: '+str(np.mean(test_recallL))+'\n')
                    file.write('\t\t\tmini test recall: '+str(np.min(test_recallL))+'\n')
                    file.write('\t\t\tmax test recall: '+str(np.max(test_recallL))+'\n')
                    file.close()
        else:
                    L=all_models_of_nn(False,L,logfile,outputs)#write all models so far to logfile
                    best_model_of_nn(True,L,logfile,outputs)#write best model so far to logfile  
    if outputs == 1 or outputs == 2:
        return (L,train_aucL,test_aucL)
    else:
        return (L,train_recallL,test_recallL)
    
def best_model_of_nn(models_sorted,L,logfile,outputs):
        #return: L sorted in descending order of train_test_auc
        if models_sorted==False:
            L.sort(key=operator.itemgetter(1),reverse=True)#sort network models in descending order of overall performance
        best_model=L[0]
        if os.path.isfile(logfile)==False:#logfile does not exist, create a new one
           file=open(logfile,'w+')
        else:
           file=open(logfile,'a')        
        if outputs == 1 or outputs == 2:
            #           (0,1             ,2        ,3        ,4        ,5        ,6        ,7       ,8       ,9       ,10      ,11      ,12          ,13         ,14        )
            #best_model=(i,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,validset_auc,nn_topology,nn_weights)
            if best_model[0]!='none':#more than 1 iterations               
                file.write('\t\t\t*best model*: neural network of iteration: '+str(best_model[0])+', model file: nn'+str(best_model[0])+'.h5, training AUC: '+str(np.round(best_model[2],3))+', test AUC: '+str(np.round(best_model[7],3))+', validation AUC: '+str(np.round(best_model[12],3))+', topology: '+str(best_model[13])+', weigts: '+str(best_model[14])+'\n')
                print('\t\t\t*best model*: neural network of iteration: '+str(best_model[0])+', model file: nn'+str(best_model[0])+'.h5, training AUC: '+str(np.round(best_model[2],3))+', test AUC: '+str(np.round(best_model[7],3))+', validation AUC: '+str(np.round(best_model[12],3))+', topology: '+str(best_model[13])+', weigts: '+str(best_model[14]))
            else:#1 iteration
                file.write('\t\t\t*best model*: neural network: '+str(best_model[0])+', model file: nn'+str(best_model[0])+'.h5, training AUC: '+str(np.round(best_model[2],3))+', test AUC: '+str(np.round(best_model[7],3))+', validation AUC: '+str(np.round(best_model[12],3))+', topology: '+str(best_model[13])+', weigts: '+str(best_model[14])+'\n')
                print('\t\t\t*best model*: neural network: '+str(best_model[0])+', model file: nn'+str(best_model[0])+'.h5, training AUC: '+str(np.round(best_model[2],3))+', test AUC: '+str(np.round(best_model[7],3))+', validation AUC: '+str(np.round(best_model[12],3))+', topology: '+str(best_model[13])+', weigts: '+str(best_model[14]))
            file.write('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model[3],3))+', training TNR: '+str(np.round(best_model[4],3))+', testing TPR: '+str(np.round(best_model[8],3))+', testing TNR: '+str(np.round(best_model[9],3))+'\n')
            file.write('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model[5],3))+', training FNR: '+str(np.round(best_model[6],3))+', testing FPR: '+str(np.round(best_model[10],3))+', testing FNR: '+str(np.round(best_model[11],3))+'\n')   
            print('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model[3],3))+', training TNR: '+str(np.round(best_model[4],3))+', testing TPR: '+str(np.round(best_model[8],3))+', testing TNR: '+str(np.round(best_model[9],3)))
            print('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model[5],3))+', training FNR: '+str(np.round(best_model[6],3))+', testing FPR: '+str(np.round(best_model[10],3))+', testing FNR: '+str(np.round(best_model[11],3)))   
            file.close()
        else:
            #            0,                1,      2,     3,         4,          5,        6  
            #best_model=(i,train_test_recall,train_p,test_p,validset_p,nn_topology,nn_weights)
            if best_model[0]!='none':#more than 1 iterations
                file.write('\t\t\t*best model*: neural network of iteration: '+str(best_model[0])+', model file: nn'+str(best_model[0])+'.h5, topology: '+str(best_model[5])+', weights: '+str(best_model[6])+'\n')
                print('\t\t\t*best model*: neural network of iteration: '+str(best_model[0])+', model file: nn'+str(best_model[0])+'.h5, topology: '+str(best_model[5])+', weights: '+str(best_model[6]))
            else:
                file.write('\t\t\t*best model*: neural network: '+str(best_model[0])+', model file: nn'+str(best_model[0])+'.h5, topology: '+str(best_model[5])+', weights: '+str(best_model[6])+'\n')
                print('\t\t\t*best model*: neural network: '+str(best_model[0])+', model file: nn'+str(best_model[0])+'.h5, topology: '+str(best_model[5])+', weights: '+str(best_model[6]))  
            file.write('\t\t\t\ttraining performance: '+str(best_model[2])+'\n')
            print('\t\t\t\ttraining performance:',str(best_model[2]))
            file.write('\t\t\t\ttesting performance: '+str(best_model[7])+'\n')
            print('\t\t\t\ttesting performance:',str(best_model[7]))
            file.write('\t\t\t\tvalidation performance: '+str(best_model[12])+'\n')
            file.close()
            print('\t\t\t\tvalidation performance: ',str(best_model[12]))            
            
def best_model(models_sorted,model1L,model2L,logfile,model1='logistic regression',model2='random forest',cross_validation=None):
       #find best model 
       #return: model1L, model2L sorted in descending order of train_test_auc (overall performance)
        if models_sorted==False:
            model1L.sort(key=operator.itemgetter(1),reverse=True)#sort logistic regression models in descending order of overall performance
            model2L.sort(key=operator.itemgetter(1),reverse=True)#sort random forest models in descending order of overall performance (training AUC + test set AUC)-absolute(training AUC - test set AUC)
        best_model1=model1L[0]
        best_model2=model2L[0]
        #(i,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)
        #(0,1             ,2        ,3        ,4        ,5        ,6        ,7       ,8       ,9       ,10      ,11      )
        if os.path.isfile(logfile)==False:#logfile does not exist, create a new one
           file=open(logfile,'w+')
        else:
           file=open(logfile,'a')        
        if float(best_model2[1]) > float(best_model1[1]):
            if cross_validation!=None:
                if best_model2[0]!='none':#more than 1 iterations               
                    file.write('\t\t\t*best model*: '+model2+' of iteration: '+str(best_model2[0])+', model file: rf'+str(best_model2[0])+'.model, training AUC: '+str(np.round(best_model2[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model2[7],3))+'\n')
                    print('\t\t\t*best model*: '+model2+' of iteration: '+str(best_model2[0])+', training AUC: '+str(np.round(best_model2[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model2[7],3)))
                else:#1 iteration
                    file.write('\t\t\t*best model*: '+model2+', model file: rf.model, training AUC: '+str(np.round(best_model2[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model2[7],3))+'\n')
                    print('\t\t\t*best model*: '+model2+', training AUC: '+str(np.round(best_model2[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model2[7],3)))           
                file.write('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model2[3],3))+', training TNR: '+str(np.round(best_model2[4],3))+'\n')
                file.write('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model2[5],3))+', training FNR: '+str(np.round(best_model2[6],3))+'\n')   
                print('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model2[3],3))+', training TNR: '+str(np.round(best_model2[4],3))+'\n')
                print('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model2[5],3))+', training FNR: '+str(np.round(best_model2[6],3))+'\n')
            else:
                if best_model2[0]!='none':#more than 1 iterations               
                    file.write('\t\t\t*best model*: '+model2+' of iteration: '+str(best_model2[0])+', model file: rf'+str(best_model2[0])+'.model, training AUC: '+str(np.round(best_model2[2],3))+', test AUC: '+str(np.round(best_model2[7],3))+'\n')
                    print('\t\t\t*best model*: '+model2+' of iteration: '+str(best_model2[0])+', training AUC: '+str(np.round(best_model2[2],3))+', test AUC: '+str(np.round(best_model2[7],3)))
                else:#1 iteration
                    file.write('\t\t\t*best model*: '+model2+', model file: rf.model, training AUC: '+str(np.round(best_model2[2],3))+', test AUC: '+str(np.round(best_model2[7],3))+'\n')
                    print('\t\t\t*best model*: '+model2+', training AUC: '+str(np.round(best_model2[2],3))+', test AUC: '+str(np.round(best_model2[7],3)))           
                file.write('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model2[3],3))+', training TNR: '+str(np.round(best_model2[4],3))+', testing TPR: '+str(np.round(best_model2[8],3))+', testing TNR: '+str(np.round(best_model2[9],3))+'\n')
                file.write('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model2[5],3))+', training FNR: '+str(np.round(best_model2[6],3))+', testing FPR: '+str(np.round(best_model2[10],3))+', testing FNR: '+str(np.round(best_model2[11],3))+'\n')   
                print('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model2[3],3))+', training TNR: '+str(np.round(best_model2[4],3))+', testing TPR: '+str(np.round(best_model2[8],3))+', testing TNR: '+str(np.round(best_model2[9],3)))
                print('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model2[5],3))+', training FNR: '+str(np.round(best_model2[6],3))+', testing FPR: '+str(np.round(best_model2[10],3))+', testing FNR: '+str(np.round(best_model2[11],3)))   
        elif float(best_model1[1]) > float(best_model2[1]):
            if cross_validation!=None:
               if best_model1[0]!='none':
                   file.write('\t\t\t*best model*: '+model1+' of iteration: '+str(best_model1[0])+', model file: model1'+str(best_model1[0])+'.model, training AUC: '+str(np.round(best_model1[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model1[7],3))+'\n')
                   print('\t\t\t*best model*: '+model1+' of iteration: '+str(best_model1[0])+', training AUC: '+str(np.round(best_model1[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model1[7],3)))
               else:#1 iteration
                   file.write('\t\t\t*best model*: '+model1+', model file: model1.model, training AUC: '+str(np.round(best_model1[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model1[7],3))+'\n')
                   print('\t\t\t*best model*: '+model1+': training AUC: '+str(np.round(best_model1[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model1[7],3)))
               file.write('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model1[3],3))+', training TNR: '+str(np.round(best_model1[4],3))+'\n')
               file.write('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model1[5],3))+', training FNR: '+str(np.round(best_model1[6],3))+'\n')          
               print('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model1[3],3))+', training TNR: '+str(np.round(best_model1[4],3)))
               print('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model1[5],3))+', training FNR: '+str(np.round(best_model1[6],3)))          
        else:#model1 and model2 have same performance
           file.write('\t\t\t2 *best models*:\n')
           if cross_validation!=None:
               file.write('\t\t\t2 *best models*:\n')            
               print('\t\t\t2 *best models*:')
               if best_model2[0]!='none':
                   file.write('\t\t\t'+model2+' of iteration: '+str(best_model2[0])+', training AUC: '+str(np.round(best_model2[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model2[7],3))+'\n')
                   print('\t\t\t'+model2+' of iteration: '+str(best_model2[0])+', training AUC: '+str(np.round(best_model2[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model2[7],3)))          
               else:
                   file.write('\t\t\t'+model2+', training AUC: '+str(np.round(best_model2[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model2[7],3))+'\n')
                   print('\t\t\t'+model2+', training AUC: '+str(np.round(best_model2[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model2[7],3)))                           
               file.write('\t\t\t'+model2+': training AUC: '+str(np.round(best_model2[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model2[7],3))+'\n')
               print('\t\t\t'+model2+': training AUC: '+str(np.round(best_model2[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model2[7],3)))         
               file.write('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model2[3],3))+', training TNR: '+str(np.round(best_model2[4],3))+'\n')
               file.write('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model2[5],3))+', training FNR: '+str(np.round(best_model2[6],3))+'\n')   
               print('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model2[3],3))+', training TNR: '+str(np.round(best_model2[4],3)))
               print('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model2[5],3))+', training FNR: '+str(np.round(best_model2[6],3)))   
               if best_model1[0]!='none':#more than 1 iterations
                   file.write('\t\t\t'+model1+' of iteration: '+str(best_model1[0])+', training AUC: '+str(np.round(best_model1[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model1[7],3))+'\n')
                   print('\t\t\t'+model1+' of iteration: '+str(best_model1[0])+', training AUC: '+str(np.round(best_model1[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model1[7],3)))             
               else:#1 iteration
                   file.write('\t\t\t'+model1+': training AUC: '+str(np.round(best_model1[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model1[7],3))+'\n')
                   print('\t\t\t'+model1+': training AUC: '+str(np.round(best_model1[2],3))+', '+str(cross_validation)+'-fold CV AUC: '+str(np.round(best_model1[7],3)))           
               file.write('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model1[3],3))+', training TNR: '+str(np.round(best_model1[4],3))+'\n')
               file.write('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model1[5],3))+', training FNR: '+str(np.round(best_model1[6],3))+'\n')          
               print('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model1[3],3))+', training TNR: '+str(np.round(best_model1[4],3)))
               print('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model1[5],3))+', training FNR: '+str(np.round(best_model1[6],3)))          
           else:
               file.write('\t\t\t2 *best models*:\n')            
               print('\t\t\t2 *best models*:')
               if best_model2[0]!='none':
                   file.write('\t\t\t'+model2+' of iteration: '+str(best_model2[0])+', training AUC: '+str(np.round(best_model2[2],3))+', test AUC: '+str(np.round(best_model2[7],3))+'\n')
                   print('\t\t\t'+model2+' of iteration: '+str(best_model2[0])+', training AUC: '+str(np.round(best_model2[2],3))+', test AUC: '+str(np.round(best_model2[7],3)))          
               else:
                   file.write('\t\t\t'+model2+', training AUC: '+str(np.round(best_model2[2],3))+', test AUC: '+str(np.round(best_model2[7],3))+'\n')
                   print('\t\t\t'+model2+', training AUC: '+str(np.round(best_model2[2],3))+', test AUC: '+str(np.round(best_model2[7],3)))                           
               file.write('\t\t\t'+model2+': training AUC: '+str(np.round(best_model2[2],3))+', test AUC: '+str(np.round(best_model2[7],3))+'\n')
               print('\t\t\t'+model2+': training AUC: '+str(np.round(best_model2[2],3))+', test AUC: '+str(np.round(best_model2[7],3)))         
               file.write('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model2[3],3))+', training TNR: '+str(np.round(best_model2[4],3))+', testing TPR: '+str(np.round(best_model2[8],3))+', testing TNR: '+str(np.round(best_model2[9],3))+'\n')
               file.write('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model2[5],3))+', training FNR: '+str(np.round(best_model2[6],3))+', testing FPR: '+str(np.round(best_model2[10],3))+', testing FNR: '+str(np.round(best_model2[11],3))+'\n')   
               print('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model2[3],3))+', training TNR: '+str(np.round(best_model2[4],3))+', testing TPR: '+str(np.round(best_model2[8],3))+', testing TNR: '+str(np.round(best_model2[9],3)))
               print('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model2[5],3))+', training FNR: '+str(np.round(best_model2[6],3))+', testing FPR: '+str(np.round(best_model2[10],3))+', testing FNR: '+str(np.round(best_model2[11],3)))   
               if best_model1[0]!='none':#more than 1 iterations
                   file.write('\t\t\t'+model1+' of iteration: '+str(best_model1[0])+', training AUC: '+str(np.round(best_model1[2],3))+', test AUC: '+str(np.round(best_model1[7],3))+'\n')
                   print('\t\t\t'+model1+' of iteration: '+str(best_model1[0])+', training AUC: '+str(np.round(best_model1[2],3))+', test AUC: '+str(np.round(best_model1[7],3)))             
               else:#1 iteration
                   file.write('\t\t\t'+model1+': training AUC: '+str(np.round(best_model1[2],3))+', test AUC: '+str(np.round(best_model1[7],3))+'\n')
                   print('\t\t\t'+model1+': training AUC: '+str(np.round(best_model1[2],3))+', test AUC: '+str(np.round(best_model1[7],3)))           
               file.write('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model1[3],3))+', training TNR: '+str(np.round(best_model1[4],3))+', testing TPR: '+str(np.round(best_model1[8],3))+', testing TNR: '+str(np.round(best_model1[9],3))+'\n')
               file.write('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model1[5],3))+', training FNR: '+str(np.round(best_model1[6],3))+', testing FPR: '+str(np.round(best_model1[10],3))+', testing FNR: '+str(np.round(best_model1[11],3))+'\n')          
               print('\t\t\t\t\t\t\t\t\t training TPR: '+str(np.round(best_model1[3],3))+', training TNR: '+str(np.round(best_model1[4],3))+', testing TPR: '+str(np.round(best_model1[8],3))+', testing TNR: '+str(np.round(best_model1[9],3)))
               print('\t\t\t\t\t\t\t\t\t training FPR: '+str(np.round(best_model1[5],3))+', training FNR: '+str(np.round(best_model1[6],3))+', testing FPR: '+str(np.round(best_model1[10],3))+', testing FNR: '+str(np.round(best_model1[11],3)))          
        file.close()
        return (model1L,model2L)

def summarize_accuracy_results(modelname,logfile,performanceL,train_accuracyL,test_accuracyL):    
    #performanceL = list of (iteration,train_accuracy+test_accuracy,train_report,test_report)
    print('==='+modelname+'===')
    f=open(logfile,'a')
    f.write('==='+modelname+'===\n')
    ##rank all models so  far in descending order of training accuracy + test accuracy
    performanceL.sort(key=operator.itemgetter(1),reverse=True)#sort models in descending order of train_test_accuracy
    for performance in performanceL:
        print('iteration: ',performance[0],' training AUC: ',np.round(performance[2],2),', test AUC: ',np.round(performance[3],2))
        f.write('iteration: '+str(performance[0])+' training AUC: '+str(np.round(performance[2],2))+', test AUC, '+str(np.round(performance[3],2))+'\n')
    ###best model so far###
    best_performance=performanceL[0]
    print('*best model*: iteration: ',best_performance[0],' training AUC: ',np.round(best_performance[2],2),', test AUC: ',np.round(best_performance[3],2))
    f.write('*best model*: iteration: '+str(best_performance[0])+' training AUC: '+str(np.round(best_performance[2],2))+', test AUC: '+str(np.round(best_performance[3],2))+'\n')
    ###average, mini and max accuracies###
    print('\t\t\t========='+modelname+'=========')
    print('\t\t\tmean training AUC: '+str(np.round(np.mean(train_accuracyL),2)))
    print('\t\t\tmin training AUC: '+str(np.round(np.min(train_accuracyL),2)))
    print('\t\t\tmax training AUC: '+str(np.round(np.max(train_accuracyL),2)))
    print('\t\t\tmean test AUC: '+str(np.round(np.mean(test_accuracyL),2)))
    print('\t\t\tmin test AUC: '+str(np.round(np.min(test_accuracyL),2)))
    print('\t\t\tmax test AUC: '+str(np.round(np.max(test_accuracyL),2)))
    f.write('\t\t\t========='+modelname+'=========\n')
    f.write('\t\t\tmean training AUC: '+str(np.round(np.mean(train_accuracyL),2))+'\n')
    f.write('\t\t\tmin training AUC: '+str(np.round(np.min(train_accuracyL),2))+'\n')
    f.write('\t\t\tmax training AUC: '+str(np.round(np.max(train_accuracyL),2))+'\n')
    f.write('\t\t\tmean test AUC: '+str(np.round(np.mean(test_accuracyL),2))+'\n')
    f.write('\t\t\tmin test AUC: '+str(np.round(np.min(test_accuracyL),2))+'\n')
    f.write('\t\t\tmax test AUC: '+str(np.round(np.max(test_accuracyL),2))+'\n')
    f.close()
                    
def summarize_performance(file,testset_aucL,testset_tprL,testset_tnrL,testset_fprL,testset_fnrL):
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

def best_model_of_each_iteration(logfile):
    #find the best model of logistric regression and random forest of each iteration
    #input: logfile e.g. "H:\data\EIS preterm prediction\results\workflow1\15dec_filtered_data_28inputs\logfile.txt"
    #output: the best model of each iteration
    h={}
    f=open(logfile,'r')
    line=f.readline().rstrip()
    while line!='':
       m=re.match('^\t+iteration:\s+(\d+)\s+random forest\s+\(training\).+$',line)
       if m:
           line=f.readline().rstrip()
           m2=re.match('^\t+random forest\s+\(testing\)\s+([0-9\.]+).+$',line)
           if m2:
                iterNum=m.group(1)
                test_auc=m2.group(1)
                if h.get(iterNum)==None:
                    h[iterNum]=[('random forest',test_auc)]
                else:
                    l=h[iterNum]
                    l.append(('random forest',test_auc))
                    h[iterNum]=l
           else:
                print('m2 not matched')
                sys.exit
       else:
           m3=re.match('^\t+iteration:\s+(\d+)\s+logistic regression\s+\(training\).+$',line)
           if m3:
               line=f.readline().rstrip()
               m4=re.match('^\t+logistic regression\s+\(testing\)\s+([0-9\.]+).+$',line)
               if m4:
                    iterNum=m3.group(1)
                    test_auc=m4.group(1)
                    if h.get(iterNum)==None:
                        h[iterNum]=[('logistic regression',test_auc)]
                    else:
                        l=h[iterNum]
                        l.append(('logistic regression',test_auc))
                        h[iterNum]=l
               else:
                    print('m4 not matched')
                    sys.exit
       line=f.readline().rstrip()
    f.close()
    rf=0
    log_reg=0
    print('keys: '+str(len(h.keys())))
    print(list(h.keys()))
    for iterNum in h.keys():
        models=h[iterNum]
        (model0,test_auc0)=models[0]    
        (model1,test_auc1)=models[1]
        if float(test_auc0) > float(test_auc1):
            ptb_predictor=model0
        else:
            ptb_predictor=model1
        print(iterNum+': ptb predictor: '+ptb_predictor)
        if ptb_predictor=='random forest':
            rf+=1
        else:
            log_reg+=1
    print('random forest ptb predictors: ',str(rf))
    print('logistic regression ptb predictors: ',str(log_reg))

def split_train_test_sets_and_get_ids(dataset,trainset_fraction,iterations,results_path):
    #split a dataset with ids into ids of training sets and ids of test sets
    data=pd.read_csv(dataset)
    (_,c)=data.shape
    cols=list(data.columns)
    test_size=1-float(trainset_fraction)    
    for i in range(int(iterations)):
        print(i)
        file=open(results_path+'trainset'+str(i)+'_ids.csv','w+')
        file.write('hospital_id\n')
        #seed=random.randint(0,2**32-1)
        seed=random.randint(0,999999) #ith seed for data split
        print('seed of train/test split: '+str(seed))
        (train_set,test_set)=split_train_test_sets(data,test_size,seed,cols[c-1])
        rows=train_set.index
        (n,_)=train_set.shape
        for k in range(n):
            pid=train_set.at[rows[k],'hospital_id']
            file.write(str(pid)+'\n')
        file.close()
        print(results_path+'\\trainset'+str(i)+'_ids.csv')
        file2=open(results_path+'\\testset'+str(i)+'_ids.csv','w+')
        file2.write('hospital_id\n')
        rows2=test_set.index
        (n2,_)=test_set.shape
        for k2 in range(n2):
            pid2=test_set.at[rows2[k2],'hospital_id']
            file2.write(str(pid2)+'\n')
        file2.close()
        print(results_path+'\\testset'+str(i)+'_ids.csv')

def get_ids(data_csv,ids_of_all_data,ids_file):
    print('Get ids of '+data_csv)
    s=sd.SelectData()
    idsL=s.get_patients_ids2(data_csv,ids_of_all_data)#get the ids of data            
    pd.DataFrame(idsL,columns=['hospital_id']).to_csv(ids_file,index=False)

def get_ids_of_training_sets(results_path,ids_file):
    ###get ids of training sets at results_path from ids_file (use N cpus to get ids of N training sets simultaneously)
    Parallel(n_jobs=-1,verbose=20,batch_size=1)(delayed(get_ids)(results_path+'trainset'+str(i)+'.csv',ids_file,results_path+'trainset'+str(i)+'_ids.csv') for i in range(100))

def get_ids_of_test_sets(results_path,ids_file):
    ###get ids of test sets at results_path from ids_file (use N cpus to get ids of N test sets simultaneously)
    Parallel(n_jobs=-1,verbose=20,batch_size=1)(delayed(get_ids)(results_path+'testset'+str(i)+'.csv',ids_file,results_path+'testset'+str(i)+'_ids.csv') for i in range(100))

def get_data_by_ids(i,datafilename,data_ids_path,ids_file,testdata_csv,skip_ids=True):
    #i, ith test set or ith training set
    #datafilename, 'testset' or 'trainset'
    #data_ids_path, path of ids of data to retrieve
    #ids_file, file containing ids of all data
    #testdata_csv, output csv file without ids
    if os.path.isfile(data_ids_path+datafilename+str(i)+'_ids.csv')==False:
       sys.exit(data_ids_path+datafilename+str(i)+'_ids.csv does not exist')
    s = sd.SelectData()
    idsL_df=s.get_data_by_ids(data_ids_path+datafilename+str(i)+'_ids.csv',ids_file)#get the spectra which have ids in ids_file and the ids of the spectra   
    ids_series=idsL_df.iloc[:,0]
    (_,c)=idsL_df.shape
    if skip_ids:
        idsL_df=idsL_df.iloc[:,1:c]#skip the ids column            
    idsL_df.to_csv(testdata_csv,index=False)
    return ids_series
        
def poly_kernel(degree,support_vector,test_instance):
    #polynomial kernel: K(y,x)=(yTx)^degree
    l=len(support_vector)
    result=0
    for i in range(l):
        result+=support_vector[i]*test_instance.iat[0,i]
    result=result**degree
    return result

def decision_function(test_instance,degree,dual_coeff,support_vectors,intercept):
    (r,_)=support_vectors.shape
    v=0
    for i in range(r):
        v+=dual_coeff[i]*poly_kernel(degree,support_vectors[i][:],test_instance)    
    v+=intercept
    return v

def poly_svm_predict(df,degree,dual_coef,support_vectors,intercept,probA,probB):
    #build a polynomial kernel SVM to predict from inputs of a dataset
    #X, inputs of dataset
    #return: auc of df, probabilities of class 1 of each instance in df
    (r,c)=df.shape
    X=df.iloc[:,0:c-1]
    probL=[]
    for i in range(r):
        d=decision_function(X.iloc[i,:],degree,dual_coef,support_vectors,intercept)
        prob = 1 / (1 + np.exp(d*probA + probB))
        probL.append(prob)
    auc=roc_auc_score(df.iloc[:,c-1],probL)
    return (probL,auc)

def load_sklearn_gp_and_display_info(model_file,testset=None):
    from joblib import load
    m=load(model_file)
    #print(str(m['gp']))
    print('hyperparameters: '+str(m['gp'].kernel_.hyperparameters))
    print('theta: '+str(m['gp'].kernel_.theta))
    
def load_sklearn_gb_and_display_info(model_file,testset=None):
    from joblib import load
    m=load(model_file)
    print(m)
    params=m.get_params()
    print(params)
    print(m.estimators_)
    print(m.estimators_[29,0].tree_)

def load_sklearn_xgb_and_display_info(model_file,testset=None):
    from joblib import load
    m=load(model_file)
    print(m)
    params=m.get_params()
    print(params)
    b=m.get_booster()
    #print(b.trees_to_dataframe())
    df=b.trees_to_dataframe()
    df.to_csv('xgbtrees.csv')
    import xgboost as xgb
    import matplotlib.pyplot as plt
    xgb.plot_tree(b,num_trees=0)
    plt.show()
    
def load_sklearn_model_and_display_info(model_file,testset=None,ensemble=False):
    from joblib import load
    m=load(model_file)
    if ensemble:
        basemodels=m.estimators_
        for i in range(0,len(basemodels)):
            print('base model: '+str(basemodels[i]))
    print(m)
    #data=pd.read_csv("D:\\EIS preterm prediction\\EIS for cervical cancer diagnosis\\ColePY.csv")
    if testset!=None:
        data=pd.read_csv(testset)
        (auc,_,_,_,_)=predict_testset(False,m,data)
        print(auc)
        (_,c)=data.shape
        #func=m.decision_function(data.iloc[:,0:c-1])
        #print(func)
    if ensemble==False:
        params=m.get_params()
        print('===parameters of model===')
        print(params)
    '''
    ###SVM parameters###
    print('gamma=',m.gamma)
    print('dual coef=', m.dual_coef_) #ndarray of shape (n_classes -1, n_SV)
    print('shape dual coef=',m.dual_coef_.shape)
    print('support vectors=',m.support_vectors_)
    print('shape of support vectors ndarray',m.support_vectors_.shape)
    print('no. of support vectors=',m.n_support_)
    print('b=',m.intercept_)
    print('probA=',m.probA_)
    print('probB=',m.probB_)
    '''    
    ##SVM pipeline parameters###
    #stepsL=params.get('steps')
    #zscore_tuple=stepsL[0]
    #zscore=zscore_tuple[1]
    #zscore=m['zscore']
    #print('zscore transform: mean=',zscore.mean_)
    #print('zscore transform: variance=',zscore.var_)
    #print('zscore transform: scale=',zscore.scale_)
    #svc_tuple=stepsL[1]
    #svc=svc_tuple[1]
    #print('gamma=',svc.gamma)
    #print('dual coef=', svc.dual_coef_) #ndarray of shape (n_classes -1, n_SV)
    #print('shape dual coef=',svc.dual_coef_.shape)
    #print('support vectors=',svc.support_vectors_)
    #print('shape of support vectors ndarray',svc.support_vectors_.shape)
    #print('no. of support vectors=',svc.n_support_)
    #print('b=',svc.intercept_)
    #print('probA=',svc.probA_)
    #print('probB=',svc.probB_)

def create_testsets_by_merging_best_EIS_readings_with_other_features(testsets_ids_path,nonEIS_data_with_ids,filter2_path,good_readings_testset_path,results_path):
    create_folder_if_not_exist(results_path)
    MP=mp.ModelsPredict()
    for i in range(100):
        num=random.randint(0,2**32-1)
        nonEIS_testdata=results_path+'non-EIS_testset'+str(num)+'.csv'
        get_data_by_ids(i,'testset',testsets_ids_path,nonEIS_data_with_ids,nonEIS_testdata,skip_ids=False)
        filter2=filter2_path+'rf'+str(i)+'.model'
        if os.path.isfile(filter2):
            MP.main(
                            i=i,
                            filter2type2='rf',
                            filter2_software2='weka',
                            filter2_path2=filter2_path,
                            results_path2=results_path,
                            create_dataset_only=True,#select good readings of patients and save them to csv file
                            good_readings_testset_i2=good_readings_testset_path+'good_readings_testset'+str(i)+'.csv',
                            testset_i_ids2=testsets_ids_path+'testset_ids'+str(i)+'.csv',
                            allreadings_with_ids2=allreadings_with_ids,
                            ordinal_encode2=ordinal_encode_of_filter2,#the discrete filter2s don't encode their discrete inputs as integers
                            weka_path2=weka_path,
                            logfile2=logfile,
                            logfile2_option='a'#open logfile to append results to it, then close it
                    )
            
        
if __name__ == "__main__":
     #df=merge_parous_features_into_boolean_feature_and_merge_nulliparous_features_into_boolean_feature("U:\\EIS preterm prediction\\my_filtered_data_28inputs_438_V1_demographics_treatment_history_obstetric_history_with_ids.csv")
     #transform_numeric_obstetric_features_to_boolean_features(df,"U:\\EIS preterm prediction\\my_filtered_data_28inputs_438_V1_demographics_treatment_history_obstetric_history_with_ids_boolean_features.csv")
     #from sklearn.preprocessing import MyPolynomialFeatures
     #p = MyPolynomialFeatures.MyPolynomialFeatures("C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_438_V1_demographics_treatment_history_obstetric_history_with_ids\\trainset0_preprocessed_balanced_trainset_size=2000_degree=4_info_gain_selected_features=30_ga_cv_of_gaussianNB_selected_features.csv")
     #data=pd.read_csv("U:\\EIS preterm prediction\\my_filtered_data_28inputs.csv")
     #(_,c)=data.shape
     #X=data.iloc[:,:c-1]#skip the last column (targets)
     #X=p.fit_transform(X)
     #print(X)
     #X.to_csv("my_filtered_data_28inputs_poly_features.csv",index=False)
    #predict_PTB_using_PTB_history('have_ptb_history_no_treatment_with_ids.csv')
    #predict_PTB_using_PTB_history('have_ptb_history_treated_with_ids.csv')
    #predict_PTB_using_PTB_history('have_ptb_history_no_treatment_with_previous_pregnancy_with_ids.csv')
    #predict_PTB_using_PTB_history('have_ptb_history_with_ids.csv')    
    #predict_PTB_using_PTB_history('have_ptb_history_treated_with_previous_pregnancy_with_ids.csv',threshold=0.1)    
    
    #divide_features_by_frequency_squared("D:\\EIS preterm prediction\\i4i MIS\\raw data\\subtract v1v2 compensation\\mis_data_c1c2c3_with_subtract_v1v2_compensation_visit2_no_missing_labels.csv","D:\\EIS preterm prediction\\i4i MIS\\raw data\\subtract v1v2 compensation\\mis_data_c1c2c3_with_subtract_v1v2_compensation_visit2_no_missing_labels_div_freqsq.csv")
    #change_ids_to_dummy_ids("my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_ids.csv","my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_dummy_ids.csv")
    #mean_of_spectra_of_each_id("438_V1_all_eis_readings_real_imag_with_dummy_ids.csv","mean_of_438_V1_all_eis_readings_real_imag_with_dummy_ids.csv",data='eis')
    #merge_features("D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs_log_transformed.csv","D:\\EIS preterm prediction\\metabolite\\merged_features_of_asymp_22wks_438_V1_8inputs_log_transformed.csv")
    #merge_features("D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs.csv","D:\\EIS preterm prediction\\metabolite\\merged_features2_of_asymp_22wks_438_V1_8inputs.csv",merge_operation='product')
    #log10_transform("D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs_outliers_removed_as_blanks.csv","D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs_outliers_removed_as_blanks_log_transformed.csv")
    #log10_transform("D:\\EIS preterm prediction\\metabolite\\merged_features_of_asymp_22wks_438_V1_8inputs.csv","D:\\EIS preterm prediction\\metabolite\\log_transformed_merged_features_of_asymp_22wks_438_V1_8inputs.csv")
    #log10_transform("D:\\EIS preterm prediction\\metabolite\\asymp_22wks_filtered_data_28inputs_with_ids.csv","D:\\EIS preterm prediction\\metabolite\\asymp_22wks_filtered_data_28inputs_log10_transformed_with_ids.csv")
    #log10_transform("D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs.csv","D:\\EIS preterm prediction\\metabolite\\data_log10.csv")
    #multiply_features_by_number_transform("D:\\EIS preterm prediction\\i4i MIS\\raw data\\no compensation\\mis_data_c1c2c3_no_compensation_visit1_visit2_no_missing_labels.csv","D:\\EIS preterm prediction\\i4i MIS\\raw data\\no compensation\\mis_data_c1c2c3_no_compensation_visit1_visit2_no_missing_labels_x100.csv",100)
    #get_no_of_inputs_of_models("H:\\data\\EIS preterm prediction\\results\\workflow1\\filter2 from sharc\\selected_unselected_eis_readings\\",classifier='rf_and_log_reg')
    #get_no_of_inputs_of_models("H:\\data\\EIS preterm prediction\\results\\workflow1\\filter2 from sharc\\selected_unselected_eis_readings_no_treatment\\",classifier='rf_and_log_reg')
    #get_no_of_inputs_of_models("H:\\data\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\",classifier='rf_and_log_reg')
    #get_no_of_inputs_of_models("H:\\data\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_no_treatment\\",classifier='rf_and_log_reg')
    #traindf=pd.read_csv("C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\ga_cv_of_gaussianNB_selected_features\\trainset0_preprocessed_balanced_trainset_size=2000_degree=-1_info_gain_selected_features=-1_ga_cv_of_gaussianNB_selected_features.csv")
    #testdf=pd.read_csv("C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\ga_cv_of_gaussianNB_selected_features\\trainset0_preprocessed_balanced_trainset_size=2000_degree=-1_info_gain_selected_features=-1_ga_cv_of_gaussianNB_selected_features.csv")
    #model=load("C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\ga_cv_of_gaussianNB_selected_features\\rf\\rf0.joblib")
    #predict_trainset_and_testset_using_sklearn_and_optimal_threshold(traindf,testdf,model)
    #load_sklearn_model_and_display_info("U:\\EIS preterm prediction\\Every Baby\\new contract 22 to 23\\code\\mlp-EIS.joblib")
    load_sklearn_model_and_display_info("U:\\EIS preterm prediction\\Every Baby\\new contract 22 to 23\\code\\ensemble1-EIS.joblib")

    #load_sklearn_model_and_display_info("D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\visit1_symp\\poly_svm3\\poly_svm0.joblib")
    #load_sklearn_model_and_display_info("C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_438_V1_demographics_treatment_history_obstetric_history\\train66test34\\rbfsvm (zscore)\\rbf_svm0.joblib")
    #load_sklearn_model_and_display_info("C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\soft voting_cv_auc_weights\\voted_classifier0.joblib",ensemble=True)
    #load_sklearn_model_and_display_info("C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_log_regression2\\stacked_ensemble0.joblib",ensemble=True)
    #load_sklearn_xgb_and_display_info("C:\\Users\\uos\\EIS preterm prediction\\results\\sklearn_pipeline\\xgb\\xgb28.joblib")
    #load_sklearn_xgb_and_display_info("F:\\EIS preterm prediction\\results\\438_V1_28inputs\\xgb\\xgb0.joblib")
    #load_sklearn_gb_and_display_info("C:\\Users\\uos\\EIS preterm prediction\\results\\sklearn_pipeline\\gb\\gb0.joblib")
    #load_sklearn_gp_and_display_info("C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\gp_matern (differential evolution)\\gp_matern0.joblib")
    #load_sklearn_gp_and_display_info("C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_with_ids\\train66test34\\gp_multiple_of_rbf\\gp22.joblib")
    #load_sklearn_model_and_display_info("C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\mlp_2_layers_(zscore)\\mlp0.joblib")
    #features_stats("D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_8inputs_log_transformed\\train80test20\\trainset_encoded_zscore_normalized_elu0.csv",normalize=False)
    #features_stats("D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_8inputs_log_transformed\\train80test20\\testset_encoded_zscore_normalized_elu0.csv",normalize=False)
    #features_stats("F:\\EIS preterm prediction\\results\\workflow1\\filter from sharc\\selected_unselected_eis_readings\\trainset_encoded_zscore_normalized_relu0.csv")
    #features_stats("D:\\EIS preterm prediction\\i4i MIS\\raw data\\divide v1v2 compensation\\mis_data_c1c2c3_with_divide_v1v2_compensation_visit1_no_missing_labels.csv",normalize=False)
    #features_stats("D:\\EIS preterm prediction\\i4i MIS\\raw data\\no compensation\\mis_data_c1c2c3_no_compensation_visit1_no_missing_labels.csv",normalize=False)
    #features_stats("D:\\EIS preterm prediction\\i4i MIS\\raw data\\subtract v1v2 compensation\\mis_data_c1c2c3_with_subtract_v1v2_compensation_visit1_no_missing_labels.csv",normalize=False)
    #features_stats("U:\\EIS preterm prediction\\438_V1_demographics_treatment_history_obstetric_history_with_ids_boolean_features.csv")
    #get_number_of_inputs_of_models("H:\\data\\EIS preterm prediction\\results\\workflow1\\filter2 from sharc\\selected_unselected_eis_readings\\",model='rf')
    #get_ids("F:\\EIS preterm prediction\\results\\workflow1\\filter from sharc\\selected_unselected_eis_readings\\trainset23.csv","F:\\EIS preterm prediction\\selected_unselected_eis_readings_with_ids.csv","F:\\EIS preterm prediction\\results\\workflow1\\filter from sharc\\selected_unselected_eis_readings\\train23_ids.csv")
    #get_ids("F:\\EIS preterm prediction\\results\\workflow1\\filter from sharc\\selected_unselected_eis_readings\\testset23.csv","F:\\EIS preterm prediction\\selected_unselected_eis_readings_with_ids.csv","F:\\EIS preterm prediction\\results\\workflow1\\filter from sharc\\selected_unselected_eis_readings\\testset23_ids.csv")
    
    #get_ids_of_patients()
    #results_path="h:\\data\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\"
    #results_path="h:\\data\\EIS preterm prediction\\results\\workflow1\\filtered_data\\"
    #ids_of_best_readings="filtered_data_28inputs_with_ids.csv"#ids of 438 patients each with the filtered (best) reading
    #get_ids_of_training_sets(results_path,ids_of_best_readings)
    #ids_of_all_readings="438_V1_4_eis_readings_28inputs_with_ids.csv"
    #get_ids_of_test_sets(results_path,ids_of_all_readings)
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_demographics_obstetric_history_10\\"
    #get_ids_of_test_sets(results_path,'438_V1_demographics_obstetric_history_with_ids.csv')
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_demographics_obstetric_history_2_parous_features_using_GA_features2\\"
    #get_ids_of_test_sets(results_path,'438_V1_demographics_obstetric_history_2_parous_features_with_ids.csv')
    #get_ids_of_training_sets(results_path,'438_V1_demographics_obstetric_history_2_parous_features_with_ids.csv')
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_demographics_obstetric_history_2_parous_features_using_GA_features\\"
    #get_ids_of_test_sets(results_path,'438_V1_demographics_obstetric_history_2_parous_features_with_ids.csv')
    #get_data_by_ids(35,'testset',"h:\\data\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\","438_V1_demographics_obstetric_history_2_parous_features_with_ids.csv",results_path+'testset35_using_ids_of_testset35_of_15dec_filtered_data_28inputs.csv')
    ###create training sets###
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_438_V1_demographics_obstetric_history_2_parous_features\\"
    #create_folder_if_not_exist(results_path)
    #for i in range(100):
    #    get_data_by_ids(i,'trainset',"h:\\data\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\","filtered_data_28inputs_438_V1_demographics_obstetric_history_2_parous_features_with_ids.csv",results_path+'trainset'+str(i)+'.csv')
    
    #split_train_test_sets_and_get_ids("438_V1_demographics_obstetric_history_with_ids.csv",0.8,100,"D:\\EIS preterm prediction\\results\\workflow1\\438_V1_demographics_obstetric_history_10\\")
    #merge_parous_features('438_V1_demographics_obstetric_history_with_ids.csv','438_V1_demographics_obstetric_history_2_parous_features_with_ids.csv')
    #extract_obstetric_features('438_V1_previous_history_and_demographics.csv','obstetric_features.csv')
    #merge_races('438_V1_previous_history_and_demographics.csv','438_V1_ethnicity.csv')
    #merge_demographics_and_obstetric_history('438_V1_previous_history_and_demographics.csv','438_V1_ethnicity.csv','obstetric_features.csv','438_V1_demographics_obstetric_history.csv')
    #best_model_of_each_iteration("H:\\data\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_no_treatment_noise_added_10_to_15features\\logfile.txt")
    #select_all_readings_of_ids("D:\\EIS preterm prediction\\msc2020\\average EIS readings_20patients_V1.csv","438_V1_4_eis_readings_28inputs_with_ids.csv","D:\\EIS preterm prediction\\msc2020\\all_EIS_readings_20patients_V1.csv")
    #csv_data='D:\\EIS preterm prediction\\poly features\\438_V1_28inputs_poly_degree4.csv'
    #arff_data='D:\\EIS preterm prediction\\data.arff'
    #csv_data='D:\\EIS preterm prediction\\trainset_balanced_reduced0.csv'
    #arff_data='D:\\EIS preterm prediction\\trainset_balanced_reduced0.arff'
    #csv_data='D:\\EIS preterm prediction\\EIS_Data\\EIS_Data\\438_V1_28inputs.csv'
    #arff_data='D:\\EIS preterm prediction\\data.arff'
    #csv_data='D:\\EIS preterm prediction\\438_V1_30inputs_demographics.csv'
    #arff_data='D:\\EIS preterm prediction\\data.arff'
    #df=pd.read_csv(csv_data)
    #dataframe_to_arff(df,arff_data)
      
    #arff_file="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow2\\outliers_preterm_reduced.arff"
    #df=pd.read_csv("C:\\Users\\uos\\EIS preterm prediction\\results\\workflow2\\outliers_preterm.csv")
    #dataframe_to_arff(True,df,arff_file)
    
    #(mean0L,mean1L,good_readings,scoresL)=template_match_filter("D:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets60_validsets20_testsets20\\trainset1_0.csv",
    #                      "438_V1_4_eis_readings_28inputs_with_ids.csv",                          
    #                      filter_option='filter4',
    #                      testset_ids_csv="D:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets60_validsets20_testsets20\\testset1_ids_0.csv",
    #                      good_readings_csv="good_readings.csv"
    #                      )
    #myplot(mean0L,mean1L,1)
    #(m0L,m1L)=compute_mean(good_readings)
    #myplot(m0L,m1L,2)
    #change_between_features_transform('my_selected_unselected_eis_readings.csv','my_selected_unselected_eis_readings_transformed.csv')
    #change_between_features_transform('my_selected_unselected_eis_readings_inversed_labels.csv','my_selected_unselected_eis_readings_inversed_labels_transformed.csv')
    #convert_neg_phase_to_pos_phase('my_selected_unselected_eis_readings.csv','my_selected_unselected_eis_readings_pos_phase.csv')
    #change_between_features_transform('my_selected_unselected_eis_readings_pos_phase.csv','my_selected_unselected_eis_readings_pos_phase_transformed.csv')
    #change_between_features_transform('my_filtered_data_28inputs.csv','my_filtered_data_28inputs_transformed.csv')
    #change_between_features_transform('my_filtered_data_unselected_eis_readings.csv','my_filtered_data_unselected_eis_readings_transformed.csv')
    #change_between_features_transform('438_V1_28inputs.csv','438_v1_28inputs_transformed.csv')
    #convert_neg_phase_to_pos_phase('438_V1_28inputs.csv','438_V1_28inputs_pos_phase.csv')
    #convert_neg_phase_to_pos_phase('my_filtered_data_28inputs.csv','my_filtered_data_28inputs_pos_phase.csv')
    #split_train_valid_and_test_sets_furthest_to_mean('D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs.csv',trainfile='trainset.csv',validfile='validset.csv',testfile='testset.csv')
    #split_train_valid_and_test_sets_furthest_to_mean('my_filtered_data_28inputs.csv',train_fraction=0.66,datatype='csv',outfile='trainset.csv')
    #(pval,auc_ci)=get_p_value_auc_ci('outfile.txt')
    #print(pval)
    #print(auc_ci)
    #weka_path='c:\\Program Files\\Weka-3-7-10\\weka.jar'
    #java_memory='2g'
    #discrete_features_to_binary_features("D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\testset_discrete_integers.arff","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\testset_discrete_binary.arff","28",weka_path,java_memory)
    ###Plot mean0L, mean1L of whole data, training set and test set 
'''
    from preprocess import split_train_test_sets
    #data=pd.read_csv('my_filtered_data_28inputs.csv')
    data=pd.read_csv('D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs.csv')
    cols=list(data.columns)
    seed0=random.randint(0,5**9)
    print(seed0)
    (train_set,test_set)=split_train_test_sets(data,0.34,seed0,cols[-1])
    (mean0L,mean1L)=compute_mean(data)
    import matplotlib.pyplot as plt
    xaxis=[i for i in range(len(mean0L))]
    plt.figure(1)#plot mean of class1 and mean of class0 of whole data
    plt.plot(xaxis,mean0L,'-bo',xaxis,mean1L,'-rs')
    (m0L,m1L)=compute_mean(train_set)
    plt.figure(2)#plot mean of class1and mean of class0 of training set
    plt.plot(xaxis,m0L,'-bo',xaxis,m1L,'-rs')
    (m0L,m1L)=compute_mean(test_set)
    plt.figure(3)#plot mean of class1and mean of class0 of test set
    plt.plot(xaxis,m0L,'-bo',xaxis,m1L,'-rs')    
    plt.show()
'''
    #pval_auc_ci=get_p_value_auc_ci("D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\p_values_auc_ci_log_reg.txt",resultsfiletype='log reg training testing')
    #print(pval_auc_ci)
    #random_forest_training_testing_p_value_and_auc_ci('20',"D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\testset.csv","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\testset.csv","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\results.txt","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\rf.rda")
    #random_forest_testing_auc_ci("D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\rf.rda","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\testset.csv","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\results2.txt")
    #log_reg_training_testing_p_value_and_auc_ci("D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\testset.csv","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\testset.csv","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\results.txt","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\log_reg.rda")
    #log_reg_testing_auc_ci("D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\log_reg.rda","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\testset.csv","D:\\EIS preterm prediction\\results\\workflow1\\my_filtered_data\\results.txt")
    #(m0L,m1L)=compute_mean("D:\\EIS preterm prediction\\results\\workflow1\\combiner_log_reg_of_demographics_treatment_history_obstetric_history_model_and_eis_model\\new_testset_rf_eis_model0.csv",datatype='csv')
    #myplot(m0L,m1L,0)
    #(m0L,m1L)=compute_mean("D:\\EIS preterm prediction\\results\\workflow1\\combiner_log_reg_of_demographics_treatment_history_obstetric_history_model_and_eis_model\\new_trainset_rf_eis_model0.csv",datatype='csv')
    #myplot(m0L,m1L,1)
    #trainset="D:\\EIS preterm prediction\\results\\workflow1\\combiner_log_reg_of_demographics_treatment_history_obstetric_history_model_and_eis_model\\new_trainset_rf_eis_model0.csv"
    #noisy_trainset='noisy_trainset.csv'
    #add_noise_based_on_mean_of_each_class(trainset,noisy_trainset,percent=20)
    #(m0L,m1L)=compute_mean(noisy_trainset,datatype='csv')            
    #myplot(m0L,m1L,0)
    #ahr_v1=pd.read_excel("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\MIS MAC_no compensation.xlsx",sheet_name='AHR_V1')
    #ahr_v2=pd.read_excel("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\MIS MAC_no compensation.xlsx",sheet_name='AHR_V2')
    #symp=pd.read_excel("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\MIS MAC_no compensation.xlsx",sheet_name='SYMP')
    #ahr_v1=pd.read_excel("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\MIS MAC_divide compensation.xlsx",sheet_name='AHR_V1')
    #ahr_v2=pd.read_excel("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\MIS MAC_divide compensation.xlsx",sheet_name='AHR_V2')
    #symp=pd.read_excel("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\MIS MAC_divide compensation.xlsx",sheet_name='SYMP')
    #ahr_v1=pd.read_excel("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\MIS MAC_subtract compensation.xlsx",sheet_name='AHR_V1')
    #ahr_v2=pd.read_excel("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\MIS MAC_subtract compensation.xlsx",sheet_name='AHR_V2')
    #symp=pd.read_excel("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\MIS MAC_subtract compensation.xlsx",sheet_name='SYMP')
    
    #ahr_v1_symp=pd.concat([ahr_v1,symp])
    #ahr_v2_symp=pd.concat([ahr_v2,symp])
    #ahr_v1_symp.to_csv("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\ahr_v1_symp_no_compensation.csv",index=False)
    #ahr_v2_symp.to_csv("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\ahr_v2_symp_no_compensation.csv",index=False)
    #ahr_v1_symp.to_csv("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\ahr_v1_symp_divide_compensation.csv",index=False)
    #ahr_v2_symp.to_csv("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\ahr_v2_symp_divide_compensation.csv",index=False)
    #ahr_v1_symp.to_csv("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\ahr_v1_symp_subtract_compensation.csv",index=False)
    #ahr_v2_symp.to_csv("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\ahr_v2_symp_subtract_compensation.csv",index=False)
    
    #ahr_v1_v2_symp=pd.concat([ahr_v1_symp,ahr_v2])
    #ahr_v1_v2_symp.to_csv("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\ahr_v1_v2_symp_no_compensation.csv",index=False)
    #ahr_v1_v2_symp.to_csv("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\ahr_v1_v2_symp_divide_compensation.csv",index=False)
    #ahr_v1_v2_symp.to_csv("D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\ahr_v1_v2_symp_subtract_compensation.csv",index=False)
    