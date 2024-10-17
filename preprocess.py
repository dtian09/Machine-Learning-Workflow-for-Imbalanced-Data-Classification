'''
Data Preprocessing Algorithms for Machine Learning
written by David Tian
'''
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import re, regex
from sklearn.model_selection import StratifiedShuffleSplit    
import random
import operator
import sys
from collections import Counter
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import gain
import os

def oversample_train_set(train_set,class0_size,class1_size,method,csv_balanced_train_file):
    #inputs: the inputs (X) and outputs (y) of a training set,
    #       csv file name
    #       random_state (random seed)
    #       method of oversampling: random, smote or adasyn
    #       column labels of the training set
    #output: oversampled training set in the named csv file
    #        oversampled training set dataframe
    (_,c)=train_set.shape
    X=train_set.iloc[:,:c-1]#get all columns except the last one
    y=train_set.iloc[:,c-1]            
    column_labels=list(train_set.columns)
    random_state2=random.randint(0,2**32-1)
    if method == 'random':
        print('Create a balanced training set using random sampling with replacement')    
        from imblearn.over_sampling import RandomOverSampler      
        ros = RandomOverSampler(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        #print('selected instances: ',ros.sample_indices_)
    elif method == 'smote':
        print('Create a balanced trainng set using SMOTE')
        import numpy as np
        X=X.values #convert X to numpy array
        y=np.array(y) #convert y to numpy array
        #print(y)
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2)
        X_resampled, y_resampled = sm.fit_resample(X, y)
    else:#adasyn
        print('create a balanced trainng set using ADASYN')
        from imblearn.over_sampling import ADASYN
        X_resampled, y_resampled = ADASYN(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2).fit_resample(X, y)
        print('balanced training set: ',len(y_resampled))
        print(sorted(Counter(y_resampled).items()))
    df1=pd.DataFrame(X_resampled,columns=column_labels[:-1])#the inputs
    df2=pd.DataFrame(y_resampled,columns=[column_labels[-1]])#the last column (targets)
    balanced_trainset=(df1.join(df2))
    if csv_balanced_train_file!='none':
        balanced_trainset.to_csv(csv_balanced_train_file,index=False)
    return balanced_trainset

def normalize_inputs_by_divide_by_max(df):
    (r,c)=df.shape
    inputs=df.iloc[:,0:c-1]
    t = df.iloc[:,c-1]
    maxL=list(inputs.max())
    i=0
    for elem in maxL:
        if elem == 0:
            print('maximum of the column at '+str(i)+'th index is 0')
            return 'divide by zero'
        i+=1
    inputs2=inputs    
    (r2,c2)=inputs.shape
    inputs2=inputs2.astype('float64')#convert data type of each column to float64 so that the decimal places are not removed
    for i in range(r2):
        for j in range(c2):
            val = float(inputs.iat[i,j]) / float(maxL[j])
            inputs2.iat[i,j] = val
    df2=inputs2.join(t,on=inputs2.index)
    return df2

def distance_of_2classes(option,data,indx1,indx2):
    #distances of the 2 classes in the EIS data at frequencies 1 and 9
    if option == 'csv':
        df=pd.read_csv(data)
    elif option == 'df':
        df=data
    else:
        print('invalid option: ',option)
        return None
    (_,c)=df.shape
    cols=df.columns
    preterm=df.loc[lambda f: df[cols[c-1]]==1]
    onterm=df.loc[lambda f: df[cols[c-1]]==0]    
    m1=preterm.mean()
    m2=onterm.mean()
    dist1=np.abs(m1.iloc[indx1]-m2.iloc[indx1]) #distance between mean of amp at freq 1 (indx1=0)
    dist2=np.abs(m1.iloc[indx2]-m2.iloc[indx2]) #distance between mean of phase at freq 9 (indx2=22)    
    #dist1=np.abs(m1.iloc[0]-m2.iloc[0]) #distance between mean of amp at freq 1
    #dist2=np.abs(m1.iloc[22]-m2.iloc[22]) #distance between mean of phase at freq 9
    #print('distance of amplitude at frequency 1: ',dist1)
    #print('distance of phase at frequency 9: ',dist2)
    return (dist1,dist2)

def distance_of_2classes2(option,data,indx1,indx2,indx3,indx4):
    #distances between the 2 classes of metabolite data at features 1, 4, 5 and 8
    if option == 'csv':
        df=pd.read_csv(data)
    elif option == 'df':
        df=data
    else:
        print('invalid option: ',option)
        return None
    (_,c)=df.shape
    cols=df.columns
    preterm=df.loc[lambda f: df[cols[c-1]]==1]
    onterm=df.loc[lambda f: df[cols[c-1]]==0]    
    m1=preterm.mean()
    m2=onterm.mean()
    dist1=np.abs(m1.iloc[indx1]-m2.iloc[indx1]) #distance between means of feature 1
    dist2=np.abs(m1.iloc[indx2]-m2.iloc[indx2]) #distance between means of feature 4
    dist3=np.abs(m1.iloc[indx3]-m2.iloc[indx3]) #distance between mean of feature 5
    dist4=np.abs(m1.iloc[indx4]-m2.iloc[indx4]) #distance between mean of feature 8
    #dist1=np.abs(m1.iloc[0]-m2.iloc[0]) #distance between means of feature 1
    #dist2=np.abs(m1.iloc[3]-m2.iloc[3]) #distance between means of feature 4
    #dist3=np.abs(m1.iloc[4]-m2.iloc[4]) #distance between mean of feature 5
    #dist4=np.abs(m1.iloc[7]-m2.iloc[7]) #distance between mean of feature 8
    print('distance at feature 1: ',dist1)
    print('distance of feature 4: ',dist2)
    print('distance at feature 5: ',dist3)
    print('distance of feature 8: ',dist4)
    return (dist1,dist2,dist3,dist4)

def data_split_using_dists(dist1,dist2,indx1,indx2,data,test_size,random_state,target):
    (train_set,test_set)=split_train_test_sets(data,test_size,random_state,target)
    (d1,d2)=distance_of_2classes('df',train_set,indx1,indx2)
    if (dist1-d1)>0 and (dist2-d2)>0:
        #print(str(dist1-d1))
        #print(str(dist2-d2))
        return (train_set,test_set)
    else:
        #print(str(dist1-d1))
        #print(str(dist2-d2))        
        random_state2=random.randint(0,2**32-1)
        while random_state2 == random_state:
            random_state2=random.randint(0,2**32-1)    
        (train_set,test_set)=data_split_using_dists(dist1,dist2,indx1,indx2,data,test_size,random_state2,target)
        return (train_set,test_set)

def cross_validation_split_and_remove_duplicates_from_valid_set(data,k,random_state2=1):
    #k-fold cross-validation split:
    #output: training sets and validation sets
    train_setL=[]
    val_setL=[]
    train_set_indicesL=[]
    val_set_indicesL=[]
    (_,c)=data.shape
    cols=list(data.columns)
    validset_size=1/k
    split = StratifiedShuffleSplit(n_splits=int(k), test_size=validset_size, random_state=random_state2)
    for train_index, val_index in split.split(data, data[cols[c-1]]):
        valsetdf=data.loc[val_index]
        #print('size of a validation set before removing duplicates: '+str(len(valsetdf.index)))        
        valsetdf=valsetdf.drop_duplicates(keep='first')
        #print('size of the validation set after removing duplicates: '+str(len(valsetdf.index)))        
        train_setL.append(data.loc[train_index])
        val_setL.append(valsetdf)
        train_set_indicesL.append(train_index)
        val_set_indicesL.append(np.array(valsetdf.index))
    return (train_setL, val_setL, train_set_indicesL, val_set_indicesL)
        
def split_train_test_sets(data,test_size2,random_state2,targetVar):
        #stratefied random split
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size2, random_state=random_state2)
        for train_index, test_index in split.split(data, data[targetVar]):
            train_set = data.loc[train_index]
            test_set = data.loc[test_index]
        return (train_set,test_set)
            
def split_train_test_sets2(dataset,trainset_fraction,iterations,results_path):
    data=pd.read_csv(dataset)
    (_,c)=data.shape
    cols=list(data.columns)
    test_size=1-float(trainset_fraction)    
    for i in range(int(iterations)):
        print(i)
        random_state=random.randint(0,2**32-1)
        (train_set,test_set)=split_train_test_sets(data,test_size,random_state,cols[c-1])
        train_set=normalize_inputs_by_divide_by_max(train_set)
        test_set=normalize_inputs_by_divide_by_max(test_set)
        train_set.to_csv(results_path+'trainset'+str(i)+'.csv',index=False)
        test_set.to_csv(results_path+'testset'+str(i)+'.csv',index=False)
        
def select_random_eis_readings(data,option,outfile,skip_ids=True):
    #randomly select an eis reading for each patient
    #data: a dataframe or 438_V1_4_eis_readings_28inputs_normal_shapes.csv
    #option: type of input data (csv file or dataframe)
    #return: a dataframe
    #        outfile, a csv file
    #        no outfile ('none')
    if option=='csv':
        data=pd.read_csv(data)
    (r,c)=data.shape
    cols=list(data.columns)
    readings_hash={}#key=patient id, value=[reading1,...,reading4]
    #put the readings of each patient into a hash table  
    for i in range(r):
        patient_id=data.iat[i,0]
        if skip_ids:
            reading=data.iloc[i,1:c]#skip the patient id
        else:
            reading=data.iloc[i,:]
        reading=list(reading)
        if readings_hash.get(patient_id)==None:
            readings_hash[patient_id]=[reading]
        else:
            l=readings_hash[patient_id]
            l.append(reading)
            readings_hash[patient_id]=l
    n=len(list(readings_hash.keys()))#no. of unique ids
    if skip_ids:
        data2=pd.DataFrame(np.zeros((n,c-1)),columns=cols[1:c])#initialize dataframe
    else:
        data2=pd.DataFrame(np.zeros((n,c)),columns=cols)        
    j=0
    #print('unique ids: '+str(n))        
    for key in list(readings_hash.keys()):
        readings=readings_hash[key]
        n=len(readings)
        if n>1:
            seed0=random.randint(0,5**9)
            random.seed(seed0)
            indx=random.randint(0,n-1)
            reading=readings[indx]
        else:
            reading=readings[0]
        data2.iloc[j,:]=reading
        j=j+1
    if outfile!='none':
        data2.to_csv(outfile,index=False)
    return data2
        
def create_filtered_data(indxL,data_csv,filtered_data_csv):
    #indxL: list of indices of the columns to use as filters
    data=pd.read_csv(data_csv,low_memory=False)
    (r,c)=data.shape
    cols=list(data.columns)
    preterm=data[data[cols[c-1]]==1]
    onterm=data[data[cols[c-1]]==0]
    (lbs,ubs)=ubs_lbs(preterm,indxL)  
    (keepIndx,removeIndx)=filter_data(preterm,lbs,ubs)#filter out the preterm instances outside whiskers of amplutudes 1 to 14
    subset=pd.concat([preterm.iloc[keepIndx,:],onterm])
    subset.to_csv(filtered_data_csv,index=False)
    print('size of dataset after filtering: '+str(r-len(removeIndx)))
    print('removed: '+str(len(removeIndx)))
    removed=preterm.iloc[removeIndx,:]
    preterm2=removed[removed[cols[c-1]]==1]
    onterm2=removed[removed[cols[c-1]]==0]
    print('no. of preterm removed: '+str(len(preterm2)))
    print('no. of onterm removed: '+str(len(onterm2)))
    
def ubs_lbs(preterm_data,indxL):#get the whiskers of the boxplot of a preterm dataframe
    #indxL: list of column indices to find upper and lower bounds
    cols=list(preterm_data.columns)
    #compute whiskers of boxplots
    lbs=[]#list of lower bounds, one per feature
    ubs=[]#list of upper bounds, one per feature
    #outliersL=[]#list of lists of outliers of the whole dataset
    for i in indxL:
        bplot=preterm_data.boxplot(column=cols[i],sym='+', return_type='dict')
        #outliers = [flier.get_ydata() for flier in bplot["fliers"]]
        #boxes = [box.get_ydata() for box in bplot["boxes"]]
        #medians = [median.get_ydata() for median in bplot["medians"]]
        whiskers=[whiskers.get_ydata() for whiskers in bplot["whiskers"]]
        #print(i)
        #print(outliers)
        #print(boxes)
        #print(medians)
        #print(whiskers)
        w1=whiskers[0]#whisker 1
        w2=whiskers[1]#whisker 2
        lb=w1[1]
        ub=w2[1]
        #print(lb)#lower bound
        #print(ub)#upper bound
        lbs.append(lb)
        ubs.append(ub)
        #outliersL.append(list(outliers)) 
    return (lbs,ubs)

def filter_data(dfpreterm,lbs,ubs):#filter out the instances outside upper and lower bounds of each feature
    #return: a list of indices of the instances with feature vals >= lb and <= ub
    keepedIndx=[]
    removedIndx=[]
    (r,_)=dfpreterm.shape
    c=len(lbs)
    for i in range(r):
        keep=True
        for j in range(c):
            val=dfpreterm.iloc[i,j]
            if val<lbs[j] or val>ubs[j]:
                keep=False
                break
        if keep==True:    
            keepedIndx.append(i)
        else:
            removedIndx.append(i)
    return (keepedIndx,removedIndx)

def compare_ids_of_2_datasets(data1_csv,data2_csv):
    #compare the ids of 2 datasets with ids columns by calling compare_2_datasets
    df1=pd.read_csv(data1_csv)
    df2=pd.read_csv(data2_csv)
    cols=list(df1.columns)
    (_,c)=df1.shape
    (_,c2)=df2.shape
    df1=convert_targets(df1,c-1)
    df2=convert_targets(df2,c2-1)
    ids1df=df1.iloc[:,[0,c-1]]#get the ids column and the target column
    cols2=list(df2.columns)
    ids2df=df2.iloc[:,[0,c2-1]]#get the ids column and the target column
    ids=cols[0]
    classes=cols[c-1]
    ids2=cols2[0]
    classes2=cols2[c2-1]
    ids1df.to_csv('ids1.csv',columns=[ids,classes],header=True,index=False)
    ids2df.to_csv('ids2.csv',columns=[ids2,classes2],header=True,index=False)
    compare_2_datasets('ids1.csv','ids2.csv',option='ids_files')
    
def compare_2_datasets(data1_csv,data2_csv,option='data_files'):
    #compare 2 ids files with class columns or 2 datasets without ids columns
    df1=pd.read_csv(data1_csv)
    df2=pd.read_csv(data2_csv)
    if option!='ids_files':#round to decimal places if the input files are not ids
        df1=df1.round(4)
        df2=df2.round(4)
    (r,c)=df1.shape
    cols=list(df1.columns)
    (r2,_)=df2.shape
    print('file1 contains '+str(r)+' rows')
    print('file2 contains '+str(r2)+' rows')
    df4=dataframes_common_instances(df1,df2)
    (same,_)=df4.shape
    print('common rows of file1 and file2: '+str(same))
    class1=df4[df4[cols[c-1]]==1]
    (r4,_)=class1.shape
    print('This includes no. of class1 instances: '+str(r4))
    #print(class1)
    class0=df4[df4[cols[c-1]]==0]
    (r5,_)=class0.shape
    print(' and no. of class0 instances: '+str(r5))
    #print(class0)
    df3=dataframes_diff(df1,df2)
    (r3,c)=df3.shape
    print('file2 does not contain '+str(r3)+' rows of file1:')
    print(df3)
    df3.to_csv('ids_of_file1_which_are_not_in_file2.csv',index=False)
    class1=df3[df3[cols[c-1]]==1]
    (r3,_)=class1.shape
    print('This includes no. of class1 instances: '+str(r3))
    class2=df3[df3[cols[c-1]]==0]
    (r4,_)=class2.shape
    print(' and no. of class0 instances: '+str(r4))    
    
def get_labels(df):
    (_,c)=df.shape
    cols=list(df.columns)
    labels=list(df[cols[c-1]].unique())
    alllabels=set()
    for label in labels:
        alllabels.add(label)
    alllabels=list(alllabels)
    alllabels.sort()
    alllabels2=''
    for label in alllabels:
        alllabels2+=str(label)+','
    alllabels2=alllabels2.rstrip(',')
    return alllabels2

def replace_rows_of_dataframe(rows_indx,df1,df2):
    #replaces the rows of df1 at rows_indx with len(rows_indx) rows of df2 starting from index 0
    #return: df1 updated with the rows of df2
    #assumption: the the length of rows_indx is <= no. of rows in df2
    (_,c)=df1.shape
    i=0#row index of df2
    for indx in rows_indx:
        for j in [k for k in range(c)]:
            df1.iat[indx,j]=df2.iat[i,j]
        i+=1
    cols=list(df1.columns)
    df1=df1.astype({cols[c-1]:int})
    return df1

def get_arff_attributes(arff_file):
    #return: a list of the attributes and the target variables
    f=open(arff_file,'r')
    all_attrs_obtained=False
    attrs=[]
    while all_attrs_obtained==False:
        line=f.readline().rstrip() 
        m=re.match('^@data\s*$',line)
        if m:
            all_attrs_obtained=True
            break;
        else:
            m2=regex.match('^@attribute\s+([\s\p{P}\w\d\^]+)\s+[numericalstg\s\{\}\d\w\p{P}]+$',line)#match continuous attribute, discrete attribute or the target variable (numeric, real or nominal)
            if m2:
                attrs.append(m2.group(1))
            #else:                
            #    m3=regex.match('^@attribute\s+([\s\'\w\d\^]+)\s+[\{\s\w\d\p{P}\}]+$',line)#match discrete attribute and the target variable (numeric or nominal)
            #    if m3:
            #        attrs.append(m3.group(1))
    f.close()
    return attrs

def convert_csv_to_arff(csv_file,arff_file,class_labels,weka_path,java_memory):
    import os
    if class_labels=='numeric':#The outcomes are real values (regression)
        cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.core.converters.CSVLoader \""+csv_file+"\" -B 1000 > \""+arff_file+"\""
        os.system(cmd)
        print(cmd)
    else:#The outcomes are class labels (classification), class_labels="last:label1,label2" (indx start from 1)
        cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.core.converters.CSVLoader \""+csv_file+"\" -B 1000 -L "+class_labels+" > \""+arff_file+"\""
        os.system(cmd)
        print(cmd)
    print('output of convert_csv_to_arff is saved to '+arff_file)

def numeric_target_to_nominal_target(arff_file,arff_file2,weka_path,java_memory):
    #convert numeric target to nominal target
    import os
    cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.filters.unsupervised.attribute.NumericToNominal -R last -i \""+arff_file+"\" -o \""+arff_file2+"\""
    os.system(cmd)
    print(cmd)

def convert_arff_to_csv(arff_file,csv_file,weka_path,java_memory):
    import os
    cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.core.converters.CSVSaver -i \""+arff_file+"\" -o \""+csv_file+"\""
    os.system(cmd)
    #print(cmd)
    
def poly_features(data,deg,csv_outfile='none',interaction_only=True):
    #input: data, a dataframe
    #       deg, degree
    #output: csv_outfile, 'none' (not need to write csv file) or a csv file name
    #        df3, dataframe
    (_,c)=data.shape
    cols=list(data.columns)
    X = data.iloc[:,:c-1]#skip the last column (targets)
    poly = PolynomialFeatures(degree=deg,interaction_only=interaction_only)
    fnames=[]
    for i in range(len(cols)-1):
        fnames.append('x'+str(i))
    X.columns=fnames
    X=poly.fit_transform(X)
    fnames=poly.get_feature_names_out()
    #print('number of polynomial features: ',len(fnames))
    y=data.iloc[:,c-1]
    #y=convert_targets(y)#convert string targets to integers
    df1=pd.DataFrame(X,index=data.index, columns=fnames)
    df2=pd.DataFrame(y,index=data.index, columns=[cols[c-1]])
    df3=df1.join(df2)    
    if csv_outfile!='none':
        df3.to_csv(csv_outfile,index=False)
        #print('output of poly_features is saved to '+csv_outfile)
    return df3

def poly_features_parallel(data,deg,n=10,interaction_only2=True):#construct polynomial features in parallel
    from utilities import split_list
    from joblib import Parallel, delayed
    (r,_)=data.shape
    indxL=[i for i in range(r)]#positional indices of rows
    sublists=split_list('ordered',indxL,n)
    ###debug###
    #for sublist in sublists:
    #   print(len(sublist))
    ###end###
    L=Parallel(n_jobs=-1)(delayed(poly_features_parallel_step)(j,sublists,data,int(deg),interaction_only2=interaction_only2) for j in range(n))        
    ###debug###
    #for df in L:
    #    (r,_)=df.shape
    #    print(r)
    ###end###
    df=pd.concat(L)
    return df

def poly_features_parallel_step(j,sublists,data,deg,interaction_only2=True):
    return poly_features(data.iloc[sublists[j],:],deg,interaction_only=interaction_only2)
    
def poly_features2(datafile,featurestype,model_inputs_output_csv,poly_features_data):
    #construct particular polynomial features for a dataset of original features or polynomial features
    #input: datafile, a dataset containing amplitude and phase features or polynomial features
    #       featurestype of datafile, 'original_features' or 'poly_features'
    #       model_inputs_output_csv (polynomial features) 
    #output: poly_features_data, a dataset containing the same polynomial features as model_inputs_output_csv
    #       dataframe, a dataset containing the same polynomial features as model_inputs_output_csv
    if os.path.isfile(datafile):
        data=pd.read_csv(datafile)
    else:
        data=datafile
    (r,n)=data.shape
    f=open(model_inputs_output_csv,'r')
    poly_features_and_class=f.readline()
    poly_features_and_class=poly_features_and_class.rstrip()#remove the newline
    f.close()
    poly_features_and_class=poly_features_and_class.split(',')
    poly_data=pd.DataFrame(np.zeros((r,len(poly_features_and_class))),columns=poly_features_and_class)    
    if featurestype=='original_features':
        for i in range(len(poly_features_and_class)-1):#construct each poly feature excluding the class variable e.g. x1 x5^2 x3 x6, x2 x3 x8^3 etc.
            poly_feature_col=pd.Series(dtype=float) #empty series
            poly_feature=poly_features_and_class[i]
            m=re.match('^\'{0,1}([x\d\^\s]+)\'{0,1}$',poly_feature)#match 'x1 x5 x3 x6', 'x7 x9', 'x13^2 x15 x26', x1 x5 x3 x6, x7 x9 or x13^2 x15 x26
            if m:
                poly_feature2=m.group(1)
                xi_L=poly_feature2.split(' ')
                for j in range(len(xi_L)):#construct this polynomial feature            
                    m2=re.match('^x(\d+)$',xi_L[j]) #match x0 or x10 etc
                    if m2:
                        indx=int(m2.group(1))#column indx is i for xi (i=0,1,2,3,...)
                        if poly_feature_col.empty==False:#x0 is a middle feature of this polynomial feature
                            poly_feature_col=poly_feature_col * data.iloc[:,indx] 
                        else:#x0 is at the 1st original feature of this polynomial feature
                            poly_feature_col=data.iloc[:,indx]
                    else:
                        m2=re.match('^x(\d+)\^(\d)$',xi_L[j]) #match x0^2 or x10^3 etc
                        if m2:
                            indx=int(m2.group(1))
                            prod=data.iloc[:,indx].pow(int(m2.group(2)))
                            if poly_feature_col.empty==False:#x0^2 is a middle feature of this polynomial feature
                                poly_feature_col=poly_feature_col * prod     
                            else:#x0^2 is at the 1st feature of this polynomial feature
                                poly_feature_col=prod
            else:
                m=re.match('^\'{0,1}x(\d+)\^(\d)\'{0,1}$',poly_feature) #match 'x0^2', 'x10^3', x0^2 or x10^3 etc
                if m:
                    indx=int(m.group(1))
                    poly_feature_col=data.iloc[:,indx].pow(int(m.group(2)))
            poly_data[poly_features_and_class[i]]=poly_feature_col
        poly_data.iloc[:,len(poly_features_and_class)-1]=data.iloc[:,n-1]
    elif featurestype=='poly_features':#input dataset is a poly features dataset, so select the specified poly features
        for poly_feature in poly_features_and_class:
            poly_data[poly_feature]=data[poly_feature]
    else:
        sys.exit('invalid featurestype: ',featurestype)
    if poly_features_data!='none' or poly_features_data!=None:#write poly feature data to file if required
        poly_data.to_csv(poly_features_data,index=False)
    return poly_data
'''
def convert_targets(y):
    #input: a 1-d array of targets 'Yes' and 'No'
    #output: a list of 1 and 0
    y=list(y)
    if y[0]==0 or y[0]==1:#The targets are already 0 and 1. Targets conversion not done by convert_targets.
        return y
    generator=(y[i] for i in range(len(y)))
    i=0
    for elem in generator:
        if elem in [1.0,'Yes','yes','y','Y']:
            y[i]=1
        elif elem in [0.0,'No','no','n','N']:
            y[i]=0
        i=i+1
    return y
'''
def convert_targets(test_set,t):
    #input: a data frame with 'Yes' and 'No' in the last column
    #       t, index of target column
    #output: the data frame with 0 and 1 in the target column
    labels=list(set(test_set.iloc[:,t]))
    cols=list(test_set.columns)
    if len(labels)==2: 
        if test_set.iat[0,t]==0 or test_set.iat[0,t]==1 or test_set.iat[0,t]==2 or test_set.iat[0,t]==3 or test_set.iat[0,t]==4 or test_set.iat[0,t]==5:#The targets are already 0 and 1. Targets conversion not done by convert_targets2
            test_set=test_set.astype({cols[t]:int})
            return test_set
        else:
            for i in range(len(test_set)):
               if test_set.iat[i,t] in [1.0, 'Yes', 'yes', 'y']:
                   test_set.iat[i,t]=1
               elif test_set.iat[i,t] in [0.0, 'No', 'no', 'n']:
                   test_set.iat[i,t]=0
               else:
                   sys.exit('target '+str(t)+' is not a boolean value')             
    else:#multi class, encode multi class labels as integers starting from 0
        l=[i for i in range(len(labels))]
        if labels == l:#multi class labels are integers starting from 0
            return (test_set, labels)
        else:
            labels_dict={}
            for i in range(len(labels)):
                labels_dict[labels[i]]=i
            labelsdf=test_set.iloc[:,t]
            for l in labels:
                labelsdf=labelsdf.replace(l,labels_dict[l])
            test_set.iloc[:,t]=labelsdf
            print(labels_dict)
    test_set=test_set.astype({cols[t]:int})
    return (test_set, labels)

def find_duplicates(df,dupfile_csv):
    duplicateRowsDF=df[df.duplicated()]
    print("Duplicate Rows except first occurrence based on all columns are :")
    duplicateRowsDF.to_csv(dupfile_csv,index=False)

def dataframes_diff2(df1,df2):
    #return: df3 = df1 - df2
    #convert df1 to a set of rows (string)
    (r,_)=df1.shape
    s=set()
    for i in range(r):
        row=df1.iloc[i,:]
        row=list(row)
        row_str=''
        for j in range(len(row)-1):
            row_str+=str(row[j])+','
        row_str+=str(row[-1])
        s.add(row_str)
    #print('s')
    #print(s)
    #convert df2 to a set of rows (strings)
    (r2,_)=df2.shape
    s2=set()
    for i in range(r2):
        row=df2.iloc[i,:]
        row=list(row)
        row_str=''
        for j in range(len(row)-1):
            row_str+=str(row[j])+','
        row_str+=str(row[-1])
        s2.add(row_str)
    #print('s2:')
    #print(s2)
    s3=s.difference(s2)
    s3=list(s3)
    #convert s3 (a list of strings) to a list of lists of strings
    for k in range(len(s3)):
        row=s3[k]
        row=row.split(',')
        s3[k]=row
    cols=list(df1.columns)
    #print(s3)
    return pd.DataFrame(s3,columns=cols)
    
def dataframes_diff(df1,df2):
    #this function only works for df1 and df2 which do not contain ids columns
    #return df3=df1-df2 (df3=rows of df1 which are not in df2)
    df3=df1[~df1.apply(tuple,1).isin(df2.apply(tuple,1))]
    #print(df3)
    return df3

def dataframes_common_instances(df1,df2):
    #get the rows of df1 which are also in df2
    df3=df1[df1.apply(tuple,1).isin(df2.apply(tuple,1))]
    #print(df3)
    return df3

def remove_duplicates(input_data_csv,output_data_csv):
    #Remove duplicate rows from a file
    #input: input_data_file e.g. "D:\\EIS preterm prediction\\resample\\poly features\\select features from whole dataset\\438_V1_28inputs_poly_degree4_balanced_resample_reduced.csv"
    #output: output_data_file: file with no dupliate rows
    df=pd.read_csv(input_data_csv)
    df=df.drop_duplicates(keep='first')
    df.to_csv(output_data_csv,index=False)
    '''
    dataL = [line.strip() for line in open(input_data_csv)]#balanced whole data
    s=set()
    i=1#skip the 1st line in the file
    while i <(len(dataL)):#remove duplicate instances (rows)
        s.add(dataL[i])
        i+=1
    s=list(s)
    fo=open(output_data_csv,"w")#write unique instances to a file
    fo.write(dataL[0]+"\n")#write variable names
    for k in range(len(s)):
        fo.write(s[k]+"\n")
    fo.close()
    '''
def remove_zero_variance_features(arff_data,arff_data2):
    import os
    cmd="java -Xmx3g -cp weka-3-9.jar weka.filters.unsupervised.attribute.RemoveUseless -M 99 -i \""+arff_data+"\" -o \""+arff_data2+"\""
    os.system(cmd)
    print(cmd)
'''
def join_data(datafile,datafile2,outfile):
    #join 2 datasets by a column
    #input: datafile, e.g. '438_V1_28inputs.csv'
    #       datafile2, e.g. 'Demographics_only.csv' 
    #output: outfile e.g. "438_V1_30inputs_demographics.csv"
    data=pd.read_csv(datafile)
    data2=pd.read_csv(datafile2)
    data2=data2.iloc[:,[0,1,2]]#get the hospital id, cervical length and ffn level
    #data3=data2.join(data.set_index('ID'), on='hospital_id', how='inner')
    data3=data2.join(data.set_index('hospital_id'), on='hospital_id', how='inner')
    data3=(data3.drop_duplicates(keep='first'))#keep the 1st occurrence of duplicates
    data3=data3.round(4)#round to 4 decimal places
    data3.to_csv(outfile,index=False)
    print('output of join_data is saved to '+outfile)
'''
def join_data(data1,data2,id_data1,id_data2,datatype='csv'):
    #join 2 datasets by a column
    #input: data1, csv file or dataframe
    #       data2, csv file or dataframe 
    #       idcol1, id column of data1
    #       idcol2, id column of data2
    #output: mergeddata
    if datatype=='csv':
        data1=pd.read_csv(data1)
        data2=pd.read_csv(data2)
    elif datatype=='df':        
        print()
    else:
        sys.exit('invalid datatype: ',datatype)
    data3=data2.join(data1.set_index(id_data1), on=id_data2, how='inner', lsuffix = '_left', rsuffix = '_right')
    data3=(data3.drop_duplicates(keep='first'))#keep the 1st occurrence of duplicates
    #data3=data3.round(4)#round to 4 decimal places
    return data3

def merge_data(datafile,datafile2,outfile):
    #merge datafile and datafile2 vertically
    #assume the targets are 0 or 1
    data=pd.read_csv(datafile,low_memory=False)
    (_,c)=data.shape
    data2=pd.read_csv(datafile2,low_memory=False)
    (_,c2)=data2.shape
    if c!=c2:
        import sys
        sys.exit('datafile and datafile2 have different dimensionalities')
    data=convert_targets(data,c-1)#convert 'Yes' and 'No' to 1 and 0
    data2=convert_targets(data2,c2-1)
    #if data2.columns[0] not in data.columns:#2 datasets have different ids columns
    #   data2.columns=data.columns
    #   print('datafile and datafile2 have different features in merge_data. The features of datafile2 is replaced with those of datafile.')
    merged=pd.concat([data,data2])
    if outfile!='none':
        merged.to_csv(outfile,index=False)
        #print('output of merge_data is saved to '+outfile)
    return merged

def chimerge_discretize(A,C,c,R,arff_input_data,arff_discrete_data):
    import os
    ####options of ChiMerge:
    #-A: significance level e.g. 0.9, 0.95, 0.99
    #-C: class attribute index (indices start from 0)
    #-c: class attribute index (indices start from 1)
    #-R: indices of the continuous attributes to discretize (indices start from 1)
    ###Poly features
    #C='35960' 
    #c='35961'
    #R='1-35960'
    ###EIS + demographics features
    #C='37'   
    #c='38'
    #R='2,5,6,7,8,9-37'
    ###EIS features
    #C='28'
    #c='29'
    #R='1-28'
    cmd="java -Xmx3g -cp weka-3-4-6.jar;. ChiMerge -i \""+arff_input_data+"\" -A "+A+" -C "+C+" -c last -R \""+R+"\" -o \""+arff_discrete_data+"\""
    os.system(cmd)
    #print(cmd)

def entropy_discretize(arff_data,arff_discrete_data):
    import os
    cmd="java -Xmx3g -cp weka-3-9.jar weka.filters.supervised.attribute.Discretize -R first-last -c last -i \""+arff_data+"\" -o \""+arff_discrete_data+"\""
    os.system(cmd)
    #print(cmd)

def equal_width_discretize(arff_data,arff_discrete_data,bins,weka_path,java_memory):
    import os
    cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.filters.unsupervised.attribute.Discretize -B "+str(bins)+" -M -1.0 -R first-last -i \""+arff_data+"\" -o \""+arff_discrete_data+"\""
    os.system(cmd)
    #print(cmd)

def equal_freq_discretize(arff_data,arff_discrete_data,bins,weka_path,java_memory):
    import os
    cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.filters.unsupervised.attribute.Discretize -F -B "+str(bins)+" -M -1.0 -R first-last -i \""+arff_data+"\" -o \""+arff_discrete_data+"\""
    os.system(cmd)
    #print(cmd)

def pki_discretize(arff_data,arff_discrete_data,weka_path,java_memory):
    #Proportional k-interval discretization method (no. of bins of equal frequency method is square root of size of the dataset)
    import os
    cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.filters.unsupervised.attribute.PKIDiscretize -R first-last -i \""+arff_data+"\" -o \""+arff_discrete_data+"\""
    os.system(cmd)
    #print(cmd)
    
def ordinal_encode(arff_discrete_data_integer,arff_discrete_train=None,enc=None,arff_discrete_test=None):
    #Build an encoder using the discrete values of arff_discrete_train
    #Use the encoder to transform arff_discrete_test if arff_discrete_test != None, else transform arff_discrete_train
    #input: arff_discrete_train
    #       arff_discrete_test (optional)
    #output: arff_discrete_data_integer
    from utilities import arff_to_dataframe, dataframe_to_arff
    import re
    if arff_discrete_train!=None:    
        df=arff_to_dataframe(arff_discrete_train)
        (_,c)=df.shape
        X=df.iloc[:,:c-1]
        y=df.iloc[:,c-1]    
    if enc==None:#build an ordinal encoder from discrete training set to encode the discrete values of features
        from sklearn.preprocessing import OrdinalEncoder
        enc=OrdinalEncoder()
        enc.fit(X)
    if arff_discrete_test==None:#encode the discrete training set as integers using the encoder
        X=pd.DataFrame(enc.transform(X),columns=list(df.columns)[:c-1])
        X=fill_missing_values('most_frequent','df',X,has_targets_column=False)
        X=X.astype(int)
    else:#encode the discrete test set as integers using the encoder
        df=arff_to_dataframe(arff_discrete_test)
        (_,c)=df.shape
        X=df.iloc[:,:c-1]
        y=df.iloc[:,c-1]
        y=y.replace(np.nan,'?') #replace nan in targets column with ?   
        X=pd.DataFrame(enc.transform(X),columns=list(df.columns)[:c-1])
        X=fill_missing_values('most_frequent','df',X,has_targets_column=False)
        X=X.astype(int)
    if '?' not in set(y.unique()):
        y=y.replace(to_replace=r'^\"?\'?(\d+)\'?\"?$', value=r'\1', regex=True)#remove any ", ' 
        y=y.astype(int)    
    df=X.join(y)
    dataframe_to_arff(df,arff_discrete_data_integer,ordinal_encoder=enc)
    return enc
   
def discretize_using_cuts(arff_data,discrete_arff_data2,discrete_arff_data,class_path,java_memory):
    import os
    #Discretize arff_data using the cuts of discrete_arff_data2 and save the discretized data to discrete_arff_data (output file)
    if class_path=='.':
        cmd="java -Xmx"+java_memory+" -cp . Discretize \""+arff_data+"\" weka \""+discrete_arff_data2+"\" \""+discrete_arff_data+"\""
    else:
        cmd="java -Xmx"+java_memory+" -cp \""+class_path+"\" Discretize \""+arff_data+"\" weka \""+discrete_arff_data2+"\" \""+discrete_arff_data+"\""
    os.system(cmd)
    #print(cmd)
    
def random_rsfs(class_path,arff_discrete_data,reducts_file,no_of_reducts):
    #random rough set feature selection (RSFS)
    #input: discrete dataset, no. of reducts to find
    #output: a reducts file (file name: arff_discrete_data+".random_rsfs_reducts")
    import os, subprocess
    cmd="java -Xmx4g -cp \""+class_path+"\" Random_RSFS \""+arff_discrete_data+"\" "+no_of_reducts+" windows"
    os.system(cmd)
    print(cmd)
    cmd="move "+arff_discrete_data+".random_rsfs_reducts "+reducts_file
    try:
        subprocess.check_call(cmd,shell=True)
    except subprocess.CalledProcessError:
        cmd="mv "+arff_discrete_data+".random_rsfs_reducts "+reducts_file    
        subprocess.call(cmd,shell=True)
    print(cmd)

def wrapper_es_fs(arff_data=None,cv_fold='5',optimalfeaturesubset_file=None,weka_3_9_4_path=None,java_memory=None,pop_size='20',generations='20',crossover_prob='0.6',mutation_prob='0.1',classifier='log reg',ridge='1.0E-8',trees='20',tree_depth='0',seed='1',no_of_cpu='4'):
    #Evolutionary Search (ES) wrapper feature selection (from Weka 9.4)
    #input: training set in arff format
    #       ES parameters
    #       classifier ('log reg' or 'random forest')
    #       parameters of classifier
    #output: optimalfeaturesubset_file
    #format of optimalfeaturesubset_file (the optimal feature subset does not contain the class index and feature indices start from 1):
    #1,3,5,7
    #
    #java -Xmx5g -cp "c:\Program Files\Weka-3-9-4\weka.jar" weka.Run WrapperSubsetEval -s "weka.attributeSelection.EvolutionarySearch -population-size 20 -generations 20 -init-op 0 -selection-op 1 -crossover-op 0 -crossover-probability 0.6 -mutation-op 0 -mutation-probability 0.1 -replacement-op 0 -seed 1 -report-frequency 20" -i "D:/EIS preterm prediction/filtered_data_28inputs.arff" -F 5 -T 0.01 -R 1 -E AUC -B weka.classifiers.functions.Logistic -- -R 1.0E-8 -M -1 -num-decimal-places 4 
    #java -Xmx5g -cp "c:\Program Files\Weka-3-9-4\weka.jar" weka.Run WrapperSubsetEval -s "weka.attributeSelection.EvolutionarySearch -population-size 20 -generations 20 -init-op 0 -selection-op 1 -crossover-op 0 -crossover-probability 0.6 -mutation-op 0 -mutation-probability 0.1 -replacement-op 0 -seed 1 -report-frequency 20" -i "D:/EIS preterm prediction/filtered_data_28inputs.arff" -F 5 -T 0.01 -R 1 -E AUC -B weka.classifiers.trees.RandomForest -- -P 100 -I 20 -num-slots 4 -K 0 -M 1.0 -V 0.001 -S 1 -depth 0
    import os
    if classifier=='log reg':
        cmd="java -Xmx"+java_memory+" -cp \""+weka_3_9_4_path+"\" weka.Run WrapperSubsetEval -s \"weka.attributeSelection.EvolutionarySearch -population-size "+pop_size+" -generations "+generations+" -init-op 0 -selection-op 1 -crossover-op 0 -crossover-probability "+crossover_prob+" -mutation-op 0 -mutation-probability "+mutation_prob+" -replacement-op 0 -seed 1 -report-frequency 20\" -i \""+arff_data+"\" -F "+cv_fold+" -T 0.01 -R 1 -E AUC -B weka.classifiers.functions.Logistic -- -R "+ridge+" -M -1 -num-decimal-places 4 > \""+optimalfeaturesubset_file+"\"" 
    elif classifier=='random forest':
        cmd="java -Xmx"+java_memory+" -cp \""+weka_3_9_4_path+"\" weka.Run WrapperSubsetEval -s \"weka.attributeSelection.EvolutionarySearch -population-size "+pop_size+" -generations "+generations+" -init-op 0 -selection-op 1 -crossover-op 0 -crossover-probability "+crossover_prob+" -mutation-op 0 -mutation-probability "+mutation_prob+" -replacement-op 0 -seed 1 -report-frequency 20\" -i \""+arff_data+"\" -F "+cv_fold+" -T 0.01 -R 1 -E AUC -B weka.classifiers.trees.RandomForest -- -P 100 -I "+trees+" -num-slots "+no_of_cpu+" -K 0 -M 1.0 -V 0.001 -S "+seed+" -depth "+tree_depth+" > \""+optimalfeaturesubset_file+"\"" 
    else:
        sys.exit('invalid classifier: '+classifier)
    os.system(cmd)
    print(cmd)
    #get the optimal feature subset and print on screen its fitness (cross validation AUC)
    #Selected attributes: 2,7,13,19,25,28 : 6
    if os.path.isfile(optimalfeaturesubset_file)==False:#resultsfile does not exist
        sys.exit(optimalfeaturesubset_file+' does not exist.')
    fileL=[line.strip() for line in open(optimalfeaturesubset_file)]
    file=open(optimalfeaturesubset_file,'w')
    for i in range(len(fileL)):
       line=fileL[i]
       m=re.match('^Current max fitness:\s+([\.\d]+)\s*$',line)   #Current max fitness: 0.8069
       if m:
           cv_auc=m.group(1)
       else:
           m=re.match('^Selected attributes:\s+([\d,]+)\s+:\s+\d+\s*$',line) #Selected attributes: 2,7,13,19,25,28 : 6
           if m:
              featuresubset=m.group(1)
              print(featuresubset+', '+cv_fold+'-fold cross validation auc: '+cv_auc)
              file.write(featuresubset+'\n')
              file.close()
              break
    return optimalfeaturesubset_file

def ga_rsfs(arff_discrete_data,reducts_file,populationSize,generations,crossover_prob,mutation_prob,fitness,class_path,weka_path,results_path,java_memory,platform):
    #single objective genetic algorithm for RSFS
    #output: reducts_file
    #format of reducts_file: features indices start from 1 and class attribute not included
    # Output format:
    # 1, 3, 5, 6, 7
    # 2, 4, 6, 8
  	
    import os, subprocess
    
    results_path=results_path+"\\"
    if fitness=="find_high_info_gain_reducts":
        cmd ="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.attributeSelection.InfoGainAttributeEval -i \""+arff_discrete_data+"\" > \""+results_path+"features_rank.output\""
        os.system(cmd)
        print(cmd)    
    cmd="java -Xmx"+java_memory+" -cp \""+class_path+"\" GA_Reducts \""+arff_discrete_data+"\" "+platform+" "+populationSize+" "+generations+" "+crossover_prob+" "+mutation_prob+" "+fitness+" \""+weka_path+"\" "+java_memory+" \""+results_path+"\""
    os.system(cmd)
    #print(cmd)    
    if platform=="windows":
        cmd="move \""+results_path+"GA_Reducts_resultsfile2\" \""+reducts_file+"\"" 
        subprocess.call(cmd,shell=True)
        #print(cmd)
        if fitness=="find_high_info_gain_reducts":
            results_path=convert_to_windows_path(results_path)
            cmd="del \""+results_path+"\\features_rank.output\""
            subprocess.call(cmd,shell=True)
            #print(cmd)  
    else:#linux
        cmd="mv \""+results_path+"GA_Reducts_resultsfile2\" \""+reducts_file+"\""
        subprocess.call(cmd,shell=True)
        #print(cmd)
        if fitness=="find_high_info_gain_reducts":
            cmd="rm \""+results_path+"features_rank.output\""
            subprocess.call(cmd,shell=True)
            #print(cmd)
                
def gls_rsfs(class_path,arff_discrete_trainset,arff_continuous_trainset,arff_continuous_testset,reducts_file):
    #3-objective genetic local search algorithm for RSFS
    import os
    cmd="java -Xmx3g -cp \""+class_path+"\" Hybrid_NSGAII \""+arff_discrete_trainset+"\" \""+arff_continuous_trainset+"\" \""+arff_continuous_testset+"\" windows > \""+reducts_file+"\""
    os.system(cmd)
    print(cmd)

def parallel_feature_select(data,reduced_data_arff,m=10):
    #select m best features in parallel based on mutual information and reduce the data using the selected features
    #1. split a dataframe of n features into m sub-dataframes of n/m features
    #2. Select the top k features from each sub-dataframe in parallel using information gain. This selects k*m features.
    #3. Select the m best features from the k*m features
    #4. Reduce the dataframe using the feature subset
    from utilities import split_list,dataframe_to_arff#,delete_files
    from joblib import Parallel, delayed
    (_,c)=data.shape
    cols=list(data.columns)
    target=cols[c-1]
    del cols[c-1]
    if len(cols) > 1000:#dimensionality > 1000
        sublists=split_list('random',cols,m)
        #print('length of sublists:',len(sublists))
        ###debug###
        #for i in range(len(sublists)):
            #sublist=sublists[i]
            #print(str(i)+': '+str(len(sublist)))
            #print(sublist)
        ###end###
        #featuresL=Parallel(n_jobs=-1)(delayed(select_top_feature_weka_parallel_step)(j,sublists,data,results_path,weka_path,java_memory) for j in range(int(k)))        
        #featuresL=Parallel(n_jobs=-1,batch_size=2,max_nbytes='5G')(delayed(select_top_feature_parallel_step)(j,sublists,data) for j in range(int(k)))        
        #featuresL=Parallel(n_jobs=-1)(delayed(select_top_feature_parallel_step)(j,sublists,data) for j in range(len(sublists)))        
        selectedfeaturesL=Parallel(n_jobs=-1,batch_size=2)(delayed(select_top_k_features_parallel_step)(j,sublists,data,k=10) for j in range(len(sublists)))        
        #L= a list of m lists of selected k best features
        selectedfeaturesL2=[]
        for selectedfeatures in selectedfeaturesL:
            selectedfeaturesL2=selectedfeaturesL2 + selectedfeatures
        selectedfeaturesL2.append(target)
        reduced_data=gain.select_top_k_features_parallel('info_gain',data[selectedfeaturesL2],m,True)
        dataframe_to_arff(reduced_data,reduced_data_arff)    
    else:#dimesionality <= 1000
        reduced_data=gain.select_top_k_features_parallel('info_gain',data,m,True)
        dataframe_to_arff(reduced_data,reduced_data_arff)
    
def select_top_k_features_parallel_step(j,sublists,data,k=1):    
    (_,c)=data.shape
    cols=list(data.columns)
    features=sublists[j]
    X=data[features]
    y=data[cols[c-1]]    
    fs=SelectKBest(mutual_info_classif, k=k).fit(X,y)
    features=np.array(features)
    selectedfeatures=features[fs.get_support()]
    del [X,y]
    return list(selectedfeatures)

def select_top_feature_weka_parallel_step(j,sublists,data,results_path,weka_path,java_memory):    
    (_,c)=data.shape
    cols=list(data.columns)
    subset_arff = results_path+'subset_'+str(j)+'.arff'
    features=sublists[j]
    features.append(cols[c-1])
    print(str(j))
    ###debug###
    #df=data[features]
    #df.to_csv(results_path+'subset_'+str(j)+'.csv',index=False)
    #(r,c)=df.shape
    #print(str(r)+', '+str(c))
    ####
    from utilities import dataframe_to_arff
    dataframe_to_arff(data[features],subset_arff)
    #df=data[features]
    #(r,c)=df.shape
    #print('df '+str(j)+': '+str(r)+', '+str(c))
    #df.to_csv(results_path+'subset_'+str(j)+'.csv',index=False)
    #labels=get_labels(df)    
    #print('labels: ',labels)
    #convert_csv_to_arff(results_path+'subset_'+str(j)+'.csv',subset_arff,'last:'+labels,weka_path,java_memory)
    outfile = results_path+'outfile_'+str(j)+'.txt'
    info_gain_fs2('1',subset_arff,outfile,weka_path,java_memory)
    features_scoresL=get_features_scores_from_weka_output_file(outfile)
    #features_scoresL=[(f1,score1)]
    (feature,score)=features_scoresL[0]
    if feature not in cols:
        feature='\''+feature+'\''#add single quotes to each feature
    return feature

def get_features_scores_from_weka_output_file(outfile):
	#format: 0.0204702    17 diag_1
    import re
    features_scoresL=[]
    fileL=[line.strip() for line in open(outfile)]
    for i in range(len(fileL)):
       line=fileL[i]
       m=re.match('^\s*([\d\.]+)\s+[\d]+\s+([\'_\w\s\^]+)$',line)
       if m:
           feature_score = m.group(1)
           feature = m.group(2)
           features_scoresL.append((feature,feature_score))
    return features_scoresL

#def info_gain_fs(No_of_features,arff_data,reduced_arff_data,weka_path,java_memory):
    ###select a specified number of top-ranked features using information gain and reduce the data using the selected features
    #import os
    #cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.filters.supervised.attribute.AttributeSelection -E \" weka.attributeSelection.InfoGainAttributeEval\" -c last -i \""+arff_data+"\" -S \"weka.attributeSelection.Ranker -N "+str(No_of_features)+"\" > \""+reduced_arff_data+"\""
    #os.system(cmd)
    #print(cmd)
def info_gain_fs(data,k,reduced_data_arff=None):
    (_,c)=data.shape
    cols=list(data.columns)
    features=cols[0:-1]
    X=data[features]
    y=data[cols[c-1]]    
    fs=SelectKBest(mutual_info_classif, k=int(k)).fit(X,y)
    features=np.array(features)
    selectedfeatures=features[fs.get_support()]
    from utilities import dataframe_to_arff
    selectedfeatures=list(selectedfeatures)
    selectedfeatures.append(cols[c-1]) #add class column
    #print(selectedfeatures)
    del [X,y]
    if dataframe_to_arff!=None:
        dataframe_to_arff(data[selectedfeatures],reduced_data_arff)
    return data[selectedfeatures]
        
def info_gain_fs2(k,arff_data,outfile,weka_path,java_memory):
    ###select k top-ranked features using information gain
    #output: a ranking of the selected k features (outfile)
    import os
    cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.attributeSelection.InfoGainAttributeEval -s \"weka.attributeSelection.Ranker -N "+k+"\" -c last -i \""+arff_data+"\" > \""+outfile+"\""
    os.system(cmd)
    #print(cmd)
    
def info_gain_fs_threshold(threshold,arff_data,reduced_arff_data,weka_path,java_memory):
    import os
    cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.filters.supervised.attribute.AttributeSelection -E \" weka.attributeSelection.InfoGainAttributeEval\" -c last -i \""+arff_data+"\" -S \"weka.attributeSelection.Ranker -T "+str(threshold)+"\" > \""+reduced_arff_data+"\""
    os.system(cmd)
    #print(cmd)

def reduce_a_weka_file(reduct_indices, weka_path, weka_file, reduced_weka_file,java_memory):
    #indices of reduct_indices start from 1 and reduct_indices does not include the class variable index 
    import os
    cmd = "java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.filters.unsupervised.attribute.Remove -R \""+reduct_indices+",last\" -V -i \""+weka_file+"\" -o \""+reduced_weka_file+"\""
    os.system(cmd)    
    #print(cmd)
 
def reduce_data(class_path,arff_data,reducts_file,arff_reduced_data,no_of_reducts):
    #Reduce a arff dataset using reducts
    #The reduced datasets are saved as arff_reduced_data+i+".arff" (i=0,1,2,...,no_of_reducts-1)
    import os
    cmd="java -Xmx2g -cp \""+class_path+"\" ReduceWekaData \""+arff_data+"\" \""+reducts_file+"\" \""+arff_reduced_data+"\" "+str(no_of_reducts)#ReduceWekaData.java appends '0.arff' to the name of the reduced data
    os.system(cmd)
    #print(cmd)

def reduce_data2(datatype,data,csv_data2,csv_reduced_data):
    #Reduce data using the features of csv_data2 (e.g. model_inputs_output_csv) and save the reduced data to csv_reduced_data
    import re, sys
    if datatype == 'csv':#input data is a csv file
        data=pd.read_csv(data)
        (r,_)=data.shape    
    elif datatype == 'df':#input data is a dataframe
        (r,_)=data.shape
    else:
        sys.exit('invalid datatype: ',datatype)
    f=open(csv_data2,'r')
    features_and_class=f.readline()
    features_and_class=features_and_class.rstrip()#remove the newline
    f.close()
    features_and_class=features_and_class.split(',')
    feature_subset=[]
    for feature in features_and_class:
        if feature in data.columns:
            feature_subset.append(feature)
        else:
            m=re.match('^\'(.+)\'$',feature)
            if m:
                f=m.group(1)#get rid of the single quotes around a feature name
                feature_subset.append(f)
            else:
                sys.exit(feature+' does not match pattern')
    data_reduced=data[feature_subset]
    data_reduced.columns=features_and_class
    if csv_reduced_data!='none':           
        data_reduced.to_csv(csv_reduced_data,index=False)
    return data_reduced

def reduce_data_using_another_data(arff_data,arff_data2,arff_reduced_data):
    #Reduce arff_data (with poly features) using the features of arff_data2 (with poly features) and save the reduced data to arff_reduced_data
    import os
    cmd="java ReduceWekaData3 \""+arff_data+"\" \""+arff_data2+"\" \""+arff_reduced_data+"\" windows" 
    os.system(cmd)
    print('output of reduce_data_using_another_data is saved to '+arff_reduced_data)        
    print(cmd)
    
def reduce_data_using_another_data2(csv_data,arff_data,arff_reduced_data,csv_reduced_data):
    #Reduce csv_data (with poly features) using the features of arff_data (with poly features) and save the reduced data to arff_reduced_data
    import os
    cmd="java ReduceData2 "+csv_data+" "+arff_data+" "+arff_reduced_data+" "+csv_reduced_data+"  windows"
    os.system(cmd)
    print('output of reduce_data_using_another_data2 is saved to '+arff_reduced_data+' and '+csv_reduced_data)        
    print(cmd)
'''    
def split_training_validation_sets(data_path,train_path,valid_path):
    #split a dataset into a training set and validation set
    import random
    n=16#no. of onterm instances in the validation set
    n2=6#no. of preterm instances in the validation set
    data=pd.read_csv(data_path,low_memory=False)
    onterm=data.loc[lambda df: data['before37weeksCell']==0]
    preterm=data.loc[lambda df: data['before37weeksCell']==1]
    (r,_)=onterm.shape #get number of rows
    k=0
    validset_onterm=pd.DataFrame(columns=data.columns)
    for i in range(n):
       indx=random.randint(0,r-1)   
       validset_onterm=validset_onterm.append(onterm.iloc[indx,:])
       k+=1
    (r2,_)=preterm.shape #get number of rows
    k=0
    validset_preterm=pd.DataFrame(columns=data.columns)
    for i in range(n2):
       indx=random.randint(0,r2-1)
       validset_preterm=validset_preterm.append(preterm.iloc[indx,:])
       k+=1
    validset=validset_onterm.append(validset_preterm)
    validset.to_csv(valid_path,index=False)   
    remove_data(data_path,valid_path,train_path)
'''
def remove_data(file1,file2,remainfile):
    df1=pd.read_csv(file1) 
    df2=pd.read_csv(file2)
    df1=df1.round(6) 
    df2=df2.round(6)
    df3=dataframes_diff(df2,df1)#df3=df2-df1
    df3.to_csv(remainfile,index=False) 

'''    
def remove_data(datafile,datafile2,outfile):
    #remove datafile from datafile2 to get the remaining subset
    #input: datafile, a csv data file with classes 1 or 0
    #       datafile2, a csv data file with classes 1 or 0
    data=pd.read_csv(datafile,low_memory=False)
    cols1=list(data.columns)
    if cols1[0] == 'hospital_id' or cols1[0] == 'ID':
        data.iloc[:,1:len(cols1)]=data.iloc[:,1:len(cols1)].round(6)
    else:#no ids column            
        data=data.round(6)#round to 6 decimal places
    #print(data.iloc[:,len(list(data.columns))-1])
    data2=pd.read_csv(datafile2,low_memory=False)
    cols2=list(data2.columns)
    if cols2[0] == 'hospital_id' or cols2[0] == 'ID':
        data2.iloc[:,1:len(cols2)]=data2.iloc[:,1:len(cols2)].round(6)
    else:#no ids column            
        data2=data2.round(6)#round to 6 decimal places
    (_,c)=data.shape
    data=convert_targets(data,c-1)
    (_,c2)=data2.shape
    data2=convert_targets(data2,c2-1)
    (_,c)=data.shape
    cols=list(data.columns)
    data[cols[c-1]]=data[cols[c-1]].astype(int)#replace 1.0 with 1 and 0.0 with 0 in targets   
    data2[cols[c-1]]=data2[cols[c-1]].astype(int)#replace 1.0 with 1 and 0.0 with 0 in targets   
    #print(data2['before37weeksCell'])
    data.to_csv(datafile,index=False)
    data2.to_csv(datafile2,index=False)
    dataL=[line.strip() for line in open(datafile)]
    data2L=[line.strip() for line in open(datafile2)]
    s1=set(dataL)#duplicate instances are removed from datafile
    s2=set(data2L)#duplicate instances are removed from datafile2
    s3=s2.difference(s1)
    l=list(s3)
    #store the subset in a dataframe and write to a file
    data3=pd.DataFrame(np.zeros((len(l),len(data.columns))),columns=data.columns)
    data3=data3.astype(object)
    for i in range(len(l)):
        l2=l[i].split(',')
        for j in range(len(l2)-1):
            if j > 0:
                data3.iloc[i,j]=float(l2[j])
            elif j==0:
                data3.iloc[i,j]=l2[j]#add hospital_id to dataframe
        data3.iloc[i,len(l2)-1]=l2[len(l2)-1]
    if outfile!='none':#save the remaining subset to outfile, otherwise only return the subset as a dataframe
        data3.to_csv(outfile,index=False)
        print(datafile+' is removed from '+datafile2+'. Output of remove_data is saved to '+outfile)
    return data3
'''
def resample(data,data2,seed,z,weka_path,java_memory):
    #create a balanced dataset using random sampling with replacement
    #Return the balanced dataset data2
    #seed: sampling seed
    #z: sample size percentage
    import os
    cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\" weka.filters.supervised.instance.Resample -B 1.0 -c last -S "+str(seed)+" -Z "+str(z)+" -i \""+data+"\" -o \""+data2+"\"" 
    os.system(cmd)
    #print(cmd)
    
def smote(data,data2,seed,p,weka_path,java_memory):
    import os
    #p: percentage of miniority instances to create for SMOTE
    cmd="java -Xmx"+java_memory+" -cp \""+weka_path+"\"  weka.filters.supervised.instance.SMOTE -C 0 -c last -K 5 -P "+str(p)+" -S "+str(seed)+" -i \""+data+"\" -o \""+data2+"\"" 
    os.system(cmd)
    #print(cmd)

def flatten(list_of_lists):
    flat_list=[]
    for sublist in list_of_lists:
      for item in sublist:
          flat_list.append(item)
    return flat_list
                   
def fill_missing_values(new_value,datatype,data,outfile=None,has_targets_column=True):
    #fill missing values of features with new_value
    #input: new_value e.g. median
    #       data, input data
    #       datatype, 'df' or 'csv'
    #output: dataframe with missing values replaced
    #        a csv file with missing values replaced    
    from sklearn.impute import SimpleImputer
    if new_value == 'mean':
        imp_data=SimpleImputer(missing_values=np.nan,strategy='mean')
    elif new_value == 'median':
        imp_data=SimpleImputer(missing_values=np.nan,strategy='median')
    elif new_value == 'most_frequent':
        imp_data=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    else:
         sys.exit('new_value is invalid: '+new_value)
    if datatype == 'csv':
        data=pd.read_csv(data)
        features=find_features_of_missing_values('df',data,has_targets_column=has_targets_column)
        for i in range(len(features)):
            a=np.array(data[features[i]])
            col_no_missing=imp_data.fit_transform(a.reshape(-1,1))
            data[features[i]]=pd.Series(flatten(list(col_no_missing)),name=features[i],index=data.index)
            #data[features[i]].fillna(pd.Series(flatten(list(col_no_missing)),index=data.index),inplace=True)
    elif datatype == 'df':
        features=find_features_of_missing_values('df',data,has_targets_column=has_targets_column)
        #print('features with missing values: ',features)
        for i in range(len(features)):
            a=np.array(data[features[i]])
            col_no_missing=imp_data.fit_transform(a.reshape(-1,1))
            #print('col_no_missing: ',pd.Series(flatten(list(col_no_missing)),name=features[i]))            
            data[features[i]]=pd.Series(flatten(list(col_no_missing)),name=features[i],index=data.index)
            #data[features[i]].fillna(pd.Series(flatten(list(col_no_missing)),index=data.index),inplace=True)
    else:
        sys.exit('datatype is invalid: '+datatype)
    if outfile != None and outfile!='none':
        data.to_csv(outfile,index=False)
    return data

def find_features_of_missing_values(datatype,data,has_targets_column=True):
    fs=[]#list of names of the features containing missing values 
    if datatype == 'csv':
        data=pd.read_csv(data)
    elif datatype != 'df':
        sys.exit('datatype is invalid: '+datatype) 
    (_,c)=data.shape
    cols=list(data.columns)
    if has_targets_column:
        c=c-1 #exclude the targets column
    for j in range(c):
        l=data.iloc[:,j].isnull()
        if np.isin(True,l):
            fs.append(cols[j])
    return fs

def complex_convert_i_to_j(num):
    #convert complext number a+bi to a+bj as Python only recognise the latter format
    import re
    m=re.match('^([^\s]+)\s*([\-\+])\s*([^\s]+)i$',num)
    if m:
        return m.group(1)+m.group(2)+m.group(3)+'j'
    else:
        m=re.match('^\s*0\s*$',num)
        if m:
            return 'zero'
        else:
            return num

def reformat_complex_number(num,decimal_places):
    #reformat 36.2458500000000 - 0.730011000000000i to 36.2459-0.7300i
    import re
    m=re.match('^([^\s]+)\s*([\-\+])\s*([^\s]+)i$',num)
    if m:
        return str(np.round(float(m.group(1)),decimal_places))+m.group(2)+str(np.round(float(m.group(3)),decimal_places))+'i'
    else:
        #print('real number: '+num)
        return 'real_number'
    
def reformat_eis_readings(eis_file_csv,decimal_places,out_file_csv):
    #reformat 36.2458500000000 - 0.730011000000000i to 36.2459-0.7300i
    data=pd.read_csv(eis_file_csv)
    (r,c)=data.shape
    rows=data.index #list of row labels
    data2=data        
    for i in range(r):
        for j in range (14):
            eis=data.at[rows[i],'EIS'+str(j+1)]                
            eis=reformat_complex_number(eis,int(decimal_places))
            data2.at[rows[i],'EIS'+str(j+1)]=eis
    data2.to_csv(out_file_csv,index=False)
    return data2

def select_ith_reading(file,outfile,k=0):
    #select the ith reading of a patient id i=0,1,2 or 3 (if a patient id has 4 readings)
    #input: file, all readings of all patient ids e.g. 438_V1_4_eis_readings.csv
    #put all readings into a hashtable
    print('select '+str(k)+' reading for each patient id')
    df=pd.read_csv(file)
    readings_hash={}
    rows=list(df.index)
    for row in rows:
        patient_id=df.at[row,'hospital_id']
        reading=list(df.loc[row])
        if readings_hash.get(patient_id)==None:
            readings_hash[patient_id]=[reading]
        else:
            l=readings_hash[patient_id]
            l.append(reading)
            readings_hash[patient_id]=l
    #select ith reading for each patient id
    file=open(outfile,'w+')    
    cols=['hospital_id']
    for i in range(14):
        cols.append('EIS'+str(i+1))
    cols.append('before37weeksCell')
    for i in range(len(cols)-1):
        file.write(cols[i]+',')#write the columns names
    file.write(cols[len(cols)-1]+'\n')
    for patient_id,readings in readings_hash.items():
        if len(readings)==1:#a patient id has 1 reading
            reading=readings[0]
            if '0' not in reading:
                for j in range(len(reading)-1):
                    file.write(str(reading[j])+',')
                file.write(str(reading[len(reading)-1])+'\n')
        else:
            reading=readings[k]
            if '0' in reading:#select next reading if kth reading contains 0
                print(reading)
                reading=readings[k+1]                
                for j in range(len(reading)-1):
                    file.write(str(reading[j])+',')
                file.write(str(reading[len(reading)-1])+'\n')
    file.close()
            
def select_eis_reading_by_real_parts(freqL,file,outfile,option='max'):
    #select readings with the largest real part at frequency 1 or select readings with the largest sum of real parts at frequencies 
    #freqL: list of frequencies starting 0 (frequency 1)
    #e.g. select eis readings by real part of frequency 1 only, set freqL=[0]
    #     select eis readings by sum of real part of frequency 1 and 2, set freqL=[0,1] 
    (df2,df)=get_real_imag(file,'none')
    preterm_hash={}#key=patient id, value=[(amplitude of frequency 1 of reading1,impedance of reading1),...]
    onterm_hash={}
    rows=df.index
    (r,_)=df.shape
    print('Select eis readings of '+option+' real parts of the frequencies: '+str(freqL)+' for preterm and vice versa for onterm')
    for i in range(r):
        patient_id=df.at[rows[i],'hospital_id']
        s=0
        for j in freqL:
            realpart_j = df.at[rows[i],'real_part_EIS'+str(j+1)]
            s+=realpart_j
        label=df.at[rows[i],'before37weeksCell']
        reading=list(df2.iloc[i,:])#complex impedance reading
        if int(label) == 1:
            if preterm_hash.get(patient_id)==None:
                preterm_hash[patient_id]=[(s,reading)]
            else:
                l=preterm_hash[patient_id]
                l.append((s,reading))
                preterm_hash[patient_id]=l
        else:        
            if onterm_hash.get(patient_id)==None:
                onterm_hash[patient_id]=[(s,reading)]
            else:
                l=onterm_hash[patient_id]
                l.append((s,reading))
                onterm_hash[patient_id]=l
    file=open(outfile,'w+')    
    cols=['hospital_id']
    for i in range(14):
        cols.append('EIS'+str(i+1))
    cols.append('before37weeksCell')
    for i in range(len(cols)-1):
        file.write(cols[i]+',')#write the columns names
    file.write(cols[len(cols)-1]+'\n')
    for patient_id,readings in preterm_hash.items():
        readings.sort(key=operator.itemgetter(0),reverse=True)#sort in descending order of sum of real parts  
        if option=='max':#select highest real part at frequency 1 for preterm
            max_realparts_reading=readings[0]
            reading=max_realparts_reading[1]
            for i in range(len(reading)-1):
                file.write(str(reading[i])+',')
            file.write(str(reading[len(reading)-1])+'\n')
        elif option=='2nd max':#2nd highest real part at frequency 1 for preterm and 2nd lowest real part at frequency 1 for onterm
            if len(readings) > 1:
                second_max_realparts_reading=readings[1]
                reading=second_max_realparts_reading[1]
                for i in range(len(reading)-1):
                    file.write(str(reading[i])+',')
                file.write(str(reading[len(reading)-1])+'\n')
        else:
            sys.exit('in select_eis_reading_by_real_parts invalid option: ',option)
    for patient_id,readings in onterm_hash.items():
        readings.sort(key=operator.itemgetter(0),reverse=False)#sort in ascending order of sum of real parts
        if option=='max':#select lowest real part at frequency 1 for onterm       
            min_realparts_reading=readings[0]
            reading=min_realparts_reading[1]
            for i in range(len(reading)-1):
                file.write(str(reading[i])+',')
            file.write(str(reading[len(reading)-1])+'\n')
        elif option=='2nd max':#2nd highest real part at frequency 1 for preterm and 2nd lowest real part at frequency 1 for onterm
            if len(readings) > 1:
                second_min_realparts_reading=readings[1]
                reading=second_min_realparts_reading[1]
                for i in range(len(reading)-1):
                    file.write(str(reading[i])+',')
                file.write(str(reading[len(reading)-1])+'\n')
        else:
            sys.exit('in select_eis_reading_by_real_parts invalid option: ',option)    
    file.close()

def select_eis_reading_by_imag_parts(file,outfile):
    #select readings with the largest imaginary part at frequency 6
    (df2,df)=get_real_imag(file,'none')
    preterm_hash={}
    onterm_hash={}
    rows=df.index
    (r,_)=df.shape
    for i in range(r):
        patient_id=df.at[rows[i],'hospital_id']
        imagpart6 = df.at[rows[i],'imag_part_EIS6']
        label=df.at[rows[i],'before37weeksCell']
        reading=list(df2.iloc[i,:])#complex impedance reading
        if int(label) == 1:
            if preterm_hash.get(patient_id)==None:
                preterm_hash[patient_id]=[(imagpart6,reading)]
            else:
                l=preterm_hash[patient_id]
                l.append((imagpart6,reading))
                preterm_hash[patient_id]=l
        else:        
            if onterm_hash.get(patient_id)==None:
                onterm_hash[patient_id]=[(imagpart6,reading)]
            else:
                l=onterm_hash[patient_id]
                l.append((imagpart6,reading))
                onterm_hash[patient_id]=l
    file=open(outfile,'w+')    
    cols=['hospital_id']
    for i in range(14):
        cols.append('EIS'+str(i+1))
    cols.append('before37weeksCell')
    for i in range(len(cols)-1):
        file.write(cols[i]+',')#write the columns names
    file.write(cols[len(cols)-1]+'\n')
    for patient_id,readings in preterm_hash.items():
        readings.sort(key=operator.itemgetter(0),reverse=False)#sort in ascending order of imaginary parts  
        min_imagpart6_reading=readings[0]
        reading=min_imagpart6_reading[1]
        for i in range(len(reading)-1):
            file.write(str(reading[i])+',')
        file.write(str(reading[len(reading)-1])+'\n')
    #print('onterm')
    for patient_id,readings in onterm_hash.items():
        readings.sort(key=operator.itemgetter(0),reverse=True)#sort in descending order of imaginary parts
        max_imagpart6_reading=readings[0]
        reading=max_imagpart6_reading[1]
        for i in range(len(reading)-1):
            file.write(str(reading[i])+',')
        file.write(str(reading[len(reading)-1])+'\n')
    file.close()
    
def select_eis_reading_by_amplitude1():
    #select the reading with the largest amplitude at frequency 1 and create selected_unselected_data_csv
    infile="438_V1_4_eis_readings.csv"
    outfile1="438_V1_28inputs_by_amp1_with_ids.csv"#amplitude and phase of selected eis readings
    outfile2="438_V1_eis_readings_by_amp1.csv"#the complex impedance of selected eis readings
    outfile3="438_V1_4_eis_readings_28inputs_with_ids.csv"
    selected_unselected_data_csv="U:\\EIS preterm prediction\\selected_unselected_438_V1_28inputs_by_amp1_with_ids.csv"#
    (df2,df)=amp_phase(infile,outfile3,True)
    (r,c)=df.shape
    preterm_hash={}#key=patient id, value=[(amplitude of frequency 1,list of amplitude and phase of reading1,EIS of reading1),(amplitude of frequency 1,list of amplitude and phase of reading2,EIS of reading2),...,]
    onterm_hash={}
    rows=df.index
    for i in range(r):
        patient_id=df.at[rows[i],'hospital_id']
        amp1=df.at[rows[i],'27_EIS_Amplitude1']
        amps_phase=list(df.iloc[i,:])
        ei_reading=list(df2.iloc[i,:])#complex impedance at 14 frequencies
        label=df.at[rows[i],'before37weeksCell']
        if int(label) == 1:#preterm
            if preterm_hash.get(patient_id)==None:
                preterm_hash[patient_id]=[(amp1,amps_phase,ei_reading)]
            else:
                l=preterm_hash[patient_id]
                l.append((amp1,amps_phase,ei_reading))
                preterm_hash[patient_id]=l
        else:        
            if onterm_hash.get(patient_id)==None:
                onterm_hash[patient_id]=[(amp1,amps_phase,ei_reading)]
            else:
                l=onterm_hash[patient_id]
                l.append((amp1,amps_phase,ei_reading))
                onterm_hash[patient_id]=l
    file=open(outfile1,'w+')#write amplitude and phase of the selected eis readings to file
    file2=open(outfile2,'w+') #write eis readings to file2
    cols=['hospital_id']
    for i in range(14):
        cols.append('27_EIS_Amplitude'+str(i+1))
    for i in range(14):
        cols.append('27_EIS_Phase'+str(i+1))
    cols.append('before37weeksCell')
    for i in range(len(cols)-1):
        file.write(cols[i]+',')#write the columns names
    file.write(cols[len(cols)-1]+'\n')
    cols2=['hospital_id']
    for i in range(14):
        cols2.append('EIS'+str(i+1))
    cols2.append('before37weeksCell')
    for i in range(len(cols2)-1):
        file2.write(cols2[i]+',')#write the columns names
    file2.write(cols2[len(cols2)-1]+'\n')
    for patient_id,readings in preterm_hash.items():#for each preterm patient, select the spectrum with the largest amplitude at lowest frequency
        readings.sort(key=operator.itemgetter(0),reverse=True)#sort in descending order of amplitude of frequency 1  
        max_amp1_reading=readings[0]
        amps_phase=max_amp1_reading[1]
        for i in range(len(amps_phase)-1):
            file.write(str(amps_phase[i])+',')#write amplitude and phase of the selected eis readings to file1
        file.write(str(amps_phase[len(amps_phase)-1])+'\n')
        ei_reading=max_amp1_reading[2]
        for i in range(len(ei_reading)-1):
            file2.write(ei_reading[i]+',')#write eis readings to file2
        file2.write(str(ei_reading[len(ei_reading)-1])+'\n')
    #print('onterm')
    for patient_id,readings in onterm_hash.items():#for each onterm patient, select the spectrum with the smallest amplitude at lowest frequency
        readings.sort(key=operator.itemgetter(0),reverse=False)#sort in ascending order of amplitude of frequency 1
        min_amp1_reading=readings[0]
        amps_phase=min_amp1_reading[1]
        for i in range(len(amps_phase)-1):
            file.write(str(amps_phase[i])+',')
        file.write(str(amps_phase[len(amps_phase)-1])+'\n')
        ei_reading=min_amp1_reading[2]
        for i in range(len(ei_reading)-1):
            file2.write(ei_reading[i]+',')#write eis readings to file2
        file2.write(str(ei_reading[len(ei_reading)-1])+'\n')        
    file.close()
    file2.close()
    import os
    if os.path.isfile(outfile1):
        remove_data(outfile1,outfile3,'unselected_438_V1_28inputs_by_amp1_with_ids.csv')
        create_selected_unselected_readings_data(outfile1,'unselected_438_V1_28inputs_by_amp1_with_ids.csv',selected_unselected_data_csv)
    else:
        sys.exit(outfile1+' does not exist\n')
        
def select_eis_reading_by_sum_of_amplitude1_2_3():
    infile="U:\\EIS preterm prediction\\438_V1_4_eis_readings.csv"
    outfile1="U:\\EIS preterm prediction\\438_V1_28inputs_by_sum_of_amp1_2_3.csv"#amplitude and phase of selected eis readings
    outfile2="U:\\EIS preterm prediction\\438_V1_eis_readings_by_sum_of_amp1_2_3.csv"#the complex impedance of selected eis readings
    (df2,df)=amp_phase(infile,'none',True)
    (r,c)=df.shape
    preterm_hash={}#key=patient id, value=[(amplitude of frequency 1,list of amplitude and phase of reading1,EIS of reading1),(amplitude of frequency 1,list of amplitude and phase of reading2,EIS of reading2),...,]
    onterm_hash={}
    rows=df.index
    for i in range(r):
        patient_id=df.at[rows[i],'hospital_id']
        amp1=df.at[rows[i],'27_EIS_Amplitude1']
        amp2=df.at[rows[i],'27_EIS_Amplitude2']
        amp3=df.at[rows[i],'27_EIS_Amplitude3']   
        s=amp1+amp2+amp3
        amps_phase=list(df.iloc[i,:])
        ei_reading=list(df2.iloc[i,:])#complex impedance at 14 frequencies
        label=df.at[rows[i],'before37weeksCell']
        if int(label) == 1:#preterm
            if preterm_hash.get(patient_id)==None:
                preterm_hash[patient_id]=[(s,amps_phase,ei_reading)]
            else:
                l=preterm_hash[patient_id]
                l.append((s,amps_phase,ei_reading))
                preterm_hash[patient_id]=l
        else:        
            if onterm_hash.get(patient_id)==None:
                onterm_hash[patient_id]=[(s,amps_phase,ei_reading)]
            else:
                l=onterm_hash[patient_id]
                l.append((s,amps_phase,ei_reading))
                onterm_hash[patient_id]=l
    file=open(outfile1,'w+')#write amplitude and phase of the selected eis readings to file
    file2=open(outfile2,'w+') #write eis readings to file2
    cols=['hospital_id']
    for i in range(14):
        cols.append('27_EIS_Amplitude'+str(i+1))
    for i in range(14):
        cols.append('27_EIS_Phase'+str(i+1))
    cols.append('before37weeksCell')
    for i in range(len(cols)-1):
        file.write(cols[i]+',')#write the columns names
    file.write(cols[len(cols)-1]+'\n')
    cols2=['hospital_id']
    for i in range(14):
        cols2.append('EIS'+str(i+1))
    cols2.append('before37weeksCell')
    for i in range(len(cols2)-1):
        file2.write(cols2[i]+',')#write the columns names
    file2.write(cols2[len(cols2)-1]+'\n')
    for patient_id,readings in preterm_hash.items():
        readings.sort(key=operator.itemgetter(0),reverse=True)#sort in descending order of sum of amplitudes of frequencies 1, 2 and 3  
        max_reading=readings[0]
        amps_phase=max_reading[1]
        for i in range(len(amps_phase)-1):
            file.write(str(amps_phase[i])+',')#write amplitude and phase of the selected eis readings to file1
        file.write(str(amps_phase[len(amps_phase)-1])+'\n')
        ei_reading=max_reading[2]
        for i in range(len(ei_reading)-1):
            file2.write(ei_reading[i]+',')#write eis readings to file2
        file2.write(str(ei_reading[len(ei_reading)-1])+'\n')
    #print('onterm')
    for patient_id,readings in onterm_hash.items():
        readings.sort(key=operator.itemgetter(0),reverse=False)#sort in ascending order of sum of amplitudes of frequencies 1, 2 and 3
        min_reading=readings[0]
        amps_phase=min_reading[1]
        for i in range(len(amps_phase)-1):
            file.write(str(amps_phase[i])+',')
        file.write(str(amps_phase[len(amps_phase)-1])+'\n')
        ei_reading=min_reading[2]
        for i in range(len(ei_reading)-1):
            file2.write(ei_reading[i]+',')#write eis readings to file2
        file2.write(str(ei_reading[len(ei_reading)-1])+'\n')        
    file.close()
    file2.close() 

def select_eis_reading_by_sum_of_amplitude1_2_3_and_phase10_11_12_13_14():
    infile="U:\\EIS preterm prediction\\438_V1_4_eis_readings.csv"
    outfile1="U:\\EIS preterm prediction\\filtered_data_by_sum_of_amp1_2_3_and_phase10_11_12_13_14.csv"#amplitude and phase of selected eis readings
    outfile2="U:\\EIS preterm prediction\\filtered_data_eis_readings_by_sum_of_amp1_2_3_and_phase10_11_12_13_14.csv"#the complex impedance of selected eis readings
    (df2,df)=amp_phase(infile,'none',True)
    (r,c)=df.shape
    preterm_hash={}#key=patient id, value=[(amplitude of frequency 1,list of amplitude and phase of reading1,EIS of reading1),(amplitude of frequency 1,list of amplitude and phase of reading2,EIS of reading2),...,]
    onterm_hash={}
    rows=df.index
    for i in range(r):
        patient_id=df.at[rows[i],'hospital_id']
        amp1=df.at[rows[i],'27_EIS_Amplitude1']
        amp2=df.at[rows[i],'27_EIS_Amplitude2']
        amp3=df.at[rows[i],'27_EIS_Amplitude3']
        phase10=df.at[rows[i],'27_EIS_Phase10']
        phase11=df.at[rows[i],'27_EIS_Phase11']
        phase12=df.at[rows[i],'27_EIS_Phase12']
        phase13=df.at[rows[i],'27_EIS_Phase13']
        phase14=df.at[rows[i],'27_EIS_Phase14']        
        s=amp1+amp2+amp3+phase10+phase11+phase12+phase13+phase14
        amps_phase=list(df.iloc[i,:])
        ei_reading=list(df2.iloc[i,:])#complex impedance at 14 frequencies
        label=df.at[rows[i],'before37weeksCell']
        if int(label) == 1:#preterm
            if preterm_hash.get(patient_id)==None:
                preterm_hash[patient_id]=[(s,amps_phase,ei_reading)]
            else:
                l=preterm_hash[patient_id]
                l.append((s,amps_phase,ei_reading))
                preterm_hash[patient_id]=l
        else:        
            if onterm_hash.get(patient_id)==None:
                onterm_hash[patient_id]=[(s,amps_phase,ei_reading)]
            else:
                l=onterm_hash[patient_id]
                l.append((s,amps_phase,ei_reading))
                onterm_hash[patient_id]=l
    file=open(outfile1,'w+')#write amplitude and phase of the selected eis readings to file
    file2=open(outfile2,'w+') #write eis readings to file2
    cols=['hospital_id']
    for i in range(14):
        cols.append('27_EIS_Amplitude'+str(i+1))
    for i in range(14):
        cols.append('27_EIS_Phase'+str(i+1))
    cols.append('before37weeksCell')
    for i in range(len(cols)-1):
        file.write(cols[i]+',')#write the columns names
    file.write(cols[len(cols)-1]+'\n')
    cols2=['hospital_id']
    for i in range(14):
        cols2.append('EIS'+str(i+1))
    cols2.append('before37weeksCell')
    for i in range(len(cols2)-1):
        file2.write(cols2[i]+',')#write the columns names
    file2.write(cols2[len(cols2)-1]+'\n')
    for patient_id,readings in preterm_hash.items():
        readings.sort(key=operator.itemgetter(0),reverse=True)#sort in descending order of sum of amplitudes of frequencies 1, 2 and 3 and phases of frequencies 10, 11, 12, 13 and 14 
        max_reading=readings[0]
        amps_phase=max_reading[1]
        for i in range(len(amps_phase)-1):
            file.write(str(amps_phase[i])+',')#write amplitude and phase of the selected eis readings to file1
        file.write(str(amps_phase[len(amps_phase)-1])+'\n')
        ei_reading=max_reading[2]
        for i in range(len(ei_reading)-1):
            file2.write(ei_reading[i]+',')#write eis readings to file2
        file2.write(str(ei_reading[len(ei_reading)-1])+'\n')
    #print('onterm')
    for patient_id,readings in onterm_hash.items():
        readings.sort(key=operator.itemgetter(0),reverse=False)#sort in ascending order of sum of amplitudes of frequencies 1, 2 and 3 and phases of frequencies 10, 11, 12, 13 and 14
        min_reading=readings[0]
        amps_phase=min_reading[1]
        for i in range(len(amps_phase)-1):
            file.write(str(amps_phase[i])+',')
        file.write(str(amps_phase[len(amps_phase)-1])+'\n')
        ei_reading=min_reading[2]
        for i in range(len(ei_reading)-1):
            file2.write(ei_reading[i]+',')#write eis readings to file2
        file2.write(str(ei_reading[len(ei_reading)-1])+'\n')        
    file.close()
    file2.close() 

def get_real_imag(data,file2,datatype='csv'):
    #get the real and imaginary parts of eis
    #input: file, "U:\EIS preterm prediction\438_V1_4_eis_readings.csv"
    #output: file2
    #        dataframe of real parts and imaginary parts
    if datatype == 'csv':
        data=pd.read_csv(data)
    (r,c)=data.shape
    cols=['hospital_id']
    for i in range(14):
        cols.append('real_part_EIS'+str(i+1))
    for i in range(14):
        cols.append('imag_part_EIS'+str(i+1))
    cols.append('before37weeksCell')
    data2=pd.DataFrame(np.zeros((r,30)),columns=cols)#ids, 28 features and class variable
    data2.hospital_id=data2.hospital_id.astype(object)#change 'hospital_id' to object type for string
    rows=data.index #list of row labels
    rows_to_drop=set()#rows containing invalid eis readings i.e. 0
    for i in range(r):
        for j in range (14):
            eis=data.at[rows[i],'EIS'+str(j+1)]
            if eis != 'none':
                eis=complex_convert_i_to_j(eis)
                if eis!='zero':                
                    real_part=np.real(complex(eis))
                    imag_part=np.imag(complex(eis))
                    data2.at[rows[i],'real_part_EIS'+str(j+1)]=np.round(real_part,4)
                    data2.at[rows[i],'imag_part_EIS'+str(j+1)]=np.round(imag_part,4)
                else:#eis is 0 (invalid reading)
                    rows_to_drop.add(i)#indices of rows to remove
            else:
                print('none')
        data2.at[rows[i],'hospital_id']=data.at[rows[i],'hospital_id']
        if data.at[rows[i],'before37weeksCell']=='Yes' or data.at[rows[i],'before37weeksCell']=='yes':
            data2.at[rows[i],'before37weeksCell']=1
        elif data.at[rows[i],'before37weeksCell']=='No' or data.at[rows[i],'before37weeksCell']=='no':
            data2.at[rows[i],'before37weeksCell']=0
        else:
            data2.at[i,'before37weeksCell']=data.at[rows[i],'before37weeksCell']
    if len(rows_to_drop)>0:#delete any invalid eis readings ie. real numbers
        patients=[]
        cols=list(data.columns)
        for indx in list(rows_to_drop):
            patients.append(data.loc[indx,cols])
        data2=data2.drop(list(rows_to_drop))
        data=data.drop(list(rows_to_drop))
        print('The rows with 0 eis readings (0) are deleted: ',str(patients))
    data2.before37weeksCell=data2.before37weeksCell.astype(int)#convert any 1.0 and 0.0 to 1 and 0
    if file2!='none':
        data2.to_csv(file2,index=False)
    return (data,data2)
   
def amp_phase(file,file2,include_ids=True,datatype='eis'):
    #compute amplitude and phase of eis
    #input: file, e.g. '438_V1_eis_prob_C.csv' (aggregation of the 4 eis readings with their Prob_Ci)
    #       include_id, whether to include id in the output file (True or False)
    #output: file2, e.g. 125_V1_28inputs.csv' (amplitude and phase of the eis readings in the input file)
    if datatype=='eis':#column names: hospital_id,EI1,EI2,EI3,...,EI14,before37weeksCell
        data=pd.read_csv(file)
    elif datatype=='mis':#column names: Identifier,Real1,Real2,Real3,...,Real16,Im1,Im2,Im3,...,Im16,PTB
        data2=pd.read_csv(file)
        #create complex impedence features: EI1,EI2,EI3,...,EI16
        (r2,c2)=data2.shape
        data=pd.DataFrame(np.zeros((r2,18)),columns=['hospital_id','EI1','EI2','EI3','EI4','EI5','EI6','EI7','EI8','EI9','EI10','EI11','EI12','EI13','EI14','EI15','EI16','PTB'])
        data=data.astype(object)
        for i in range(r2):
            data.iat[i,0]=data2.iat[i,0]
            k=1
            for j in range(16):
                real_part=data2.iat[i,j+1]
                img_part=data2.iat[i,j+17]
                data.iat[i,k]=complex(real_part,img_part)
                #print(data.iat[i,k])
                k+=1
            data.iat[i,17]=data2.iat[i,c2-1]
        ##debug
        #data.to_csv('data_ei_features.csv',index=False)
    else:
        sys.exit('invalid option in function amp_phase: ',datatype)
    if include_ids==True:
        cols=['hospital_id']
    else:
        cols=[]
    if datatype=='eis':#EIS data has 14 frequencies
        for i in range(14):
            cols.append('27_EIS_Amplitude'+str(i+1))
        for i in range(14):
            cols.append('27_EIS_Phase'+str(i+1))
        cols.append('before37weeksCell')
    elif datatype=='mis':#MIS data has 16 frequencies
        for i in range(16):
            cols.append('Amplitude'+str(i+1))
        for i in range(16):
            cols.append('Phase'+str(i+1))
        cols.append('PTB')
    else:
        sys.exit('invalid option in function amp_phase: ',data)
    (r,c)=data.shape
    if datatype=='eis':
        if include_ids==True:
            data2=pd.DataFrame(np.zeros((r,30)),columns=cols)#ids, 28 features and class variable
        else:
            data2=pd.DataFrame(np.zeros((r,29)),columns=cols)#28 features and class variable
    elif datatype=='mis':
        if include_ids==True:
            data2=pd.DataFrame(np.zeros((r,34)),columns=cols)#ids, 32 features and class variable
        else:
            data2=pd.DataFrame(np.zeros((r,33)),columns=cols)#32 features and class variable                  
    if include_ids==True:
        data2.hospital_id=data2.hospital_id.astype(object)#change 'hospital_id' to object type for string
    rows=data.index #list of row labels
    rows_to_drop=set()#rows containing invalid 0 eis readings 
    if datatype=='eis':
        freq=14
    elif datatype=='mis':
        freq=16
    for i in range(r):
        for j in range (freq):
            ei=data.at[rows[i],'EI'+str(j+1)]
            if ei != 'none':
                if datatype=='eis':
                    ei=complex_convert_i_to_j(ei)
                if ei!='zero':                
                    amp=np.absolute(complex(ei))
                    phase=np.angle(complex(ei),deg=True)#phase in degrees
                    if datatype=='eis':
                        data2.at[i,'27_EIS_Amplitude'+str(j+1)]=np.round(amp,4)
                        data2.at[i,'27_EIS_Phase'+str(j+1)]=np.round(phase,4)
                    elif datatype=='mis':
                        data2.at[i,'Amplitude'+str(j+1)]=np.round(amp,4)
                        data2.at[i,'Phase'+str(j+1)]=np.round(phase,4)                        
                else:#eis is 0 (invalid reading)
                    rows_to_drop.add(i)#index of the row to delete
            else:
                print('none')
        if include_ids==True:
            data2.at[i,'hospital_id']=data.at[rows[i],'hospital_id']
        if datatype=='eis':
            if data.at[rows[i],'before37weeksCell']=='Yes' or data.at[rows[i],'before37weeksCell']=='yes':
                data2.at[i,'before37weeksCell']=1
            elif data.at[rows[i],'before37weeksCell']=='No' or data.at[rows[i],'before37weeksCell']=='no':
                data2.at[i,'before37weeksCell']=0
            else:
                data2.at[i,'before37weeksCell']=data.at[rows[i],'before37weeksCell']
        elif datatype=='mis':
             data2.at[i,'PTB']=data.at[rows[i],'PTB']
    if len(rows_to_drop)>0:#delete any invalid eis readings ie. real numbers
        patients=[]
        cols=list(data.columns)
        for indx in list(rows_to_drop):
            patients.append(data.loc[indx,cols])
        data2=data2.drop(list(rows_to_drop))
        data=data.drop(list(rows_to_drop))
        print('The rows with at least one 0 impedance reading are deleted: ')
        for patient in patients:
            print(patient)
    if datatype=='eis':
        data2.before37weeksCell=data2.before37weeksCell.astype(int)#convert any 1.0 and 0.0 to 1 and 0
    elif datatype=='mis':
        data2.PTB=data2.PTB.astype(int)#convert any 1.0 and 0.0 to 1 and 0       
    #print(data2.info())
    if file2!='none':
        data2.to_csv(file2,index=False)
    return (data,data2)

def convert_to_python_path(datapath):
    #convert a path e.g. "C:\Users\uos\data.csv" to path "C:/Users/uos/data.csv" 
    #from pathlib import Path 
    datapath=datapath.replace('\\','\\\\')
    #datapath=Path(datapath)
    print('file path: ',datapath)
    return str(datapath)

def convert_to_windows_path(datapath):
    #convert a path e.g. "C:\\Users\\uos\\data.csv" to path "C:\\Users\\uos\\data.csv" 
    from pathlib import Path 
    datapath=datapath.replace('\\\\','\\')
    datapath=Path(datapath)
    #print('file path: ',datapath)
    return str(datapath)

def get_removed_eis_readings():
    #df1: all single EIS readings
    #df2: filtered_data.csv (the selected readings)
    #return df3=df1-df2
    data1=r"U:\EIS preterm prediction\438_V1_4_eis_readings.csv"
    data1=convert_to_python_path(data1)
    df1_original=pd.read_csv(data1)
    reformatted_file_csv="U:\\EIS preterm prediction\\438_V1_4_eis_readings_reformatted.csv"
    df1=reformat_eis_readings(data1,0,reformatted_file_csv)
    data2=r"U:\EIS preterm prediction\filtered_data.csv"
    data2=convert_to_python_path(data2)
    reformatted_file_csv="U:\\EIS preterm prediction\\filtered_data_reformatted.csv"
    df2=reformat_eis_readings(data2,0,reformatted_file_csv)
    df3=dataframes_diff(df1,df2)
    rowsindx=df3.index
    df3=df1_original.loc[rowsindx,list(df1_original.columns)]#get the original readings
    data3=r"U:\EIS preterm prediction\removed_data.csv"
    data3=convert_to_python_path(data3)
    df3.to_csv(data3,index=False)
    print(df3)

def create_selected_unselected_readings_data(selected_data_csv,unselected_data_csv,selected_unselected_data_csv):
    #selected_data="filtered_data_28_inputs_with_ids.csv"
    #amp_phase("removed_data.csv","removed_data_28_inputs_with_ids.csv",True)
    #unselected_data="removed_data_28_inputs_with_ids.csv"
    #selected_data="centroid_filtered_data.csv"
    #unselected_data="centroid_removed_data.csv"
    #selected_data="centroid_filtered_data_with_ids.csv"
    #unselected_data="centroid_removed_data_with_ids.csv"
    df1=pd.read_csv(selected_data_csv)
    df2=pd.read_csv(unselected_data_csv)
    (r,c)=df1.shape
    cols=list(df1.columns)
    ones=pd.DataFrame(np.ones((r,1)),columns=["selected_reading"])
    (r2,c2)=df2.shape
    cols2=list(df2.columns)
    zeros=pd.DataFrame(np.zeros((r2,1)),columns=["selected_reading"])
    df1[cols[c-1]]=ones
    df2[cols2[c2-1]]=zeros
    df1=df1.rename(columns={cols[c-1]:"selected_reading"})
    df2=df2.rename(columns={cols2[c2-1]:"selected_reading"})
    df1.selected_reading=df1.selected_reading.astype(int)
    df2.selected_reading=df2.selected_reading.astype(int)
    print(df1.info())
    print(df2.info())
    df3=pd.concat([df1,df2])
    #df3.to_csv("selected_unselected_eis_readings_with_ids.csv",index=False)
    #df3.to_csv("centroid_selected_unselected_readings.csv",index=False)
    df3.to_csv(selected_unselected_data_csv,index=False)

def create_unselected_readings_preterm_onterm_data(preterm_onterm_data_csv,unselected_data_csv,preterm_onterm_unselected_data_csv):
    #merge filtered_data_28inputs.csv with unselected_data_28inputs.csv into a dataset with labels 0 (onterm),1 (preterm), 2 (unselected)
    #
    df1=pd.read_csv(preterm_onterm_data_csv)
    df2=pd.read_csv(unselected_data_csv)
    (r,c)=df1.shape
    cols=list(df1.columns)
    (r2,c2)=df2.shape
    label_two=pd.DataFrame(2*np.ones((r2,1)),columns=["label"])
    cols2=list(df2.columns)
    df2[cols2[c2-1]]=label_two
    df1=df1.rename(columns={cols[c-1]:"label"})
    df2=df2.rename(columns={cols2[c2-1]:"label"})
    df1.label=df1.label.astype(int)
    df2.label=df2.label.astype(int)
    print(df1.info())
    print(df2.info())
    df3=pd.concat([df1,df2])
    df3.to_csv(preterm_onterm_unselected_data_csv,index=False)
    
def select_eis_readings_by_peak_amplitude(option,all_readings_file,selected_readings_file):
    import operator
    #For each preterm patient, select the reading with the highest peak (maximum) amplitude across the 14 frequencies; and for each onterm patient, select the reading with the lowest peak amplitude across the 14 frequencies
    dataL = [line.strip() for line in open(all_readings_file)]
    preterm_hash={}#key=patient id, value=[(mean amplitude,reading1),(mean amplitude,reading2),...,]
    onterm_hash={}
    for line in dataL[1:len(dataL)-1]:#skip the 1st line of columns names
        reading=line.split(',')
        i=1
        ampL=[]
        while i <=14:#first 14 columns are amplitudes
            ampL.append(float(reading[i]))
            i+=1
        patient_id=reading[0]
        label=reading[len(reading)-1]
        if int(label) == 1:#preterm
            max_amp=np.max(ampL)
            if preterm_hash.get(patient_id)==None:
                preterm_hash[patient_id]=[(max_amp,reading)]
            else:
                readings=preterm_hash[patient_id]
                readings.append((max_amp,reading))
                preterm_hash[patient_id]=readings
        else:
            min_amp=np.min(ampL)        
            if onterm_hash.get(patient_id)==None:
                onterm_hash[patient_id]=[(min_amp,reading)]
            else:
                readings=onterm_hash[patient_id]
                readings.append((min_amp,reading))
                onterm_hash[patient_id]=readings
    #sort the readings of each patient in descending order of their peak amplitude and select the readings with max peak amplitude for preterm patients and the readings of minimum peak amplitude for onterm patients
    file=open(selected_readings_file,'w+')
    file.write(dataL[0]+'\n')#write the columns names
    #print('preterm')
    for patient_id,readings in preterm_hash.items():
        readings.sort(key=operator.itemgetter(0),reverse=True)#sort in descending order of peak amplitude   
        print(readings)
        if option == 'preterm_max_peak_amp_onterm_min_peak_amp':
            max_amp_reading=readings[0]
            reading=max_amp_reading[1]
        elif option == 'preterm_min_peak_amp_onterm_max_peak_amp':
            min_amp_reading=readings[len(readings)-1]
            reading=min_amp_reading[1]            
        for i in range(len(reading)-1):
            file.write(reading[i]+',')
        file.write(reading[len(reading)-1]+'\n')
    #print('onterm')
    for patient_id,readings in onterm_hash.items():
        readings.sort(key=operator.itemgetter(0),reverse=False)#sort in ascending order of peak amplitude
        print(readings)
        if option == 'preterm_max_peak_amp_onterm_min_peak_amp':
            min_amp_reading=readings[0]
            reading=min_amp_reading[1]
        elif option == 'preterm_min_peak_amp_onterm_max_peak_amp':
            max_amp_reading=readings[len(readings)-1]
            reading=max_amp_reading[1]        
        for i in range(len(reading)-1):
            file.write(reading[i]+',')
        file.write(reading[len(reading)-1]+'\n')
    file.close()

def balance_train_set(train_set,class0_size,class1_size,random_state2,method,csv_balanced_train_file=None):
    #inputs: the inputs (X) and outputs (y) of a training set,
    #       csv file name
    #       random_state (random seed)
    #       method of oversampling: random, smote or adasyn
    #       column labels of the training set
    #output: a balanced training set in the named csv file
    X=train_set.iloc[:,:len(list(train_set.columns))-1]
    y=train_set.iloc[:,len(list(train_set.columns))-1]    
    y=list(y)        
    column_labels=list(train_set.columns)
    #random_state2=random.randint(0,2**32-1)
    random_state2=int(random_state2)
    if method == 'random_sampling_with_replacement':
        print('random sampling with replacement')    
        from imblearn.over_sampling import RandomOverSampler      
        ros = RandomOverSampler(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        #print('selected instances: ',ros.sample_indices_)
    elif method == 'smote':
        print('SMOTE')
        import numpy as np
        X=X.values #convert X to numpy array
        y=np.array(y) #convert y to numpy array
        #print(y)
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2)
        X_resampled, y_resampled = sm.fit_resample(X, y)
    elif method=='borderline_smote':
        print('Borderline SMOTE')
        from imblearn.over_sampling import BorderlineSMOTE
        sm = BorderlineSMOTE(sampling_strategy={0:class0_size,1:class1_size},k_neighbors=5,random_state=random_state2)
        X_resampled, y_resampled = sm.fit_resample(X, y)
    elif method=='adasyn':
        print('ADASYN')
        from imblearn.over_sampling import ADASYN
        X_resampled, y_resampled = ADASYN(sampling_strategy='auto',random_state=random_state2).fit_resample(X, y)
        #print('balanced training set: ',len(y_resampled))
        print(sorted(Counter(y_resampled).items()))
    #save data to csv files
    df1=pd.DataFrame(X_resampled,columns=column_labels[:-1])#the inputs
    df2=pd.DataFrame(y_resampled,columns=[column_labels[-1]])#the targets
    balanced_train_set=df1.join(df2)
    (r,c)=balanced_train_set.shape
    print('balanced trainset size: ',r)
    cols=list(balanced_train_set.columns)
    labels=[0,1]
    class1=train_set[train_set[cols[-1]]==1]
    class0=train_set[train_set[cols[-1]]==0]
    (c1,_)=class1.shape
    (c0,_)=class0.shape
    for i in labels:
        classidf=balanced_train_set[balanced_train_set[cols[c-1]]==i]
        (ci,_)=classidf.shape
        print('class '+str(i)+' size: '+str(ci))
        unique=classidf.drop_duplicates(keep='first')
        print('unique instances of class: '+str(i)+': '+str(len(unique)))
        if i==0 and len(unique) < c0:
            print('total no. of unique instances of class0: '+str(c0))
            print(str(c0-len(unique))+' instances of class0 are not included in oversampled training set')
        elif i==1 and len(unique) < c1:
            print('total no. of unique instances of class1: '+str(c1))
            print(str(c1-len(unique))+' instances of class1 are not included in oversampled training set')
    if csv_balanced_train_file!=None:
        balanced_train_set.to_csv(csv_balanced_train_file,index=False)        
    return balanced_train_set

if __name__ == "__main__":    
    #df1=pd.read_csv('my_filtered_data_28inputs_and_have_ptb_history_treated_with_ids.csv')
    #df2=pd.read_csv('my_filtered_data_28inputs_and_have_ptb_history_treated_with_no_delivery_with_ids.csv')
    #df3=dataframes_diff2(df1,df2)
    #df3=df1.compare(df2,align_axis=0)
    #print(df3)
    #df3.to_csv('my_filtered_data_28inputs_and_have_ptb_history_treated_with_delivery_with_ids.csv', index=False)
    
    #get_removed_eis_readings()
    '''
    test="u:\\EIS preterm prediction\\testset0.25_poly_features.csv"
    reduced_test="u:\\EIS preterm prediction\\testset0.25_reduced.arff"
    reduced_test_discrete="u:\\EIS preterm prediction\\testset0.25_poly_features_reduced_discrete.arff"
    train="u:\\EIS preterm prediction\\trainset0.75_resample_poly_features.arff"
    discrete_train="u:\\EIS preterm prediction\\trainset0.75_resample_poly_features_discrete.arff"
    discrete_train_reduced="u:\\EIS preterm prediction\\trainset0.75_resample_poly_features_discrete_reduced.arff"
    A='0.95'
    C='35960' 
    c='35961'
    R='1-35960'
    chimerge_discretize(A,C,c,R,train,'temp_file.arff')
    remove_zero_variance_features('temp_file.arff',discrete_train)    
    info_gain_fs('10',discrete_train,discrete_train_reduced)
    convert_csv_to_arff(test,'testset0.25_poly_features.arff')
    reduce_data_using_another_data('testset0.25_poly_features.arff',discrete_train_reduced,reduced_test)
    discretize_using_cuts(reduced_test,discrete_train_reduced,reduced_test_discrete)
    '''
    #class_path="u:\\featureselection\\src;u:\\featureselection\\bin"
    #arff_train="trainset0.75_discrete0.95.arff"
    #arff_train="C:\\Users\\uos\\EIS preterm prediction\\resample\\poly features\\select features from training sets\\balanced_train_resample_reduced97.arff_balanced_reduced30_discrete.arff"
    #random_rsfs(class_path,arff_train,'1')
    #discrete_train="D:\\EIS preterm prediction\\resample\\poly features\\select features from training sets\\balanced_train_resample_reduced97_reduced_resample_discrete.arff"
    #discrete_train="D:\\EIS preterm prediction\\resample\\poly features\\select features from training sets\\balanced_train_resample_reduced97_discrete.arff"
    #cont_train="D:\\EIS preterm prediction\\resample\\poly features\\select features from training sets\\balanced_train_resample_reduced97.arff"
    #cont_test="D:\\EIS preterm prediction\\resample\\poly features\\select features from training sets\\test97_reduced.arff"
    #gls_rsfs(class_path,discrete_train,cont_train,cont_test,discrete_train+".gls_reducts")
    #convert_arff_to_csv('good_trainset0.arff','good_trainset0.csv')
    #convert_arff_to_csv('good_testset0.arff','good_testset0.csv')   
    #remove_data('filtered_data_28inputs_no_treatment_with_ids.csv','filtered_data_28inputs_with_ids.csv','filtered_data_28inputs_treatment_with_ids.csv')
    #remove_data('my_filtered_data_28inputs_with_ids.csv', '438_V1_4_eis_readings_28inputs_with_ids.csv','removed_from_my_filtered_data_28inputs_with_ids.csv')
    #besttrainset="C:\\Users\\uos\\EIS preterm prediction\\resample\\poly features\\select features from training sets\\balanced_train_resample97.csv"
    #idsL=get_patients_ids(besttrainset)#ids of patients in best training set
    #convert_arff_to_csv('good_trainset91.arff','good_trainset91.csv')
    #get_template_probs('438_V1_30inputs_demographics.csv',"U:\\EIS preterm prediction\\epithelium templates\\prob_o_prob_c.csv",1,'prob_of_ids.csv')
    #get_4_eis('438_V1_30inputs_demographics.csv','eis_cervical_length_ffn_visits_labour_onset_progesterone_cervical_cerclage.csv','Visit 1','438_V1_4_eis_readings.csv')
    #data=pd.read_csv("C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\testset_reduced.csv")
    #fs=find_features_of_missing_values('df',data)
    #print(fs)
    #fill_missing_values('median','df',data,fs,'testset_reduced4.csv')
    #fill_missing_values('csv','prob_of_ids.csv',['Prob_O.2','Prob_C.2'],'prob_of_ids2.csv')
    #(ids1reading,ids2readings,ids3readings,ids4readings)=count_EIS_readings_of_patients('438_V1_4_eis_readings.csv')
    #print('patients with 4 EIS readings: '+str(ids4readings))
    #get_temp_prob2(ids4readings,'prob_of_ids2.csv',1)
    #incorp_prob_C('prob_of_ids.csv','438_V1_4_eis_readings.csv','438_V1_eis_prob_C.csv')
    #amp_phase('438_V1_4_eis_readings.csv','438_V1_4_eis_readings_28inputs_with_ids.csv')
    #amp_phase('my_filtered_data.csv','my_filtered_data_28inputs_with_ids.csv')
    #amp_phase('my_filtered_data_2nd_max.csv','my_filtered_data_2nd_max_28inputs_with_ids.csv')
    #amp_phase('filtered_data_1st_readings.csv','filtered_data_1st_readings_28inputs_with_ids.csv')
    #amp_phase("D:\\EIS preterm prediction\\i4i MIS\\raw data\\divide by air reference\\mis_data_C1C2C3_divide_by_air_no_missing_labels.csv","D:\\EIS preterm prediction\\i4i MIS\\raw data\\divide by air reference\\amp_phase_of_mis_data_C1C2C3_divide_by_air_no_missing_labels.csv",datatype='mis')
    #joined_data=join_data("D:\\EIS preterm prediction\\EIS_Data\\EIS_Data\\438_V1_9inputs_with_ids.csv",'Demographics_only.csv','hospital_id','hospital_id')
    #joined_data=join_data("U:\\EIS preterm prediction\\my_filtered_data_28inputs_with_ids.csv","U:\\EIS preterm prediction\\438_V1_have_ptb_history.csv",'hospital_id','hospital_id')
    #joined_data.to_csv("U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_with_ids.csv",index=False)
    #joined_data.to_csv("438_V1_previous_history_and_demographics.csv",index=False)
    #joined_data=join_data("filtered_data_28inputs_no_treatment_with_ids.csv",'438_V1_previous_history_and_demographics.csv','hospital_id','hospital_id')
    #joined_data.to_csv("filtered_data_28inputs_no_treatment_previous_history_and_demographics.csv",index=False)    
    #joined_data=join_data("filtered_data_28inputs_with_ids.csv",'438_V1_demographics_obstetric_history_2_parous_features_with_ids.csv','hospital_id','hospital_id')
    #joined_data.to_csv('filtered_data_28inputs_438_V1_demographics_obstetric_history_2_parous_features_with_ids.csv',index=False)
    joined_data=join_data("438_V1_28inputs_selected_by_filter_with_ids.csv",'have_ptb_history_with_ids.csv','hospital_id','hospital_id')
    joined_data.to_csv('438_V1_28inputs_selected_by_filter_and_have_ptb_history_with_ids.csv',index=False)

    #ids2L=get_patients_ids('C:\\Users\\uos\\EIS preterm prediction\\trainset28.csv',"filtered_data_and_demographics_30_inputs.csv")
    #print(ids2L)
    #get_demo_details(ids2L,'filtered_data_and_demographics_30_inputs.csv','demo_data_of_trainset28.csv')
    #data=pd.read_csv("D:\\EIS preterm prediction\\working papers\\biomedical engineering\\IEEEtran\\example_dataset\\example_dataset_20_instances_3.csv")
    #poly_features(data,2,"D:\\EIS preterm prediction\\working papers\\biomedical engineering\\IEEEtran\\example_dataset\\poly_features_data_5features_20_instances_3.csv")
    #convert_csv_to_arff('merged_data_poly_features.csv','merged_data_poly_features.arff','last:0,1')
    #trainset_csv=r'C:\Users\uos\EIS preterm prediction\merged data\best training set\trainset_balanced_reduced34.csv'           
    #traincsv="C:\Users\uos\EIS preterm prediction\merged data\best training set\trainset_balanced_reduced34.csv"
    #path=path.replace('\\','/')
    #traincsv=Path(path)
    #print(traincsv)
    #convert_arff_to_csv(trainarff,traincsv)
    #testcsv="C:\Users\uos\EIS preterm prediction\merged data\best training set\testset34.csv"
    #create_unselected_readings_preterm_onterm_data('my_filtered_data_28inputs_with_ids.csv','removed_from_my_filtered_data_28inputs_with_ids.csv','my_filtered_data_unselected_eis_readings_with_ids.csv')    
    #trainset_balanced_arff="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow2\\trainset_balanced.arff"
    #trainset_balanced_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\trainset6_balanced.csv"
    #trainset_balanced_unique="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\trainset6_balanced_unique.csv"
    #convert_arff_to_csv(trainset_balanced_arff,trainset_balanced_csv,"C:\\Program Files\\Weka-3-7-10\\weka.jar",'2g')
    #convert_arff_to_csv("U:\\EIS preterm prediction\\trainset_subset.arff","U:\\EIS preterm prediction\\trainset_subset.csv","C:\\Program Files\\Weka-3-7-10\\weka.jar",'1g')
    #remove_duplicates(trainset_balanced_csv,trainset_balanced_unique)
    #df=pd.read_csv("C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\trainset84_balanced.csv")
    #dupfile_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\duplicates_in_trainset84_balanced.csv"
    #find_duplicates(df,dupfile_csv)
    #attrs=get_arff_attributes("C:\\Users\\uos\\EIS preterm prediction\\results2\\testset_ga_reduced.arff")
    #print(attrs)
    #select_random_eis_readings("U:\\EIS preterm prediction\\data filter\\438_V1_4_eis_readings_28inputs_normal_shapes.csv",'csv',"U:\\EIS preterm prediction\\data filter\\438_V1_28inputs_random.csv")
    #select_random_eis_readings("d:\\EIS preterm prediction\\438_V1_4_eis_readings_28inputs_with_ids.csv",'csv',"d:\\EIS preterm prediction\\438_V1_28inputs_unfiltered_random.csv")
    #select_random_eis_readings('438_V1_4_eis_readings_28inputs_no_treatment_with_ids.csv','csv','438_V1_28inputs_no_treatment_random.csv')    
    #java_memory='6g'
    #class_path='ga_rsfs.jar' 
    #arff_discrete_data=r"C:\Users\uos\EIS preterm prediction\results\workflow1\trainset_discrete.arff"
    #platform='windows'
    #populationSize='100'
    #generations='50'
    #crossover_prob='0.6'
    #mutation_prob='0.033'
    #fitness='find_reducts'
    #weka_path=r"C:\Program Files\Weka-3-7-10\weka.jar" 
    #weka_path=convert_to_python_path(weka_path)
    #equal_freq_discretize("D:\\EIS preterm prediction\\results\\workflow1\\filtered_data\\trainset.arff","D:\\EIS preterm prediction\\results\\workflow1\\filtered_data\\trainset_disc.arff",5,weka_path,java_memory)
    #results_path=r"C:\Users\uos\EIS preterm prediction\results\workflow1\\"
    #reducts_file=r"C:\Users\uos\EIS preterm prediction\results\workflow1\\reductsfile"
    #arff_discrete_data=convert_to_python_path(arff_discrete_data)
    #results_path=convert_to_python_path(results_path)
    #reducts_file=convert_to_python_path(reducts_file)
    #ga_rsfs(arff_discrete_data,reducts_file,populationSize,generations,crossover_prob,mutation_prob,fitness,class_path,weka_path,results_path,java_memory,platform)
    #all_readings_file="U:\\EIS preterm prediction\\438_V1_4_eis_readings_28inputs_with_ids.csv"
    #selected_readings_file=data="U:\\EIS preterm prediction\\preterm_max_peak_amp_onterm_min_peak_amp_with_ids.csv"
    #select_eis_readings_by_peak_amplitude('preterm_max_peak_amp_onterm_min_peak_amp',all_readings_file,selected_readings_file)
    #selected_readings_file=data="U:\\EIS preterm prediction\\preterm_min_peak_amp_onterm_max_peak_amp_with_ids.csv"
    #select_eis_readings_by_peak_amplitude('preterm_min_peak_amp_onterm_max_peak_amp',all_readings_file,selected_readings_file)
    #select_eis_reading_by_amplitude1()
    #select_eis_reading_by_sum_of_amplitude1_2_3()    
    #select_eis_reading_by_sum_of_amplitude1_2_3_and_phase10_11_12_13_14()        
    #data=r"U:\EIS preterm prediction\filtered_data.csv"
    #data=convert_to_python_path(data)
    #reformatted_file_csv="U:\\EIS preterm prediction\\filtered_data_reformatted.csv"
    #reformat_eis_readings(data,6,reformatted_file_csv)
    #get_real_imag("U:\\EIS preterm prediction\\filtered_data_reformatted.csv","U:\\EIS preterm prediction\\filtered_data_reformatted_real_imag.csv")
    #select_eis_reading_by_real_parts([0],"438_V1_4_eis_readings.csv","my_filtered_data.csv")
    #select_eis_reading_by_real_parts([0],"438_V1_4_eis_readings.csv","my_filtered_data_2nd_max.csv",option='2nd max')
    #select_eis_reading_by_imag_parts("d:\\EIS preterm prediction\\438_V1_4_eis_readings_with_ids.csv","d:\\EIS preterm prediction\\438_V1_eis_readings_largest_imagparts.csv")       
    #select_ith_reading("d:\\EIS preterm prediction\\438_V1_4_eis_readings.csv","d:\\EIS preterm prediction\\filtered_data_1st_readings.csv",k=0)
    #select_ith_reading("d:\\EIS preterm prediction\\438_V1_4_eis_readings.csv","d:\\EIS preterm prediction\\filtered_data_2nd_readings.csv",k=1)
    #split_train_test_sets2("U:\\EIS preterm prediction\\metabolite\\asymp_22wks_8inputs.csv",0.70,100,"C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\31oct19_100iter_asymp_metabolite\\") 
    #compare_ids_of_2_datasets("D:\\EIS preterm prediction\\i4i MIS\\raw data\\no compensation\\mis_data_c1c2c3_no_compensation_visit1_visit2_no_missing_labels.csv","D:\\EIS preterm prediction\\i4i MIS\\raw data\\Di\\ahr_v1_v2_symp_no_compensation.csv")
    #compare_2_datasets('no_ids','filtered_data_28inputs_no_treatment.csv','filtered_data_28inputs_no_treatment2.csv')
    #poly_features2("C:\\Users\\uos\\EIS preterm prediction\\testset82.csv",'original_features',"C:\\Users\\uos\\EIS preterm prediction\\rf82.model_inputs_output.csv","testset_reduced.csv")
    #poly_features2("C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\10oct19_36iter_filtered_data\\testset30.csv",'original_features',"C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\10oct19_36iter_filtered_data\\log_reg30.model_inputs_output.csv","testset30_reduced.csv")
    #poly_features2("C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\22oct19_20_iter_selected_unselected_eis_readings\\testset9.csv",'original_features',"C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\22oct19_20_iter_selected_unselected_eis_readings\\rf9.model_inputs_output.csv","testset9_reduced.csv")
    #poly_features2("C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\9oct19_59iter_filtered_data\\testset_good_readings2.csv",'original_features',"C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\9oct19_59iter_filtered_data\\rf2.model_inputs_output.csv","testset2_reduced.csv")
    #discretize_using_cuts("testset9_reduced.arff","C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\22oct19_20_iter_selected_unselected_eis_readings\\rf9.discrete_cuts","testset9_reduced_discrete.arff",".","6g")
    #select_eis_reading_by_amplitude1()
    #info_gain_fs('30',"C:\\Users\\uos\\EIS preterm prediction\\poly features\\438_V1_28inputs_poly_degree4.arff","C:\\Users\\uos\\EIS preterm prediction\\poly features\\438_V1_28inputs_poly_degree4_reduced.arff",weka_path,'13g') 
    #ordinal_encode('D:\\EIS preterm prediction\\results\\workflow1\\cst_asymp_8inputs_filtered_data_28inputs\\trainset_discrete_integers.arff',arff_discrete_train='D:\\EIS preterm prediction\\results\\workflow1\\cst_asymp_8inputs_filtered_data_28inputs\\trainset_discrete.arff')
    ###select a random reading for each patient id of a testset
    #all_readings=pd.read_csv("d:\\EIS preterm prediction\\438_V1_4_eis_readings_28inputs_with_ids.csv")
    #for i in range(100):
        #testset_ids='d:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets70_validsets15_testsets15\\testset1_ids_'+str(i)+'.csv'
        #testset_ids='d:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets66_validsets17_testsets17\\testset1_ids_'+str(i)+'.csv'      
        #testset_ids='d:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets66_validsets17_testsets17\\validset1_ids_'+str(i)+'.csv'      
        #testset_ids=pd.read_csv(testset_ids)
        #cols=list(all_readings.columns)
        #testset=all_readings.loc[lambda arg: np.isin(all_readings[cols[0]],testset_ids)]
        #select_random_eis_readings(testset,'df','d:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets66_validsets17_testsets17\\testset1_random_'+str(i)+'.csv')
        #select_random_eis_readings(testset,'df','d:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets66_validsets17_testsets17\\validset1_random_'+str(i)+'.csv')
    ###select a 2nd max real part reading for each patient id of a training set and a validset
    #second_max_readings='my_filtered_data_2nd_max_28inputs_with_ids.csv'
    #second_max_readings=pd.read_csv(second_max_readings)
    #cols=list(second_max_readings.columns)
    #for i in range(100):
    #    trainset_ids='d:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets66_validsets17_testsets17\\trainset1_ids_'+str(i)+'.csv'      
    #    trainset_ids=pd.read_csv(trainset_ids)
    #    trainset=second_max_readings.loc[lambda arg: np.isin(second_max_readings[cols[0]],trainset_ids)]
    #    trainset=trainset.iloc[:,1:len(cols)]#skip ids column
    #    validset_ids='d:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets66_validsets17_testsets17\\validset1_ids_'+str(i)+'.csv'      
    #    validset_ids=pd.read_csv(validset_ids)
    #    validset=second_max_readings.loc[lambda arg: np.isin(second_max_readings[cols[0]],validset_ids)]
    #    validset=validset.iloc[:,1:len(cols)]
    #    trainset.to_csv('d:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets66_validsets17_testsets17\\trainset1_2nd_max_'+str(i)+'.csv',index=False)
    #    validset.to_csv('d:\\EIS preterm prediction\\trainsets1trainsets2\\my_filtered_data\\trainsets66_validsets17_testsets17\\validset1_2nd_max_'+str(i)+'.csv',index=False)
    #wrapper_es_fs(arff_data='D:\\EIS preterm prediction\\filtered_data_28inputs.arff',cv_fold='5',optimalfeaturesubset_file='es_results_log_reg',weka_3_9_4_path='c:\\Program Files\\Weka-3-9-4\\weka.jar',results_path='D:\\EIS preterm prediction\\\\',java_memory='2g',pop_size='20',generations='20',crossover_prob='0.6',classifier='log reg',ridge='1.0E-8')
    #wrapper_es_fs(arff_data='D:\\EIS preterm prediction\\filtered_data_28inputs.arff',cv_fold='5',optimalfeaturesubset_file='es_results_rf',weka_3_9_4_path='c:\\Program Files\\Weka-3-9-4\\weka.jar',results_path='D:\\EIS preterm prediction\\\\',java_memory='2g',pop_size='20',generations='20',crossover_prob='0.6',classifier='random forest',trees='20',tree_depth='0',seed='1',no_of_cpu='4')
