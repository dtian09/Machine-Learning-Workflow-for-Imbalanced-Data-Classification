'''
Pipeline: oversampling, polynomial feature construction, information gain feature selection, GA feature selection, cross-valiation and training of Logistic Regression and Random Forest 
'''
import os
import pandas as pd
import numpy as np
import random
import preprocess as prep
import classifiers as cl
import postprocess as post
import sys
import operator
import re
import math 
pd.set_option('mode.chained_assignment', None)
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from joblib import dump#, load
from joblib import Parallel, delayed
#import workflow_nn
#import cb_oversample
#import matlab.engine
#import subprocess
#from sklearn.model_selection import GridSearchCV
import utilities
import ModelsPredict as mp
#from combine_weighted_sum import select_best_eis_then_predict_ptb
from sklearn.metrics import roc_auc_score
#import SelectData as SD
mini_noise=0
max_noise=0.001
#noise_percent=30
noise_percent=10
mini_features=10#smallest size of feature subsets of GA
max_features=20
#max_features=15#maximum size of feature subsets of GA
#mini_features=-1#feature subsets of random sizes of GA
#max_features=-1#feature subsets of random sizes of GA
results_path=''
trainsets1trainsets2Path=''
tree_depth='0' #max depth
ga_path=".\\ga_rsfs.jar"
#java GA_Reducts <discrete data file> > <windows or linux> <population size> <generations> <crossover prob> <mutation prob> <fitnessfunctiontype (find_reducts or find_high_info_gain_reducts)> <weka_path> <java_memory>
wrapper_es_fs=True
discretize_method="equal_freq"
bins=100 #equal frequency discretization for EIS data
#bins=80 #equal frequency discretization for EIS data
#bins=50 #equal frequency discretization for metabolite data
#discretize_method="equal_width"
#bins=10 #equal width discretization for metabolite data
populationSize=100
generations=50
#generations=1
crossover_prob=0.6
mutation_prob=0.033
fitness='find_reducts'
#fitness='find_high_info_gain_reducts'
#number_of_reducts=80
#number_of_reducts=40 #use the best number_of_reducts reducts from all the reducts found by the GA to train classifiers
number_of_reducts=10
#number_of_reducts=1
my_os="windows"
classifiersTypes='continuous classifiers only'
#classifiersTypes='discrete classifiers only'
#classifiersTypes='discrete and continuous classifiers'
#print(classifiersTypes)
cv=None #cross validation (default: no cross validation)
regularization_of_best_log_reg=[-999,-999,-999]# [j,reduct_size,regularization]
trees_of_best_rf=[-999,-999,-999] #[j,reduct_size,trees]
stop_cond=None #stop condition for tuning regularization of logistic regression
stop_cond2=None #stop condition for tuning trees of random forest   
dim=None #dimensionality of data
compare_with_set_of_all_features=True
core_features=None
predict_ptb_of_each_id_using_all_spectra_of_id=False
trainset_pids=None #ids of original training instances
testset_pids=None #ids of original testing instances
treesL=None
rf_or_lr_feature_select=None
   
def workflow1(results_path2,
              logfile,
              option='split_train_test_sets',#split_train_test_sets, train_test_sets or preprocess
              rf_or_lr_feature_select2=None,#reduce training set using random forest-selected features or logistic regression-selected features or none 
              logfile_option='w',
              trainsets1trainsets2Path2=None,
              dataset=None,
              original_train_set='train_set',
              add_noise_option='no_noise',
              noise_percent2=10,
              train_set_csv=None,
              test_set_csv=None,
              preprocessed_train_set_csv=None,
              cross_validation=None,
              iteration_number=0,
              trainset_fraction=0.66,
              testset_fraction=0.17,
              balanced_trainset_size=1984,
              oversampling_method2='oversample_class1_and_class0_separately_using_repeated_bootstraps',
              iterations=100,
              degree=-1,
              interaction_only2=True,#interaction features: included: x0, x1, x0*x1, etc. excluded: x0^2, x0^2*x1, etc. x0*x1 and x0^2*x1 contain the same information. When the dataset contains one of the features, the other feature is redundant and should be removed from the dataset.    
              k=-1,
              wrapper_es_fs2=False,
              discretize_method2="equal_width",
              bins2=100,
              fitness2='find_reducts',
              populationSize2='100',
              generations2='50',
              crossover_prob2='0.6',
              mutation_prob2='0.033',
              number_of_reducts2=-1,
              mini_features2=10,
              max_features2=15,
              classifiersTypes2='continuous classifiers',
              reg=['1E-8'],
              trees=['20'],
              tree_depth2='0',
              stop_cond_reg=5, #stop condition for tuning regularization of logistic regression
              stop_cond_trees=5,#stop condition for tuning trees of random forest
              seed0='none', #seed for data split (optional)
              seed='123456', #seed for oversampling and bootstrap of random forest
              no_of_cpu='4',
              weka_path='c:\\Program Files\\Weka-3-7-10\\weka.jar',
              java_memory='4g',
              compare_with_set_of_all_features2=True,
              core_features2=None, #indices of core features starting from 1 e.g. '4,8' (alpha and delta alpha features of cervical cancer dataset)
              predict_ptb_of_each_id_using_all_spectra_of_id2=False
              ):
    #input: dataset2, a csv dataset
    #       trainset_size #fraction of training set e.g. 0.66
    #       size, size of the balanced training set after oversampling
    #       m, no. of iterations
    #       degree, degree of polynomial features e.g. d=4
    #       k, no. of features to select using information gain feature selection
    #       reg, l2 regularization of logistic regression
    #       trees, no. of trees of random forest
    #       seed0, seed for data split
    #       seed, seed for oversampling and bootstrap of random forest
    #       results_path, path of results
    #       logfile
    #output: models, results files, training sets (.arff) and test sets (.arff) in results_path
    #print('training: training set -> oversampling')
    #print('                       -> construct polynomial features')
    #print('                       -> information gain feature selection')
    #print('                       -> genetic algorithm feature selection')
    #print('                       -> cross validation and train logistic regression and random forest\n')
    #print('testing: test set -> logistic regression and random forest')
    global number_of_reducts
    global mini_features
    global max_features
    global results_path
    global tree_depth
    global trainsets1trainsets2Path
    global fitness
    global experiment
    global classifiersTypes
    global discretize_method
    global bins
    global cv #k-fold cross validation (default: cv=None) e.g. cv=10 (10 fold cv)
    global noise_percent
    global wrapper_es_fs
    global populationSize
    global generations
    global crossover_prob
    global mutation_prob
    global stop_cond
    global stop_cond2
    global compare_with_set_of_all_features
    global core_features
    global predict_ptb_of_each_id_using_all_spectra_of_id
    global w1 #weight of ptb prediction of an id based on c1, c2 and c3 spectra of the id at visit 1
    global w2 #weight of ptb prediction of an id based on c1, c2 and c3 spectra of the id at visit 2
    global trainset_pids
    global testset_pids
    global treesL 
    global rf_or_lr_feature_select
    
    if rf_or_lr_feature_select2!=None:
        if rf_or_lr_feature_select2!='lr' and rf_or_lr_feature_select2!='rf':
            sys.exit('rf_or_lr_feature_select2='+str(rf_or_lr_feature_select2)+' is invalid. rf_or_lr_feature_select2=lr or rf.')       
        elif cross_validation==None:
            cross_validation=5
        if rf_or_lr_feature_select2=='rf':
            print('reduce training set using random forest-selected features')
        elif rf_or_lr_feature_select2=='lr':
            print('reduce training set using logistic regression-selected features')
        print('cross validation=',str(cross_validation))    
    if option=='train_test_sets' or option == 'split_train_test_sets':
        log_regL=[]#list of (iteration,train_test_auc,train_auc,train_tpr,train_tnr,test_auc,test_tpr,test_tnr) for logistic regression models where train_test_auc=(train_auc+test_auc)-absolute(train_auc-test_auc)
        rf_L=[]#list of (iteration,train_test_auc,train_auc,train_tpr,train_tnr,test_auc,test_tpr,test_tnr) for random forest models 
        log_reg_train_aucL=[]
        rf_train_aucL=[]
        if cv==None:#training and testing
            log_reg_test_aucL=[]     
            rf_test_aucL=[]        
    if type(reg)!=list:
        sys.exit('reg is not a list: '+str(reg))
    if type(trees)!=list:
        sys.exit('trees is not a list: '+str(trees))
    compare_with_set_of_all_features=compare_with_set_of_all_features2
    core_features=core_features2
    noise_percent=noise_percent2
    classifiersTypes=classifiersTypes2
    experiment=option
    wrapper_es_fs=wrapper_es_fs2
    populationSize=populationSize2
    generations=generations2
    crossover_prob=crossover_prob2
    mutation_prob=mutation_prob2
    number_of_reducts=number_of_reducts2
    fitness=fitness2
    discretize_method=discretize_method2
    bins=bins2
    stop_cond=stop_cond_reg
    stop_cond2=stop_cond_trees
    cv=cross_validation
    mini_features=int(mini_features2)
    max_features=int(max_features2)
    results_path=results_path2
    tree_depth=tree_depth2
    predict_ptb_of_each_id_using_all_spectra_of_id=predict_ptb_of_each_id_using_all_spectra_of_id2
    treesL=trees
    rf_or_lr_feature_select=rf_or_lr_feature_select2
    logfile=os.path.normpath(logfile)
    file=open(logfile,logfile_option)
    file.write('====Workflow1 is running====\n')
    file.write('####Parameters Setting####\n')
    if predict_ptb_of_each_id_using_all_spectra_of_id:
        print('predict ptb of each id using all the spectra of id')
        file.write('predict ptb of each id using all the spectra of id\n')
    if cv != None:#k-fold cross validation
        if isinstance(cv,int) and cv >= 2:
            file.write(str(cv)+'-fold cross validation\n')
        else:
            print('Number of folds of cross validation must be an integer >= 2')
            sys.exit(-1)
        file.write('dataset='+dataset+'\n')
    else:
        file.write('option='+option+'\n')
        if option=='split_train_test_sets':
            file.write('dataset='+dataset+'\n')
        elif option=='train_test_sets':
            file.write('train_set_csv: '+train_set_csv+'\n')
            file.write('test_set_csv: '+test_set_csv+'\n')
    file.write('results path='+results_path+'\n')
    file.write('classifiers types: '+classifiersTypes2+'\n')
    file.write('degree of polynomial features='+str(degree)+'\n')
    file.write('number of features selected using information gain='+str(k)+'\n')
    if wrapper_es_fs:
        file.write('Wrapper Evolutionary Search Feature Selection\n')
        file.write('population size='+str(populationSize)+'\n')
        file.write('generations='+str(generations)+'\n')
        file.write('crossover prob='+str(crossover_prob)+'\n')
        file.write('mutation prob='+str(mutation_prob)+'\n')
    else:
        if number_of_reducts > 0:
            file.write('Genetic Algorithm Feature Selection\n')
            file.write('fitness='+fitness+'\n')
            file.write('population size='+str(populationSize)+'\n')
            file.write('generations='+str(generations)+'\n')
            file.write('crossover prob='+str(crossover_prob)+'\n')
            file.write('mutation prob='+str(mutation_prob)+'\n')
            file.write('discretization method: '+discretize_method+'\n')
            if discretize_method=='equal_freq':
                file.write('bins: '+str(bins)+'\n')
                print('bins: ',str(bins))
            file.write('number of feature subsets to find='+str(number_of_reducts)+'\n')
            if core_features!=None:
                print('\t\t\tcore features: ',core_features)
                file.write('\t\t\tcore features: '+core_features+'\n')
            if mini_features == -1 and max_features == -1:
                file.write('select optimal feature subsets of random sizes from all the populations of genetic algorithm\n')
            file.write('minimum size of feature subsets selected by genetic algorithms='+str(mini_features)+'\n')
            file.write('maximum size of feature subsets selected by genetic algorithms='+str(max_features)+'\n')
            if compare_with_set_of_all_features2:
                file.write('compare optimal feature subsets with the set of all features\n')
        else:#all features are used to train models
            file.write('Use all the features of the training set to train classifiers.\n')
            if classifiersTypes2=='discrete classifiers only' or classifiersTypes2=='discrete and continuous classifiers':
                file.write('discretization method: '+discretize_method+'\n')
                #print('discretization method: '+discretize_method)
                if discretize_method=='equal_freq':
                    file.write('bins: '+str(bins)+'\n')
                    print('bins: ',str(bins))
    file.write('size of balanced training set after resampling='+str(balanced_trainset_size)+'\n')
    file.write('ridge coefficient of l2 regularization of logistic regression='+str(reg)+'\n')
    file.write('number of trees of random forest='+str(trees)+'\n')
    file.write('random seed of oversampling and random forest='+str(seed)+'\n')
    file.close()    
    #read in datasets and convert targets from Yes(No) to 1(0)
    if cv != None or option == 'preprocess':#k-fold cross validation
        L=[iteration_number]           
        if os.path.isfile(dataset):
            train_set=pd.read_csv(dataset)
            cols=list(train_set.columns)
            if predict_ptb_of_each_id_using_all_spectra_of_id or cols[0] == 'Identifier' or cols[0] == 'hospital_id' or cols[0] == 'id' or cols[0] == 'Id' or cols[0] == 'ID':
                (_,c)=train_set.shape
                trainset_pids=train_set.iloc[:,0]
                train_set=train_set.iloc[:,1:c] #removed ids column
        else:
            sys.exit('training set does not exist: '+dataset)            
    elif option=='train_test_sets':#select optimal models based on testing performance
        L = [iteration_number]
        if os.path.isfile(train_set_csv):
            train_set=pd.read_csv(train_set_csv)
            cols=list(train_set.columns)
            if cols[0] == 'Identifier' or cols[0] == 'hospital_id' or cols[0] == 'id' or cols[0] == 'Id' or cols[0] == 'ID':
                (_,c)=train_set.shape
                train_set=train_set.iloc[:,1:c] #removed ids column from training set               
            if os.path.isfile(test_set_csv):
                test_set=pd.read_csv(test_set_csv)
                if cols[0] == 'Identifier' or cols[0] == 'hospital_id' or cols[0] == 'id' or cols[0] == 'Id' or cols[0] == 'ID':
                    (_,c)=test_set.shape
                    test_set=test_set.iloc[:,1:c] #removed ids column from testset
            else:
                sys.exit('test set does not exist: '+test_set_csv)
        else:
            sys.exit('train set does not exist: '+train_set_csv)     
    elif option == 'split_train_test_sets':#split data into training and testing sets, select optimal models based on testing performance
        if os.path.isfile(dataset):
            data=pd.read_csv(dataset)
            cols=list(data.columns)
            if cols[0] == 'Identifier' or cols[0] == 'hospital_id' or cols[0] == 'id' or cols[0] == 'Id' or cols[0] == 'ID':
                (_,c)=data.shape
                data=data.iloc[:,1:c] #removed ids column from whole dataset
        else:
            sys.exit('dataset does not exist: '+dataset)
        (_,c)=data.shape
        cols=list(data.columns)
        test_size=1-float(trainset_fraction)
        #(dist1,dist2)=prep.distance_of_2classes('df',data,0,22)#distance between mean of amplitude at frequency 1 of preterm and that of onterm, distance between mean of phase of preterm at frequency 9 and that of onterm.
        L = [i for i in range(int(iterations))]
        file=open(logfile,'a')
        file.write('training set percentage: '+str(trainset_fraction*100)+'\n')
        file.write('test set percentage: '+str(test_size*100)+'\n')
        file.close()    
    else:
        print('invalid option: ',option)
        sys.exit(-1)
    for i in L:
        iteration=i
        ###delete any old discrete cuts files of random forest and logistic regression of previous experiments
        old_cuts_file_rf=results_path+'rf'+str(i)+'.discrete_cuts'
        old_cuts_file_log_reg=results_path+'log_reg'+str(i)+'.discrete_cuts'
        utilities.delete_files([old_cuts_file_rf,old_cuts_file_log_reg])
        files_to_delete=[]
        ###finished deleting old discrete cuts files
        file=open(logfile,'a')
        if option=='preprocess':
            file.write("\titeration: "+str(iteration_number)+'\n')
        else:    
            file.write("\titeration: "+str(i)+'\n')
        print("\titeration: "+str(i))
        file.write('seed (oversampling and random forest): '+str(seed)+'\n')
        #print('seed (oversampling and random forest): '+str(seed))
        step=1
        if cv != None:#k-fold cross validation
           print('\t###Training Pipeline###')
            #print('\t###'+str(cv)+'-fold cross validation and training classifiers###')
        elif option == 'split_train_test_sets':            
            #seed0=random.randint(0,5**9) #ith seed for data split 
            seed0=random.randint(0,999999) #ith seed for data split
            file.write('seed0: '+str(seed0)+'\n')
            #print('seed0 (data split): '+str(seed0))
            (train_set,test_set)=prep.split_train_test_sets(data,test_size,seed0,cols[c-1])#split whole dataset consisting of 28 features (14 amplitudes and 14 phases) into training and test sets
            #(train_set,test_set)=prep.data_split_using_dists(dist1,dist2,data,test_size,seed2,cols[c-1])
            if predict_ptb_of_each_id_using_all_spectra_of_id:
                (_,c)=train_set.shape
                trainset_pids=train_set.iloc[:,0]
                train_set=train_set.iloc[:,1:c-1] #removed ids column
                testset_pids=test_set.iloc[:,0]
                test_set=test_set.iloc[:,1:c-1] #removed ids column                
            file.writelines(["\t\t\titeration: "+str(i)+"\n","\t\t\t1.Split dataset into training set and test set\n"])
            file.close()
            print('\t###Training Pipeline###')
            print('\t\t\t1.Split dataset into training set and test set')            
        elif option == 'preprocess':
            test_set=train_set
            print('\t###Preprocess Training Data###')
        else:
            sys.exit('invalid option: '+str(option))
        (r,c)=train_set.shape
        train_set=prep.convert_targets(train_set,c-1)
        train_set=prep.fill_missing_values('median','df',train_set)
        train_set.to_csv(results_path+'trainset'+str(i)+'_no_missing_vals.csv',index=False)
        files_to_delete.append(results_path+'trainset'+str(i)+'_no_missing_vals.csv')
        if add_noise_option=='add_noise':
            #traindf=utilities.add_noise_to_amp_phase(traindf)
            #print('added noise to EIS data')
            #traindf2=utilities.add_noise_to_metabolite(traindf,percent=noise_percent)
            #traindf2=utilities.add_noise_based_on_mean_of_each_class(traindf,percent=noise_percent)
            train_set2=utilities.add_noise(train_set,mini=mini_noise,maxi=max_noise)
            train_set=pd.concat([train_set,train_set2])#merge the noisy training data with the original training set
            train_set.to_csv(results_path+'trainset.csv',index=False)
            train_set=pd.read_csv(results_path+'trainset.csv')#read the training set into a dataframe so that each row has a unique row label
        elif add_noise_option != 'no_noise':
            print('invalid noise option: ',add_noise_option)
            return -1        
        if cv==None and option!= 'preprocess':   
           test_set=prep.convert_targets(test_set,c-1)
           test_set=prep.fill_missing_values('median','df',test_set)                
           test_set.to_csv(results_path+'testset'+str(i)+'_no_missing_vals.csv',index=False)
           files_to_delete.append(results_path+'testset'+str(i)+'_no_missing_vals.csv')
        ga_input_data=''
        if balanced_trainset_size == 'oversample class1 to size of class0':
            balanced_trainset_size = 999999
        elif balanced_trainset_size == 'undersample class0 to size of class1 with replacement':
            balanced_trainset_size = 111111
        elif balanced_trainset_size == 'undersample class0 to size of class1 without replacement':
            balanced_trainset_size = 222222
        if int(degree) < 2 and int(k) > c-1:#select k features from all the original features
            sys.exit('no. of features to select using information gain > total no. of features of the dataset')
        elif int(balanced_trainset_size) > 0 and int(degree) >= 2 and int(k) >= 1:
            ###training set -> oversampling or undersampling -> construct polynomial features -> information gain feature selection -> (reduced balanced training set in arff, reduced original train set in arff, reduced testset in arff)
            #seeds=[seed,seed2]
            if cv!=None or option=='preprocess':#cross validation
                step=preprocess_trainset(step,3,results_path+'trainset'+str(i)+'_no_missing_vals.csv','none',train_set,'none','trainset_balanced_reduced.arff','original_trainset_reduced.arff','none',results_path,logfile,balanced_trainset_size,degree,str(k),seed,weka_path,java_memory,oversampling_method=oversampling_method2,interaction_only=interaction_only2)
            else:                
                step=preprocess_trainset(step,3,results_path+'trainset'+str(i)+'_no_missing_vals.csv',results_path+'testset'+str(i)+'_no_missing_vals.csv',train_set,'none','trainset_balanced_reduced.arff','original_trainset_reduced.arff','testset_reduced.arff',results_path,logfile,balanced_trainset_size,degree,str(k),seed,weka_path,java_memory,oversampling_method=oversampling_method2,interaction_only=interaction_only2)
            ga_input_data='trainset_balanced_reduced.arff'
        elif int(balanced_trainset_size) > 0 and int(degree) < 2 and int(k) >= 1:#do not generate polynomial features, but select features using information gain
            ###training set -> oversampling or undersampling -> information gain feature selection -> (reduced balanced training set in arff, reduced original train set in arff, reduced testset in arff)
            if cv!=None or option=='preprocess':#cross validation
                step=oversample_trainset2(step,3,results_path+'trainset'+str(i)+'_no_missing_vals.csv','none',train_set,'none','trainset_balanced_reduced.arff','original_trainset_reduced.arff','none',results_path,logfile,balanced_trainset_size,k,seed,weka_path,java_memory,oversampling_method=oversampling_method2)
            else:    
                step=oversample_trainset2(step,3,results_path+'trainset'+str(i)+'_no_missing_vals.csv',results_path+'testset'+str(i)+'_no_missing_vals.csv',train_set,'none','trainset_balanced_reduced.arff','original_trainset_reduced.arff','testset_reduced.arff',results_path,logfile,balanced_trainset_size,k,seed,weka_path,java_memory,oversampling_method=oversampling_method2)
            ga_input_data='trainset_balanced_reduced.arff'
        elif int(balanced_trainset_size) > 0 and int(degree) >= 2 and int(k) < 1:
            ###training set -> oversampling or undersampling -> construct polynomial features -> (balanced training set in arff, original training set in arff, testset in arff)
            if cv!=None or option=='preprocess':#cross validation
                step=preprocess_trainset(step,3,results_path+'trainset'+str(i)+'_no_missing_vals.csv','none',train_set,'none','trainset_balanced.arff','original_trainset.arff','none',results_path,logfile,balanced_trainset_size,degree,str(k),seed,weka_path,java_memory,oversampling_method=oversampling_method2,interaction_only=interaction_only2)
            else:                
                step=preprocess_trainset(step,3,results_path+'trainset'+str(i)+'_no_missing_vals.csv',results_path+'testset'+str(i)+'_no_missing_vals.csv',train_set,'none','trainset_balanced.arff','original_trainset.arff','testset.arff',results_path,logfile,balanced_trainset_size,degree,str(k),seed,weka_path,java_memory,oversampling_method=oversampling_method2,interaction_only=interaction_only2)
            ga_input_data='trainset_balanced.arff'
        elif int(balanced_trainset_size) > 0 and int(degree) < 2 and int(k) < 1:#ovesampling or undersampling without generating polynomial features and selecting features using information gain
            ###training set -> oversampling or undersampling -> (balanced training set in arff, original training set in arff, testset in arff)
            if cv!=None or option=='preprocess':#cross validation
                step=oversample_trainset(step,3,results_path+'trainset'+str(i)+'_no_missing_vals.csv','none',train_set,'none','trainset_balanced.arff','original_trainset.arff','none',results_path,logfile,balanced_trainset_size,seed,weka_path,java_memory,oversampling_method=oversampling_method2)
            else:
                step=oversample_trainset(step,3,results_path+'trainset'+str(i)+'_no_missing_vals.csv',results_path+'testset'+str(i)+'_no_missing_vals.csv',train_set,'none','trainset_balanced.arff','original_trainset.arff','testset.arff',results_path,logfile,balanced_trainset_size,seed,weka_path,java_memory,oversampling_method=oversampling_method2)
            ga_input_data='trainset_balanced.arff'
            #seeds=[seed,seed2]
            #step=oversample_trainset(step,'no_noise','cb_oversampling',results_path+'trainset'+str(i)+'.csv',results_path+'testset'+str(i)+'.csv',train_set,'trainset_balanced.csv','trainset_balanced.arff','original_trainset.arff','testset.arff',results_path,logfile,balanced_trainset_size,seeds,weka_path,java_memory)
        elif int(balanced_trainset_size) < 0 and int(degree) >= 2 and int(k) >= 1:
            ###training set -> construct polynomial features -> information gain feature selection -> (reduced training set in arff, reduced original training set in arff, reduced testset in arff)
            if cv!=None or option == 'preprocess':#cross validation
                step=preprocess_trainset2(i,step,results_path+'trainset'+str(i)+'_no_missing_vals.csv','none',train_set,'none','trainset_reduced.arff','original_trainset_reduced.arff','none',results_path,logfile,degree,k,weka_path,java_memory,interaction_only=interaction_only2)           
            else:
                step=preprocess_trainset2(i,step,results_path+'trainset'+str(i)+'_no_missing_vals.csv',results_path+'testset'+str(i)+'_no_missing_vals.csv',train_set,'none','trainset_reduced.arff','original_trainset_reduced.arff','testset_reduced.arff',results_path,logfile,degree,k,weka_path,java_memory,interaction_only=interaction_only2)           
            ga_input_data='trainset_reduced.arff'
        elif int(balanced_trainset_size) < 0 and int(degree) < 2 and int(k) >= 1:
            ###training set -> information gain feature selection -> (reduced training set in arff, reduced original training set in arff, reduced testset in arff)
            if cv!=None or option=='preprocess':#cross validation
                step=preprocess_trainset2(i,step,results_path+'trainset'+str(i)+'_no_missing_vals.csv','none',train_set,'none','trainset_reduced.arff','original_trainset_reduced.arff','none',results_path,logfile,degree,k,weka_path,java_memory)           
            else:                
                step=preprocess_trainset2(i,step,results_path+'trainset'+str(i)+'_no_missing_vals.csv',results_path+'testset'+str(i)+'_no_missing_vals.csv',train_set,'none','trainset_reduced.arff','original_trainset_reduced.arff','testset_reduced.arff',results_path,logfile,degree,k,weka_path,java_memory)           
            ga_input_data='trainset_reduced.arff'
        elif int(balanced_trainset_size) < 0 and int(degree) >= 2 and int(k) < 1:
            ###training set -> construct polynomial features -> (training set in arff, original training set in arff, testset in arff)
            if cv!=None or option == 'preprocess':#cross validation
                step=preprocess_trainset2(i,step,results_path+'trainset'+str(i)+'_no_missing_vals.csv','none',train_set,'none','trainset.arff','none','none',results_path,logfile,degree,k,weka_path,java_memory,interaction_only=interaction_only2)           
            else:                
                step=preprocess_trainset2(i,step,results_path+'trainset'+str(i)+'_no_missing_vals.csv',results_path+'testset'+str(i)+'_no_missing_vals.csv',train_set,'none','trainset.arff','none','testset.arff',results_path,logfile,degree,k,weka_path,java_memory,interaction_only=interaction_only2)           
            ga_input_data='trainset.arff'
        elif int(balanced_trainset_size) < 0 and int(degree) < 2 and int(k) < 1:#no preprocessing before running ga_rsfs
            ###training set -> data format conversion -> (training set in arff, testset in arff)
            train_set=train_set.astype(object)
            utilities.dataframe_to_arff(train_set,results_path+'trainset.arff')
            utilities.mycopyfile(results_path+'trainset.arff',results_path+'original_trainset.arff')
            if cv==None:#training and testing experiment
                utilities.dataframe_to_arff(test_set,results_path+'testset.arff')
            ga_input_data='trainset.arff'
        else:
            print('invalid options for balanced_trainset_size, degree or k. Valid options: balanced_trainset_size > 1 or balanced_trainset_size < 0, degree > 1 or degree < 0, k > 1 or k < 0')                              
        if ga_input_data == 'trainset_balanced_reduced.arff':
            #balanced training set with selected features -> evolutinary search feature selection or GA_RSFS -> train and test models
            original_training_set_arff='original_trainset_reduced.arff'
            testset_arff='testset_reduced.arff'
        elif ga_input_data == 'trainset_balanced.arff':
            #balanced training set with all features -> evolutionary search feature selection or GA_RSFS -> train and test models
            original_training_set_arff='original_trainset.arff'   
            testset_arff='testset.arff'
        elif ga_input_data == 'trainset_reduced.arff':
            #unbalanced training set with selected features-> evolutionary search feature selection or GA_RSFS -> train and test models
            original_training_set_arff='original_trainset_reduced.arff'
            testset_arff='testset_reduced.arff'
        elif ga_input_data == 'trainset.arff':
            #unbalanced training set with all features-> evolutionary search feature selection or GA_RSFS -> train and test models
            original_training_set_arff='original_trainset.arff'
            testset_arff='testset.arff'
        else:
            sys.exit('invalid ga_input_data: '+ga_input_data)
        files_to_delete.append(results_path+ga_input_data)
        files_to_delete.append(results_path+original_training_set_arff)
        if test_set_csv!=None:#training and testing experiment
            files_to_delete.append(results_path+testset_arff)            
        if option=='preprocess' and rf_or_lr_feature_select==None and number_of_reducts <= 0:#oversampled training set or information gain reduced training set
            if preprocessed_train_set_csv!=None:#save the preprocessed training set to a user-specified csv file
                prep.convert_arff_to_csv(results_path+ga_input_data,preprocessed_train_set_csv,weka_path,java_memory)
                utilities.delete_files(files_to_delete)
                print('finished preprocessing data')
                return True            
            else:#save the preprocessed training set to csv file with a default name 
                prep.convert_arff_to_csv(results_path+ga_input_data,train_set_csv+'_preprocessed_balanced_trainset_size='+str(balanced_trainset_size)+'_degree='+str(degree)+'_info_gain_selected_features='+str(k)+'_'+str(iteration_number)+'.csv',weka_path,java_memory)
                prep.convert_arff_to_csv(results_path+original_training_set_arff,train_set_csv+'_original_training_set_preprocessed_balanced_trainset_size='+str(balanced_trainset_size)+'_degree='+str(degree)+'_info_gain_selected_features='+str(k)+'_'+str(iteration_number)+'.csv',weka_path,java_memory)
                if test_set_csv!=None:#training and testing experiment
                    prep.convert_arff_to_csv(results_path+testset_arff,train_set_csv+'_testset_preprocessed_balanced_trainset_size='+str(balanced_trainset_size)+'_degree='+str(degree)+'_info_gain_selected_features='+str(k)+'_'+str(iteration_number)+'.csv',weka_path,java_memory)       
                utilities.delete_files(files_to_delete)
                print('finished preprocessing data')
                return True
        file=open(logfile,'a')
        if wrapper_es_fs:
            print('\t\t\t'+str(step)+'. Evolutionary Search Feature Selection\n')
            file.write('\t\t\t'+str(step)+'. Evolutionary Search Feature Selection\n')
            step+=1
        elif wrapper_es_fs==False and number_of_reducts>0:
            print('\t\t\t'+str(step)+'. Genetic Algorithm Feature Selection')
            file.write('\t\t\t'+str(step)+'. Genetic Algorithm Feature Selection\n')
            step+=1
            if core_features!=None:
                print('\t\t\tcore features: ',core_features)
                file.write('\t\t\tcore features: '+core_features+'\n')
        if cv!=None:
            print('\t\t\t'+str(step)+'. Cross-validation and training classifiers\n')
            file.write('\t\t\t'+str(step)+'. Cross-validating and training classifiers\n')
        file.close()
        if wrapper_es_fs:
                   print('Wrapper Evolutionary Search Feature Selection')
                   if cv==None:
                       sys.exit('Cross validation (cv) can not be set to None when using wrapper evolutionary search feature selection')
                   prep.wrapper_es_fs(arff_data=results_path+ga_input_data,cv_fold=str(cv),optimalfeaturesubset_file=results_path+'featuresubset_log_reg',weka_3_9_4_path=weka_path,java_memory=java_memory,pop_size=str(populationSize),generations=str(generations),crossover_prob=str(crossover_prob),mutation_prob=str(mutation_prob),classifier='log reg',ridge=str(reg))
                   prep.wrapper_es_fs(arff_data=results_path+ga_input_data,cv_fold=str(cv),optimalfeaturesubset_file=results_path+'featuresubset_rf',weka_3_9_4_path=weka_path,java_memory=java_memory,pop_size=str(populationSize),generations=str(generations),crossover_prob=str(crossover_prob),mutation_prob=str(mutation_prob),classifier='random forest',trees=str(trees),tree_depth='0',seed=str(seed),no_of_cpu=str(no_of_cpu))
                   #get the optimal feature subset of logistic regression and that of random forest and write them into a reductsfile
                   if os.path.isfile(results_path+'featuresubset_log_reg'):
                       file=open(results_path+'featuresubset_log_reg','r')
                       featuresubset=file.readline()
                       featuresubset=featuresubset.strip()
                       file.close()
                   else:
                       sys.exit('file: '+results_path+'featuresubset_log_reg'+' does not exist')
                   if os.path.isfile(results_path+'featuresubset_rf'):
                       file=open(results_path+'featuresubset_rf','r')
                       featuresubset2=file.readline()
                       featuresubset2=featuresubset2.strip()
                       file.close()
                   else:
                       sys.exit('file: '+results_path+'featuresubset_rf'+' does not exist')
                   reductsfile=results_path+'featuresubsets_log_reg_and_random_forest'+str(i)+'.txt'
                   file=open(reductsfile,'w')
                   file.write(featuresubset+'\n')
                   file.write(featuresubset2)
                   file.close()
                   (models,models_inputs_output,performance,models_discrete_cuts)=select_optimal_feature_subsets_and_parameters_then_train_classifiers(results_path+ga_input_data,results_path+original_training_set_arff,'none',reg,trees,seed,results_path,weka_path,java_memory,reductsfile=reductsfile,cross_validation=cv)                   
        else:#run rough set-based wrapper feature selection method (rough set-based GA feature selection + k-fold CV), then train a classifier
             #training set -> GA RSFS -> K best reducts -> determine an optimal reduct using CV of logistic regression -> determine an optimal regularization of logistic regression using CV -> train logistic regression using optimal reduct and optimal regularization -> logistic regression
             #                                          -> determine an optimal reduct using CV of random forest -> determine an optimal trees of random forest using CV -> train random forest using optimal reduct and optimal trees -> random forest                    
                if int(number_of_reducts) > 0: #or classifiersTypes == 'discrete and continuous classifiers' or classifiersTypes == 'discrete classifiers only':#training only discrete classifiers on some or all features; training only continuous classifiers on some features; training both continuous and discrete classifiers on some or all features 
                    (reductsfile,trainset_discrete_arff)=discretize_ga_rsfs(i,results_path+ga_input_data,discretize_method,bins,populationSize,generations,crossover_prob,mutation_prob,fitness,number_of_reducts,ga_path,results_path,weka_path,java_memory)
                    #each reduct does not include the class variable index
                    if option=='preprocess' and rf_or_lr_feature_select==None:
                        prep.convert_arff_to_csv(results_path+ga_input_data,preprocessed_train_set_csv,weka_path,java_memory)
                        files_to_delete.append(trainset_discrete_arff)
                        utilities.delete_files(files_to_delete)
                        print('finished using GA feature selection to select best '+str(number_of_reducts)+' feature subsets')
                        print('The selected feature subsets are saved to ',reductsfile)
                        print('The input data of GA feature selection is saved to ',preprocessed_train_set_csv)
                        return True
                    elif rf_or_lr_feature_select!=None:
                        if cv!=None:
                            reduced_trainset_arff=select_optimal_feature_subsets_and_parameters_then_train_classifiers(results_path+ga_input_data,results_path+original_training_set_arff,'none',reg,trees,seed,results_path,weka_path,java_memory,weka_train_discrete_file=trainset_discrete_arff,reductsfile=reductsfile,cross_validation=cv)
                            #save the preprocessed training set to a user-specified csv file
                            prep.convert_arff_to_csv(reduced_trainset_arff,preprocessed_train_set_csv,weka_path,java_memory)
                            files_to_delete.append(trainset_discrete_arff)
                            utilities.delete_files(files_to_delete)
                            print('finished select features using CV of random forest or CV of logistic regression')
                            return True            
                        else:
                            sys.exit('rf_or_lr_feature_select='+rf_or_lr_feature_select+', cv==None')
                    elif cv!=None:#cross validation
                        ###1. select an optimal feature subset for logistic regression using CV from the feature subsets of GA and the set of all features, then, select an optimal regularization for logistic regression using CV
                        ###2. select an optimal feature subset for random forestusing using CV from the feature subsets of GA and the set of all features, then, select an optimal number of trees for random forest using CV
                        ###3. train a logistic regression using the optimal feature subset of logistic regression and the optimal regularization
                        ###4. train a random forest using the optimal feature subset of random forest and the optimal number of trees
                        (models,models_inputs_output,performance,models_discrete_cuts)=select_optimal_feature_subsets_and_parameters_then_train_classifiers(results_path+ga_input_data,results_path+original_training_set_arff,'none',reg,trees,seed,results_path,weka_path,java_memory,weka_train_discrete_file=trainset_discrete_arff,reductsfile=reductsfile,cross_validation=cv)
                    else:#training and testing or split training and testing                  
                        (models,models_inputs_output,performance,models_discrete_cuts)=select_optimal_feature_subsets_and_parameters_then_train_classifiers(results_path+ga_input_data,results_path+original_training_set_arff,results_path+testset_arff,reg,trees,seed,results_path,weka_path,java_memory,weka_train_discrete_file=trainset_discrete_arff,reductsfile=reductsfile)
                else:#training only continuous classifiers on all the features
                    if cv!=None:#cross validation
                        (models,models_inputs_output,performance,models_discrete_cuts)=select_optimal_feature_subsets_and_parameters_then_train_classifiers(results_path+ga_input_data,results_path+original_training_set_arff,'none',reg,trees,seed,results_path,weka_path,java_memory,cross_validation=cv)                
                    else:#training and testing or split training and testing
                        (models,models_inputs_output,performance,models_discrete_cuts)=select_optimal_feature_subsets_and_parameters_then_train_classifiers(results_path+ga_input_data,results_path+original_training_set_arff,results_path+testset_arff,reg,trees,seed,results_path,weka_path,java_memory)                        
        files_to_delete.append(trainset_discrete_arff)
        utilities.delete_files(files_to_delete)
        if len(models) == 1:
            file=open(logfile,'a')
            file.write('1 model trained at iteration: '+str(iteration)+'\n')
            file.close()
            print('1 model trained at iteration: ',iteration)
            return '1 model'
        elif len(models) == 2:
            file=open(logfile,'a')
            if regularization_of_best_log_reg[0]==-1:#set of all features
                file.write('optimal combination of feature subset and regularization: all features='+str(regularization_of_best_log_reg[1])+', regularization='+str(regularization_of_best_log_reg[2])+'\n')
                print('optimal combination of feature subset and regularization: all features='+str(regularization_of_best_log_reg[1])+', regularization='+str(regularization_of_best_log_reg[2]))
            else:
                file.write('optimal combination of feature subset and regularization: feature subset size='+str(regularization_of_best_log_reg[1])+', regularization='+str(regularization_of_best_log_reg[2])+'\n')                
                print('optimal combination of feature subset and regularization: feature subset size='+str(regularization_of_best_log_reg[1])+', regularization='+str(regularization_of_best_log_reg[2]))                
            if trees_of_best_rf[0]==-1:
                file.write('optimal combination of feature subset and trees: all features='+str(trees_of_best_rf[1])+', trees='+str(trees_of_best_rf[2])+'\n')
                print('optimal combination of feature subset and trees: all features='+str(trees_of_best_rf[1])+', trees='+str(trees_of_best_rf[2]))
            else:
                file.write('optimal combination of feature subset and trees: feature subset size='+str(trees_of_best_rf[1])+', trees='+str(trees_of_best_rf[2])+'\n')
                print('optimal combination of feature subset and trees: feature subset size='+str(trees_of_best_rf[1])+', trees='+str(trees_of_best_rf[2]))
            file.close()
            best_log_reg=models[0]
            best_rf=models[1]
            files_to_delete=[]
            if classifiersTypes == 'discrete classifiers only':
                utilities.mycopyfile(best_rf,results_path+'rf'+str(i)+'.model')
                utilities.mycopyfile(models_inputs_output[1],results_path+'rf'+str(i)+'.model_inputs_output.csv')
                files_to_delete.append(best_rf)
                files_to_delete.append(models_inputs_output[1])
            else:
                utilities.mycopyfile(best_log_reg,results_path+'log_reg'+str(i)+'.model')
                utilities.mycopyfile(models_inputs_output[0],results_path+'log_reg'+str(i)+'.model_inputs_output.csv')
                utilities.mycopyfile(best_rf,results_path+'rf'+str(i)+'.model')
                utilities.mycopyfile(models_inputs_output[1],results_path+'rf'+str(i)+'.model_inputs_output.csv')
                files_to_delete.append(best_log_reg)
                files_to_delete.append(models_inputs_output[0])
                files_to_delete.append(best_rf)
                files_to_delete.append(models_inputs_output[1])
            if len(models_discrete_cuts)==1:#a best classifier contains discrete features and another best classifier contains continuous features
                if models_discrete_cuts[0]==best_log_reg+".discrete_cuts":
                    utilities.mycopyfile(models_discrete_cuts[0],results_path+'log_reg'+str(i)+'.discrete_cuts')
                elif models_discrete_cuts[0]==best_rf+".discrete_cuts":
                    utilities.mycopyfile(models_discrete_cuts[0],results_path+'rf'+str(i)+'.discrete_cuts')
                files_to_delete.append(models_discrete_cuts[0])
            elif len(models_discrete_cuts)==2:#both best classifiers contain discrete features
                utilities.mycopyfile(models_discrete_cuts[0],results_path+'log_reg'+str(i)+'.discrete_cuts')
                utilities.mycopyfile(models_discrete_cuts[1],results_path+'rf'+str(i)+'.discrete_cuts')
                files_to_delete.append(models_discrete_cuts[0])
                files_to_delete.append(models_discrete_cuts[1])
            utilities.delete_files(files_to_delete)
            log_reg_performance=performance[0]#performance of best logistic regression: (train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)
            rf_performance=performance[1]#performance of best random forest: (train_test_auc2,train_auc2,train_tpr2,train_tnr2,train_fpr2,train_fnr2,test_auc2,test_tpr2,test_tnr2,test_fpr2,test_fnr2)
            train_auc=log_reg_performance[1]
            train_tpr=log_reg_performance[2]
            train_tnr=log_reg_performance[3]
            train_fpr=log_reg_performance[4]
            train_fnr=log_reg_performance[5]
            train_auc2=rf_performance[1]
            train_tpr2=rf_performance[2]
            train_tnr2=rf_performance[3]
            train_fpr2=rf_performance[4]
            train_fnr2=rf_performance[5]
            if cv!=None:#cross validation
                    train_xval_auc=log_reg_performance[0]                
                    xval_auc=log_reg_performance[6]#only xval auc is computed, tpr, fpr etc. not computed
                    train_xval_auc2=rf_performance[0]
                    xval_auc2=rf_performance[6]
                    log_reg_performance=(iteration,train_xval_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,xval_auc,-999,-999,-999,-999)
                    rf_performance=(iteration,train_xval_auc2,train_auc2,train_tpr2,train_tnr2,train_fpr2,train_fnr2,xval_auc2,-999,-999,-999,-999)                
            else:
                file=open(logfile,'a')
                if option=='train_test_sets':
                    testset=test_set_csv
                else:#split_train_test_sets
                    testset=results_path+'testset'+str(iteration)+'_no_missing_vals.csv'
                print('#####Predict testset#####')
                file.write('#####Predict testset.csv#####\n')
                file.close()
                if predict_ptb_of_each_id_using_all_spectra_of_id:
                   MP=mp.ModelsPredict()
                   (_,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)=MP.predict_using_all_spectra_of_each_patient(results_path+'rf'+str(iteration)+'.model','rf',results_path+'rf'+str(iteration)+'.model_inputs_output.csv',testset,w1,w2,results_path)
                   rf_performance=(test_auc,test_tpr,test_tnr,test_fpr,test_fnr)
                   (_,test_auc2,test_tpr2,test_tnr2,test_fpr2,test_fnr2)=MP.predict_using_all_spectra_of_each_patient(results_path+'log_reg'+str(iteration)+'.model','log_reg',results_path+'log_reg'+str(iteration)+'.model_inputs_output.csv',testset,w1,w2,results_path)
                   log_reg_performance=(test_auc2,test_tpr2,test_tnr2,test_fpr2,test_fnr2)           
                else:
                    if classifiersTypes=='discrete classifiers only':
                        rf_performance=MP.main(
                                i=iteration,
                                model_software2='weka',
                                modeltype2='random forest',
                                filter2_software2=None,                
                                testset_i2=testset,#test set for the ith filter
                                ordinal_encode2=True,
                                model_path2=results_path,
                                results_path2=results_path,
                                logfile2=logfile,
                                logfile2_option='a',
                                weka_path2=weka_path,
                                java_memory2=java_memory
                                )
                        log_reg_performance=(-999,-999,-999,-999,-999)#dummy results
                    else:
                        (rf_performance,log_reg_performance)=MP.main(
                                i=iteration,
                                model_software2='weka',
                                modeltype2='random forest and log regression',
                                filter2_software2=None,                
                                testset_i2=testset,
                                ordinal_encode2=True,
                                model_path2=results_path,
                                results_path2=results_path,
                                logfile2=logfile,
                                logfile2_option='a',
                                weka_path2=weka_path,
                                java_memory2=java_memory
                                )
                test_auc=log_reg_performance[0]
                test_tpr=log_reg_performance[1]
                test_tnr=log_reg_performance[2]
                test_fpr=log_reg_performance[3]
                test_fnr=log_reg_performance[4]
                test_auc2=rf_performance[0]
                test_tpr2=rf_performance[1]
                test_tnr2=rf_performance[2]
                test_fpr2=rf_performance[3]
                test_fnr2=rf_performance[4]
                train_test_auc=train_auc+test_auc
                train_test_auc2=train_auc2+test_auc2
                log_reg_performance=(iteration,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)
                rf_performance=(iteration,train_test_auc2,train_auc2,train_tpr2,train_tnr2,train_fpr2,train_fnr2,test_auc2,test_tpr2,test_tnr2,test_fpr2,test_fnr2)      
                file=open(logfile,'a')
                file.write('####performance on '+testset+'####\n')
                file.write('random forest:\n')
                file.write('model'+str(iteration)+'='+results_path+'rf'+str(iteration)+'.model\n')
                file.write('testset auc: '+str(test_auc2)+'\n')
                file.write('testset tpr: '+str(test_tpr2)+'\n')
                file.write('testset tnr: '+str(test_tnr2)+'\n')
                file.write('testset fpr: '+str(test_fpr2)+'\n')
                file.write('testset fnr: '+str(test_fnr2)+'\n')
                file.write('logistic regression:\n')
                file.write('model'+str(iteration)+'='+results_path+'log_reg'+str(iteration)+'.model\n')
                file.write('testset auc: '+str(test_auc)+'\n')
                file.write('testset tpr: '+str(test_tpr)+'\n')
                file.write('testset tnr: '+str(test_tnr)+'\n')
                file.write('testset fpr: '+str(test_fpr)+'\n')
                file.write('testset fnr: '+str(test_fnr)+'\n')
                file.close()
                if option == 'split_train_test':
                    utilities.summarize_results(logfile,log_reg_performance,rf_performance,log_regL,log_reg_train_aucL,log_reg_test_aucL,rf_L,rf_train_aucL,rf_test_aucL,model1='logistic regression',model2='random forest')                 
    file=open(logfile,'a')
    file.write('\t\t\t====Workflow1 finished====\n')
    file.write('The logistic regression models are saved to: '+results_path+'log_regN.model (N is the iteration number and N starts from 0)')
    file.write('\nThe random forest models are saved to: '+results_path+'rfN.model (N is the iteration number and N starts from 0)')
    file.write('\nThe results file is saved to: '+logfile)
    file.write('\nThe other files (training sets, balanced training sets, test sets and model_inputs_output files etc.) are saved to: '+results_path)
    file.close()
    print('\t\t\t====Workflow1 finished====')
    return (rf_performance,log_reg_performance)

def preprocess_trainset(step,option,trainset_csv,testset_csv,traindf,trainset_balanced_reduced_csv,trainset_balanced_reduced_arff,original_trainset_reduced_arff,testset_reduced_arff,results_path,logfile,balanced_trainset_size,degree,k,seed,weka_path,java_memory,oversampling_method='oversample_class1_and_class0_separately_using_repeated_bootstraps',interaction_only=True):
    ####training set -> oversampling->polynomial features construction -> information gain feature selection -> reduced balanced training set####
    #trainset_csv, training set with original features in a csv file
    #traindf, dataframe containing the training set
    #output: trainset_balanced_reduced_arff .arff file
    if os.path.isfile(logfile)==False:#if logfile does not exist, create a new one
       file=open(logfile,'w+')
    else:
       file=open(logfile,'a')
    (r,c)=traindf.shape
    cols=list(traindf.columns)
    preterm=traindf[traindf[cols[c-1]]==1]
    onterm=traindf[traindf[cols[c-1]]==0]    
    if balanced_trainset_size==999999:#Oversample class1 to the size of class0
        train_balanced=oversample_class1_to_size_of_class0(preterm,onterm)
    elif balanced_trainset_size==111111:#Undersample class0 to the size of class1 with replacement
        train_balanced=undersample_class0_to_size_of_class1(preterm,onterm,replacement=True)
    elif balanced_trainset_size==222222:#Undersample class0 to the size of class1 without replacement
        train_balanced=undersample_class0_to_size_of_class1(preterm,onterm,replacement=False)
    else:
        class_size=int(balanced_trainset_size/2)
        (class1,_)=preterm.shape
        (class0,_)=onterm.shape
        if class_size > class1 and class_size < class0:#undersample class0 to N and oversample class1 to N where N < size of class0
            print('oversample class1 to '+str(class_size)+' and undersample class0 to '+str(class_size))
            train_balanced=oversample_class1_and_undersample_class0(traindf,balanced_trainset_size,file,step,seed,results_path,weka_path,java_memory)
        else:#oversample class0 and class1 respectively to N where N > size of class0 > size of class1
            print('\t\t\t'+str(step)+'. Oversample the training set to a balanced training set')
            if oversampling_method == 'oversample_class1_and_class0_separately_using_repeated_bootstraps':
                train_balanced=oversample_class1_and_class0_separately_using_repeated_bootstraps(preterm,onterm,balanced_trainset_size)
                print('oversampling with repeated bootstraps')
                file.write('oversample_class1_and_class0_separately_using_repeated_bootstraps\n')
            elif oversampling_method == 'random_sampling_with_replacement':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'random_sampling_with_replacement')
                file.write('random sampling with replacement\n')
            elif oversampling_method == 'smote':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'smote')
                file.write('smote\n')
            elif oversampling_method == 'borderline_smote':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'borderline_smote')
                file.write('borderline smote\n')
            elif oversampling_method=='adasyn':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'adasyn')
                file.write('adasyn\n')    
            else:
                sys.exit('oversampling_method is invalid: '+oversampling_method)
                '''
                train_balanced=oversampling_with_recursive_boostrap(option,step,file,trainset_csv,traindf,balanced_trainset_size,seed,results_path,weka_path,java_memory)            
                file.write('oversampling with recursive bootstrap\n')
                '''
    step+=1
    if int(degree) > 1:
        print('\t\t\t'+str(step)+'. Construct polynomial features from the training set')
        file.write('\t\t\t'+str(step)+'. Construct polynomial features from the training set\n')
        (_,c)=traindf.shape
        no_of_poly_features=int(math.factorial((c-1)+int(degree))/(math.factorial(c-1)*math.factorial(int(degree))))
        print('\t\t\t  total Number of polynomial features: '+str(no_of_poly_features))
        file.write('\t\t\t total Number of polynomial features (excluding degree-0 polynomial feature \'1\'): '+str(no_of_poly_features-1)+'\n')    
        print('\t\t\tdegree of polynomial features: ',degree)    
        #train_balanced=prep.poly_features(train_balanced,int(degree),'none')  
        train_balanced=prep.poly_features_parallel(train_balanced,int(degree),n=10,interaction_only2=interaction_only)
        #print(train_balanced)
        if int(k) >= no_of_poly_features:#selecting all the poly features except the degree-0 poly feature '1' (redundant feature)
            k=no_of_poly_features-1
        step+=1
    if int(k) >= 1:    
        print('\t\t\t'+str(step)+'. Select '+str(k)+' features using information gain.')
        file.write('\t\t\t'+str(step)+'. Select '+str(k)+' features using information gain.\n')
        file.close()
        (_,c)=train_balanced.shape
        if c-1 <= 1000 or int(k) < 30:
            #utilities.dataframe_to_arff(train_balanced,results_path+'trainset_balanced.arff')
            prep.info_gain_fs(train_balanced,k,reduced_data_arff=results_path+trainset_balanced_reduced_arff)
        else:
            prep.parallel_feature_select(train_balanced,results_path+trainset_balanced_reduced_arff,m=int(k))
        utilities.get_model_inputs_output('arff',results_path+trainset_balanced_reduced_arff,results_path+'model_inputs_output.csv')
        step+=1
    else:#Do not select features using information gain
        if int(degree) > 1:#remove degree-0 poly feature (useless feature) from all the poly features
            del train_balanced['1']
        utilities.get_model_inputs_output('df',train_balanced,results_path+'model_inputs_output.csv')        
        utilities.dataframe_to_arff(train_balanced,results_path+trainset_balanced_reduced_arff)
    if testset_csv!='none':
        utilities.construct_poly_features_of_another_dataset('original_features',testset_csv,results_path+'model_inputs_output.csv','none',results_path+testset_reduced_arff)
    if original_trainset_reduced_arff!='none':
        utilities.construct_poly_features_of_another_dataset('original_features',trainset_csv,results_path+'model_inputs_output.csv','none',results_path+original_trainset_reduced_arff)
    if trainset_balanced_reduced_csv!='none':#save the reduced balanced training set to csv file 
        prep.convert_arff_to_csv(results_path+trainset_balanced_reduced_arff,results_path+trainset_balanced_reduced_csv,weka_path,java_memory)            
    del [traindf,train_balanced]#release RAM allocated to the dataframe
    return step    

def preprocess_trainset2(iteration,step,trainset_csv,testset_csv,traindf,trainset_reduced_csv,trainset_reduced_arff,original_trainset_reduced_arff,testset_reduced_arff,results_path,logfile,degree,k,weka_path,java_memory,interaction_only=True):
    ####training set -> polynomial features construction -> information gain feature selection -> (reduced poly features training set, reduced poly features testset)
    #trainset_csv, training set with original features in a csv file
    #traindf, dataframe containing the training set
    #output: trainset_reduced_arff .arff file    
    if os.path.isfile(logfile)==False:#if logfile does not exist, create a new one
       file=open(logfile,'w+')
    else:
       file=open(logfile,'a')
    if int(degree) > 1:
        (_,c)=traindf.shape
        print('\t\t\t'+str(step)+'.Construct polynomial features from the training set')
        file.write('\t\t\t'+str(step)+'.Construct polynomial features from the training set\n')
        print('\t\t\t\tdegree of polynomial features: ',degree)
        no_of_poly_features=int(math.factorial((c-1)+int(degree))/(math.factorial(c-1)*math.factorial(int(degree))))
        print('\t\t\t\tTotal Number of polynomial features: '+str(no_of_poly_features))
        file.write('\t\t\t Total Number of polynomial features: '+str(no_of_poly_features)+'\n')    
        traindf=prep.poly_features(traindf,int(degree),'none',interaction_only=interaction_only)
        del traindf['1']#remove the degree-0 poly feature (useless feature) from all the poly features
        step+=1
    else:
        no_of_poly_features=0
    if int(k) >= 1:
        if int(k) >= no_of_poly_features and no_of_poly_features > 0:#selecting all the poly features except the degree-0 poly feature '1' (redundant feature)
           k=no_of_poly_features-1
        print('\t\t\t'+str(step)+'. Select '+str(k)+' features using information gain.')
        file.write('\t\t\t'+str(step)+'. Select '+str(k)+' features using information gain.\n')
        file.close()
        (_,c)=traindf.shape
        if c-1 <= 1000 or int(k) < 30:
            utilities.dataframe_to_arff(traindf,results_path+'trainset.arff')
            prep.info_gain_fs(traindf,k,reduced_data_arff=results_path+trainset_reduced_arff)
        else:
            prep.parallel_feature_select(traindf,results_path+trainset_reduced_arff,m=int(k))
        step+=1
    else:#do not select features using information gain        
        utilities.dataframe_to_arff(traindf,results_path+trainset_reduced_arff)
    utilities.get_model_inputs_output('arff',results_path+trainset_reduced_arff,results_path+'model_inputs_output.csv')        
    if testset_csv!='none':
        utilities.reduce_data(testset_csv,results_path+'model_inputs_output.csv',results_path+'testset_reduced.csv')
        testdf=pd.read_csv(results_path+'testset_reduced.csv')
        utilities.dataframe_to_arff(testdf,results_path+testset_reduced_arff)
    if original_trainset_reduced_arff != 'none':#reduce the original training set using the features of the reduced training set
        utilities.reduce_data(trainset_csv,results_path+'model_inputs_output.csv',results_path+'trainset_reduced.csv')
        traindf=pd.read_csv(results_path+'trainset_reduced.csv')
        utilities.dataframe_to_arff(traindf,results_path+original_trainset_reduced_arff)
    if trainset_reduced_csv!='none':#save the reduced balanced training set to csv file 
        trainset_reduced_df=utilities.arff_to_dataframe(results_path+trainset_reduced_arff)
        trainset_reduced_df.to_csv(results_path+trainset_reduced_csv,index=False)
        del [trainset_reduced_df]#release RAM allocated to the dataframe
    utilities.delete_files([results_path+'trainset_poly_features.arff',results_path+'model_inputs_output.csv'])        
    del [traindf]#release RAM allocated to the dataframe      
    return step

def oversample_trainset(step,option,trainset_csv,testset_csv,traindf,trainset_balanced_csv,trainset_balanced_arff,original_trainset_arff,testset_arff,results_path,logfile,balanced_trainset_size,seed,weka_path,java_memory,oversampling_method='oversample_class1_and_class0_separately_using_repeated_bootstraps'):
    ####training set -> oversampling-> balanced training set####
    #trainset_csv, training set with original features in a csv file
    #traindf, dataframe containing the training set
    #output: trainset_balanced_reduced_arff .arff file
    if os.path.isfile(logfile)==False:#if logfile does not exist, create a new one
       file=open(logfile,'w+')
    else:
       file=open(logfile,'a')
    '''
    if option == 'cb_oversampling':
        print('\t\t\t'+str(step)+'.Constraint-based Oversampling of the training set to a balanced training set')
        eng = matlab.engine.start_matlab()
        lb=5
        #ub=35
        cols=list(traindf.columns)
        (_,c)=traindf.shape
        preterm=traindf[traindf[cols[c-1]]==1]
        onterm=traindf[traindf[cols[c-1]]==0]
        (class1,_)=preterm.shape
        (class0,_)=onterm.shape
        #ub=round(round(int(balanced_trainset_size)/2)/class1) + 20
        ub=120
        #ub=119
        lb2=1
        #ub2=15
        #ub2=round(round(int(balanced_trainset_size)/2)/class0) + 20
        ub2=120
        #ub2=119
        objectfunc='max_std'
        seed1=int(seed[0]) #used by Matlab to set start point of optimization problem
        seed2=seed[1] #used to select patients of each class to oversample
        no_of_solutions=10
        print('class1 bounds: lb=',lb,' ub=',ub)
        print('class0 bounds: lb2=',lb2,' ub2=',ub2)
        print('seed1: ',seed1)
        print('seed2: ',seed2)
        train_balanced=cb_oversample.cb_oversampling(traindf,int(balanced_trainset_size),lb,ub,lb2,ub2,objectfunc,seed1,seed2,no_of_solutions,eng)
        eng.quit()
    else:
    '''
    (r,c)=traindf.shape
    cols=list(traindf.columns)
    preterm=traindf[traindf[cols[c-1]]==1]
    onterm=traindf[traindf[cols[c-1]]==0]
    if balanced_trainset_size==999999:#Oversample class1 to the size of class0
       train_balanced=oversample_class1_to_size_of_class0(preterm,onterm)
    elif balanced_trainset_size==111111:#Undersample class0 to the size of class1 with replacement
         train_balanced=undersample_class0_to_size_of_class1(preterm,onterm,replacement=True)
    elif balanced_trainset_size==222222:#Undersample class0 to the size of class1 without replacement
         train_balanced=undersample_class0_to_size_of_class1(preterm,onterm,replacement=False)
    else:
         class_size=int(balanced_trainset_size/2)
         (class1,_)=preterm.shape
         (class0,_)=onterm.shape
         if class_size > class1 and class_size < class0:#oversample class1 to balanced_size/2 and undersample class0 to balanced_size/2
             print('oversample class1 to '+str(class_size)+' and undersample class0 to '+str(class_size))
             train_balanced=oversample_class1_and_undersample_class0(traindf,balanced_trainset_size,file,step,seed,results_path,weka_path,java_memory)
         else:#oversample class0 and class1 respectively to N where N > size of class0 > size of class1
            print('\t\t\t'+str(step)+'. Oversample the training set to a balanced training set')
            if oversampling_method == 'oversample_class1_and_class0_separately_using_repeated_bootstraps':
                train_balanced=oversample_class1_and_class0_separately_using_repeated_bootstraps(preterm,onterm,balanced_trainset_size)
                print('oversampling with repeated bootstraps')
                file.write('oversample_class1_and_class0_separately_using_repeated_bootstraps\n')
            elif oversampling_method == 'random_sampling_with_replacement':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'random_sampling_with_replacement')
                file.write(oversampling_method+'\n')
            elif oversampling_method == 'smote':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'smote')
                file.write('smote\n')
            elif oversampling_method == 'borderline_smote':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'borderline_smote')
                file.write('borderline smote\n')
            elif oversampling_method=='adasyn':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'adasyn')
                file.write('adasyn\n')    
            else:
                sys.exit('oversampling_method is invalid: '+oversampling_method)
                '''
                train_balanced=oversampling_with_recursive_boostrap(option,step,file,trainset_csv,traindf,balanced_trainset_size,seed,results_path,weka_path,java_memory)            
                file.write('oversampling with recursive bootstrap\n')
                '''
    utilities.dataframe_to_arff(train_balanced,results_path+trainset_balanced_arff)
    utilities.dataframe_to_arff(traindf,results_path+original_trainset_arff)
    if testset_csv!='none':
        utilities.dataframe_to_arff(pd.read_csv(testset_csv),results_path+testset_arff)
    del [traindf,train_balanced]#release RAM allocated to the dataframe
    step+=1
    return step

def oversample_trainset2(step,option,trainset_csv,testset_csv,traindf,trainset_balanced_reduced_csv,trainset_balanced_reduced_arff,original_trainset_reduced_arff,testset_reduced_arff,results_path,logfile,balanced_trainset_size,k,seed,weka_path,java_memory,oversampling_method='oversample_class1_and_class0_separately_using_repeated_bootstraps'):
    ####training set -> oversampling-> information gain feature selection -> (reduced balanced training set, reduced original training set, reduced test set)####
    #trainset_csv, original training set in a csv file
    #traindf, dataframe containing the original training set
    #output: trainset_balanced_reduced_arff .arff file
    if os.path.isfile(logfile)==False:#if logfile does not exist, create a new one
       file=open(logfile,'w+')
    else:
       file=open(logfile,'a')
    '''
    if option == 'cb_oversampling':
        print('\t\t\t'+str(step)+'.Constraint-based Oversampling of the training set to a balanced training set')
        eng = matlab.engine.start_matlab()
        lb=5
        #ub=35
        #ub=50
        ub=120
        cols=list(traindf.columns)
        (_,c)=traindf.shape
        preterm=traindf[traindf[cols[c-1]]==1]
        onterm=traindf[traindf[cols[c-1]]==0]
        (class1,_)=preterm.shape
        (class0,_)=onterm.shape
        #ub=round(round(int(balanced_trainset_size)/2)/class1) + 20
        lb2=1
        #ub2=15
        #ub2=20
        ub2=120
        #ub2=round(round(int(balanced_trainset_size)/2)/class0) + 20
        objectfunc='max_std'
        seed1=int(seed[0]) #used by Matlab to set start point of optimization problem
        seed2=seed[1] #used to select patients of each class to oversample
        #seed=random.randint(0,2**32-1)
        no_of_solutions=10
        print('class1 bounds: lb=',lb,' ub=',ub)
        print('class0 bounds: lb2=',lb2,' ub2=',ub2)
        print('seed1: ',seed1)
        print('seed2: ',seed2)
        train_balanced=cb_oversample.cb_oversampling(traindf,int(balanced_trainset_size),lb,ub,lb2,ub2,objectfunc,seed1,seed2,no_of_solutions,eng)
        eng.quit()
        #labels=prep.get_labels(traindf)
    else:
    '''
    (r,c)=traindf.shape
    cols=list(traindf.columns)
    preterm=traindf[traindf[cols[c-1]]==1]
    onterm=traindf[traindf[cols[c-1]]==0]
    if balanced_trainset_size==999999:#Oversample class1 to the size of class0
       train_balanced=oversample_class1_to_size_of_class0(preterm,onterm)
    elif balanced_trainset_size==111111:#Undersample class0 to the size of class1 with replacement
         train_balanced=undersample_class0_to_size_of_class1(preterm,onterm,replacement=True)
    elif balanced_trainset_size==222222:#Undersample class0 to the size of class1 without replacement
         train_balanced=undersample_class0_to_size_of_class1(preterm,onterm,replacement=False)
    else:
         class_size=int(balanced_trainset_size/2)
         (class1,_)=preterm.shape
         (class0,_)=onterm.shape
         if class_size > class1 and class_size < class0:#oversample class1 to balanced_size/2 and undersample class0 to balanced_size/2
             print('oversample class1 to '+str(class_size)+' and undersample class0 to '+str(class_size))
             train_balanced=oversample_class1_and_undersample_class0(traindf,balanced_trainset_size,file,step,seed,results_path,weka_path,java_memory)
         else:#oversample class0 and class1 respectively to N where N > size of class0 > size of class1
            print('\t\t\t'+str(step)+'. Oversample the training set to a balanced training set')
            if oversampling_method == 'oversample_class1_and_class0_separately_using_repeated_bootstraps':
                train_balanced=oversample_class1_and_class0_separately_using_repeated_bootstraps(preterm,onterm,balanced_trainset_size)
                print('oversampling with repeated bootstraps')
                file.write('oversample_class1_and_class0_separately_using_repeated_bootstraps\n')
            elif oversampling_method == 'random_sampling_with_replacement':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'random_sampling_with_replacement')
                file.write('random sampling with replacement\n')
            elif oversampling_method == 'smote':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'smote')
                file.write('smote\n')
            elif oversampling_method == 'borderline_smote':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'borderline_smote')
                file.write('borderline smote\n')
            elif oversampling_method=='adasyn':
                train_balanced=prep.balance_train_set(traindf,class_size,int(balanced_trainset_size-class_size),seed,'adasyn')
                file.write('adasyn\n')    
            else:
                sys.exit('oversampling_method is invalid: '+oversampling_method)
                '''
                train_balanced=oversampling_with_recursive_boostrap(option,step,file,trainset_csv,traindf,balanced_trainset_size,seed,results_path,weka_path,java_memory)            
                file.write('oversampling with recursive bootstrap\n')
                '''
    step+=1
    print('\t\t\t'+str(step)+'. Select '+str(k)+' features using information gain.')
    file.write('\t\t\t'+str(step)+'. Select '+str(k)+' features using information gain.\n')
    file.close()
    (_,c)=train_balanced.shape
    if c-1 <= 1000 or int(k) < 30:
        utilities.dataframe_to_arff(train_balanced,results_path+'trainset_balanced.arff')
        prep.info_gain_fs(train_balanced,k,reduced_data_arff=results_path+trainset_balanced_reduced_arff)      
    else:
        prep.parallel_feature_select(train_balanced,results_path+trainset_balanced_reduced_arff,m=int(k))
    utilities.get_model_inputs_output('arff',results_path+trainset_balanced_reduced_arff,results_path+'model_inputs_output.csv')   
    #reduce the test set using the features of the reduced training set
    train_balanced_reduced=utilities.arff_to_dataframe(results_path+trainset_balanced_reduced_arff)        
    cols=list(train_balanced_reduced.columns)
    if testset_csv!='none':
        testdf=pd.read_csv(testset_csv)
        utilities.dataframe_to_arff(testdf[cols],results_path+testset_reduced_arff)
    if trainset_balanced_reduced_csv!='none':#save the reduced balanced training set to csv file        
        train_balanced_reduced.to_csv(results_path+trainset_balanced_reduced_csv)
    if original_trainset_reduced_arff != 'none':
        utilities.dataframe_to_arff(traindf[cols],results_path+original_trainset_reduced_arff)
    if testset_csv!='none':
        del [traindf,train_balanced,train_balanced_reduced,testdf]
    else:
        del [traindf,train_balanced,train_balanced_reduced]
    step+=1
    return step
        
def discretize_ga_rsfs(i,weka_train_file,discretize_method,bins,populationSize,generations,crossover_prob,mutation_prob,fitness,number_of_reducts,ga_path,results_path,weka_path,java_memory):  
    #input: trainset in arff format
    #       GA parameters
    #output: reductsfile
    #        discretized trainset in arff format
    if discretize_method == 'pki':
       df=utilities.arff_to_dataframe(weka_train_file)
       (r,c)=df.shape
       bins=int(np.floor(np.sqrt(r)))
       prep.equal_freq_discretize(weka_train_file,results_path+'trainset_discrete.arff',bins,weka_path,java_memory)      
       #prep.pki_discretize(weka_train_file,results_path+'trainset_discrete.arff',weka_path,java_memory)
    elif discretize_method == 'equal_freq':
        prep.equal_freq_discretize(weka_train_file,results_path+'trainset_discrete.arff',bins,weka_path,java_memory)
    elif discretize_method == 'equal_width':
        prep.equal_width_discretize(weka_train_file,results_path+'trainset_discrete.arff',bins,weka_path,java_memory)
    else:
        print('invalid discretization method', discretize_method)
        return -1
    if int(number_of_reducts) > 0:
        reductsfile=results_path+"ga_reducts"+str(i)+".txt"
        prep.ordinal_encode(results_path+'trainset_discrete_integers.arff',arff_discrete_train=results_path+'trainset_discrete.arff')
        prep.ga_rsfs(results_path+'trainset_discrete_integers.arff',reductsfile,str(populationSize),str(generations),str(crossover_prob),str(mutation_prob),str(fitness),ga_path,weka_path,results_path,java_memory,my_os)   
        utilities.delete_files([results_path+'trainset_discrete_integers.arff'])
        return (reductsfile,results_path+'trainset_discrete.arff')
    else:
        return ('no feature selection',results_path+'trainset_discrete.arff')        

def select_optimal_feature_subsets_and_parameters_then_train_classifiers(weka_train_file,weka_original_train_file,weka_test_file,regularizer,trees,seed,results_path,weka_path,java_memory,weka_train_discrete_file=None,reductsfile=None,cross_validation=None):
    #best models with discretized inputs
    best_log_reg=results_path+"best_log_reg.model"
    best_rf=results_path+"best_rf.model"
    best_log_reg_inputs_output=results_path+"best_log_reg.model_inputs_output.csv"
    best_rf_inputs_output=results_path+"best_rf.model_inputs_output.csv" 
    #best models with continuous inputs
    best_log_reg2=results_path+"best_log_reg2.model"
    best_rf2=results_path+"best_rf2.model"
    best_log_reg_inputs_output2=results_path+"best_log_reg2.model_inputs_output.csv"
    best_rf_inputs_output2=results_path+"best_rf2.model_inputs_output.csv"
    global dim
    traindf=utilities.arff_to_dataframe(weka_train_file)
    (_,c)=traindf.shape
    dim=c-1
    #print('dim: ',dim)
    ###train and test logistic regression and random forest using set of all features and feature subsets
    if rf_or_lr_feature_select!=None:#reduce training set using random forest-selected features or logistic regression-selected features
        reduced_trainset_arff=train_test_classifiers_using_all_features_and_using_feature_subsets("continuous_data",number_of_reducts,reductsfile,weka_train_file,weka_original_train_file,weka_test_file,regularizer,trees,seed,results_path,best_log_reg2,best_rf2,best_log_reg_inputs_output2,best_rf_inputs_output2,weka_path,java_memory,cross_validation=cross_validation)
        return reduced_trainset_arff
    elif classifiersTypes=='discrete classifiers only':
       if cross_validation==None:#train and test classifier
           prep.discretize_using_cuts(weka_test_file,weka_train_discrete_file,results_path+'testset_discrete.arff','.',java_memory)
       if weka_original_train_file != 'none':
            num=random.randint(0,2**32-1)
            prep.discretize_using_cuts(weka_original_train_file,weka_train_discrete_file,results_path+'original_trainset_discrete'+str(num)+'.arff','.',java_memory)        
            (best_log_reg_performance,best_rf_performance,best_log_reg_discrete_cuts,best_rf_discrete_cuts)=train_test_classifiers_using_all_features_and_using_feature_subsets("discretized_data",number_of_reducts,reductsfile,weka_train_discrete_file,results_path+'original_trainset_discrete'+str(num)+'.arff',results_path+'testset_discrete.arff',regularizer,trees,seed,results_path,best_log_reg,best_rf,best_log_reg_inputs_output,best_rf_inputs_output,weka_path,java_memory,cross_validation=cross_validation)
       else:
            (best_log_reg_performance,best_rf_performance,best_log_reg_discrete_cuts,best_rf_discrete_cuts)=train_test_classifiers_using_all_features_and_using_feature_subsets("discretized_data",number_of_reducts,reductsfile,weka_train_discrete_file,'none',results_path+'testset_discrete.arff',regularizer,trees,seed,results_path,best_log_reg,best_rf,best_log_reg_inputs_output,best_rf_inputs_output,weka_path,java_memory,cross_validation=cross_validation)
    elif classifiersTypes=='continuous classifiers only':
       (best_log_reg_performance2,best_rf_performance2)=train_test_classifiers_using_all_features_and_using_feature_subsets("continuous_data",number_of_reducts,reductsfile,weka_train_file,weka_original_train_file,weka_test_file,regularizer,trees,seed,results_path,best_log_reg2,best_rf2,best_log_reg_inputs_output2,best_rf_inputs_output2,weka_path,java_memory,cross_validation=cross_validation)
    elif classifiersTypes=='discrete and continuous classifiers':#both discrete and continuous classifiers
       if cross_validation==None:
           prep.discretize_using_cuts(weka_test_file,weka_train_discrete_file,results_path+'testset_discrete.arff','.',java_memory)
       if weka_original_train_file != 'none':
            num=random.randint(0,2**32-1)
            prep.discretize_using_cuts(weka_original_train_file,weka_train_discrete_file,results_path+'original_trainset_discrete'+str(num)+'.arff','.',java_memory)        
            (best_log_reg_performance,best_rf_performance,best_log_reg_discrete_cuts,best_rf_discrete_cuts)=train_test_classifiers_using_all_features_and_using_feature_subsets("discretized_data",number_of_reducts,reductsfile,weka_train_discrete_file,results_path+'original_trainset_discrete'+str(num)+'.arff',results_path+'testset_discrete.arff',regularizer,trees,seed,results_path,best_log_reg,best_rf,best_log_reg_inputs_output,best_rf_inputs_output,weka_path,java_memory,cross_validation=cross_validation)
       else:
            (best_log_reg_performance,best_rf_performance,best_log_reg_discrete_cuts,best_rf_discrete_cuts)=train_test_classifiers_using_all_features_and_using_feature_subsets("discretized_data",number_of_reducts,reductsfile,weka_train_discrete_file,'none',results_path+'testset_discrete.arff',regularizer,trees,seed,results_path,best_log_reg,best_rf,best_log_reg_inputs_output,best_rf_inputs_output,weka_path,java_memory,cross_validation=cross_validation)
       (best_log_reg_performance2,best_rf_performance2)=train_test_classifiers_using_all_features_and_using_feature_subsets("continuous_data",number_of_reducts,reductsfile,weka_train_file,weka_original_train_file,weka_test_file,regularizer,trees,seed,results_path,best_log_reg2,best_rf2,best_log_reg_inputs_output2,best_rf_inputs_output2,weka_path,java_memory,cross_validation=cross_validation)
    else:
       sys.exit('invalid classifiersTypes: '+classifiersTypes)
    models=[]#models=[best logistic regression, best random forest]
    performance=[]#performance=[performance of best logistic regression, performance of best random forest]
    models_inputs_output=[]#[inputs and output of best logistic regression, inputs and output of best random forest]
    models_discrete_cuts=[]#[discrete cuts of best logistic regression, discrete cuts of best random forest] or []
    if classifiersTypes=='discrete classifiers only':
        models.append(best_log_reg)
        performance.append(best_log_reg_performance)
        models_inputs_output.append(best_log_reg_inputs_output)
        models_discrete_cuts.append(best_log_reg_discrete_cuts)
        models.append(best_rf)
        performance.append(best_rf_performance)   
        models_inputs_output.append(best_rf_inputs_output)
        models_discrete_cuts.append(best_rf_discrete_cuts)
    elif classifiersTypes=='continuous classifiers only':
        models.append(best_log_reg2)
        performance.append(best_log_reg_performance2)
        models_inputs_output.append(best_log_reg_inputs_output2)
        models.append(best_rf2)
        performance.append(best_rf_performance2)
        models_inputs_output.append(best_rf_inputs_output2) 
    elif classifiersTypes=='discrete and continuous classifiers':#both discrete and continuous classifiers; get the best of discrete classifier and continuos classifer
        if best_log_reg_performance[0] > best_log_reg_performance2[0]:
            models.append(best_log_reg)
            performance.append(best_log_reg_performance)
            models_inputs_output.append(best_log_reg_inputs_output)
            models_discrete_cuts.append(best_log_reg_discrete_cuts)
        else:
            models.append(best_log_reg2)
            performance.append(best_log_reg_performance2)
            models_inputs_output.append(best_log_reg_inputs_output2)     
        if best_rf_performance[0] > best_rf_performance2[0]:
            models.append(best_rf)
            performance.append(best_rf_performance)   
            models_inputs_output.append(best_rf_inputs_output)
            models_discrete_cuts.append(best_rf_discrete_cuts)
        else:
            models.append(best_rf2)
            performance.append(best_rf_performance2)
            models_inputs_output.append(best_rf_inputs_output2)
    return (models,models_inputs_output,performance,models_discrete_cuts)

def remove_single_feature_subsets(reductsL):
    reductsL2=[]
    for reduct in reductsL:
        m=re.match('^\d+$',reduct)        
        if m==None:#reduct size > 1, so keep it
            reductsL2.append(reduct)
    reductsL=reductsL2
    del [reductsL2]
    return reductsL

def get_feature_subsets(reductsL,min_size,max_size,num_of_reducts):
    #get the feature subsets between min_size and max_size from all the feature subsets found by ga_rsfs
    #print('get feature subsets of sizes >= min_size and <= max_size')
    reductsL2=[]
    for reduct in reductsL:
        m=re.match('^\d+$',reduct)#an one-feature-subset        
        if m==None:#reduct size > 1, so find its size
            fs=reduct.split(',')
            size=len(fs)
            if size >= min_size and size <= max_size:
                reductsL2.append(reduct)
                #print(reduct)
                #if len(reductsL2)==int(num_of_reducts):
                #    break;
    if reductsL2==[]:
        sys.exit('Error: GA feature selection did not find any high quality feature subsets (reducts) of between minimum size='+str(min_size)+' and maximum size='+str(max_size)+'\n Suggestion: set different min_size and max_size')
    if len(reductsL2) >= num_of_reducts:
        indxL=random.sample([i for i in range(len(reductsL2))],num_of_reducts)
    else:
        indxL=[i for i in range(len(reductsL2))]
    reductsL3=[]
    for indx in indxL:
        reductsL3.append(reductsL2[indx])
    return reductsL3

def get_top_k_feature_subsets(reductsL,k):
    #get the top-k ranked feature subsets found by ga_rsfs
    i=0
    reductsL2=[]
    while i<k:
        reduct=reductsL[i]
        reductsL2.append(reduct)
        i+=1
    return reductsL2

def get_random_feature_subsets(reductsL,number_of_reducts):
    if number_of_reducts >= len(reductsL):
        indxL=range(len(reductsL))
    else:
        k=300#top ranked k reducts
        if len(reductsL) < k:#if total no. of reducts < k, randomly select number_of_reducts from all the reducts
            indxL=random.sample(range(len(reductsL)),number_of_reducts)
        else:#select randomly number_of_reducts from the top ranked k reducts        
            indxL=random.sample(range(k),number_of_reducts)
    reductsL2=[]
    for indx in indxL:
        reduct=reductsL[indx]
        if reduct!='':
            reductsL2.append(reduct)
    return reductsL2

def get_feature_subsets_containing_core_features(reductsL,corefeatures):
    #get those feature subsets contain core features e.g. alpha and delta alpha features of cervical cancer dataset
    #reductsL, list of reducts (string) e.g. ["3,5,6","1,2,7,8"]
    #corefeatures, indices of core features starting from 1 (e.g. "4,8")
    corefeatures=corefeatures.split(',')
    corefeatures=set(corefeatures)
    reductsL2=[]
    for reduct in reductsL:
        reduct2=reduct.split(',')
        reduct2=set(reduct2)
        if corefeatures.issubset(reduct2):
            reductsL2.append(reduct)
    if len(reductsL2)>0:
        reductsL=reductsL2
        print('no. of reducts containing the core features: ',len(reductsL2))
    else:
        print('None of the reducts contains the core features: ',str(corefeatures))
    del [reductsL2]
    return reductsL
    
def train_test_classifiers_using_all_features_and_using_feature_subsets(option,number_of_reducts,reductsfile,weka_train_file,weka_original_train_file,weka_test_file,regularizer,trees,seed,results_path,best_log_reg,best_rf,best_log_reg_inputs_output,best_rf_inputs_output,weka_path,java_memory,cross_validation=None):
    #1.train and test models using all the features
    #2.select an optimal feature subset for logistic regression (if GA feature selection was performed), then, select an optimal regularization
    #3.select an optimal feature subset for random forest (if GA feature selection was performed), then, select an optimal trees
    #4.train logistic regression and random forest using the optimal feature subsets and optimal regularization and trees
    #
    #return: best_log_reg_performance
    #        best_rf_performance
    #option: discrete_data or continuous_data
    best_log_reg_performance=''
    best_rf_performance=''
    global regularization_of_best_log_reg
    global trees_of_best_rf
    if int(number_of_reducts) == 0 or reductsfile == 'no feature selection': #no feature selection is done, use all features to train classifiers
        (best_log_reg_performance,best_rf_performance)=determine_optimal_models_parameters_of_all_features_then_train_classifiers2(option,weka_train_file,weka_original_train_file,weka_test_file,regularizer,trees,seed,results_path,weka_path,java_memory,cross_validation=cross_validation)
        best_log_reg_performance = list(best_log_reg_performance)
        best_rf_performance = list(best_rf_performance)
        regularization_of_best_log_reg=best_log_reg_performance[-1]
        trees_of_best_rf=best_rf_performance[-1]
        if option=='continuous_data':
            utilities.mycopyfile(best_log_reg_performance[11],best_log_reg)#copy best logistic regression
            utilities.get_model_inputs_output('arff',best_log_reg_performance[12],best_log_reg_inputs_output)             
        utilities.mycopyfile(best_rf_performance[11],best_rf)#copy best random forest
        utilities.get_model_inputs_output('arff',best_rf_performance[12],best_rf_inputs_output)
        tempfiles=[]
        if option=='discretized_data':
            utilities.mycopyfile(best_rf_performance[12],best_rf+".discrete_cuts")               
        tempfiles.append(best_log_reg_performance[11])#model
        tempfiles.append(best_rf_performance[11])#model       
        tempfiles.append(best_log_reg_performance[12])
        tempfiles.append(best_rf_performance[12])
        utilities.delete_files(tempfiles)
        best_log_reg_performance=tuple(best_log_reg_performance)
        best_rf_performance=tuple(best_rf_performance)
        if option=='discretized_data':
            return (best_log_reg_performance,best_rf_performance,best_log_reg+".discrete_cuts",best_rf+".discrete_cuts")
        else:
            return (best_log_reg_performance,best_rf_performance)
    elif int(number_of_reducts) > 0:
        reductsL=[line.strip() for line in open(reductsfile)]
        if wrapper_es_fs==False:#remove single feature subsets from feature subsets which are found by GA RSFS
            reductsL=remove_single_feature_subsets(reductsL)
        if core_features!=None:
            reductsL=get_feature_subsets_containing_core_features(reductsL,core_features)
        if mini_features == -1 and max_features == -1:
            reductsL=get_random_feature_subsets(reductsL,int(number_of_reducts))
        elif int(mini_features) >0 and int(mini_features) <= int(max_features):
            reductsL=get_feature_subsets(reductsL,int(mini_features),int(max_features),int(number_of_reducts))
        else:
            sys.exit('invalid mini_features or max_features')
        #reductsL=get_top_k_feature_subsets(reductsL,int(number_of_reducts))
        log_reg_and_rf_performanceL=Parallel(n_jobs=-1,batch_size=10)(delayed(determine_optimal_models_parameters_of_a_feature_subset_then_train_classifiers)(
                                             j,
                                             reductsL,
                                             option,
                                             weka_train_file,
                                             weka_original_train_file,
                                             weka_test_file,
                                             [regularizer[0]],#use the 0th regularization
                                             [treesL[0]],#use the 0th trees
                                             seed,
                                             results_path,
                                             weka_path,
                                             java_memory,
                                             cross_validation=cross_validation) for j in range(len(reductsL)))        
         #                    0            1       2           3         4         5         6        7          8       9       10         11                 12                     13        
         #rf_performance=[train_xval_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,xval_auc,-999,    -999,     -999,   -999,results_path+rf_model,weka_train_file,optimal_trees_of_feature_subset]
        ###Select the optimal feature subset to be the one with the highest (training auc+CV auc) and the smallest size 
        if compare_with_set_of_all_features:
            (log_reg_performance,rf_performance)=determine_optimal_models_parameters_of_all_features_then_train_classifiers2(option,weka_train_file,weka_original_train_file,weka_test_file,[regularizer[0]],[treesL[0]],seed,results_path,weka_path,java_memory,cross_validation=cross_validation)
            #                    0            1       2           3         4         5         6        7          8       9       10         11                 12                     13        
            #rf_performance=[train_xval_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,xval_auc,-999,    -999,     -999,   -999,results_path+rf_model,weka_train_file,optimal_trees_of_feature_subset]
            log_reg_and_rf_performanceL.append((log_reg_performance,rf_performance))
        log_reg_performanceL=[]
        rf_performanceL=[]
        for log_reg_and_rf_performance in log_reg_and_rf_performanceL:
            log_reg_performance=log_reg_and_rf_performance[0]
            rf_performance=log_reg_and_rf_performance[1]
            log_reg_performanceL.append(log_reg_performance)
            rf_performanceL.append(rf_performance)
        ###set the criterion of a feature subset to (training auc+xval auc or training auc)-regularization*feature subset size/dimensionality of training set so that for 0 < lambda < 1, smaller weights are assigned to feature subset size/dimensionality of training set than the weights to (training_auc+cv_auc)
        reg=0.2 #The cost of the feature subset is regularization*feature subset size/dimensionality of training set; larger the size of a feature subset, the bigger its cost
        for log_reg_performance in log_reg_performanceL:
            train_xval_auc=log_reg_performance[0]
            regularization=log_reg_performance[-1]
            subset_size=regularization[1]
            criterion=train_xval_auc-reg*subset_size/dim
            log_reg_performance[0]=criterion
        for rf_performance in rf_performanceL:
            train_xval_auc=rf_performance[0]
            trees=rf_performance[-1]
            subset_size=trees[1]
            criterion=train_xval_auc-reg*subset_size/dim
            rf_performance[0]=criterion
        #print('criterion of feature subset: training auc + CV auc or training auc - '+str(lbda)+'*feature subset size/dimensionality of training set')
        log_reg_performanceL.sort(key=operator.itemgetter(0),reverse=True)#sort by criterion of feature subset
        rf_performanceL.sort(key=operator.itemgetter(0),reverse=True)
        best_log_reg_performance=log_reg_performanceL[0]
        best_rf_performance=rf_performanceL[0]
        regularization=best_log_reg_performance[-1]
        trees=best_rf_performance[-1]
        index_of_optimal_feature_subset=regularization[0] #index of best feature subset for logistic regression
        index_of_optimal_feature_subset2=trees[0] #index of best feature subset for random forest
        if rf_or_lr_feature_select=='lr':
           print('reduce training set using logistic regression-selected features')
           j=index_of_optimal_feature_subset
           reduct=reductsL[j]
           prep.reduce_a_weka_file(reduct,weka_path,weka_train_file,weka_train_file+'_reduced.arff',java_memory)
           return weka_train_file+'_reduced.arff' #reduced training set using random forest-select features or logistic regression-selected features
        elif rf_or_lr_feature_select=='rf': 
           print('reduce training set using random forest-selected features') 
           j=index_of_optimal_feature_subset2
           reduct=reductsL[j]
           prep.reduce_a_weka_file(reduct,weka_path,weka_train_file,weka_train_file+'_reduced.arff',java_memory)
           return weka_train_file+'_reduced.arff' #reduced training set using random forest-select features or logistic regression-selected features
        ###get the regularization of the optimal logistic regression corresponding to the optimal feature subset based on (training auc+Cv auc)
        if option=='continuous_data':
            if index_of_optimal_feature_subset==-1:#all features
                  (best_log_reg_performance,_)=determine_optimal_models_parameters_of_all_features_then_train_classifiers2(option,weka_train_file,weka_original_train_file,weka_test_file,regularizer,treesL,seed,results_path,weka_path,java_memory,cross_validation=cross_validation,log_reg=True,rf=False)
            else:
                  j=index_of_optimal_feature_subset #index of best feature subset for logistic regression
                  (best_log_reg_performance,_)=determine_optimal_models_parameters_of_a_feature_subset_then_train_classifiers(j,reductsL,option,weka_train_file,weka_original_train_file,weka_test_file,regularizer,treesL,seed,results_path,weka_path,java_memory,cross_validation=cross_validation,log_reg=True,rf=False)                    
        ###get the trees of the optimal random forest corresponding to the optimal feature subset based on (training auc+Cv auc)
        if index_of_optimal_feature_subset2==-1:
           (_,best_rf_performance)=determine_optimal_models_parameters_of_all_features_then_train_classifiers2(option,weka_train_file,weka_original_train_file,weka_test_file,regularizer,treesL,seed,results_path,weka_path,java_memory,cross_validation=cross_validation,log_reg=False,rf=True)
        else:
            j=index_of_optimal_feature_subset2 
            (_,best_rf_performance)=determine_optimal_models_parameters_of_a_feature_subset_then_train_classifiers(j,reductsL,option,weka_train_file,weka_original_train_file,weka_test_file,regularizer,treesL,seed,results_path,weka_path,java_memory,cross_validation=cross_validation,log_reg=False,rf=True)            
        regularization_of_best_log_reg=best_log_reg_performance[-1]#update global regularization_of_best_log_reg
        trees_of_best_rf=best_rf_performance[-1]#update global trees_of_best_rf
        if option=='continuous_data':
            utilities.mycopyfile(best_log_reg_performance[11],best_log_reg)#copy best logistic regression
            utilities.get_model_inputs_output('arff',best_log_reg_performance[12],best_log_reg_inputs_output)     
        utilities.mycopyfile(best_rf_performance[11],best_rf)#copy best random forest
        utilities.get_model_inputs_output('arff',best_rf_performance[12],best_rf_inputs_output)
        tempfiles=[]
        tempfiles.append(best_log_reg_performance[11])
        tempfiles.append(best_log_reg_performance[12])
        tempfiles.append(best_rf_performance[11])
        tempfiles.append(best_rf_performance[12])        
        for rf_performance in rf_performanceL:
              tempfiles.append(rf_performance[11])#model
              tempfiles.append(rf_performance[12])#train set     
        if option=='discretized_data':
                utilities.mycopyfile(best_rf_performance[12],best_rf+".discrete_cuts")
        else:
                for log_reg_performance in log_reg_performanceL:
                        tempfiles.append(log_reg_performance[11])#model
                        tempfiles.append(log_reg_performance[12])#training set
        utilities.delete_files(tempfiles)
        best_log_reg_performance=tuple(best_log_reg_performance)       
        best_rf_performance=tuple(best_rf_performance)
        if option=='discretized_data':
            return (best_log_reg_performance,best_rf_performance,best_log_reg+".discrete_cuts",best_rf+".discrete_cuts")
        else:
            return (best_log_reg_performance,best_rf_performance)

def determine_optimal_trees_of_random_forest(treesL='',
                                             j=None,#jth reduct
                                             inputs=None,#no. of inputs
                                             modelfile='rf_trees=',
                                             testresults='testresults_rf_trees=',
                                             trainresults='trainresults_rf_trees=',
                                             weka_train_file=None,
                                             weka_test_file=None,
                                             weka_original_train_file=None,
                                             seed=0,
                                             weka_path=None,
                                             java_memory=None,
                                             cross_validation=None,
                                             discretized_data=False #whether the input data is discrete or continuous
                                             ):
                best_rf_performance=[0,0,0,0,0,0,0,0,0,0,0,None,None,(None,0,None)]#(train_auc+test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,results_path+rf_model,reduced_weka_train_file,optimal_trees_of_feature_subset)     
                optimal_trees_of_feature_subset=''
                files_to_delete=[]
                count=0 #count of consecutive models after a best model
                if discretized_data:#encode discrete values of training set, original training set and test set using integers                               
                    num=random.randint(0,2**32-1)
                    enc=prep.ordinal_encode(results_path+'trainset_discrete_integers'+str(num)+'.arff',arff_discrete_train=weka_train_file)
                    if cross_validation==None:#train and test a classifier
                        #print('encode discrete intervals of test set as integers')
                        prep.ordinal_encode(results_path+'testset_discrete_integers'+str(num)+'.arff',enc=enc,arff_discrete_test=weka_test_file)
                        discretized_testset=results_path+'testset_discrete_integers'+str(num)+'.arff'
                    else:
                        discretized_testset='none'
                    if weka_original_train_file != 'none':            
                            #print('encode discrete intervals of original training set as integers')
                            prep.ordinal_encode(results_path+'original_trainset_discrete_integers'+str(num)+'.arff',enc=enc,arff_discrete_test=weka_original_train_file)            
                            discretized_original_trainset=results_path+'original_trainset_discrete_integers'+str(num)+'.arff'
                    else:
                        discretized_original_trainset='none'
                #print('treesL:',treesL)
                for trees in treesL:#train random forests of different trees until CV auc or test auc stays same or gets worse for some consecutive trees
                    rf_model=modelfile+str(trees)+'.model'
                    rf_testresults=testresults+str(trees)+'.txt'
                    rf_trainresults=trainresults+str(trees)+'.txt'
                    model_and_resultsfiles_rf=[rf_model,rf_testresults,rf_trainresults]
                    if discretized_data:
                       (_,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)=train_test_rf(False,str(trees),tree_depth,str(seed),results_path+'trainset_discrete_integers'+str(num)+'.arff',discretized_original_trainset,discretized_testset,results_path,model_and_resultsfiles_rf,weka_path,java_memory,'none','none',cross_validation=cross_validation)
                    else:      
                       (_,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)=train_test_rf(False,str(trees),tree_depth,str(seed),weka_train_file,weka_original_train_file,weka_test_file,results_path,model_and_resultsfiles_rf,weka_path,java_memory,'none','none',cross_validation=cross_validation)
                    files_to_delete.append(results_path+rf_testresults)
                    files_to_delete.append(results_path+rf_trainresults)
                    '''
                    if train_test_auc - best_rf_performance[0] > 0.01:
                        if best_rf_performance[11]!=None:
                            files_to_delete.append(best_rf_performance[11])#delete the old best model (not best model any more)
                        best_rf_performance=[train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,results_path+rf_model,weka_train_file]
                        optimal_trees_of_feature_subset=(j,inputs,trees)
                        best_rf_performance.append(optimal_trees_of_feature_subset)
                        count=0
                    '''
                    if (train_test_auc - best_rf_performance[0])>0: #and int(trees) < int(optimal_trees_of_feature_subset[1]):#current model has a slightly better performance than the best model, but fewer trees than the best model 
                        if best_rf_performance[11]!=None:
                            files_to_delete.append(best_rf_performance[11])#delete the old best model (not best model any more)
                        best_rf_performance=[train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,results_path+rf_model,weka_train_file]
                        optimal_trees_of_feature_subset=(j,inputs,trees)
                        best_rf_performance.append(optimal_trees_of_feature_subset)
                        count=0                        
                    else:#performace of this model not better than the best model or this model has a little better performance but more complex than the best model
                        files_to_delete.append(results_path+rf_model)
                        count+=1
                        if count==stop_cond2:
                            break
                if discretized_data:
                    files_to_delete.append(results_path+'trainset_discrete_integers'+str(num)+'.arff')
                    files_to_delete.append(discretized_testset)
                    files_to_delete.append(discretized_original_trainset)
                utilities.delete_files(files_to_delete)
                return best_rf_performance

def determine_optimal_regularization_of_log_reg(regularizerL=None,
                                             j=None,#jth reduct
                                             inputs=None,#no. of inputs
                                             modelfile='log_reg_reg=',
                                             testresults='testresults_logistic_reg=',
                                             trainresults='trainresults_logistic_reg=',
                                             weka_train_file=None,
                                             weka_test_file=None,
                                             weka_original_train_file=None,
                                             seed=0,
                                             weka_path=None,
                                             java_memory=None,
                                             cross_validation=None
                                             ):
        best_log_reg_performance=[0,0,0,0,0,0,0,0,0,0,0,None,None,(None,0,None)]#(train_auc+test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,results_path+log_reg_model,reduced_weka_train_file,optimal_regularization_of_feature_subset)
        optimal_regularization_of_feature_subset=''                             # 0                 , 1       , 2       , 3       ,  4      , 5       ,  6     ,  7     ,    8   ,   9    , 10     , 11                       , 12                    , 13 
        files_to_delete=[]
        count=0
        for reg in regularizerL:#train logistic regression of different regularization until CV auc or test auc does not improve for 3 consecutive regularizations           
            log_reg_model=modelfile+str(reg)+'.model'
            log_reg_testresults=testresults+str(reg)+'.txt'
            log_reg_trainresults=trainresults+str(reg)+'.txt'
            model_and_resultsfiles_log_reg=[log_reg_model,log_reg_testresults,log_reg_trainresults]                        
            (_,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)=train_test_log_reg(False,reg,weka_train_file,weka_original_train_file,weka_test_file,results_path,model_and_resultsfiles_log_reg,weka_path,java_memory,'none','none',cross_validation=cross_validation)
            files_to_delete.append(results_path+log_reg_testresults)
            files_to_delete.append(results_path+log_reg_trainresults)
            '''
            if train_test_auc - best_log_reg_performance[0] > 0.01:
                if best_log_reg_performance[11]!=None:
                    files_to_delete.append(best_log_reg_performance[11])#delete the old best model (not best model any more)
                if weka_original_train_file != 'none':
                    best_log_reg_performance=[train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,results_path+log_reg_model,weka_train_file]                     
                else:
                    best_log_reg_performance=[train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,results_path+log_reg_model,weka_train_file]                                     
                optimal_regularization_of_feature_subset=(j,inputs,reg)
                best_log_reg_performance.append(optimal_regularization_of_feature_subset)                                    
                count=0 #initialize count for the current best model
            '''
            if (train_test_auc - best_log_reg_performance[0])>0:# and float(reg) > float(optimal_regularization_of_feature_subset[1]):#current model has a similar performance to the best model, but smaller weights (larger regularization) than the best model 
                if best_log_reg_performance[11]!=None:
                   files_to_delete.append(best_log_reg_performance[11])#delete the old best model (not best model any more)
                if weka_original_train_file != 'none':
                    best_log_reg_performance=[train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,results_path+log_reg_model,weka_train_file]                     
                else:
                    best_log_reg_performance=[train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,results_path+log_reg_model,weka_train_file]                                         
                optimal_regularization_of_feature_subset=(j,inputs,reg)                                    
                best_log_reg_performance.append(optimal_regularization_of_feature_subset)                                    
                count=0 #initialize count for the current best model                        
            else:
                files_to_delete.append(results_path+log_reg_model)
                count+=1
                if count==stop_cond:#stop tuning model when stop_cond models are not better than the best model
                   break
        utilities.delete_files(files_to_delete)
        return best_log_reg_performance

def determine_optimal_models_parameters_of_all_features_then_train_classifiers2(option,weka_train_file,weka_original_train_file,weka_test_file,regularizerL,treesL,seed,results_path,weka_path,java_memory,cross_validation=None,log_reg=True,rf=True):
    #cross-validate, then, train (or train, then, test) a logistic regression and a random forest on all features of a training set respectively. 
    if treesL==['10*features']:
        #global dim
        treesL=[10*dim]
    if option=='discretized_data':
        option2=True
        best_log_reg_performance=[0,0,0,0,0,0,0,0,0,0,0,None,None,(None,0,None)]
        best_rf_performance=determine_optimal_trees_of_random_forest(treesL=treesL,
                                             j=-1,#jth reduct. -1 denotes all the features
                                             inputs=dim,
                                             modelfile='rf_trees=',
                                             testresults='testresults_rf_trees=',
                                             trainresults='trainresults_rf_trees=',
                                             weka_train_file=weka_train_file,
                                             weka_test_file=weka_test_file,
                                             weka_original_train_file=weka_original_train_file,
                                             seed=seed,
                                             weka_path=weka_path,
                                             java_memory=java_memory,
                                             cross_validation=cross_validation,
                                             discretized_data=option2
                                             )
        return (best_log_reg_performance,best_rf_performance)        
    else:#continuous data
        option2=False
        best_log_reg_performance=[0,0,0,0,0,0,0,0,0,0,0,None,None,(None,0,None)]
        best_rf_performance=[0,0,0,0,0,0,0,0,0,0,0,None,None,(None,0,None)]
        if rf:
            best_rf_performance=determine_optimal_trees_of_random_forest(treesL=treesL,
                                             j=-1,#jth reduct. -1 denotes all the features
                                             inputs=dim,
                                             modelfile='rf_trees=',
                                             testresults='testresults_rf_trees=',
                                             trainresults='trainresults_rf_trees=',
                                             weka_train_file=weka_train_file,
                                             weka_test_file=weka_test_file,
                                             weka_original_train_file=weka_original_train_file,
                                             seed=seed,
                                             weka_path=weka_path,
                                             java_memory=java_memory,
                                             cross_validation=cross_validation,
                                             discretized_data=option2
                                             )
        if log_reg:
            best_log_reg_performance=determine_optimal_regularization_of_log_reg(regularizerL=regularizerL,
                                             j=-1,#jth reduct
                                             inputs=dim,#no. of inputs
                                             modelfile='log_reg_reg=',
                                             testresults='testresults_logistic_reg=',
                                             trainresults='trainresults_logistic_reg=',
                                             weka_train_file=weka_train_file,
                                             weka_test_file=weka_test_file,
                                             weka_original_train_file=weka_original_train_file,
                                             seed=seed,
                                             weka_path=weka_path,
                                             java_memory=java_memory,
                                             cross_validation=cross_validation
                                             )
        return (best_log_reg_performance,best_rf_performance)

def determine_optimal_models_parameters_of_a_feature_subset_then_train_classifiers(j,reductsL,option,weka_train_file,weka_original_train_file,weka_test_file,regularizerL,treesL,seed,results_path,weka_path,java_memory,cross_validation=None,log_reg=True,rf=True):
    #cross-validate and train (or train and test) a logistic regression and a random forest on 1 reduced training set respectively on 1 cpu. Train N pairs of logistic regression and random forest on reduced training sets simultaneously on N CPUs.  
    files_to_delete=[]
    num=random.randint(0,2**32-1)
    reduced_weka_train_file=results_path+'trainset_reduced'+str(num)+'.arff'
    reduct=reductsL[j]
    reduct_size=len(reduct.split(','))
    prep.reduce_a_weka_file(reduct,weka_path,weka_train_file,reduced_weka_train_file,java_memory)
    reduced_weka_test_file='none'
    reduced_weka_original_train_file='none'
    if weka_original_train_file != 'none':
        reduced_weka_original_train_file=results_path+"original_trainset_reduced"+str(num)+".arff"
        prep.reduce_a_weka_file(reduct,weka_path,weka_original_train_file,reduced_weka_original_train_file,java_memory)
        files_to_delete.append(reduced_weka_original_train_file)
    else:
        reduced_weka_original_train_file='none'
    #cross validation and train logistic regression and random forest
    if treesL==['10*features']:
        treesL=[10*dim]
    if option=='discretized_data':
        option2=True
        num=random.randint(0,2**32-1)
        best_rf_performance=determine_optimal_trees_of_random_forest(treesL=treesL,
                                             j=j,#jth reduct.
                                             inputs=reduct_size,
                                             modelfile='rf'+str(num)+'_trees=',
                                             testresults='testresults_rf'+str(num)+'_trees=',
                                             trainresults='trainresults_rf'+str(num)+'_trees=',
                                             weka_train_file=reduced_weka_train_file,
                                             weka_test_file=reduced_weka_test_file,
                                             weka_original_train_file=reduced_weka_original_train_file,
                                             seed=seed,
                                             weka_path=weka_path,
                                             java_memory=java_memory,
                                             cross_validation=cross_validation,
                                             discretized_data=option2
                                             )
        utilities.delete_files(files_to_delete)
        best_log_reg_performance=[0,0,0,0,0,0,0,0,0,0,0,None,None,(None,0,None)]
        return (best_log_reg_performance,best_rf_performance)
    else:#continuous data
        option2=False
        num=random.randint(0,2**32-1)
        best_log_reg_performance=[0,0,0,0,0,0,0,0,0,0,0,None,None,(None,0,None)]
        best_rf_performance=[0,0,0,0,0,0,0,0,0,0,0,None,None,(None,0,None)]
        if rf:
            best_rf_performance=determine_optimal_trees_of_random_forest(treesL=treesL,
                                                 j=j,#jth reduct
                                                 inputs=reduct_size,
                                                 modelfile='rf'+str(num)+'_trees=',
                                                 testresults='testresults_rf'+str(num)+'_trees=',
                                                 trainresults='trainresults_rf'+str(num)+'_trees=',
                                                 weka_train_file=reduced_weka_train_file,
                                                 weka_test_file=reduced_weka_test_file,
                                                 weka_original_train_file=reduced_weka_original_train_file,
                                                 seed=seed,
                                                 weka_path=weka_path,
                                                 java_memory=java_memory,
                                                 cross_validation=cross_validation,
                                                 discretized_data=option2
                                                 )
        if log_reg:
            best_log_reg_performance=determine_optimal_regularization_of_log_reg(regularizerL=regularizerL,
                                             j=j,#jth reduct
                                             inputs=reduct_size,#no. of inputs
                                             modelfile='log_reg'+str(num)+'_reg=',
                                             testresults='testresults_logistic'+str(num)+'_reg=',
                                             trainresults='trainresults_logistic'+str(num)+'_reg=',
                                             weka_train_file=reduced_weka_train_file,
                                             weka_test_file=reduced_weka_test_file,
                                             weka_original_train_file=reduced_weka_original_train_file,
                                             seed=seed,
                                             weka_path=weka_path,
                                             java_memory=java_memory,
                                             cross_validation=cross_validation
                                             )
        utilities.delete_files(files_to_delete)
        return (best_log_reg_performance,best_rf_performance)

def oversample_class1_and_undersample_class0(traindf,balanced_trainset_size,file,step,seed,results_path,weka_path,java_memory):
    ###some instances of class0 and class1 are not included in the balanced dataset.
    cols=list(traindf.columns)
    (r,c)=traindf.shape
    z=str(np.round(balanced_trainset_size/r,4)*100)#size of the balanced training set as a percentage of the size of the imbalanced training set 
    labels=list(traindf.iloc[:,c-1])
    rows_indx=list(traindf.index)
    trainL=[]  
    for i in range(len(rows_indx)):
        indx_class=[rows_indx[i],labels[i]]
        trainL.append(indx_class)
    traindf_indx=pd.DataFrame(trainL,columns=['indx',cols[c-1]])
    num=random.randint(0,999999)
    utilities.dataframe_to_arff(traindf_indx,results_path+'trainset'+str(num)+'_indx.arff')
    #print('Oversample class1 and undersample class0 of the training set to create a balanced training set.')
    if file!='none' and step!='none':
        file.write('\t\t\t'+str(step)+'.Oversample class1 and undersample class0 of the training set to create a balanced training set\n')
    prep.resample(results_path+'trainset'+str(num)+'_indx.arff',results_path+'trainset'+str(num)+'_indx_balanced.arff',seed,z,weka_path,java_memory)
    train_balanced_indx=utilities.arff_to_dataframe(results_path+'trainset'+str(num)+'_indx_balanced.arff')
    cols=list(train_balanced_indx.columns)
    train_balanced_indx=train_balanced_indx.astype({cols[0]:int})#cast the 0th column (indices) to int type
    indx=list(train_balanced_indx.iloc[:,0])
    train_balanced=traindf.loc[indx]
    (r,c)=train_balanced.shape
    #print('balanced trainset size: ',r)
    cols=list(train_balanced.columns)
    labels=list(set(labels))
    for i in range(len(labels)):
        classidf=train_balanced[train_balanced[cols[c-1]]==i]
        (ci,_)=classidf.shape
        #print('class '+str(i)+' size: '+str(ci))
        #unique=classidf.drop_duplicates(keep='first')
        #print('unique instances of class: '+str(i)+': '+str(len(unique)))
    utilities.delete_files([results_path+'trainset'+str(num)+'_indx.arff',results_path+'trainset'+str(num)+'_indx_balanced.arff'])
    del [train_balanced_indx]
    return train_balanced

def undersample_class0_to_size_of_class1(class1df,class0df,replacement=True):
    #Undersample class0 to the size of class1 with replacement or without replacement
    #1. Resample s instances of class0 with replacement where s is size of class1
    #2. Combine the resampled instances with class1 to create a balanced dataset
    #3. Return the balanced dataset
    ###some instances of class0 are not included in the balanced dataset.
    if replacement:
        print('undersample class0 (majority class) to size of class1 (minority class) with replacement')
    else:
        print('undersample class0 (majority class) to size of class1 (minority class) without replacement')    
    (c1,_)=class1df.shape
    (c0,_)=class0df.shape
    if c1 > c0:
        print('size of class1: ',c1,' > size of class0: ',c0)
        print('size of class1 should be less than class0')
        sys.exit(-1)
    elif c1==c0:
        print('size of class: ',c1,' = size of class0: ',c0)
        train_balanced=pd.concat([class1df,class0df])
        del [class1df,class0df]
        return train_balanced
    seed=random.randint(0,999999)
    print('seed of undersampling',seed)
    indxL=list(class0df.index)
    from sklearn.utils import resample
    subsampleIndx=list(resample(indxL, n_samples=c1, replace=replacement, random_state=seed))
    class0_undersample=class0df.loc[subsampleIndx]
    train_balanced=pd.concat([class0_undersample,class1df])
    (r,c)=train_balanced.shape
    print('balanced trainset size: ',r)
    cols=list(train_balanced.columns)
    labels=[0,1]
    for i in labels:
        classidf=train_balanced[train_balanced[cols[c-1]]==i]
        (ci,_)=classidf.shape
        print('class '+str(i)+' size: '+str(ci))
        unique=classidf.drop_duplicates(keep='first')
        print('unique instances of class: '+str(i)+': '+str(len(unique)))   
    del [class0df,class1df]
    return train_balanced

def oversample_class1_to_size_of_class0(class1df,class0df):
    #Oversample class1 to the size of class0
    #1. create K bootstraps of class1 where K=size of class0/size of class1
    #2. Combine the bootstraps to create an oversampled class1 with equal size as size of class0
    #3. Combine the oversampled class1 with class0 to create a balanced training set
    #input: class1df, dataframe of class1 data
    #       class0df, dataframe of class0 data
    #output: balanced training set
    ###some instances of class1 are not included in the balanced dataset.
    print('oversample class1 (preterm) to size of class0 (onterm)')
    (c1,_)=class1df.shape
    (c0,_)=class0df.shape
    if c1 > c0:
        print('size of class1: ',c1,' > size of class0: ',c0)
        print('size of class1 should be less than class0')
        sys.exit(-1)
    elif c1==c0:
        print('size of class: ',c1,' = size of class0: ',c0)
        train_balanced=pd.concat([class1df,class0df])
        del [class1df,class0df]
        return train_balanced
    number_of_bootstraps=int(np.ceil(c0/c1))#round up to nearest integer
    from sklearn.utils import resample
    bootstrapIndxL=[]
    indxL=list(class1df.index)
    for i in range(number_of_bootstraps):
        seed=random.randint(0,999999)
        #print('seed of bootstrap',seed)
        bootstrapIndx=list(resample(indxL, n_samples=c1, replace=True, random_state=seed))
        bootstrapIndxL+=bootstrapIndx
    class1_oversample=class1df.loc[bootstrapIndxL]
    train_balanced=pd.concat([class1_oversample,class0df])
    (r,c)=train_balanced.shape
    print('balanced trainset size: ',r)
    cols=list(train_balanced.columns)
    labels=[0,1]
    for i in labels:
        classidf=train_balanced[train_balanced[cols[c-1]]==i]
        (ci,_)=classidf.shape
        print('class '+str(i)+' size: '+str(ci))
        unique=classidf.drop_duplicates(keep='first')
        print('unique instances of class: '+str(i)+': '+str(len(unique)))   
    del [class0df,class1df]
    return train_balanced

def oversample_class1_and_class0_separately_using_repeated_bootstraps(class1df,class0df,balanced_trainset_size): 
    #Oversample class1 and class0 seperately
    #1. Create M bootstraps of class0 where M=balanced_trainset_size/2/size of class0 to form an oversampled class0
    #2. Add copies of any class0 instances which are not included in the oversampled class0 
    #3. If size of overrsampled class0 < size of balanced training set/2
    #   draw more instances of class0 with replacement and add them to oversampled class0 to get the size (balanced training set size /2)
    #4. Repeat 1 to 3 on class1 to create an oversampled class1
    #5. Combine the oversampled class0 with the oversampled class1 to create a balanced training set
    #A bootstrap contains roughly 63% of the training set. 
    #To include all instances of the training set into the oversampled training set, at least 10 bootstraps (more bootstraps for the minority class) are created for each class  
    #input: class1df, dataframe of class1 data
    #       class0df, dataframe of class0 data
    #       balanced_trainset_size
    #output: balanced training set
    #print('oversample class1 and class0 individually')
    from sklearn.utils import resample
    (c1,_)=class1df.shape
    (c0,_)=class0df.shape
    if c1 > c0:
        print('size of class1: ',c1,' > size of class0: ',c0)
        print('size of class1 should be less than class0')
        sys.exit(-1)
    elif c1==c0:
        print('size of class1: ',c1,' = size of class0: ',c0)
        train_balanced=pd.concat([class1df,class0df])
        del [class1df,class0df]
        return train_balanced    
    #create bootstraps of majority class to include in oversampled training set
    number_of_bootstraps0=int(np.floor(balanced_trainset_size/2/c0)-1)#increase diff0 by creating 1 bootstrap less than (balanced_trainset_size/2/c0) bootstraps of majority class    
    bootstrapIndxL0=[]
    indxL0=list(class0df.index)
    for i in range(number_of_bootstraps0):
        seed=random.randint(0,999999)
        bootstrapIndx0=list(resample(indxL0, n_samples=c0, replace=True, random_state=seed))
        bootstrapIndxL0+=bootstrapIndx0
    class0_oversample=class0df.loc[bootstrapIndxL0]
    (c0_oversample,_)=class0_oversample.shape
    #add copies of any class0 instances not included in the oversampled class0
    diff0 = int(balanced_trainset_size/2) - c0_oversample        
    df=prep.dataframes_diff(class0df.drop_duplicates(keep='first'),class0_oversample.drop_duplicates(keep='first'))
    (r,_)=df.shape
    if r>0:   
        number_of_copies=int(np.floor(diff0/r))
        L=[class0_oversample]
        for i in range(number_of_copies):
            L.append(df)
        class0_oversample=pd.concat(L) 
    (c0_oversample,_)=class0_oversample.shape    
    if c0_oversample < balanced_trainset_size/2:#if size of overrsampled class0 < size of balanced training set/2, draw more instances of class0 with replacement and add them to oversampled class0 to get the size (balanced training set size /2)
        diff0 = int(balanced_trainset_size/2) - c0_oversample
        seed=random.randint(0,999999)
        bootstrapIndx0=list(resample(indxL0, n_samples = diff0, replace=True, random_state=seed))
        resample0=class0df.loc[bootstrapIndx0]
        class0_oversample=pd.concat([class0_oversample,resample0])    
    '''
    #create copies of majority class to include in oversampled training set
    number_of_copies=int(np.floor(balanced_trainset_size/2/c0))
    print('number_of_copies:',number_of_copies)
    L=[]
    for i in range(number_of_copies):
       L.append(class0df)
    class0_oversample=pd.concat(L)
    indxL0=list(class0df.index)
    if number_of_copies*c0 < balanced_trainset_size/2:#need to draw more instances of class0
        diff0 = int(balanced_trainset_size/2 - number_of_copies*c0)
        seed=random.randint(0,999999)
        bootstrapIndxL0=list(resample(indxL0, n_samples = diff0, replace=True, random_state=seed))
        class0_oversample=pd.concat([class0_oversample,class0df.loc[bootstrapIndxL0]])
    '''    
    #create bootstraps of minority class to include in oversampled training set
    bootstrapIndxL1=[]
    indxL1=list(class1df.index)
    number_of_bootstraps1=int(np.floor(balanced_trainset_size/2/c1))  
    for i in range(number_of_bootstraps1):
        seed=random.randint(0,999999)
        bootstrapIndx1=list(resample(indxL1, n_samples=c1, replace=True, random_state=seed))
        bootstrapIndxL1+=bootstrapIndx1
    class1_oversample=class1df.loc[bootstrapIndxL1]
    (c1_oversample,_)=class1_oversample.shape
    (c0_oversample,_)=class0_oversample.shape
    # add copies of any class1 instances not included in the oversampled class1
    diff1 = c0_oversample - c1_oversample        
    df=prep.dataframes_diff(class1df.drop_duplicates(keep='first'),class1_oversample.drop_duplicates(keep='first'))
    (r,_)=df.shape
    if r > 0:
       number_of_copies=int(np.floor(diff1/r))
       L=[class1_oversample]
       for i in range(number_of_copies):
           L.append(df)
       class1_oversample=pd.concat(L)
    (c1_oversample,_)=class1_oversample.shape    
    if c1_oversample < c0_oversample:#if size of overrsampled class1 < size of oversampled class0, draw more instances of class1 with replacement and add them to oversampled class1 to get same size of class0
        diff1 = c0_oversample - c1_oversample
        seed=random.randint(0,999999)
        bootstrapIndx1=list(resample(indxL1, n_samples = diff1, replace=True, random_state=seed))
        resample1=class1df.loc[bootstrapIndx1]
        class1_oversample=pd.concat([class1_oversample,resample1])
    train_balanced=pd.concat([class0_oversample,class1_oversample])
    (r,c)=train_balanced.shape
    print('balanced trainset size: ',r)
    cols=list(train_balanced.columns)
    labels=[0,1]
    for i in labels:
        classidf=train_balanced[train_balanced[cols[c-1]]==i]
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
    del [class0df,class1df]
    return train_balanced
    
def oversampling_with_recursive_boostrap(option,step,file,trainset_csv,traindf,balanced_trainset_size,seed,results_path,weka_path,java_memory):
        #Oversample both class1 and class0 to N where N > size of class0 > size of class1 
        #
        ###some instances of the imbalanced dataset are not included in the balanced dataset.
        ###However, the size of the balanced dataset is normally a little greater than the specified size balanced_trainset_size
        #Input: D, an imbalanced dataset 
        #       S, size of the balanced dataset
        #Output: D', a balanced dataset
        #1. D' <- resample_with_replacement(D,S) where |D'|=S and D' is a balanced dataset
        #2. D'_unique <- remove_duplicates(D')
        #3. D'=add_remaining_samples_to_balanced_dataset(a,D,D'_unique,D')
        #4. return D'
        #
        #function add_remaining_samples_to_balanced_dataset(a,D,D'_unique,Balanced_data)
        #input: a, threshold
        #       D, data set
        #       D'_unique, dataset containing the unique instances of a resampled dataset or bootstrap
        #       Balanced_data, balanced data set
        #output: Balanced_data       
        #1. S <- D - D'_unique where S is the subset data of D which are not included in Balanced_data
        #2. if(|S|>a) 
        #3. then { S'<- boostrap(S)
        #4.        Balanced_data <- Balanced_data + S'
        #5.        S'_unique <- remove_duplicates(S')
        #6.        return add_remaining_samples_to_balanced_dataset(a,S,S'_unique,Balanced_data)
        #7. }           
        #8. else { S' <- boostrap(S)
        #9.        Balanced_data <- Balanced_data + S'
        #10.      return Balanced_data
        #11.}
        #print('oversampling with recursive bootstrap')
        labels=prep.get_labels(traindf)
        (r,c)=traindf.shape
        cols=list(traindf.columns)
        z=str(np.round(balanced_trainset_size/r,4)*100)#size of the balanced training set as a percentage of the size of the imbalanced training set 
        labels=list(traindf.iloc[:,c-1])
        rows_indx=list(traindf.index)
        trainL=[]  
        for i in range(len(rows_indx)):
            indx_class=[rows_indx[i],labels[i]]
            trainL.append(indx_class)
        traindf_indx=pd.DataFrame(trainL,columns=['indx',cols[c-1]])
        num=random.randint(0,999999)
        utilities.dataframe_to_arff(traindf_indx,results_path+'trainset'+str(num)+'_indx.arff')
        if file!='none' and step!='none':
            file.write('\t\t\t'+str(step)+'.Oversample the training set to a balanced training set\n')
        prep.resample(results_path+'trainset'+str(num)+'_indx.arff',results_path+'trainset'+str(num)+'_indx_balanced.arff',seed,z,weka_path,java_memory)
        df=utilities.arff_to_dataframe(results_path+'trainset'+str(num)+'_indx_balanced.arff')
        df.to_csv(results_path+'trainset'+str(num)+'_indx_balanced.csv',index=False)
        prep.remove_duplicates(results_path+'trainset'+str(num)+'_indx_balanced.csv',results_path+'trainset'+str(num)+'_indx_balanced_unique.csv')
        train_balanced_unique=pd.read_csv(results_path+'trainset'+str(num)+'_indx_balanced_unique.csv',low_memory=False)
        train_balanced=pd.read_csv(results_path+'trainset'+str(num)+'_indx_balanced.csv')
        if option == 1:
            train_balanced_indx=add_remaining_samples_to_balanced_trainset(3,traindf_indx,train_balanced_unique,train_balanced,results_path,labels,seed,weka_path,java_memory)                
        elif option == 2:
            train_balanced_indx=add_remaining_samples_to_balanced_trainset2(3,traindf_indx,train_balanced_unique,train_balanced,results_path,labels,seed,weka_path,java_memory)
        elif option == 3:
            train_balanced_indx=add_remaining_samples_to_balanced_trainset3(3,traindf_indx,train_balanced_unique,train_balanced,results_path,labels,seed,weka_path,java_memory)
        else:
            sys.exit('invalid option in oversampling_with_recursive_boostrap: ',option)
        ###debug###
        #train_balanced_indx_unique=train_balanced_indx.drop_duplicates(keep='first')
        #(r,c)=train_balanced_indx_unique.shape
        #print('unique: ',r,', c:',c)
        #cols=list(train_balanced_indx.columns)
        #n=train_balanced_indx[cols[c-1]].nunique()
        #print('classes: ',n)
        ###
        cols=list(train_balanced_indx.columns)
        train_balanced_indx=train_balanced_indx.astype({cols[0]:int})#cast the 0th column (indices) to int type
        train_balanced_indx2=balance_trainset(train_balanced_indx,balanced_trainset_size)
        indx=list(train_balanced_indx2.iloc[:,0])
        train_balanced=traindf.loc[indx]
        (r,c)=train_balanced.shape
        #print('balanced trainset size: ',r)
        cols=list(train_balanced.columns)
        labels=list(set(labels))
        for i in range(len(labels)):
            classidf=train_balanced[train_balanced[cols[c-1]]==i]
            (ci,_)=classidf.shape
            #print('class '+str(i)+' size: '+str(ci))
            #unique=classidf.drop_duplicates(keep='first')
            #print('unique instances of class: '+str(i)+': '+str(len(unique)))   
        del [df,traindf]
        utilities.delete_files([results_path+'trainset'+str(num)+'_indx.arff',results_path+'trainset'+str(num)+'_indx_balanced_unique.csv',results_path+'trainset'+str(num)+'_indx_balanced.csv',results_path+'trainset'+str(num)+'_indx_balanced.arff'])
        return train_balanced

def balance_trainset(train_balanced_indx,balanced_trainset_size):
    #increase or decrease each class to balanced_trainset_size/2 by adding or removing instances to/from each class
    #input: train_balanced_indx, rows indices of balanced training set
    #       balanced_trainset_size, size of balanced training set
    (_,c)=train_balanced_indx.shape
    cols=list(train_balanced_indx.columns)
    n=train_balanced_indx[cols[c-1]].nunique()
    L=[]
    L2=[]
    if n>2:#multi class
        classi=int(np.round(balanced_trainset_size/n))
        for i in range(n):
            classidf=train_balanced_indx[train_balanced_indx[cols[c-1]]==i]
            (ci,_)=classidf.shape
            L.append((ci,classidf))
            L2.append(classi)
    elif n==2:
        class1=int(balanced_trainset_size/2)#specified class1 size
        class0=balanced_trainset_size-class1#specified class0 size
        (_,c)=train_balanced_indx.shape
        cols=list(train_balanced_indx.columns)
        class1df=train_balanced_indx[train_balanced_indx[cols[c-1]]==1]
        class0df=train_balanced_indx[train_balanced_indx[cols[c-1]]==0]
        (c1,_)=class1df.shape#c1: size of class1 data of balanced training set
        (c0,_)=class0df.shape#c0: size of class0 data of balanced training set
        L=[(c0,class0df), (c1,class1df)]
        L2=[class0, class1]
    balanced=pd.DataFrame(columns=cols)
    for i in range(len(L)):
        (c,df)=L[i]
        classi=L2[i]
        if c > classi:
            di=c-classi
            #remove diff instances from a class classdf as follows:
            #0. Initialize the no. of the selected instances for removal to 0
            #1. Initialize the no. of the selected unique instances for removal to 0
            #2. if no. of the selected instances < diff and no. of selected unique instances < total no. of unique instances  
            #3. then select an unique instance to remove from the class;
            #4.      go to step 2;
            #5. else if no. of the selected instances < diff and no. of selected unique instances == total no. of unique instances
            #6. then go to step 1.
            #6. else if no. of selected instances == diff
            #7. then return classdf - the selected instances to remove
            #df=remove_instances(df,di,set())
            indxL=list(df.index)
            to_remove_indx=[]
            to_remove_indx=random.sample(indxL,di)#sample di instance without replacement
            df=df.drop(to_remove_indx)            
            (r,_)=df.shape
        elif c < classi:
            di=classi-c
            indxL=list(df.index)#list of row labels of classi
            to_add_indx=[]
            for _ in range(di):#resample di instances with replacement
                to_add_indx.append(random.choice(indxL))
            to_add=df.loc[to_add_indx]
            df=pd.concat([df,to_add])
            (r,_)=df.shape
        balanced=pd.concat([balanced,df])
    return balanced

def remove_instances(classdf,diff,to_remove_indx):
    to_remove_unqiue_rows=set()
    unique=classdf.drop_duplicates(keep='first')#get the unique instances of this class
    indx=set(classdf.index) #set of indices (row labels)
    remainL=list(indx-to_remove_indx)#the labels of remaining rows for selection for removal
    for r in remainL:     
        row=str(classdf.loc[r])
        if len(to_remove_indx) < diff:
            if r not in to_remove_indx and row not in to_remove_unqiue_rows:
                to_remove_unqiue_rows.add(row)
                to_remove_indx.add(r) 
            if len(to_remove_unqiue_rows)==len(unique):
                return remove_instances(classdf,diff,to_remove_indx)
        elif len(to_remove_indx) == diff:
            to_keep_indx=list(indx-to_remove_indx)
            return classdf.loc[to_keep_indx]
        else:
            print('to_remove_indx > diff')

def add_remaining_samples_to_balanced_trainset(diff_threshold,traindf,train_balanced_unique,train_balanced,results_path,labels,seed,weka_path,java_memory):
    #for a training set of binary class
    #replace r class0 instances of the balanced training set with a bootstrap of the remaining r class0 instances (r > diff_threshold)
    num=random.randint(0,2**32-1)
    traindf=traindf.round(4)#round to 4 decimal places
    train_balanced_unique=train_balanced_unique.round(4)#round to 4 decimal places
    df=prep.dataframes_diff(traindf,train_balanced_unique)
    (r,_)=df.shape
    #print('difference of traindf - train_balanced_unique: '+str(r))
    if r > diff_threshold:
        df.to_csv(results_path+'df'+str(num)+'.csv',index=False)    
        prep.convert_csv_to_arff(results_path+'df'+str(num)+'.csv',results_path+'df'+str(num)+'.arff',"last:"+labels,weka_path,java_memory)
        prep.resample(results_path+'df'+str(num)+'.arff',results_path+'df_resampled'+str(num)+'.arff',seed,'100',weka_path,java_memory)#boostrap 1
        prep.resample(results_path+'df_resampled'+str(num)+'.arff',results_path+'df_resampled2'+str(num)+'.arff',seed,'100',weka_path,java_memory)#boostrap 2
        prep.convert_arff_to_csv(results_path+'df_resampled2'+str(num)+'.arff',results_path+'df_resampled2'+str(num)+'.csv',weka_path,java_memory)
        df_resampled2_data=pd.read_csv(results_path+'df_resampled2'+str(num)+'.csv',low_memory=False)        
        (_,c)=train_balanced.shape
        cols=list(train_balanced.columns)
        pos=train_balanced[train_balanced[cols[c-1]]==1]    
        neg=train_balanced[train_balanced[cols[c-1]]==0]
        (n,_)=neg.shape
        indx=random.sample([i for i in range(n)],r)#randomly select r indices from the list of all the indices of the neg instances of train_balanced 
        neg=prep.replace_rows_of_dataframe(indx,neg,df_resampled2_data)#replace the rows at r0 indices with the neg training instances
        train_balanced=pd.concat([neg,pos])
        prep.remove_duplicates(results_path+'df_resampled2'+str(num)+'.csv',results_path+'df_resampled2_unique'+str(num)+'.csv')
        df_resampled2_unique=pd.read_csv(results_path+'df_resampled2_unique'+str(num)+'.csv')
        df2=prep.dataframes_diff(df,df_resampled2_unique)
        (r2,_)=df2.shape
        #print('difference of df - df_resampled2_unique: '+str(r2))
        utilities.delete_files([results_path+'df'+str(num)+'.csv',results_path+'df'+str(num)+'.arff',results_path+'df_resampled'+str(num)+'.arff',results_path+'df_resampled2'+str(num)+'.arff',results_path+'df_resampled2'+str(num)+'.csv',results_path+'df_resampled2_unique'+str(num)+'.csv'])
        return train_balanced
    elif r <= diff_threshold:
        train_balanced=pd.concat([train_balanced,df,df,df,df,df,df])#add the remaining samples and their 5 duplicates
        (_,c)=train_balanced.shape
        cols=list(train_balanced.columns)
        pos=train_balanced[train_balanced[cols[c-1]]==1]    
        neg=train_balanced[train_balanced[cols[c-1]]==0]
        (n,_)=pos.shape
        (n2,_)=neg.shape
        d=n2-n
        if d>0:#if more neg than pos, add a subsample of pos
            pos.to_csv(results_path+'pos'+str(num)+'.csv',index=False)
            prep.convert_csv_to_arff(results_path+'pos'+str(num)+'.csv',results_path+'pos'+str(num)+'.arff',"last:"+labels,weka_path,java_memory)
            z=str(int(d)/n*100)#subsample d instances from the pos class (n instances) of the training set
            prep.resample(results_path+'pos'+str(num)+'.arff',results_path+'pos_subsample'+str(num)+'.arff',str(seed),str(z),weka_path,java_memory)
            prep.convert_arff_to_csv(results_path+'pos_subsample'+str(num)+'.arff',results_path+'pos_subsample'+str(num)+'.csv',weka_path,java_memory)
            df=pd.read_csv(results_path+'pos_subsample'+str(num)+'.csv')
            train_balanced=pd.concat([train_balanced,df])
            utilities.delete_files([results_path+'pos'+str(num)+'.csv',results_path+'pos'+str(num)+'.arff',results_path+'pos_subsample'+str(num)+'.arff',results_path+'pos_subsample'+str(num)+'.csv'])
        elif d<0:#more pos than neg, add a subsample of neg
            d=-d
            neg.to_csv(results_path+'neg'+str(num)+'.csv',index=False)
            prep.convert_csv_to_arff(results_path+'neg'+str(num)+'.csv',results_path+'neg'+str(num)+'.arff',"last:"+labels,weka_path,java_memory)
            z=str(int(d)/n2*100)#subsample d instances from the neg class (n2 instances) of the training set
            prep.resample(results_path+'neg'+str(num)+'.arff',results_path+'neg_subsample'+str(num)+'.arff',str(seed),str(z),weka_path,java_memory)
            prep.convert_arff_to_csv(results_path+'neg_subsample'+str(num)+'.arff',results_path+'neg_subsample'+str(num)+'.csv',weka_path,java_memory)
            df=pd.read_csv(results_path+'neg_subsample'+str(num)+'.csv')
            train_balanced=pd.concat([train_balanced,df])
            utilities.delete_files([results_path+'neg'+str(num)+'.csv',results_path+'neg'+str(num)+'.arff',results_path+'neg_subsample'+str(num)+'.arff',results_path+'neg_subsample'+str(num)+'.csv'])
    return train_balanced
    
def add_remaining_samples_to_balanced_trainset2(diff_threshold,traindf,train_balanced_unique,train_balanced,results_path,labels,seed,weka_path,java_memory):
    #for a training set of binary class
    #Recursively replace r class0 instances of the balanced training set with a bootstrap of the remaining r class0 instances (r > diff_threshold) 
    num=random.randint(0,2**32-1)
    traindf=traindf.round(4)#round to 4 decimal places
    train_balanced_unique=train_balanced_unique.round(4)#round to 4 decimal places
    df=prep.dataframes_diff(traindf,train_balanced_unique)#get the remaining samples
    (r,_)=df.shape
    #print('difference of traindf - train_balanced_unique: '+str(r))
    if r > int(diff_threshold):#if the training set does not contain r neg instances (r > diff_threshold), add a bootstrap of those r training instances recursively
        df.to_csv(results_path+'df'+str(num)+'.csv',index=False)    
        prep.convert_csv_to_arff(results_path+'df'+str(num)+'.csv',results_path+'df'+str(num)+'.arff',"last:"+labels,weka_path,java_memory)
        prep.resample(results_path+'df'+str(num)+'.arff',results_path+'df_bootstrap'+str(num)+'.arff',seed,'100',weka_path,java_memory)#bootstrap1
        prep.resample(results_path+'df_bootstrap'+str(num)+'.arff',results_path+'df_bootstrap2'+str(num)+'.arff',seed,'100',weka_path,java_memory)#bootstrap 2
        prep.convert_arff_to_csv(results_path+'df_bootstrap2'+str(num)+'.arff',results_path+'df_bootstrap2'+str(num)+'.csv',weka_path,java_memory)
        df_bootstrap=pd.read_csv(results_path+'df_bootstrap2'+str(num)+'.csv',low_memory=False)        
        (_,c)=train_balanced.shape
        cols=list(train_balanced.columns)
        pos=train_balanced[train_balanced[cols[c-1]]==1]    
        neg=train_balanced[train_balanced[cols[c-1]]==0]
        (n,_)=neg.shape
        indx=random.sample([i for i in range(n)],r)#randomly select r indices from the neg instances of train_balanced 
        neg=prep.replace_rows_of_dataframe(indx,neg,df_bootstrap)#replace the rows at r0 indices with the boostrap
        train_balanced=pd.concat([neg,pos])
        prep.remove_duplicates(results_path+'df_bootstrap2'+str(num)+'.csv',results_path+'df_bootstrap2_unique'+str(num)+'.csv')
        df_bootstrap_unique=pd.read_csv(results_path+'df_bootstrap2_unique'+str(num)+'.csv')
        utilities.delete_files([results_path+'df'+str(num)+'.csv',results_path+'df'+str(num)+'.arff',results_path+'df_bootstrap'+str(num)+'.arff',results_path+'df_bootstrap2'+str(num)+'.arff',results_path+'df_bootstrap2'+str(num)+'.csv',results_path+'df_bootstrap2_unique'+str(num)+'.csv'])
        return add_remaining_samples_to_balanced_trainset(diff_threshold,df,df_bootstrap_unique,train_balanced,results_path,labels,str(seed),weka_path,java_memory)
    elif r == diff_threshold or r > 0:
        train_balanced=pd.concat([train_balanced,df,df,df,df,df,df])#add the remaining samples and their 5 duplicates
        (_,c)=train_balanced.shape
        cols=list(train_balanced.columns)
        pos=train_balanced[train_balanced[cols[c-1]]==1]    
        neg=train_balanced[train_balanced[cols[c-1]]==0]
        (n,_)=pos.shape
        (n2,_)=neg.shape
        d=n2-n
        if d>0:#more neg than pos, add a subsample of pos
            pos.to_csv(results_path+'pos'+str(num)+'.csv',index=False)
            prep.convert_csv_to_arff(results_path+'pos'+str(num)+'.csv',results_path+'pos'+str(num)+'.arff',"last:"+labels,weka_path,java_memory)
            z=str(int(d)/n*100)#subsample d instances from the pos class (n instances) of the training set
            prep.resample(results_path+'pos'+str(num)+'.arff',results_path+'pos_subsample'+str(num)+'.arff',str(seed),str(z),weka_path,java_memory)
            prep.convert_arff_to_csv(results_path+'pos_subsample'+str(num)+'.arff',results_path+'pos_subsample'+str(num)+'.csv',weka_path,java_memory)
            df=pd.read_csv(results_path+'pos_subsample'+str(num)+'.csv')
            train_balanced=pd.concat([train_balanced,df])
        utilities.delete_files([results_path+'pos'+str(num)+'.csv',results_path+'pos'+str(num)+'.arff',results_path+'pos_subsample'+str(num)+'.arff',results_path+'pos_subsample'+str(num)+'.csv'])
        return train_balanced
    else:#r==0 
        return train_balanced

def add_remaining_samples_to_balanced_trainset3(diff_threshold,traindf,train_balanced_unique,train_balanced,results_path,labels,seed,weka_path,java_memory):
    #for a training set of binary class
    #Recursively add remaining r class0 instances (r > diff_threshold) to the training set to give an imbalanced training set; then, add more class1 duplicates to balance the imbalanced training set 
    num=random.randint(0,2**32-1)
    traindf=traindf.round(4)#round to 4 decimal places
    train_balanced_unique=train_balanced_unique.round(4)#round to 4 decimal places
    df=prep.dataframes_diff(traindf,train_balanced_unique)#get the remaining samples
    (r,c)=df.shape
    #print('difference of traindf - train_balanced_unique: '+str(r))
    if r > int(diff_threshold):#if the training set does not contain r instances (r > diff_threshold), add a bootstrap of those r training instances recursively
        cols=list(df.columns)
        if df[cols[c-1]].nunique()==1:#instances of remaining samples belong to same class
            train_balanced=pd.concat([train_balanced,df])
            return train_balanced
        else:
            utilities.dataframe_to_arff(df,results_path+'df'+str(num)+'.arff')
            prep.resample(results_path+'df'+str(num)+'.arff',results_path+'df_bootstrap'+str(num)+'.arff',seed,'100',weka_path,java_memory)#bootstrap1
            prep.resample(results_path+'df_bootstrap'+str(num)+'.arff',results_path+'df_bootstrap2'+str(num)+'.arff',seed,'100',weka_path,java_memory)#bootstrap 2
            df_bootstrap=utilities.arff_to_dataframe(results_path+'df_bootstrap2'+str(num)+'.arff')            
            cols=list(df_bootstrap.columns)
            df_bootstrap=df_bootstrap.astype({cols[-1]:int})#convert last column (targets) to int
            ###debug##
            #print('df_bootstrap classes: ',df_bootstrap[cols[-1]].unique())
            ###
            (_,c)=train_balanced.shape
            train_balanced=pd.concat([train_balanced,df_bootstrap])
            ###debug##
            #cols=list(train_balanced.columns)
            #n=train_balanced[cols[c-1]].nunique()
            #print('classes: ',n)
            ###
            df_bootstrap.to_csv(results_path+'df_bootstrap2'+str(num)+'.csv',index=False)
            prep.remove_duplicates(results_path+'df_bootstrap2'+str(num)+'.csv',results_path+'df_bootstrap2_unique'+str(num)+'.csv')
            df_bootstrap_unique=pd.read_csv(results_path+'df_bootstrap2_unique'+str(num)+'.csv')
            del [df_bootstrap]
            utilities.delete_files([results_path+'df'+str(num)+'.csv',results_path+'df'+str(num)+'.arff',results_path+'df_bootstrap'+str(num)+'.arff',results_path+'df_bootstrap2'+str(num)+'.arff',results_path+'df_bootstrap2'+str(num)+'.csv',results_path+'df_bootstrap2_unique'+str(num)+'.csv'])
            return add_remaining_samples_to_balanced_trainset3(diff_threshold,df,df_bootstrap_unique,train_balanced,results_path,labels,str(seed),weka_path,java_memory)
    elif r <= diff_threshold and r > 0:#add duplicates of the r instances to the training set    
        train_balanced=pd.concat([train_balanced,df,df,df,df,df,df,df,df,df,df,df])#add the remaining samples and their 10 duplicates
        (_,c)=train_balanced.shape
        cols=list(train_balanced.columns)
        ###debug##
        #n=train_balanced[cols[c-1]].nunique()
        #print('classes: ',n)
        ###
        if train_balanced[cols[c-1]].nunique() == 2:
            class1=train_balanced[train_balanced[cols[c-1]]==1]    
            class0=train_balanced[train_balanced[cols[c-1]]==0]
            (n1,_)=class1.shape
            (n0,_)=class0.shape
            d=n0-n1
            if d>0:#more class0 than class1, add d class1 instances
                indx=list(class1.index)
                indx2=[]
                for i in range(d):
                    indx2.append(random.choice(indx))#random sampling with replacement                
                train_balanced=pd.concat([train_balanced,class1.loc[indx2]])
            del [class1,class0]
            utilities.delete_files([results_path+'class1'+str(num)+'.arff',results_path+'class1_subsample'+str(num)+'.arff'])
        return train_balanced
    else:#r==0
        return train_balanced

def train_test_rf(option,
                  trees,
                  tree_depth,
                  seed,
                  trainset,
                  original_trainset,
                  testset,
                  results_path,
                  model_and_resultsfiles,
                  weka_path,
                  java_memory,
                  logfile,
                  iteration,
                  cross_validation=None
                  ):
        ####k-fold cross validation, then train a random forest         
        #         or
        #    train a random forest
        #input: option, True (print prediction of instances in logfile) or False (not print prediction of instances in logfile)  
        #       cross_validation, None (default) or no. of fold of CV    
        #       trainset, a training set in arff format
        #       testset, a test set in arff format
        #       results_path, path where model and resultsfiles are saved
        #       model_and_resultsfiles=[model,testresults_file,testresults_file2,trainresults_file,trainresults_file2]
        #       logfile, 'none' or a filename
        #       iteration, training iteration ('none' or a number)
        #output: model
        #        performance of model
        num=random.randint(0,2**32-1)
        model=model_and_resultsfiles[0]
        if cross_validation!=None:#cross validation and train a random forest
            xvalresults_file=model_and_resultsfiles[1]
            trainresults_file=model_and_resultsfiles[2]
            c=cl.Classifier(trainset,testset,weka_path,java_memory)        
            c.random_forest_xval(str(cross_validation),str(trees),str(tree_depth),str(seed),results_path+model,results_path+xvalresults_file)
            xval_auc=post.get_auc(results_path+xvalresults_file)
        else:#training a random forest        
            trainresults_file=model_and_resultsfiles[2]
            c=cl.Classifier(trainset,testset,weka_path,java_memory)        
            c.random_forest_train(str(trees),str(tree_depth),str(seed),results_path+model)
        if original_trainset != 'none':
           c.random_forest_predict(results_path+model,original_trainset,results_path+trainresults_file)
           c.random_forest_predict2(results_path+model,original_trainset,results_path+'rf_trainset_output'+str(num)+'.txt')
        else:
           c.random_forest_predict(results_path+model,trainset,results_path+trainresults_file)
           c.random_forest_predict2(results_path+model,trainset,results_path+'rf_trainset_output'+str(num)+'.txt')
        train_auc=post.get_auc(results_path+trainresults_file)        
        (predL_train,train_tpr,train_tnr,train_fpr,train_fnr)=post.model_output(results_path+'rf_trainset_output'+str(num)+'.txt')        
        utilities.delete_files([results_path+trainresults_file,results_path+'rf_trainset_output'+str(num)+'.txt'])
        if logfile!='none':
            if os.path.isfile(logfile)==False:#logfile does not exist, create a new one
                file=open(logfile,'w+')
            else:
                file=open(logfile,'a')
            if iteration!='none':
                file.write('\t\t\t====Results of Random Forest of Iteration '+str(iteration)+'====\n')
                print('\t\t\t====Results of Random Forest of Iteration '+str(iteration)+'====\n')
            else:
                file.write('\t\t\t====Results of Random Forest====\n')            
                print('\t\t\t====Results of Random Forest====\n')            
            if cross_validation!=None:#write training AUC and k-fold cross validation AUC
                    file.write('\t\t\t\t\tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                    file.write('\t\t\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')        
                    file.write('\t\t\t'+str(cross_validation)+'-fold CV'+'\t\t'+str(np.round(xval_auc,3))+'\n')    
                    file.close() 
                    print('\t\t\t\t\tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')                
                    print('\t\t\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')        
                    print('\t\t\t'+str(cross_validation)+'-fold CV'+'\t\t'+str(np.round(xval_auc,3))+'\n')             
            else:#write training AUC                                        
                    file.write('\t\t\t\t\tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                    file.write('\t\t\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')        
                    file.close()
                    print('\t\t\t\t\tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                    print('\t\t\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')        
        if option:
            file=open(logfile,'a')
            file.write('======Prediction of Training Instances of Random Forest=======\n\n')
            file.close()
            post.write_to_file2(predL_train,train_tpr,train_tnr,train_fpr,train_fnr,logfile)                
        if cross_validation!=None:
            train_xval_auc=train_auc+xval_auc            
            return (c,train_xval_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,xval_auc,-999,-999,-999,-999)     
        else:#training performance
            return (c,train_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,-999,-999,-999,-999,-999)        
    
def train_test_log_reg(option,
                       reg,
                       trainset,
                       original_trainset,
                       testset,
                       results_path,
                       model_and_resultsfiles,
                       weka_path,
                       java_memory,
                       logfile,
                       iteration,
                       cross_validation=None
                       ):
        ####k-fold cross validation and train a logisitc regression
        ####   or
        ####train logistic regression 
        #input: option, True (write predicted probabilities of instances into logfile) or False
        #       reg, l2 regularization 
        #       trainset, a training set in arff format
        #       testset, a test set in arff format
        #       results_path, path where model and resultsfiles are saved
        #       model_and_resultsfiles=[model,testresults_file,testresults_file2,trainresults_file,trainresults_file2]
        #       logfile, 'none' or a filename
        #       iteration, training iteration ('none' or a number)
        #       cross_validation (None or no. of folds)
        #output: model
        #        performance of model
        num=random.randint(0,2**32-1)
        model=model_and_resultsfiles[0]
        if cross_validation!=None:
            xvalresults_file=model_and_resultsfiles[1]
            trainresults_file=model_and_resultsfiles[2]            
            c=cl.Classifier(trainset,testset,weka_path,java_memory)
            c.log_reg_xval(str(cross_validation),str(reg),results_path+model,results_path+xvalresults_file)
            xval_auc=post.get_auc(results_path+xvalresults_file)
        else:
            trainresults_file=model_and_resultsfiles[2]
            c=cl.Classifier(trainset,testset,weka_path,java_memory)
            c.log_reg_train(str(reg),results_path+model)
        if original_trainset != 'none':
           c.log_reg_predict(results_path+model,original_trainset,results_path+trainresults_file)
           c.log_reg_predict2(results_path+model,original_trainset,results_path+'log_reg_trainset_output'+str(num)+'.txt')
        else:
           c.log_reg_predict(results_path+model,trainset,results_path+trainresults_file)
           c.log_reg_predict2(results_path+model,trainset,results_path+'log_reg_trainset_output'+str(num)+'.txt')
        train_auc=post.get_auc(results_path+trainresults_file)
        (predL_train,train_tpr,train_tnr,train_fpr,train_fnr)=post.model_output(results_path+'log_reg_trainset_output'+str(num)+'.txt')                           
        utilities.delete_files([results_path+trainresults_file,results_path+'log_reg_trainset_output'+str(num)+'.txt'])
        if logfile!='none':
            if os.path.isfile(logfile)==False:#logfile does not exist, create a new one
                file=open(logfile,'w+')
            else:
                file=open(logfile,'a')
            if iteration!='none':
                file.write('\t\t\t====Results of Logistic Regression of Iteration '+str(iteration)+'===\n')
                print('\t\t\t====Results of Logistic Regression of Iteration '+str(iteration)+'===\n')
            else:
                file.write('\t\t\t====Results of Logistic Regression====\n')
                print('\t\t\t====Results of Logistic Regression====\n')
            if cross_validation!=None:#write training AUC and k-fold cross validation AUC
                    file.write('\t\t\t\t\tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                    file.write('\t\t\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')        
                    file.write('\t\t\t'+str(cross_validation)+'-fold CV'+'\t\t'+str(np.round(xval_auc,3))+'\n')    
                    file.close() 
                    print('\t\t\t\t\tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')                
                    print('\t\t\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')        
                    print('\t\t\t'+str(cross_validation)+'-fold CV'+'\t\t'+str(np.round(xval_auc,3))+'\n')             
            else:#write training AUC 
                    file.write('\t\t\t\t\tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')
                    file.write('\t\t\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')        
                    file.close()
                    print('\t\t\t\t\tAUC\tTPR(sensitivity) \tTNR(specificity) \tFPR \tFNR\n')                
                    print('\t\t\ttraining'+'\t'+str(np.round(train_auc,3))+'\t'+str(np.round(train_tpr,3))+'\t\t\t'+str(np.round(train_tnr,3))+'\t\t\t'+str(np.round(train_fpr,3))+'\t'+str(np.round(train_fnr,3))+'\n')        
        if option:
            file=open(logfile,'a')
            file.write('======Prediction of Training Instances of Logistic Regression=======\n\n')
            file.close()
            post.write_to_file2(predL_train,train_tpr,train_tnr,train_fpr,train_fnr,logfile)                
        if cross_validation!=None:
            train_xval_auc=train_auc+xval_auc            
            return (c,train_xval_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,xval_auc,-999,-999,-999,-999)
        else:#training performance
            return (c,train_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,-999,-999,-999,-999,-999)            
                    
def split_training_and_testing(iterations2,
                               testset_fraction2=None,
                               dataset=None,
                               data_path=None,
                               results_path=None,
                               preprocessed_ext=None,
                               logfile=None,
                               oversampling_method='oversample_class1_and_class0_separately_using_repeated_bootstraps',
                               compare_with_set_of_all_features=True,):
    #results_path='d:\\EIS preterm prediction\\results\\workflow1\\cl_ffn_V1\\\\'
    #dataset='cl_ffn_V1.csv'
    #dataset='cl_ffn_V1_no_treatment.csv'
    #dataset='438_V1_28inputs.csv'
    #dataset='438_V1_28inputs_no_treatment.csv'
    #dataset="D:\\EIS preterm prediction\\i4i MIS\\raw data\\mis_data_C1C2C3_no_missing_labels.csv"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\workflow1\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\workflow1_2\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\workflow1_3\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\workflow1_4\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\workflow1_5\\\\"
    #results_path='d:\\EIS preterm prediction\\results\\workflow1\\cl_ffn_V1_no_treatment\\\\'
    #results_path='d:\\EIS preterm prediction\\results\\workflow1\\cl_ffn_V1_no_treatment_3\\\\'
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_28inputs_no_treatment_random\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_28inputs\\holdout\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_28inputs_no_treatment\\holdout\\\\"
    #dataset="438_V1_28inputs_no_treatment_random.csv"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_28inputs_unfiltered_random\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_28inputs_xval\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_438_V1_8inputs_xval\\\\"    
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_438_V1_8inputs\\\\"    
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\data_log10_highest_info_gain\\\\"
    #dataset="filtered_data_28inputs.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\data_log10.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\data_log10_highest_info_gain.csv"
    #dataset='438_V1_previous_history_and_demographics2_7features.csv'
    #dataset='438_V1_previous_history_and_demographics2.csv'
    #dataset='438_V1_treatment_history2.csv'
    #dataset='d:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_1input_log_transformed.csv'
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_previous_history_and_demographics2\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_previous_history_and_demographics2_2\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_previous_history_and_demographics2_3\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_previous_history_and_demographics2_4\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_previous_history_and_demographics2_5\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_previous_history_and_demographics2_7features\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_treatment_history2\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_438_V1_1input_log_transformed_5\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_438_V1_1input_log_transformed_6\\\\"
    #logfile=results_path+"logfile.txt"
    weka_path2='c:\\Program Files\\Weka-3-7-10\\weka.jar'
    java_memory2='4g'    
    utilities.create_folder_if_not_exist(results_path)
    dataset=utilities.remove_ids_column_from_dataset(dataset,results_path)
    #trainset_fraction2=0.66                     
    #testset_fraction2=0.34
    #trainset_fraction2=0.75                     
    #testset_fraction2=0.25
    trainset_fraction2=1-testset_fraction2
    #add_noise_option2='add_noise'
    add_noise_option2='no_noise'
    noise_percent=10
    #noise_percent=20
    #noise_percent=30
    #iterations2=100
    #balanced_trainset_size2=-1
    #balanced_trainset_size2=200#oversample preterm class to N (balanced training set size=2N) and undersample onterm class to N where size of preterm class < N < size of onterm class
    #balanced_trainset_size2='oversample class1 to size of class0'
    #balanced_trainset_size2='undersample class0 to size of class1 with replacement' 
    balanced_trainset_size2='undersample class0 to size of class1 without replacement'
    #balanced_trainset_size2=400
    #balanced_trainset_size2=600
    #balanced_trainset_size2=1000
    #balanced_trainset_size2=1200
    #balanced_trainset_size2=2000    
    #balanced_trainset_size2=1984
    degree2=-1
    #degree2=4 #degree of poly features of PTB classifier
    #degree2=3
    #k2=50
    k2=-1#do not select features using information gain
    #k2=6
    #k2=7
    #k2=10
    #k2=16
    #wrapper_es_fs=True #wrapper Evolutionary Search feature selection
    wrapper_es_fs=False #GA RSFS
    if wrapper_es_fs:
        weka_path2='c:\\Program Files\\Weka-3-9-4\\weka.jar' #for Evolutionary Search Feature Selection
        mini_features=-1
        max_features=-1
        number_of_reducts=2#2 optimal feature subsets: 1 for logistic regression and 1 for random forest
        number_of_cpu=4
        populationSize=20 #for Evolutionary Search feature selection (use defaut setting for GA feature selection)
        generations=20 #for Evolutionary Search feature selection (use defaut setting for GA feature selection)
        crossover_prob=0.6
        mutation_prob=0.1 #default mutation prob
        discretize_method='N/A'
        classifiersTypes='continuous classifiers only'
    else:#GA feature selection
        weka_path2='c:\\Program Files\\Weka-3-7-10\\weka.jar' #for logistic regression and random forest
        mini_features=-1
        max_features=-1
        mini_features=10
        #max_features=15
        max_features=16
        number_of_reducts=0 #no GA feature selection performed
        #number_of_reducts=10
        #number_of_reducts=20
        #number_of_reducts=30
        #number_of_reducts=40
        number_of_cpu='N/A'
        #discretize_method="equal_freq"
        discretize_method="pki" #proportional k-interval discretization (number of bins= square root of size of training set)
        #discretize_method="equal_width"
        ##no. of bins determines the size of reducts: the more bins, the smaller the reducts sizes and vice versa.
        ##set no. of bins to find reducts of medium sizes and small sizes
        #bins=50 #no. of bins of equal frequency discretization
        #bins=30 #smaller the no. of bins, the more instances in each bin and vice versa
        populationSize=100 
        generations=50 
        #generations=10 
        crossover_prob=0.6
        mutation_prob=0.033
        #classifiersTypes='discrete and continuous classifiers'
        classifiersTypes='continuous classifiers only'
        #classifiersTypes='discrete classifiers only'        
    #reg2=list(np.logspace(-4,4,10))#10 regularizations between 10^-4 and 10^4
    #reg2.append(1.0)#add regularization 1.0
    #reg2.sort()#sort in ascending order
    reg2=[1e-8] 
    #trees2=[20]
    #trees2='30'              
    #trees2='40'              
    trees2=[50]
    #trees2='100'
    stop_cond_reg2=5 #stop condition for tuning regularization of logistic regression
    stop_cond_trees2=5 #stop condition for tuning trees of random forest 
    workflow1(
              option='split_train_test_sets',
              #option='split_train_valid_test_sets',
              trainset_fraction=trainset_fraction2,                     
              testset_fraction=testset_fraction2,
              #validset_fraction=0.17,
              dataset=dataset,
              iterations=iterations2,
              classifiersTypes2=classifiersTypes,
              add_noise_option=add_noise_option2,
              noise_percent2=noise_percent,
              seed='0',
              results_path2=results_path,
              balanced_trainset_size=balanced_trainset_size2,
              oversampling_method2=oversampling_method,
              degree=degree2,
              k=k2,
              wrapper_es_fs2=wrapper_es_fs,
              no_of_cpu=number_of_cpu,
              populationSize2=populationSize,
              generations2=generations,
              crossover_prob2=crossover_prob,
              mutation_prob2=mutation_prob,
              fitness2='find_reducts',
              number_of_reducts2=number_of_reducts,
              discretize_method2=discretize_method,
              bins2=bins,
              mini_features2=mini_features,
              max_features2=max_features,
              reg=reg2,
              trees=trees2,
              stop_cond_reg=stop_cond_reg2,
              stop_cond_trees=stop_cond_trees2,
              logfile=logfile,
              logfile_option='w',
              weka_path=weka_path2,
              java_memory=java_memory2,
              compare_with_set_of_all_features2=compare_with_set_of_all_features
            )

def training_and_testing(iterations,
                         trainsets=None,
                         testsets=None,
                         data_path=None,
                         results_path=None,
                         preprocessed_ext=None,
                         logfile=None,
                         oversampling_method='oversample_class1_and_class0_separately_using_repeated_bootstraps',
                         compare_with_set_of_all_features=True,
                         predict_ptb_of_each_id_using_all_spectra_of_id=False
                        ):
    utilities.create_folder_if_not_exist(results_path)
    add_noise_option2='no_noise'
    #add_noise_option2='add_noise'
    #noise_percent=10
    #noise_percent=20
    #noise_percent=30
    #balanced_trainset_size2=200#oversample preterm class to N (size of preterm class < N < size of onterm class and balanced training set size=2N) and undersample onterm class to N
    #balanced_trainset_size2='oversample class1 to size of class0'
    #balanced_trainset_size2='undersample class0 to size of class1 with replacement' 
    #balanced_trainset_size2='undersample class0 to size of class1 without replacement'
    #balanced_trainset_size2=400
    #balanced_trainset_size2=600
    #balanced_trainset_size2=1000
    #balanced_trainset_size2=1200
    #balanced_trainset_size2=1984
    #balanced_trainset_size2=2000
    #balanced_trainset_size2=3000    
    #balanced_trainset_size2=4000    
    balanced_trainset_size2=-1
    degree2=-1
    #degree2=2
    #degree2=3
    #degree2=4 #degree of poly features of PTB classifier
    #k2=-1#do not use information gain feature selection
    k2=15
    #k2=26
    #k2=30
    #k2=50
    #k2=60
    #k2=100
    #k2=10
    #k2=2
    #k2=3
    #k2=5
    #k2=6
    #k2=7
    #k2=10
    #k2=21
    #wrapper_es_fs=True #wrapper Evolutionary Search feature selection
    wrapper_es_fs=False #GA RSFS
    if wrapper_es_fs:
        classifiersTypes='continuous classifiers only'
        weka_path2='c:\\Program Files\\Weka-3-9-4\\weka.jar' #for Evolutionary Search Feature Selection
        mini_features=-1
        max_features=-1
        number_of_reducts=2#2 optimal feature subsets: 1 for logistic regression and 1 for random forest
        number_of_cpu=4
        populationSize=20 #for Evolutionary Search feature selection (use defaut setting for GA feature selection)
        generations=20 #for Evolutionary Search feature selection (use defaut setting for GA feature selection)
        crossover_prob=0.6
        mutation_prob=0.1 #default mutation prob
        discretize_method='N/A'
    else:#GA feature selection
        #classifiersTypes='discrete and continuous classifiers'
        classifiersTypes='continuous classifiers only'
        #classifiersTypes='discrete classifiers only'
        weka_path2='c:\\Program Files\\Weka-3-7-10\\weka.jar' #for logistic regression and random forest
        #mini_features=-1
        #max_features=-1
        mini_features=10
        max_features=16
        #max_features=15
        #mini_features=15
        #max_features=25
        number_of_reducts=0 #no GA feature selection performed
        #number_of_reducts=1
        #number_of_reducts=5
        #number_of_reducts=10
        #number_of_reducts=20
        #number_of_reducts=30
        #number_of_reducts=40
        number_of_cpu='N/A'
        #discretize_method="equal_freq"
        discretize_method="pki" #proportional k-interval discretization (number of bins= square root of size of training set)
        #discretize_method="equal_width"
        ##no. of bins determines the size of reducts: the more bins, the smaller the sizes of the reducts and vice versa.
        ##set no. of bins to find reducts of medium sizes and small sizes
        #bins=10
        #bins=20 #smallest bins with dependency degree 1
        #bins=50 #no. of bins of equal frequency discretization
        #bins=30 #smaller the no. of bins, the more instances in each bin and vice versa
        #bins=60
        #bins=70
        #bins=80
        #bins=90
        #bins=100
        populationSize=100 
        generations=50 
        #generations=10
        crossover_prob=0.6
        mutation_prob=0.033
    java_memory2='4g'
    #reg2=list(np.logspace(-4,4,10))#10 regularizations between 10^-4 and 10^4
    #reg2.append(1.0)#add regularization 1.0
    #reg2.sort()#sort in ascending order
    #reg2=[1e-4] 
    #reg2=[1e-8,1e-4,1.0,2e1,1.7e2,1.3e3,4e4]
    reg2=[1e-8] #the best regularization is normally 1e-8
    trees2=['20']
    #trees2='30'              
    #trees2='40'              
    #trees2=['50']
    #trees2='100'
    #trees2=[20,30,50,60,80,100,130,150,180,200]
    #trees2=['50','100','150','200']
    #trees2=[20,30,50,80,100,150,200,250,300]
    #trees2=['10*features']
    #trees2=[20]
    #trees2=[20,30,50,60,80]
    #trees2=[10,20,30]#,40,50,60,70,80,90,100]
    #trees2=[250]
    #trees2=['10']#,'20','30','40','50','60','70','80','90','100'] #good start: trees=10*no. of features
    #trees2=[100]
    #trees2=['250','200','150','100','80','60','40','20','10']
    stop_cond_reg2=5 #stop condition for tuning regularization of logistic regression
    stop_cond_trees2=5 #stop condition for tuning trees of random forest 
    log_regL=[]
    rf_L=[]
    log_reg_train_aucL=[]
    log_reg_test_aucL=[]
    rf_train_aucL=[]
    rf_test_aucL=[]
    f=open(logfile,'w')#delete the contents of the previous logfile
    f.close()
    for iteration in range(iterations):
        if preprocessed_ext!=None:#preprocessed training sets (with preprocessed extensions)
            trainsets2=trainsets+str(iteration)+'.'+preprocessed_ext+str(iteration)+'.csv'#trainset2_0.csv_preprocessed_balanced_trainset_size=1984_degree=4_info_gain_selected_features=30_0.csv
            balanced_trainset_size2=-1
            degree2=-1
            k2=-1
            #transform the test set to the same features as those of the preprocessed training set
            #testsets2=testsets+str(iteration)+'.preprocessed.csv'
            #utilities.construct_poly_features_of_another_dataset('original_features',data_path+testsets+str(iteration)+'.csv',data_path+trainsets2,data_path+testsets2,'none')
        else:#original training sets (.csv)
            trainsets2=data_path+trainsets+str(iteration)+'.csv'
        if testsets=='outliers':
            testsets2="D:\\EIS preterm prediction\\i4i MIS\\raw data\\outliers.csv"
        else:
            testsets2=data_path+testsets+str(iteration)+'.csv'
        (rf_performance,log_reg_performance)=workflow1(
                  iteration_number=iteration,
                  option='train_test_sets',
                  classifiersTypes2=classifiersTypes,
                  train_set_csv=trainsets2,
                  test_set_csv=testsets2,
                  add_noise_option=add_noise_option2,
                  noise_percent2=noise_percent,
                  seed='0',
                  results_path2=results_path,
                  balanced_trainset_size=balanced_trainset_size2,
                  oversampling_method2=oversampling_method,
                  degree=degree2,
                  k=k2,
                  wrapper_es_fs2=wrapper_es_fs,
                  no_of_cpu=number_of_cpu,
                  populationSize2=populationSize,
                  generations2=generations,
                  crossover_prob2=crossover_prob,
                  mutation_prob2=mutation_prob,
                  fitness2='find_reducts',
                  number_of_reducts2=number_of_reducts,
                  discretize_method2=discretize_method,
                  bins2=bins,
                  mini_features2=mini_features,
                  max_features2=max_features,
                  reg=reg2,
                  trees=trees2,
                  stop_cond_reg=stop_cond_reg2,
                  stop_cond_trees=stop_cond_trees2,
                  logfile=logfile,
                  logfile_option='a',
                  weka_path=weka_path2,
                  java_memory=java_memory2,
                  compare_with_set_of_all_features2=compare_with_set_of_all_features,
                  predict_ptb_of_each_id_using_all_spectra_of_id2=predict_ptb_of_each_id_using_all_spectra_of_id
                )
        print('####summary of training and testing performances####')
        file=open(logfile,'a')
        file.write('\n####summary of training and testing performances####\n')
        file.close()
        utilities.summarize_results(logfile,log_reg_performance,rf_performance,log_regL,log_reg_train_aucL,log_reg_test_aucL,rf_L,rf_train_aucL,rf_test_aucL)
        
def cross_validation_training_and_testing(iterations2,
                                          cross_validation2=5,
                                          trainsets=None,
                                          testsets=None,
                                          data_path=None,
                                          results_path=None,
                                          preprocessed_ext=None,
                                          balanced_trainset_size2=-1,
                                          oversampling_method='oversample_class1_and_class0_separately_using_repeated_bootstraps',
                                          degree2=-1,
                                          interaction_only=True,
                                          k2=-1,
                                          wrapper_es_fs=False,
                                          #classifiersTypes='discrete and continuous classifiers',
                                          classifiersTypes='continuous classifiers only',
                                          #classifiersTypes='discrete classifiers only',
                                          mini_features=-1,
                                          max_features=-1,
                                          number_of_reducts=0, #no GA feature selection performed
                                          discretize_method="pki", #proportional k-interval discretization (number of bins= square root of size of training set)
                                          #discretize_method="equal_width",
                                          bins=30, #smaller the no. of bins, the more instances in each bin and vice versa
                                          #bins=60,
                                          #bins=70,
                                          #bins=80,
                                          #bins=90,
                                          #bins=100,
                                          populationSize=100, 
                                          generations=50,
                                          #generations=10
                                          crossover_prob=0.6,
                                          mutation_prob=0.033,
                                          reg2=[1e-8], #the best regularization is normally 1e-8
                                          trees2=[20,30,40],
                                          add_noise_option2='no_noise',
                                          #add_noise_option2='add_noise'
                                          noise_percent=10,
                                          stop_cond_reg2=11, #stop condition for tuning regularization of logistic regression
                                          stop_cond_trees2=11, #stop condition for tuning trees of random forest
                                          logfile=None,
                                          compare_with_set_of_all_features=True,
                                          predict_ptb_of_each_id_using_all_spectra_of_id=False,
                                          final_prob='average of majority probs'
                                          ):
    #loop:
    # 1. split dataset into training set and test set
    # 2. training set -> workflow1 (k-fold cross validation and training classifiers) -> classifiers, training AUC and CV auc
    # 3. test set -> classifiers -> test AUC
    #end loop 
    utilities.create_folder_if_not_exist(results_path)
    if wrapper_es_fs:
        classifiersTypes='continuous classifiers only'
        weka_path2='c:\\Program Files\\Weka-3-9-4\\weka.jar' #for Evolutionary Search Feature Selection
        mini_features=-1
        max_features=-1
        number_of_reducts=2#2 optimal feature subsets: 1 for logistic regression and 1 for random forest
        number_of_cpu=4
        populationSize=20 #for Evolutionary Search feature selection (use defaut setting for GA feature selection)
        generations=20 #for Evolutionary Search feature selection (use defaut setting for GA feature selection)
        crossover_prob=0.6
        mutation_prob=0.1 #default mutation prob
        discretize_method='N/A'
    else:#GA feature selection        
        weka_path2='c:\\Program Files\\Weka-3-7-10\\weka.jar' #for logistic regression and random forest        
        number_of_cpu='N/A'
    java_memory2='4g'
    log_regL=[] #list of log reg performance=(iteration,train_xval_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,xval_auc,-999,-999,-999,-999)
    rf_L=[] #list of random forest performance==(iteration,train_xval_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,xval_auc,-999,-999,-999,-999)
    log_regL2=[] #list of  log reg performance=(i,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)
    rf_L2=[] #list of random forest performance=(i,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr)
    log_reg_train_aucL=[]
    log_reg_xval_aucL=[]
    rf_train_aucL=[]
    rf_xval_aucL=[]
    rf_testset_aucL=[]
    rf_testset_tprL=[]
    rf_testset_tnrL=[]
    rf_testset_fprL=[]
    rf_testset_fnrL=[]
    log_reg_testset_aucL=[]
    log_reg_testset_tprL=[]
    log_reg_testset_tnrL=[]
    log_reg_testset_fprL=[]
    log_reg_testset_fnrL=[]    
    
    f=open(logfile,'w')#delete the contents of the previous logfile
    f.close()
    '''
    ###for good_trainset2###
    iterations2=set([i for i in range(iterations2+10)]) 
    to_delete=set([i+20 for i in range(10)]) #skip good_trainset220 to good_trainset229 which have very small amount of data
    iterations2=iterations2.difference(to_delete)
    iterations2=list(iterations2)
    for iteration in iterations2:
    '''
    for iteration in range(iterations2):          
        if preprocessed_ext!=None:#preprocessed training sets (with preprocessed extensions) e.g. trainset99.csv_preprocessed_balanced_trainset_size=2000_degree=4_info_gain_selected_features=30_99.csv 
            trainsets2=trainsets+str(iteration)+'.'+preprocessed_ext+str(iteration)+'.csv'#trainset2_0.csv_preprocessed_balanced_trainset_size=1984_degree=4_info_gain_selected_features=30_0.csv
            balanced_trainset_size2=-1
            degree2=-1
            k2=-1
        else:#original training sets (.csv)
            trainsets2=trainsets+str(iteration)+'.csv'
        if os.path.isfile(data_path+trainsets2)==False:
            sys.exit('training set does not exist: '+data_path+trainsets2)
        if testsets=='outliers':
            testsets2="D:\\EIS preterm prediction\\i4i MIS\\raw data\\outliers.csv"
        else:
            testsets2=data_path+testsets+str(iteration)+'.csv'
        if os.path.isfile(testsets2)==False:
            sys.exit('test set does not exist: '+testsets2)
        
        (rf_performance,log_reg_performance)=workflow1(
                  iteration_number=iteration,
                  cross_validation=cross_validation2,
                  classifiersTypes2=classifiersTypes,
                  dataset=data_path+trainsets2,
                  add_noise_option=add_noise_option2,
                  noise_percent2=noise_percent,
                  seed='0',
                  results_path2=results_path,
                  balanced_trainset_size=balanced_trainset_size2,
                  oversampling_method2=oversampling_method,
                  degree=degree2,
                  interaction_only2=interaction_only,
                  k=k2,
                  wrapper_es_fs2=wrapper_es_fs,
                  no_of_cpu=number_of_cpu,
                  populationSize2=populationSize,
                  generations2=generations,
                  crossover_prob2=crossover_prob,
                  mutation_prob2=mutation_prob,
                  fitness2='find_reducts',
                  number_of_reducts2=number_of_reducts,
                  discretize_method2=discretize_method,
                  bins2=bins,
                  mini_features2=mini_features,
                  max_features2=max_features,
                  reg=reg2,
                  trees=trees2,
                  stop_cond_reg=stop_cond_reg2,
                  stop_cond_trees=stop_cond_trees2,
                  logfile=logfile,
                  logfile_option='a',
                  weka_path=weka_path2,
                  java_memory=java_memory2,
                  compare_with_set_of_all_features2=compare_with_set_of_all_features,
                  predict_ptb_of_each_id_using_all_spectra_of_id2=predict_ptb_of_each_id_using_all_spectra_of_id
                )
        print('####summary of training and cross validation performances####')
        file=open(logfile,'a')
        file.write('\n####summary of training and cross validation performances####\n')
        file.close()
        utilities.summarize_results(logfile,log_reg_performance,rf_performance,log_regL,log_reg_train_aucL,log_reg_xval_aucL,rf_L,rf_train_aucL,rf_xval_aucL,model1='logistic regression',model2='random forest',cross_validation=cross_validation2)       
        predict_test_set(iteration=iteration,
                     logfile=logfile,
                     log_reg_performance=log_reg_performance,
                     rf_performance=rf_performance,
                     log_regL=log_regL2,
                     rf_L=rf_L2,
                     rf_testset_aucL=rf_testset_aucL,
                     rf_testset_tprL=rf_testset_tprL,
                     rf_testset_tnrL=rf_testset_tnrL,
                     rf_testset_fprL=rf_testset_fprL,
                     rf_testset_fnrL=rf_testset_fnrL,
                     rf_train_aucL=rf_train_aucL,
                     log_reg_train_aucL=log_reg_train_aucL,
                     log_reg_testset_aucL=log_reg_testset_aucL,
                     log_reg_testset_tprL=log_reg_testset_tprL,
                     log_reg_testset_tnrL=log_reg_testset_tnrL,
                     log_reg_testset_fprL=log_reg_testset_fprL,
                     log_reg_testset_fnrL=log_reg_testset_fnrL, 
                     predict_ptb_of_each_id_using_all_spectra_of_id=predict_ptb_of_each_id_using_all_spectra_of_id,
                     testsets=testsets,
                     data_path=data_path,
                     results_path=results_path,
                     weka_path=weka_path2,
                     java_memory=java_memory2,
                     final_prob=final_prob
                     )

def predict_test_set(iteration=None,
                     logfile=None,
                     log_reg_performance=None, #training and CV performance of logistic regression
                     rf_performance=None, #training and CV performance of random forest
                     log_regL=None,
                     rf_L=None,
                     rf_train_aucL=None,
                     log_reg_train_aucL=None,
                     rf_testset_aucL=None,
                     rf_testset_tprL=None,
                     rf_testset_tnrL=None,
                     rf_testset_fprL=None,
                     rf_testset_fnrL=None,
                     log_reg_testset_aucL=None,
                     log_reg_testset_tprL=None,
                     log_reg_testset_tnrL=None,
                     log_reg_testset_fprL=None,
                     log_reg_testset_fnrL=None, 
                     predict_ptb_of_each_id_using_all_spectra_of_id=None,
                     data_path=None,
                     model_path=None,
                     results_path=None,
                     testsets=None,
                     weka_path=None,
                     java_memory=None
                     ):
        ###use ith model to predict ith testset
        utilities.create_folder_if_not_exist(results_path)
        print('#####Predict test set#####')
        file=open(logfile,'a')
        file.write('#####Predict testset.csv#####\n')
        file.close()
        testsets2=data_path+testsets+str(iteration)+'.csv'
        if os.path.isfile(testsets2)==False:
           testsets2=data_path+testsets
           if os.path.isfile(testsets2)==False:
               sys.exit(testsets2+' does not exist')
        MP=mp.ModelsPredict()
        if predict_ptb_of_each_id_using_all_spectra_of_id==True:   
           if classifiersTypes=='discrete classifiers only':
               MP.set_model_number(iteration)
               MP.set_ordinal_encode(True)
               (_,testset_auc,testset_tpr,testset_tnr,testset_fpr,testset_fnr)=MP.predict_using_all_spectra_of_each_patient(
                                                                                                       results_path+'rf'+str(iteration)+'.model',
                                                                                                       'rf',
                                                                                                       results_path+'rf'+str(iteration)+'.model_inputs_output.csv',
                                                                                                       testsets2,
                                                                                                       results_path
                                                                                                       )
               testset_auc2=-999 #dummy results
               testset_tpr2=-999
               testset_tnr2=-999
               testset_fpr2=-999
               testset_fnr2=-999 
           else:
               (_,testset_auc,testset_tpr,testset_tnr,testset_fpr,testset_fnr)=MP.predict_using_all_spectra_of_each_patient(
                       results_path+'rf'+str(iteration)+'.model',
                       'rf',
                       results_path+'rf'+str(iteration)+'.model_inputs_output.csv',
                       testsets2,
                       results_path
                       )
               (_,testset_auc2,testset_tpr2,testset_tnr2,testset_fpr2,testset_fnr2)=MP.predict_using_all_spectra_of_each_patient(
                       results_path+'log_reg'+str(iteration)+'.model',
                       'log_reg',
                       results_path+'log_reg'+str(iteration)+'.model_inputs_output.csv',
                       testsets2,
                       results_path
                       )
        else:
            if classifiersTypes=='discrete classifiers only':#test performance of the random forest with discrete valued inputs
               (testset_auc,testset_tpr,testset_tnr,testset_fpr,testset_fnr)=MP.main(
                        i=iteration,
                        model_software2='weka',
                        modeltype2='random forest',
                        filter2_software2=None,                
                        testset_i2=testsets2,#test set for the ith filter
                        ordinal_encode2=True,
                        model_path2=model_path,
                        results_path2=results_path,
                        logfile2=logfile,
                        logfile2_option='a',
                        weka_path2=weka_path,
                        java_memory2=java_memory
                        )
               testset_auc2=-999 #dummy results for logistic regression
               testset_tpr2=-999
               testset_tnr2=-999
               testset_fpr2=-999
               testset_fnr2=-999
            else:#test performance of the classifiers with continuous valued inputs  
                (testset_auc,testset_tpr,testset_tnr,testset_fpr,testset_fnr),(testset_auc2,testset_tpr2,testset_tnr2,testset_fpr2,testset_fnr2)=MP.main(
                        i=iteration,
                        model_software2='weka',
                        modeltype2='random forest and log regression',
                        filter2_software2=None,                
                        testset_i2=testsets2,
                        ordinal_encode2=True,
                        model_path2=model_path,
                        results_path2=results_path,
                        logfile2=logfile,
                        logfile2_option='a',
                        weka_path2=weka_path,
                        java_memory2=java_memory
                        )
        rf_testset_aucL.append(testset_auc)
        rf_testset_tprL.append(testset_tpr)
        rf_testset_tnrL.append(testset_tnr)
        rf_testset_fprL.append(testset_fpr)
        rf_testset_fnrL.append(testset_fnr)
        log_reg_testset_aucL.append(testset_auc2)
        log_reg_testset_tprL.append(testset_tpr2)
        log_reg_testset_tnrL.append(testset_tnr2)
        log_reg_testset_fprL.append(testset_fpr2)
        log_reg_testset_fnrL.append(testset_fnr2)
        print('####summary of performances####')
        file=open(logfile,'a')
        file.write('\n####summary of performances####\n')
        file.close()
        #log reg performance2=(iteration,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,tset_fpr,test_fnr)
        #                    (0,         1            ,2        ,3        ,4        ,5        ,6        ,7       ,8       ,9       ,10      ,11  )
        if log_reg_performance!=None and rf_performance!=None:#training and testing performance
            rf_performance2=(rf_performance[0],
                             rf_performance[1]+testset_auc,
                             rf_performance[2],
                             rf_performance[3],
                             rf_performance[4],
                             rf_performance[5],
                             rf_performance[6],
                             testset_auc,
                             testset_tpr,
                             testset_tnr,
                             testset_fpr,
                             testset_fnr)
            log_reg_performance2=(log_reg_performance[0],
                                  log_reg_performance[1]+testset_auc2,
                                  log_reg_performance[2],
                                  log_reg_performance[3],
                                  log_reg_performance[4],
                                  log_reg_performance[5],
                                  log_reg_performance[6],
                                  testset_auc2,                              
                                  testset_tpr2,
                                  testset_tnr2,
                                  testset_fpr2,
                                  testset_fnr2)            
        else:#testing performance
            log_reg_performance2=(iteration,
                                  testset_auc2,
                                  -999,#dummy training auc
                                  -999,
                                  -999,
                                  -999,
                                  -999,
                                  testset_auc2,                              
                                  testset_tpr2,
                                  testset_tnr2,
                                  testset_fpr2,
                                  testset_fnr2)             
            rf_performance2=(iteration,
                             testset_auc,
                             -999,#dummy training auc
                             -999,
                             -999,
                             -999,
                             -999,
                             testset_auc,
                             testset_tpr,
                             testset_tnr,
                             testset_fpr,
                             testset_fnr)
        utilities.summarize_results(logfile,log_reg_performance2,rf_performance2,log_regL,log_reg_train_aucL,log_reg_testset_aucL,rf_L,rf_train_aucL,rf_testset_aucL,model1='logistic regression',model2='random forest')       
        
def testing_performance_of_combining_filter_with_predictor(
                                             predictors_path=None,
                                             predictortype=None,
                                             filter2_path=None,
                                             filter2type='rf',
                                             ordinal_encode_of_filter2=False,
                                             testsets_ids_path=None,
                                             good_readings_testset_path=None,
                                             results_path=None,
                                             allreadings_with_ids="438_V1_4_eis_readings_28inputs_with_ids.csv",
                                             logfile=None,
                                             weka_path='c:\\Program Files\\Weka-3-7-10\\weka.jar',
                                             java_memory='4g'
                                             ):
    test_aucL=[]
    utilities.create_folder_if_not_exist(results_path)
    file=open(logfile,'a')
    file.write('####testing performance of combining a filter with a predictor####\n')
    file.write('filters path: '+filter2_path+'\n')
    file.write('filters type: '+filter2type+'\n')
    file.write('predictors path: '+predictors_path+'\n')
    file.write('testsets_ids_path: '+testsets_ids_path+'\n')
    file.write('good_readings_testset_path: '+good_readings_testset_path+'\n')
    file.write('results_path: '+results_path+'\n')
    file.write('logfile: '+logfile+'\n')
    file.close()
    print('####testing performance of combining a filter with a predictor####')
    print('filters path: '+filter2_path)
    print('filters type: '+filter2type)
    print('predictors path: '+predictors_path)
    print('testsets_ids_path: '+testsets_ids_path)
    print('good_readings_testset_path: '+good_readings_testset_path)
    print('results_path: '+results_path)
    print('logfile: '+logfile)
    #l=[i for i in range(100)]
    #l2=[i for i in range(8)]
    #l3=set(l)-set(l2)
    #l3=list(l3)
    #for i in l3:
    for i in range(100):
        print('iteration: ',i)
        file=open(logfile,'a')
        file.write('iteration: '+str(i)+'\n')
        filter2=filter2_path+filter2type+str(i)+'.model'
        predictor=predictors_path+predictortype+str(i)+'.model'
        predL=select_best_eis_then_predict_ptb(i,
                                               predictors_path,
                                               filter2_path,
                                               filter2,
                                               predictor,
                                               predictortype,
                                               allreadings_with_ids,
                                               testsets_ids_path,
                                               good_readings_testset_path,
                                               results_path,
                                               weka_path,
                                               java_memory,
                                               filter2type=filter2type,
                                               ordinal_encode_of_filter2=ordinal_encode_of_filter2,
                                               logfile=logfile)            
        print('#####EIS Readings Filter: '+filter2type+' and PTB Predictor: '+predictortype+'#####')
        file.write('#####EIS Readings Filter: '+filter2type+' and PTB Predictor: '+predictortype+'#####\n')
        if predL=='no filter':
            print('no filter')
            file.write('no filter\n')
            file.close()
        elif predL=='no predictor':
            print('no predictor')
            file.write('no predictor\n')
            file.close()
        else:
            probL=[]
            targetL=[]
            for m in range(len(predL)):
                    pred=predL[m]
                    prob=pred[1]
                    probL.append(prob)
                    target=pred[3]
                    targetL.append(float(target))
            test_auc=roc_auc_score(targetL,probL)
            test_aucL.append(test_auc)
            print('test auc: ',np.round(test_auc,3))
            file.write('test auc: '+str(np.round(test_auc,3))+'\n')
            #summarize results after predicting each test set
            mean_auc=np.mean(test_aucL)
            max_auc=np.max(test_aucL)
            min_auc=np.min(test_aucL)
            print('mean test auc: '+str(np.round(mean_auc,3)))
            print('max test auc: '+str(np.round(max_auc,3)))
            print('min test auc: '+str(np.round(min_auc,3)))
            file.write('mean test auc: '+str(np.round(mean_auc,3))+'\n')
            file.write('max test auc: '+str(np.round(max_auc,3))+'\n')
            file.write('min test auc: '+str(np.round(min_auc,3))+'\n')
            file.close()
        
if __name__ == "__main__":   
    '''
    ###Split filtered_eis_readings_with_ids and all_eis_readings_with_ids into training sets, validation sets (if validset_size > 0) and testsets by ids
    #output: trainset1_i.csv (ith training set of filtered_eis_readings)
    #        trainset2_i.csv
    #        testset1_i.csv
    #        testset2_i.csv
    #        validset1_i.csv (if validset_size > 0)
    #        validset2_i.csv (if validset_size > 0)
    #        trainset1_ids_i.csv
    #        testset1_ids_i.csv
    #        validset1_ids_i.csv
    #        trainset2_ids_i.csv
    #        testset2_ids_i.csv
    #        validset2_ids_i.csv
    #        seeds.txt (random seeds of data splits)
    #        trainsets1_indx.txt
    #        trainsets2_indx.txt    
    trainset_size=0.66 #each training set 52.8%, each validation set: 13.2%, the test set: 34%
    testset_size=1-float(trainset_size)
    results_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\"
    utilities.split_train_valid_testsets_by_ids(
                               trainsets1_and_trainsets2=True,
                               trainsets2_only=False,
                               trainsets1_only=False,
                               filtered_eis_readings_with_ids='filtered_data_28inputs_with_ids.csv',
                               selected_unselected_eis_readings_with_ids='selected_unselected_eis_readings_with_ids.csv',
                               results_path=results_path,
                               iterations=100,
                               trainset_size=trainset_size,
                               testset_size=testset_size)
    '''
    '''
    ###split dataset into K training sets and K test sets (e.g. K=100)
    #output: trainseti.csv (i=0,...,iterations)
    #        testseti.csv
    #dataset='filtered_data_28inputs.csv'
    #dataset='438_V1_demographics.csv'
    #dataset='438_V1_treatment_history2.csv'
    #dataset='438_V1_no_preterm_birth.csv'
    #dataset='438_V1_treatment_history2_and_demographics.csv'
    #dataset='438_V1_previous_history_and_demographics2.csv'
    #dataset='438_V1_demographics_obstetric_history_2_parous_features.csv'
    #dataset='438_V1_demographics_obstetric_history.csv'
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs_log_transformed.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_1input_log_transformed.csv"
    #dataset="D:\\EIS preterm prediction\\438_V1_28inputs_and_438_V1_demographics_obstetric_history_2_parous_features_with_ids.csv"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_28inputs_and_438_V1_demographics_obstetric_history_2_parous_features\\"
    dataset="D:\\EIS preterm prediction\\i4i MIS\\raw data\\divide by air reference\\mean_of_amp_phase_of_mis_data_C1C2C3_divide_by_air_no_missing_labels.csv"
    results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\mean_of_amp_phase_of_mis_data_C1C2C3_divide_by_air_no_missing_labels\\'
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_438_V1_1input_log_transformed_7\\"
    #df=pd.read_csv(dataset)
    #df=df.drop(columns='abxCell')#delete Abx feature as it has too many missing values (57%)
    #df.to_csv('438_V1_demographics_obstetric_history3.csv',index=False)
    #dataset='438_V1_demographics_obstetric_history3.csv'
    #dataset='cl_ffn_V1.csv'
    #dataset='cl_ffn_V1_no_treatment.csv'
    #dataset='438_V1_demographics_obstetric_history2.csv' 
    #dataset="D:\\EIS preterm prediction\\EIS for cervical cancer diagnosis\\ColePY.csv"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\asymp_22wks_438_V1_8inputs_log_transformed_xval5\\"
    #results_path="D:\\EIS preterm prediction\\results\\cervical_cancer_eis\\log_reg_and_random_forest\\balanced 2000\\"
    #testset_size=0.15
    testset_size=0.2
    #testset_size=0.34    
    iterations=100
    utilities.split_train_testsets(dataset,testset_size,iterations,results_path)
    '''
    '''
    ###split MIS data into training and testing sets by ids###
    from mis_data import split_train_testsets_by_ids
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\divide by air reference\\mis_data_C1C2C3_divide_by_air_no_missing_labels.csv"
    testset_size=0.2
    iterations=100
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\'
    data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\no compensation\\mis_data_c1c2c3_no_compensation_visit1_visit2_no_missing_labels.csv"
    results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (no compensation)\\'    
    split_train_testsets_by_ids(data,testset_size,iterations,results_path)
    '''
    '''
    ###generate preprocessed training sets and original training sets for experiments (this avoid repeat the preprocessing steps on the same training sets)
    #data_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\cervical_cancer_eis\\\\"
    data_path='D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\visit1_symp\\\\'
    for i in range(100):
        workflow1(                                                       
                         option='preprocess',
                         #train_set_csv="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\trainset2_"+str(i)+".csv",
                         #train_set_csv="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\trainset1_"+str(i)+".csv",
                         #train_set_csv="D:\\EIS preterm prediction\\results\\cervical_cancer_eis\\trainset"+str(i)+".csv",
                         train_set_csv=data_path+'trainset'+str(i)+'.csv',
                         results_path2=data_path,
                         iterations=100,
                         iteration_number=i,
                         #balanced_trainset_size=1984,
                         balanced_trainset_size=2000,
                         degree=4,
                         k=30,
                         #logfile=data_path+'logfile_preprocess_trainset2_i.txt',
                         #logfile=data_path+'logfile_preprocess_trainset1_i.txt',
                         logfile=data_path+'logfile_preprocess_trainset.txt',
                         logfile_option='a'
                 )
    '''
    #split_training_and_testing()
    ###cross validate, then, train filters on original training sets (training sets: trainset2_i.csv, test sets: testset2_i.csv)#####
    #data_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\discrete filters\\filters9\\\\"
    #cross_validation_training_and_testing(100,
    #                                      cross_validation2=5,
    #                                      trainsets='trainset2_',
    #                                      testsets='testset2_',
    #                                      data_path=data_path,
    #                                      results_path=results_path,
    #                                      compare_with_set_of_all_features=True)
    #
    #results_path="D:\\EIS preterm prediction\\results\\cervical_cancer_eis\\log_reg_and_random_forest\\undersample class0 to class1\\"
    #results_path="D:\\EIS preterm prediction\\results\\cervical_cancer_eis\\log_reg_and_random_forest\\balanced 2000\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal data\\"
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal data\\good trainsets\\'
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal data\\good trainsets2\\'
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal data2\\"
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\good trainsets2\\'
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\good trainsets2_2\\'
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\"
    #data_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_28inputs_and_438_V1_demographics_obstetric_history_2_parous_features\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\\\"
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\mean_of_amp_phase_of_mis_data_C1C2C3_divide_by_air_no_missing_labels\\\\'
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\good trainsets2_3\\'
    #results_path=data_path
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\mean_of_amp_phase_of_mis_data_C1C2C3_divide_by_air_no_missing_labels2\\\\'
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each spectrum\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_balanced_training_set\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_balanced_training_set_2\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_balanced_training_set_3\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_good_trainsets\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_good_trainsets_2\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_good_trainsets_3\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_good_trainsets2\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_good_trainsets2_2\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_good_trainsets2_3\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_good_trainsets2_4\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal_data_good_trainsets2_5\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\outliers\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3_GA_feature_selection\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3_info_gain_feature_selection\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\trainset_normal_data_plus_outliers\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_balanced_training_set\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_balanced_training_set2\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_balanced_training_set3\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_oversample_class1_to_class0\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_oversample_class0_and_class1_separately\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_oversample_class0_and_class1_separately2\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_oversample_class0_and_class1_separately3\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_oversample_class0_and_class1_separately4\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_balanced_training_set_GA_feature_selection_5foldCV\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_GA_feature_selection_5foldCV\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_info_gain_feature_selection_5foldCV\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data_plus_outliers_balanced_training_set_5foldCV\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\testset_normal_data2_plus_outliers2\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\balanced train size 2000\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\balanced train size 2000_poly_degree4_info_gain30_ga\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\balanced train size -1\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\balanced train size 2000_poly_degree4_info_gain100_ga\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\balanced train size 2000_poly_degree4_info_gain100\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\normal data\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\normal data\\balanced train size 5000\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\normal data\\balanced train size 2000_poly_degree4_info_gain30_ga\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\normal data\\balanced train size 2000_ga\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\good_trainset\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\normal data\\good_trainset\\balanced train size 5000\\\\"
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\good_trainset\\balanced train size 2000_poly_degree4_info_gain30_ga\\\\"

    '''
    training_and_testing(100,
                         trainsets='trainset',
                         #trainsets='trainset_normal_data_plus_outliers',
                         testsets='testset',
                         #testsets='outliers',
                         #testsets='testset_normal_data_plus_outliers',
                         #testsets='testset_normal_data2_plus_outliers2',
                         data_path=data_path,
                         results_path=results_path,
                         logfile=results_path+'logfile.txt',
                         compare_with_set_of_all_features=True,
                         predict_ptb_of_each_id_using_all_spectra_of_id=True,
                         w1=0.5, #weight of ptb prediction of an id based on c1, c2 and c3 spectra of the id at visit 1. The w1 represents how much the user trusts the predicted prob of PTB based on c1, c2 and c3 of visit 1
                         #w2=0.8 #weight of ptb prediction of an id based on c1, c2 and c3 spectra of the id at visit 2. The w2 represents how much the user trusts the predicted prob of PTB based on c1, c2 and c3 of visit 2
                         w2=0.5
                        )
    '''
    '''
    #normalize all training sets and test sets before cross_validation_training_and_testing
    data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(divide_by_air_reference)\\\\"
    #normalized_data_path=data_path+'mini_max_normalized data\\\\'
    normalized_data_path=data_path+'mini_max_normalized data2\\\\'
    utilities.normalize_training_and_test_sets(data_path,
                                     normalized_data_path,
                                     iterations=100,
                                     train_set='trainset',
                                     test_set='testset',
                                     normalize_method='minmax',
                                     mini=0,
                                     maxi=10)    
    '''    
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(divide_by_air_reference)\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(divide_by_air_reference)\\mis_data_C1C2C3_divide_by_air_no_missing_labels_x100\\\\"
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\mis_data_C1C2C3_divide_by_air_no_missing_labels_x100\\\\'
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(divide_by_air_reference)\\mis_data_C1C2C3_divide_by_air_no_missing_labels_x100\\results4\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(divide_by_air_reference)\\normal data2\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(divide_by_air_reference)\\visit1_symp\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(divide_by_air_reference)\\visit2_symp\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\visit1_symp\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\visit2_symp\\\\"
    data_path='D:\\EIS preterm prediction\\results\\Di\\ahr_v1_symp_no_compensation\\'
    #data_path='D:\\EIS preterm prediction\\results\\Di\\ahr_v2_symp_no_compensation\\'
    #data_path='D:\\EIS preterm prediction\\results\\Di\\ahr_v1_v2_symp_no_compensation\\'
    #results_path=normalized_data_path
    #data_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_1input_log_transformed_3\\"
    #results_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_1input_log_transformed_7\\"
    #results_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_1input_log_transformed_9\\"

    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\normal data\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\visit1_symp\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\visit2_symp\\\\"
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3 (441 spectra)\\normal data\\\\'
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3 (441 spectra)\\\\'
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\normal data2\\\\"
    results_path=data_path
    #results_path=data_path+'balanced_trainset_size2000_degree4_info_gain_selected_features30_ga\\\\'
    #results_path=data_path+'balanced_trainset_size2000_degree4_info_gain_selected_features30_ga2\\\\'
    #results_path=data_path+'balanced_trainset_size6000\\\\'
    #results_path=data_path+'balanced_trainset_size5000\\\\'
    #results_path=data_path+'balanced_trainset_size2000\\\\'
    #results_path=data_path+'balanced_trainset_size2000_ga_subsets_of_10_to_15_features\\\\'
    #results_path=data_path+'balanced_trainset_size2000_discretized\\\\'
    ###parameters of workflow1
    #balanced_trainset_size2=-1
    #balanced_trainset_size2='undersample class0 to size of class1 with replacement' 
    #balanced_trainset_size2='undersample class0 to size of class1 without replacement'
    #balanced_trainset_size2=200#combination of oversampling and undersampling: 
                               #If set balanced training set size to 2N where size of preterm class < N < size of onterm class (majority class), 
                               #oversample preterm class to N  
                               #and undersample onterm class to N e.g. N=100 (balanced training set size=2N).
    #balanced_trainset_size2='oversample class1 to size of class0'
    #balanced_trainset_size2='undersample class0 to size of class1 with replacement' 
    #balanced_trainset_size2='undersample class0 to size of class1 without replacement'
    #balanced_trainset_size2=400
    #balanced_trainset_size2=600
    #balanced_trainset_size2=1000
    #balanced_trainset_size2=1200
    #balanced_trainset_size2=1984
    balanced_trainset_size2=2000
    #balanced_trainset_size2=3000    
    #balanced_trainset_size2=4000
    #balanced_trainset_size2=5000
    #balanced_trainset_size2=6000
    #balanced_trainset_size2=7000
    oversampling_method='oversample_class1_and_class0_separately_using_repeated_bootstraps'
    #oversampling_method='smote'
    #oversampling_method='random sampling with replacement'
    #oversampling_method='borderline smote'
    #oversampling_method='adasyn'
    #degree2=-1
    degree2=2
    #degree2=3
    #degree2=4 #degree of poly features of PTB classifier
    #degree2=5
    #k2=-1#do not use information gain feature selection
    #k2=int(32*3/4)
    k2=30
    #k2=50
    #k2=60
    #k2=100
    #k2=10
    #k2=2
    #k2=3
    #k2=5
    #k2=6
    #k2=7
    #k2=10
    #k2=21
    #wrapper_es_fs=True #wrapper Evolutionary Search feature selection
    wrapper_es_fs=False #GA RSFS
    mini_features=2
    #max_features=2        
    #mini_features=-1
    #max_features=-1        
    #mini_features=10
    #max_features=15
    #max_features=16
    #mini_features=15
    max_features=25
    #number_of_reducts=0 #no GA feature selection performed
    number_of_reducts=5
    #number_of_reducts=10
    #number_of_reducts=40
    #discretize_method="equal_freq"
    #discretize_method="pki" #proportional k-interval discretization (number of bins= square root of size of training set)
    discretize_method="equal_width"
    ##no. of bins determines the size of reducts: the more bins, the smaller the sizes of the reducts and vice versa.
    ##set no. of bins to find reducts of medium sizes and small sizes
    #bins=10
    #bins=20 #smallest bins with dependency degree 1
    #bins=50 #no. of bins of equal frequency discretization
    bins=30 #smaller the no. of bins, the more instances in each bin and vice versa
    #bins=40
    #bins=60
    #bins=70
    #bins=80
    #bins=90
    #bins=100
    populationSize=100 
    #generations=50 
    generations=10
    crossover_prob=0.6
    mutation_prob=0.033
    #reg=list(np.logspace(-4,4,10))#10 regularizations between 10^-4 and 10^4
    #reg.append(1.0)#add regularization 1.0
    #reg.sort()#sort in ascending order
    #reg=[1e-4] 
    #reg=[1e-8,1e-9,1e-4,1e-2,1.0,2e1,1.7e2,1.3e3]
    reg=[1e-8] #the best regularization is normally 1e-8
    #reg=[1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    #trees=[20,30,50,60,80,100,130,150,180,200]
    #trees=['50','100','150','200']
    #trees=[20,30,50,80,100,150,200,250,300,350,400]
    #trees=[20,30,40,50,80,100,150,200,250,300]
    trees=['10*features'] #a good number of trees
    #trees=[10,15,20]
    #trees=[20,30,40]#,60,80,100] 
    #trees=[20,30,40,50,60,70,80,90,100]
    #trees=[200,250,300,150,100,50,20]
    #trees=['10']#,'20','30','40','50','60','70','80','90','100'] #good start: trees=10*no. of features
    #trees=[100]
    #trees=[200,250,300,350]
    #trees=['250','200','150','100','80','60','40','20','10']
    stop_cond_reg=11 #stop condition for tuning regularization of logistic regression
    stop_cond_trees=11 #stop condition for tuning trees of random forest
    #stop_cond_reg=5 #stop condition for tuning regularization of logistic regression
    #stop_cond_trees=5 #stop condition for tuning trees of random forest
    #stop_cond_reg=6 #stop condition for tuning regularization of logistic regression
    #stop_cond_trees=6 #stop condition for tuning trees of random forest
    compare_with_set_of_all_features2=True
    predict_ptb_of_each_id_using_all_spectra_of_id2=True
    classifiersTypes2='continuous classifiers only'
    #classifiersTypes2='discrete classifiers only'
    #classifiersTypes2='discrete and continuous classifiers'
    cross_validation_training_and_testing(100,
                                          cross_validation2=5,
                                          trainsets='trainset',
                                          testsets='testset',
                                          #preprocessed_ext='csv_preprocessed_balanced_trainset_size=2000_degree=4_info_gain_selected_features=30_',
                                          #testsets='outliers',
                                          #testsets='testset_of_normal_data_and_outliers',
                                          #testsets='testset_normal_data_plus_outliers',
                                          #trainsets='good_trainset',
                                          #testsets='good_testset',
                                          #trainsets='good_trainset2',
                                          #testsets='good_testset2',
                                          #testsets='good_testset2_with_outliers_removed_by_pca_trained_on_good_trainset2',
                                          #data_path=data_path,
                                          data_path=data_path,
                                          results_path=results_path,
                                          logfile=results_path+'logfile.txt',
                                          #logfile=results_path+'logfile2.txt',
                                          #logfile=results_path+'logfile3.txt',                                          
                                          #logfile=results_path+'logfile4.txt',                                          
                                          #logfile=results_path+'logfile_good_testset2_with_outliers_removed_by_pca_trained_on_good_trainset2.txt',
                                          #logfile=results_path+'logfile_outliers.txt',
                                          compare_with_set_of_all_features=compare_with_set_of_all_features2,
                                          predict_ptb_of_each_id_using_all_spectra_of_id=predict_ptb_of_each_id_using_all_spectra_of_id2,
                                          #add_noise_option2='no_noise',  
                                          add_noise_option2='add_noise',
                                          noise_percent=10,
                                          balanced_trainset_size2=balanced_trainset_size2,
                                          oversampling_method= oversampling_method,
                                          degree2=degree2,
                                          k2=k2,
                                          wrapper_es_fs=wrapper_es_fs,
                                          classifiersTypes=classifiersTypes2,
                                          mini_features=mini_features,
                                          max_features=max_features,
                                          number_of_reducts=number_of_reducts,
                                          discretize_method=discretize_method,
                                          bins=bins, 
                                          populationSize=populationSize, 
                                          generations=generations,
                                          crossover_prob=crossover_prob,
                                          mutation_prob=mutation_prob,
                                          reg2=reg, #the best regularization is normally 1e-8
                                          trees2=trees,
                                          stop_cond_reg2=stop_cond_reg, #stop condition for tuning regularization of logistic regression
                                          stop_cond_trees2=stop_cond_trees, #stop condition for tuning trees of random forest
                                          )  
    '''
    ###determine optimal feature subsets and models parameters using a test set, then output the optimal classifiers
    data_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\\\"
    results_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\predictors6\\\\"
    training_and_testing(100,
                         trainsets='trainset1_',
                         preprocessed_ext='csv_preprocessed_balanced_trainset_size=1984_degree=4_info_gain_selected_features=30_',
                         testsets='testset1_',
                         data_path=data_path,
                         results_path=results_path,
                         logfile2=results_path+"logfile.txt"
                        )
    '''
    '''
    ###testing performance of combining filter with predictor
    #filter2_path='h:\\data\\EIS preterm prediction\\results\\workflow1\\filter2 from sharc\\selected_unselected_eis_readings\\'
    filter2_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\discrete filters\\filters9\\"
    filter2type='rf'
    predictors_path="h:\\data\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\"#model2s (EIS-based models)
    predictortype='rf'
    #testsets_ids_path="d:\\EIS preterm prediction\\trainsets1trainsets2\\filtered_data\\trainsets66_percent\\"
    #testsets_ids_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\"
    #testsets_ids_path=predictors_path
    testsets_ids_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\"
    #good_readings_testset_path="H:\\data\\EIS preterm prediction\\results\\workflow1\\validate filters\\15dec_filtered_data_28inputs\\"
    good_readings_testset_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\good_readings_of_testsets\\"
    #results_path="h:\\data\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\testing_performance_of_combining_filter_with_predictor\\"   
    results_path="D:\\EIS preterm prediction\\results\\workflow1\\filtered_data_3\\testing_performance_of_combining_filter_with_predictor\\"   
    logfile=results_path+'logfile.txt'    
    testing_performance_of_combining_filter_with_predictor(
                                             predictors_path=predictors_path,
                                             predictortype=predictortype,
                                             filter2_path=filter2_path,
                                             testsets_ids_path=testsets_ids_path,
                                             good_readings_testset_path=good_readings_testset_path,
                                             results_path=results_path,
                                             filter2type=filter2type,
                                             #ordinal_encode_of_filter2=False,
                                             ordinal_encode_of_filter2=True,
                                             logfile=logfile,
                                             weka_path='c:\\Program Files\\Weka-3-7-10\\weka.jar',
                                             java_memory='4g'
                                             )
'''
'''
###reduce a weka training data and a weka test data using python dataframe and save the reduced data as weka files
    traindf=utilities.arff_to_dataframe(weka_train_file)
    (_,c)=traindf.shape
    reduct=reduct.split(',')
    for i in range(len(reduct)):
        reduct[i]=int(reduct[i])-1 #change indices of features to start from 0 rather than 1
    reduct.append(c-1)#add class variable index
    print('reduct: '+str(j))
    print(reduct)
    traindf=traindf.iloc[:,reduct]    
    if option=='continuous_data':
        cols=list(traindf.columns)
        y=traindf[cols[-1]]
        y=y.replace(to_replace=r'^\"?\'?(\d+)\'?\"?$', value=r'\1', regex=True)
        traindf[cols[-1]]=y
        traindf=traindf.astype(float)
        traindf=traindf.astype({cols[-1]:int})
    utilities.dataframe_to_arff(traindf,reduced_weka_train_file)
    testdf=utilities.arff_to_dataframe(weka_test_file)
    (_,c)=testdf.shape
    testdf=testdf.iloc[:,reduct]
    if option=='continuous_data':
        cols=list(testdf.columns)
        y=testdf[cols[-1]]
        y=y.replace(to_replace=r'^\"?\'?(\d+)\'?\"?$', value=r'\1', regex=True)
        testdf[cols[-1]]=y
        testdf=testdf.astype(float)
        testdf=testdf.astype({cols[-1]:int})
    utilities.dataframe_to_arff(testdf,reduced_weka_test_file)
'''
'''
###reduce a weka data file using python dataframe###
        traindf=utilities.arff_to_dataframe(weka_original_train_file)
        (_,c)=traindf.shape
        traindf=traindf.iloc[:,reduct]
        if option=='continuous_data':
            cols=list(traindf.columns)
            y=traindf[cols[-1]]
            y=y.replace(to_replace=r'^\"?\'?(\d+)\'?\"?$', value=r'\1', regex=True)
            traindf[cols[-1]]=y
            traindf=traindf.astype(float)
            traindf=traindf.astype({cols[-1]:int})
        utilities.dataframe_to_arff(traindf,reduced_weka_original_train_file)
'''

'''               
def nth_test_set(dataset,dataset2,m,n,data_format,results_path):
    #get the nth test set in m iterations e.g. m=100
    #input: dataset1, a csv dataset
    #       dataset2, a csv dataset ('none' if not given)
    #       m, no. of iterations
    #       n, nth test set
    #       data_format, csv or arff
    data=pd.read_csv(dataset)
    (_,c)=data.shape
    cols=list(data.columns)
    data=prep.convert_targets2(data,c-1)
    data.before37weeksCell=data.before37weeksCell.astype(int)#replace any 1.0 with 1 and 0.0 with 0 in targets
    if dataset2!='none':
        data2=pd.read_csv(dataset2)
        (_,c2)=data2.shape
        data2=prep.convert_targets2(data2,c2-1)
        data2.before37weeksCell=data2.before37weeksCell.astype(int)#replace any 1.0 with 1 and 0.0 with 0 in targets
        if data2.columns[0] not in data.columns:#dataset2 had different features to dataset1 e.g. x2^2 x3 (dataset2) and 'x2^2 x3' (dataset1) 
            print('dataset1 and dataset2 have different features. The features of dataset2 is replaced with those of dataset1.')
            data2.columns=data.columns
    test_size=0.34 #proportion of test set
    for i in range(m):
        random_state=random.randint(0,2**32-1)
        (train_set,test_set)=prep.split_train_test_sets(data,test_size,random_state,cols[c-1])#split whole dataset consisting of 28 features (14 amplitudes and 14 phases) into training and test sets
        if dataset2!='none':#split dataset2 into training and test sets; then, merge the training set with that of dataset1 and merge the test set with that of dataset1
            (train_set2,test_set2)=prep.split_train_test_sets(data2,test_size,random_state,cols[c-1])
            dfs=[train_set,train_set2]
            train_set=pd.concat(dfs)#merge training sets of dataset1 and dataset2
            dfs=[test_set,test_set2]
            test_set=pd.concat(dfs)#merge test sets of dataset1 and dataset2
        if i==n:
            if data_format=='csv':
                outfile=results_path+'testset'+str(i)+'.csv'
                test_set.to_csv(outfile,index=False)#testset.csv consists of the 28 features 
                print(str(n)+'th test_set is saved to '+outfile)
                break
            elif data_format=='arff':
                outfile=results_path+'testset'+str(i)+'.arff'
                test_set.to_csv(results_path+'testset.csv',index=False)#testset.csv consists of the 28 features             
                prep.convert_csv_to_arff(results_path+'testset.csv',outfile,"last:0,1") 
                print(str(n)+'th test_set is saved to '+outfile)
                break       
            else:
                import sys
                sys.exit('invalid format: '+data_format)                    
'''



