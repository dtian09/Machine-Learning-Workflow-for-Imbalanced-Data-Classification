#!/usr/bin/python

"""
===================================================================
Pipeline: oversampling, feature selection and supervised learning
===================================================================
"""


#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#from sklearn import datasets
#from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, ARDRegression
#from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
import sys
#sys.path.append('D:\\EIS preterm prediction')
from preprocess import convert_to_python_path, cross_validation_split_and_remove_duplicates_from_valid_set, split_train_test_sets, fill_missing_values
#from sklearn.metrics import classification_report
#from imblearn.over_sampling import RandomOverSampler      
from imblearn.pipeline import Pipeline
#from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern, DotProduct, Exponentiation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, MyPolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
from joblib import dump
from joblib import Parallel, delayed
import random
from utilities import delete_files, get_model_inputs_output, reduce_data, create_folder_if_not_exist, predict_testset, predict_trainset_and_testset_using_sklearn_and_optimal_threshold, summarize_accuracy_results
import numpy as np
from workflows import workflow1, get_random_feature_subsets
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
import os
import operator
import warnings

def grid_search_cv(model,
                   modelname,
                   traindata,
                   cv,
                   param_grid,
                   randomsearchCV=False,
                   n_iter=200, #number of parameter settings to search during random search CV
                   select_EIS_features=-1,#select EIS features from all the features and train model on EIS features
                   select_demo_features=-1,#select demographics, treatment and obstetric history features from all the features and train models on the selected features
                   polynomial_degree_within_cv=-1, #degree of polynomial features to be constructed
                   transform_to_polynomial_features_of_a_dataset=False, #select the same polynomial features as the ones of a dataset
                   dataset_with_polynomial_features_csv=None, #a dataset (.csv file) consisting of polynomial features output by rough set-based GA feature selection 
                   select_original_features_of_a_dataset=False, #select the same original features as the ones of a dataset
                   dataset_with_original_features_csv=None, #a dataset (.csv file) consisting of original features output by a feature selection method
                   interaction_only_within_cv=False, #interaction features only e.g. include polynomial features such as ab, abc and exclude a^2, b^2, etc
                   oversampling_method_within_cv=-1,
                   balanced_trainset_size_within_cv=-1,
                   fs_select_from_model_within_cv=False,
                   k_within_cv=-1,
                   transform_inputs_within_cv='zscore',
                   mini=0,
                   maxi=1,
                   training_score='roc_auc'):
    #determine optimal parameters of classification algorithms using cross validation of pipeline: training set -> oversampling -> feature selection -> normalization -> classifier training
    #note: the training set can be balanced via oversampling
    steps=[]
    if select_EIS_features != -1:
        from sklearn.compose import ColumnTransformer, make_column_selector
        cols=list(traindata.columns)
        selectEISfeatures = ColumnTransformer(transformers=[("selectEISfeatures", 'passthrough', select_EIS_features),
                                                            ('selectclassvariable','passthrough',make_column_selector(pattern=cols[-1]))],
                                              remainder="drop")
        (_,c)=traindata.shape
        X1=traindata.iloc[:,:c-1]
        y1=traindata.iloc[:,c-1]
        selectEISfeatures.fit(X1,y1)
        print(select_EIS_features)
        steps.append(('selectEISfeatures',selectEISfeatures))#add the fitted columnTransformer to pipeline      
    elif select_demo_features != -1:
        from sklearn.compose import ColumnTransformer, make_column_selector
        cols=list(traindata.columns)
        selectDemofeatures = ColumnTransformer(transformers=[("selectDemofeatures", 'passthrough', select_demo_features),
                                                             ('selectclassvariable','passthrough',make_column_selector(pattern=cols[-1]))],
                                              remainder="drop")
        (_,c)=traindata.shape
        X1=traindata.iloc[:,:c-1]
        y1=traindata.iloc[:,c-1]
        selectDemofeatures.fit(X1,y1)
        print(select_demo_features)
        steps.append(('selectDemofeatures',selectDemofeatures))#add the fitted columnTransformer to pipeline        
    if transform_to_polynomial_features_of_a_dataset:
        mypoly = MyPolynomialFeatures.MyPolynomialFeatures(dataset_with_polynomial_features_csv)
        steps.append(('mypoly',mypoly))
        print('construct the same polynomial features as the ones of a dataset')
    elif select_original_features_of_a_dataset: #select the same original features as the ones of a dataset
        f=open(dataset_with_original_features_csv,'r')
        original_features_and_class=f.readline()
        original_features_and_class=original_features_and_class.rstrip()#remove the newline
        original_features_and_class=original_features_and_class.split(",")
        original_features=original_features_and_class[:-1]
        f.close()
        from sklearn.compose import ColumnTransformer, make_column_selector
        cols=list(traindata.columns)
        selectOriginalfeatures = ColumnTransformer(transformers=[("selectOriginalfeatures", 'passthrough', original_features),
                                                                 ('selectclassvariable','passthrough',make_column_selector(pattern=original_features_and_class[-1]))],
                                                   remainder="drop")
        (_,c)=traindata.shape
        X1=traindata.iloc[:,:c-1]
        y1=traindata.iloc[:,c-1]
        selectOriginalfeatures.fit(X1,y1)
        print('select the same features as the ones of a dataset')
        steps.append(('selectOriginalfeatures',selectOriginalfeatures))
    if polynomial_degree_within_cv > 1:
        poly = PolynomialFeatures(degree=polynomial_degree_within_cv,interaction_only=interaction_only_within_cv)
        steps.append(('poly_features',poly))
        steps.append(('remove features with lowest variance',VarianceThreshold(threshold=0)))
        print('construct polynomial features of degree='+str(polynomial_degree_within_cv)+' during each training-test process of CV')
    if modelname=='knn':#select features using knn then train knn
        rfecv = RFECV(
                 estimator=KNeighborsClassifier(n_neighbors=3),
                 step=1,
                 cv=cv,
                 scoring=training_score,
                 min_features_to_select=5,
                 n_jobs=-1
                )
        steps.append(('rfecv',rfecv))
    if oversampling_method_within_cv=='random_sampling_with_replacement':
        from imblearn.over_sampling import RandomOverSampler
        random_state2=random.randint(0,999999) 
        if balanced_trainset_size_within_cv =='bootstrap':
            print('random sampling with replacement within each fold of CV')
            print('sampling strategy: ',balanced_trainset_size_within_cv)
            from imblearn.under_sampling import RandomUnderSampler
            (r,_)=traindata.shape
            train_size=r*(cv-1)/cv #size of training data of 5-fold CV = whole training data x 4/5
            print(train_size)
            class0_size=int(train_size/2)
            class1_size=int(train_size-class0_size)
            print(class0_size)
            print(class1_size)
            #create a balanced bootstrap
            steps.append(('ro',RandomOverSampler(sampling_strategy='minority',random_state=random_state2)))                   
            steps.append(('ru',RandomUnderSampler(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2)))                   
        elif isinstance(balanced_trainset_size_within_cv,int) and balanced_trainset_size_within_cv > 0:
            print('random sampling with replacement within each fold of CV')
            print('sampling strategy: '+str(balanced_trainset_size_within_cv))    
            class0_size=int(balanced_trainset_size_within_cv/2)
            class1_size=balanced_trainset_size_within_cv - class0_size
            steps.append(('ro',RandomOverSampler(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2)))       
        else: 
            steps.append(('ro',RandomOverSampler(sampling_strategy=balanced_trainset_size_within_cv,random_state=random_state2)))                       
    elif oversampling_method_within_cv == 'smote':
            random_state2=random.randint(0,999999) 
            print('SMOTE within each fold of CV')
            print('sampling strategy: '+str(balanced_trainset_size_within_cv))
            from imblearn.over_sampling import SMOTE
            #if balanced_trainset_size_within_cv > 0:
            if isinstance(balanced_trainset_size_within_cv,int) and balanced_trainset_size_within_cv > 0:
                   class0_size=int(balanced_trainset_size_within_cv/2)
                   class1_size=balanced_trainset_size_within_cv - class0_size
                   steps.append(('smote', SMOTE(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2)))
            else: 
                steps.append(('smote',SMOTE(sampling_strategy=balanced_trainset_size_within_cv,random_state=random_state2)))                       
    elif oversampling_method_within_cv == 'svmsmote':
            random_state2=random.randint(0,999999) 
            print('SVMSMOTE within each fold of CV')
            print('sampling strategy: '+str(balanced_trainset_size_within_cv))
            from imblearn.over_sampling import SVMSMOTE
            if isinstance(balanced_trainset_size_within_cv,int) and balanced_trainset_size_within_cv > 0:
                   class0_size=int(balanced_trainset_size_within_cv/2)
                   class1_size=balanced_trainset_size_within_cv - class0_size
                   steps.append(('svmsmote', SVMSMOTE(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2)))
            else: 
                steps.append(('svmsmote',SVMSMOTE(sampling_strategy=balanced_trainset_size_within_cv,random_state=random_state2)))                       
    elif oversampling_method_within_cv == 'kmeanssmote':
            random_state2=random.randint(0,999999) 
            print('KMeansSMOTE within each fold of CV')
            print('sampling strategy: '+str(balanced_trainset_size_within_cv))
            from imblearn.over_sampling import KMeansSMOTE
            if isinstance(balanced_trainset_size_within_cv,int) and balanced_trainset_size_within_cv > 0:
                   class0_size=int(balanced_trainset_size_within_cv/2)
                   class1_size=balanced_trainset_size_within_cv - class0_size
                   steps.append(('kmeanssmote', KMeansSMOTE(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2)))
            else: 
                steps.append(('kmeanssmote',KMeansSMOTE(sampling_strategy=balanced_trainset_size_within_cv,random_state=random_state2)))                    
    elif oversampling_method_within_cv=='borderline_smote':
            print('Borderline SMOTE within each fold of CV')
            print('sampling strategy: '+str(balanced_trainset_size_within_cv))
            random_state2=random.randint(0,999999) 
            from imblearn.over_sampling import BorderlineSMOTE
            if isinstance(balanced_trainset_size_within_cv,int) and balanced_trainset_size_within_cv > 0:
                   class0_size=int(balanced_trainset_size_within_cv/2)
                   class1_size=balanced_trainset_size_within_cv - class0_size
                   steps.append(('borderlinesmote',BorderlineSMOTE(sampling_strategy={0:class0_size,1:class1_size},k_neighbors=5,random_state=random_state2)))
            else:
              steps.append(('borderlinesmote',BorderlineSMOTE(sampling_strategy=balanced_trainset_size_within_cv,k_neighbors=5,random_state=random_state2)))                           
    elif oversampling_method_within_cv=='adasyn':
            print('ADASYN within each fold of CV')
            print('sampling strategy: '+str(balanced_trainset_size_within_cv))
            random_state2=random.randint(0,999999) 
            from imblearn.over_sampling import ADASYN
            if isinstance(balanced_trainset_size_within_cv,int) and balanced_trainset_size_within_cv > 0:
                    class0_size=int(balanced_trainset_size_within_cv/2)
                    class1_size=balanced_trainset_size_within_cv - class0_size
                    steps.append(('adasyn',ADASYN(sampling_strategy={0:class0_size,1:class1_size},random_state=random_state2)))
            else:
              steps.append(('adasyn',ADASYN(sampling_strategy=balanced_trainset_size_within_cv,random_state=random_state2)))               
    elif oversampling_method_within_cv==-1:
        print('oversampling is not performed on each training set of CV.')
    else:
        sys.exit('invalid oversampling method ='+str(oversampling_method_within_cv)+' or balanced_trainset_size_within_cv='+str(balanced_trainset_size_within_cv))
    if modelname!='rf' and fs_select_from_model_within_cv:
        random_state2=random.randint(0,999999) 
        fs=SelectFromModel(DecisionTreeClassifier(criterion='entropy',random_state=random_state2))
        steps.append(('fs',fs))
        print('select features from decision tree')
    elif modelname =='rf' and fs_select_from_model_within_cv:
        random_state2=random.randint(0,999999) 
        fs=SelectFromModel(RandomForestClassifier(n_jobs=-1,class_weight='balanced',n_estimators=50,random_state=random_state2))
        steps.append(('fs',fs))
        print('select features from random forest')
    elif k_within_cv > 0:
            fs=SelectKBest(mutual_info_classif, k=k_within_cv)
            steps.append(('fs',fs))
            print('select '+str(k_within_cv)+' features using info gain within each fold of CV')
    steps.append(('remove features with close to 0 variance',VarianceThreshold(threshold=variance_threshold_within_cv)))
    if transform_inputs_within_cv=='zscore':
            zscore=preprocessing.StandardScaler()                
            steps.append(('zscore',zscore))
    elif transform_inputs_within_cv=='minmax':
            minmax=preprocessing.MinMaxScaler(feature_range=(int(mini),int(maxi)))
            steps.append(('minmax',minmax))    
    elif transform_inputs_within_cv==None:#useoriginal features to train model
                print('Normalization of features is not performed before training',modelname)
    else:
         sys.exit('invalid transform inputs: ',transform_inputs_within_cv)
    steps.append((modelname, model))               
    pipe = Pipeline(steps=steps)
    #(_,c)=traindata.shape
    #X1=traindata.iloc[:,:c-1]
    #y1=traindata.iloc[:,c-1]
    #pipe.fit(X1,y1)            
    (_,_,train_set_indicesL,val_set_indicesL)=cross_validation_split_and_remove_duplicates_from_valid_set(traindata,cv,random_state2=1)
    train_val_splits=[]
    for k in range(len(train_set_indicesL)):
        train_val_splits.append((train_set_indicesL[k],val_set_indicesL[k]))
    if randomsearchCV:
        search = RandomizedSearchCV(pipe, param_grid, n_iter=n_iter, scoring=training_score, cv=cv)                          
    else:
        search = GridSearchCV(pipe, param_grid, scoring=training_score, n_jobs=-1, cv=train_val_splits)    
    return search
       
def select_optimal_feature_subset_using_recursive_feature_elimination_cv(trainset=None,
                                                                         training_score='roc_auc',
                                                                         min_features_to_select = 5,  # Minimum number of features to consider
                                                                         cv=5,
                                                                         estimator='decisiontree'
                                                                         ):
        (_,c)=trainset.shape
        X1=trainset.iloc[:,:c-1]
        y1=trainset.iloc[:,c-1]
        X1=X1.astype(float)
        if estimator=='decisiontree':      
            clf = DecisionTreeClassifier(criterion='entropy')
        elif estimator=='gp':
            clf = GaussianProcessClassifier(kernel=1.0*RBF(1.0))
        elif estimator=='gaussianNB':
            clf = GaussianNB()
        elif estimator=='rf':
            clf=RandomForestClassifier(n_jobs=-1,n_estimators=30,random_state=1)        
        elif estimator=='knn':
            clf=KNeighborsClassifier(n_neighbors=5)
            clf=clf.fit(X1,y1)
        elif estimator=='ARD':#automatic relevance determination
            clf=ARDRegression()
        else:
            sys.exit('unknown estimator: '+str(estimator))
        
        rfecv = RFECV(
                 estimator=clf,
                 step=1,
                 cv=cv,
                 scoring=training_score,
                 min_features_to_select=min_features_to_select,
                 n_jobs=-1
                )
        rfecv.fit(X1, y1)               
        cols=list(trainset.columns)
        fsL=[]
        mask=rfecv.get_support()
        for i in range(c-1):
            if mask[i]:
                fsL.append(cols[i])
        X1=rfecv.transform(X1)
        X1=pd.DataFrame(X1, index=trainset.index, columns=fsL)        
        reduced_trainset=X1.join(y1)
        print("Optimal number of features selected by rfecv: %d" % rfecv.n_features_)
        print(str(cv)+"-fold cv "+training_score+": ",rfecv.cv_results_['mean_test_score'])
        print("selected features: ",list(reduced_trainset.columns))
        return reduced_trainset

def ensemble_naive_bayes(reductsfile="ga_reducts.txt",
                         number_of_reducts=30,
                         trainset=None,
                         voting_strategy='soft'):
    #build an ensemble of gaussian naive bayes classifiers from the feature subsets output by GA feature selection
    reductsL=[line.strip() for line in open(reductsfile)]
    reductsL=get_random_feature_subsets(reductsL,number_of_reducts)
    nb = GaussianNB()                                                          
    modelsL=Parallel(n_jobs=-1,batch_size=10)(delayed(cv_and_train_model_using_feature_subset)(j,reductsL,trainset,nb,nb_param) for j in range(len(reductsL)))
    eclf = VotingClassifier(estimators=modelsL, #[('svc', svc), ('rf', rf), ('gnb', gnb)]
                            voting=voting_strategy,
                            weights=None)
    return eclf
'''
def cv_and_train_model_using_feature_subset(j,reductsL,trainset,clf,clf_param,training_score='roc_auc',cv=5):
    #reduce the training set using jth feature subset, then, find best parameter of classifier using cross validation on the reduced training set and train a classifier using the best parameter
    #return the classifier
    (_,c)=trainset.shape
    cols=(trainset.columns)
    reduct=reductsL[j]
    indicesL=reduct.split(',')
    fsL=[]
    for i in range(len(indicesL)):
        indx=int(indicesL[i])-1 #change indices to start from 0 by subtracting 1 from the indices which start from 1 
        fsL.append(cols[indx])
    fsL.append(cols[c-1])
    trainset2=trainset[fsL]
    (_,c)=trainset2.shape
    X=trainset2.iloc[:,:c-1]
    y=trainset2.iloc[:,c-1]
    X=X.astype(float)
    search = GridSearchCV(clf, clf_param, scoring=training_score, cv=cv)
    search.fit(X,y)
    best_params=search.best_params_
    print("Best parameter: ", best_params)
    print("CV "+str(training_score)+" of best parameter=%0.3f" % search.best_score_)
    trainedmodel=search.best_estimator_
    return ('model'+str(j),trainedmodel)
'''
def cv_score_of_feature_subset(j,reductsL,trainset,clf,training_score,cv):
    #reduce the training set using jth feature subset, then, cross-validate a classifier on the reduced training set
    #return the CV performance of the classifier
    (_,c)=trainset.shape
    cols=(trainset.columns)
    reduct=reductsL[j]
    indicesL=reduct.split(',')
    fsL=[]
    for i in range(len(indicesL)):
        indx=int(indicesL[i])-1 #change indices to start from 0 by subtracting 1 from the indices which start from 1 
        fsL.append(cols[indx])
    fsL.append(cols[c-1])
    trainset2=trainset[fsL]
    #print('reduct: ',trainset2.columns)
    (_,c)=trainset2.shape
    X=trainset2.iloc[:,:c-1]
    y=trainset2.iloc[:,c-1]
    X=X.astype(float)
    scores = cross_val_score(clf, X, y, scoring=training_score, cv=cv)
    cv_performance=scores.mean()
    return (cv_performance,trainset2)

def select_optimal_feature_subset_using_cv(reductsfile="ga_reducts.txt",
                                           number_of_reducts=30,
                                           trainset=None,
                                           model='rf',                                          
                                           training_score='roc_auc',
                                           cv=5,
                                           C=100, #regularization of SVM (misclassification cost)
                                           activation='relu',# parameters of MLP
                                           hidden_layer_sizes=(16,16),
                                           solver='lbfgs',
                                           max_iter=2000,
                                           alpha=1e-3,
                                           learning_rate_init=1e-2,
                                           early_stopping=True,
                                           n_estimators_rf=50,#no. of trees of random forest
                                           n_estimators_xgb=80, #no. of trees of xgboost and xgboost random forest
                                           max_depth=3 #max depth of xgb and xgboost random forest
                                           ):
    #select an optimal feature subset from a set of feature subsets output by GA feature selection according to the CV performance of a classifier
    #input: reductsfile of GA feature selection
    #       trainset, training data dataframe
    #       model
    #       training score: roc_auc or balanced accuracy
    #       k-fold cv
    #output: reduced training set using the best reduct selected using CV of the model
    #format of reductsfile: features indices start from 1 and class attribute not included
    # 1, 3, 5, 6, 7
    # 2, 4, 6, 8
    #from sklearn.model_selection import cross_val_score
    
    seed=0        
    if model=='rf':
        clf=RandomForestClassifier(n_jobs=-1,class_weight='balanced',n_estimators=n_estimators_rf,random_state=seed)        
    elif model=='softmaxreg':
        clf=LogisticRegression(max_iter=10000, multi_class='multinomial', solver='lbfgs', C=1e-8,class_weight='balanced')
    elif model=='logreg':
        clf = LogisticRegression(max_iter=10000, solver='lbfgs', C=1e-8,class_weight='balanced')
    elif model=='gaussianNB':
        clf = GaussianNB()
    elif model=='gp':
        clf = GaussianProcessClassifier(kernel=1.0*RBF(1.0))
    elif model=='decisiontree':
        clf=DecisionTreeClassifier(criterion='entropy')
    else:
        sys.exit('invalid model in select_optimal_feature_subset_using_cv',model)
    print('select an optimal feature subset based on the CV performance of '+model)   
    best_cv_performance=0
    reduced_train_set_using_best_feature_subset=None
    reductsL=[line.strip() for line in open(reductsfile)]
    reductsL=get_random_feature_subsets(reductsL,number_of_reducts)                                                          
    cv_performanceL=Parallel(n_jobs=-1,batch_size=10)(delayed(cv_score_of_feature_subset)(j,reductsL,trainset,clf,training_score,cv) for j in range(len(reductsL)))
    #get CV performance of set of all features
    (_,c)=trainset.shape
    X=trainset.iloc[:,:c-1]
    y=trainset.iloc[:,c-1]
    X=X.astype(float)
    scores = cross_val_score(clf, X, y, scoring=training_score, cv=cv)
    cv_performance_of_all_features=scores.mean()
    cv_performanceL.append((cv_performance_of_all_features,trainset))#compare performance of all the features with the performance of feature subsets 
    cv_performanceL.sort(key=operator.itemgetter(0),reverse=True)
    best_cv_performance_reduced_trainset=cv_performanceL[0]
    best_cv_performance=best_cv_performance_reduced_trainset[0]
    reduced_train_set_using_best_feature_subset=best_cv_performance_reduced_trainset[1]
    print('best cv '+training_score+'='+str(best_cv_performance))
    return reduced_train_set_using_best_feature_subset

def training_testing(trainset_file='trainset',
                     testset_file='testset',
                     select_EIS_features=-1,
                     select_demo_features=-1,
                     testset_path=None
                     ):
        create_folder_if_not_exist(results_path)
        if predict_ptb_of_each_id_using_all_spectra_of_id:
           import ModelsPredict as mp
        seed=123456
        #seed=0
        #seed=1
        print('seed:',seed)
        f=open(logfile,'w')
        f.write(option+'\n')
        if option=='training_testing_on_whole_dataset' or option=='split_training_testing':
            f.write('dataset='+dataset+'\n')
        f.write('results_path='+results_path+'\n')
        if transform_inputs_within_cv=='minmax':
            f.write('minmax normalization of features: '+transform_inputs_within_cv+', minimum='+str(mini)+', maximum='+str(maxi)+'\n')
        elif transform_inputs_within_cv=='zscore':
            f.write('zscore normalization of features\n')
        else:
            f.write('no normalization of inputs is used\n')
        if option=='split_training_testing' or option=='training_testing_on_whole_dataset':
            f.write(dataset+'\n')
            print(dataset+'\n')
        if option=='training_testing':
            f.write('trainset_file: '+trainset_file+'\n')
            f.write('testset_file: '+testset_file+'\n')
            if preprocessed_trainset_file_ext!=None:
                f.write('preprocessed_trainset_file_ext: '+preprocessed_trainset_file_ext+'\n')
        if option=='split_training_testing':    
            f.write('training size (%): '+str((1-test_size)*100)+'\n')
            f.write('testing size (%): '+str(test_size*100)+'\n')
            print('training size (%): '+str((1-test_size)*100))
            print('testing size (%): '+str(test_size*100))
        if option=='split_training_testing' or option=='training_testing':    
            f.write('iterations: '+str(iterations)+'\n')
        if remove_duplicates_from_valid_set:
            f.write('Remove duplicates from the validation sets of k-fold CV ='+str(remove_duplicates_from_valid_set)+'\n')
        if model=='linear_svm':
            print('parameters of linear svm: ',linear_svm_param_grid)
            f.write('parameters of linear svm: '+str(linear_svm_param_grid)+'\n')
        elif model=='poly_svm':
            print('parameters of polynomial kernel svm: ',poly_svm_param_grid)
            f.write('parameters of poly svm: '+str(poly_svm_param_grid)+'\n')
        elif model=='rbf_svm':
            print('parameters of rbf kernel svm: ',rbf_svm_param_grid)
            f.write('parameters of rbf svm: '+str(rbf_svm_param_grid)+'\n')       
        elif model=='softmaxreg':
            print('parameters of softmax regression: ',logreg_param)
            f.write('parameters of softmax regression: '+str(logreg_param)+'\n')
        elif model=='logreg':
            print('parameters of logistic regression: ',logreg_param)
            f.write('parameters of logistic regression: '+str(logreg_param)+'\n')
        elif model=='mlp':
            print('parameters of mlp: ',mlp_param_grid)
            f.write('parameters of mlp: '+str(mlp_param_grid)+'\n')
        elif model=='xgb':
            f.write('parameter of xgboost: '+str(xgb_param)+'\n')
        elif model=='lgbm':
            f.write('parameter of light gradient boosting: '+str(lgbm_param)+'\n')
        elif model=='xgbrf':
            f.write('parameter of xgboost random forest: '+str(xgb_param)+'\n')
        elif model=='gp_rbf':
            f.write('parameter of gaussian process with rbf kernel: '+str(gp_rbf_param)+'\n')
        elif model=='gp_matern':
            f.write('parameter of gaussian process with matern kernel: '+str(gp_matern_param)+'\n')
        elif model=='stacked_ensemble':
            f.write('stacked ensemble\n')
        f.close()
        performanceL=[]
        train_aucL=[]
        test_aucL=[]
        tpr_tnr_fpr_fnrL=[] #list of (int(interation),(tpr2+tnr2)/2,tpr,tnr,fpr,fnr,tpr2,tnr2,fpr2,fnr2)
        print('iterations: ',iterations)
        #l=list(range(26,iterations))
        #for iteration in l:
        single_train_set=False
        for iteration in range(iterations):
            print('iteration: ',iteration)
            seed0=random.randint(0,2**32-1)
            f=open(logfile,'a')
            if option=='training_testing':
                f.write('iteration: '+str(iteration)+'\n')
                f.close()
                if preprocessed_trainset_file_ext!=None:#train classifiers on preprocessed training set
                    if os.path.isfile(data_path+trainset_file+str(iteration)+preprocessed_trainset_file_ext):
                        train_set=pd.read_csv(data_path+trainset_file+str(iteration)+preprocessed_trainset_file_ext)
                    else:
                        sys.exit('preprocessed trainset does not exist: '+data_path+trainset_file+str(iteration)+preprocessed_trainset_file_ext)                              
                else:#train classifiers on training set
                    if data_path!=None and os.path.isfile(data_path+trainset_file+str(iteration)+'.csv'):
                        train_set=pd.read_csv(data_path+trainset_file+str(iteration)+'.csv')
                    elif os.path.isfile(trainset_file):
                        train_set=pd.read_csv(trainset_file)
                        single_train_set=True
                    else:
                        sys.exit('trainset does not exist: '+data_path+trainset_file+str(iteration)+'.csv does not exist. '+trainset_file+' does not exist.')  
                cols=list(train_set.columns)
                class_var=cols[-1]               
                if cols[0] == 'Identifier' or cols[0] == 'hospital_id':
                    (_,c)=train_set.shape
                    train_set=train_set.iloc[:,1:c] #removed ids column                
            elif option=='preprocess':
                f.write('iteration: '+str(iteration)+'\n')
                f.close()
                if os.path.isfile(data_path+trainset_file+str(iteration)+'.csv'):
                    train_set=pd.read_csv(data_path+trainset_file+str(iteration)+'.csv')
                else:
                    sys.exit('trainset does not exist: '+data_path+trainset_file+str(iteration)+'.csv')
                cols=list(train_set.columns)
                class_var=cols[-1]
                print('class var:'+str(class_var))
                if cols[0] == 'Identifier' or cols[0] == 'hospital_id':
                    (_,c)=train_set.shape
                    train_set=train_set.iloc[:,1:c] #removed ids column                
            elif option=='split_training_testing':
                f.write('iteration: '+str(iteration)+'\n')
                f.close()
                data=pd.read_csv(dataset)
                cols=list(data.columns)
                class_var=cols[-1]
                if cols[0] == 'Identifier' or cols[0] == 'hospital_id':
                    (_,c)=data.shape
                    data=data.iloc[:,1:c] #removed ids column                
                cols=list(data.columns)                
                (train_set,test_set)=split_train_test_sets(data,test_size,seed0,cols[-1]) #cols[-1] is the last element in list
                train_set.to_csv(results_path+'train_set.csv',index=False)
                test_set.to_csv(results_path+'test_set.csv',index=False)
                train_set=pd.read_csv(results_path+'train_set.csv')
                test_set=pd.read_csv(results_path+'test_set.csv')
            elif option=='training_testing_on_whole_dataset':
                data=pd.read_csv(dataset)
                train_set=data
                cols=list(train_set.columns)
                class_var=cols[-1]               
                if cols[0] == 'Identifier' or cols[0] == 'hospital_id':
                    (_,c)=train_set.shape
                    train_set=train_set.iloc[:,1:c] #removed ids column                
                f.write('===training and testing on whole dataset===\n')
                f.close()
            else:
                sys.exit('invalid option: ',option)
            if select_EIS_features != -1:
                #print(select_EIS_features)
                EIS_features_and_class = select_EIS_features + [class_var] #do not use append method to add class variable to EIS_features as doing so will add class variable to select_EIS_features                
                train_set = train_set[EIS_features_and_class]
                #print(train_set.columns)
                trainset2_file='trainset_EIS'
            elif select_demo_features != -1:
                demo_features_and_class = select_demo_features + [class_var]
                train_set = train_set[demo_features_and_class]
                trainset2_file='trainset_Demo'
            else:
                trainset2_file='trainset'
            #preprocessing trainings set         
            train_set=fill_missing_values('median','df',train_set)        
            (_,c)=train_set.shape
            X=train_set.iloc[:,:c-1]
            y=train_set.iloc[:,c-1]
            train_set.to_csv(results_path+'trainset_no_missing_values'+str(iteration)+'.csv',index=False)
            files_to_delete=[results_path+'trainset_no_missing_values'+str(iteration)+'.csv']            
            if balanced_trainset_size2!=-1 or degree2!=-1 or k2!=-1 or number_of_reducts > 0:
                print('oversampling and/or polynomial features construction and/or information gain feature selection and/or random forest feature selection (logistic regression feature selection)')
                f=open(logfile,'a')
                f.write('balanced training set size: '+str(balanced_trainset_size2)+'\n')
                f.write(oversampling_method+'\n')
                f.write('degree of polynomial features: '+str(degree2)+'\n')
                f.write('no. of features to select using information gain: '+str(k2)+'\n')
                f.close()
                weka_path2=weka_path
                java_memory2='4g'                
                workflow1(                                                       
                             option='preprocess',
                             dataset=results_path+'trainset_no_missing_values'+str(iteration)+'.csv',
                             preprocessed_train_set_csv=results_path+trainset2_file+str(iteration)+'.preprocessed.csv',
                             results_path2=results_path,
                             iteration_number=iteration,
                             balanced_trainset_size=balanced_trainset_size2,
                             oversampling_method2=oversampling_method,
                             degree=degree2,
                             interaction_only2=interaction_only,
                             k=k2,
                             populationSize2=100,
                             generations2=50,
                             #populationSize2=50,
                             #generations2=5,                             
                             crossover_prob2=0.6,
                             mutation_prob2=0.033,
                             fitness2='find_reducts',
                             discretize_method2='pki',
                             number_of_reducts2=number_of_reducts,
                             mini_features2=-1,
                             max_features2=-1,
                             trees=[20],
                             seed='123456',
                             logfile=results_path+'logfile_workflow1.txt',
                             weka_path=weka_path2,
                             java_memory=java_memory2
                           )
                if number_of_reducts==0:#oversampled training data or constructed polynomial features or information gain feature selection
                    train_set=pd.read_csv(results_path+trainset2_file+str(iteration)+'.preprocessed.csv')
                    if option=='preprocess':
                        train_set.to_csv(results_path+trainset2_file+str(iteration)+'_preprocessed_balanced_trainset_size='+str(balanced_trainset_size2)+'_degree='+str(degree2)+'_info_gain_selected_features='+str(k2)+'.csv',index=False)
                        print('preprocessed training set is saved to:',results_path+trainset2_file+str(iteration)+'_preprocessed_balanced_trainset_size='+str(balanced_trainset_size2)+'_degree='+str(degree2)+'_info_gain_selected_features='+str(k2)+'.csv')
                elif number_of_reducts > 0:#use GA feature selection to select numerous optimal feature subsets, then select an optimal feature subset from these best feature subset using CV of a classifier
                    train_set_preprocessed=pd.read_csv(results_path+trainset2_file+str(iteration)+'.preprocessed.csv')
                    if model=='ensemble_naive_bayes':
                        print('build ensemble naive bayes from the feature subsets output by GA feature selection.')
                    else:
                        print('select an optimal feature subset from '+str(number_of_reducts)+' feature subsets found by GA feature selection based on '+str(cv)+'-fold CV performance of '+model_of_select_optimal_feature_subset_using_cv)
                        train_set=select_optimal_feature_subset_using_cv(reductsfile=results_path+"ga_reducts"+str(iteration)+".txt",
                                                                         number_of_reducts=number_of_reducts,
                                                                         trainset=train_set_preprocessed,
                                                                         training_score=training_score,
                                                                         cv=cv,
                                                                         model=model_of_select_optimal_feature_subset_using_cv
                                                                         )
                        f=open(logfile,'a')
                        f.write('select an optimal feature subset from '+str(number_of_reducts)+' feature subsets found by GA feature selection based on '+str(cv)+'-fold CV performance of '+model_of_select_optimal_feature_subset_using_cv+'\n')
                        f.close()
                    if option=='preprocess':
                        train_set.to_csv(results_path+trainset2_file+str(iteration)+'_preprocessed_balanced_trainset_size='+str(balanced_trainset_size2)+'_degree='+str(degree2)+'_info_gain_selected_features='+str(k2)+'_ga_cv_of_'+model_of_select_optimal_feature_subset_using_cv+'_selected_features.csv',index=False)
                        print('preprocessed training set is saved to:',results_path+trainset2_file+str(iteration)+'_preprocessed_balanced_trainset_size='+str(balanced_trainset_size2)+'_degree='+str(degree2)+'_info_gain_selected_features='+str(k2)+'_ga_cv_of_'+model_of_select_optimal_feature_subset_using_cv+'_selected_features.csv')
                files_to_delete.append(results_path+trainset2_file+str(iteration)+'.preprocessed.csv')
            if recursive_feature_elimination_cv:
                if balanced_trainset_size2!=-1 or degree2!=-1 or k2!=-1 or number_of_reducts > 0:
                    train_set2=pd.read_csv(results_path+trainset2_file+str(iteration)+'.preprocessed.csv')
                else:
                    train_set2 = train_set
                train_set = select_optimal_feature_subset_using_recursive_feature_elimination_cv(trainset=train_set2,
                                                                                                 training_score=training_score,
                                                                                                 min_features_to_select = 5,  # Minimum number of features to consider
                                                                                                 cv=cv,
                                                                                                 estimator=estimator_of_rfe_cv
                                                                                                 )
                print('find an optimal feature subset using recursive feature elimination '+str(cv)+'-fold CV of '+estimator_of_rfe_cv)
                f=open(logfile,'a')
                f.write('find an optimal feature subset using recursive feature elimination '+str(cv)+'-fold CV of '+estimator_of_rfe_cv+'\n')
                f.close()
                if option=='preprocess':
                   train_set.to_csv(results_path+trainset_file+str(iteration)+'_preprocessed_balanced_trainset_size='+str(balanced_trainset_size2)+'_degree='+str(degree2)+'_info_gain_selected_features='+str(k2)+'_rfe_cv_of_decision_tree_selected_features.csv',index=False)
                   print('preprocessed training set is saved to:',results_path+trainset_file+str(iteration)+'_preprocessed_balanced_trainset_size='+str(balanced_trainset_size2)+'_degree='+str(degree2)+'_info_gain_selected_features='+str(k2)+'_rfe_cv_of_decision_tree_selected_features.csv')
                   files_to_delete.append(results_path+trainset_file+str(iteration)+'_preprocessed_balanced_trainset_size='+str(balanced_trainset_size2)+'_degree='+str(degree2)+'_info_gain_selected_features='+str(k2)+'.csv')
                   files_to_delete.append(results_path+'trainset'+str(iteration)+'.preprocessed.csv')
            if option=='preprocess':
                    delete_files(files_to_delete)#delete temporary files of this iteration
                    continue
            elif balanced_trainset_size2==-1 and degree2==-1 and k2==-1 and number_of_reducts < 1:
                print('original training set -> classifier training')
                (_,c)=train_set.shape
                X=train_set.iloc[:,:c-1]
                y=train_set.iloc[:,c-1]
            else:
                    get_model_inputs_output('df',train_set,results_path+'trainset'+str(iteration)+'.preprocessed_inputs_output.csv')
                    (_,c)=train_set.shape
                    X1=train_set.iloc[:,:c-1]
                    y1=train_set.iloc[:,c-1]
                    X1=X1.astype(float)
                    ####Reduce the original training set using the features of the preprocessed training set (preprocessed training set can have polynomial features or original features)
                    reduce_data(results_path+'trainset_no_missing_values'+str(iteration)+'.csv',results_path+'trainset'+str(iteration)+'.preprocessed_inputs_output.csv',results_path+'original_trainset'+str(iteration)+'.preprocessed.csv')
                    train_set=pd.read_csv(results_path+'original_trainset'+str(iteration)+'.preprocessed.csv')
                    (_,c)=train_set.shape
                    X=train_set.iloc[:,:c-1]
                    y=train_set.iloc[:,c-1]
                    files_to_delete.append(results_path+'trainset'+str(iteration)+'.preprocessed.csv')
                    files_to_delete.append(results_path+'trainset'+str(iteration)+'.preprocessed_inputs_output.csv')
                    files_to_delete.append(results_path+'original_trainset'+str(iteration)+'.preprocessed.csv')            
            f=open(logfile,'a')
            f.write('balanced_trainset_size_within_cv='+str(balanced_trainset_size_within_cv)+'\n')
            f.write('oversampling_method_within_cv='+str(oversampling_method_within_cv)+'\n')
            f.write('fs_select_from_model_within_cv='+str(fs_select_from_model_within_cv)+'\n')
            f.write('k_within_cv='+str(k_within_cv)+'\n')
            f.close()
            ###training a classifier
            if balanced_trainset_size2!=-1 or degree2!=-1 or k2!=-1 or number_of_reducts > 0:
                    traindata=X1.join(y1)
            else:
                    traindata=X.join(y)         
            if option=='preprocess':
                continue           
            elif model=='ensemble_naive_bayes':
                print('===ensemble naive bayes===')
                print('voting strategy:',voting_strategy)
                search=ensemble_naive_bayes(reductsfile=results_path+"ga_reducts"+str(iteration)+".txt",
                                            number_of_reducts=number_of_reducts,
                                            trainset=train_set_preprocessed,
                                            voting_strategy=voting_strategy)            
            elif model=='stacked_ensemble':
                print('===stacked ensemble===')
                print('===1: rbf svm===')
                from joblib import load
                ###load each classifier trained using the best parameters setting found by CV
                f=open(logfile,'a')
                f.write('===1. rbf svm===\n')
                #load svc training pipeline with best parameter settings
                svc=load(svc_model)                
                f.write('model='+svc_model+'\n')
                f.write('cv auc='+str(svc_cv_auc)+'\n')
                print('===2. Gaussian Processes===')
                f.write('===2. Gaussian Processes===\n')
                #load gp training pipeline with best parameter settings 
                gp=load(gp_model)
                f.write('model='+gp_model+'\n')
                f.write('cv auc='+str(gp_cv_auc)+'\n')
                print('===3. mlp===')
                f.write('===3. MLP ===\n')
                #load mlp training pipeline with best parameter settings
                mlp=load(mlp_model)                         
                f.write('model='+mlp_model+'\n')
                f.write('cv auc='+str(mlp_cv_auc)+'\n')
                print('===4. Gaussian Naive Bayes===')
                f.write('===4. Gaussian Naive Bayes===\n')
                nb=load(nb_model)
                f.write('model='+nb_model+'\n')
                f.write('cv auc='+str(nb_cv_auc)+'\n')              
                f.close()
                base_models = [('svc', svc),
                               ('gp',gp),
                               ('mlp',mlp),
                               ('nb',nb)]
                #combiner=LogisticRegression(max_iter=10000, solver='lbfgs', C=1e-8,class_weight='balanced') 
                ###set the parameters of the combiner to the ones of the base model, then, train the combiner 
                if combiner=='gp':
                    combiner=gp
                    param_grid=gp_rbf_with_noise_param
                elif combiner=='nb':
                    combiner=nb
                    param_grid=nb_param
                elif combiner=='svc':
                    combiner=svc
                    param_grid=rbf_svm_param_grid
                elif combiner=='mlp':
                    combiner=mlp
                    param_grid=mlp_param_grid
                else:
                    sys.exit('invalid combiner: '+combiner)
                ensemble = StackingClassifier(
                    estimators = base_models,
                    final_estimator = combiner,                    
                    stack_method='predict_proba',#predict prob of class using each base model
                    #stack_method='predict',#predict class using each base model
                    #cv = 'prefit') #the estimators are prefit and will not be refitted 
                    cv = cv)                
                ####use out-of-sample predictions of CV of based models to train combiner
                f=open(logfile,'a')
                f.write('===use out-of-sample predictions of CV of based models to train combiner===\n')
                final_estimatorSearch = GridSearchCV(ensemble.final_estimator,param_grid,scoring=training_score, cv=cv)
                #ensemble = RandomizedSearchCV(ensemble, stacked_ensemble_param, scoring=training_score, cv=cv)            
            elif model=='voted_classifier':
                print('===voting classifier===')
                print('voting strategy:',voting_strategy)
                if voting_strategy=='hard (base models=ensembles)':
                    print('===1: ensemble1===')
                    f=open(logfile,'a')
                    f.write('===voting classifier===\n')
                    f.write('voting strategy: '+voting_strategy+'\n')                
                    ###load each classifier trained using the best parameters setting found by CV
                    f.write('===1. ensemble1===\n')
                    from joblib import load                   
                    e1=load(ensemble1)
                    f.write('model='+ensemble1+'\n')
                    print('===2. ensemble2===')
                    f.write('===2. ensemble2===\n')
                    e2=load(ensemble2)
                    f.write('model='+ensemble2+'\n')
                    print('===3. ensemble3===')
                    f.write('===3. ensemble3===\n')
                    e3=load(ensemble3)
                    f.write('model='+ensemble3+'\n')
                    print('===4. ensemble4===')
                    f.write('===4. ensemble4===\n')
                    e4=load(ensemble4)
                    f.write('model='+ensemble4+'\n')
                    print('===5. ensemble5===')
                    f.write('===5. ensemble5===\n')
                    e5=load(ensemble5)
                    f.write('model='+ensemble5+'\n')
                    estimators=[('ensemble1',e1),
                                ('ensemble2',e2),
                                ('ensemble3',e3),
                                ('ensemble4',e4),
                                ('ensemble5',e5)]                    
                    eclf = VotingClassifier(estimators=estimators,
                                            voting='hard',
                                            weights=None)
                else:
                    print('===1: rbf svm===')
                    f=open(logfile,'a')
                    f.write('===voting classifier===\n')
                    f.write('voting strategy: '+voting_strategy+'\n')                
                    ###load each classifier trained using the best parameters setting found by CV
                    f.write('===1. rbf svm===\n')
                    from joblib import load
                    #load svc training pipeline with best parameter settings
                    svc=load(svc_model)
                    f.write('model='+svc_model+'\n')
                    f.write('cv auc='+str(svc_cv_auc)+'\n')
                    print('===2. Gaussian Processes===')
                    f.write('===2. Gaussian Processes===\n')
                    #load gp training pipeline with best parameter settings 
                    gp=load(gp_model)
                    f.write('model='+gp_model+'\n')
                    f.write('cv auc='+str(gp_cv_auc)+'\n')
                    print('===3. mlp===')
                    f.write('===3. MLP ===\n')
                    #load mlp training pipeline with best parameter settings
                    mlp=load(mlp_model)                         
                    f.write('model='+mlp_model+'\n')
                    f.write('cv auc='+str(mlp_cv_auc)+'\n')
                    print('===4. Gaussian Naive Bayes===')
                    f.write('===4. Gaussian Naive Bayes===\n')
                    nb=load(nb_model)
                    f.write('model='+nb_model+'\n')
                    f.write('cv auc='+str(nb_cv_auc)+'\n')                
                    #print('===5. Random Forest===')
                    #f.write('===5. Random Forest===\n')
                    #rf=load(rf_model)
                    #f.write('model='+rf_model+'\n')
                    #f.write('cv auc='+str(rf_cv_auc)+'\n')      
                    if voting_strategy=='soft':
                        if weights=='cv_auc':
                            print('weight of a model = CV AUC of model/sum of CV AUCs of models')
                            print('prob of ensemble = weight1 x prob1 + weight1 x prob2 +...+weight_k x prob_k where weight1 is weight of model1, prob1 is prob of model1 etc.')
                            f.write('weight of a model = CV AUC of model/sum of CV AUCs of models\n')
                            f.write('prob of ensemble = weight1 x prob1 + weight1 x prob2 +...+weight_k x prob_k where weight1 is weight of model1, prob1 is prob of model1 etc.\n')
                            f.close()
                            total_auc=svc_cv_auc + gp_cv_auc + mlp_cv_auc + nb_cv_auc
                            svc_w=svc_cv_auc/total_auc
                            gp_w=gp_cv_auc/total_auc
                            mlp_w=mlp_cv_auc/total_auc
                            nb_w=nb_cv_auc/total_auc
                            ws=[svc_w,gp_w,mlp_w,nb_w]
                            estimators=[('svc',svc),('gp',gp),('mlp',mlp),('nb',nb)]                             
                        elif weights=='cv_auc2':
                            print('compute weights of models as CV AUC of model/the smallest CV AUC')
                            f.write('compute weights of models as CV AUC of model/the smallest CV AUC\n')
                            f.close()                        
                            #modelsL=[[CV AUC1,weight1,model1],[CV AUC2,weight2,model2],...]
                            modelsL=[[svc_cv_auc,-1,('svc',svc)],
                                      [gp_cv_auc,-1,('gp',gp)],
                                      [mlp_cv_auc,-1,('mlp',mlp)],
                                      [nb_cv_auc,-1,('nb',nb)]]
                            modelsL.sort(key=operator.itemgetter(0),reverse=False)#sort models in ascending order of CV AUC
                            m=modelsL[0]
                            smallest_cv_auc=m[0]
                            no_of_base_models=len(modelsL)                        
                            for i in range(0,no_of_base_models):#compute weights of models as CV AUC of model/the smallest CV AUC
                                model_i=modelsL[i]
                                cv_auc=model_i[0]
                                w=cv_auc/smallest_cv_auc
                                model_i[1]=w
                                modelsL[i]=model_i                        
                            estimators=[]            
                            ws=[]
                            for model_i in modelsL:
                                estimators.append(model_i[-1])
                                ws.append(model_i[1])
                        elif isinstance(weights,list):#models have user-specified weights
                            estimators=[('svc',svc),
                                        ('gp',gp),
                                        ('mlp',mlp),
                                        ('nb',nb),
                                        ('rf',rf)]
                            ws=weights
                            print('weights of base models: weight of svc='+str(weights[0])+', weight of gp='+str(weights[1])+', weight of mlp='+str(weights[2])+', weight of nb='+str(weights[3])+', weight of rf='+str(weights[4])+'\n')
                            f.write('weights of base models: weight of svc='+str(weights[0])+', weight of gp='+str(weights[1])+', weight of mlp='+str(weights[2])+', weight of nb='+str(weights[3])+', weight of rf='+str(weights[4])+'\n')
                            f.close()
                        else:
                            sys.exit('invalid weights: '+str(weights))
                        eclf = VotingClassifier(estimators=estimators,
                                                voting=voting_strategy,
                                                weights=ws)
                    elif voting_strategy=='hard':#hard voting
                        estimators=[('svc',svc),
                                ('gp',gp),
                                ('mlp',mlp),
                                ('nb',nb)]
                        eclf = VotingClassifier(estimators=estimators,
                                            voting=voting_strategy,
                                            weights=None)
                ensemble=eclf                
            elif model=='gp_polynomial':
                kernel = Exponentiation(DotProduct(sigma_0=0.1),3)
                gp = GaussianProcessClassifier(kernel=kernel)
                search = GridSearchCV(gp, gp_polynomial_param, scoring=training_score, cv=cv)
            elif model=='gp_rbf':#gaussian process
                print('===gaussian process with rbf kernel===')
                kernel = RBF(1.0)
                #kernel = RBF(length_scale=1.0)
                #kernel = 1*RBF(1)
                #kernel = RBF(1)*RBF(1)                
                #kernel = RBF(2)+RBF(1) #from SE(long)+SE(short) of page 13 of Automatic Kernel Construction PhD theis  
                #kernel=1.0*RBF(1.0) + DotProduct(sigma_0=1)
                ###use different length scale for all the features
                gp = GaussianProcessClassifier(kernel=kernel)
                #search = GridSearchCV(gp, gp_param, scoring=training_score, cv=cv)
                search=grid_search_cv(gp,
                                      'gp',
                                      traindata,
                                      cv,
                                      gp_rbf_param,
                                      polynomial_degree_within_cv=polynomial_degree_within_cv,
                                      interaction_only_within_cv=interaction_only_within_cv,
                                      balanced_trainset_size_within_cv=balanced_trainset_size_within_cv,
                                      oversampling_method_within_cv=oversampling_method_within_cv,
                                      fs_select_from_model_within_cv=fs_select_from_model_within_cv,
                                      k_within_cv=k_within_cv,
                                      transform_inputs=transform_inputs,
                                      mini=mini,
                                      maxi=maxi,                                      
                                      training_score=training_score)
            elif model=='gp_rbf_with_noise':#GP with matern kernel with noise
                print('===gaussian process with RBF kernel with noise===')
                kernel=ConstantKernel(constant_value=1.0) * RBF(length_scale=1) + WhiteKernel(noise_level=1.0) #constantkernel is variance of signal (y), whitekernel is variance of noise
                gp = GaussianProcessClassifier(kernel=kernel)
                search=grid_search_cv(gp,
                                      'gp',
                                      traindata,
                                      cv,
                                      gp_rbf_with_noise_param,
                                      polynomial_degree_within_cv=polynomial_degree_within_cv,
                                      interaction_only_within_cv=interaction_only_within_cv,
                                      balanced_trainset_size_within_cv=balanced_trainset_size_within_cv,
                                      oversampling_method_within_cv=oversampling_method_within_cv,
                                      fs_select_from_model_within_cv=fs_select_from_model_within_cv,
                                      k_within_cv=k_within_cv,
                                      transform_inputs_within_cv=transform_inputs_within_cv,
                                      mini=mini,
                                      maxi=maxi,                                      
                                      training_score=training_score)
            elif model=='gp_matern':#gaussian process
                print('===gaussian process with matern kernel===')
                kernel = Matern(length_scale=1)
                gp = GaussianProcessClassifier(kernel=kernel)
                search=grid_search_cv(gp,
                                      'gp',
                                      traindata,
                                      cv,
                                      gp_matern_param,
                                      polynomial_degree_within_cv=polynomial_degree_within_cv,
                                      interaction_only_within_cv=interaction_only_within_cv,
                                      balanced_trainset_size_within_cv=balanced_trainset_size_within_cv,
                                      oversampling_method_within_cv=oversampling_method_within_cv,
                                      fs_select_from_model_within_cv=fs_select_from_model_within_cv,
                                      k_within_cv=k_within_cv,
                                      transform_inputs_within_cv=transform_inputs_within_cv,
                                      mini=mini,
                                      maxi=maxi,                                      
                                      training_score=training_score)
            elif model=='gp_matern_with_noise':#GP with matern kernel with noise
                print('===gaussian process with matern kernel with noise===')
                kernel=ConstantKernel(constant_value=1.0) * Matern(length_scale=1) + WhiteKernel(noise_level=1.0) #constantkernel is variance of signal (y), whitekernel is variance of noise
                gp = GaussianProcessClassifier(kernel=kernel)
                search=grid_search_cv(gp,
                                      'gp',
                                      traindata,
                                      cv,
                                      gp_matern_with_noise_param,
                                      polynomial_degree_within_cv=polynomial_degree_within_cv,
                                      interaction_only_within_cv=interaction_only_within_cv,
                                      balanced_trainset_size_within_cv=balanced_trainset_size_within_cv,
                                      oversampling_method_within_cv=oversampling_method_within_cv,
                                      fs_select_from_model_within_cv=fs_select_from_model_within_cv,
                                      k_within_cv=k_within_cv,
                                      transform_inputs_within_cv=transform_inputs_within_cv,
                                      mini=mini,
                                      maxi=maxi,                                      
                                      training_score=training_score)
            elif model=='linear_svm': 
                print('===linear svm===')    
                cclf = CalibratedClassifierCV(base_estimator=LinearSVC(max_iter=10000,class_weight='balanced',random_state=seed), cv=cv)
                traindata=X1.join(y1)
                search=grid_search_cv(cclf,'cclf',traindata,cv,linear_svm_param_grid,transform_inputs=transform_inputs,mini=mini,maxi=maxi,training_score=training_score)
            elif model=='poly_svm':
                    print('===polynomial kernel svm===')
                    svc=SVC(kernel='poly', probability=True, class_weight='balanced',random_state=seed)                        
                    search=grid_search_cv(svc,
                                      'svc',
                                      traindata,
                                      cv,
                                      poly_svm_param_grid,
                                      balanced_trainset_size_within_cv=balanced_trainset_size_within_cv,
                                      oversampling_method_within_cv=oversampling_method_within_cv,
                                      fs_select_from_model_within_cv=fs_select_from_model_within_cv,
                                      k_within_cv=k_within_cv,
                                      transform_inputs_within_cv=transform_inputs_within_cv,
                                      mini=mini,
                                      maxi=maxi,                                      
                                      training_score=training_score)
            elif model=='rbf_svm':    
                print('===rbf svm===')
                '''
                #tune class weights
                class0_count=y.value_counts()[0] 
                class1_count=y.value_counts()[1]
                (r,)=y.shape
                #set w0 and w1 to give balanced class_weight 
                w0 = r / (2*class0_count) #class0 is majority class   w0=0.58
                w1 = r / (2*class1_count) #class1 is minority class   w1=3.53  this ensures that w0 * class0_count = w1 * class1_count (class_weight='balanced')
                rbf_svm_param_grid['svc__class_weight']=['balanced',
                                  None,
                                  #{0: w0-0.01, 1: w1},{0: w0-0.02, 1: w1},{0: w0-0.05, 1: w1}, #keep w1 fixed and decrease w0 
                                  #{0: w0+0.01, 1: w1},{0: w0+0.02, 1: w1},{0: w0+0.05, 1: w1}, #keep w1 fixed and increase w0                              
                                  #{0: w0, 1: w1-0.01},{0: w0, 1: w1-0.02},{0: w0, 1: w1-0.05}, #keep w0 fixed and decrease w1
                                  #{0: w0, 1: w1+0.01},{0: w0, 1: w1+0.02},{0: w0, 1: w1+0.05}] #keep w0 fixed and increase w1s                                
                                  {0: w0, 1: w1+0.01},{0: w0, 1: w1+0.02},{0: w0, 1: w1+0.05},{0: w0, 1: w1+1},{0: w0, 1: w1+2},{0: w0, 1: w1+3},{0: w0, 1: 2*w1}] #keep w0 fixed and increase w1s                                
                
                print('tune class weights of SVM',rbf_svm_param_grid['svc__class_weight'])
                f=open(logfile,'a')
                f.write(str(rbf_svm_param_grid['svc__class_weight'])+'\n')
                f.close()
                '''                
                svc=SVC(kernel='rbf',probability=True, class_weight='balanced',random_state=seed)#setting balanced is: w_j = n/(k * n_j) and C_j = C * w_j where k is no. of classes; n is number of training instances; n_j is size of class j; w_j is weight of jth class, C_j is misclassificastion cost for class j. C is the penalty for misclassification. reference: Machine Learning with Python Cookbook_ Practical Solutions from Preprocessing to Deep Learning.pdf                                        
                search=grid_search_cv(svc,
                                      'svc',
                                      traindata,
                                      cv,
                                      rbf_svm_param_grid,
                                      #select_EIS_features=select_EIS_features,#select EIS features from all the features and train model on EIS features
                                      #select_demo_features=select_demo_features,#select demographics, treatment and obstetric history features from all the features and train models on the selected features
                                      balanced_trainset_size_within_cv=balanced_trainset_size_within_cv,
                                      oversampling_method_within_cv=oversampling_method_within_cv,
                                      fs_select_from_model_within_cv=fs_select_from_model_within_cv,
                                      k_within_cv=k_within_cv,
                                      transform_inputs_within_cv=transform_inputs_within_cv,
                                      mini=mini,
                                      maxi=maxi,                                      
                                      training_score=training_score)
                
            elif model=='sigmoid_svm':
                print('===sigmoid svm===')
                svc=SVC(kernel='sigmoid',probability=True,class_weight='balanced')
                traindata=X1.join(y1)
                search=grid_search_cv(svc,'svc',traindata,cv,sigmoid_svm_param_grid,transform_inputs_within_cv=transform_inputs_within_cv,mini=mini,maxi=maxi,training_score=training_score)
            elif model=='knn':
                knn = KNeighborsClassifier(n_neighbors=3)
                search = GridSearchCV(knn, knn_param, scoring=training_score, cv=cv)
                #search=grid_search_cv(knn,'knn',traindata,cv,knn_param,transform_inputs_within_cv=None,mini=None,maxi=None,training_score=training_score)
            elif model=='gaussianNB':
                nb = GaussianNB()
                search=grid_search_cv(nb,
                                      'gaussianNB',
                                      traindata,
                                      cv,
                                      nb_param,
                                      select_EIS_features=-1,#select EIS features from all the features and train model on EIS features
                                      select_demo_features=-1,#select demographics, treatment and obstetric history features from all the features and train models on the selected features
                                      balanced_trainset_size_within_cv=balanced_trainset_size_within_cv,
                                      oversampling_method_within_cv=oversampling_method_within_cv,
                                      fs_select_from_model_within_cv=fs_select_from_model_within_cv,
                                      k_within_cv=k_within_cv,
                                      transform_inputs_within_cv=transform_inputs_within_cv,
                                      mini=mini,
                                      maxi=maxi,                                      
                                      training_score=training_score)
            elif model=='decisiontree':
                dtree = DecisionTreeClassifier()
                search = GridSearchCV(dtree, dtree_param, scoring=training_score, cv=cv)           
            elif model=='rf':
                rf=RandomForestClassifier(n_jobs=-1,class_weight='balanced',n_estimators=50,random_state=seed)
                #search = GridSearchCV(rf, rf_param, scoring=training_score, cv=cv)
                search=grid_search_cv(rf,
                                      'rf',
                                      traindata,
                                      cv,
                                      rf_param,
                                      select_EIS_features=-1,#select EIS features from all the features and train model on EIS features
                                      select_demo_features=-1,#select demographics, treatment and obstetric history features from all the features and train models on the selected features                                  
                                      balanced_trainset_size_within_cv=balanced_trainset_size_within_cv,
                                      oversampling_method_within_cv=oversampling_method_within_cv,
                                      fs_select_from_model_within_cv=fs_select_from_model_within_cv,
                                      k_within_cv=k_within_cv,
                                      transform_inputs_within_cv=None,
                                      mini=None,
                                      maxi=None,                                      
                                      training_score=training_score)               
            elif model == 'mlp':
                mlp=MLPClassifier(random_state=1, max_iter=300)
                search=grid_search_cv(mlp,
                                      'mlp',
                                      traindata,
                                      cv,
                                      mlp_param_grid,
                                      balanced_trainset_size_within_cv=balanced_trainset_size_within_cv,
                                      oversampling_method_within_cv=oversampling_method_within_cv,
                                      fs_select_from_model_within_cv=fs_select_from_model_within_cv,
                                      k_within_cv=k_within_cv,
                                      transform_inputs_within_cv=transform_inputs_within_cv,
                                      mini=mini,
                                      maxi=maxi,                                      
                                      training_score=training_score,
                                      randomsearchCV=randomsearchCV,
                                      n_iter=n_iter)
            elif model=='softmaxreg':#multinomial logistic regression                                               
                softmaxreg = LogisticRegression(max_iter=10000, multi_class='multinomial', solver='lbfgs', C=1e-8,class_weight='balanced')
                search = GridSearchCV(softmaxreg, logreg_param, scoring=training_score, n_jobs=-1, cv=cv)           
            elif model=='logreg':#logistic regression
                logreg = LogisticRegression(max_iter=10000, solver='lbfgs', C=1e-8,class_weight='balanced')
                #search = GridSearchCV(logreg, logreg_param, scoring=training_score, n_jobs=-1, cv=cv)    
                search=grid_search_cv(logreg,
                                      'logreg',
                                      traindata,
                                      cv,
                                      logreg_param,
                                      select_EIS_features=select_EIS_features,#select EIS features from all the features and train model on EIS features
                                      select_demo_features=select_demo_features,#select demographics, treatment and obstetric history features from all the features and train models on the selected features
                                      polynomial_degree_within_cv=polynomial_degree_within_cv,
                                      balanced_trainset_size_within_cv=balanced_trainset_size_within_cv,
                                      oversampling_method_within_cv=oversampling_method_within_cv,
                                      fs_select_from_model_within_cv=fs_select_from_model_within_cv,
                                      k_within_cv=k_within_cv,
                                      transform_inputs_within_cv=transform_inputs_within_cv,
                                      mini=mini,
                                      maxi=maxi,                                      
                                      training_score=training_score)
            elif model=='lgbm':#light gradient boosting machine
                print('===lgbm===')    
                lgbm = lgb.LGBMClassifier(num_leaves=31)
                search = GridSearchCV(lgbm, lgbm_param, scoring=training_score, n_jobs=-1, cv=cv)
            elif model=='gb' or model=='xgb' or model=='xgbrf': #sklearn gb or xgboost or xgboost random forest
                (r,c)=traindata.shape
                cols=list(traindata.columns)
                preterm=traindata[traindata[cols[c-1]]==1]
                onterm=traindata[traindata[cols[c-1]]==0]
                (s1,_)=preterm.shape
                (s0,_)=onterm.shape
                if model=='gb':
                   gb=GradientBoostingClassifier()
                   gb_param=xgb_param
                   search = GridSearchCV(gb, gb_param, scoring=training_score, n_jobs=-1, cv=cv)
                elif model=='xgb':
                    print('===Xgboost===')               
                    #scale_pos_weight is ratio of class0 size (majority class) to class1 size (minority class)
                    #xgb=XGBClassifier(n_estimators=10,objective='multi:softmax',eval_metric='logloss',random_state=seed,n_jobs=-1)        
                    xgb=XGBClassifier(scale_pos_weight=np.sqrt(s0/s1),eval_metric='logloss', use_label_encoder=False, n_jobs=-1)#2 class dataset
                    search = GridSearchCV(xgb, xgb_param, scoring=training_score, n_jobs=-1, cv=cv)
                elif model=='xgbrf':
                    print('===xgboost random forest===')
                    xgb=XGBRFClassifier(scale_pos_weight=np.sqrt(s0/s1), use_label_encoder=False, n_jobs=-1)
                    search = GridSearchCV(xgb, xgb_param, scoring=training_score, n_jobs=-1, cv=cv)
            ###train classifier using the pipeline
            print('===training classifier===')
            if option=='preprocess':
                continue
            elif balanced_trainset_size2!=-1 or degree2!=-1 or k2!=-1 or number_of_reducts > 0: 
                if model=='voted_classifier':
                    ensemble.fit(X1,y1)
                elif model=='stacked_ensemble':
                    ensemble.fit(X1,y1)                    
                    final_estimatorSearch.fit(X1,y1)                    
                elif model=='ensemble_naive_bayes':
                    ensemble.fit(X1,y1)                    
                else:
                    search.fit(X1,y1)#cv and train classifier on pre-processed training set                 
            else:
                if model=='voted_classifier':
                    ensemble.fit(X,y)
                elif model=='stacked_ensemble':
                    ensemble.fit(X,y)                    
                    final_estimatorSearch.fit(X,y)                    
                elif model=='ensemble_naive_bayes':
                    ensemble.fit(X,y)                    
                else:
                    search.fit(X,y)#cv and train classifier on original training set                    
            if option=='preprocess':
                continue
            if model == 'stacked_ensemble' or model=='ensemble_naive_bayes':
                #trainedmodel=ensemble               
                #f=open(logfile,'a')
                #f.write(str(ensemble)+'\n')
                #f.close()
                ##CV results of combiner using out-of-sample predictions of CV of base models               
                ensemble.final_estimator=final_estimatorSearch.best_estimator_
                trainedmodel=ensemble
                best_params=final_estimatorSearch.best_params_
                print("Best parameter: ", best_params)
                print(str(cv)+"-fold CV "+str(training_score)+" of best parameter=%0.3f" % final_estimatorSearch.best_score_)
                f=open(logfile,'a')
                f.write("Best parameter: "+str(best_params)+'\n')
                f.write(str(cv)+"-fold CV "+str(training_score)+" of best parameter="+str(np.round(final_estimatorSearch.best_score_,3))+'\n')
                f.close()
                cv_results=final_estimatorSearch.cv_results_
                print("std of "+str(cv)+"-fold CV "+str(training_score)+'='+str(np.round(cv_results['std_test_score'][0],3)))
            elif model == 'voted_classifier': 
                trainedmodel=ensemble               
                f=open(logfile,'a')
                f.write(str(ensemble)+'\n')
                f.close()
            else: 
                trainedmodel=search.best_estimator_
                best_params=search.best_params_
                print("Best parameter: ", best_params)
                print(str(cv)+"-fold CV "+str(training_score)+" of best parameter=%0.3f" % search.best_score_)
                f=open(logfile,'a')
                f.write("Best parameter: "+str(best_params)+'\n')
                f.write(str(cv)+"-fold CV "+str(training_score)+" of best parameter="+str(np.round(search.best_score_,3))+'\n')
                f.close()
                cv_results=search.cv_results_
                print("std of "+str(cv)+"-fold CV "+str(training_score)+'='+str(np.round(cv_results['std_test_score'][0],3)))
                #print('cv results: ',search.cv_results_)          
            if option=='preprocess':
                continue
            if model == 'mlp':
                print('trainedmodel:',str(trainedmodel))
                mlp=trainedmodel.named_steps['mlp']
                print('no. of outputs of network: ',mlp.n_outputs_)
                print('activation of output: ',mlp.out_activation_)
            if option=='preprocess':
                continue
            else:
                modelfile=results_path+model+str(iteration)+'.joblib'
                model_inputs_output_csv=results_path+model+str(iteration)+'.model_inputs_output.csv'
                if model=='stacked_ensemble':
                    get_model_inputs_output('df',traindata,model_inputs_output_csv)       
                else:
                    get_model_inputs_output('df',X.join(y),model_inputs_output_csv)       
                dump(trainedmodel,modelfile)                    
            #p=search.get_params()
            #print(p['estimator__steps'])
            #s=p['estimator__steps']
            #print(s[0])
            #t=s[0]
            #print(t[1].get_feature_names_out())
            ####testing classifier
            if option=='preprocess':
                continue
            elif optimal_threshold:
                if option=='training_testing_on_whole_dataset':
                    test_set=pd.read_csv(dataset)                        
                elif option=='training_testing':
                   if testset_path != None:
                       if os.path.isfile(testset_path+testset_file+str(iteration)+'.csv'):
                           test_set=pd.read_csv(testset_path+testset_file+str(iteration)+'.csv')
                       else:
                           print('testset does not exist: testset='+testset_path+testset_file+str(iteration)+'.csv does not exist.')     
                   else:
                       if data_path!=None and os.path.isfile(data_path+testset_file+str(iteration)+'.csv'):
                           test_set=pd.read_csv(data_path+testset_file+str(iteration)+'.csv')
                       elif os.path.isfile(testset_file):
                           test_set=pd.read_csv(testset_file)
                       else:
                           print('testset does not exist: testset='+data_path+testset_file+str(iteration)+'.csv does not exist. '+test_set+' does not exist.')                       
                   if select_EIS_features != -1:
                      test_set = test_set[EIS_features_and_class]
                   elif select_demo_features != -1:
                      test_set = test_set[demo_features_and_class]
                elif option=='split_training_testing':
                   print('split training testing')
                cols=list(test_set.columns)
                if cols[0] == 'Identifier' or cols[0] == 'hospital_id':
                       (_,c)=test_set.shape
                       test_set=test_set.iloc[:,1:c] #removed ids column
                test_set=fill_missing_values('median','df',test_set)
                if preprocessed_trainset_file_ext!=None or degree2!=-1 or k2!=-1 or number_of_reducts > 0 or recursive_feature_elimination_cv: #rf_or_lr_feature_select!=None:####Transform the original test set to the same features of the preprocessed training set
                             test_set.to_csv(results_path+'testset_no_missing_values'+str(iteration)+'.csv',index=False)
                             reduce_data(results_path+'testset_no_missing_values'+str(iteration)+'.csv',model_inputs_output_csv,results_path+'original_testset'+str(iteration)+'.preprocessed.csv')        
                             test_set=pd.read_csv(results_path+'original_testset'+str(iteration)+'.preprocessed.csv')
                             files_to_delete.append(results_path+'testset_no_missing_values'+str(iteration)+'.csv')
                             files_to_delete.append(results_path+'original_testset'+str(iteration)+'.preprocessed.csv')
                if (model=='voted_classifier' and voting_strategy=='hard') or (model=='voted_classifier' and voting_strategy=='hard (base models=ensembles)') or (model=='ensemble_naive_bayes' and voting_strategy=='hard'):
                        #print('base_models_take_columnTransformer_selected_features='+str(base_models_take_columnTransformer_selected_features))        
                        optimalthreshold,auc,tpr,tnr,fpr,fnr,auc2,tpr2,tnr2,fpr2,fnr2 = predict_trainset_and_testset_using_sklearn_and_optimal_threshold(train_set,test_set,trainedmodel,display_info=False,ensemble_with_hard_voting=True,mini_no_of_class1_votes=mini_no_of_class1_votes,base_models_take_columnTransformer_selected_features=base_models_take_columnTransformer_selected_features,score=score,reward_scale=reward_scale,max_diff=max_diff,logfile=logfile)
                elif model=='voted_classifier' and voting_strategy=='soft' and weights=='cv_auc':
                        optimalthreshold,auc,tpr,tnr,fpr,fnr,auc2,tpr2,tnr2,fpr2,fnr2 = predict_trainset_and_testset_using_sklearn_and_optimal_threshold(train_set,test_set,trainedmodel,display_info=False,ensemble_with_hard_voting=False,ensemble_with_cv_auc_voting=True,base_models_take_columnTransformer_selected_features=base_models_take_columnTransformer_selected_features,score=score,reward_scale=reward_scale,max_diff=max_diff,logfile=logfile)
                else:
                        optimalthreshold,auc,tpr,tnr,fpr,fnr,auc2,tpr2,tnr2,fpr2,fnr2 = predict_trainset_and_testset_using_sklearn_and_optimal_threshold(train_set,test_set,trainedmodel,display_info=False,ensemble_with_hard_voting=False,base_models_take_columnTransformer_selected_features=base_models_take_columnTransformer_selected_features,score=score,reward_scale=reward_scale,max_diff=max_diff,logfile=logfile)
                f=open(logfile,'a')
                f.write('score of selecting the optimal threshold: '+score+'\n')
                f.write('optimal threshold='+str(np.round(optimalthreshold,2))+'\n')
            else:#classify training data and test data using 0.5 threshold
                ###performance on training set
                (auc,tpr,tnr,fpr,fnr)=predict_testset(False,trainedmodel,X.join(y))#predict the classes of original training set using 0.5 threshold
                print('training AUC: ',auc)    
                ###performance on test set (testset_file)
                if predict_ptb_of_each_id_using_all_spectra_of_id:
                    if os.path.isfile(data_path+testset_file+str(iteration)+'.csv'):
                        MP=mp.ModelsPredict()
                        (_,auc2,tpr2,tnr2,fpr2,fnr2)=MP.predict_using_all_spectra_of_each_patient(modelfile,
                                                                                                        model,
                                                                                                        model_inputs_output_csv,
                                                                                                        data_path+testset_file+str(iteration)+'.csv',
                                                                                                        results_path,
                                                                                                        software='sklearn',
                                                                                                        #final_prob='average of majority probs'
                                                                                                        final_prob='average of all probs'
                                                                                                        )
                    else:
                        print('testset does not exist: testset='+data_path+testset_file+str(iteration)+'.csv does not exist.')
                else:#predict PTB using each instance e.g. spectrum
                        if option=='training_testing_on_whole_dataset':
                            test_set=pd.read_csv(dataset)                        
                        elif option=='training_testing':
                            if os.path.isfile(data_path+testset_file+str(iteration)+'.csv'):                            
                                test_set=pd.read_csv(data_path+testset_file+str(iteration)+'.csv')
                            else:
                                print('testset does not exist: testset='+data_path+testset_file+str(iteration)+'.csv does not exist.')
                        if cols[0] == 'Identifier' or cols[0] == 'hospital_id':
                            (_,c)=test_set.shape
                            test_set=test_set.iloc[:,1:c] #removed ids column
                        test_set=fill_missing_values('median','df',test_set)   
                        if preprocessed_trainset_file_ext!=None or degree2!=-1 or k2!=-1 or number_of_reducts > 0 or recursive_feature_elimination_cv: #rf_or_lr_feature_select!=None:####Transform the original test set to the same features of the preprocessed training set
                            test_set.to_csv(results_path+'testset_no_missing_values'+str(iteration)+'.csv',index=False) #results_path+'trainset'+str(iteration)+'.preprocessed_inputs_output.csv'
                            reduce_data(results_path+'testset_no_missing_values'+str(iteration)+'.csv',model_inputs_output_csv,results_path+'original_testset'+str(iteration)+'.preprocessed.csv')        
                            test_set=pd.read_csv(results_path+'original_testset'+str(iteration)+'.preprocessed.csv')
                            files_to_delete.append(results_path+'testset_no_missing_values'+str(iteration)+'.csv')
                            files_to_delete.append(results_path+'original_testset'+str(iteration)+'.preprocessed.csv')
                        (auc2,tpr2,tnr2,fpr2,fnr2)=predict_testset(False,trainedmodel,test_set)#predict the classes of original test set using 0.5 threshold
                        print('test AUC: ',auc2)
                f=open(logfile,'a')
                f.write('classify data using 0.5 threshold\n')
            ###write threshold, TPR, TNR, FPR, FNR to logfile###                
            if optimal_threshold:
                f.write('performance on training set using optimal threshold: \n')
            else:
                f.write('performance on training set using 0.5 threshold:\n')
            f.write('tpr='+str(np.round(tpr,2))+'\n')
            f.write('tnr='+str(np.round(tnr,2))+'\n')
            f.write('fpr='+str(np.round(fpr,2))+'\n')
            f.write('fnr='+str(np.round(fnr,2))+'\n')
            if optimal_threshold:
                f.write('performance on test set using optimal threshold: \n')
            else:
                f.write('performance on test set using 0.5 threshold:\n')
            f.write('tpr='+str(np.round(tpr2,2))+'\n')
            f.write('tnr='+str(np.round(tnr2,2))+'\n')
            f.write('fpr='+str(np.round(fpr2,2))+'\n')
            f.write('fnr='+str(np.round(fnr2,2))+'\n')                
            f.close()
            ###summarize training aucs, testing aucs of models trained up to this iteration###
            performance=(iteration,(auc+auc2)/2,auc,auc2)
            performanceL.append(performance)
            train_aucL.append(auc)
            test_aucL.append(auc2)                
            summarize_accuracy_results(model,logfile,performanceL,train_aucL,test_aucL)
            if score == 'my_score':
                s = (tpr2+tnr2)/2 #my score: (tpr+tnr)/2 is an unbiased classification measure
            elif score == 'my_score2':
                s = (tpr2+tnr2)/2+(tpr2-tnr2)*reward_scale
            elif score == 'my_score3':
                s = (tpr2+tnr2)/2 + reward_scale * (tpr2-tnr2)/np.abs(tpr2-tnr2) * np.exp(-(tpr2-tnr2-max_diff))              
            elif score == 'G-mean':
                s = np.sqrt(tpr2*tnr2) #G-mean=sqrt(tpr*tnr) is an unbiased classification measure       
            elif score == 'youden':#youden's index = sensitivity + specificity - 1
                s = tpr+tnr-1
            if optimal_threshold:
                tpr_tnr_fpr_fnr=(iteration,s,tpr,tnr,fpr,fnr,tpr2,tnr2,fpr2,fnr2,optimalthreshold)
            else:
                tpr_tnr_fpr_fnr=(iteration,s,tpr,tnr,fpr,fnr,tpr2,tnr2,fpr2,fnr2,0.5)
            tpr_tnr_fpr_fnrL.append(tpr_tnr_fpr_fnr)
            tpr_tnr_fpr_fnrL.sort(key=operator.itemgetter(1),reverse=True)#sort models in descending order of score of test set    
            ###compute mean test TPR, mean test TNR, std of test TPR, std of test TNR of all the models trained up to this iteration###
            tpr2L=[]
            tnr2L=[]
            for tpr_tnr_fpr_fnr in tpr_tnr_fpr_fnrL:
                tpr2L.append(tpr_tnr_fpr_fnr[6])
                tnr2L.append(tpr_tnr_fpr_fnr[7])
            mean_tpr2=np.mean(tpr2L)
            mean_tnr2=np.mean(tnr2L)
            std_tpr2=np.std(tpr2L)
            std_tnr2=np.std(tnr2L)
            std_train_auc=np.std(train_aucL)
            std_test_auc=np.std(test_aucL)
            print('=== mean test TPR, mean test TNR, std of test TPR, std of test TNR, std of training AUC and std of test AUC===')
            print('mean test TPR: '+str(np.round(mean_tpr2,2))+', mean test TNR: '+str(np.round(mean_tnr2,2))+', std of test TPR: '+str(np.round(std_tpr2,2))+', std of test TNR: '+str(np.round(std_tnr2,2))+', std of training AUC: '+str(np.round(std_train_auc,2))+', std of testing AUC: '+str(np.round(std_test_auc,2)))
            print('===ranking of models by tpr and tnr of test set===')
            for t in tpr_tnr_fpr_fnrL:
                print('iteration: '+str(t[0])+', threshold='+str(np.round(t[-1],2))+', training performance: TPR='+str(np.round(t[2],2))+', TNR='+str(np.round(t[3],2))+', FPR='+str(np.round(t[4],2))+', FNR='+str(np.round(t[5],2)))
                print('\t\t testing performance: TPR='+str(np.round(t[6],2))+', TNR='+str(np.round(t[7],2))+', FPR='+str(np.round(t[8],2))+', FNR='+str(np.round(t[9],2)))
            delete_files(files_to_delete)#delete temporary files of this iteration
            if option=='training_testing_on_whole_dataset' or single_train_set:
                break
        ####write performances of all models to logfile when all iterations completed###
        f=open(logfile,'a')
        f.write('=== mean test TPR, mean test TNR, std of test TPR, std of test TNR, std of training AUC and std of test AUC===\n')
        f.write('mean test TPR: '+str(np.round(mean_tpr2,2))+', mean test TNR: '+str(np.round(mean_tnr2,2))+', std of test TPR: '+str(np.round(std_tpr2,2))+', std of test TNR: '+str(np.round(std_tnr2,2))+', std of training AUC: '+str(np.round(std_train_auc,2))+', std of testing AUC: '+str(np.round(std_test_auc,2)))
        f.write('===ranking of models by TPR and TNR of test set===\n')
        for t in tpr_tnr_fpr_fnrL:
            f.write('iteration: '+str(t[0])+', optimal threshold='+str(np.round(t[-1],2))+', training performance: TPR='+str(np.round(t[2],2))+', TNR='+str(np.round(t[3],2))+', FPR='+str(np.round(t[4],2))+', FNR='+str(np.round(t[5],2))+'\n')
            f.write('\t\t\t testing performance: TPR='+str(np.round(t[6],2))+', TNR='+str(np.round(t[7],2))+', FPR='+str(np.round(t[8],2))+', FNR='+str(np.round(t[9],2))+'\n')
        f.close()
        print('====finished running experiments====')
        
if __name__ == "__main__":
    ###===              define global variables              ===###  
    ###hyperparameters settings of classifiers###
    gammas_of_rbf_kernel=['scale','auto',2e-15,2e-13,2e-11,2e-9,2e-7,2e-5,2e-3,2e-1,0.1,2,4,8] #wide range of gammas
    #scale is using 1 / (n_features * X.var()) as gamma
    #auto is using 1 / n_features as gamma
    #gammas_of_rbf_kernel=[0.01,0.02,0.04,0.05,0.0625,0.08,0.1,0.125,0.15,0.2] #neighborhood of the best gamma found early: 0.0625
    #gammas_of_rbf_kernel=[0.01,0.05,0.1,0.5,1,1.2,1.5,1.8,2,2.2,2.5,2.8,3,3.2,3.5,3.8,4] #neighborhood of the best gamma found early: 2
    #gammas_of_rbf_kernel=[0.001,0.005,0.01,0.05,0.1,0.12,0.15,0.17,0.2,]# 0.3, 0.5, 0.7, 1, 1.2, 1.6, 2, 2.5, 3, 3.5, 4]
    gammas_of_sigmoid_kernel=['scale','auto',0.001,0.005,0.01,0.1]
    C=[0.1, 1, 1e1, 1e2, 1e3, 1e4] #C is regularization of SVM
    #C=[0.1, 0.5, 1, 5, 1e1] #neighborhood of best C found early: 1
    #C=[5000,1e4,1.5e4]
    #C=[1e4, 1.5e4, 2e4, 2.5e4, 3e4]
        
    rbf_svm_param_grid = {'svc__C': C,
                          'svc__gamma': gammas_of_rbf_kernel,
                          'svc__class_weight': [{0:3, 1:1}],#[None],#['balanced'],
                          'svc__random_state':[123456]
                        }
   
    #rbf_svm_param_grid = {'svc__C': [10000.0], 
    #                     'svc__class_weight': ['balanced'],
    #                     'svc__gamma': [2e-05],
    #                     'svc__random_state': [123456]
    #                     }
    
    #rbf_svm_param_grid = {'svc__C': [10.0],
    #                      'svc__class_weight': ['balanced'], 
    #                      'svc__gamma': [0.002],
    #                      'svc__random_state': [123456]
    #                      }

    #rbf_svm_param_grid={'svc__C': [1], 
    #                    'svc__class_weight': ['balanced'],
    #                    'svc__gamma': ['scale','auto'],
    #                    'svc__random_state':[123456]
    #                    }
       
    #rbf_svm_param_grid={'svc__C': [0.05,0.5,1], 
    #                    'svc__class_weight': ['balanced'],
    #                    'svc__gamma': ['auto','scale'],
    #                    'svc__random_state':[0,1,123456]
    #                    }
    degrees_of_poly_kernel=[3,4,5,6] #2 
    #degrees_of_poly_kernel=[3,4,5,6,7]
    #degrees_of_poly_kernel=[7,8,9,10,11,12]
 
    poly_svm_param_grid = {
                          'svc__gamma': ['auto','scale',0.1], #auto is 1/no. of features, scale is 1/(no. of features * variance)
                          'svc__C': C,
                          'svc__degree': degrees_of_poly_kernel,
                          'svc__class_weight': ['balanced']
                          }
        
    sigmoid_svm_param_grid={'svc__C': C,
                            'svc__gamma': gammas_of_sigmoid_kernel,
                            'svc__class_weight': ['balanced']
                           }
    
    linear_svm_param_grid= {'svc__base_estimator__C': C,
                            'svc__class_weight': ['balanced']
                           }       
    #MLP parameters
    #activationL=['tanh']
    activationL=['relu'] #tanh, logistic
    #activationL=['logistic'] #tanh, logistic
    #activationL=['relu','tanh','logistic']  
    #hidden_layer_sizesL=[(2,),(5,),(8,),(10,),(12,),(14,),(16,),(20,),(25,),(30,),(35,),(40,)]   
    #hidden_layer_sizesL=[(10,),(16,),(20,),(25,),(30,),(40,)]
    hidden_layer_sizesL=[(2,2),(4,4),(6,6),(10,10),(12,12),(14,14),(16,16),(20,20),(22,22),(25,25),(27,27),(30,30)]
    #hidden_layer_sizesL=[(6,6),(8,8),(10,10),(12,12),(14,14),(16,16)]
    #hidden_layer_sizesL=[(6,6),(8,8),(10,10),(12,12),(14,14),(16,16),(18,18),(20,20),(22,22),(25,25),(27,27),(30,30)]
    #hidden_layer_sizesL=[(2,2,2),(5,5,5),(10,10,10),(12,12,12),(14,14,14),(16,16,16),(20,20,20),(25,25,25),(30,30,30)]
    #hidden_layer_sizesL=[(30,6,6),(30,8,8),(30,10,10),(12,12),(14,14),(16,16)]
    #hidden_layer_sizesL=[(16,),(20,),(25,),(30,),(40,),(8,8),(12,12),(16,16),(20,20),(25,25),(30,30)]
    ###create all possible 2-layer architectures 
    #hidden_layer_sizesL=[]
    #for i in [2,5,6,8,10,12,14,16,18,20,25,28,30]:
    #    for j in [2,5,6,8,10,12,14,16,18,20,25,28,30]:
    #        hidden_layer_sizesL.append((i,j))
    ###create some randomly generated 3-layer architectures
    '''
    hidden_layer_sizesL=[]
    sizesL=[2,5,8,10,12,14,16,18,20,25,28,30,35,40]
    for i in range(200):
        indx=random.randint(0,len(sizesL)-1)
        indx2=random.randint(0,len(sizesL)-1)
        indx3=random.randint(0,len(sizesL)-1)
        hidden_layer_sizesL.append((sizesL[indx],sizesL[indx2],sizesL[indx3]))
    '''
    solverL=['adam']#,'lbfgs' ]
    #max_iterL=[2000]
    #max_iterL=[4000,8000]
    max_iterL=[8000]
    alphaL=[0, 0.0001, 0.0005, 0.005, 0.025, 0.05, 0.1, 0.2, 0.5, 1] #l2 regularization
    learning_rate_initL=[0.001]#,0.01]
    early_stopL=[True]#,False]
    random_stateL=[123456]
    mlp_param_grid={ 'mlp__activation': activationL,
                     'mlp__hidden_layer_sizes': hidden_layer_sizesL,
                     'mlp__max_iter': max_iterL,
                     'mlp__solver': solverL,
                     'mlp__alpha': alphaL,
                     'mlp__learning_rate_init': learning_rate_initL,
                     'mlp__early_stopping': early_stopL,
                     'mlp__random_state': random_stateL
                    }
    
    knn_param={'n_neighbors': [5,3],
                 'metric': ['minkowski','euclidean'],
                 'n_jobs': [-1]
                }
     
    ###parameters of GP within a pipeline###
    gp_rbf_param = {
                #'poly_features__degree': [2],
                #'poly_features__degree': [3],#4,5,6], #degree of polynomial feature within each fold of CV
                'gp__n_restarts_optimizer': [3],
                'gp__kernel__length_scale_bounds': [(1e-5,1e6),'fixed'],
                #'gp__kernel__length_scale': [1,0.5,2,3],
                #'gp__kernel__length_scale': [1e-5+0.5e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 1.5, 1e1, 1e2, 1e3, 1e4, 1e5-0.5e5], #length scale bounds: between 1e-5 and 1e5
                #'gp__kernel__length_scale':[np.array([1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,1,2,3,4,5,6,7,8,9,10,11,12,13,14]).reshape(-1,1)], #scale lengths for PTB history feature (1st feature) and 28 eis features; use with 'fmin_l_bfgs_b' or differential evolution optimizer
                #'gp__kernel__length_scale':[np.array([1,1,2,3**2,4**2,5**2,6**2,7**2,8**2,9**9,10**2,11**2,12**2,13**2,14**2,1,2,3**2,4**2,5**2,6**2,7**2,8**2,9**9,10**2,11**2,12**2,13**2,14**2]).reshape(-1,1)], #scale lengths for PTB history feature (1st feature) and 28 eis features; use with 'fmin_l_bfgs_b' or differential evolution optimizer
                #'gp__kernel__length_scale':[np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,1,2,3,4,5,6,7,8,9,10,11,12,13,14]).reshape(-1,1)], #scale lengths for 28 eis features; use with 'fmin_l_bfgs_b' or differential evolution optimizer
                #'gp__kernel__length_scale':[np.array([1,2,3**2,4**2,5**2,6**2,7**2,8**2,9**9,10**2,11**2,12**2,13**2,14**2,1,2,3**2,4**2,5**2,6**2,7**2,8**2,9**9,10**2,11**2,12**2,13**2,14**2]).reshape(-1,1)], #scale lengths for 28 eis features; use with 'fmin_l_bfgs_b' or differential evolution optimizer
                #'gp__kernel__length_scale':[np.array([1,1,1,1,1,1,1,1,2,4,5,7,8,9,10,1,1,1,1,1,1,1,1,1,1,1,3,6]).reshape(-1,1)], #scale lengths for 28 eis features; use with 'fmin_l_bfgs_b' or differential evolution optimizer
                #'gp__kernel__length_scale':[np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,28,28,28,28,28,28,28,28]).reshape(-1,1)], #scale lengths for PTB history feature (1st feature) and 28 eis features; use with 'fmin_l_bfgs_b' or differential evolution optimizer
                'gp__kernel__length_scale':[np.array([1,1,1,1,1,1,1,1,1,2,4,5,7,8,9,10,1,1,1,1,1,1,1,1,1,1,1,3,6]).reshape(-1,1)], #scale lengths for PTB history feature (1st feature) and 28 eis features; use with 'fmin_l_bfgs_b' or differential evolution optimizer
                'gp__random_state': [0,1,123456],
                'gp__optimizer':['fmin_l_bfgs_b']               
                }
    #kernel=ConstantKernel(constant_value=1.0) * RBF(length_scale=1) + WhiteKernel(noise_level=1.0) where constantkernel is variance of signal (y), whitekernel is variance of noise
   
    gp_rbf_with_noise_param = {
             'gp__kernel__k1__k1__constant_value': [0.1,0.5,1,1.5,2], #variance of signal y (constantkernel), the best variance is 0.1
             #'gp__kernel__k1__k1__constant_value': [0.01,0.05,0.1,0.2,0.3], #variance of signal y (constantkernel), search in the neighbourhood of the best variance 0.1 found previously
             'gp__kernel__k1__k2__length_scale':[1], #length scale of RBF kernel
             #'gp__kernel__k1__k2__length_scale':[np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,28,28,28,28,28,28,28,28]).reshape(-1,1)], #scale lengths for PTB history feature (1st feature) and 28 eis features; use with 'fmin_l_bfgs_b' or differential evolution optimizer
             #'gp__kernel__k1__k2__length_scale':[np.array([1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,1,2,3,4,5,6,7,8,9,10,11,12,13,14]).reshape(-1,1)], #scale lengths for PTB history feature (1st feature) and 28 eis features; use with 'fmin_l_bfgs_b' or differential evolution optimizer
             #'gp__kernel__k1__k2__length_scale':[np.array([1,1,2,3**2,4**2,5**2,6**2,7**2,8**2,9**9,10**2,11**2,12**2,13**2,14**2,1,2,3**2,4**2,5**2,6**2,7**2,8**2,9**9,10**2,11**2,12**2,13**2,14**2]).reshape(-1,1)], #scale lengths for PTB history feature (1st feature) and 28 eis features; use with 'fmin_l_bfgs_b' or differential evolution optimizer
             #'gp__kernel__k1__k2__length_scale':[np.array([5,7,1,6,4,2,8,3]).reshape(-1,1)], #scale lengths for metabolite data (8 features) 
             'gp__kernel__k1__k2__length_scale_bounds': [(1e-7,1e6)],
             'gp__kernel__k2__noise_level': [0.001,0.1,0.5,1,1.5], #variance of noise (white kernel)
             'gp__n_restarts_optimizer': [3],
             'gp__random_state': [123456],
             'gp__optimizer':['fmin_l_bfgs_b']               
            }
    
    gp_polynomial_param = {'kernel__kernel__sigma_0': [0, 0.1, 1, 1.5],
                           'kernel__exponent':[2,3,4]
                          }
    #parameter of Matern kernel within a pipeline
    gp_matern_param = {
                #'poly_features__degree': [2],#3,4,5,6], #degree of polynomial feature within each fold of CV
                'gp__n_restarts_optimizer': [1],
                'gp__random_state': [123456],
                'gp__kernel__length_scale': [1],
                #'gp__kernel__length_scale': [1e-5+0.5e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 1.5, 1e1, 1e2, 1e3, 1e4, 1e5-0.5e5],
                #'gp__kernel__length_scale': [1e-2, 1e-1, 0.5, 1], #length scale tunes the difference between xi and xj in the kernel; setting length scale to < 1 increases the difference between xi and xj in the kernel and vice versa
                #'gp__kernel__length_scale_bounds': [(1e-5,2)],
                #'gp__kernel__length_scale':[1e-2, 1e-1, 0.5, 1],
                #'gp__kernel__length_scale_bounds':['fixed'],#if fixed, length scale is not tuned during training.
                #'gp__kernel__length_scale': [100],
                #'gp__kernel__length_scale':[np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(-1,1)], #scale lengths for 29 features; use with differential evolution optimizer
                #'gp__optimizer':["fmin_diff_evolution"],
                'gp__optimizer':['fmin_l_bfgs_b'],
                'gp__kernel__nu': [0.5,1.5,2.5, np.inf] #nu=inf is equivalent to RBF kernel 
                }
    
    #kernel=ConstantKernel(constant_value=1.0) * Matern(length_scale=1) + WhiteKernel(noise_level=1.0) where constantkernel is variance of signal (y), whitekernel is variance of noise
    gp_matern_with_noise_param = {
             'gp__kernel__k1__k1__constant_value': [0.1,0.5,1,1.5,2], #variance of signal y (constantkernel)
             'gp__kernel__k1__k2__nu': [0.5,1.5,2.5], #nu of matern kernel
             'gp__kernel__k1__k2__length_scale':[1], #length scale of matern kernel
             'gp__kernel__k2__noise_level': [0.001,0.1,0.5,1,1.5], #variance of noise (white kernel)
             'gp__n_restarts_optimizer': [1],
             'gp__random_state': [123456],
             'gp__optimizer':['fmin_l_bfgs_b']               
            }
    
    '''
    #kernel=RBF()+whitekernel
    gp_param = {'n_restarts_optimizer': [3],
                #'kernel__k1__k2__length_scale':[0.01,0.03,0.05,0.1,0.2,0.5,1,1.5,2], #length scale of 1.0*RBF(1.0) kernel
                'kernel__k2__noise_level': [0.01,0.05,0.1,0.2,0.5,1,1.2,1.5,2], #noise level of white kernel; kernel=RBF()+whitekernel
                'random_state': [0]
                }
    '''
    nb_param = {'gaussianNB__var_smoothing': np.logspace(0,-9, num=10)
                #'gaussianNB__var_smoothing': np.logspace(0,9, num=10)
                }
    
    logreg_param={'logreg__C': [0,1e-8,0.001,0.003,0.005,0.007,0.009,0.1,0.3,0.5,0.7,0.9,1]#+list(np.logspace(0,4,10)) #Inverse of regularization strength, smaller values specify stronger regularization
                  #'logreg__C': [1] #default
                 } 
    
    ###parameters of rf within a pipeline   
    #trees_of_rf=[15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 170, 200]
    trees_of_rf=[20, 30, 50, 60, 70, 80, 90, 100, 120, 150, 170, 200]
    rf_param = {'rf__n_estimators': trees_of_rf,
                     #'min_samples_split': [30, 40, 50],
                     #'min_samples_leaf': [10, 20, 30, 40, 50],                     
                'rf__class_weight': ['balanced'],
                #'rf__random_state': [0,1,123456789],
                'rf__n_jobs': [-1]
               }
    '''
    rf_param = {'n_estimators': trees_of_rf,
                     #'min_samples_split': [30, 40, 50],
                     #'min_samples_leaf': [10, 20, 30, 40, 50],                     
                'class_weight': ['balanced'],
                'random_state': [0,1],
                'n_jobs': [-1]
               }
    '''    
    dtree_param = { 'criterion': ['gini', 'entropy'],
                    'class_weight': ['balanced',None]
                  }
    xgb_param = {'eta': [0.5, 1, 2, 3], #learning rate: step size shrinkage used in update to prevents overfitting. 
                 'reg_lambda': [0, 1e-3, 1e-2, 0.1, 1, 10], #L2 regularization term on weights. Increasing this value will make model more conservative.
                 #'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],#to speed up training on very high dimensionality
                 'early_stopping_rounds': [5, 10], #seems to generate warning messages
                 #'n_estimators': [10,15,20,30],
                 'n_estimators': [10,15,20,30,50,80,100,150,200,300],
                 #'n_estimators': [30,50,80,100,150,200,300],#,500,600,800,1000],
                 #'n_estimators': [200,400,600,800,1000,1500,2000],
                 'max_depth': [3,4],#5,6],
                 'random_state': [123456]
                 }
    lgbm_param = {
                    'num_leaves': [31,40,50,60,70],
                    'learning_rate': [0.01, 0.1, 1],
                    'n_estimators': [20, 40, 80, 100]
                 }
        
    stacked_ensemble_param={
                            #logistic regression                
                            #'final_estimator__C': [1e-8] #[0,1e-8,0.001,0.003,0.005,0.007,0.009,0.1,0.3,0.5,0.7,0.9,1]#+list(np.logspace(0,4,10)) #C of logistic regression combiner
                            #gaussian processes
                            #'final_estimator__gp__kernel__k1__k1__constant_value': [0.1],#0.5,1,1.5,2], #variance of signal y (constantkernel), the best variance is 0.1
                            #'final_estimator__gp__kernel__k1__k2__length_scale':[1], #length scale of RBF kernel
                            #'final_estimator__gp__kernel__k1__k2__length_scale_bounds': [(1e-7,1e6)],
                            #'final_estimator__gp__kernel__k2__noise_level': [0.001], #0.1,0.5,1,1.5], #variance of noise (white kernel)
                            #'final_estimator__gp__n_restarts_optimizer': [3],
                            #'final_estimator__gp__random_state': [123456],
                            #'final_estimator__gp__optimizer':['fmin_l_bfgs_b']
                            #gaussian naive bayes
                            #'final_estimator__gaussianNB__var_smoothing': [1e-05] #naive bayes (this parameter setting is the one of the corresponding base model)
                            'final_estimator__var_smoothing': [1e-5]#,1e-9] #naive bayes (this parameter setting is the one of the corresponding base model)
                            #svm
                            #'final_estimator__svc__C': [1000.0], #svm (this parameter setting is the one of the corresponding base model)
                            #'final_estimator__svc__gamma': [2e-05]
                            #mlp
                            #'final_estimator__mlp__hidden_layer_sizes': [(6, 6)] #mlp (this parameter setting is the one of the corresponding base model)
                           }
    
    #cv=10 
    cv=5
    #cv=3
    #model='softmaxreg'
    #model='logreg'
    #model='mlp'
    #model='linear_svm'
    #model='poly_svm'
    #model='rbf_svm'
    #model='sigmoid_svm'
    #model='gb' #sklearn gradient boosting
    #model='xgb'
    #model='xgbrf'
    #model='lgbm'
    model='rf'
    #model='knn'
    #model='rnn' #radius nearest neighbor    
    #model='gaussianNB'
    #model='gp_rbf' #gaussian process with rbf kernel
    #model='gp_rbf_with_noise'
    #model='gp_matern' #gaussian process with matern kernel
    #model='gp_matern_with_noise'
    #model='gp_polynomial' #gaussian process with polynomial kernel
    #model='decisiontree'
    #model='stacked_ensemble'
    #model='voted_classifier'
    base_models_take_columnTransformer_selected_features=False
    #model='ensemble_naive_bayes'
    combiner='svc'
    voting_strategy='soft' #soft weights
    #voting_strategy='hard' #prob of class1 is based on the votes of class labels output by the base models 
    #voting_strategy='hard (base models=ensembles)'
    #mini_no_of_class1_votes='optimize'
    mini_no_of_class1_votes=1 #minority vote, mini_no_of_class1_votes=1,2,..,(no. of models)/2 - 1                         
    #mini_no_of_class1_votes='majority' #majority vote, threshold=0.5                          
    #===base models of ensemble for EIS + PTB history ===
    svc_model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\rbfsvm (zscore)_5\\rbf_svm0.joblib"
    gp_model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\gp_with_noise\\gp_rbf_with_noise0.joblib"               
    mlp_model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\mlp_2layers_5\\mlp0.joblib"
    nb_model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\gaussianNB_3\\gaussianNB0.joblib"              
    #rf_model="C:\\Users\\uos\\EIS preterm prediction\\results\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\rf_3\\rf0.joblib"
    svc_cv_auc=0.84
    gp_cv_auc=0.85
    mlp_cv_auc=0.87
    nb_cv_auc=0.82
    #rf_cv_auc=0.81
    #weights=[1, 1, 1, 1]     
    #===base models of ensemble for EIS data ===
    #svc_model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_5\\rbf_svm0.joblib"
    #gp_model='C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gp_with_noise\\gp_rbf_with_noise0.joblib'
    #mlp_model='C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\mlp_2_layers_(zscore)\\mlp0.joblib'
    #nb_model='C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gaussianNB\\gaussianNB0.joblib'
    #svc_cv_auc=0.78
    #gp_cv_auc=0.77
    #mlp_cv_auc=0.80
    #nb_cv_auc=0.69
    #weights=[1, 1, 1, 1] 
    weights='cv_auc' #'weight of a model = CV AUC of model/sum of CV AUCs of models'
                     # prob of PTB = weight1 x prob1 + weight1 x prob2 +...+weight_k x prob_k where weight1 is weight of model1, prob1 is prob of PTB of model1 etc.')
    #weights='cv_auc2' #weight of a model = CV AUC of model/ smallest CV AUC of models
                      #prob of ensemble = (weight1 x prob1 + weight1 x prob2 +...+weight_k x prob_k)/k
    ##base models are ensembles
    #ensemble1="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\soft voting_weights=1\\voted_classifier0.joblib"
    #ensemble2="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\stacked_ensemble_svc\\stacked_ensemble0.joblib"
    #ensemble3="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\stacked_ensemble_mlp\\stacked_ensemble0.joblib"
    #ensemble4="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\stacked_ensemble_gp\\stacked_ensemble0.joblib"
    #ensemble5="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\stacked_ensemble_nb\\stacked_ensemble0.joblib"
    #combiner='nb'
    #svc_cv_auc=0.78
    #gp_cv_auc=0.77
    #mlp_cv_auc=0.80
    #nb_cv_auc=0.69
    #weights='cv_auc' #weight=CV AUC of model / smallest CV AUC    
    #weights=[1, 1, 1, 1]     
    #weights='cv_auc' #'weight of a model = CV AUC of model/sum of CV AUCs of models'
                     # prob of PTB = weight1 x prob1 + weight1 x prob2 +...+weight_k x prob_k where weight1 is weight of model1, prob1 is prob of PTB of model1 etc.')
    #weights='cv_auc2' #weight of a model = CV AUC of model/ smallest CV AUC of models
                      #prob of ensemble = (weight1 x prob1 + weight1 x prob2 +...+weight_k x prob_k)/k
    ###parameters setting of pipeline###
    #score='youden' #youden's index = sensitivity + specificity - 1
    score='my_score' #my score = (tpr+tnr)/2
    #score='my_score2' #my score2 favours higher TPR than TNR, my score2=(tpr+tnr)/2+(tpr-tnr)*reward_scale where (tpr-tnr) is a reward (+ve) if tpr > tnr and is a penalty (-ve) if tpr < tnr
    #score='my_score3' #my score3 = (tpr+tnr)/2 + reward_scale * (tpr-tnr)/|tpr-tnr| * e^-(tpr-tnr-max difference); 
                      #favours threshold with higher TPR than TNR at most max difference between TPR and TNR; 
                      #reward_scale*(tpr-tnr) * e^-(tpr-tnr-max difference) is reward; 
                      #reward scale controls the contribution of reward
                      #(tpr-tnr)/|tpr-tnr| gives positive reward when TPR > TNR; negative reward when TPR < TNR and 0 reward when TPR=TNR; 
                      #e^-(tpr-tnr-max difference) gives reward when TPR-TNR <= max difference; gives maximum reward when TPR < TNR 
    max_diff=0.3
    reward_scale=1/4 
    #reward_scale=1/3
    #reward_scale=1/2
    #reward_scale=1/5
    #reward_scale=1/6
    #score='G-mean' #G-mean = sqrt(tpr*tnr)
    optimal_threshold=True
    #optimal_threshold=False
    #predict_ptb_of_each_id_using_all_spectra_of_id=True
    predict_ptb_of_each_id_using_all_spectra_of_id=False
    #remove_duplicates_from_valid_set=False
    remove_duplicates_from_valid_set=True
    #print('remove_duplicates_from_valid_set:',remove_duplicates_from_valid_set)    
    test_size=0.34
    #test_size=0.3
    #test_size=0.2
    #test_size=0.15
    #test_size=0.5
    #test_size=0.7
    #option='training_testing_on_whole_dataset'
    #option='split_training_testing'
    option='training_testing' #use already created training sets and test sets 
    #option='preprocess'
    if option=='split_training_testing':
        data_path=None
        trainset_file=None
        testset_file=None
        preprocessed_trainset_file_ext=None
    elif option=='training_testing':
        dataset=None
        data_path=None
        #trainset_file="U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_ids.csv"
        trainset_file="U:\\EIS preterm prediction\\my_filtered_data_28inputs_no_treatment_with_ids.csv"
        #trainset_file='trainset'
        preprocessed_trainset_file_ext=None
        #preprocessed_trainset_file_ext='_preprocessed_balanced_trainset_size=-1_degree=-1_info_gain_selected_features=-1_ga_cv_of_gaussianNB_selected_features.csv'#ga cv of naive bayes    
        #preprocessed_trainset_file_ext='_preprocessed_balanced_trainset_size=2000_degree=-1_info_gain_selected_features=14.csv'    
        #preprocessed_trainset_file_ext='_preprocessed_balanced_trainset_size=2000_degree=-1_info_gain_selected_features=-1_ga_cv_of_rf_selected_features.csv'    #rswr+ga cv of random forest
        #preprocessed_trainset_file_ext='_preprocessed_balanced_trainset_size=1000_degree=-1_info_gain_selected_features=-1_ga_cv_of_gaussianNB_selected_features.csv'#rswr+ga cv of naive bayes    
        #testset_file='testset_good_readings'
        #testset_file='valset'
        #testset_file='testset' 
        #testset_file="U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_treated_with_ids.csv"
        testset_file="U:\\EIS preterm prediction\\my_filtered_data_28inputs_treated_with_ids.csv"
        #trainset_file='trainset_encoded_zscore_normalized_relu'
        #testset_file='testset_encoded_zscore_normalized_relu'        
        #trainset_file='trainset_encoded_zscore_normalized_elu'
        #testset_file='testset_encoded_zscore_normalized_elu'
        #trainset_file='trainset_encoded_zscore_normalized'
        #testset_file='testset_encoded_zscore_normalized'
        #trainset_file='trainset_encoded_mini_max_normalized'
        #testset_file='testset_encoded_mini_max_normalized' 
        #trainset_file='trainset_encoded_mini_max_normalized_relu'
        #testset_file='testset_encoded_mini_max_normalized_relu'
        #trainset_file='trainset_encoded_mini_max_normalized_relu_balanced_trainset_size=1000_'
        #testset_file='testset_encoded_mini_max_normalized_relu_balanced_trainset_size=1000_'
        #testset_file='testset_of_normal_data_and_outliers'
        #testset_file='testset_normal_data_plus_outliers'
    elif option=='preprocess':#prepocess training set files 
        dataset=None
        trainset_file='trainset'
        testset_file=None
        preprocessed_trainset_file_ext=None
    elif option=='training_testing_on_whole_dataset':
        data_path=None
        trainset_file=None
        testset_file=None
        preprocessed_trainset_file_ext=None
    iterations=100
    #iterations=1
    #iterations=5
    ###To use all features to train a classifier, select_demo_features=-1 and select_EIS_features=-1
    select_demo_features=-1
    select_EIS_features=-1
    #select_EIS_features=['Amplitude1','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7','Amplitude8','Amplitude9','Amplitude10','Amplitude11','Amplitude12','Amplitude13','Amplitude14','Phase1','Phase2','Phase3','Phase4','Phase5','Phase6','Phase7','Phase8','Phase9','Phase10','Phase11','Phase12','Phase13','Phase14']
    #select_demo_features=['parous_with_1_or_more_preterm_delivery','parous_with_1_or_more_term_delivery','nulliparous_with_no_pregnancy','nulliparous_with_previous_miscarriage','nulliparous_due_to_other_causes','parous_with_previous_miscarriage','Ethnicity','Age','Smoker','Alcohol_in_pregnancy','Non_prescribed_drugs_in_pregnancy','BMI','Number_of_previous_pregnancies','number_previous_early_miscarriages','number_previous_TOPs','colpCell','no_preterm_birthsCell','no_term_birthsCell','no_MTLCell','cervical_cerclageCell','progesteroneCell','tocolysisCell','visits_steroidsCell']
    randomsearchCV=False #use exhaustive search CV instead of random search CV
    #randomsearchCV=True, #use random search CV instead of exhaustive search CV
    n_iter=200 #number of parameter settings to search during random search CV
    #===perform data preprocessing before CV and training classifier ===#
    ###oversampling -> polynomial features construction -> information gain feature selection of Workflow1
    #oversampling_and_poly_features_construct_and_info_gain_fs=True #use workflow1 to oversample, construct polynomial features and select features by information gain
    #oversampling_and_poly_features_construct_and_info_gain_fs=False #use workflow1 to oversample, construct polynomial features and select features by information gain
    balanced_trainset_size2=-1
    #balanced_trainset_size2='oversample class1 to size of class0' #this is 'auto' option of smote
    #balanced_trainset_size2='undersample class0 to size of class1 with replacement' 
    #balanced_trainset_size2='undersample class0 to size of class1 without replacement'
    #balanced_trainset_size2=1515 #size of balanced dataset when whole dataset is used as training and testing sets
    #balanced_trainset_size2=400#combination of oversampling and undersampling: 
                               #If set balanced training set size to N where size of preterm class < N/2 < size of onterm class (majority class), 
                               #oversample preterm class to N/2  
                               #and undersample onterm class to N/2 e.g. N=300.
    #balanced_trainset_size2=500 #if balanced_trainset_size2 > size of training set, this is 'all' option of smote
    #balanced_trainset_size2=600
    #balanced_trainset_size2=800
    #balanced_trainset_size2=1000
    #balanced_trainset_size2=1200
    #balanced_trainset_size2=1500
    #balanced_trainset_size2=1984
    #balanced_trainset_size2=2000
    #oversampling_method='oversample_class1_and_class0_separately_using_repeated_bootstraps'
    #oversampling_method='smote'
    oversampling_method='random_sampling_with_replacement'
    #oversampling_method='borderline_smote'
    #oversampling_method='adasyn'
    degree2=-1 #construct polynomial features with the degree
    #degree2=2
    #degree2=3
    #degree2=4
    #interaction_only=True
    interaction_only=False
    k2=-1 #no. of feature to select using information gain after polynomial features construction
    #k2=30
    #k2=150
    #k2=2
    #k2=7
    #k2=14
    #k2=int(32*3/4) #select the most informative 3/4 of features
    #k2=int(32/2) #select the most informative 1/2 of features
    #k2=int(8/2)
    #k2=1000
    number_of_reducts=0 #no. of feature subsets output by GA CV feature selection from information gain-reduced training data
    #number_of_reducts=30 #no. of feature subsets output by GA CV feature selection
    #number_of_reducts=40 #no. of feature subsets output by GA CV feature selection
    #number_of_reducts=5 #no. of feature subsets output by GA CV feature selection
    recursive_feature_elimination_cv=False #run rfe cv instead of GA feature selection 
    #recursive_feature_elimination_cv=True
    if recursive_feature_elimination_cv:
        number_of_reducts=0
    ###estimator of recursive feature elimination CV 
    #estimator_of_rfe_cv='decisiontree'
    #estimator_of_rfe_cv='gaussianNB'
    estimator_of_rfe_cv='rf'
    #estimator_of_rfe_cv='knn'
    #estimator_of_rfe_cv='ARD' #Bayesian ARD regression
    ###classifier of select_optimal_feature_subset_using_cv which is called if number_of_reducts > 0. This implements GA CV feature selection
    #model_of_select_optimal_feature_subset_using_cv='gp'
    model_of_select_optimal_feature_subset_using_cv='gaussianNB'
    #model_of_select_optimal_feature_subset_using_cv='rf'
    #model_of_select_optimal_feature_subset_using_cv='logreg'    
    #===perform data preprocessing during each fold of CV===
    polynomial_degree_within_cv=-1
    #polynomial_degree_within_cv=2
    interaction_only_within_cv=False
    balanced_trainset_size_within_cv=-1
    #balanced_trainset_size_within_cv=400#combination of oversampling and undersampling: 
                               #If set balanced training set size to N where size of preterm class < N/2 < size of onterm class (majority class), 
                               #oversample preterm class to N/2  
                               #and undersample onterm class to N/2 e.g. N=300.
    #balanced_trainset_size_within_cv=500
    #balanced_trainset_size_within_cv=600
    #balanced_trainset_size_within_cv=700
    #balanced_trainset_size_within_cv=800
    #balanced_trainset_size_within_cv=1000
    #balanced_trainset_size_within_cv=1200
    #balanced_trainset_size_within_cv=1500
    #balanced_trainset_size_within_cv=2000
    #balanced_trainset_size_within_cv='all' #resample both classes
    #balanced_trainset_size_within_cv='minority'#only resample the minority class
    #balanced_trainset_size_within_cv='bootstrap' #a bootstrap is a resampled training set of the same size as the training set
    oversampling_method_within_cv=-1
    #oversampling_method_within_cv='smote'
    #oversampling_method_within_cv='svmsmote'
    #oversampling_method_within_cv='kmeanssmote'
    #oversampling_method_within_cv='random_sampling_with_replacement'
    #oversampling_method_within_cv='borderline_smote'
    #oversampling_method_within_cv='adasyn'
    #fs_select_from_model_within_cv=True #select features from decision tree within each fold of CV
    fs_select_from_model_within_cv=False #select features from decision tree within each fold of CV
    k_within_cv=-1 #select k features using information gain within each fold of CV
    #k_within_cv=14
    #k_within_cv=20    
    #k_within_cv=30
    #===Normalize inputs as a preprocessing step during each fold of CV===#
    transform_inputs_within_cv=None
    #transform_inputs_within_cv='zscore' #more suitable for rbf kernel and polynomial kernel
    #transform_inputs_within_cv='minmax' #more suitable for sigmoid kernel
    #mini=-1 #lower bound of minmax scaler
    mini=0 #lower bound of minmax scaler
    maxi=1 #upper bound of minmax scaler
    variance_threshold_within_cv=0 #remove features with variance < variance threshold
    #training_score='balanced_accuracy' #the average of recall obtained on each class using threshold 0.5
    #training_score='f1' #F1 = 2 * (precision * recall) / (precision + recall); 
                        #0.5 threshold is used compute precision and recall; 
                        #The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.
    training_score='roc_auc'
    #training_score='recall'
    weka_path='U:\\EIS preterm prediction\\weka-3-7-10.jar'
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
    warnings.filterwarnings('ignore')
    #====path of the whole dataset===
    #dataset="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filters from sharc\\selected_unselected_eis_readings\\trainset0.csv"
    #dataset="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\trainset0.csv"
    #dataset="U:\\EIS preterm prediction\\438_V1_demographics_treatment_history_obstetric_history_with_ids.csv"
    #dataset="438_V1_28inputs_and_438_V1_demographics_treatment_history_obstetric_history_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_438_V1_demographics_treatment_history_obstetric_history_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\438_V1_demographics_treatment_history_obstetric_history_with_ids.csv"
    dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_no_treatment_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_previous_pregnancy_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_no_treatment_with_previous_pregnancy_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\have_ptb_history_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\have_ptb_history_no_treatment_with_ids.csv"
    #dataset="D:\\EIS preterm prediction\\i4i MIS\\raw data\\mis_data_C1C2C3_no_missing_labels.csv"
    #dataset='u:\\EIS preterm prediction\\my_filtered_data_28inputs_with_ids.csv'
    #dataset='d:\\EIS preterm prediction\\my_selected_unselected_eis_readings.csv'
    #dataset="D:\\EIS preterm prediction\\my_filtered_data_unselected_eis_readings.csv"
    #dataset="U:\\EIS preterm prediction\\EIS_Data\\EIS_Data\\438_V1_28inputs.csv"
    #dataset="U:\\EIS preterm prediction\\438_V1_28inputs_no_treatment.csv"
    #dataset="mean_of_438_V1_4_eis_readings_28inputs_no_treatment.csv"
    #dataset="mean_of_438_V1_4_eis_readings_28inputs_no_treatment_real_parts_only.csv"
    #dataset="cl_ffn_V1.csv"
    #dataset="cl_ffn_V1_no_treatment.csv"
    #dataset="u:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\data_log10_highest_info_gain.csv"
    #dataset="u:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs_log_transformed.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_1input_log_transformed.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs_outliers_removed_as_blanks_log_transformed.csv"
    #dataset="U:\\EIS preterm prediction\\438_V1_obstetric_history_2_parous_features.csv"
    #dataset="U:\\EIS preterm prediction\\438_V1_obstetric_history.csv"    
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_3_best_features_log_transformed.csv"
    #===path of training sets and test sets===
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_and_438_V1_demographics_treatment_history_obstetric_history\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\5-fold cv\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_previous_pregnancy\\5-fold cv\\"
    #data_path="c:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\5-fold cv\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\ga_cv_of_gaussianNB_selected_features\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_with_ids\\train66test34\\" 
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_with_ids\\train66test34\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train80test20\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train80test20\\ga_cv_of_gaussianNB_selected_features\\"
    #data_path="G:\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\"
    #data_path="F:\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\"
    #data_path="F:\\EIS preterm prediction\\results\\my_filtered_data_28inputs_with_ids\\train66test34\\"
    #data_path="F:\\EIS preterm prediction\\results\\workflow1\\filter from sharc\\selected_unselected_eis_readings\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filters from sharc\\selected_unselected_eis_readings\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filters from sharc\\selected_unselected_eis_readings\\random_sampling_with_replacement\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\"
    #data_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_438_V1_demographics_treatment_history_obstetric_history\\train66test34\\"    
    #testset_path="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter_and_438_V1_demographics_treatment_history_obstetric_history\\train66test34\\"
    #===path of results file and logfile===
    #results_path="U:\\EIS preterm prediction\\working papers\\journal of biomedical signal processing and control\\revision\\k-fold cv\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_and_438_V1_demographics_treatment_history_obstetric_history\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\438_V1_demographics_treatment_history_obstetric_history_with_ids_boolean_features\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_438_V1_demographics_treatment_history_obstetric_history\\train66test34\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_demographics_treatment_history_obstetric_history\\"
    #results_path="U:\\EIS preterm prediction\\results\\438_V1_28inputs\\"
    #results_path="U:\\EIS preterm prediction\\results\\438_V1_28inputs_no_treatment\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\have_ptb_history_no_treatment\\"
    #results_path="U:\\EIS preterm prediction\\results\\mean_of_438_V1_4_eis_readings_28inputs_no_treatment_real_parts_only\\"
    results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\5-fold cv\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_previous_pregnancy\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\5-fold cv\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_previous_pregnancy\\5-fold cv\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment_with_previous_pregnancy\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history\\"
    #results_path="F:\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\"
    #results_path="F:\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\asymp_22wks_438_V1_8inputs\\train80test20\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\asymp_22wks_438_V1_8inputs\\"    
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\asymp_22wks_438_V1_8inputs_log_transformed\\train80test20\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\asymp_22wks_438_V1_8inputs_log_transformed\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\sklearn_pipeline\\selected_unselected_eis_readings\\"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\sklearn_pipeline\\"
    #results_path="F:\\EIS preterm prediction\\results\\sklearn_pipeline\\"
    #results_path="F:\\EIS preterm prediction\\results\\my_filtered_data_28inputs_with_ids\\"
    #results_path="F:\\EIS preterm prediction\\results\\have_ptb_history_with_ids\\"
    #results_path="F:\\EIS preterm prediction\\results\\sklearn_pipeline\\selected_unselected_eis_readings\\"
    #results_path="F:\\EIS preterm prediction\\results\\sklearn_pipeline\\obstetric_treatment_history\\"
    #results_path="F:\\EIS preterm prediction\\results\\sklearn_pipeline\\obstetric_history\\"
    #results_path="F:\\EIS preterm prediction\\results\\sklearn_pipeline\\selected_unselected_eis_readings\\autoencoder_reduced\\"
    create_folder_if_not_exist(results_path)
    #results_path=results_path+'rbfsvm (minmax)\\'
    #results_path=results_path+'rbfsvm (minmax)_2\\'
    #results_path=results_path+'rbfsvm (minmax)_bootstrap\\'
    #results_path=results_path+'rbfsvm (minmax)_my_score2_2\\'
    #results_path=results_path+'rbfsvm (zscore)\\' 
    #results_path=results_path+'rbfsvm (zscore)_6\\'
    #results_path=results_path+'rbfsvm (zscore)_bootstrap\\'
    #results_path=results_path+'rbfsvm (zscore)_my_score2\\'
    #results_path=results_path+'rbfsvm (zscore)_my_score3\\'
    #results_path=results_path+'rbfsvm (zscore)_youden\\' 
    #results_path=results_path+'rbfsvm (zscore)_G-mean\\' 
    #results_path=results_path+'poly_svm (zscore)\\' 
    #results_path=results_path+'poly_svm (minmax)\\'
    #results_path=results_path+'gp\\'
    #results_path=results_path+'gp_with_different_length_scale_4\\'
    #results_path=results_path+'gp_with_noise_2\\'
    #results_path=results_path+'gp_with_noise_2\\'    
    #results_path=results_path+'gp_with_noise_3\\'    

    #results_path=results_path+'gp_with_noise_and_different_length_scale\\'
    #results_path=results_path+'gp_polynomial features (degree 2)\\'
    #results_path=results_path+'gp_polynomial features (degree 2)_2\\'
    #results_path=results_path+'gp_matern\\'
    #results_path=results_path+'gp_matern_2\\'
    #results_path=results_path+'gp_matern (fixed length scale)\\'
    #results_path=results_path+'gp_matern (differential evolution)\\'
    #results_path=results_path+'gp_matern (fmin_l_bfgs_b)\\'
    #results_path=results_path+'gp_matern_polynomial features (degree 2)\\'
    #results_path=results_path+'gp_matern_with_noise\\'
    results_path=results_path+'rf\\'
    #results_path=results_path+'rf_2\\'
    #results_path=results_path+'gaussianNB\\'
    #results_path=results_path+'gaussianNB_3\\'
    #results_path=results_path+'logreg\\'
    #results_path=results_path+'logreg_2\\'
    #results_path=results_path+'xgb\\'
    #results_path=results_path+'xgb_2\\'
    #results_path=results_path+'lgb\\'
    #results_path=results_path+'knn\\'
    #results_path=results_path+'mlp\\'
    #results_path=results_path+'mlp_1_layer (zscore)\\'
    #results_path=results_path+'mlp_1_layer (minmax)\\' 
    #results_path=results_path+'mlp_2_layers_(zscore)_2\\'
    #results_path=results_path+'mlp_3_layers_(zscore)\\'
    #results_path=results_path+'mlp_3_layers_(minmax)_2\\'
    #results_path=results_path+'mlp_2layers_6\\' 
    #results_path=results_path+'hard voting\\'
    #results_path=results_path+'hard voting_2\\'
    #results_path=results_path+'hard voting (base models=ensembles)\\'
    #results_path=results_path+'soft voting (5 models)\\'
    #results_path=results_path+'soft voting_(gridsearchCV of each base model)\\'
    #results_path=results_path+'soft voting_(gridsearchCV of each base model)_f1\\'
    #results_path=results_path+'soft voting_(G-mean score)\\'
    #results_path=results_path+'soft voting_(youden score)\\' 
    #results_path=results_path+'soft voting_(my_score2)\\' 
    #results_path=results_path+'soft voting_weights=1\\' 
    #results_path=results_path+'soft voting_cv_auc_weights\\'  
    #results_path=results_path+'soft voting_custom_weights\\' 
    #results_path=results_path+'stacked_ensemble_log_regression\\'    
    #results_path=results_path+'stacked_ensemble_log_regression2\\'
    #results_path=results_path+'stacked_ensemble_gp\\'    
    #results_path=results_path+'stacked_ensemble_nb2\\'   
    #results_path=results_path+'stacked_ensemble_svc2\\'
    #results_path=results_path+'stacked_ensemble_mlp\\'
    #results_path=data_path
    #results_path=data_path+'gp_product_of_rbf_and_rbf\\'
    #results_path=data_path+'gp_sum_of_rbf_and_rbf\\'
    #results_path=data_path+'gp_matern_kernel2\\'         
    #results_path=data_path+'knn4\\'
    #results_path=data_path+'ensemble_naive_bayes (hard voting)\\'    
    create_folder_if_not_exist(results_path)
    logfile=results_path+'logfile.txt'
    #logfile=results_path+'logfile_optimal_threshold.txt'
    #logfile=results_path+'logfile_0.5_threshold.txt'
    #logfile=results_path+'logfile_preprocess_ga_cv_of_gaussianNB_selected_features.txt'
    #logfile=results_path+'logfile_preprocess_rswr_ga_cv_of_gaussianNB_selected_features.txt'
    training_testing(trainset_file=trainset_file,
                     testset_file=testset_file,
                     select_EIS_features=select_EIS_features,
                     select_demo_features=select_demo_features,
                     testset_path=None)                        
    
#ranking in ascending order by p-value, metabolite feature no. e.g. 1 is lactate and 8 is BCAA (reference: project_meeting2oct20.ppt)
        #1                 3
        #2                 6
        #3                 8
        #4                 5
        #5                 1
        #6                 4
        #7                 2
        #8                 7
    #feature ranking in descending order of difference between mean of preterm and mean of term, EIS feature no. (reference: project_meeting2oct20.ppt (slide: Mean of Preterm vs. Mean of Onterm (Best EIS Spectra)))
    # 1                    1
    # 1                    2
    # 1                    3
    # 1                    4
    # 1                    5
    # 1                    6
    # 1                    7
    # 1                    8
    # 1                     9
    # 1                    16
    # 1                    17
    # 1                    18
    # 1                    19
    # 1                    20
    # 1                    21
    # 1                   22
    # 1                  23
    # 1                  24
    # 1                 25
    # 1                26
    #28,               10
    #28,               11
    #28,               12
    #28,               13
    #28                14
    #28,               15 
    #28,                27 
    #28,                28 
    #feature ranking in ascending order by p-value, EIS feature no. (reference: project_meeting2oct20.ppt (slide: Mean of Preterm vs. Mean of Onterm (Best EIS Spectra)))
    # 1                    1
    # 1                    2
    # 1                    3
    # 1                    4
    # 1                    5
    # 1                    6
    # 1                    7
    # 1                    8
    #2                9 
    #4               10
    #5               11
    #7               12
    #8               13
    #9               14
    #10               15 
    # 1                    16
    # 1                    17
    # 1                    18
    # 1                    19
    # 1                    20
    # 1                    21
    # 1                   22
    # 1                  23
    # 1                  24
    # 1                 25
    # 1                26    
    #3             27 
    #6             28 
'''
    ###preprocess training set
    #ros = RandomOverSampler(sampling_strategy=dict_size,
    #                         random_state=1)
    #ros = RandomOverSampler(sampling_strategy='minority', #oversample the minority classes only
    #                        random_state=1)
    #tuned parameters of pipeline
    #param_grid = {#small and large values of C
           #'fs__estimator__C': [1e-8,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]+list(np.logspace(0,4,10)),
    #       'fs__estimator__criterion': ['gini', 'entropy'],#feature selection using decision tree
    #       'logistic__C': np.logspace(0,4,10)
           #'logistic__C': [1e-8,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]+list(np.logspace(0,4,10))
           #'C': [1e-8,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]+list(np.logspace(0,4,10)) #C of logistic regression
    #       }
    #param_grid = { #large values of C
                   #'fs__estimator__C': np.logspace(0,4,10),
                  #'logistic__C': np.logspace(0,4,10)}
    
    ##Select best parameters of pipeline using grid search and CV###
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    # Define a pipeline to search for the best combination of PCA truncation
    # and classifier regularization.
    #pca = PCA()
    # set the tolerance to a large value to make the example faster
    decisiontree = DecisionTreeClassifier(criterion='entropy',class_weight='balanced')
    logistic = LogisticRegression(max_iter=1000, solver='liblinear', C=1e-8,class_weight='balanced')
    #pipe = Pipeline(steps=[('ros',ros),('pca', pca), ('logistic', logistic)])

    rf=RandomForestClassifier(n_jobs=-1,class_weight='balanced',n_estimators=50,random_state=seed)
    #xgb=XGBClassifier(n_estimators=10,objective='multi:softmax',random_state=seed,n_jobs=-1)
    xgb=XGBClassifier(n_estimators=10,random_state=seed,n_jobs=-1)#2 class dataset

    #fs=SelectFromModel(rf,threshold='mean')
    #fs=SelectFromModel(DecisionTreeClassifier(criterion='entropy',random_state=seed))
    fs_logistic=SelectFromModel(logistic)
    fs_decisiontree=SelectFromModel(decisiontree)
    zscore=preprocessing.StandardScaler()
    #pipe = Pipeline(steps=[('ros',ros),('fs',fs),('logistic', logistic)]) #preprocessing + classification pipeline
    #pipe = Pipeline(steps=[('ros',ros),('fs',fs)]) #preprocessing pipeline
    #pipe = Pipeline(steps=[('fs',fs_logistic),('zscore',zscore),('logistic', logistic)]) #preprocessing + classification pipeline
    pipe = Pipeline(steps=[('fs',fs_decisiontree),('zscore',zscore),('logistic', logistic)]) #preprocessing + classification pipeline
    #pipe = Pipeline(steps=[('fs',fs),('logistic', logistic)]) #preprocessing + classification pipeline
    gridsearch = GridSearchCV(pipe, param_grid, n_jobs=-1,cv=5)
    best_model=gridsearch.fit(X1,y1)
    best_params=gridsearch.best_params_
    print("Best parameter (CV score=%0.3f):" % gridsearch.best_score_)
    print(best_params)
    #pc=best_params['pca__n_components']
    #C=best_params['fs__estimator__C']
    #print('Best C of selectFromModel:',C)
    criterion=best_params['fs__estimator__criterion']
    print('Best criterion of selectFromModel:',criterion)
    C2=best_params['logistic__C']
    print('Best C of logistic regression',C2)
    ###Set the pipeline with the best parameters and fit it on the training set 
    #pipe.set_params(pca__n_components=pc,logistic__C=C,logistic__solver='newton-cg')
    ###look at the selected features
    #pipe.set_params(fs__estimator__C=C)
    pipe.set_params(fs__estimator__criterion=criterion)
    pipe=pipe.fit(X1,y1)
    cols=X1.columns
    print('selected features: ',len(cols[pipe['fs'].get_support()]))
    print(cols[pipe['fs'].get_support()])
    X2=pipe['fs'].transform(X1)#reduce the balanced training set using the selected features
    (_,c)=X2.shape
    print('no. of selected features: ',c)
    print('best_model: ',best_model)
    ###preprocess the training set using the fitted pipeline with the best found parameters 
    #X1,y1=pipe['ros'].fit_resample(X,y)#oversample the training set to get a balanced training set
    #X=pipe['fs'].transform(X) #reduce the original training set using the selected features
    print('===logistic regression===')
    #logistic.fit(X2,y1)#train model on reduced balanced training set
    #pred=logistic.predict(X3)
    #train_accuracy=logistic.score(X3,y)
    #print('training accuracy: ',train_accuracy)
    #train_report=classification_report(y,pred)
    #print(train_report)
    (auc,tpr,tnr,fpr,fnr)=predict_testset(False,best_model,pd.DataFrame(X,index=X.index).join(y))
    print('training AUC: ',auc)
    #(_,c)=test_set.shape
    #X2=test_set.iloc[:,:c-1]
    #y2=test_set.iloc[:,c-1]
    #(auc2,tpr2,tnr2,fpr2,fnr2)=predict_testset(False,best_model,pd.DataFrame(X2,index=test_set.index).join(y2))
    (auc2,tpr2,tnr2,fpr2,fnr2)=predict_testset(False,best_model,test_set)
    print('test AUC: ',auc2)
    #pred2=logistic.predict(X4)
    #test_accuracy=logistic.score(X4,y2)
    #print('testing accuracy: ',test_accuracy)
    #test_report=classification_report(y2,pred2)
    #print(test_report)
    dump(best_model,log_reg_file)
    log_reg_performance=(i,auc+auc2,auc,tpr,tnr,fpr,fnr,auc2,tpr2,tnr2,fpr2,fnr2)
    #log_reg_performance=(int(i),train_accuracy+test_accuracy,train_report,test_report)
    #log_reg_train_accuracyL.append(train_accuracy)
    #log_reg_test_accuracyL.append(test_accuracy)
    #utilities.summarize_accuracy_results('logistic regression',logfile,log_reg_performanceL,log_reg_train_accuracyL,log_reg_test_accuracyL)
'''    
    