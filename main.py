"""
Interface to run machine learning pipelines
"""
import workflows
import utilities

def cross_val_split():
    #split a dataset into k training sets and k validation sets using k-fold cross validation
    import pandas as pd
    from preprocess import cross_validation_split_and_remove_duplicates_from_valid_set
    dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_previous_pregnancy_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_no_treatment_with_ids.csv"
    data=pd.read_csv(dataset)
    k=5
    results_path='C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_previous_pregnancy\\'+str(k)+'-fold cv\\'
    #results_path='C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\'+str(k)+'-fold cv\\'
    utilities.create_folder_if_not_exist(results_path)
    trainsetL,valsetL,_,_=cross_validation_split_and_remove_duplicates_from_valid_set(data,k)
    for i in range(k):
        trainset=trainsetL[i]
        valset=valsetL[i]
        trainset.to_csv(results_path+'trainset'+str(i)+'.csv',index=False)
        valset.to_csv(results_path+'valset'+str(i)+'.csv',index=False)
        
def split_data_into_training_sets_and_test_sets_by_ids():
    from mis_data import split_train_testsets_by_ids
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\divide by air reference\\mis_data_C1C2C3_divide_by_air_no_missing_labels.csv"
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\'
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\no compensation\\mis_data_c1c2c3_no_compensation_visit1_visit2_no_missing_labels.csv"
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (phase correction and no air compensation)\\'    
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\no compensation\\mis_data_c1c2c3_no_compensation_visit1_visit2_no_missing_labels_amplitude.csv"
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\no compensation\\mis_data_c1c2c3_no_compensation_visit1_visit2_no_missing_labels.csv"
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\no compensation\\mis_data_c1c2c3_no_compensation_visit1_no_missing_labels.csv"
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\no compensation\\mis_data_c1c2c3_no_compensation_visit2_no_missing_labels_amplitude_phase_div_freqsq.csv"
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\divide v1v2 compensation\\mis_data_c1c2c3_with_divide_v1v2_compensation_visit2_no_missing_labels_freq9to16_div_freqsq.csv"
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\divide v1v2 compensation\\mis_data_c1c2c3_with_divide_v1v2_compensation_visit2_no_missing_labels_div_freqsq.csv"
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\divide v1v2 compensation\\mis_data_c1c2c3_with_divide_v1v2_compensation_visit1_no_missing_labels.csv"
    #data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\subtract v1v2 compensation\\mis_data_c1c2c3_with_subtract_v1v2_compensation_visit1_no_missing_labels_div_freqsq.csv"
    data="D:\\EIS preterm prediction\\i4i MIS\\raw data\\subtract v1v2 compensation\\mis_data_c1c2c3_with_subtract_v1v2_compensation_visit2_no_missing_labels_div_freqsq.csv"

    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\real_img_features\\' 
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\visit1\\real_img_features\\' 
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\visit2\\real_img_features_div_freqsq\\' 
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\amplitude_phase_div_freqsq\\' 
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\visit1\\amplitude_phase_div_freqsq\\' 
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\visit2\\amplitude_phase_div_freqsq\\' 
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (divide v1v2 compensation)\\visit2\\freq9to16_div_freqsq\\' 
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (divide v1v2 compensation)\\visit2\\div_freqsq\\' 
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (divide v1v2 compensation)\\visit1\\' 
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (subtract v1v2 compensation)\\visit1\\div_freqsq\\' 
    results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (subtract v1v2 compensation)\\visit2\\div_freqsq\\' 
    
    testset_size=0.2
    iterations=100
    split_train_testsets_by_ids(data,testset_size,iterations,results_path)
    
def split_data_into_training_sets_and_test_sets():
    ###split dataset into K training sets and K test sets (e.g. K=100)
    #output: trainseti.csv (i=0,...,iterations)
    #        testseti.csv
    #dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_438_V1_demographics_treatment_history_obstetric_history_with_ids.csv"
    dataset="U:\\EIS preterm prediction\\438_V1_28inputs_selected_by_filter_and_438_V1_demographics_treatment_history_obstetric_history_with_ids.csv"    
    #dataset="U:\\EIS preterm prediction\\438_V1_28inputs_selected_by_filter_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_with_ids.csv"
    #dataset="U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_with_ids.csv"
    #dataset='filtered_data_28inputs.csv'
    #dataset='438_V1_demographics.csv'
    #dataset='438_V1_treatment_history2.csv'
    #dataset='438_V1_no_preterm_birth.csv'
    #dataset='438_V1_treatment_history2_and_demographics.csv'
    #dataset='438_V1_previous_history_and_demographics2.csv'
    #dataset='438_V1_demographics_obstetric_history_2_parous_features.csv'
    #dataset='438_V1_demographics_obstetric_history.csv'
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_8inputs_log_transformed.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_lactate_log_transformed.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_glucose_log_transformed.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_1input_log_transformed.csv"
    #dataset="D:\\EIS preterm prediction\\metabolite\\asymp_22wks_438_V1_1input_log_transformed_outliers_removed.csv"
    #dataset="D:\\EIS preterm prediction\\438_V1_28inputs_and_438_V1_demographics_obstetric_history_2_parous_features_with_ids.csv"
    #results_path="D:\\EIS preterm prediction\\results\\workflow1\\438_V1_28inputs_and_438_V1_demographics_obstetric_history_2_parous_features\\"
    #dataset="D:\\EIS preterm prediction\\i4i MIS\\raw data\\divide by air reference\\mean_of_amp_phase_of_mis_data_C1C2C3_divide_by_air_no_missing_labels.csv"
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\mean_of_amp_phase_of_mis_data_C1C2C3_divide_by_air_no_missing_labels\\'
    #results_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_lactate_log_transformed\\train66test34\\"
    #results_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_glucose_log_transformed\\train66test34\\"
    #results_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_1input_log_transformed\\train66test34\\"
    #results_path="F:\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\"
    #results_path="F:\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train80test20\\"   
    #results_path="F:\\EIS preterm prediction\\results\\my_filtered_data_28inputs_with_ids\\train80test20\\"   
    #results_path="F:\\EIS preterm prediction\\results\\my_filtered_data_28inputs_with_ids\\train66test34\\"   
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_with_ids\\train66test34\\"   
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_with_ids\\train66test34\\"   
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_438_V1_demographics_treatment_history_obstetric_history\\train66test34\\"
    results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter_and_438_V1_demographics_treatment_history_obstetric_history\\train66test34\\"
    utilities.create_folder_if_not_exist(results_path)
    
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
    #testset_size=0.2
    testset_size=0.34    
    #testset_size=0.35
    iterations=100
    utilities.split_train_testsets(dataset,testset_size,iterations,results_path,seeds_file="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_438_V1_demographics_treatment_history_obstetric_history\\train66test34\\seeds.txt")
    
def cross_validation_training_and_testing():
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\amplitude_phase_div_freqsq\\' 
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\visit2\\real_img_features_div_freqsq\\' 
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\real_img_features\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(divide_by_air_reference)\\mis_data_C1C2C3_divide_by_air_no_missing_labels_x100\\\\"
    #results_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3 (divide by air reference)\\mis_data_C1C2C3_divide_by_air_no_missing_labels_x100\\\\'
    #results_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(divide_by_air_reference)\\mis_data_C1C2C3_divide_by_air_no_missing_labels_x100\\results4\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\visit1_symp\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\visit2_symp\\\\"
    #data_path='D:\\EIS preterm prediction\\results\\Di\\ahr_v1_symp_no_compensation\\'
    #data_path='D:\\EIS preterm prediction\\results\\Di\\ahr_v2_symp_no_compensation\\'
    #data_path='D:\\EIS preterm prediction\\results\\Di\\ahr_v1_v2_symp_no_compensation\\'
    #results_path=normalized_data_path
    #data_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_1input_log_transformed_3\\"
    #results_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_1input_log_transformed_7\\"
    #results_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_1input_log_transformed_9\\"
    #results_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_1input_log_transformed_10\\\\"

    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\normal data\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\visit1_symp\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\visit2_symp\\\\"
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3 (441 spectra)\\normal data\\\\'
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3 (441 spectra)\\\\'
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb_prediction_of_each_patient_using_c1c2c3_(no compensation)\\normal data2\\\\"
    #data_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_8inputs_log_transformed\\"
    #data_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_1input_log_transformed_outliers_removed\\train85%_test15%\\"
    #data_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_lactate_log_transformed\\train66test34\\"
    #data_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_glucose_log_transformed\\train66test34\\"
    #data_path="D:\\EIS preterm prediction\\results\\metabolite\\asymp_22wks_438_V1_1input_log_transformed\\train66test34\\"
    data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (divide v1v2 compensation)\\visit2\\freq9to16_div_freqsq\\' 
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (divide v1v2 compensation)\\visit2\\div_freqsq\\' 
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (divide v1v2 compensation)\\visit1\\' 
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (divide v1v2 compensation)\\visit1\\div_freqsq\\' 
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (divide v1v2 compensation)\\visit1\\freq9to16_div_freqsq\\' 

    #results_path=data_path
    #results_path=data_path+"results_with_added_noise\\"    
    results_path=data_path+"rf_lr_results\\"
    logfile=results_path+'logfile3.txt'
    ###parameters of workflow1
    #add_noise_option='add_noise'
    add_noise_option='no_noise'
    #balanced_trainset_size2=-1
    #balanced_trainset_size2='undersample class0 to size of class1 with replacement' 
    #balanced_trainset_size2='undersample class0 to size of class1 without replacement'
    #balanced_trainset_size2=200#combination of oversampling and undersampling: 
                               #If set balanced training set size to 2N where size of preterm class < N < size of onterm class (majority class), 
                               #oversample preterm class to N  
                               #and undersample onterm class to N e.g. N=100 (balanced training set size=2N).
    #balanced_trainset_size2='oversample class1 to size of class0'
    #balanced_trainset_size2='undersample class0 to size of class1'
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
    #balanced_trainset_size2='1500 with oversample class1 and class0 separately'
    oversampling_method='oversample_class1_and_class0_separately_using_repeated_bootstraps'
    #oversampling_method='smote'
    #oversampling_method='random sampling with replacement'
    #oversampling_method='borderline smote'
    #oversampling_method='adasyn'
    #degree2=-1
    #degree2=2
    degree2=3
    #degree2=4 #degree of poly features of PTB classifier
    #degree2=5
    #k2=-1#do not use information gain feature selection
    #k2=16
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
    #wrapper_es_fs=False #GA RSFS
    #mini_features=3
    #max_features=3        
    mini_features=-1 #(Randomly select best reducts from all the reducts found by GA)
    max_features=-1        
    #mini_features=10
    #max_features=15
    #max_features=16
    #mini_features=15
    #max_features=25
    number_of_reducts=0 #no GA feature selection performed
    #number_of_reducts=10
    #number_of_reducts=20
    #number_of_reducts=40
    #discretize_method="equal_freq"
    #discretize_method="pki" #proportional k-interval discretization (number of bins= square root of size of training set)
    discretize_method="equal_width"
    ##no. of bins determines the size of reducts: the more bins, the smaller the sizes of the reducts and vice versa.
    ##set no. of bins to find reducts of medium sizes and small sizes
    if discretize_method=='pki':
        bins=None #pki discretization automatically determines no. of bins
    #bins=6
    #bins=10
    #bins=20 #smallest bins with dependency degree 1
    #bins=50 #no. of bins of equal frequency discretization
    bins=30 #smaller the no. of bins, the more instances in each bin and vice versa
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
    #reg=list(np.logspace(-4,4,10))#10 regularizations between 10^-4 and 10^4
    #reg.append(1.0)#add regularization 1.0
    #reg.sort()#sort in ascending order
    #reg=[1e-4] 
    #reg=[1e-8,1e-9,1e-4,1e-2,1.0,2e1,1.7e2,1.3e3]
    reg=[1e-8] #the best regularization is normally 1e-8
    #reg=[1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    #reg=[1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1]
    #trees=[20,30,50,60,80,100,130,150,180,200]
    #trees=['50','100','150','200']
    #trees=[20,30,50,80,100,150,200,250,300,350,400]
    #trees=[20,30,40,50,80,100,150,200,250,300]
    #trees=[10,15,20] #10*dimensionality of data is a good parameter setting
    #trees=[15,20,30,40,60,80,100,120,150,160,200] 
    #trees=[20,30,40,50,60,70,80,90,100]
    #trees=[200,250,300,150,100,50,20]
    trees=['20','10','15']#,'30','40','50','60','70','80','90','100'] #good start: trees=10*no. of features
    #trees=[100]
    #trees=[200,250,300,350]
    #trees=['250','200','150','100','80','60','40','20','10']
    #stop_cond_reg=11 #stop condition for tuning regularization of logistic regression
    #stop_cond_trees=11 #stop condition for tuning trees of random forest
    #stop_cond_reg=5 #stop condition for tuning regularization of logistic regression
    #stop_cond_trees=5 #stop condition for tuning trees of random forest
    stop_cond_reg=6 #stop condition for tuning regularization of logistic regression
    stop_cond_trees=10 #stop condition for tuning trees of random forest
    compare_with_set_of_all_features2=True
    predict_ptb_of_each_id_using_all_spectra_of_id2=True
    #predict_ptb_of_each_id_using_all_spectra_of_id2=False
    #classifiersTypes2='continuous classifiers only'
    #classifiersTypes2='discrete classifiers only'
    #classifiersTypes2='discrete and continuous classifiers'
    #iterations=1
    iterations=100
    #final_prob='average of majority probs'
    final_prob='average of all probs'
    workflows.cross_validation_training_and_testing(iterations,
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
                                          data_path=data_path,
                                          results_path=results_path,
                                          logfile=logfile,
                                          #logfile=results_path+'logfile_good_testset2_with_outliers_removed_by_pca_trained_on_good_trainset2.txt',
                                          #logfile=results_path+'logfile_outliers.txt',
                                          compare_with_set_of_all_features=compare_with_set_of_all_features2,
                                          predict_ptb_of_each_id_using_all_spectra_of_id=predict_ptb_of_each_id_using_all_spectra_of_id2,
                                          add_noise_option2=add_noise_option,
                                          #noise_percent=10,
                                          balanced_trainset_size2=balanced_trainset_size2,
                                          oversampling_method=oversampling_method,
                                          degree2=degree2,
                                          k2=k2,
                                          #wrapper_es_fs=wrapper_es_fs,
                                          #classifiersTypes=classifiersTypes2,
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
                                          final_prob=final_prob
                                          )  

def predict_test_sets_using_weka(data_path=None,
                      model_path=None,
                      results_path=None,
                      logfilename=None,
                      testsets=None,
                      predict_ptb_of_each_id_using_all_spectra_of_id=False,
                      modelnumlist=None
                      ):
    #predict test set using Weka
    log_regL=[] #list of log reg performance=(iteration,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test tpr,test tnr,test fpr,test fnr)
    rf_L=[] #list of random forest performance==(iteration,train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test tpr,test tnr,test fpr,test fnr)
    log_reg_train_aucL=[]
    rf_train_aucL=[]
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
    
    logfile=results_path+logfilename
    utilities.create_folder_if_not_exist(results_path)
    for iteration in modelnumlist:
        workflows.predict_test_set(iteration=iteration,
                         logfile=logfile,
                         log_regL=log_regL,
                         rf_L=rf_L,
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
                         model_path=model_path,
                         results_path=results_path,
                         weka_path='c:\\Program Files\\Weka-3-7-10\\weka.jar',
                         java_memory='4g'
                         )

def select_best_spectra_using_eis_spectrum_selection_filter(testset_ids_csv,results_path,selected_spectra_csv):
    
    ###weka eis spectrum selection filter
    eisfilter='C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filters from sharc\\selected_unselected_eis_readings\\rf23.model'
    eisfilter_inputs_output_csv='C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filters from sharc\\selected_unselected_eis_readings\\rf23.model_inputs_output.csv'
    eisfilter_discrete_cuts_file='C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filters from sharc\\selected_unselected_eis_readings\\rf23.discrete_cuts'
    eisfilter_path='C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filters from sharc\\'
    eisfilter_type='rf'
    eisfilter_software='weka'
    all_eis_readings_of_ids="U:\\EIS preterm prediction\\438_V1_all_eis_readings_28inputs_with_ids.csv"                 
    
    '''
    ###sklearn eis spectrum selection filter (an example sklearn filter)
    eisfilter="C:\\Users\\uos\\EIS preterm prediction\\results\\sklearn_pipeline\\selected_unselected_eis_readings\\rf\\rf0.joblib"
    eisfilter_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\sklearn_pipeline\\selected_unselected_eis_readings\\rf\\rf0.model_inputs_output.csv"
    eisfilter_discrete_cuts_file=None
    eisfilter_path="C:\\Users\\uos\\EIS preterm prediction\\results\\sklearn_pipeline\\selected_unselected_eis_readings\\rf\\"
    eisfilter_software='sklearn'
    all_eis_readings_of_ids="U:\\EIS preterm prediction\\438_V1_all_eis_readings_28inputs_with_ids.csv"                 
    eisfilter_type='rf'   
    '''
    java_memory='4g'
    weka_path="U:\\EIS preterm prediction\\weka-3-7-10.jar"
    
    import ModelsPredict as mp
    MP=mp.ModelsPredict(select_readings_parallel=True,
                 eisfilter_path=eisfilter_path,
                 eisfilter_type=eisfilter_type,
                 eisfilter_software=eisfilter_software,
                 results_path=results_path,
                 all_eis_readings_of_ids=all_eis_readings_of_ids,
                 weka_path=weka_path,
                 java_memory=java_memory
                 )
    (selectedspectra,_)=MP.select_readings_using_filtering(eisfilter,
                                                           eisfilter_inputs_output_csv,
                                                           filter2_discrete_cuts_file=eisfilter_discrete_cuts_file,
                                                           testset_ids=testset_ids_csv)
    selectedspectra.to_csv(results_path+selected_spectra_csv,index=False)

if __name__ == "__main__":
    #cross_val_split()
    #split_data_into_training_sets_and_test_sets_by_ids()
    #split_data_into_training_sets_and_test_sets()
    #trainset_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\trainset7.csv"
    #testset_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\testset7.csv"
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\xgb\\xgb7.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\xgb\\trainset7.preprocessed_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\438_V1_28inputs_selected_by_filter\\train66test34\\xgb\\"
    '''
    trainset_csv="U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_previous_pregnancy_with_ids.csv"
    '''
    ###EIS and PTB history test data
    testset_csv="U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_treated_with_ids.csv"
    #testset_csv="U:\\EIS preterm prediction\\my_filtered_data_28inputs_and_have_ptb_history_treated_with_previous_pregnancy_with_ids.csv"
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\rbfsvm (zscore)_5\\rbf_svm0.joblib"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\rbfsvm (zscore)_5\\"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\rbfsvm (zscore)_5\\rbf_svm0.model_inputs_output.csv"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\xgb\\xgb0.model_inputs_output.csv"
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\xgb\\xgb0.joblib"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\xgb\\"
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_previous_pregnancy\\rbfsvm (zscore)_3\\rbf_svm0.joblib"   
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_previous_pregnancy\\rbfsvm (zscore)_3\\"
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\gp_with_noise\\gp_rbf_with_noise0.joblib"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\gp_with_noise\\"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\gp_with_noise\\gp_rbf_with_noise0.model_inputs_output.csv"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\mlp_2layers\\mlp0.model_inputs_output.csv"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\mlp_2\\mlp0.model_inputs_output.csv"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\mlp_2layers_2\\mlp0.model_inputs_output.csv"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\mlp_2layers_3\\mlp0.model_inputs_output.csv" #test AUC: 0.76 (TPR 0.63)
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\mlp_2layers_5\\mlp0.model_inputs_output.csv" #test AUC: 0.73 (TPR 0.79)
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\mlp_2layers_6\\mlp0.model_inputs_output.csv"
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\mlp_2layers_6\\mlp0.joblib" 
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\mlp_2layers_6\\"    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\lgb\\lgbm0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\lgb\\lgbm0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\lgb\\"
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\gaussianNB_3\\gaussianNB0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\gaussianNB_3\\gaussianNB0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\gaussianNB_3\\"
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\rf_3\\rf0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\rf_3\\rf0.model_inputs_output_csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\rf_3\\"
    ##ensembles###
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\soft voting_equal_weights\\voted_classifier0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\soft voting_equal_weights\\voted_classifier0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\soft voting_equal_weights\\"   
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\soft voting_cv_auc_weights\\voted_classifier0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\soft voting_cv_auc_weights\\voted_classifier0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\soft voting_cv_auc_weights\\"   
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_log_regression2\\stacked_ensemble0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_log_regression2\\stacked_ensemble0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_log_regression2\\"

    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_gp\\stacked_ensemble0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_gp\\stacked_ensemble0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_gp\\"

    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_naivebayes\\stacked_ensemble0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_naivebayes\\stacked_ensemble0.model_inputs_output.csv"
    #results_path=model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_naivebayes\\"
    
    #model="C:\\Users\\uos\\Desktop\\stacked_ensemble_naivebayes\\stacked_ensemble0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\Desktop\\stacked_ensemble_naivebayes\\stacked_ensemble0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\Desktop\\stacked_ensemble_naivebayes\\"
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_svm\\stacked_ensemble0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_svm\\stacked_ensemble0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_svm\\"    

    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_mlp\\stacked_ensemble0.joblib"    
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_mlp\\stacked_ensemble0.model_inputs_output.csv"    
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_mlp\\"    

    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\soft voting_custom_weights\\voted_classifier0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\soft voting_custom_weights\\voted_classifier0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\soft voting_custom_weights\\"
        
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_nb2\\stacked_ensemble0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_nb2\\stacked_ensemble0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_nb2\\"
 
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_nb3\\stacked_ensemble0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_nb3\\stacked_ensemble0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\stacked_ensemble_nb3\\"
 
    model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\rf_4\\rf0.joblib"
    results_path="C:\\Users\\uos\EIS preterm prediction\\results\\my_filtered_data_28inputs_and_have_ptb_history_no_treatment\\rf_4\\"
    
    ###EIS test data
    #testset_csv="U:\\EIS preterm prediction\\my_filtered_data_28inputs_treated_with_ids.csv"
    #trainset_csv="U:\\EIS preterm prediction\\my_filtered_data_28inputs_no_treatment_with_ids.csv"
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (minmax)_bootstrap\\rbf_svm0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (minmax)_bootstrap\\rbf_svm0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (minmax)_bootstrap\\"

    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_bootstrap\\rbf_svm0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_bootstrap\\rbf_svm0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_bootstrap\\"
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_5\\rbf_svm0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_5\\rbf_svm0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_5\\"
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_2\\rbf_svm0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_2\\rbf_svm0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_2\\"
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_3\\rbf_svm0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_3\\rbf_svm0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_3\\"
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_4\\rbf_svm0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_4\\rbf_svm0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\rbfsvm (zscore)_4\\"
        
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gp_with_noise\\gp_rbf_with_noise0.model_inputs_output.csv"
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gp_with_noise\\gp_rbf_with_noise0.joblib"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gp_with_noise\\"
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gp_with_noise_3\\gp_rbf_with_noise0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gp_with_noise_3\\gp_rbf_with_noise0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gp_with_noise_3\\"
        
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gp\\gp_rbf0.model_inputs_output.csv"  
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gp\\gp_rbf0.joblib"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gp\\"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\mlp_2_layers_(zscore)\\mlp0.model_inputs_output.csv"
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\mlp_2_layers_(zscore)\\mlp0.joblib"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\mlp_2_layers_(zscore)\\"
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gaussianNB_3\\gaussianNB0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gaussianNB_3\\gaussianNB0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\gaussianNB_3\\"
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\soft voting_equal_weights\\voted_classifier0.joblib"
    #model_inputs_output_csv="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\soft voting_equal_weights\\voted_classifier0.model_inputs_output.csv"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\soft voting_equal_weights\\"        
    
    #model="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\soft voting_weights=1\\voted_classifier0.joblib"
    #results_path="C:\\Users\\uos\\EIS preterm prediction\\results\\my_filtered_data_28inputs_no_treatment\\soft voting_weights=1\\"
        
    
    import pandas as pd
    from joblib import load
    model=load(model)
    #traindf=pd.read_csv(trainset_csv)
    testdf=pd.read_csv(testset_csv)
    #cols=list(traindf.columns)
    cols=list(testdf.columns)
    #if cols[0] == 'Identifier' or cols[0] == 'hospital_id':
    #    (_,c)=traindf.shape
    #    traindf=traindf.iloc[:,1:c] #removed ids column                
    if cols[0] == 'Identifier' or cols[0] == 'hospital_id':
        (_,c)=testdf.shape
        testdf=testdf.iloc[:,1:c] #removed ids column                
    #utilities.predict_trainset_and_testset_using_sklearn_and_optimal_threshold(traindf,testdf,model,display_info=False,score='my_score',results_path=results_path,trainset_csv=trainset_csv,testset_csv=testset_csv)
    threshold=0.53
    utilities.predict_testset_using_sklearn_model_and_threshold(testdf,model,results_path,testset_csv=testset_csv,threshold=threshold)
    
    ##hard voting (majority vote) classifier###
    #threshold=0.34 #optimal threshold of hard voting classifier
    #optimal_thresholds_of_base_models=[0.49, 0.1, 0.46, 0.29]#[0.4891266729761132, 0.10443106929176338, 0.4635999742367338, 0.287141411252805]
    #utilities.predict_testset_using_sklearn_model_and_threshold(testdf,model,results_path,testset_csv=testset_csv,ensemble_with_hard_voting=True,optimal_thresholds_of_base_models=optimal_thresholds_of_base_models,threshold=threshold)

    
    '''
    #cross_validation_training_and_testing()
    #data_path="D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1 c2 c3\\normal data\\"
    #results_path=data_path+"testset_normal_data_plus_outliers_oversample_class0_and_class1_separately4\\"
    #data_path='D:\\EIS preterm prediction\\results\\mis\\ptb prediction of each patient using c1c2c3 (no compensation)\\' 
    #data_path='C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\15dec_filtered_data_28inputs\\' #path of test set
    data_path='u:\\EIS preterm prediction\\'
    model_path='C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\' 
    #model_path=data_path
    results_path=model_path
    testsets="438_V1_28inputs_selected_by_filter.csv"
    #testsets="testset_normal_data_plus_outliers"
    #testsets='testset'
    #testsets='testset_good_readings'
    #testsets='trainset'
    logfilename='logfile2.txt'
    #prediction_list_option='prediction list' # 'no prediction list'
    predict_ptb_of_each_id_using_all_spectra_of_id=False,
    modelnumlist=[87]
    #modelnumlist=[92]
    #modelnumlist=range(100)
    predict_test_sets_using_weka(data_path=data_path,
                      model_path=model_path,
                      results_path=results_path,
                      logfilename=logfilename,
                      testsets=testsets,
                      predict_ptb_of_each_id_using_all_spectra_of_id=predict_ptb_of_each_id_using_all_spectra_of_id,
                      modelnumlist=modelnumlist)
    '''
    '''
    #testset_ids_csv="U:\\EIS preterm prediction\\438_V1_28inputs_with_ids.csv"
    results_path='C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\filters from sharc\\selected_unselected_eis_readings\\'
    #selected_spectra_csv="438_V1_28inputs_selected_by_filter.csv"
    select_best_spectra_using_eis_spectrum_selection_filter('ids3.csv',results_path,'ids3_best_readings_weka.csv')
    '''