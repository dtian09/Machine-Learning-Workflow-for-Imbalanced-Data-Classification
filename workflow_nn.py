#Program to 1. Train and validate multiple neural networks (MLPs and DNNs) of different topologies in parallel on multiple CPU cores (a network is validated on a validation set while it is being trained in order to determine which epoch to stop training)
#           2. Select an optimal network among all the networks trained based on their validation performances.
#           3. Evaluate the performance of the selected neural network in step 2 on a test set.
#Data pre-processing: normalization of input features, oversampling/undersampling (optional) and genetic algorithm feature selection (optional) of the training set can be performed before trainig neural networks to improve performance of the neural networks
#To run on a PC with multiple CPU cores: 
#   python workflow_nn.py
#To run on a HPC using SLURM scheduler:  
#   sbatch slurm_script.sh

import preprocess as prep
import os, sys
import workflows as wf
import numpy as np
#import re
from keras import models
from keras.models import load_model
from keras import layers
from keras import regularizers
import tensorflow as tf
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.exceptions import DataConversionWarning
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import random
import operator
from keras.utils.np_utils import to_categorical
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adamax
import utilities
from joblib import dump, load, Parallel, delayed
from utilities import zscore_normalize_inputs, min_max_normalize_inputs, norm_normalize_inputs, power_transform_normalize_inputs

networkstype='multilayer'
singlelayer_sizeL=[]
multilayers_sizeL=[]
discretize_method="equal_freq"
bins=80
ga_path="d:\\EIS preterm prediction\\ga_rsfs.jar"
populationSize=100
generations=50
#generations=10
crossover_prob=0.6
mutation_prob=0.033
fitness='find_reducts'
#fitness='find_high_info_gain_reducts'
#number_of_reducts=40#use number_of_reducts top-ranked reducts to train models
number_of_reducts=10#use number_of_reducts top-ranked reducts to train models
results_path=''
mini_features=10
max_features=15
iterative_training=1
trainsets1trainsets2Path=''
learning_rate=0.002
repeat=10
parallel_training_validate=False
parallel_train_validate_nn_of_random_initial_weights=True
num_jobs = 8 #number of CPU cores to use
      
def nn_train(normalizeInputs,outputs,trainset,valset,nnlayers,nn_file,classes_weights={0:0.5, 1:0.5},epochs=1000):        
        (_,c)=trainset.shape
        X_train=trainset.iloc[:,:c-1]
        X_val=valset.iloc[:,:c-1]
        X_train=X_train.to_numpy()
        X_val=X_val.to_numpy()
        t_train = trainset.iloc[:,c-1]
        t_val = valset.iloc[:,c-1]
        t_train=t_train.to_numpy()
        t_val=t_val.to_numpy()
        if normalizeInputs == 'zscore':#normalize inputs of training and test sets
            (X_train,X_val,X_scaler)=zscore_normalize_inputs(X_train,X_val)
            print('zscore normalize inputs')
        elif normalizeInputs == 'mini_max':
            (X_train,X_val,X_scaler)=min_max_normalize_inputs(X_train,X_val)
            print('mini-max normalize inputs')
        elif normalizeInputs == 'norm':
            (X_train,X_scaler) = norm_normalize_inputs(X_train)
            X_val = X_scaler.transform(X_val)
            print('norm normalize inputs')
        elif normalizeInputs == 'yb':#Yeo-Johnson (work for pos and neg input values)
            (X_train,X_scaler) = power_transform_normalize_inputs(X_train,'yb')
            X_val = X_scaler.transform(X_val)
            print('yeo-johnson normalize inputs')
        elif normalizeInputs == 'bc':#Box-Cox (work for pos input values only)
            (X_train,X_scaler) = power_transform_normalize_inputs(X_train,'bc')
            X_val = X_scaler.transform(X_val)
            print('box-cox normalize inputs')
        else:
            print('Normalization of inputs not performed')
        (_,number_of_features)=X_train.shape        
        # Start neural network
        network = models.Sequential()
        h1=nnlayers[0]#create the input layer and the 1st hidden layer
        # Add fully connected layer with a ReLU activation function
        #activation_function="sigmoid"
        activation_function="relu"
        print('hidden layer activation function: '+activation_function)
        reg=0.01
        network.add(layers.Dense(units=int(h1), activation=activation_function, kernel_regularizer=regularizers.l2(reg),input_shape=(number_of_features,)))
        network.add(layers.Dropout(0.5))
        if len(nnlayers)>1:
            i=1
            while i < len(nnlayers):              
                h=nnlayers[i]
                network.add(layers.Dense(units=int(h), kernel_regularizer=regularizers.l2(reg), activation=activation_function))
                network.add(layers.Dropout(0.5))
                i+=1
        #opt = RMSprop(learning_rate=0.001)  # Root Mean Square Propagation
        #opt = RMSprop(learning_rate=0.0001)
        #opt = RMSprop(learning_rate=0.00016)
        #opt = RMSprop(learning_rate=0.0005)
        #opt = RMSprop(learning_rate=0.0008)
        #opt=SGD(learning_rate=0.001) #stochastic gradient descent
        #opt=Adadelta(learning_rate=1.0)#default
        #opt=Adamax(learning_rate=0.002)#default
        #opt=Adam(learning_rate=0.001)#default 
        opt=Adamax(learning_rate=learning_rate)        
        if outputs == 1:
            # Add fully connected layer with a sigmoid activation function
            network.add(layers.Dense(units=1, activation="sigmoid"))
            # Compile neural network
            network.compile(loss="binary_crossentropy", # Cross-entropy
            optimizer=opt,
            metrics=["accuracy"])
        else:#softmatx outputs
            # Add fully connected layer with a sigmoid activation function
            network.add(layers.Dense(units=outputs, activation="softmax"))
            # Compile neural network
            network.compile(loss="categorical_crossentropy", # Cross-entropy
            optimizer=opt,
            metrics=["accuracy"])
            #transform targets to 1 hot-encoding format
            t_train=to_categorical(t_train)#0th column is prob of label 0, 1st column is prob of label 1    
            t_val=to_categorical(t_val)           
        # Set callback functions to early stop training and save the best model so far
        callbacks = [EarlyStopping(monitor="val_loss", patience=10),
                     ModelCheckpoint(filepath=nn_file,
                                     monitor="val_loss",
                                     save_best_only=True)]        
        #Train neural network
        history = network.fit(X_train, # Features
                              t_train, # Target vector
                              epochs=epochs, # Number of epochs
                              #epochs=1000,
                              #epochs=2000,
                              callbacks=callbacks, # Early stopping
                              verbose=0, # Print description after each epoch
                              batch_size=100, # Number of observations per batch
                              #batch_size=200, # Number of observations per batch
                              class_weight=classes_weights,
                              validation_data=(X_val, t_val))
        del [network,X_train,X_val,t_train,t_val,history]
        return (nn_file,callbacks,X_scaler)
        
def iter_train(network,X_scaler,nn_file,outputs,trainset,original_trainset,validset,callbacks,normalizeInputs,m,epochs,classes_weights,
                    best_nn_file,
                    best_nn_train_valid_auc,
                    best_nn_train_auc,
                    best_nn_train_tpr,
                    best_nn_train_tnr,
                    best_nn_train_fpr,
                    best_nn_train_fnr,
                    best_nn_valid_auc,
                    best_nn_valid_tpr,
                    best_nn_valid_tnr,
                    best_nn_valid_fpr,
                    best_nn_valid_fnr):
        #input: a pre-trained network
        #       m, iterations of training process
        #output: the network after iterative training
        (_,c)=trainset.shape
        X_train=trainset.iloc[:,:c-1]#get all the columns except the last one
        X_val=validset.iloc[:,:c-1]
        X_train=X_train.to_numpy()
        X_val=X_val.to_numpy()
        t_train = trainset.iloc[:,c-1]
        t_val = validset.iloc[:,c-1]
        t_train=t_train.to_numpy()
        t_val=t_val.to_numpy()
        try:        
           X_train=X_scaler.transform(X_train)
        except DataConversionWarning:
           print('Inputs of training set are converted to float64 by min-max scaler')
        try:
           X_val=X_scaler.transform(X_val)
        except DataConversionWarning:
           print('Inputs of validation set are converted to float64 by min-max scaler')
        if outputs == 2:
            #transform targets to 1 hot-encoding format
            t_train=to_categorical(t_train)#0th column is prob of label 0, 1st column is prob of label 1    
            t_val=to_categorical(t_val)            
        for i in range(m):
            network.fit(X_train, # Features
                    t_train, # Target vector
                    epochs=epochs, # Number of epochs
                    callbacks=callbacks, # Early stopping
                    verbose=0, # Print description after each epoch
                    batch_size=100, # Number of observations per batch
                    class_weight=classes_weights,
                    validation_data=(X_val, t_val))
            network=load_model(nn_file)
            (valid_auc,valid_tpr,valid_tnr,valid_fpr,valid_fnr)=nn_predict(network,validset,X_scaler)
            if original_trainset.equals(trainset) == False:
                (train_auc,train_tpr,train_tnr,train_fpr,train_fnr)=nn_predict(network,original_trainset,X_scaler)     
            else:#trainset is the original training set
                (train_auc,train_tpr,train_tnr,train_fpr,train_fnr)=nn_predict(network,trainset,X_scaler)
            train_valid_auc=train_auc+valid_auc
            #print('iteration: ',i)
            #print('valid auc: ',valid_auc)
            #print('training auc: ',train_auc)
            if train_valid_auc > best_nn_train_valid_auc:
                   best_nn_train_valid_auc=train_valid_auc
                   best_nn_train_auc=train_auc
                   best_nn_valid_auc=valid_auc
                   best_nn_train_tpr=train_tpr
                   best_nn_train_tnr=train_tnr
                   best_nn_train_fpr=train_fpr
                   best_nn_train_fnr=train_fnr
                   best_nn_valid_tpr=valid_tpr
                   best_nn_valid_tnr=valid_tnr
                   best_nn_valid_fpr=valid_fpr
                   best_nn_valid_fnr=valid_fnr
                   wf.mycopyfile(nn_file,best_nn_file)
            utilities.delete_files([nn_file])
        del [network,X_train,X_val,t_train,t_val,valid_auc,valid_tpr,valid_tnr,valid_fpr,valid_fnr,train_auc,train_tpr,train_tnr,train_fpr,train_fnr]
        return (best_nn_file,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr)
        
def nn_predict(network,testset,X_scaler):
        (_,c)=testset.shape
        X_test=testset.iloc[:,:c-1]
        X_test=X_test.to_numpy()        
        t_test=testset.iloc[:,c-1]
        t_test=t_test.to_numpy()
        if X_scaler!='none':#normalize testset using the scaler of training set
            try:
                X_test = X_scaler.transform(X_test)
            except DataConversionWarning:
                print('Inputs of test set are converted to float64 by min-max scaler')
        prob = network.predict(X_test)
        #print(prob)
        #print(prob.shape)
        y = network.predict_classes(X_test)
        #print(y)
        #print(y.shape)
        (_,c)=prob.shape
        #print(predicted_targets)
        if c == 1:#single output
            auc=roc_auc_score(t_test, prob)
        elif c == 2:#2 outputs (1st column is prob of class 1)
            auc=roc_auc_score(t_test, prob[:,1])
        elif int(c) > 2:#> 2 outputs
            sys.exit('> 2 outputs in nn_predict function')
        tn, fp, fn, tp = confusion_matrix(t_test,y).ravel()
        tnr=tn/(tn+fp)
        tpr=tp/(tp+fn)
        fpr=1-tnr
        fnr=1-tpr
        print('auc: ',str(auc))
        return (auc,tpr,tnr,fpr,fnr)
'''
def singlelayernetwork(normalizeInputs,outputs,class_weight,train_set,original_trainset,valid_set,layer_sizeL,epochs,results_path,m):
        #input: layer_sizeL = [net1_layer_size,net2_layer_size,...,netk_layer_size]
        #outputs, number of output units (if 1 output unit, a logistic output is created, else softmax units are created)    
        #m, iterations of training and testing a network topology
        best_nn_train_test_auc=0
        best_nn_train_auc=0
        best_nn_test_auc=0        
        (_,c)=train_set.shape
        num=random.randint(0,999999)
        best_nn_file=results_path+'best_nn'+str(num)+'.h5'
        for layer_size in layer_sizeL:
            print('single layer size of network: '+str(layer_size))
            for i in range(repeat):#repeat initializing weights to random values and train and test the network topology
                print('repeat: ',str(i))
                rseed=random.randint(0,999999)
                np.random.seed(rseed)   #set seed of keras network weights initializer
                tf.random.set_seed(rseed)#set seed of tensorflow network weights initializer
                num=random.randint(0,999999)
                nn_file=results_path+'nn'+str(num)+'.h5'
                (nn_file,callbacks,X_scaler)=nn_train(normalizeInputs,outputs,train_set,valid_set,[layer_size],nn_file,class_weight,epochs)
                network=load_model(nn_file)
                (test_auc,test_tpr,test_tnr,test_fpr,test_fnr)=nn_predict(network,valid_set,X_scaler)
                (train_auc,train_tpr,train_tnr,train_fpr,train_fnr)=nn_predict(network,original_trainset,X_scaler)
                train_test_auc=train_auc+test_auc
                if train_test_auc > best_nn_train_test_auc:
                   best_nn_train_test_auc=train_test_auc
                   best_nn_train_auc=train_auc
                   best_nn_test_auc=test_auc
                   best_nn_train_tpr=train_tpr
                   best_nn_train_tnr=train_tnr
                   best_nn_train_fpr=train_fpr
                   best_nn_train_fnr=train_fnr
                   best_nn_test_tpr=test_tpr
                   best_nn_test_tnr=test_tnr
                   best_nn_test_fpr=test_fpr
                   best_nn_test_fnr=test_fnr
                   utilities.mycopyfile(nn_file,best_nn_file)
                   best_nn_topology=(c-1,layer_size,outputs)
                   best_nn_weights=(c-1)*layer_size+layer_size*outputs
                utilities.delete_files([nn_file])
                if m>0:
                    (best_nn_file,best_nn_train_test_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_test_auc,best_nn_test_tpr,best_nn_test_tnr,best_nn_test_fpr,best_nn_test_fnr)=iter_train(network,X_scaler,nn_file,outputs,train_set,original_trainset,valid_set,callbacks,X_scaler,m,epochs,class_weight,best_nn_file,best_nn_train_test_auc,best_nn_train_auc,best_nn_test_auc)
                del [network]
            print('validset auc: '+str(best_nn_test_auc))
            print('training auc: '+str(best_nn_train_auc))
            print('topology: '+str(best_nn_topology))
            print('weights: '+str(best_nn_weights))
        return (best_nn_file,best_nn_train_test_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_test_auc,best_nn_test_tpr,best_nn_test_tnr,best_nn_test_fpr,best_nn_test_fnr,best_nn_topology,best_nn_weights,X_scaler)
'''
def network(normalizeInputs,outputs,class_weight,traindf,original_traindf,validsetdf,epochs,results_path,m):
        #input: list of single layer networks or a list of multilayer networks 
        #       format of single layer networks: layers_sizeL=[net1_layer_size,net2_layer_size,...]
        #       format of multilayer networks: layers_sizeL=[[net1_1st_layer_size,net1_2nd_layer_size,...],[net2_1st_layer_size,net2_2nd_layer_size,...],...] 
        #       traindf, traning set dataframe
        #       original_traindf, original training set dataframe
        #       validsetdf, validation set dataframe
        #       outputs, number of output units (if 1 output unit, a logistic output is created, else softmax units are created)    
        #       m, iterations of training and testing a network topology with a specific random initial weights 
        if networkstype=='singlelayer':
            layers_sizeL=singlelayer_sizeL
        elif networkstype == 'multilayer':
            layers_sizeL=multilayers_sizeL
        else:
            sys.exit('wrong network type')
        (_,c)=traindf.shape
        for layers_size_of_a_network in layers_sizeL:#get a network topology
            print('network topology: '+str(layers_size_of_a_network))
            if networkstype=='singlelayer':
                layers_size_of_a_network=[layers_size_of_a_network]
            if parallel_train_validate_nn_of_random_initial_weights:#Parallelly train networks of same topology with random initial weights
                performanceL=Parallel(n_jobs=-1)(delayed(train_validate_nn_random_weights_parallel_step)(j,layers_size_of_a_network,traindf,original_traindf,validsetdf,normalizeInputs,outputs,class_weight,epochs,iterative_training) for j in range(repeat))
                #performanceL = list of (best_nn_file,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr,X_scaler)        
                #                                   0,                      1,                2,                3,                4,                5,                6,                7,                8,               9,               10,                11,     12   
                performanceL.sort(key=operator.itemgetter(2),reverse=True)#sort by train_valid AUC
                best_nn_performance=performanceL[0]
                best_nn_train_auc=best_nn_performance[2]
                best_nn_test_auc=best_nn_performance[7]                
                best_nn_topology=(c-1,layers_size_of_a_network,outputs)
                if len(layers_size_of_a_network)==1:#a single layer network
                   best_nn_weights=(c-1)*layers_size_of_a_network[0]+layers_size_of_a_network[0]*outputs
                elif len(layers_size_of_a_network)==2:#a network of 2-hidden layers
                   best_nn_weights=(c-1)*layers_size_of_a_network[0]+layers_size_of_a_network[0]*layers_size_of_a_network[1]+layers_size_of_a_network[1]*outputs
                elif len(layers_size_of_a_network)==3:#a network of 3 hidden layers
                   best_nn_weights=(c-1)*layers_size_of_a_network[0]+layers_size_of_a_network[0]*layers_size_of_a_network[1]+layers_size_of_a_network[1]*layers_size_of_a_network[2]+layers_size_of_a_network[2]*outputs                       
                else:
                   print('no. of hidden layers > 3')
                   best_nn_weights='weights of a network with > 3 hidden layers'                   
                X_scaler=best_nn_performance[len(best_nn_performance)-1]
                best_nn_performance=list(best_nn_performance)
                best_nn_performance[len(best_nn_performance)-1]=best_nn_topology
                best_nn_performance.append(best_nn_weights)
                best_nn_performance.append(X_scaler)
                i=1
                while i < len(performanceL):
                    performance=performanceL[i]
                    utilities.delete_files([performance[0]])
                    del performance
                    i+=1
                print('validset auc: '+str(best_nn_test_auc))
                print('training auc: '+str(best_nn_train_auc))
                print('topology: '+str(best_nn_topology))
                print('weights: '+str(best_nn_weights))
                del performanceL
                return tuple(best_nn_performance)
            else:#sequentially train networks of same topology with random initial weights
                (best_nn_file,best_nn_train_test_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_test_auc,best_nn_test_tpr,best_nn_test_tnr,best_nn_test_fpr,best_nn_test_fnr,X_scaler)=train_validate_nn_random_weights_sequential(normalizeInputs,outputs,class_weight,traindf,original_traindf,validsetdf,layers_size_of_a_network,epochs,results_path,m)
                best_nn_topology=(c-1,layers_size_of_a_network,outputs)
                if len(layers_size_of_a_network)==1:#single layer networks
                    best_nn_weights=(c-1)*layers_size_of_a_network[0]+layers_size_of_a_network[0]*outputs
                elif len(layers_size_of_a_network)==2:#networks of 2-hidden layers
                   best_nn_weights=(c-1)*layers_size_of_a_network[0]+layers_size_of_a_network[0]*layers_size_of_a_network[1]+layers_size_of_a_network[1]*outputs
                elif len(layers_size_of_a_network)==3:#networks of 3 hidden layers
                   best_nn_weights=(c-1)*layers_size_of_a_network[0]+layers_size_of_a_network[0]*layers_size_of_a_network[1]+layers_size_of_a_network[1]*layers_size_of_a_network[2]+layers_size_of_a_network[2]*outputs                       
                else:
                   print('no. of hidden layers > 3')
                   best_nn_weights='weights of a network with > 3 hidden layers'
                print('validset auc: '+str(best_nn_test_auc))
                print('training auc: '+str(best_nn_train_auc))
                print('topology: '+str(best_nn_topology))
                print('weights: '+str(best_nn_weights))
                return (best_nn_file,best_nn_train_test_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_test_auc,best_nn_test_tpr,best_nn_test_tnr,best_nn_test_fpr,best_nn_test_fnr,best_nn_topology,best_nn_weights,X_scaler)

def train_validate_nn_random_weights_parallel_step(j,layers_size_of_a_network,traindf,original_traindf,validsetdf,normalizeInputs,outputs,class_weight,epochs,m):
        #train and validate a network with random initial weights    
        #traindf, traning set dataframe
        #original_traindf, original training set dataframe
        #validsetdf, validation set dataframe
        #m, iterations of training and testing a network topology with a specific random initial weights 
        print('network: ',str(j))
        rseed=random.randint(0,999999)
        np.random.seed(rseed)   #set seed of keras network weights initializer
        tf.random.set_seed(rseed)#set seed of tensorflow network weights initializer
        num=random.randint(0,999999)
        best_nn_file=results_path+'best_nn'+str(num)+'.h5'
        best_nn_train_valid_auc=0
        best_nn_train_auc=0
        best_nn_valid_auc=0 
        (_,c)=traindf.shape
        nn_file=results_path+'nn'+str(num)+'.h5'
        (nn_file,callbacks,X_scaler)=nn_train(normalizeInputs,outputs,traindf,validsetdf,layers_size_of_a_network,nn_file,class_weight,epochs)
        network=load_model(nn_file)
        (valid_auc,valid_tpr,valid_tnr,valid_fpr,valid_fnr)=nn_predict(network,validsetdf,X_scaler)
        (train_auc,train_tpr,train_tnr,train_fpr,train_fnr)=nn_predict(network,original_traindf,X_scaler)                    
        if m>0:#iterative training of the topology to get a better network
             (best_nn_file,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr)=iter_train(network,nn_file,outputs,traindf,original_traindf,validsetdf,callbacks,X_scaler,m,epochs,class_weight,best_nn_file,best_nn_train_valid_auc,best_nn_train_auc,best_nn_valid_auc)
        else:                    
             train_valid_auc=train_auc+valid_auc
             best_nn_train_valid_auc=train_valid_auc
             best_nn_train_auc=train_auc
             best_nn_valid_auc=valid_auc
             best_nn_train_tpr=train_tpr
             best_nn_train_tnr=train_tnr
             best_nn_train_fpr=train_fpr
             best_nn_train_fnr=train_fnr
             best_nn_valid_tpr=valid_tpr
             best_nn_valid_tnr=valid_tnr
             best_nn_valid_fpr=valid_fpr
             best_nn_valid_fnr=valid_fnr
             utilities.mycopyfile(nn_file,best_nn_file)
        utilities.delete_files([nn_file])
        del [network,train_valid_auc,valid_auc,valid_tpr,valid_tnr,valid_fpr,valid_fnr,train_auc,train_tpr,train_tnr,train_fpr,train_fnr]
        return (best_nn_file,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr,X_scaler)
      
def train_validate_nn_random_weights_sequential(normalizeInputs,outputs,class_weight,traindf,original_traindf,validsetdf,layers_size_of_a_network,epochs,results_path,m):          
            #train and validate a network with random initial weights    
            #traindf, traning set dataframe
            #original_traindf, original training set dataframe
            #validsetdf, validation set dataframe
            #m, iterations of training and testing a network topology with a specific random initial weights        
            (_,c)=traindf.shape
            num=random.randint(0,999999)
            best_nn_file=results_path+'best_nn'+str(num)+'.h5'
            best_nn_train_valid_auc=0
            best_nn_train_auc=0
            best_nn_valid_auc=0 
            for i in range(repeat):#repeat initializing weights to random values and train and valid the network topology
                print('network: ',str(i))
                rseed=random.randint(0,999999)
                np.random.seed(rseed)   #set seed of keras network weights initializer
                tf.random.set_seed(rseed)#set seed of tensorflow network weights initializer
                num=random.randint(0,999999)
                nn_file=results_path+'nn'+str(num)+'.h5'
                (nn_file,callbacks,X_scaler)=nn_train(normalizeInputs,outputs,traindf,validsetdf,layers_size_of_a_network,nn_file,class_weight,epochs)
                network=load_model(nn_file)
                (valid_auc,valid_tpr,valid_tnr,valid_fpr,valid_fnr)=nn_predict(network,validsetdf,X_scaler)
                (train_auc,train_tpr,train_tnr,train_fpr,train_fnr)=nn_predict(network,original_traindf,X_scaler)                    
                if m>0:#iterative training of the topology to get a better network
                    (best_nn_file,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr)=iter_train(network,nn_file,outputs,traindf,original_traindf,validsetdf,callbacks,X_scaler,m,epochs,class_weight,best_nn_file,best_nn_train_valid_auc,best_nn_train_auc,best_nn_valid_auc)
                else:                    
                    train_valid_auc=train_auc+valid_auc
                    best_nn_train_valid_auc=train_valid_auc
                    best_nn_train_auc=train_auc
                    best_nn_valid_auc=valid_auc
                    best_nn_train_tpr=train_tpr
                    best_nn_train_tnr=train_tnr
                    best_nn_train_fpr=train_fpr
                    best_nn_train_fnr=train_fnr
                    best_nn_valid_tpr=valid_tpr
                    best_nn_valid_tnr=valid_tnr
                    best_nn_valid_fpr=valid_fpr
                    best_nn_valid_fnr=valid_fnr
                    utilities.mycopyfile(nn_file,best_nn_file)
                utilities.delete_files([nn_file])
                del [network,train_valid_auc,valid_auc,valid_tpr,valid_tnr,valid_fpr,valid_fnr,train_auc,train_tpr,train_tnr,train_fpr,train_fnr]
            return (best_nn_file,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr,X_scaler)

def train_validate_nn(weka_train_file,weka_original_train_file,weka_validset_file,reductsfile,normalizeInputs,outputs,epochs,class_weight,results_path,weka_path,java_memory):
    num=random.randint(0,999999)
    best_nn_inputs_output=results_path+'best_nn_inputs_output'+str(num)+'.csv'
    #train and validate networks on all the features of the training set
    traindf=utilities.arff_to_dataframe(weka_train_file)
    if weka_original_train_file!='none':
        traindf2=utilities.arff_to_dataframe(weka_original_train_file)
    else:
        traindf2=traindf
    validsetdf=utilities.arff_to_dataframe(weka_validset_file)
    ###train a network using all the features of the training set and valid its performance on the original training set
    print('Train a network using all the features of the training set and validate its performance')
    (best_nn_file,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr,best_nn_topology,best_nn_weights,X_scaler)=network(normalizeInputs,outputs,class_weight,traindf,traindf2,validsetdf,epochs,results_path,iterative_training)
    utilities.get_model_inputs_output('df',validsetdf,best_nn_inputs_output)
    nn_performance=(best_nn_file,best_nn_inputs_output,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr,best_nn_topology,best_nn_weights,X_scaler)
    if int(number_of_reducts) == 0 or reductsfile == 'no feature selection': #no feature selection is done, use all features to train classifiers
        return nn_performance
    elif int(number_of_reducts) > 0:###training a network on each reduct and valid its performance on the original training set
        reductsL=[line.strip() for line in open(reductsfile)]
        reductsL=wf.remove_single_feature_subsets(reductsL)
        if mini_features == -1 and max_features == -1:
            print('select random reducts')
            reductsL=wf.get_random_feature_subsets(reductsL,number_of_reducts)
        elif mini_features >0 and mini_features <= max_features:
            reductsL=wf.get_feature_subsets(reductsL,mini_features,max_features,number_of_reducts)
        else:
            sys.exit('invalid mini_features or max_features')
        if parallel_training_validate:
            #performanceL=Parallel(n_jobs=num_jobs,batch_size=10)(delayed(train_validate_nn_parallel_step)(j,reductsL,weka_train_file,weka_original_train_file,weka_validset_file,normalizeInputs,outputs,class_weight,epochs,iterative_training) for j in range(len(reductsL)))
            #performanceL=Parallel(n_jobs=num_jobs,batch_size=5)(delayed(train_validate_nn_parallel_step)(j,reductsL,weka_train_file,weka_original_train_file,weka_validset_file,normalizeInputs,outputs,class_weight,epochs,iterative_training) for j in range(len(reductsL)))
            performanceL=Parallel(n_jobs=num_jobs)(delayed(train_validate_nn_parallel_step)(j,reductsL,weka_train_file,weka_original_train_file,weka_validset_file,normalizeInputs,outputs,class_weight,epochs,iterative_training) for j in range(len(reductsL)))
            #performanceL = list of (best_nn_file,best_nn_inputs_output,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr,best_nn_topology,best_nn_weights,X_scaler)        
            performanceL.append(nn_performance)#add the performance of the network trained on all features of the training set
            performanceL.sort(key=operator.itemgetter(2),reverse=True)#sort by train_valid AUC
            best_nn_performance=performanceL[0]
            i=1
            while i < len(performanceL):
               performance=performanceL[i] 
               utilities.delete_files([performance[0],performance[1]])
               del performance
               i+=1
            del [traindf,traindf2,validsetdf,performanceL] 
            return best_nn_performance
        else:#sequential training and validation
            del [traindf,traindf2,validsetdf]
            return train_validate_nn_sequential(reductsL,weka_train_file,weka_original_train_file,weka_validset_file,normalizeInputs,outputs,class_weight,epochs,iterative_training)
        
def train_validate_nn_sequential(reductsL,trainset,original_trainset,validset,normalizeInputs,outputs,class_weight,epochs,m):
    #train and validate network on reduced training sets and reduced validation sets     
    #m, iterations of training and testing a network topology with a specific random initial weights                
    best_train_valid_auc=0
    best_train_auc=0
    best_valid_auc=0
    best_train_tpr=0
    best_train_tnr=0
    best_train_fpr=0
    best_train_fnr=0
    best_valid_tpr=0
    best_valid_tnr=0
    best_valid_fpr=0
    best_valid_fnr=0
    best_nn_file=''
    best_weights=''
    best_topology=''
    num=random.randint(0,999999)
    best_nn_inputs_output=results_path+'nn'+str(num)+'.model_inputs_output.csv'
    j=0 #jth reduct
    for reduct in reductsL:
        traindf=utilities.arff_to_dataframe(trainset)
        validsetdf=utilities.arff_to_dataframe(validset)
        (_,c)=traindf.shape
        reduct=reduct.split(',')
        for i in range(len(reduct)):
            reduct[i]=int(reduct[i])-1 #change the indices of features to start from 0 rather than 1
        reduct.append(c-1)#add class variable index
        print('reduct '+str(j))
        print(reduct)
        traindf=traindf.iloc[:,reduct]
        validsetdf=validsetdf.iloc[:,reduct]
        if original_trainset != 'none':
            original_traindf=utilities.arff_to_dataframe(original_trainset)
            original_traindf=original_traindf.iloc[:,reduct]
        else:
            original_traindf=traindf  
        (_,c)=traindf.shape        
        (nn_file,train_valid_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,valid_auc,valid_tpr,valid_tnr,valid_fpr,valid_fnr,topology,weights,X_scaler)=network(normalizeInputs,outputs,class_weight,traindf,original_traindf,validsetdf,epochs,results_path,m)
        if train_valid_auc > best_train_valid_auc:
            best_train_valid_auc=train_valid_auc
            best_train_auc=train_auc
            best_valid_auc=valid_auc
            best_train_tpr=train_tpr
            best_train_tnr=train_tnr
            best_train_fpr=train_fpr
            best_train_fnr=train_fnr
            best_valid_tpr=valid_tpr
            best_valid_tnr=valid_tnr
            best_valid_fpr=valid_fpr
            best_valid_fnr=valid_fnr
            best_nn_file=nn_file
            best_weights=weights
            best_topology=topology
            utilities.get_model_inputs_output('df',validsetdf,best_nn_inputs_output)
        else:
            utilities.delete_files([nn_file])
        del [nn_file,train_valid_auc,valid_auc,valid_tpr,valid_tnr,valid_fpr,valid_fnr,train_auc,train_tpr,train_tnr,train_fpr,train_fnr]
        j+=1
    return (best_nn_file,best_nn_inputs_output,best_train_valid_auc,best_train_auc,best_train_tpr,best_train_tnr,best_train_fpr,best_train_fnr,best_valid_auc,best_valid_tpr,best_valid_tnr,best_valid_fpr,best_valid_fnr,best_topology,best_weights,X_scaler)
    
def train_validate_nn_parallel_step(j,reductsL,trainset,original_trainset,validset,normalizeInputs,outputs,class_weight,epochs,m):
        #train and validate a network on a reduced training set and a reduced validation set  
        #m, iterations of training and testing a network topology with a specific random initial weights                    
        reduct=reductsL[j]#indices of features start from 1
        traindf=utilities.arff_to_dataframe(trainset)
        validsetdf=utilities.arff_to_dataframe(validset)
        (_,c)=traindf.shape
        reduct=reduct.split(',')
        for i in range(len(reduct)):
            reduct[i]=int(reduct[i])-1 #change indices of features to start from 0 rather than 1
        reduct.append(c-1)#add class variable index
        print('reduct: '+str(j))
        print(reduct)
        traindf=traindf.iloc[:,reduct]
        validsetdf=validsetdf.iloc[:,reduct]
        if original_trainset != 'none':
            original_traindf=utilities.arff_to_dataframe(original_trainset)
            original_traindf=original_traindf.iloc[:,reduct]
        else:
            original_traindf=traindf  
        (_,c)=traindf.shape
        (best_nn_file,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr,best_nn_topology,best_nn_weights,X_scaler)=network(normalizeInputs,outputs,class_weight,traindf,original_traindf,validsetdf,epochs,results_path,m)
        num=random.randint(0,999999)
        best_nn_inputs_output=results_path+'nn'+str(num)+'.model_inputs_output.csv'
        utilities.get_model_inputs_output('df',validsetdf,best_nn_inputs_output)
        return (best_nn_file,best_nn_inputs_output,best_nn_train_valid_auc,best_nn_train_auc,best_nn_train_tpr,best_nn_train_tnr,best_nn_train_fpr,best_nn_train_fnr,best_nn_valid_auc,best_nn_valid_tpr,best_nn_valid_tnr,best_nn_valid_fpr,best_nn_valid_fnr,best_nn_topology,best_nn_weights,X_scaler)

###construct networks###
    #singlelayer_sizeL=[i+1 for i in range(36)]#[1,2,3,4,5,6,7,...,40]            
    #layer_sizeL=[i+5 for i in range(96)]#[5,6,7,...,100]
    #print('singlelayer_sizeL: '+str(singlelayer_sizeL))
    ###2-layer networks
    #weights=inputs*h1+h1*h2+h2*2
    #constraints on sizes of topologies: 
    #h1 >= h2, 
    #weights >=weights_lower
    #weights <=weights_upper 
    #nodes=[i+16 for i in range(24)]
    #twolayers_sizeL=[]
    #for j in range(len(nodes)):
    #    for k in range(len(nodes)):
    #        if nodes[j] >= nodes[k]:
    #            twolayers_sizeL.append([nodes[j],nodes[k]])  
    #twolayers_sizeL=[[20,20]]#best network topology
    #twolayers_sizeL=[[21,16]]#best network topology
    #twolayers_sizeL=[[23,22]]#best network topology, 596 weights
    #print('2-layer networks sizes: '+str(twolayers_sizeL))
    #multilayers_sizeL=twolayers_sizeL
    ###3-layer networks
    #weights=inputs*h1+h1*h2+h2*h3+h3*2
    #constraints on sizes of topologies: 
    #weights >=weights_lower
    #weights <=weights_upper    
    #threelayers_sizeL=[]
    #ws_lower=596
    #ws_upper=600
    #nodes=[i+5 for i in range(24)]
    #for j in range(len(nodes)):
    #    for k in range(len(nodes)):
    #        for l in range(len(nodes)):
    #            ws=(c-1)*nodes[j]+nodes[j]*nodes[k]+nodes[k]*nodes[l]+2*nodes[l]
    #            if ws >= ws_lower and ws <= ws_upper:
    #                threelayers_sizeL.append([nodes[j],nodes[k],nodes[l]])
    #print('3-layer networks sizes: '+str(threelayers_sizeL))
    #multilayers_sizeL=threelayers_sizeL
    #rescale inputs to [0,1]
    
def main(dataset,results_path2,logfile,trainsets1trainsets2Path2,weka_path,java_memory,option ='split_train_valid_test_sets',balanced_trainset_size=1984,fitness2='find_reducts',mini_features2=10,max_features2=15,number_of_reducts2=40,normalizeInputs='mini_max',trainset_size=0.66,testset_size=0.17,validset_size=0.17,iteration=1,logfile_option='w',networkstype2='multilayer', singlelayer_sizeL2=[10], multilayers_sizeL2=[[16,16]],outputs=2,epochs=1000,iterations=100,degree=4,k=30,seed0='random seed',seed='123456',repeat2=10,learning_rate2=0.002,iterative_training2=1,parallel_train_validate_nn_of_random_initial_weights2=True,parallel_training_validate2=False,ga_path2="d:\\EIS preterm prediction\\ga_rsfs.jar"):
    global networkstype
    global singlelayer_sizeL
    global multilayers_sizeL
    global results_path
    global mini_features
    global max_features
    global number_of_reducts
    global iterative_training
    global trainsets1trainsets2Path
    global fitness
    global ga_path
    global repeat #train a topology a number of times with each time initializing weights to random values before training
    global parallel_training_validate
    global learning_rate
    global parallel_train_validate_nn_of_random_initial_weights
    
    trainsets1trainsets2Path=trainsets1trainsets2Path2
    mini_features=int(mini_features2)
    max_features=int(max_features2)
    number_of_reducts=number_of_reducts2
    networkstype=networkstype2
    singlelayer_sizeL=singlelayer_sizeL2
    multilayers_sizeL=multilayers_sizeL2
    results_path=results_path2
    iterative_training=iterative_training2
    fitness=fitness2
    ga_path=ga_path2
    repeat=int(repeat2)
    learning_rate=learning_rate2
    parallel_training_validate=parallel_training_validate2
    parallel_train_validate_nn_of_random_initial_weights=parallel_train_validate_nn_of_random_initial_weights2
    
    data=pd.read_csv(dataset)
    (_,c)=data.shape
    cols=list(data.columns)
    data=prep.convert_targets2(data,c-1)
    for col in cols[:-1]:#all columns except the last one (targets column)
        data.astype({col:float})#convert inputs to float type
    #neg, pos = np.bincount(data[cols[c-1]])
    #total = neg + pos
    #print('Total instances: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    #weight_for_0 = (1 / neg)*(total)/2.0 
    #weight_for_1 = (1 / pos)*(total)/2.0
    weight_for_0 = 1
    #weight_for_1 = 8
    weight_for_1 = 1    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    #print('Weight for class 0: {:.2f}'.format(weight_for_0))
    #print('Weight for class 1: {:.2f}'.format(weight_for_1))
    train_aucL=[]
    test_aucL=[]    
    logfile=os.path.normpath(logfile)
    file=open(logfile,'w+')
    file.write('====workflow_nn is running====\n')
    file.write('####Parameters Setting####\n')
    file.write('dataset='+dataset+'\n')
    file.write('results path='+results_path+'\n')
    file.write('iterations='+str(iterations)+'\n')
    file.write('option='+option+'\n')
    print('option='+option)
    if option =='split_train_valid_test_sets':
        file.write('training set percentage='+str(trainset_size*100)+'\n')
        file.write('validation set percentage='+str(validset_size*100)+'\n')
        file.write('test set percentage='+str(testset_size*100)+'\n')        
    else:        
        file.write('trainsets1trainsets2Path='+trainsets1trainsets2Path2+'\n')
    file.write('normalization of inputs='+normalizeInputs+'\n')
    file.write('degree of polynomial features='+str(degree)+'\n')
    file.write('number of features to select using information gain='+str(k)+'\n')
    file.write('size of balanced training set after oversampling='+str(balanced_trainset_size)+'\n')
    file.write('number of feature subsets selected by genetic algorithm='+str(number_of_reducts)+'\n')
    file.write('fitness='+fitness+'\n')
    file.write('minimum size of feature subsets selected by genetic algorithms='+str(mini_features)+'\n')
    file.write('maximum size of feature subsets selected by genetic algorithms='+str(max_features)+'\n')
    if mini_features == -1 and max_features == -1:
        file.write('select feature subsets of random sizes from all the reducts found by genetic algorithm\n')
    file.close()
    performanceL=[]#list of (iteration,train_test_auc,train_auc,train_tpr,train_tnr,test_auc,test_tpr,test_tnr) where train_test_auc=(train_auc+test_auc)
    train_aucL=[]
    test_aucL=[]
    iterL=[]
    if option == 'split_train_valid_test_sets' or option == 'trainsets1' or option == 'trainsets2':
        iterL = [i for i in range(int(iterations))]
    elif option == 'ith_iteration of trainsets1' or option == 'ith_iteration of trainsets2' or option == 'ith_iteration of split_train_valid_test_sets':
        iterL = [int(iteration)] #ith iteration 
    for i in iterL:    
        nn=results_path+'nn'+str(i)+'.h5'#best network of ith iteration
        nn_inputs_output=results_path+'nn'+str(i)+'.model_inputs_output.csv'
        nn_scaler=results_path+'nn_scaler'+str(i)+'.joblib' #normalization parameters setting of best network of ith iteration
        print(i)
        file=open(logfile,'a')
        file.write(str(i)+'\n')
        print("\t\t\titeration: "+str(i))
        file.write('seed (oversampling): '+str(seed)+'\n')
        print('seed (oversampling): '+str(seed))
        #seed0=random.randint(0,2**32-1) #ith seed for data split 
        seed0=random.randint(0,999999) #ith seed for data split
        file.write('seed0 (data split): '+str(seed0)+'\n')
        print('seed0 (data split): '+str(seed0))        
        if option == 'split_train_valid_test_sets' or option == 'ith_iteration of split_train_valid_test_sets':
            if trainset_size+testset_size+validset_size != 1:
                sys.exit('trainset_size + testset_size + validset_size must be 1.')
            elif validset_size > 0:
                testset_validset_size=testset_size+validset_size    
                testset_size2=testset_size/(testset_size+validset_size)
                (train_set,validset_testset)=prep.split_train_test_sets(data,testset_validset_size,seed0,cols[c-1])
                validset_testset.to_csv(results_path+'validset_testset.csv',index=False)
                validset_testset=pd.read_csv(results_path+'validset_testset.csv')
                (valid_set,test_set)=prep.split_train_test_sets(validset_testset,testset_size2,seed0,cols[c-1])
                file.writelines(["\t\t\titeration: "+str(i)+"\n","\t\t\t1.Split dataset into training set, validation set and test set\n"])
                file.close()
            elif validset_size == 0:
                (train_set,test_set)=prep.split_train_test_sets(data,testset_size,seed0,cols[c-1])                     
                print('\t\t\t1.Split dataset1 into training set and test set')
                file.write('\t\t\t1.Split dataset1 into training set and test set\n')
                file.close()                
            step=2
        elif option == 'trainsets1' or option == 'ith_iteration of trainsets1':#training sets for training PTB classifiers
            print('Train PTB classifiers on training set1s and test them on validation set1s')
            file.write('Train PTB classifiers on training set1s and test them on validation set1s.\n')
            file.close()
            if os.path.isfile(trainsets1trainsets2Path+'trainset1_'+str(i)+'.csv'):
                train_set=pd.read_csv(trainsets1trainsets2Path+'trainset1_'+str(i)+'.csv')
            else:
                sys.exit('trainset1_i does not exist')
            if os.path.isfile(trainsets1trainsets2Path+'validset1_'+str(i)+'.csv'):
                valid_set=pd.read_csv(trainsets1trainsets2Path+'validset1_'+str(i)+'.csv')
            elif os.path.isfile(trainsets1trainsets2Path+'testset1_'+str(i)+'.csv'):
                valid_set=pd.read_csv(trainsets1trainsets2Path+'testset1_'+str(i)+'.csv')
            else:
                sys.exit('validset1_i and testset1_i do not exist')
            step=1
        elif option == 'trainsets2' or option == 'ith_iteration of trainsets2' :#training sets for training filter2
            print('Train filter2s on training set2s and test them on validation set2s')
            file.write('Train filter2s on training set2s and test them on validation set2s.\n')
            file.close()
            if os.path.isfile(trainsets1trainsets2Path+'trainset2_'+str(i)+'.csv'):
                train_set=pd.read_csv(trainsets1trainsets2Path+'trainset2_'+str(i)+'.csv')
            else:
                sys.exit('trainset2_i does not exist')
            if os.path.isfile(trainsets1trainsets2Path+'validset2_'+str(i)+'.csv'):
                valid_set=pd.read_csv(trainsets1trainsets2Path+'validset2_'+str(i)+'.csv')
            elif os.path.isfile(trainsets1trainsets2Path+'testset2_'+str(i)+'.csv'):
                valid_set=pd.read_csv(trainsets1trainsets2Path+'testset2_'+str(i)+'.csv')
            else:
                sys.exit('validset2_i and testset2_i do not exist')
            step=1
        else:
            sys.exit('invalid option: '+str(option))
        (_,c)=train_set.shape
        train_set=prep.convert_targets2(train_set,c-1)
        valid_set=prep.convert_targets2(valid_set,c-1)
        train_set=prep.fill_missing_values('median','df',train_set,'none')        
        valid_set=prep.fill_missing_values('median','df',valid_set,'none')
        train_set.to_csv(results_path+'trainset.csv',index=False) 
        valid_set.to_csv(results_path+'validset.csv',index=False)
        (r,_)=train_set.shape        
        if int(balanced_trainset_size) > 1 and int(degree) >= 2 and int(k) >= 1:
            ###training set -> oversampling -> construct polynomial features -> information gain feature selection -> reduced balanced training set
            if int(balanced_trainset_size) > r: 
                step=wf.preprocess_trainset(step,3,results_path+'trainset.csv',results_path+'validset.csv',train_set,'none','trainset_balanced_reduced.arff','original_trainset_reduced.arff','validset_reduced.arff',results_path,logfile,balanced_trainset_size,degree,str(k),seed,weka_path,java_memory)    
                ga_data='trainset_balanced_reduced.arff'
            else:
                sys.exit('balanced_trainset_size: '+str(balanced_trainset_size)+' < original training set size:'+str(r)+'\n')     
        elif int(balanced_trainset_size) > 1 and int(degree) < 2 and int(k) >= 1:#do not generate polynomial features, but select features using information gain
            ###training set -> oversampling -> information gain feature selection -> reduced balanced training set
            if int(balanced_trainset_size) > r: 
                step=wf.oversample_trainset2(step,'no_noise',3,results_path+'trainset.csv',results_path+'validset.csv',train_set,'none','trainset_balanced_reduced.arff','original_trainset_reduced.arff','validset_reduced.arff',results_path,logfile,balanced_trainset_size,str(k),seed,weka_path,java_memory)
                ga_data='trainset_balanced_reduced.arff'
            else:
                sys.exit('balanced_trainset_size: '+str(balanced_trainset_size)+' < original training set size:'+str(r)+'\n')     
        elif int(balanced_trainset_size) > 1 and int(degree) < 2 and int(k) < 1:#do not both generate polynomial features and select features using information gain
            ###training set -> oversampling -> balanced training set
            if int(balanced_trainset_size) > r: 
                step=wf.oversample_trainset(step,'no_noise',3,results_path+'trainset.csv',results_path+'validset.csv',train_set,'none','trainset_balanced.arff','original_trainset.arff','validset.arff',results_path,logfile,balanced_trainset_size,seed,weka_path,java_memory)
                ga_data='trainset_balanced.arff'
            else:
                sys.exit('balanced_trainset_size: '+str(balanced_trainset_size)+' < original training set size:'+str(r)+'\n')     
        elif int(balanced_trainset_size) < 0 and int(degree) >= 2 and int(k) >= 1:
            ###training set -> construct polynomial features -> information gain feature selection -> reduced training set
            step=wf.preprocess_trainset2(step,'no_noise',results_path+'trainset.csv',results_path+'validset.csv',train_set,'none','trainset_reduced.arff','original_trainset_reduced.arff','validset_reduced.arff',results_path,logfile,degree,str(k),weka_path,java_memory)           
            ga_data='trainset_reduced.arff'
        elif int(balanced_trainset_size) < 0 and int(degree) < 2 and int(k) >= 1:
            ###training set -> information gain feature selection -> reduced training set
            step=wf.preprocess_trainset2(step,'no_noise',results_path+'trainset.csv',results_path+'validset.csv',train_set,'none','trainset_reduced.arff','original_trainset_reduced.arff','validset_reduced.arff',results_path,logfile,str(k),weka_path,java_memory)           
            ga_data='trainset_reduced.arff'
        else:
            print('invalid options for balanced_trainset_size, degree or k. Valid options: balanced_trainset_size > 1 or balanced_trainset_size < 0, degree > 1 or degree < 0, k > 1 or k < 0')                
        reductsfile=results_path+"ga_reducts"        
        print('\t\t\t'+str(step+1)+'.Genetic Algorithm Feature Selection')
        print('\t\t\t'+str(step+2)+'.Train and test neural networks on the unreduced and reduced training sets and test sets')
        print('\t\t\tdiscretization method: '+discretize_method+', bins: '+str(bins))
        file=open(logfile,'a')
        file.write('\t\t\t'+str(step+1)+'.Genetic Algorithm Feature Selection\n')
        file.write('\t\t\t\t discretization method: '+discretize_method+', bins: '+str(bins)+'\n')
        file.write('\t\t\t'+str(step+2)+'.Train and test neural networks on the unreduced and reduced training sets and test sets\n')
        file.close()       
        ##oversampling the classes only (do not perform polynomial features construction and information gain feature selection)
        #class0, class1 = np.bincount(train_set[cols[c-1]])
        #class0_size2=class0 #class0 is the majority class
        #class1_size2=class0
        #print('class 0 after oversampling: '+str(class0_size2))
        #print('class 1 after oversampling: '+str(class1_size2))
        #train_set=oversample_train_set(train_set,class0_size=class0_size2,class1_size=class1_size2,method='random',csv_balanced_train_file=results_path+"trainset_balanced_reduced.csv")
        #print('oversampling training set is done')                
        #discretize -> ga_rsfs -> training and testing classifiers on the reduced training set and test set
        print('ga feature selection')
        reductsfile=results_path+"ga_reducts"               
        if ga_data == 'trainset_balanced_reduced.arff':
            (reductsfile,trainset_discrete_arff)=wf.discretize_ga_rsfs(i,results_path+'trainset_balanced_reduced.arff',discretize_method,bins,populationSize,generations,crossover_prob,mutation_prob,fitness,number_of_reducts,ga_path,results_path,weka_path,java_memory)            
            best_performance=train_validate_nn(results_path+'trainset_balanced_reduced.arff',results_path+'original_trainset_reduced.arff',results_path+'validset_reduced.arff',reductsfile,normalizeInputs,outputs,epochs,class_weight,results_path,weka_path,java_memory)
        elif ga_data == 'trainset_balanced.arff':
            (reductsfile,trainset_discrete_arff)=wf.discretize_ga_rsfs(i,results_path+'trainset_balanced.arff',discretize_method,bins,populationSize,generations,crossover_prob,mutation_prob,fitness,number_of_reducts,ga_path,results_path,weka_path,java_memory)            
            best_performance=train_validate_nn(results_path+'trainset_balanced.arff',results_path+'original_trainset.arff',results_path+'validset.arff',reductsfile,normalizeInputs,outputs,epochs,class_weight,results_path,weka_path,java_memory)
        elif ga_data == 'trainset_reduced.arff':
            (reductsfile,trainset_discrete_arff)=wf.discretize_ga_rsfs(i,results_path+'trainset_reduced.arff',discretize_method,bins,populationSize,generations,crossover_prob,mutation_prob,fitness,number_of_reducts,ga_path,results_path,weka_path,java_memory)            
            best_performance=train_validate_nn(results_path+'trainset_reduced.arff',results_path+'original_trainset_reduced.arff',results_path+'validset_reduced.arff',reductsfile,normalizeInputs,outputs,epochs,class_weight,results_path,weka_path,java_memory)
        #   best_performance = (best_nn,                       #0   (best network of iterative training of a network topology)
        #                       best_nn_inputs_output,         #1
        #                       best_nn_train_validset_auc,    #2
        #                       best_nn_train_auc,             #3
        #                       best_nn_train_tpr,             #4
        #                       best_nn_train_tnr,             #5
        #                       best_nn_train_fpr,             #6
        #                       best_nn_train_fnr,             #7
        #                       best_nn_validset_auc,          #8
        #                       best_nn_validset_tpr,          #9
        #                       best_nn_validset_tnr,          #10 
        #                       best_nn_validset_fpr,          #11
        #                       best_nn_validset_fnr,          #12
        #                       best_nn_topology,              #13
        #                       best_nn_weights,               #14
        #                       X_scaler)                      #15                       
        utilities.mycopyfile(best_performance[0],nn)#copy best network of this iteration
        utilities.mycopyfile(best_performance[1],nn_inputs_output)#copy inputs and output of the best network of iteration
        utilities.delete_files([trainset_discrete_arff,best_performance[0],best_performance[1]])
        train_auc=best_performance[3]
        train_tpr=best_performance[4]
        train_tnr=best_performance[5]
        train_fpr=best_performance[6]
        train_fnr=best_performance[7]
        validset_auc=best_performance[8]
        nn_topology=best_performance[13]
        nn_weights=best_performance[14]
        dump(best_performance[15],nn_scaler)#copy normalization setting of the best network of ith iteration to file
        del best_performance
        #evaluate the best network of this iteration on the testset
        test_auc=-999
        test_tpr=-999
        test_tnr=-999
        test_fpr=-999
        test_fnr=-999
        testset_csv=''
        if option == 'trainsets1':
            if os.path.isfile(trainsets1trainsets2Path+'testset1_'+str(i)+'.csv'):
                testset_csv=trainsets1trainsets2Path+'testset1_'+str(i)+'.csv'
            else:
                sys.exit(trainsets1trainsets2Path+'testset1_'+str(i)+'.csv does not exist')
            test_set=pd.read_csv(testset_csv)
        elif option == 'trainsets2':
            if os.path.isfile(trainsets1trainsets2Path+'testset2_'+str(i)+'.csv'):
                testset_csv=trainsets1trainsets2Path+'testset2_'+str(i)+'.csv'
            else:
                sys.exit(trainsets1trainsets2Path+'testset2_'+str(i)+'.csv does not exist')
            test_set=pd.read_csv(testset_csv)
        (_,c)=test_set.shape
        test_set=prep.convert_targets2(test_set,c-1)
        test_set=prep.fill_missing_values('median','df',test_set,'none')  
        network=load_model(nn)
        testset_features=set(list(test_set.columns))
        file=open(nn_inputs_output,'r')
        model_features=file.readline().rstrip() #remove the trailing \n from the line
        model_features=set(model_features.split(','))
        file.close()
        if model_features.issubset(testset_features):#the inputs of model are original features
            print('inputs of model are original features')
            test_set=prep.reduce_data2('df',test_set,nn_inputs_output,'none')
        else:#the inputs of model are polynomial features
            print('inputs of model are polynomial features')
            if option == 'split_train_valid_test_sets' or option == 'ith_iteration of split_train_valid_test_sets':
                testset_csv=results_path+'testset.csv'
                test_set.to_csv(testset_csv,index=False)
            test_set=utilities.construct_poly_features_of_another_dataset('original_features',testset_csv,nn_inputs_output,'none','none')
        X_scaler=load(nn_scaler)
        (test_auc,test_tpr,test_tnr,test_fpr,test_fnr)=nn_predict(network,test_set,X_scaler)
        del [test_set,network,X_scaler]
        train_test_auc=train_auc+test_auc
        if option == 'ith_iteration of trainsets1' or option == 'ith_iteration of trainsets2':
            print('neural network: trainset auc='+str(train_auc)+', trainset tpr='+str(train_tpr)+', testset auc='+str(test_auc)+', testset tpr='+str(test_tpr))
            file=open(logfile,'a')
            file.write('neural network: trainset auc='+str(train_auc)+', trainset tpr='+str(train_tpr)+', testset auc='+str(test_auc)+', testset tpr='+str(test_tpr)+'\n')
            file.close()
            performance=(int(iteration),train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,validset_auc,nn_topology,nn_weights)
            return performance                
        elif option == 'split_train_valid_test_sets':
            ###summarize performance of classifiers so far
            performance=(int(i),train_test_auc,train_auc,train_tpr,train_tnr,train_fpr,train_fnr,test_auc,test_tpr,test_tnr,test_fpr,test_fnr,validset_auc,nn_topology,nn_weights)
            (performanceL,train_aucL,test_aucL)=utilities.summarize_results_nn(i,logfile,performance,performanceL,train_aucL,test_aucL)#L=list of (iteration,train_test_auc,train_auc,train_tpr,train_fnr,test_auc,test_tpr,test_fnr,nn_topology,nn_weights) where train_test_auc=(train_auc+test_auc)   
        
if __name__ == "__main__":    
    #bins=100 #for EIS data
    #bins=50 #for metabolite data
    bins=80 #for EIS + metabolite data
    #dataset="U:\\EIS preterm prediction\\EIS_Data\\EIS_Data\\438_V1_28inputs.csv",
    #dataset="U:\\EIS preterm prediction\\metabolite\\asymp_22wks_filtered_data_28inputs_no_treatment.csv",
    dataset="d:\\EIS preterm prediction\\filtered_data_28inputs.csv"
    results_path="d:\\EIS preterm prediction\\results\\workflow_nn\\filtered_data_28inputs\\\\"
    #results_path2="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow_nn\\asymp_22wks_filtered_data_28inputs_no_treatment\\",  
    trainsets1trainsets2Path="d:\\EIS preterm prediction\\trainsets1trainsets2\\filtered_data_28inputs\\trainsets66_percent\\"
    logfile=results_path+"logfile2.txt"
    fitness='find_reducts'
    #mini_features=15
    #max_features=30
    mini_features=25
    max_features=25
    #mini_features=10
    #max_features=25    
    #number_of_reducts=40
    #number_of_reducts=20
    #number_of_reducts=10
    number_of_reducts=5    
    #number_of_reducts=1
    #number_of_reducts=0
    normalizeInputs='mini_max'
    #normalizeInputs='zscore'
    #normalizeInputs='norm'
    #normalizeInputs='yb'
    #normalizeInputs='yb_zscore'
    #trainset_size=0.66
    #testset_size=0.17
    #validset_size=0.17
    #iteration=1
    logfile_option='w'
    option ='split_train_valid_test_sets'
    #option ='ith iteration of split_train_valid_test_sets'
    #option='trainsets1'
    #option='trainsets2'
    balanced_trainset_size=1984
    networkstype='multilayer'
    #singlelayer_sizeL2=[10]
    multilayers_sizeL=[[16,16]]
    outputs=2    #number of output units (if 1 output unit, a logistic output is created, else softmax units are created)
    iterations=100 #
    repeat2=10
    #repeat2=1 #no. of networks of the same topology but with different inital random weights 
    degree=4
    #degree=-1
    k='30'
    #k='20'
    #k=-1
    seed='123456' #oversampling seed
    iterative_training=0 #no. of times to retrain a network after it is trained for the 1st time
    learning_rate=0.002
    epochs=1000
    ga_path="d:\\EIS preterm prediction\\ga_rsfs.jar"
    parallel_train_validate_nn_of_random_initial_weights=True #parallelly train and validate networks of the same topology but different initial random weights
    parallel_training_validate=False #parallel train and validate network topologies with different inputs. When parallel_training_validate is True, parallel_train_validate_nn_of_random_initial_weights should be set to False. Because, cannot run 2 parallel for loops in parallel.
    #parallel_training_validate=True    
    num_jobs=8

    main(dataset=dataset,
         option=option,
         normalizeInputs=normalizeInputs,
         balanced_trainset_size=balanced_trainset_size,
         iterations=iterations,
         degree=degree,
         k=k,
         fitness2=fitness,
         mini_features2=mini_features,
         max_features2=max_features,
         number_of_reducts2=number_of_reducts,
         networkstype2=networkstype,
         multilayers_sizeL2=multilayers_sizeL,
         epochs=epochs,
         learning_rate2=learning_rate,
         repeat2=repeat2,
         outputs=outputs,
         iterative_training2=iterative_training,
         parallel_train_validate_nn_of_random_initial_weights2=parallel_train_validate_nn_of_random_initial_weights,
         parallel_training_validate2=parallel_training_validate,
         trainsets1trainsets2Path2=trainsets1trainsets2Path,                                         
         results_path2=results_path,
         logfile=logfile,
         seed=seed,
         ga_path2=ga_path,
         weka_path="C:\\Program Files\\Weka-3-7-10\\weka.jar",
         java_memory="2g")   