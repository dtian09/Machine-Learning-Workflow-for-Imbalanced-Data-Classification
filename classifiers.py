import os
#from keras import models
#from keras import layers
#from keras import regularizers
#import tensorflow as tf
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#import pandas as pd

class Classifier:
    
    def __init__(self,trainset,testset,weka_path,java_memory):
        #trainset, trainset in arff format
        #testset, testset in arff format
        #weka_path, e.g."c:\\Program Files\\Weka-3-7-10\\weka.jar"
        #java_memory, e.g. 3g
        self.trainset=trainset
        self.testset=testset
        self.weka_path=weka_path
        self.java_memory=java_memory
    '''
    def __init__(self,trainset_csv,testset_csv):
        self.trainset=pd.read_csv(trainset_csv)
        self.testset=pd.read_csv(testset_csv)
    '''     
    def set_trainset(self,trainset2):
        self.trainset=trainset2
    
    def set_testset(self,testset2):
        self.testset=testset2
                
    def log_reg_xval(self,k,ridge,model,results):
        #k-fold cross validate a logistic regression on a training set, then, save the model and the cv results
        #java -cp "C:\Program Files\Weka-3-7-10\weka.jar" weka.classifiers.functions.Logistic -t "C:\Program Files\Weka-3-7-10\data\breast-cancer.arff" -x 10 -R 1E-8 -i -M -1 -v -d "D:\downloads\log_reg.model > resultsfile"
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.Logistic -t \""+self.trainset+"\" -x "+k+" -R "+ridge+" -M -1 -v -i -d \""+model+"\" > \""+results+"\""    
        os.system(cmd)
        #print(cmd)
        
    def log_reg_train(self,ridge,model):
        #train a logistic regression and saves the model
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.Logistic -R "+ridge+" -M -1 -v -no-cv -t \""+self.trainset+"\" -d \""+model+"\""    
        os.system(cmd)
        #print(cmd)

    def log_reg_predict(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.Logistic -o -i -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        #print(cmd)

    def log_reg_predict2(self,model,new_data,results):
        #run model to predict the probabilities of classes
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.Logistic -p 0 -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        #print(cmd)
        
    def random_forest_xval(self,k,trees,tree_depth,seed,model,results):
        #k-fold cross validate a random forest on a training set, then save the model and the results
        #java -cp "C:\Program Files\Weka-3-7-10\weka.jar" weka.classifiers.trees.RandomForest -t "C:\Program Files\Weka-3-7-10\data\breast-cancer.arff" -x 10 -I 20 -depth 0 -K 0 -i -v -d "D:\downloads\rf.model"
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.trees.RandomForest -t \""+self.trainset+"\" -x "+k+" -I "+trees+" -depth "+tree_depth+" -K 0 -S "+seed+" -v -i -d \""+model+"\" > \""+results+"\""
        os.system(cmd)
        #print(cmd)
        
    def random_forest_train(self,trees,tree_depth,seed,model):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.trees.RandomForest -I "+trees+" -depth "+tree_depth+" -K 0 -S "+seed+" -v -no-cv -t \""+self.trainset+"\" -d \""+model+"\""
        os.system(cmd)
        #print(cmd)
    
    def random_forest_predict(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.trees.RandomForest -o -i -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        #print(cmd)

    def random_forest_predict2(self,model,new_data,results):
        #run model to output the probabilities of prediction
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.trees.RandomForest -p 0 -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        #print(cmd)
    
    def rbf_network_xval(self,k,model,results,numOfClusters='9',minStdDev='0.01',ridge='10',seed='123456789'):
        #k-fold CV a RBF network
        #weka.classifiers.functions.RBFNetwork -B 9 -S 123456789 -R 1.0E-8 -M -1 -W 0.01
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.RBFNetwork -t \""+self.trainset+"\" -x "+k+" -B "+numOfClusters+" -R "+ridge+" -S "+seed+" -M -1 -W "+minStdDev+" -i -v -d \""+model+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
     
    def rbf_network_train(self,model,numOfClusters='9',minStdDev='0.01',ridge='10',seed='123456789'):
        #train a RBF network
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.RBFNetwork -no-cv -t \""+self.trainset+"\" -B "+numOfClusters+" -R "+ridge+" -S "+seed+" -M -1 -W "+minStdDev+" -v -d \""+model+"\""
        os.system(cmd)
        print(cmd)
    
    def rbf_network_predict(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.RBFNetwork -o -i -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
    
    def rbf_network_predict2(self,model,new_data,results):
        #use a RBF network to predict probabilities of classes
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.RBFNetwork -p 0 -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        
    def rbf_classifier_xval(self,k,model,results,numOfFunctions='9',ridge='10',seed='123456789'):
        #k-fold CV a RBF classifier
        #weka.classifiers.functions.RBFClassifier -N 9 -R 0.01 -L 1.0E-6 -C 2 -G -O -A -P 1 -E 1 -S 1
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.RBFClassifier -t \""+self.trainset+"\" -x "+k+" -N "+numOfFunctions+" -R "+ridge+" -S "+seed+" -L 1.0E-6 -C 2 -G -O -A -P 1 -E 1 -v -i -d \""+model+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)                

    def rbf_classifier_train(self,model,numOfFunctions='9',ridge='10',seed='123456789'):
        #train a RBF classifier
        #weka.classifiers.functions.RBFClassifier -N 9 -R 0.01 -L 1.0E-6 -C 2 -G -O -A -P 1 -E 1 -S 1
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.RBFClassifier -no-cv -t \""+self.trainset+"\" -N "+numOfFunctions+" -R "+ridge+" -S "+seed+" -L 1.0E-6 -C 2 -G -O -A -P 1 -E 1 -v -d \""+model+"\""
        os.system(cmd)
        print(cmd)
        
    def rbf_classifier_predict(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.RBFClassifier -o -i -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
    
    def rbf_classifier_predict2(self,model,new_data,results):
        #use a RBF classifier to predict probabilities of classes
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.RBFClassifier -p 0 -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)

    def naive_bayes_train(self,model):
        #train a Naive Bayes and saves the model
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.bayes.NaiveBayes -v -no-cv -t \""+self.trainset+"\" -d \""+model+"\""    
        os.system(cmd)
        print(cmd)
    
    def naive_bayes_predict(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.bayes.NaiveBayes -o -i -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
    
    def naive_bayes_predict2(self,model,new_data,results):
        #run model to output the probabilities of prediction
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.bayes.NaiveBayes -p 0 -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
    
    def poly_libsvm_xval(self,k,C,degree,model,results,gamma='1/dimensionality'):
        if gamma=='1/dimensionality':
            gamma='0'
        seed="123456"
        #java -cp .\weka.jar weka.Run LibSVM -S 0 -K 1 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -B -x 5 -v -seed 123456 -t "D:\downloads\datasets2016\datasets\ionosphere_train.arff" -d "D:\downloads\datasets2016\datasets\libsvm.model" 
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.Run LibSVM -S 0 -K 1 -D "+degree+" -G "+gamma+" -R 0.0 -N 0.5 -M 40.0 -C "+C+" -E 0.001 -P 0.1 -B -seed "+seed+" -v -x "+k+" -t \""+self.trainset+"\" -d \""+model+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)

    def poly_libsvm_train(self,C,degree,model,gamma='1/dimensionality'):
        if gamma=='1/dimensionality':
            gamma='0'
        seed="123456"
        #java -cp .\weka.jar weka.Run LibSVM -S 0 -K 1 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -B -seed 123456 -t "D:\downloads\datasets2016\datasets\ionosphere_train.arff" -d "D:\downloads\datasets2016\datasets\libsvm.model"
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.Run LibSVM -S 0 -K 1 -D "+degree+" -G "+gamma+" -R 0.0 -N 0.5 -M 40.0 -C "+C+" -E 0.001 -P 0.1 -B -seed "+seed+" -v -no-cv -t \""+self.trainset+"\" -d \""+model+"\""
        os.system(cmd)
        print(cmd)
    
    def poly_libsvm_predict(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.Run LibSVM -o -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
    
    def poly_libsvm_predict2(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.Run LibSVM -p 0 -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
        
    def poly_svm_xval(self,k,C,degree,model,results,preprocess='normalize'):
        if preprocess=='normalize':#mini-max normalize to [0,1]
            preprocess='0'
        elif preprocess=='zscore':
            preprocess='1'
        elif preprocess==None:#no normalize or zscore
            preprocess='2'
        #weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.1" -calibrator "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4"
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.SMO -C "+C+" -L 0.001 -P 1.0E-12 -M -N "+preprocess+" -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 0 -E "+degree+"\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\" -t \""+self.trainset+"\" -x "+k+" -v -d \""+model+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)

    def poly_svm_train(self,C,degree,model,preprocess='normalize'):
        if preprocess=='normalize':#mini-max normalize to [0,1]
            preprocess='0'
        elif preprocess=='zscore':
            preprocess='1'
        elif preprocess==None:#no normalize or zscore
            preprocess='2'
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.SMO -C "+C+" -L 0.001 -P 1.0E-12 -M -N "+preprocess+" -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 0 -E "+degree+"\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\" -v -no-cv -t \""+self.trainset+"\" -d \""+model+"\""
        os.system(cmd)
        print(cmd)
    
    def poly_svm_predict(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.SMO -o -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
    
    def poly_svm_predict2(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.SMO -p 0 -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
        
    def rbf_svm_xval(self,k,C,gamma,model,results,preprocess='normalize'):
        if preprocess=='normalize':#mini-max normalize to [0,1]
            preprocess='0'
        elif preprocess=='zscore':
            preprocess='1'
        elif preprocess==None:#no normalize or zscore
            preprocess='2'
        #weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.1" -calibrator "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4"
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.SMO -C "+C+" -L 0.001 -P 1.0E-12 -M -N "+preprocess+" -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 0 -G "+gamma+"\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\" -t \""+self.trainset+"\" -x "+k+" -v -d \""+model+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)

    def rbf_svm_train(self,C,gamma,model,preprocess='normalize'):
        if preprocess=='normalize':#mini-max normalize to [0,1]
            preprocess='0'
        elif preprocess=='zscore':
            preprocess='1'
        elif preprocess==None:#no normalize or zscore
            preprocess='2'
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.SMO -C "+C+" -L 0.001 -P 1.0E-12 -M -N "+preprocess+" -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 0 -G "+gamma+"\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\" -v -no-cv -t \""+self.trainset+"\" -d \""+model+"\""
        os.system(cmd)
        print(cmd)
    
    def rbf_svm_predict(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.SMO -o -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
    
    def rbf_svm_predict2(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.classifiers.functions.SMO -p 0 -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
    
    def fuzzy_classifier_xval(self,k,model,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.Run MultiObjectiveEvolutionaryFuzzyClassifier -generations 40 -populationSize 100 -seed 1 -maxSimilarity 0.4 -minV 30.0 -maxV 2.0 -maxRules 12 -maxLabels 3 -evaluationMeasure 1 -algorithm 0 -reportFrequency 20 -logFile \"C:\\Program Files\\Weka-3-9-4\" -num-decimal-places 4 -t \""+self.trainset+"\" -x "+k+" -v -d \""+model+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
    
    def fuzzy_classifier_xval_testing(self,k,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.Run MultiObjectiveEvolutionaryFuzzyClassifier -generations 40 -populationSize 100 -seed 1 -maxSimilarity 0.4 -minV 30.0 -maxV 2.0 -maxRules 12 -maxLabels 3 -evaluationMeasure 1 -algorithm 0 -reportFrequency 20 -logFile \"C:\\Program Files\\Weka-3-9-4\" -num-decimal-places 4 -t \""+self.trainset+"\" -x "+k+" -v -d \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
        
    def fuzzy_classifier_train_test(self,model,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.Run MultiObjectiveEvolutionaryFuzzyClassifier -generations 40 -populationSize 100 -seed 1 -maxSimilarity 0.4 -minV 30.0 -maxV 2.0 -maxRules 12 -maxLabels 3 -evaluationMeasure 1 -algorithm 0 -reportFrequency 20 -logFile \"C:\\Program Files\\Weka-3-9-4\" -num-decimal-places 4 -t \""+self.trainset+"\" -T \""+self.testset+"\" -d \""+model+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
    
    def fuzzy_classifier_predict(self,model,new_data,results):
        cmd="java -Xmx"+self.java_memory+" -cp \""+self.weka_path+"\" weka.Run MultiObjectiveEvolutionaryFuzzyClassifier -o -l \""+model+"\" -T \""+new_data+"\" > \""+results+"\""
        os.system(cmd)
        print(cmd)
        
if __name__ == "__main__":
    #trainset="D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\trainset.arff"
    #testset="D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\testset.arff"
    trainset="D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\trainset.arff"
    testset="D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\testset.arff"
    
    weka_path="c:\\Program Files\\Weka-3-9-4\\weka.jar"
    java_memory='2g'
    from utilities import arff_to_dataframe, fill_missing_values
    from preprocess import convert_csv_to_arff
    df=arff_to_dataframe(trainset)
    df=fill_missing_values('median','df',df)
    df.to_csv("D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\trainset_no_missing.csv",index=False)
    convert_csv_to_arff("D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\trainset_no_missing.csv","D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\trainset_no_missing.arff","last:0,1",weka_path,java_memory)
    df=arff_to_dataframe(testset)
    df=fill_missing_values('median','df',df)
    df.to_csv("D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\testset_no_missing.csv",index=False)
    convert_csv_to_arff("D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\testset_no_missing.csv","D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\testset_no_missing.arff","last:0,1",weka_path,java_memory)
    #c=Classifier("D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\trainset_no_missing.arff","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\testset_no_missing.arff",weka_path,java_memory)
    c=Classifier("D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\trainset_no_missing.arff","D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\testset_no_missing.arff",weka_path,java_memory)

    #c.fuzzy_classifier('3',"D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\fuzzyrules.model","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\results3foldCV.txt")
    #c.fuzzy_classifier_train_test("D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\fuzzyrules.model","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\train_test_results.txt")
    c.fuzzy_classifier_train_test("D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\fuzzyrules.model","D:\\EIS preterm prediction\\fuzzy classifiers\\model3\\train_test_results.txt")

    #c.fuzzy_classifier_xval('5',"D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\fuzzyrules.model","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\results5foldCV.txt")
    #c.fuzzy_classifier_xval_testing('5',"D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\fuzzyrules.model","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\testset_no_missing.arff","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\results5foldCV_and_testing.txt")
    #c.fuzzy_classifier_xval_testing('3',"D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\fuzzyrules.model","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\testset_no_missing.arff","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\results3foldCV_and_testing.txt")
    #c.fuzzy_classifier_xval_testing('10',"D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\fuzzyrules.model","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\testset_no_missing.arff","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\results10foldCV_and_testing.txt")

    #c.fuzzy_classifier_predict("D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\fuzzyrules.model","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\testset_no_missing.arff","D:\\EIS preterm prediction\\fuzzy classifiers\\model2\\test_results5foldCV.txt")    