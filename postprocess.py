"""
Programs to process the outputs of classifiers
author: David Tian (dtian09@gmail.com; d.tian@sheffield.ac.uk)
"""
from types import ModuleType
import numpy as np
import pandas as pd
import re
import os, sys
import ModelsPredict as mp
#programs to process the outputs of models

def model_output(resultsfile):
    #output a model's predictions in descending order of probabilities
    #input: resultsfile, a weka results file with probabilities of predictions
    #output: predictions in descending order of probabilities
    #format of resultsfile
    #=== Predictions on test data ===
    #inst#     actual  predicted error prediction
    # 1        1:0        1:0       0.8 
    # 2        1:0        1:0       1 
    # 3        1:0        1:0       1 
    # 4        2:1        1:0   +   0.95 
    # 5        1:0        1:0       0.8 
    # 6        1:0        1:0       0.9 
    # 7        1:0        1:0       1 
    # 8        1:0        1:0       1 
    # 9        1:0        1:0       1 
    #10        1:0        1:0       0.95 
    #11        2:1        1:0   +   1 
    import operator, os, sys
    if os.path.isfile(resultsfile)==False:#resultsfile does not exist
        sys.exit(resultsfile+' does not exist.')
    fileL=[line.strip() for line in open(resultsfile)]
    predL=[]#list of tuples (patient no.,probability of preterm,predicted class,actual class)
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
    for i in range(len(fileL)):
       line=fileL[i]
       m=re.match('^(.+)\s+(\d:\d)\s+(\d:\d)\s+\+{0,1}\s+(0\.\d+)$',line)#probability of prediction e.g. 0.85
       if m:#prediction of labelled data with probability of prediction < 1 e.g. 0.85
          inst=m.group(1)
          actual_class=m.group(2)
          l=actual_class.split(':')
          actual_class=l[1]#actual class of instance
          output_class=m.group(3)
          l=output_class.split(':')
          output_class=l[1]#predicted class of instance
          if output_class=='0' or output_class=='No':#onterm prediction (prob of onterm > prob of preterm, threshold 0.5)
              onterm_prob=float(m.group(4))
              preterm_prob=1-onterm_prob
          else:#preterm prediction: '1' or Yes (prob of preterm > prob of onterm, threshold 0.5)
              preterm_prob=m.group(4)
          if output_class != actual_class:
              error='+'
          else:
              error=' '
          #count TP, TN, FP, FN, P and N
          if output_class==actual_class and (output_class=='Yes' or output_class=='1'):
              tp+=1
          elif output_class==actual_class and (output_class=='No' or output_class=='0'):
              tn+=1
          elif (actual_class=='Yes' or actual_class=='1') and (output_class=='No' or output_class=='0'):
              fn+=1
          else:
              fp+=1   
          if actual_class=='Yes' or actual_class=='1':
              p+=1
          else:
              n+=1
          predL.append((inst,float(preterm_prob),output_class,actual_class,error))
       else:
           m=re.match('^(.+)\s+(\d:\d)\s+(\d:\d)\s+\+{0,1}\s+([01])$',line)#100% or 0% probability
           if m:#prediction of labelled data with probability of prediction = 1
               inst=m.group(1)
               actual_class=m.group(2)
               l=actual_class.split(':')
               actual_class=l[1]#actual class of instance
               output_class=m.group(3)
               l=output_class.split(':')
               output_class=l[1]
               if output_class=='0' or output_class=='No':#onterm prediction (prob of onterm > prob of preterm, threshold 0.5)
                   onterm_prob=float(m.group(4))
                   preterm_prob=1-onterm_prob
               else:#preterm prediction (prob of preterm > prob of onterm)
                   preterm_prob=m.group(4)
               if output_class != actual_class:
                   error='+'
               else:
                   error=' '
               #count TP, TN, P and N
               if output_class==actual_class and (output_class=='Yes' or output_class=='1'):
                   tp+=1
               elif output_class==actual_class and (output_class=='No' or output_class=='0'):
                   tn+=1
               elif (actual_class=='Yes' or actual_class=='1') and (output_class=='No' or output_class=='0'):
                   fn+=1
               else:
                   fp+=1   
               if actual_class=='Yes' or actual_class=='1':
                   p+=1
               else:
                   n+=1
               predL.append((inst,float(preterm_prob),output_class,actual_class,error))
           else:#prediction of unlabelled data with probability of prediction < 1
               m=re.match('^(.+)\s+\d:\?\s+(\d:\d)\s+(0\.\d+)$',line)#probability of prediction e.g. 0.85
               if m:
                   inst=m.group(1)
                   actual_class='unknown'
                   error='unknown'
                   output_class=m.group(2)
                   l=output_class.split(':')
                   output_class=l[1]
                   if output_class=='0' or output_class=='No':#onterm prediction (prob of onterm > prob of preterm)
                       preterm_prob=1-float(m.group(3))
                   else:#preterm prediction (prob of preterm > prob of onterm)
                       preterm_prob=m.group(3)
                   predL.append((inst,float(preterm_prob),output_class,actual_class,error))
               else:
                    m=re.match('^(.+)\s+\d:\?\s+(\d:\d)\s+([01])$',line)#100% or 0% probability
                    if m:#prediction of labelled data with probability of prediction =1
                       inst=m.group(1)
                       actual_class='unknown'
                       error='unknown'
                       output_class=m.group(2)
                       l=output_class.split(':')
                       output_class=l[1]
                       if output_class=='0' or output_class=='No':#onterm prediction (prob of onterm > prob of preterm)
                           preterm_prob=1-float(m.group(3))
                       else:#preterm prediction (prob of preterm > prob of onterm)
                           preterm_prob=m.group(3)
                       predL.append((inst,float(preterm_prob),output_class,actual_class,error))
    predL=sorted(predL,key=operator.itemgetter(1),reverse=True)#sort patients in descending order of preterm probability
    #TPR (sensitivity) and TNR (specificity)
    if p>0 and n>0:#compute performance for known actual targets
        tpr=tp/p
        tnr=tn/n
        fpr=fp/(fp+tn)
        fnr=fn/(fn+tp)
    return (predL,tpr,tnr,fpr,fnr)

def get_auc(resultsfile):
    fileL=[line.strip() for line in open(resultsfile)]
    if fileL!=[]:
       k=0
       found=False        
       while(found==False):
           line=fileL[k]
           m=re.match('^.+(ROC Area).+$',line)
           if m:
               found=True
           k+=1
       line=fileL[k]#go to next line which contains AUC at the 6th column
       resultsL=line.split(' ')
       resultsL=list(filter(None,resultsL))
       auc=resultsL[6]
    else:
       print('CVresultsfileL is empty')
    if auc=='?':#for unknown classes
        return auc
    else:
        return float(auc)

def plot_roc_curve(i,model_path,modeltype,testsets_path,results_path,weka_path,java_memory,fig_name=None):
    #input: i, model number e.g. rf0.model 
    #       modeltype, 'rf' or 'log_reg' 
    #       predL, a list of tupes (inst,float(preterm_prob),output_class,actual_class,error)
    #
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    MP=mp.ModelsPredict()
    model=model_path+modeltype+str(i)+'.model'
    if os.path.isfile(model) == False:
        sys.exit('model file does not exist: '+model)
    else:
       model_inputs_output_csv=model_path+modeltype+str(i)+'.model_inputs_output.csv'
       discrete_cuts_file=model_path+modeltype+str(i)+'.discrete_cuts'
       (testset_auc,predL,_,_,_,_)=MP.predict_using_weka_model('prediction list',testsets_path+'testset_good_readings'+str(i)+'.csv',discrete_cuts_file,model,modeltype,model_inputs_output_csv,results_path,weka_path,java_memory)           
       #print(predL)
       print('auc='+str(testset_auc))
       #targets=np.empty((len(predL),1),dtype=int)
       #scores=np.empty((len(predL),1),dtype=float)
       targets=[]
       scores=[]
       for i in range(len(predL)):
            tup=predL[i]
            #scores=np.insert(scores,i,0,1-float(tup[1]))#prob of class 0
            #scores=np.insert(scores,i,1,float(tup[1]))#prob of class 1
            #scores=np.insert(scores,i,float(tup[1]))#prob of class 1
            scores.append(float(tup[1]))
            targets.append(int(tup[3]))
       #for i in range(len(predL)):
       #     tup=predL[i]
       #     targets=np.insert(targets,i,int(tup[3]))
       #print(targets)
       #print(scores)
       fpr = dict()
       tpr = dict()
       roc_auc = dict()
       thresholds = dict()
       fpr[1], tpr[1], thresholds[1] = roc_curve(targets, scores,pos_label=1)
       print(fpr)
       print(tpr)
       print(thresholds)
       roc_auc[1] = auc(fpr[1], tpr[1])
       print(roc_auc[1])
       plt.figure()
       lw = 2
       plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % float(roc_auc[1]))
       plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
       plt.xlim([0.0, 1.0])
       plt.ylim([0.0, 1.05])
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title('Receiver operating characteristic')
       plt.legend(loc="lower right")
       plt.show()
       #plt.savefig(results_path+fig_name)
    
def write_to_file(auc,predL,tpr,tnr,fpr,fnr,resultsfile):
    #write prediction results in a file in descending order of probabilities
    fw=open(resultsfile,'w+')
    if auc=='?':
        fw.write('AUC: unknown\tTPR(sensitivity): unknown\tTNR(specificity): unknown\tFPR: unknown\tFNR: unknown\n')
    else:#auc is a number        
        fw.write('AUC: '+str(round(auc,3))+'\tTPR(sensitivity): '+str(round(tpr,3))+'\tTNR(specificity): '+str(round(tnr,3))+'\tFPR: '+str(round(fpr,3))+'\tFNR: '+str(round(fnr,3))+'\n')
    fw.write("Patient\t\tProb of preterm (class 1)\t\tPredicted class\t\tActual class\t\tError\n")
    for i in range(len(predL)):
          tup=predL[i]
          fw.write('  '+tup[0]+'\t\t  '+str(round(tup[1],2))+'\t\t\t'+tup[2]+'\t\t '+tup[3]+'\t\t'+tup[4]+'\n')
    fw.close()
    print('results have been written to '+resultsfile)

def write_to_file2(predL,tpr,tnr,fpr,fnr,logfile):
    #write prediction results in a file in descending order of probabilities
    if os.path.isfile(logfile)==False:#logfile does not exist, create a new one
       file=open(logfile,'w+')
    else:
       file=open(logfile,'a')
    file.write("Patient\t\tProb of preterm (class 1)\t\tPredicted class\t\tActual class\t\tError\n")
    for i in range(len(predL)):
          tup=predL[i]
          file.write('  '+tup[0]+'\t\t  '+str(round(tup[1],2))+'\t\t\t\t  '+tup[2]+'\t\t\t  '+tup[3]+'\t\t\t '+tup[4]+'\n')
    file.close()
    
def extract_rules(resultsfile,rulesfile):
    #extract rules from a decision tree output by C4.5
    #input: resultsfile, weka output file of c4.5
    #output: rulesfile, decision rules of the tree
    #format of decision tree:
    #
    #x7 x21 x22 x23 <= -22.154657
    #|   x11^2 x12 x18 <= -147.870696
    #|   |   x10^2 x13 x15 <= -297.63112
    #|   |   |   x13 x16 x17 x24 <= -13702.757626
    #|   |   |   |   x12 x14 x24 x26 <= -14467.740624: 0 (12.0)
    #|   |   |   |   x12 x14 x24 x26 > -14467.740624: 1 (37.0)
    #
    import re
    f=open(rulesfile,'w')
    L = [line.rstrip() for line in open(resultsfile)]
    cond_hash={}#key=position in a rule (position start from 0), value=a condition e.g. x25^2 x27 > -38785.361015'   
    rule_hash={}#key=attribute, value=condition of attribute e.g. key='x25^2 x27', value='> -38785.361015',key='x5 x15 x16 x22', value='> -20575.64255: 0'
    attr_hash={}#key=position of attribute, value=attribute
    #k=0#order of attribute in a rule 
    for i in range(len(L)):
        m=re.match('^\'{0,1}([x\d\^\s]+)\'{0,1}\s*([<=>]+)\s*([\-\d+\.]+)$',L[i])#match a 1st condition of a rule e.g. c <= -22.154657, x9 x12^3 <= 29.174856 or  x25^2 x27 > -38785.361015 etc.
        if m:#match the 1st condition of a rule
            val=m.group(3)
            cond_hash[0]=m.group(1)#insert attribute at position 0 in a rule
            if m.group(1) not in rule_hash:#this is a new condition
                rule_hash[m.group(1)]=m.group(2)+m.group(3)
            else:#this variable is in another condition of this rule
                test=rule_hash[m.group(1)]
                m2=re.match('^([<=>]+)\s*([\-\d+\.]+)$',test)
                if m2:
                    val_old=m2.group(2)
                    if val_old == val:#same value for the same variable, overwrite the old condition
                       rule_hash[m.group(1)]=m.group(2)+val
                    else:
                       test+='#'+m.group(2)+val #append this condition to the last condition e.g. key='x10^2 x13 x15', value='>-297.63112#<=-85.057599'
                       rule_hash[m.group(1)]=test
                else:
                    m3=re.match('^(.+#.+)$',test)
                    if m3:#this variable appears in numerous conditions
                       #print(m3.group(1))
                       test+='#'+m.group(2)+val #append this condition to the last condition e.g. key='x10^2 x13 x15', value='>-297.63112#<=-85.057599'
                       rule_hash[m.group(1)]=test
                    else:
                       m4=re.match('^([<=>]+)\s*([\-\d+\.]+):([\s\(\)\w\./]+)$',test)
                       if m4:#match the only condition and outcome of another rule, overwrite it
                          rule_hash[m.group(2)]=m.group(3)+val
            #print(attrL)
            #print(rule_hash)
        else: 
            m=re.match('^([\|\s]+)\'{0,1}([x\d\^\s]+)\'{0,1}\s*([<=>]+)\s*([\-\d+\.]+)$',L[i])#match a middle condition of a rule e.g. |    |   c <= -22.154657, x9 x12^3 <= 29.174856 or  |   x25^2 x27 > -38785.361015 etc.
            if m:
                val=m.group(4)
                l=m.group(1)
                l2=l.split(' ')
                pos=0
                for q in range(len(l2)):
                    if l2[q]=='|':
                        pos+=1
                attr_hash[pos]=m.group(2)#insert attribute
                if m.group(2) not in rule_hash:#this is a new condition
                    rule_hash[m.group(2)]=m.group(3)+val
                else:#this variable is in another condition of this rule
                    test=rule_hash[m.group(2)]
                    m2=re.match('^[<=>]+\s*([\-\d+\.]+)$',test)
                    if m2:
                        val_old=m2.group(1)
                        if val_old==val:#this is a conflict condition as same value for the same variable, overwrite the old condition
                           rule_hash[m.group(2)]=m.group(3)+val
                        else:
                           test+='#'+m.group(3)+val #append this condition to the last condition e.g. key='x10^2 x13 x15', value='>-297.63112#<=-85.057599'
                           rule_hash[m.group(2)]=test    
                    else:
                        m3=re.match('^(.+#.+)$',test)
                        if m3:#this variable appears in numerous conditions
                           #print(m3.group(1))
                           test+='#'+m.group(3)+val #append this condition to the last condition e.g. key='x10^2 x13 x15', value='>-297.63112#<=-85.057599'
                           rule_hash[m.group(2)]=test
                        else:
                            m4=re.match('^([<=>]+)\s*([\-\d+\.]+):([\s\(\)\w\./]+)$',test)
                            if m4:#match the last condition and outcome of another rule, overwrite it
                                rule_hash[m.group(2)]=m.group(3)+val
                    #print(attrL)
                #print(rule_hash)
            else:
                m=re.match('^([\|\s]*)\'{0,1}([x\d\^\s]+)\'{0,1}\s*([<=>]+)\s*([\-\d+\.]+):([\s\(\)\w\./]+)$',L[i])#match the last condition and the outcome of a rule e.g. |    |   c <= -22.154657: 0 (36.0), | x9 x12^3 <= 29.174856: 1 (45.0/1.0) etc.
                if m:
                    l=m.group(1)
                    m2=re.match('^([\|\s]+)$',l)
                    if m2:#match '|  |  |' etc
                        l2=m2.group(1)
                        l3=l2.split(' ')
                        pos=0
                        for q in range(len(l3)):
                            if l3[q]=='|':
                                pos+=1
                        attr_hash[pos]=m.group(2)#insert the attribute of the last condition of a rule
                    else:#first and only condition of a rule
                        attr_hash[0]=m.group(2)
                    rule_hash[m.group(2)]=m.group(3)+m.group(4)+':'+m.group(5).strip()
                    print(attr_hash)
                    print(rule_hash)
                    #print the rule
                    rule=''
                    for key,value in attr_hash.items():
                        if value==m.group(2): #get the key of the last attribute in the condition
                            pos=key
                            break
                    for j in range(pos):#get all the conditions up to the last condition in the rule
                        m=re.match('^([<=>]+)\s*([\-\d+\.]+):([\s\(\)\w\./]+)$',rule_hash[attr_hash[j]])
                        if m==None:#does not match the last condition and the outcome of another rule, so add it to rule
                           rule+=attr_hash[j]+rule_hash[attr_hash[j]]+' and '
                        print(rule)
                    rule+=attr_hash[pos]+rule_hash[attr_hash[pos]]
                    print(rule)
                    f.write(rule+'\n')
    f.close()

def get_rules_of_one_class(rulesfile,class_label,outfile):
    #get the rules of 1 class
    #input: rulesfile, output file by extract_rules function
    #       class_label
    #output: outfile, file containing the rules of class_label
    import re
    f=open(outfile,'w')
    rulesL=[line.rstrip() for line in open(rulesfile)]
    k=1
    for i in range(len(rulesL)):
        rule=rulesL[i]
        m=re.match('^.+:\s*(\w+)\s*[\(\)\d\.]+$',rule)
        if m:
            if class_label==m.group(1):
                print(str(k)+'.'+rule)
                f.write(str(k)+'.'+rule+'\n')
                k+=1
    f.close()
    
if __name__ == "__main__":

    #rf92.model and log_reg23.model"
    i=92
    model_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\15dec_filtered_data_28inputs\\"
    modeltype='rf'
    testsets_path="C:\\Users\\uos\\EIS preterm prediction\\results\\workflow1\\validate filters\\15dec_filtered_data_28inputs\\"
    results_path=model_path
    weka_path='weka-3-7-10.jar'
    plot_roc_curve(i,model_path,modeltype,testsets_path,results_path,weka_path,java_memory='2g')

    #(predL,tpr,tnr)=model_output('testresults_random_forest223.txt',0.5)
    '''
    (predL,tpr,tnr)=model_output('C:\\Users\\uos\\EIS preterm prediction\\log_reg_testset_output.txt',0.5)
    auc=get_auc('C:\\Users\\uos\\EIS preterm prediction\\testresults_logistic28.txt')
    write_to_file(auc,predL,tpr,tnr,'results.txt')
    print('AUC: '+str(round(auc,3)))
    print('TPR(sensitivity): '+str(round(tpr,3)))
    print('TNR(specificity): '+str(round(tnr,3)))
    print("Patient\t\tProb of preterm\t\tPredicted class\t\tActual class\t\tError\t\tThreshold")
    for i in range(len(predL)):
          tup=predL[i]
          print('  '+tup[0]+'\t\t  '+str(round(tup[1],2))+'\t\t\t  '+tup[2]+'\t\t\t  '+tup[3]+'\t\t  '+tup[4]+'\t\t  '+str(tup[5]))
    '''
    #extract_rules("U:\\EIS preterm prediction\\best training set\\c4_5_tree.txt","U:\\EIS preterm prediction\\best training set\\c4_5_rules.txt")
    #get_rules_of_one_class("U:\\EIS preterm prediction\\best training set\\c4_5_rules.txt",'1',"U:\\EIS preterm prediction\\best training set\\c4_5_rules_oneclass.txt")
