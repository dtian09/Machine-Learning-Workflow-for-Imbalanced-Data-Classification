import numpy as np
import pandas as pd
import preprocess as prep
import random

class SelectData:
    'class to select training sets and test sets'
    
    def __init__(self) :
        self.demo_data_of_ids='demo_data_of_ids.csv' #demo data of the patients in the best training set (balanced_train_resample97.csv)
        self.V1_30inputs_demographics='438_V1_30inputs_demographics.csv'
        self.filtered_data_demographics='filtered_data_and_demographics_30inputs.csv'#filtered EIS of the 438 patients
        
    def select_eis_features(self):
        #return: indices of the 28 eis features and class variable in "438_V1_30inputs_demographics.csv"
        indx=[i+13 for i in range(29)]#28 EIS features and class variable
        return indx
    '''
    def get_data_from_poly_features_data(csv_poly_features_data,csv_data):
        #get the data of original features corresponding to a polynominal features data
        #return: a dataframe of the original features data (28 features and class variable)
        #csvfile1="C:\\Users\\uos\\EIS preterm prediction\\resample\\poly features\\select features from training sets\\balanced_train_resample97.csv"
        idsL=prep.get_patients_ids(csv_poly_features_data)
        indx=select_eis_features()
        data=pd.read_csv("438_V1_30inputs_demographics.csv",low_memory=False)
        eisdata=data[:,indx]  
        return eisdata
    '''
    def compare_preterm_patients_ids_with_best_training_set(self,csvdata):
        #compare ids of preterm patients of a training set with ids of preterm patients of the best training set
        #input: csv file of 28 EIS features data
        data=pd.read_csv(self.demo_data_of_ids)
        preterm=data.loc[lambda df: data['before37weeksCell']=='Yes']
        idsL=preterm.loc[:,'hospital_id']
        print('no. of preterm in demo_data_of_ids.csv: ',str(len(idsL)))
        data2=pd.read_csv(csvdata)
        preterm2=data2.loc[lambda df: data2['before37weeksCell']=='Yes']
        preterm2.to_csv('preterm2.csv',index=False)
        idsL2=prep.get_patients_ids2('preterm2.csv')#ids of preterm patients in the 91th training set selected using the 5 categories percentages
        print('no. of preterm in ',csvdata,': ',str(len(idsL2)))
        common=set(idsL).intersection(set(idsL2))
        diff1=set(idsL)-set(idsL2)
        diff2=set(idsL2)-set(idsL)
        print('no. of preterm patients in both the best training set and the 91th training set: ',str(len(common)))
        print('no. of preterm patients only in the best training set, but not in the 91th training set: ',str(len(diff1)))
        print('no. of preterm patients only in the 91th training set, but not in the best training set: ',str(len(diff2)))

    def compare_preterm_patients_ids(self,csvdata1,csvdata2):
        #compare preterm patients ids of csvdata1 with those of csvdata2
        #input: csvdata1, csv file1 of 28 EIS features
        #       csvdata2, csv file2 of 28 EIS features
        #output: common, preterm patients in csvdata1 and csvdata2
        #        diff1, preterm patients in csvdata1 but not in csvdata2
        #        diff2, preterm patients in csvdata2 but not in csvdata1
        #        total_diff, diff1 + diff2
        data=pd.read_csv(csvdata1)
        preterm=data.loc[lambda df: data['before37weeksCell']=='Yes']
        preterm.to_csv('preterm.csv',index=False)
        idsL=prep.get_patients_ids2('preterm.csv')#ids of preterm patients in the csvdata1
        print('no. of preterm in demo_data_of_ids.csv: ',str(len(idsL)))
        data2=pd.read_csv(csvdata2)
        preterm2=data2.loc[lambda df: data2['before37weeksCell']=='Yes']
        preterm2.to_csv('preterm2.csv',index=False)
        idsL2=prep.get_patients_ids2('preterm2.csv')#ids of preterm patients in the csvdata2
        print('no. of preterm in ',csvdata2,': ',str(len(idsL2)))
        common=set(idsL).intersection(set(idsL2))
        diff1=set(idsL)-set(idsL2)
        diff2=set(idsL2)-set(idsL)
        print('no. of preterm patients in both csvdata1 and csvdata2: ',str(len(common)))
        print('no. of preterm patients only in csvdata1, but not in csvdata2: ',str(len(diff1)))
        print('no. of preterm patients only in csvdata2, but not in csvdata1: ',str(len(diff2)))
        print('total no. of preterm patients in either csvdata1 or csvdata2 only: ',str(len(diff1)+len(diff2)))
        return (len(common),len(diff1),len(diff2),len(diff1)+len(diff2))
    
    def count_preterm_patients_of_demographic_types(self,csv_poly_features_data,demofile):
        #count the percentage of each following demographic type of preterm patients in a dataset:
        #1.	13_cervical_lenCell <= 17: Yes
        #2.	Ethnicity = White AND Alcohol_in_pregnancy=No AND Non_prescribed_drugs_in_pregnancy=No AND Age > 25: Yes
        #3.	Smoker=No and Non_prescribed_drugs_in_pregnancy=No and Number_of_previous_pregnancies>1 and 13_cervical_lenCell<=18: before37weeksCell=Yes
        #4.	13_cervical_lenCell > 17 and Alcohol_in_pregnancy=Yes: Yes
        #5.	13_cervical_lenCell <= 27 and 13_cervical_lenCell > 17 and Alcohol_in_pregnancy = No and 15_ffn_level_valueCell > 8 and 15_ffn_level_valueCell <= 30: Yes
        #input: a csv file of poly features data ("C:\\Users\\uos\\EIS preterm prediction\\resample\\poly features\\select features from training sets\\balanced_train_resample97.csv")
        #       demofile, 'filtered_data_and_demographics_30inputs.csv' or '438_V1_30inputs_demographics.csv'
        #return: proportion of each type of preterm patients in the input data
        idsL=prep.get_patients_ids(csv_poly_features_data,demofile)
        #data=pd.read_csv(self.V1_30inputs_demographics,low_memory=False)
        data=pd.read_csv(demofile,low_memory=False)
        data2=data.loc[lambda df: np.isin(data['hospital_id'],idsL)]#get the demo graphic data and the eis data of the patients in csv_poly_features_data
        (r,_)=data2.shape
        print(str(r)+' patients of '+csv_poly_features_data+' has demographic information in '+demofile)
        #(t1,t2,t3,t4,t5,p,indxS)=self.satisfy_rules(data2)
        (t1,t2,t3,t4,t5,p,indxS)=self.satisfy_rules2(data2)
        preterm=data2.loc[lambda df: np.logical_or(data2['before37weeksCell']=='Yes',data2['before37weeksCell']==1)]
        (r,_)=preterm.shape
        print('total number of preterm patients: ',r)
        print('A patient is counted k times, if it satifies k rules')
        print('number of preterm patients satisfying at least 1 rule: ',str(len(indxS)))
        print('number of preterm patients satisfying numrous rules: ',str(p))
        print('percentage of type 1 preterm patients: ',str(round(t1/r*100,3)),'%')
        print('percentage of type 2 preterm patients: ',str(round(t2/r*100,3)),'%')
        print('percentage of type 3 preterm patients: ',str(round(t3/r*100,3)),'%')
        print('percentage of type 4 preterm patients: ',str(round(t4/r*100,3)),'%')
        print('percentage of type 5 preterm patients: ',str(round(t5/r*100,3)),'%')
        print('percentage of the 5 types of preterm patients: ',str(round((len(indxS))/r*100,2)),'%')
        return (round(t1/r,6),round(t2/r,6),round(t3/r,6),round(t4/r,6),round(t5/r,6))
    
    def count_preterm_patients_of_poly_features_properties(self,csv_poly_features_data,rules):
        #count the percentage of each following type of preterm patients in a dataset:
        #1.if x12 x14 x24 x26 <= 0.616152 AND x8 x11 x27^2 > 1970.982982 AND x9 x26^2 x27 > -24114.220548 then preterm
        #2.if x12 x14 x24 x26 <= 0.616152 AND x25^2 x27 <= -10288.074643 AND x3 x9 x26 x27 > 617115.379755 AND x6 x9 x26 x27 <= 544842.487511 AND x1^2 x15^2 > 82874.672947 then preterm
        #3.if x10^2 x13 x15 <= -71.817328 AND x9 x26^2 x27 > -84698.804034 AND x12 x14 x24 x26 > -10092.102417 AND x10^2 x13 x15 <= -85.057599 AND x7 x27^3 <= -78812.187917 AND x9 x12 x14 > -47.628541 AND x10 x12 x25 x27 > 5233.891303 then preterm        
        #4.if x9 x12^3 <= 20.615217 Then preterm
        #5.if x5 x15 x16 x22 <= -12997.728047 AND x5 x15 x16 x22 <= -18454.165577 AND x6 x12^3 <= 363.295587 AND x11^2 x12 x18 <= -271.405861 AND x11^2 x12 x18 > -523.960997 AND x1 x17 x18 x23 > -1142560.135815 AND x7 x21 x22 x23 <= -927304.981822 then preterm
        #6.if x5 x15 x16 x22 > -12997.728047 then preterm
        #7.if x6 x12^3 <= 353.137753 AND x11^2 x12 x18 <= -271.405861 AND x9 x12^3 <= 134.081455 AND x9 x26^2 x27 > -228708.503972 AND x1 x17 x18 x23 <= -154464.662097 AND x7 x21 x22 x23 > -539426.798594 then preterm
        #8.if x6 x12^3 <= 363.295587 AND x7 x21 x22 x23 <= -610844.978474 AND x10^2 x13 x15 > -213.678826 then preterm
        #9.if x10^2 x13 x15 > -290.612788 then preterm         
        #input: a csv file of poly features data e.g. "U:\EIS preterm prediction\best training set\train_poly_features97.csv"
        #       rules, a list of rules (strings)
        #return: proportion of each type of preterm patients in the input data
        import operator
        data=pd.read_csv(csv_poly_features_data,low_memory=False)
        (ts,p,indxS)=self.satisfy_rules_of_poly_features(data,rules)
        t1=ts[0]
        t2=ts[1]
        t3=ts[2]
        t4=ts[3]
        t5=ts[4]
        t6=ts[5]
        t7=ts[6]
        t8=ts[7]
        t9=ts[8]
        preterm=data.loc[lambda df: np.logical_or(data['before37weeksCell']=='Yes',data['before37weeksCell']==1)]
        (r,_)=preterm.shape
        print('total number of preterm patients: ',r)
        print('A patient is counted k times, if it satifies k rules')
        print('number of preterm patients satisfying at least 1 rule: ',str(len(indxS)))
        print('number of preterm patients satisfying numrous rules: ',str(p))
        print('percentage of type 1 preterm patients: ',str(round(t1/r*100,3)),'% ('+str(t1)+' patents)')
        print('percentage of type 2 preterm patients: ',str(round(t2/r*100,3)),'% ('+str(t2)+' patients)')
        print('percentage of type 3 preterm patients: ',str(round(t3/r*100,3)),'% ('+str(t3)+' patients)')
        print('percentage of type 4 preterm patients: ',str(round(t4/r*100,3)),'% ('+str(t4)+' patients)')
        print('percentage of type 5 preterm patients: ',str(round(t5/r*100,3)),'% ('+str(t5)+' patients)')
        print('percentage of type 6 preterm patients: ',str(round(t6/r*100,3)),'% ('+str(t6)+' patients)')
        print('percentage of type 7 preterm patients: ',str(round(t7/r*100,3)),'% ('+str(t7)+' patients)')
        print('percentage of type 8 preterm patients: ',str(round(t8/r*100,3)),'% ('+str(t8)+' patients)')
        print('percentage of type 9 preterm patients: ',str(round(t9/r*100,3)),'% ('+str(t9)+' patients)')
        print('===ranked categories by proportion of all preterm patients in whole dataset (percentages/100)===')
        fractions=[(round(t1/r,6),'type 1: '+rules[0]),
                   (round(t2/r,6),'type 2: '+rules[1]),
                   (round(t3/r,6),'type 3: '+rules[2]),
                   (round(t4/r,6),'type 4: '+rules[3]),
                   (round(t5/r,6),'type 5: '+rules[4]),
                   (round(t6/r,6),'type 6: '+rules[5]),
                   (round(t7/r,6),'type 7: '+rules[6]),
                   (round(t8/r,6),'type 8: '+rules[7]),
                   (round(t9/r,6),'type 9: '+rules[8])]
        fractions.sort(key=operator.itemgetter(0),reverse=True)
        for i in range(len(fractions)):
            print(fractions[i])
        fractions2=[round(t1/r,6),round(t2/r,6),round(t3/r,6),round(t4/r,6),round(t5/r,6),round(t6/r,6),round(t7/r,6),round(t8/r,6),round(t9/r,6)]
        print('sum of the percentages of the 1st 8 types: ',str(round(np.sum(fractions2[0:8])*100,3)),'%')
        print('percentage of preterm patients of the 9 types: ',str(round((len(indxS))/r*100,2)),'%')
        return fractions
    
    def satisfy_rules_of_poly_features(self,data,rules):
        #count patients satisfying the 9 rules (types):
        #1.if x12 x14 x24 x26 <= 0.616152 AND x8 x11 x27^2 > 1970.982982 AND x9 x26^2 x27 > -24114.220548 then preterm
        #2.if x12 x14 x24 x26 <= 0.616152 AND x25^2 x27 <= -10288.074643 AND x3 x9 x26 x27 > 617115.379755 AND x6 x9 x26 x27 <= 544842.487511 AND x1^2 x15^2 > 82874.672947 then preterm
        #3.if x10^2 x13 x15 <= -71.817328 AND x9 x26^2 x27 > -84698.804034 AND x12 x14 x24 x26 > -10092.102417 AND x10^2 x13 x15 <= -85.057599 AND x7 x27^3 <= -78812.187917 AND x9 x12 x14 > -47.628541 AND x10 x12 x25 x27 > 5233.891303 then preterm        
        #4.if x9 x12^3 <= 20.615217 Then preterm
        #5.if x5 x15 x16 x22 <= -12997.728047 AND x5 x15 x16 x22 <= -18454.165577 AND x6 x12^3 <= 363.295587 AND x11^2 x12 x18 <= -271.405861 AND x11^2 x12 x18 > -523.960997 AND x1 x17 x18 x23 > -1142560.135815 AND x7 x21 x22 x23 <= -927304.981822 then preterm
        #6.if x5 x15 x16 x22 > -12997.728047 then preterm
        #7.if x6 x12^3 <= 353.137753 AND x11^2 x12 x18 <= -271.405861 AND x9 x12^3 <= 134.081455 AND x9 x26^2 x27 > -228708.503972 AND x1 x17 x18 x23 <= -154464.662097 AND x7 x21 x22 x23 > -539426.798594 then preterm
        #8.if x6 x12^3 <= 363.295587 AND x7 x21 x22 x23 <= -610844.978474 AND x10^2 x13 x15 > -213.678826 then preterm
        #9.if x10^2 x13 x15 > -290.612788 then preterm        
        #input: data, a dataframe of polynomial features
        #       rules, a list of rules (strings)
        #output: (ts,p,indxS)
        t1=0#size of type 1
        t2=0#size of type 2
        t3=0#size of type 3
        t4=0#size of type 4
        t5=0#size of type 5
        t6=0
        t7=0
        t8=0
        t9=0
        indxS=set()#unique preterm patients satisfying at least 1 of the 9 rules
        p=0#no. of preterm patients satisfing numerous rules
        rs=data.index
        for r in rs:#if a patient satisfies k rules, it is counted k times
            k=0#no. of rules a patient satisfies
            if self.match_rule(data,r,rules[0]):
                t1+=1
                k+=1
                indxS.add(r)
            if self.match_rule(data,r,rules[1]):
                t2+=1
                k+=1
                indxS.add(r)
            if self.match_rule(data,r,rules[2]):
                t3+=1
                k+=1
                indxS.add(r)
            if self.match_rule(data,r,rules[3]):
                t4+=1
                k+=1
                indxS.add(r)
            if self.match_rule(data,r,rules[4]):
                t5+=1
                k+=1
                indxS.add(r)
            if self.match_rule(data,r,rules[5]):
                t6+=1
                k+=1
                indxS.add(r)
            if self.match_rule(data,r,rules[6]):
                t7+=1
                k+=1
                indxS.add(r)
            if self.match_rule(data,r,rules[7]):
                t8+=1
                k+=1
                indxS.add(r)
            if self.match_rule(data,r,rules[8]):
                t9+=1
                k+=1
                indxS.add(r)
            if k>1:#this patient satisfies numerous rules
                p+=1
            #print(i,'th patient satisfies ',k,' rules')
            ts=[t1,t2,t3,t4,t5,t6,t7,t8,t9]
        return (ts,p,indxS)
    
    def match_rule(self,data, r, rule):
        #match a rule against a row r (a patient) of data
        #input: data, a dataframe
        #       r, a row label
        #       rule, a string representing a rule e.g. "x12 x14 x24 x26 <= 0.616152 AND x8 x11 x27^2 > 1970.982982 AND x9 x26^2 x27 > -24114.220548:1"
        #return True if all the conditions are satisfied by this row of data or False otherwise
        import re
        ruleL=rule.split(':')
        condsL=ruleL[0]
        RHS=ruleL[1]
        condsL=condsL.split('AND')
        #check LHS of the rule
        result=True
        for i in range(len(condsL)):
            m=re.match('^\'{0,1}([x\d\^\s]+)\'{0,1}\s*([<=>]+)\s*([\-\d+\.]+)$',condsL[i].strip())
            if m:
                attr=m.group(1).strip()
                cond=m.group(2)
                val=float(m.group(3).strip())
                if cond=='<=':
                    result=result and data.at[r,attr]<=val
                elif cond=='>':
                    result =result and data.at[r,attr]>val
                elif cond=='=':
                    result =result and data.at[r,attr]==val
                else:
                    print('invalid condition: '+condsL[i])
                    break
            else:
                print(condsL[i]+'does not match pattern')
                break
        #check RHS of the rule
        if RHS=='1' or RHS=='Yes':
           result=result and (data.at[r,'before37weeksCell']==1 or data.at[r,'before37weeksCell']=='Yes')
        else:
           result=result and (data.at[r,'before37weeksCell']==0 or data.at[r,'before37weeksCell']=='No')
        return result                
        
    def satisfy_rules(self,data):
        #count patients satisfying the 5 rules (types):
        #1.	13_cervical_lenCell <= 17: Yes
        #2.	Ethnicity = White AND Alcohol_in_pregnancy=No AND Non_prescribed_drugs_in_pregnancy=No AND Age > 25: Yes
        #3.	Smoker=No and Non_prescribed_drugs_in_pregnancy=No and Number_of_previous_pregnancies>1 and 13_cervical_lenCell<=18: before37weeksCell=Yes
        #4.	13_cervical_lenCell > 17 and Alcohol_in_pregnancy=Yes: Yes
        #5.	13_cervical_lenCell <= 27 and 13_cervical_lenCell > 17 and Alcohol_in_pregnancy = No and 15_ffn_level_valueCell > 8 and 15_ffn_level_valueCell <= 30: Yes 
        #input: data, a dataframe of demographic information of the patients
        t1=0#size of type 1
        t2=0#size of type 2
        t3=0#size of type 3
        t4=0#size of type 4
        t5=0#size of type 5
        indxS=set()#unique preterm patients satisfying at least 1 of the 5 rules
        p=0#no. of preterm patients satisfing numerous rules
        (r,n)=data.shape
        for i in range(r):#if a patient satisfies k rules, it is counted k times
            k=0#no. of rules a patient satisfies
            if data.iat[i,11]<=17 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1): #type 1: 13_cervical_lenCell <= 17: Yes           
                t1+=1
                k+=1
                indxS.add(i)
            if data.iat[i,1]=='White' and data.iat[i,5]=='No' and data.iat[i,6]=='No' and data.iat[i,3]>25 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                t2+=1
                k+=1
                indxS.add(i)
            if data.iat[i,4]=='No' and data.iat[i,5]=='No' and data.iat[i,8] > 1 and data.iat[i,11]<=18 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                t3+=1
                k+=1
                indxS.add(i)
            if data.iat[i,11]>17 and data.iat[i,5]=='Yes' and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                t4+=1
                k+=1
                indxS.add(i)
            if data.iat[i,11]<=27 and data.iat[i,11] > 17 and data.iat[i,5]=='No' and data.iat[i,12]>8 and data.iat[i,12]<=30 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                t5+=1
                k+=1
                indxS.add(i)
            if k>1:#this patient satisfies numerous rules
                p+=1
            #print(i,'th patient satisfies ',k,' rules')
        return (t1,t2,t3,t4,t5,p,indxS)
    
    def satisfy_rules2(self,data):
        #count patients satisfying the 5 rules (types):
        #1.	If 13_cervical_lenCell <= 17 Then Preterm=1
        #2.	If 13_cervical_lenCell >17 and 13_cervical_lenCell <= 27 and BMI <= 22.4 Then Preterm=1
        #3.	Ethnicity = White AND Alcohol_in_pregnancy=No AND Non_prescribed_drugs_in_pregnancy=No AND Age > 25: Yes
        #   or 
        #3.	If 13_cervical_lenCell >17 and 13_cervical_lenCell <= 27 and Smoker=Yes and 15_ffn_level_valueCell <= 23 Then Preterm=1
        #4.	If 13_cervical_lenCell <=27.5 and Ethnicity=White Then Preterm=1
        #5.	If Non_prescribed_drugs_in_pregnancy=No AND Age > 21 AND number_previous_TOPs = 0 AND number_previous_early_miscarriages=1 Then Preterm=1
        #input: data, a dataframe of demographic information of the patients
        t1=0#size of type 1
        t2=0#size of type 2
        t3=0#size of type 3
        t4=0#size of type 4
        t5=0#size of type 5
        indxS=set()#unique preterm patients satisfying at least 1 of the 5 rules
        p=0#no. of preterm patients satisfing numerous rules
        (r,n)=data.shape
        for i in range(r):#if a patient satisfies k rules, it is counted k times
            k=0#no. of rules a patient satisfies
            if data.iat[i,11]<=17 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1): #type 1: 13_cervical_lenCell <= 17: Yes           
                t1+=1
                k+=1
                indxS.add(i)
            if data.iat[i,11]>17 and data.iat[i,11]<=27 and data.iat[i,7]<=22.4 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                t2+=1
                k+=1
                indxS.add(i)
            if data.iat[i,1]=='White' and data.iat[i,5]=='No' and data.iat[i,6]=='No' and data.iat[i,3]>25 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
            #if data.iat[i,11]>17 and data.iat[i,11]<=27 and data.iat[i,4]=='Yes' and data.iat[i,12]<=23 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                t3+=1
                k+=1
                indxS.add(i)
            if data.iat[i,11]<=27.5 and data.iat[i,1]=='White' and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                t4+=1
                k+=1
                indxS.add(i)
            if data.iat[i,6]=='No' and data.iat[i,3] > 21 and data.iat[i,10]==0 and data.iat[i,9]==1 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                t5+=1
                k+=1
                indxS.add(i)
            if k>1:#this patient satisfies numerous rules
                p+=1
            #print(i,'th patient satisfies ',k,' rules')
        return (t1,t2,t3,t4,t5,p,indxS)
    
    def select_train_test_sets_using_preterm_categories(self,data,rules,m,p1,p2,p3,p4,p5,prop_of_trainset,arff_trainset,arff_testset):
        #======Preterm Rules of best training set (test AUC 0.75, training AUC 0.829)======
        #e.g. 5 demographic preterm rules:
        #1.	13_cervical_lenCell <= 17: Yes
        #2.	Ethnicity = White AND Alcohol_in_pregnancy = No AND Non_prescribed_drugs_in_pregnancy = No AND Age > 25: Yes
        #3.	Smoker=No and Non_prescribed_drugs_in_pregnancy=No and Number_of_previous_pregnancies>1 and 13_cervical_lenCell<=18: before37weeksCell=Yes
        #4.	13_cervical_lenCell > 17 and Alcohol_in_pregnancy=Yes: Yes
        #5.	13_cervical_lenCell <= 27 and 13_cervical_lenCell > 17 and Alcohol_in_pregnancy = No and 15_ffn_level_valueCell > 8 and 15_ffn_level_valueCell <= 30: Yes
        #Each rule represents a category (group) of preterm patients in the best training set.
        #
        #input: data, dataframe representing "438_V1_30inputs_demographics.csv" or 'filtered_data_and_demographics_30inputs.csv'
        #       rules, a list of rules (strings)
        #       m, no. of pairs of training and test sets
        #       proportion of the training set (size of training set/size of whole dataset)
        #       p1, proportion of preterm category 1 (rule 1) 
        #       p2, proportion of preterm category 2 (rule 2) 
        #       p3, proportion of preterm category 3 (rule 3)
        #       p4, proportion of preterm category 4 (rule 4)
        #       p5, proportion of preterm category 5 (rule 5)
        #output: m pairs of training set, test set (i.e. test set = whole dataset - training set)
        #
        (r,_)=data.shape
        #(indx_c1,indx_c2,indx_c3,indx_c4,indx_c5,indx_onterm,indx_preterm)=self.preterm_demographic_categories(data)       
        #(indx_c1,indx_c2,indx_c3,indx_c4,indx_c5,indx_onterm,indx_preterm)=self.preterm_demographic_categories2(data)              
        (indx_c1,indx_c2,indx_c3,indx_c4,indx_c5,indx_onterm,indx_preterm)=self.preterm_poly_features_categories(data,rules)        
        print('preterm: ',str(len(indx_preterm)))
        print('on-term: ',str(len(indx_onterm)))
        print('category 1: '+rules[0])
        print('category 2: '+rules[1])
        print('category 3: '+rules[2])
        print('category 4: '+rules[3])
        print('category 5: '+rules[4])        
        print('size of preterm category 1 in whole dataset: ',str(len(indx_c1)))
        print('size of preterm category 2 in whole dataset: ',str(len(indx_c2)))
        print('size of preterm category 3 in whole dataset: ',str(len(indx_c3)))
        print('size of preterm category 4 in whole dataset: ',str(len(indx_c4)))
        print('size of preterm category 5 in whole dataset: ',str(len(indx_c5)))
        preterm_trainset=round(len(indx_preterm)*prop_of_trainset) #no. of preterm patients in the training set
        print('preterm_trainset: ',str(preterm_trainset))         
        c1=round(preterm_trainset*p1) #size of preterm category 1 in training set
        c2=round(preterm_trainset*p2)
        c3=round(preterm_trainset*p3)
        c4=round(preterm_trainset*p4)
        c5=round(preterm_trainset*p5)
        print('no. of preterm of category 1 in training set: ',str(c1))
        print('no. of preterm of category 2 in training set: ',str(c2))
        print('no. of preterm of category 3 in training set: ',str(c3))
        print('no. of preterm of category 4 in training set: ',str(c4))
        print('no. of preterm of category 5 in training set: ',str(c5))       
        for j in range(m):#create m pairs of training and test sets
            #select preterm patients to include in the training set
            print(j)
            if len(indx_c1) >= c1:
                indx_c1_trainset=random.sample(indx_c1,c1)#indices of category 1 patients to include in the training set 
            else:
                print('No. of category 1 patients to include in training set (',str(c1),') > total no. of category 1 patients in whole dataset (',str(len(indx_c1)),')')          
            if len(indx_c2) >= c2:
                indx_c2_trainset=random.sample(indx_c2,c2)
            else:
                print('No. of category 2 patients to include in training set (',str(c2),') > total no. of category 2 patients in whole dataset (',str(len(indx_c2)),')')          
            if len(indx_c3) >= c3:
                indx_c3_trainset=random.sample(indx_c3,c3)
            else:
                print('No. of category 3 patients to include in training set (',str(c3),') > total no. of category 3 patients in whole dataset (',str(len(indx_c3)),')')          
            if len(indx_c4) >= c4:
                indx_c4_trainset=random.sample(indx_c4,c4)
            else:
                print('No. of category 4 patients to include in training set (',str(c4),') > total no. of category 4 patients in whole dataset (',str(len(indx_c4)),')')          
            if len(indx_c5) >= c5:
                indx_c5_trainset=random.sample(indx_c5,c5)
            else: 
                print('No. of category 5 patients to include in training set (',str(c5),') > total no. of category 5 patients in whole dataset (',str(len(indx_c5)),')')          
            #select on-term patients
            onterm_trainset=round(len(indx_onterm)*prop_of_trainset)#size of onterm class in the training set
            indx_onterm_trainset = random.sample(indx_onterm,onterm_trainset)#select onterm patients to include in the training set
            indx_data = [i for i in range(r)]#all instances of the whole dataset
            #indx_eis=self.select_eis_features()#indices of the 28 EIS features and class variable
            if len(set(indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset)) < preterm_trainset:#add more preterm instances if no. of preterm instances collected < no. of preterm instances required
                d=preterm_trainset-len(set(indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset))
                indx_other_preterm=random.sample(list(set(indx_preterm)-set(indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset)),d)
                #trainset=data.iloc[list(set(indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset+indx_other_preterm+indx_onterm_trainset)),indx_eis]#create training set of 28 eis features
                #trainset_ids=data.iloc[list(set(indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset+indx_other_preterm+indx_onterm_trainset)),0]#get the ids of training set              
                trainset=data.iloc[list(set(indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset+indx_other_preterm+indx_onterm_trainset)),:]#create training set of polynomial features
                indx_testset=list(set(indx_data)-set(indx_onterm_trainset+indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset+indx_other_preterm))                    
                #print(indx_other_preterm)
            else:            
                #trainset=data.iloc[list(set(indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset+indx_onterm_trainset)),indx_eis]#create training set of 28 eis features
                #trainset_ids=data.iloc[list(set(indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset+indx_onterm_trainset)),0]#get the ids of training set                 
                trainset=data.iloc[list(set(indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset+indx_onterm_trainset)),:]#create training set of polynomial features
                indx_testset=list(set(indx_data)-set(indx_onterm_trainset+indx_c1_trainset+indx_c2_trainset+indx_c3_trainset+indx_c4_trainset+indx_c5_trainset))        
            #testset=data.iloc[indx_testset,indx_eis]#create testset of 28 eis features
            testset=data.iloc[indx_testset,:]#create testset of polynomial features
            #testset_ids=data.iloc[indx_testset,0]#get ids of testset
            trainset.to_csv('trainset.csv',index=False)
            testset.to_csv('testset.csv',index=False)
            #trainset_ids.to_csv('trainset_ids'+'_'+str(j)+'.csv',header=True,index=False)
            #testset_ids.to_csv('testset_ids'+'_'+str(j)+'.csv',header=True,index=False)            
            ##convert csv format to arff format
            (_,col)=trainset.shape
            if trainset.iloc[0,col-1]==0 or trainset.iloc[0,col-1]==1:
                label=str(col)+':0,1'
            elif trainset.iloc[0,col-1]=='No' or trainset.iloc[0,col-1]=='Yes':
                label=str(col)+':No,Yes'
            else:
                label=str(col)+':no,yes'
            prep.convert_csv_to_arff('trainset.csv',arff_trainset+'_'+str(j)+'.arff',label)
            prep.convert_csv_to_arff('testset.csv',arff_testset+'_'+str(j)+'.arff',label)
            print(arff_trainset+'_'+str(j)+'.arff')
            (r2,_)=trainset.shape
            #print('r: ',str(r))
            #print('r2: ',str(r2))
            print('proportion of training set: ',str(r2/r))
            print(arff_testset+'_'+str(j)+'.arff')
            (r3,_)=testset.shape
            #print('r3: ',str(r3))
            print('proportion of test set: ',str(r3/r))
    
    def preterm_poly_features_categories(self,data,rules):
        #collect the patients of each polynomial features preterm category 
        #input: data, a dataframe representing polynomial feature data e.g. 438_V1_28inputs_poly_degree4.csv
        (r,n)=data.shape        
        indx_c1=[] #indices of category 1 patients in the dataset
        indx_c2=[] #indices of category 2 patients in the dataset
        indx_c3=[] #indices of category 3 patients in the dataset
        indx_c4=[] #indices of category 4 patients in the dataset
        indx_c5=[] #indices of category 5 patients in the dataset
        indx_onterm=[] #indices of on-term patients in the dataset
        indx_preterm=[]#indices of preterm patients in the dataset
        rs=data.index
        i=0#indx of a row
        for r in rs:#if a patient satisfies k rules, it is counted k times
            if data.at[r,"before37weeksCell"]=='No' or data.at[r,"before37weeksCell"]==0:#on-term
               indx_onterm.append(i)            
            else:
               indx_preterm.append(i)
            if self.match_rule(data,r,rules[0]):
               indx_c1.append(i)                
            if self.match_rule(data,r,rules[1]):
                indx_c2.append(i)            
            if self.match_rule(data,r,rules[2]):
                indx_c3.append(i)
            if self.match_rule(data,r,rules[3]):
                indx_c4.append(i) 
            if self.match_rule(data,r,rules[4]):
                indx_c5.append(i)
            i+=1
        return (indx_c1,indx_c2,indx_c3,indx_c4,indx_c5,indx_onterm,indx_preterm)
    
    def preterm_demographic_categories(self,data):
        #collect the patients of each demographic preterm category in 438_V1_30inputs_demographics.csv
        #1.	13_cervical_lenCell <= 17: Yes
        #2.	Ethnicity = White AND Alcohol_in_pregnancy = No AND Non_prescribed_drugs_in_pregnancy = No AND Age > 25: Yes
        #3.	Smoker=No and Non_prescribed_drugs_in_pregnancy=No and Number_of_previous_pregnancies>1 and 13_cervical_lenCell<=18: before37weeksCell=Yes
        #4.	13_cervical_lenCell > 17 and Alcohol_in_pregnancy=Yes: Yes
        #5.	13_cervical_lenCell <= 27 and 13_cervical_lenCell > 17 and Alcohol_in_pregnancy = No and 15_ffn_level_valueCell > 8 and 15_ffn_level_valueCell <= 30: Yes
        #input: data, a dataframe representing 'filtered_data_and_demographics_30inputs.csv' or "438_V1_30inputs_demographics.csv"
        (r,n)=data.shape        
        indx_c1=[] #indices of category 1 patients in the whole dataset
        indx_c2=[] #indices of category 2 patients in the whole dataset
        indx_c3=[] #indices of category 3 patients in the whole dataset
        indx_c4=[] #indices of category 4 patients in the whole dataset
        indx_c5=[] #indices of category 5 patients in the whole dataset
        indx_onterm=[] #indices of on-term patients in the whole dataset
        indx_preterm=[]#indices of preterm patients in the whole dataset
        for i in range(r):#A patient is collected once into the training set no matter how many rules he/she satisfies
            if data.iat[i,n-1]=='No' or data.iat[i,n-1]==0:#on-term
               indx_onterm.append(i)
            else:#preterm
               indx_preterm.append(i)
            if data.iat[i,11]<=17 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1): #type 1: 13_cervical_lenCell <= 17: Yes           
                indx_c1.append(i) #preterm category 1
            if data.iat[i,1]=='White' and data.iat[i,5]=='No' and data.iat[i,6]=='No' and data.iat[i,3]>25 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                indx_c2.append(i) #preterm category 1
            if data.iat[i,4]=='No' and data.iat[i,5]=='No' and data.iat[i,8] > 1 and data.iat[i,11]<=18 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                indx_c3.append(i) #preterm category 1
            if data.iat[i,11]>17 and data.iat[i,5]=='Yes' and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                indx_c4.append(i)    
            if data.iat[i,11]<=27 and data.iat[i,11] > 17 and data.iat[i,5]=='No' and data.iat[i,12]>8 and data.iat[i,12]<=30 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                indx_c5.append(i)
        return (indx_c1,indx_c2,indx_c3,indx_c4,indx_c5,indx_onterm,indx_preterm)

    def preterm_demographic_categories2(self,data):
        #collect the patients of each demographic preterm category from 'filtered_data_and_demographics_30inputs.csv'
        #1.	If 13_cervical_lenCell <= 17 Then Preterm=1
        #2.	If 13_cervical_lenCell >17 and 13_cervical_lenCell <= 27 and BMI <= 22.4 Then Preterm=1
        #3.	Ethnicity = White AND Alcohol_in_pregnancy=No AND Non_prescribed_drugs_in_pregnancy=No AND Age > 25: Yes
        #   or 
        #3.	If 13_cervical_lenCell >17 and 13_cervical_lenCell <= 27 and Smoker=Yes and 15_ffn_level_valueCell <= 23 Then Preterm=1
        #4.	If 13_cervical_lenCell <=27.5 and Ethnicity=White Then Preterm=1
        #5.	If Non_prescribed_drugs_in_pregnancy=No AND Age > 21 AND number_previous_TOPs = 0 AND number_previous_early_miscarriages=1 Then Preterm=1      
        #input: data, a dataframe representing 'filtered_data_and_demographics_30inputs.csv' or "438_V1_30inputs_demographics.csv"
        (r,n)=data.shape        
        indx_c1=[] #indices of category 1 patients in the whole dataset
        indx_c2=[] #indices of category 2 patients in the whole dataset
        indx_c3=[] #indices of category 3 patients in the whole dataset
        indx_c4=[] #indices of category 4 patients in the whole dataset
        indx_c5=[] #indices of category 5 patients in the whole dataset
        indx_onterm=[] #indices of on-term patients in the whole dataset
        indx_preterm=[]#indices of preterm patients in the whole dataset
        for i in range(r):#A patient is collected once into the training set no matter how many rules he/she satisfies
            if data.iat[i,n-1]=='No' or data.iat[i,n-1]==0:#on-term
               indx_onterm.append(i)
            else:#preterm
               indx_preterm.append(i)
            if data.iat[i,11]<=17 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1): #type 1: 13_cervical_lenCell <= 17: Yes           
                indx_c1.append(i) #preterm category 1
            if data.iat[i,11]>17 and data.iat[i,11]<=27 and data.iat[i,7]<=22.4 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                indx_c2.append(i) #preterm category 2
            if data.iat[i,1]=='White' and data.iat[i,5]=='No' and data.iat[i,6]=='No' and data.iat[i,3]>25 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                indx_c3.append(i) #preterm category 3            
            #if data.iat[i,11]>17 and data.iat[i,11]<=27 and data.iat[i,4]=='Yes' and data.iat[i,12]<=23 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
            if data.iat[i,11]<=27.5 and data.iat[i,1]=='White' and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                indx_c4.append(i)    
            if data.iat[i,6]=='No' and data.iat[i,3] > 21 and data.iat[i,10]==0 and data.iat[i,9]==1 and (data.iat[i,n-1]=='Yes' or data.iat[i,n-1]==1):
                indx_c5.append(i)
        return (indx_c1,indx_c2,indx_c3,indx_c4,indx_c5,indx_onterm,indx_preterm)
    
    def select_train_test_sets_of_age_group(self,min_age,max_age):
        #select training and test sets consisting of patients of ages >=min_age and <= max_age
        #output: training and test sets consisting of EIS features of the patients of the age group        
        #        training and test sets consisting of demographic features and EIS features of the patients of the age group
        data_path="438_V1_30inputs_demographics.csv"
        data=pd.read_csv(data_path,low_memory=False)
        indx=[0,3,len(list(data.columns))-1]#hosital_id, age and class variable
        age=data.iloc[:,indx]#hoospitical_id, age and class columns
        #age.info() #display information  
        indx=[i+13 for i in range(29)]#EIS features and class variable
        indx.insert(0,3)#insert age at index 0
        indx.insert(0,0)#insert hopital_id at index 0
        eis_data=data.iloc[:,indx]
        random_state=random.randint(0,2**32-1)
        test_size=0.34 #proportion of test set
        #test_size=0.25
        #train_size=1-test_size
        (train_set_age,test_set_age)=self.split_train_test_sets(age,test_size,random_state)
        #test_set_age.info()
        unwanted=test_set_age.loc[lambda df: np.logical_or(test_set_age['Age']>max_age,test_set_age['Age']<min_age)]#patients of age over 40 and less than 20 are not tested
        unwanted2=train_set_age.loc[lambda df: np.logical_or(train_set_age['Age']>max_age,train_set_age['Age']<min_age)]#patients of age over 40 and less than 20 are not included in training set
        #print('unwanted: ',unwanted)
        #train_set_age=train_set_age.append(unwanted)#add unwanted ages to training set
        test_set_age=test_set_age.loc[lambda df: np.logical_and(test_set_age['Age']>=min_age,test_set_age['Age']<=max_age)]#get the patients of ages 20s to 40s for testing
        train_set_age=train_set_age.loc[lambda df: np.logical_and(train_set_age['Age']>=min_age,train_set_age['Age']<=max_age)]#get the patients of ages 20s to 40s for training
        (train_set,test_set)=self.split_train_test_sets(eis_data,test_size,random_state)#training and test sets consisting of EIS features of the patients of the age group
        (train_set2,test_set2)=self.split_train_test_sets(data,test_size,random_state)#training and test sets consisting of demographic features and EIS features of the patients of the age group   
        train_set=train_set.loc[lambda df: np.isin(train_set['hospital_id'], list(unwanted2['hospital_id']),invert=True)]#remove unwanted patients from training set
        test_set=test_set.loc[lambda df: np.isin(test_set['hospital_id'], list(unwanted['hospital_id']),invert=True)]#remove unwanted patients from test set
        train_set2=train_set2.loc[lambda df: np.isin(train_set2['hospital_id'], list(unwanted2['hospital_id']),invert=True)]#remove unwanted patients from training set2
        test_set2=test_set2.loc[lambda df: np.isin(test_set2['hospital_id'], list(unwanted['hospital_id']),invert=True)]#remove unwanted patients from test set2    
        #test_set=test_set.iloc[:,1:len(list(data.columns))]#remove hospitial id column from test set
        #train_set.to_csv('trainset'+str(train_size)+'_ages'+str(min_age)+'to'+str(max_age)+'.csv',index=False)
        #test_set.to_csv('testset'+str(test_size)+'_ages'+str(min_age)+'to'+str(max_age)+'.csv',index=False)
        return (train_set,test_set,train_set2,test_set2)

    def difference_of_datasets(self):
            #find difference of each pair of csv datasets
            total_diffL=[]
            compared=set()#pairs of datasets which have been compared
            #for i in range(100):
            #    for j in range(100):
            l=[i for i in range(100)]
            l2=[i for i in range(47)]
            l3=list(set(l)-set(l2))#[47,48,...,99]
            for i in l3:
                for j in range(100):
                    if i != j and (i,j) not in compared:
                        (_,_,_,total_diff)=self.compare_preterm_patients_ids('good_trainset'+str(i)+'.csv','good_trainset'+str(j)+'.csv')#compare ids of preterm patients of 2 csv files
                        total_diffL.append(total_diff)
                        compared.add((i,j))
                        print('total difference of (',i,',',j,'): ',total_diff)    
            total_diffL.sort(reverse=True)
            print('difference of each pair of datasets: \n',total_diffL)
            print('mini difference between 2 datasets: ',total_diffL[len(total_diffL)-1])
            print('max difference between 2 datasets: ',total_diffL[0])
            print('average difference between 2 datasets: ',np.mean(total_diffL))
            print('standard deviation of difference between 2 datasets: ',np.std(total_diffL))
            
    def get_patients_ids(self,csvfile,idsfile):
        #get the ids of the patients in csvfile from 438_V1_30inputs_demographics.csv or 'filtered_data_and_demographics_30_inputs.csv' 
        #input: csvfile, a polynomial features dataset
        #       idsfile, 438_V1_30inputs_demographics.csv or filtered_data_and_demographics_30_inputs.csv 
        #return: the ids in an array
        #remove_duplicates(csvfile,'temp_file.csv')
        #data=pd.read_csv('temp_file.csv',low_memory=False)
        data=pd.read_csv(csvfile,low_memory=False)
        data2=pd.read_csv(idsfile,low_memory=False)
        data=data.round(4)
        data2=data2.round(4)
        #indx=[i for i in range(27)]#amplitudes 2 to 14, phases 1 to 14 (the 1st amplitude is removed during training/test split) and the class variable
        #indx=indx+[len(list(data.columns))-1]#add class index (last column)
        (_,c)=data.shape
        (_,c2)=data2.shape
        cols=[j for j in range(29)]
        cols2=set(cols)-set([0])#remove cols '1' from polynomial features dataset
        cols3=list(cols2)
        cols3.append(c-1)#add the class variable
        #print(cols3)
        eisdata=data.iloc[:,cols3]#amplitudes 2 to 14 and phases 1 to 14 and the class variable
        #eisdata=data.iloc[:,indx]#amplitudes 2 to 14 and phases 1 to 14 (the 1st amplitude is removed during training/test split) and the class variable
        #indx=[i+14 for i in range(28)]#28 columns = 27 features (amplitudes 2 to 14, phases 1 to 14) + class variable
        indx=[i+13 for i in range(29)]#29 columns = 28 features (amplitudes 1 to 14, phases 1 to 14) + class variable   
        all_eisdata=data2.iloc[:,indx]
        (_,c)=all_eisdata.shape
        all_eisdata=prep.convert_targets(all_eisdata,c-1)
        eisdata.before37weeksCell=eisdata.before37weeksCell.astype(int)#replace 1.0 with 1 and 0.0 with 0 in targets 
        all_eisdata.before37weeksCell=all_eisdata.before37weeksCell.astype(int)#replace 1.0 with 1 and 0.0 with 0 in targets 
        (n,c)=eisdata.shape
        (n2,_)=all_eisdata.shape
        idsL=[]   
        for i in range(n):
            patient=eisdata.iloc[i,:]
            found=False
            for j in range(n2):
                patient2=all_eisdata.iloc[j,:]
                for k in range(c):    
                    val=patient.iloc[k]
                    val2=patient2.iloc[k]
                    if val!=val2:
                       break
                    else:
                       found=True
                #patient is found
                if found==True:
                   #print(str(i),'th patient is found.')
                   idsL.append(data2.iat[j,0]) #get the id of the patient
                   break
            if found==False:
                print(str(i),'th patient of ',csvfile,' is not found in '+idsfile)
        return idsL

    def get_patients_ids2(self,csvfile,idsfile):
        #get the ids of the patients in csvfile from 438_V1_30inputs_demographics.csv 
        #input: csvfile, dataset
        #       idsfile, 
        #return: the ids in a list
        prep.remove_duplicates(csvfile,'temp_file.csv')
        eisdata=pd.read_csv('temp_file.csv',low_memory=False)
        data=pd.read_csv(idsfile,low_memory=False)
        eisdata=eisdata.round(2)
        data=data.round(2)
        indx=[i for i in range(1,30)]#skip id column in idsfile
        all_eisdata=data.iloc[:,indx]
        (_,c)=eisdata.shape
        eisdata=prep.convert_targets(eisdata,c-1)#convert 'Yes' and 'No' to 1 and 0
        (_,c2)=all_eisdata.shape
        all_eisdata=prep.convert_targets(all_eisdata,c2-1)#convert 'Yes' to 1 and 'No' to 0
        cols=list(eisdata.columns)
        if cols[-1]=='before37weeksCell':
            eisdata.before37weeksCell=eisdata.before37weeksCell.astype(int)#replace 1.0 with 1 and 0.0 with 0 in targets 
            eisdata.before37weeksCell=all_eisdata.before37weeksCell.astype(int)#replace 1.0 with 1 and 0.0 with 0 in targets 
        elif cols[-1]=='selected_reading':
            eisdata.selected_reading=eisdata.selected_reading.astype(int)#replace 1.0 with 1 and 0.0 with 0 in targets 
            eisdata.selected_reading=all_eisdata.selected_reading.astype(int)#replace 1.0 with 1 and 0.0 with 0 in targets 
        else:
            import sys
            sys.exit('class column is neither before37weeksCell nor select_reading')
        (n,c)=eisdata.shape    
        (n2,_)=all_eisdata.shape
        idsL=[]   
        for i in range(n):
            print(i)
            patient=eisdata.iloc[i,:]#a patient whose id to find
            found=False
            for j in range(n2):
                patient2=all_eisdata.iloc[j,:]#all patients with ids
                for k in range(c):    
                    val=patient.iloc[k]
                    val2=patient2.iloc[k]
                    if val!=val2:
                       break
                    else:
                       found=True
                if found==True:
                   idsL.append(data.iat[j,0]) #get the id of the patient
                   break
        return list(set(idsL)) #remove any duplicate ids in idsL
    
    def get_demo_details(self,idsL,csvfile,demofile):
        #get the demographics details of patient ids
        #input: idsL, list of ids
        #       csvfile, file containing demographics of all patients
        #output: demofile, file containing demographics of the ids
        data=pd.read_csv(csvfile,low_memory=False)
        data=data.loc[lambda arg: np.isin(data['hospital_id'],idsL)]
        (_,c)=data.shape
        indx=[i for i in range(13)]#first 13 columns
        indx.append(c-1)#add class index
        demo_data=data.iloc[:,indx]
        demo_data.to_csv(demofile,index=False)
    
    def get_data_by_ids(self,idsfile,csvfile,outfile):
        #get data of the ids in idsfile from csvfile 
        ids=pd.read_csv(idsfile)#ids used to find data
        data=pd.read_csv(csvfile)#whole dataset
        ids=list(ids.iloc[:,0])
        data2=data.loc[lambda arg: np.isin(data['hospital_id'],ids)]
        (r,_)=data2.shape
        if r < len(ids):
            r=set(r)
            ids=set(ids)
            diff=ids.difference(r)
            print(csvfile+' does not include these ids: '+str(diff))
        data2.to_csv(outfile,index=False)
    
    def get_template_probs(self,file1,file2,visit,prob_file):
        #get the Prob.O1,...,Prob.O4 and Prob.C1,...,Prob.C4 of the patients ids from 'prob_o_prob_c.csv'
        #input: file1, csv file containing the patients ids e.g. '438_V1_30inputs_demographics.csv'
        #       file2, csv file containing the probabilities e.g. 'prob_o_prob_c.csv' 
        #       visit, visit no. e.g. 1
        #output: prob_file
        data=pd.read_csv(file1)
        data2=pd.read_csv(file2)
        ids=data["hospital_id"]
        idsL=list(ids)
        print(len(idsL))
        ids2=data2['hospital_id']
        data3=data2.loc[lambda arg: np.logical_and(np.isin(ids2,idsL),data2['visit']==visit)]
        #(r,c)=data3.shape
        #print('total no. of ids including duplicates: '+str(r))
        data3.to_csv(prob_file,index=False)
        #For each patient id get the 1st probability for each Prob.Oi and Prob.Ci 
        #indxL=[]#list of indices of 1st occurrences of ids
        #ids2L=list(ids2)
        #ids_found=set()
        #(r,_)=data2.shape
        #for i in range(r):
        #    for j in range(len(idsL)):#patient ids
        #        if idsL[j] == ids2L[i] and idsL[j] not in ids_found:
        #            indxL.append(i)
        #            ids_found.add(idsL[j])
        #            break        
        #data4=data2.loc[indxL,:]
        #data4.to_csv('prob_of_ids_unique.csv',index=False)
    
    def get_4_eis(self,file1,file2,visit,eisfile):
        #get the 4 EIS readings of a hospital visit (e.g. 1) for each patient in file1
        #input: file1, csv file containing the patients ids e.g. '438_V1_30inputs_demographics.csv'
        #       file2, csv file containing the EIS readings, cervical length, ffn, visits, labour_onset, progesterone, cervical_cerclage and outcome of 1633 patients (retrieved from the table T in Matlab) e.g. 'eis_cervical_length_ffn_visits_labour_onset_progesterone_cervical_cerclage.csv'
        #       visit, the hospital visit no. 
        #output: eisfile, csv file of patient ids, eis readings and the outcome
        data=pd.read_csv(file1)
        data2=pd.read_csv(file2)
        ids=data["hospital_id"]
        idsL=list(ids)
        print('no. of patient ids whose EIS are retrieved: '+str(len(idsL)))
        data3=data2.loc[lambda arg: np.logical_and(np.isin(data2['hospital_id'],idsL),data2['visits_typeCell']==visit)]
        (r,_)=data3.shape
        print('total no. of eis readings of all the patient ids: '+str(r))
        eis=[i+6 for i in range(14)]
        data3=data3.iloc[:,[0]+eis+[4]]
        data3.to_csv(eisfile,index=False)

    def count_EIS_readings_of_patients(self,file1):
        #count the no. of EIS readings of each patient
        #input: file1, '438_V1_4_eis_readings.csv'
        #output: the no. of patients with 1 EIS reading, 2 EIS readings, 3 EIS readings and 4 EIS readings respectively
        data=pd.read_csv(file1)
        ids=data['hospital_id']
        idsL=list(ids)
        ids2=set(list(idsL))
        ids2=list(ids2)
        c1=0 #patients having 1 EIS reading
        c2=0 #patients having 2 EIS readings
        c3=0 #patients having 3 EIS readings
        c4=0 #patients having 4 EIS readings
        ids1reading=[] #ids with 1 EIS reading
        ids2readings=[] #ids with 2 EIS readings
        ids3readings=[] #ids with 3 EIS readings
        ids4readings=[] #ids with 4 EIS readings
        for i in range(len(ids2)):
            c=idsL.count(ids2[i])
            if c==1:
                c1+=1
                ids1reading.append(ids2[i])
            elif c==2:
                c2+=1
                ids2readings.append(ids2[i])
            elif c==3:
                c3+=1
                ids3readings.append(ids2[i])
            else:
                c4+=1
                ids4readings.append(ids2[i])
        print(str(c1)+' patients have 1 EIS reading.')
        print(str(c2)+' patients have 2 EIS reading.')
        print(str(c3)+' patients have 3 EIS reading.')
        print(str(c4)+' patients have 4 EIS reading.')
        return (ids1reading,ids2readings,ids3readings,ids4readings)
    
    def get_temp_prob2(self,ids,file,visit):
        #Get the Prob.Oi and Prob.Ci of the patient ids
        #input: patient ids, list of ids
        #       file, 'prob_of_ids.csv' (Prob.Oi and Prob.Ci of the 438 patients at visit 1)
        #       visit, e.g. 1
        #A patient with 1 EIS reading should have Prob.O1 and Prob.C1
        #A patient with 2 EIS readings should have Prob.O1, Prob.O2, Prob.C1 and Prob.C2
        #A patient with 3 EIS readings should have Prob.O1, Prob.O2, Prob.O3, Prob.C1, Prob.C2 and Prob.C3
        #A patient with 4 EIS readings should have Prob.O1, Prob.O2, Prob.O3, Prob.O4, Prob.C1, Prob.C2, Prob.C3 and Prob.C4  
        data=pd.read_csv(file)
        #patients with 1 EIS reading
        data2=data.loc[lambda arg: np.logical_and(np.isin(data['hospital_id'],ids),data['visit']==visit)]
        (r,_)=data2.shape
        print(data2)
        '''
        c1_1reading=0 #no. of patients with 1 EIS reading and 1 probability: Prob.O1 or Prob.O2
        c2_1reading=0 ##no. of patients with 1 EIS reading and 2 probabilities: Prob.O1 and Prob.O2
        for i in range(r):
            if data1reading.loc[i,'Prob_O.1']==np.nan and data1reading.loc[i,'Prob_C.1'] != np.nan:
                c1_1reading+=1
            elif data1reading.loc[i,'Prob_O.1'] != np.nan and data1reading.loc[i,'Prob_C.1'] == np.nan:
                c1_1reading+=1
            elif data1reading.loc[i,'Prob_O.1']==np.nan and data1reading.loc[i,'Prob_C.1'] == np.nan:
                c2_1reading+=1
        #patients with 2 EIS readings
        data2reading=data.loc[lambda arg: np.logical_and(np.isin(data['hospital_id'],ids2reading),data['visits_typeCell']==visit)]
        (r,_)=data2reading
        c1_2reading=0 #no. of patients with 2 EIS reading and 1 probability: Prob.O1 or Prob.O2
        c2_2reading=0 ##no. of patients with 2 EIS reading and 2 probabilities: Prob.O1 and Prob.O2
        for i in range(r):
            if data1reading.loc[i,'Prob_O.1']==np.nan and data1reading.loc[i,'Prob_C.1'] != np.nan:
                c1_1reading+=1
            elif data1reading.loc[i,'Prob_O.1'] != np.nan and data1reading.loc[i,'Prob_C.1'] == np.nan:
                c1_1reading+=1
            elif data1reading.loc[i,'Prob_O.1']==np.nan and data1reading.loc[i,'Prob_C.1'] == np.nan:
                c2_1reading+=1
        '''
    def incorp_prob(self,probfile,probtype,eisfile,outfile):
        #incorporate Prob.Ci or Prob.Oi to EIS readings
        #input: probfile, 'prob_of_ids.csv'
        #       eisfile, '438_V1_4_eis_readings.csv'
        #       outfile, new dataset with the new features
        probs=pd.read_csv(probfile)
        probs2=probs.loc[lambda arg: probs['visit']==1]
        (r,_)=probs2.shape
        r_labels=probs2.index #row labels
        eisdata=pd.read_csv(eisfile)
        (_,c)=eisdata.shape
        eisdata2=pd.DataFrame(np.zeros((r,c)),columns=eisdata.columns)
        for i in range(r):
            print(i)
            if probtype=='C':#Prob.Ci (probability of matching columnar template)
                eisdata2.loc[i,'EIS1']=self.create_feature('C',r_labels[i],probs2,eisdata,'EIS1')
                eisdata2.loc[i,'EIS2']=self.create_feature(r_labels[i],probs2,eisdata,'EIS2')
                eisdata2.loc[i,'EIS3']=self.create_feature(r_labels[i],probs2,eisdata,'EIS3')
                eisdata2.loc[i,'EIS4']=self.create_feature(r_labels[i],probs2,eisdata,'EIS4')
                eisdata2.loc[i,'EIS5']=self.create_feature(r_labels[i],probs2,eisdata,'EIS5')
                eisdata2.loc[i,'EIS6']=self.create_feature(r_labels[i],probs2,eisdata,'EIS6')
                eisdata2.loc[i,'EIS7']=self.create_feature(r_labels[i],probs2,eisdata,'EIS7')
                eisdata2.loc[i,'EIS8']=self.create_feature(r_labels[i],probs2,eisdata,'EIS8')
                eisdata2.loc[i,'EIS9']=self.create_feature(r_labels[i],probs2,eisdata,'EIS9')
                eisdata2.loc[i,'EIS10']=self.create_feature(r_labels[i],probs2,eisdata,'EIS10')
                eisdata2.loc[i,'EIS11']=self.create_feature(r_labels[i],probs2,eisdata,'EIS11')
                eisdata2.loc[i,'EIS12']=self.create_feature(r_labels[i],probs2,eisdata,'EIS12')
                eisdata2.loc[i,'EIS13']=self.create_feature(r_labels[i],probs2,eisdata,'EIS13')
                eisdata2.loc[i,'EIS14']=self.create_feature(r_labels[i],probs2,eisdata,'EIS14')
            elif probtype=='O':#Prob.Oi (probability of matching squaremous template)
                eisdata2.loc[i,'EIS1']=self.create_feature('O',r_labels[i],probs2,eisdata,'EIS1')
                eisdata2.loc[i,'EIS2']=self.create_feature(r_labels[i],probs2,eisdata,'EIS2')
                eisdata2.loc[i,'EIS3']=self.create_feature(r_labels[i],probs2,eisdata,'EIS3')
                eisdata2.loc[i,'EIS4']=self.create_feature(r_labels[i],probs2,eisdata,'EIS4')
                eisdata2.loc[i,'EIS5']=self.create_feature(r_labels[i],probs2,eisdata,'EIS5')
                eisdata2.loc[i,'EIS6']=self.create_feature(r_labels[i],probs2,eisdata,'EIS6')
                eisdata2.loc[i,'EIS7']=self.create_feature(r_labels[i],probs2,eisdata,'EIS7')
                eisdata2.loc[i,'EIS8']=self.create_feature(r_labels[i],probs2,eisdata,'EIS8')
                eisdata2.loc[i,'EIS9']=self.create_feature(r_labels[i],probs2,eisdata,'EIS9')
                eisdata2.loc[i,'EIS10']=self.create_feature(r_labels[i],probs2,eisdata,'EIS10')
                eisdata2.loc[i,'EIS11']=self.create_feature(r_labels[i],probs2,eisdata,'EIS11')
                eisdata2.loc[i,'EIS12']=self.create_feature(r_labels[i],probs2,eisdata,'EIS12')
                eisdata2.loc[i,'EIS13']=self.create_feature(r_labels[i],probs2,eisdata,'EIS13')
                eisdata2.loc[i,'EIS14']=self.create_feature(r_labels[i],probs2,eisdata,'EIS14')
            else:
                print('invalid probability type: '+probtype)
            h_id=probs2.loc[r_labels[i],'hospital_id']
            eisdata2.loc[i,'hospital_id']=h_id
            eis_of_h_id=eisdata.loc[lambda arg: eisdata['hospital_id']==h_id]
            r_labels2=eis_of_h_id.index
            eisdata2.loc[i,'before37weeksCell']=eis_of_h_id.loc[r_labels2[0],'before37weeksCell']
        eisdata2.to_csv(outfile,index=False)           
    
    def create_feature(self,probtype,i,probs2,eisdata,eis_col):
        #incorporate each probability to a reading to create a new feature
        #input: probtype, 'C' or 'O' 
        #       i, row label of a patient in probs2
        #       probs2, the probabilities of a patient
        #       eisdata, the 4 readings of a patient
        #       eis_col, e.g. 'EIS1'
        #output: a new feature of a patient
        h_id=probs2.loc[i,'hospital_id']
        eisdata2=eisdata.loc[lambda arg: eisdata['hospital_id']==h_id]
        (n,_)=eisdata2.shape #n, no. of readings of this patient id
        indx=eisdata2.index #row labels
        #print('n: ',str(n))
        #print(eisdata2)
        s=0 #sum of prob_Ci * ith reading (complex number)
        t=0 #sum of prob_Ci
        if probtype=='C':    
            prob_C1=probs2.at[i,'Prob_C.1']
            if np.isnan(prob_C1)==False and prob_C1 != 0: # if Prob_C1 is not missing and not 0 
                reading1=eisdata2.at[indx[0],eis_col]
                if prep.complex_convert_i_to_j(reading1)!='real_number':
                    s+=np.round(prob_C1,4)*complex(prep.complex_convert_i_to_j(reading1))
                    t+=np.round(prob_C1,4)
            prob_C2=probs2.at[i,'Prob_C.2']
            if np.isnan(prob_C2)==False and prob_C2 != 0 and n > 1: #there is 2nd reading and Prob_C2 is not missing and not 0
                reading2=eisdata2.at[indx[1],eis_col]
                if prep.complex_convert_i_to_j(reading2)!='real_number':
                    s+=np.round(prob_C2,4)*complex(prep.complex_convert_i_to_j(reading2))
                    t+=np.round(prob_C2,4)
            prob_C3=probs2.at[i,'Prob_C.3']
            if np.isnan(prob_C3)==False and prob_C3 != 0 and n > 2: #there is 3rd reading and Prob_C3 is not missing and not 0
                reading3=eisdata2.at[indx[2],eis_col]
                if prep.complex_convert_i_to_j(reading3)!='real_number':
                    s+=np.round(prob_C3,4)*complex(prep.complex_convert_i_to_j(reading3))
                    t+=np.round(prob_C3,4)
            prob_C4=probs2.at[i,'Prob_C.4']
            if np.isnan(prob_C4)==False and prob_C4 !=0 and n > 3: #there is 4th reading and Prob_C4 is not missing and not 0
                reading4=eisdata2.at[indx[3],eis_col]
                if prep.complex_convert_i_to_j(reading4)!='real_number':
                    s+=np.round(prob_C4,4)*complex(prep.complex_convert_i_to_j(reading4))
                    t+=np.round(prob_C4,4)
        elif probtype=='O':
            prob_O1=probs2.at[i,'Prob_O.1']
            if np.isnan(prob_O1)==False and prob_O1 != 0: # if Prob_O1 is not missing and not 0 
                reading1=eisdata2.at[indx[0],eis_col]
                if prep.complex_convert_i_to_j(reading1)!='real_number':
                    s+=np.round(prob_O1,4)*complex(prep.complex_convert_i_to_j(reading1))
                    t+=np.round(prob_O1,4)
            prob_O2=probs2.at[i,'Prob_O.2']
            if np.isnan(prob_O2)==False and prob_O2 != 0 and n > 1: #there is 2nd reading and Prob_O2 is not missing and not 0
                reading2=eisdata2.at[indx[1],eis_col]
                if prep.complex_convert_i_to_j(reading2)!='real_number':
                    s+=np.round(prob_O2,4)*complex(prep.complex_convert_i_to_j(reading2))
                    t+=np.round(prob_O2,4)
            prob_O3=probs2.at[i,'Prob_O.3']
            if np.isnan(prob_O3)==False and prob_O3 != 0 and n > 2: #there is 3rd reading and Prob_O3 is not missing and not 0
                reading3=eisdata2.at[indx[2],eis_col]
                if prep.complex_convert_i_to_j(reading3)!='real_number':
                    s+=np.round(prob_O3,4)*complex(prep.complex_convert_i_to_j(reading3))
                    t+=np.round(prob_O3,4)
            prob_O4=probs2.at[i,'Prob_O.4']
            if np.isnan(prob_O4)==False and prob_O4 !=0 and n > 3: #there is 4th reading and Prob_O4 is not missing and not 0
                reading4=eisdata2.at[indx[3],eis_col]
                if prep.complex_convert_i_to_j(reading4)!='real_number':
                    s+=np.round(prob_O4,4)*complex(prep.complex_convert_i_to_j(reading4))
                    t+=np.round(prob_O4,4)
        else:
            print('invalid probability type: '+probtype)
        if t!=0:
            return np.round(s/t,4)
        else:
            return 'none'

  
if __name__ == "__main__":
    ##main program##
    #polynomial features rules
    '''
    rules=["x12 x14 x24 x26 <= 0.616152 AND x8 x11 x27^2 > 1970.982982 AND x9 x26^2 x27 > -24114.220548:1", 
               "x12 x14 x24 x26 <= 0.616152 AND x25^2 x27 <= -10288.074643 AND x3 x9 x26 x27 > 617115.379755 AND x6 x9 x26 x27 <= 544842.487511 AND x1^2 x15^2 > 82874.672947:1",
               "x10^2 x13 x15 <= -71.817328 AND x9 x26^2 x27 > -84698.804034 AND x12 x14 x24 x26 > -10092.102417 AND x10^2 x13 x15 <= -85.057599 AND x7 x27^3 <= -78812.187917 AND x9 x12 x14 > -47.628541 AND x10 x12 x25 x27 > 5233.891303:1",        
               "x9 x12^3 <= 20.615217:1",
               "x5 x15 x16 x22 <= -12997.728047 AND x5 x15 x16 x22 <= -18454.165577 AND x6 x12^3 <= 363.295587 AND x11^2 x12 x18 <= -271.405861 AND x11^2 x12 x18 > -523.960997 AND x1 x17 x18 x23 > -1142560.135815 AND x7 x21 x22 x23 <= -927304.981822:1",
               "x5 x15 x16 x22 > -12997.728047:1",
               "x6 x12^3 <= 353.137753 AND x11^2 x12 x18 <= -271.405861 AND x9 x12^3 <= 134.081455 AND x9 x26^2 x27 > -228708.503972 AND x1 x17 x18 x23 <= -154464.662097 AND x7 x21 x22 x23 > -539426.798594:1",
               "x6 x12^3 <= 363.295587 AND x7 x21 x22 x23 <= -610844.978474 AND x10^2 x13 x15 > -213.678826:1",
               "x10^2 x13 x15 > -290.612788:1"]
    '''
    s=SelectData()
    #print('==count preterm patients in the whole dataset===')
    #s.count_preterm_patients_of_demographic_types('C:\\Users\\uos\\EIS preterm prediction\\trainset28.csv',s.filtered_data_demographics)
    #s.count_preterm_patients_of_poly_features_properties("C:\\Users\\uos\\EIS preterm prediction\\poly features\\438_V1_28inputs_poly_degree4.csv",rules)
    #print('===count preterm patients in best training set===')
    #s.count_preterm_patients_of_poly_features_properties("U:\\EIS preterm prediction\\best training set\\train_poly_features97.csv",rules)
    #(_,_,trainset,testset)=s.select_train_test_sets_of_age_group(20,40)#select training and test sets of ages 20 to 40 years old
    #data=trainset.append(testset)
    #data=pd.read_csv("438_V1_30inputs_demographics.csv")
    #data=pd.read_csv(s.filtered_data_demographics)
    #data=pd.read_csv("C:\\Users\\uos\\EIS preterm prediction\\poly features\\438_V1_28inputs_poly_degree4.csv")
    #m=100;#no. of pairs of training and test sets
    #m=1
    #p1=12.195/100
    #p2=58.5366/100
    #p3=9.756/100
    #p4=9.756/100
    #p5=12.195/100
    #p1=14.63/100
    #p2=12.2/100
    #p3=68.29/100
    #p4=41.46/100
    #p5=29.27/100
    #prop_of_trainset=0.66 #best training set (66%), test set (34%)
    #prop_of_trainset=0.75
    #s.select_train_test_sets_using_preterm_categories(data,m,p1,p2,p3,p4,p5,prop_of_trainset,'good_trainset_ages20to40','good_testset_ages20to40')
    #s.select_train_test_sets_using_preterm_categories(data,m,p1,p2,p3,p4,p5,prop_of_trainset,'good_trainset_ids','good_testset_ids')
    #s.select_train_test_sets_using_preterm_categories(data,m,p1,p2,p3,p4,p5,prop_of_trainset,'C:\\Users\\uos\\EIS preterm prediction\\good trainsets\\good_trainset','C:\\Users\\uos\\EIS preterm prediction\\good trainsets\\good_testset')    
    #s.compare_preterm_patients_ids_with_best_training_set('good_trainset91.csv')#compare ids of preterm patients of the best training set with ids of preterm patients of a good training set selected using the 5 preterm categories
    #s.difference_of_datasets()
    ###ranked categories by fraction of preterm patients in the best training set 'train_poly_features97.csv'
    #(0.341463, '6: x5 x15 x16 x22 > -12997.728047:1')
    #(0.243902, '8: x6 x12^3 <= 363.295587 AND x7 x21 x22 x23 <= -610844.978474 AND x10^2 x13 x15 > -213.678826:1')
    #(0.195122, '4: x9 x12^3 <= 20.615217:1')
    #(0.146341, '1: x12 x14 x24 x26 <= 0.616152 AND x8 x11 x27^2 > 1970.982982 AND x9 x26^2 x27 > -24114.220548:1')
    #(0.146341, '3: x10^2 x13 x15 <= -71.817328 AND x9 x26^2 x27 > -84698.804034 AND x12 x14 x24 x26 > -10092.102417 AND x10^2 x13 x15 <= -85.057599 AND x7 x27^3 <= -78812.187917 AND x9 x12 x14 > -47.628541 AND x10 x12 x25 x27 > 5233.891303:1')
    #rules=['x5 x15 x16 x22 > -12997.728047:1',
    #       'x6 x12^3 <= 363.295587 AND x7 x21 x22 x23 <= -610844.978474 AND x10^2 x13 x15 > -213.678826:1',
    #       'x9 x12^3 <= 20.615217:1',
    #       'x12 x14 x24 x26 <= 0.616152 AND x8 x11 x27^2 > 1970.982982 AND x9 x26^2 x27 > -24114.220548:1',
    #       'x10^2 x13 x15 <= -71.817328 AND x9 x26^2 x27 > -84698.804034 AND x12 x14 x24 x26 > -10092.102417 AND x10^2 x13 x15 <= -85.057599 AND x7 x27^3 <= -78812.187917 AND x9 x12 x14 > -47.628541 AND x10 x12 x25 x27 > 5233.891303:1'
    #       ]
    #p1=0.341
    #p2=0.244
    #p3=0.195
    #p4=0.146
    #p5=0.146
    #print(rules)
    #s.select_train_test_sets_using_preterm_categories(data,rules,m,p1,p2,p3,p4,p5,prop_of_trainset,'C:\\Users\\uos\\EIS preterm prediction\\good trainsets\\poly features rules\\good_trainset','C:\\Users\\uos\\EIS preterm prediction\\good trainsets\\poly features rules\\good_testset')
    #patients_ids_no_treatment.csv contains the patients who had no treatment and contains missing values for antibiotics etc.
    #s.get_data_by_ids('no_treatment_with_previous_pregnancy_ids.csv','my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_ids.csv','my_filtered_data_28inputs_and_have_ptb_history_no_treatment_with_previous_pregnancy_with_ids.csv')
    s.get_data_by_ids('treated_with_previous_pregnancy_ids.csv','my_filtered_data_28inputs_treated_with_ids.csv','my_filtered_data_28inputs_treated_with_previous_pregnancy_with_ids.csv')
    #s.get_data_by_ids('patients_ids_no_treatment.csv','438_V1_4_eis_readings_28inputs_with_ids.csv','438_V1_4_eis_readings_28inputs_no_treatment_with_ids.csv')
    #s.get_data_by_ids('patients_ids_no_treatment.csv','filtered_data_28inputs_with_ids.csv','filtered_data_28inputs_no_treatment_with_ids.csv')
    #s.get_data_by_ids('patients_ids_no_treatment.csv','438_V1_28inputs_by_sum_of_amp1_2_3_and_phase10_11_12_13_14.csv','filtered_data_28inputs_by_sum_of_amp1_2_3_and_phase10_11_12_13_14_no_treatment_with_ids.csv')    
    #s.get_data_by_ids('patients_ids_no_treatment.csv','438_V1_28inputs_by_amp1_with_ids.csv','438_V1_28inputs_by_amp1_no_treatment_with_ids.csv')
    #s.get_data_by_ids('patients_ids_no_treatment.csv','selected_unselected_eis_readings_with_ids.csv','selected_unselected_eis_readings_no_treatment_with_ids.csv')
    #patients_ids_no_treatment2.csv contains the patients who had no treatment and no missing values for antibiotics etc.
    #s.get_data_by_ids('patients_ids_no_treatment2.csv','filtered_data_28inputs_with_ids.csv','filtered_data_28inputs_no_treatment2_with_ids.csv')
    #merge testset1_i (17%) with validset1_i (17%) to form testset1_validset1_i (34%)
    #for i in range(100):
    #    testsetid='U:\\EIS preterm prediction\\trainsets1trainsets2\\asymp_22wks_filtered_data_28inputs_no_treatment\\trainsets66_percent\\testset1_ids_'+str(i)+'.csv'
    #    validsetid='U:\\EIS preterm prediction\\trainsets1trainsets2\\asymp_22wks_filtered_data_28inputs_no_treatment\\trainsets66_percent\\validset1_ids_'+str(i)+'.csv'
    #    testset_validsetid='U:\\EIS preterm prediction\\trainsets1trainsets2\\asymp_22wks_filtered_data_28inputs_no_treatment\\trainsets66_percent\\testset1_validset1_ids_'+str(i)+'.csv'
    #    df=pd.read_csv(testsetid)
    #    df2=pd.read_csv(validsetid)
    #    df3=pd.concat([df,df2])
    #    df3.to_csv(testset_validsetid,index=False)