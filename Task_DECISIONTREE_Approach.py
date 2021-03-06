# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:49:14 2017

@author: u56648
"""

###TextPharse Classification (Given Text into c1,c2,...cn)
###MultiClass classification using MultinomialNB()
"""
Here every request(description) that is getting from the user will get classified in to any one of the following categories

    #ACC - Access/Privelege 
    #APP - Applications
    #COM - Communication
    #INQ - Inquiry
    #LDT - Laptop/Desktop
    #LOA - Loan Accessories
    #MOD - Mobile Devices
    #NPS - Network/Printer/Scanner
    #OTH - Other
    #SEC - Security
    #SFT - Software
    #STR - Storage & Backup
            
    @input - Get the user request from the user
    @output - Returns any one of the above category      
"""

import csv 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.svm import SVC1
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def load_file():
    with open('C:\\Users\\U56648\\Desktop\\book2.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        reader.next()
        data =[]
        target = []
        for row in reader:
            # skip missing data
            if row[0] and row[1]:
                data.append(row[0])
                target.append(row[1])
        return data,target

def preprocess():
    """preprocess creates the term frequency matrix for the review data set"""
    data,target = load_file()
    count_vectorizer = CountVectorizer(ngram_range=(1, 2))
    data = count_vectorizer.fit_transform(data)   
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)
    return tfidf_data,count_vectorizer

def learn_model(data,target,data_for_test_tfidf,inputdesc):
    """ preparing data for split validation. 60% training, 40% test"""
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.25,random_state=33)
    ##Building a classifer using NavieBayes Algorithm
    #classifier = SVC().fit(data_train,target_train)
    classifier = DecisionTreeClassifier().fit(data_train,target_train)
    joblib.dump(classifier,'SVC_classifier')

    predicted = classifier.predict(data_test)
     
    """For the Demostration purpose the sample Ticket Description must be entered by the user and test it on this classifier """
    demo_predicted1 = classifier.predict(data_for_test_tfidf)  
    print "*********"
    for desc, category in zip(inputdesc, demo_predicted1):
       print('%r => %s' % (desc, demo_predicted1))
       
    print "*********"  
    evaluate_model(target_test,predicted)

def evaluate_model(target_true,target_predicted):
    """Evaluvate the Model """
    print "Classification of Request description(Using DECISION TREE Algorithm)"
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))

def demo_classifer(inputdesc):
    """To demonstrate in real time with a given request description. This demo_classifer will do the classification of the given description"""
    
def main():
    """Driver Fucntion to call other functions """
    data,target = load_file()
    tf_idf,count_vectorizer = preprocess()   

    features = count_vectorizer.get_feature_names()
    print features

    answer=True
    while(answer):
        print("""
               1.Enter your Description to Classify
               2.Exit/Quit
               """)
        answer=raw_input("What would you like to do?")
        if answer=="1":
             inputdesc= [raw_input("Enter the Request Description to classify:")] 
             data_for_test = count_vectorizer.transform(inputdesc)
             data_for_test_tfidf = TfidfTransformer(use_idf=False).transform(data_for_test)
             learn_model(tf_idf,target,data_for_test_tfidf,inputdesc)
        elif answer=="2":
             print("\n Goodbye") 
             answer = None
        else:
            print("\n Not Valid Choice Try again")
            
main()

    


