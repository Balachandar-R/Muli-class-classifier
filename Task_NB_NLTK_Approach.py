# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:36:24 2017

@author: u56648
"""
import csv 
import random
import nltk

#with open('C:\\Users\\U56648\\Desktop\\review.csv') as csv_file:
with open('C:\\Users\\U56648\\Desktop\\Test_task_Book1.csv') as csv_file:
    reader = csv.reader(csv_file,delimiter=",",quotechar='"')
    reader.next()
    data =[]
    target = []
    for row in reader:
    # skip missing data
        if row[0] and row[1]:
            data.append(row[0])
            target.append(row[1])
            
random.shuffle(data)
