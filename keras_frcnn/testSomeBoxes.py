# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:47:34 2020

@author: Josua
"""
import cv2
input_path ="../dataset/box.txt"
with open(input_path,'r') as f:
        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split
            #filename = "../dataset/trainImages/0.png"
            img = cv2.imread(filename)
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            #cv2.imshow("img",img)
            img = cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.imshow("img",img)
            cv2.waitKey(0)                
            #closing all open windows  
            cv2.destroyAllWindows()  