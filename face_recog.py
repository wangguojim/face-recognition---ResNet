# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:56:26 2018

@author: Administrator
"""

import dlib
import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt

# --------------加载数据----------------#
detector = dlib.cnn_face_detection_model_v1('data/mmod_human_face_detector.dat')
#detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('data/dlib_face_recognition_resnet_model_v1.dat')

# -------------------------------------------

imagePath = 'data/images/stars/'   
pnum=np.size(os.listdir(imagePath));                                           #图像的目录
data = np.zeros((1,128))                                                       #定义一个128维的空向量data
label = []    
IMAGEs=list(np.zeros((pnum,1)));  
LABELs=list(np.zeros((pnum,1)));                                                                  #定义空的list存放人脸的标签
index=0;
Face_Vector=np.zeros((128,pnum))
for file in os.listdir(imagePath):                                             #开始一张一张索引目录中的图像
    if '.jpg' in file or '.png' in file:
        fileName = file
        labelName = file.split('_')[0]                                          #获取标签名
        print('current image: ', file)
        #print('current label: ', labelName)        
        img = cv2.imread(imagePath + file)                                     #使用opencv读取图像数据
        if img.shape[0]*img.shape[1] > 500000:                                 #如果图太大的话需要压缩，这里像素的阈值可以自己设置
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        dets = detector(img, 1)                                                #使用检测算子检测人脸，返回的是所有的检测到的人脸区域
        
        for k, d in enumerate(dets):
            rec = dlib.rectangle(d.rect.left(),d.rect.top(),d.rect.right(),d.rect.bottom())
            shape = sp(img, rec)                                               #获取landmark
            face_descriptor = facerec.compute_face_descriptor(img, shape)      #使用resNet获取128维的人脸特征向量
            Face_Vector[:,index]=face_descriptor                               #记录所有脸特征
            faceArray = np.array(face_descriptor).reshape((1, 128))            #转换成numpy中的数据结构
            data = np.concatenate((data, faceArray))   
            #显示人脸区域                        #拼接到事先准备好的data当中去
            label.append(labelName)                                            #保存标签
            cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0), 2)
            
        #cv2.waitKey(2)
        #cv2.imshow('image', img)
        
        IMAGEs[index]=img;
        LABELs[index]=file
        index+=1
        

data = data[1:, :]                                                             #因为data的第一行是空的128维向量，所以实际存储的时候从第二行开始
np.savetxt('faceData.txt', data, fmt='%f')                                     #保存人脸特征向量合成的矩阵到本地


#labelFile=open('label.txt','w')                                      
#json.dump(label, labelFile)                                                    #使用json保存list到本地
#labelFile.close()
cv2.destroyAllWindows() 



Face_dist=np.zeros((pnum,pnum))
for i in range(pnum):
    for j in range(pnum):
        Face_dist[i,j]=np.linalg.norm(Face_Vector[:,i]-Face_Vector[:,j])


#for k in range(pnum):
#    for s in range(blcoksize):        
#        plt.imshow(IMAGEs[k*blcoksize+s])
#        plt.show()
import copy
Knn=5  #K最近邻
record=np.zeros((pnum,Knn))
blcoksize=3
ss=int(Knn/blcoksize)+1

for i in range(pnum):
    plt.figure(figsize=(blcoksize*4,ss*4),dpi=80) 
    plt.subplot(ss,blcoksize,1)
    plt.imshow(IMAGEs[i])    
    u=copy.copy(Face_dist[:,i])
    u[i]=1;
    for k in range(Knn):
        j=np.argmin(u)
        u[j]=k+10
        record[i,k]=j
        plt.subplot(ss,blcoksize,k+2)
        plt.imshow(IMAGEs[j])
        plt.title(LABELs[j][:-4])
    plt.show()
    
    
# ------------------------------------------------
#def findNearestClassForImage(face_descriptor, faceLabel):
#    temp =  face_descriptor - data
#    e = np.linalg.norm(temp,axis=1,keepdims=True)
#    min_distance = e.min() 
#    print('distance: ', min_distance)
#    if min_distance > threshold:
#        return 'other'
#    index = np.argmin(e)
#    return faceLabel[index]
##--------------------------------------------------
#def recognition(img):
#    dets = detector(img, 1)
#    for k, d in enumerate(dets):
#        
#        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#            k, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()))
#        rec = dlib.rectangle(d.rect.left(),d.rect.top(),d.rect.right(),d.rect.bottom())
#        print(rec.left(),rec.top(),rec.right(),rec.bottom())
#        shape = sp(img, rec)
#        face_descriptor = facerec.compute_face_descriptor(img, shape)        
#        
#        class_pre = findNearestClassForImage(face_descriptor, label)
#        print(class_pre)
#        cv2.rectangle(img, (rec.left(), rec.top()+10), (rec.right(), rec.bottom()), (0, 255, 0), 2)
#        cv2.putText(img, class_pre , (rec.left(),rec.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
#
#    cv2.imshow('image', img)
#
## ----------------------------------------------------
#labelFile=open('label.txt','r')
#label = json.load(labelFile)                                                   #载入本地人脸库的标签
#labelFile.close()
#    
#data = np.loadtxt('faceData.txt',dtype=float)                                  #载入本地人脸特征向量
#
#cap = cv2.VideoCapture(0)
#fps = 10
#size = (640,480)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#videoWriter = cv2.VideoWriter('video.MP4', fourcc, fps, size)
#
#while(1):
#    ret, frame = cap.read()
#    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
#    recognition(frame)
#    videoWriter.write(frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#    
#cap.release()
#videoWriter.release()
#cv2.destroyAllWindows()                                                        #关闭所有的窗口