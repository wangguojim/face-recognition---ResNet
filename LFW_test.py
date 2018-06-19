# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:51:55 2018

@author: Administrator
"""

import dlib
import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt

#数据导入
#imagePath = 'data/images/lfw/'  
#savePath = 'data/images/lfwtest/'
#
#for file in os.listdir(imagePath):
#    subpath=imagePath+file+"/"
#    print(np.size(os.listdir(subpath)))
#    if np.size(os.listdir(subpath))>1:
#        for pic in os.listdir(subpath):
#            img = cv2.imread(subpath + pic)
#            cv2.imwrite(savePath+pic, img)       
#            plt.imshow(img)
#            plt.show()
   
# --------------加载数据----------------#
detector = dlib.cnn_face_detection_model_v1('data/mmod_human_face_detector.dat')
#detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('data/dlib_face_recognition_resnet_model_v1.dat')

# -------------------------------------------

imagePath = 'data/images/lfwtest/'   
pnum=np.size(os.listdir(imagePath));                                           #图像的目录

data = np.zeros((1,128))                                                       #定义一个128维的空向量data
label = []    
IMAGEs=list(np.zeros((0,1)))  
LABELs=list(np.zeros((0,1)))                                                                 #定义空的list存放人脸的标签
index=0;
Face_Vector=list(np.zeros((0,1))) 
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
            Face_Vector.append(np.array(face_descriptor))                                #记录所有脸特征
            faceArray = np.array(face_descriptor).reshape((1, 128))            #转换成numpy中的数据结构
            data = np.concatenate((data, faceArray))   
            #显示人脸区域                        #拼接到事先准备好的data当中去
            label.append(labelName)                                            #保存标签
            cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0), 2)
            IMAGEs.append(img)
            LABELs.append(file)            
        
        index+=1
        if index>=100:
            break        

data = data[1:, :]                                                             #因为data的第一行是空的128维向量，所以实际存储的时候从第二行开始
np.savetxt('faceData.txt', data, fmt='%f')                                     #保存人脸特征向量合成的矩阵到本地


labelFile=open('label.txt','w')                                      
json.dump(label, labelFile)                                                    #使用json保存list到本地
labelFile.close()
cv2.destroyAllWindows() 

face_num=int (np.size(Face_Vector)/128)
Face_dist=np.zeros((face_num,face_num))
for i in range(face_num):
    for j in range(face_num):
        Face_dist[i,j]=np.linalg.norm(Face_Vector[i]-Face_Vector[j])


for i in range(face_num):
    u=Face_dist[:,i]
    u[i]=1
    j=np.argmin(u)
    plt.subplot(1,2,1)
    plt.imshow(IMAGEs[i])
    #plt.title(LABELs[i][:-4])
    plt.subplot(1,2,2)
    plt.imshow(IMAGEs[j])
    #plt.title(LABELs[j][:-4])    
    plt.show()
    
    
import scipy.io as sio    
sio.savemat('Face_Vector',{'Face_Vector':Face_Vector,'IMAGEs':IMAGEs,'LABELs':LABELs})
data=sio.loadmat('Face_Vector.mat')
IMAGEs=data.get('IMAGEs')
Face_Vector=data.get('Face_Vector')
LABELs=data.get('LABELs')