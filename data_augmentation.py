#coding=utf-8
import cv2
from math import *
import numpy as np
import os
import codecs

img_path = '/home/xxx/ICDAR2015/ch4_training_images/'
txt_path = '/home/xxx/ICDAR2015/ch4_training_localization_transcription_gt/'
save_path1 = '/home/xxx/img_result/'
save_path2 = '/home/xxx/txt_result/'
#angle = [-90，-75, -60, -45, -30, -15, 0, 7.5, 15, 22.5, 30, 37.5, 45, 52.5, 60, 67.5, 75, 82.5, 90]
angle = 0 # angle的值从上面选

img_list = os.listdir(img_path) 
txt_list = os.listdir(txt_path)

# 对图片进行旋转 取外接矩形补全 不改变原有图片的大小
def img_rotate(image, angle, img_name):
    h, w = image.shape[:2]
    
    #旋转后的尺寸 注意这里图片的h和w要为整数
    #angle = radians(angle)
    h_New = int(w*fabs(sin(radians(angle))) + h*fabs(cos(radians(angle))))
    w_New = int(h*fabs(sin(radians(angle))) + w*fabs(cos(radians(angle))))
    
    matRotation = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1)

    matRotation[0,2] += (w_New - w)/2  #重点在这步
    matRotation[1,2] += (h_New - h)/2  #重点在这步

    imgRotation = cv2.warpAffine(image, matRotation, (w_New, h_New), borderValue=(255,255,255))

    cv2.imwrite(save_path1 + 'ro_'+str(angle)+'_'+ img_name +'.jpg', imgRotation)

# 得出旋转后的坐标平移delta值
def txt_rotate(image, angle):
    h, w = image.shape[:2]
    #旋转后的尺寸
#     heightNew = int(width*fabs(sin(radians(angle)))+height*fabs(cos(radians(angle))))
#     widthNew = int(height*fabs(sin(radians(angle)))+width*fabs(cos(radians(angle))))
    #angle = radians(angle)
    h_New = w*fabs(sin(radians(angle))) + h*fabs(cos(radians(angle)))
    w_New = h*fabs(sin(radians(angle))) + w*fabs(cos(radians(angle)))
    
    delta_x = (w_New - w)/2
    delta_y = (h_New - h)/2
    
    return delta_x, delta_y

# 对坐标绕一个中心点旋转
def transfor(x1, y1, image, angle, dx, dy):
    h, w = image.shape[:2] # 输入图片的高和宽
    # 原图片的中心点坐标
    x_ = w / 2
    y_ = h / 2
    #angle = radians(angle)
    x = (x1 - x_)*cos(radians(angle)) - (y1 - y_)*sin(radians(angle)) + x_ + dx
    y = (x1 - x_)*sin(radians(angle)) + (y1 - y_)*cos(radians(angle)) + y_ + dy

    return x, y

for img_ in img_list:
    img_name = img_.split('.jpg')[0]

    img = cv2.imread(img_path + img_)
    img_rotate(img, angle, img_name) # 先将图片进行旋转

    delta_x, delta_y = txt_rotate(img, angle) # 得到每个坐标平移的delta值
    
    try:
        with open(txt_path + 'gt_'+ img_name + '.txt', 'r') as f:
            lines = f.readlines()
    except:
        print(img_name)
        continue
    print(img_name)
    for line in lines:
        line = line.split(',')
        #line[0] = filter(str.isdigit, line[0])
        if line[0].startswith(codecs.BOM_UTF8):
            line[0] = line[0].split('\xef\xbb\xbf')[1]
        pt1 = [float(line[0]), float(line[1])]
        pt2 = [float(line[2]), float(line[3])]
        pt3 = [float(line[4]), float(line[5])]
        pt4 = [float(line[6]), float(line[7])]
        
        pt_x = np.zeros((4, 1))
        pt_y = np.zeros((4, 1))
        # 对坐标的旋转 因为图片取整的原因 所以坐标的旋转有误差
        pt_x[0, 0], pt_y[0, 0] = transfor(pt1[0], pt1[1], img, angle, delta_x, delta_y)
        pt_x[1, 0], pt_y[1, 0] = transfor(pt2[0], pt2[1], img, angle, delta_x, delta_y)
        pt_x[2, 0], pt_y[2, 0] = transfor(pt3[0], pt3[1], img, angle, delta_x, delta_y)
        pt_x[3, 0], pt_y[3, 0] = transfor(pt4[0], pt4[1], img, angle, delta_x, delta_y)
#         ptt = np.zeros((4,2))
#         ptt[0,0], ptt[0,1] = transfor(pt1[0], pt1[1], img, angle, delta_x, delta_y)
#         ptt[1,0], ptt[1,1] = transfor(pt2[0], pt2[1], img, angle, delta_x, delta_y)
#         pt3[2,0], pt3[2,1] = transfor(pt3[0], pt3[1], img, angle, delta_x, delta_y)
#         pt4[3,0], pt4[3,1] = transfor(pt4[0], pt4[1], img, angle, delta_x, delta_y)
        
        with open(save_path2 + 'ro_'+ str(angle)+'_'+img_name +'.txt', 'a') as f1:
            f1.writelines(str('%.2f')%(pt_x[0, 0]))
            f1.writelines(',')
            f1.writelines(str('%.2f')%(pt_y[0, 0]))
            f1.writelines(',')
            f1.writelines(str('%.2f')%(pt_x[1, 0]))
            f1.writelines(',')
            f1.writelines(str('%.2f')%(pt_y[1, 0]))
            f1.writelines(',')
            f1.writelines(str('%.2f')%(pt_x[2, 0]))
            f1.writelines(',')
            f1.writelines(str('%.2f')%(pt_y[2, 0]))
            f1.writelines(',')
            f1.writelines(str('%.2f')%(pt_x[3, 0]))
            f1.writelines(',')
            f1.writelines(str('%.2f')%(pt_y[3, 0]))
            f1.writelines(',')
            f1.writelines(str(line[-1]))   
