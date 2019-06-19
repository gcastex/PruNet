"""
Disclaimer:

This code is based on code by Peter Ruch, see his prunhild repository
See: https://github.com/gfrogat/prunhild
Snippets of code also borrowed from Arun Joseph pruning code.
https://github.com/00arun00/Pruning-Pytorch/blob/master/prune.py
"""

import numpy as np
from mtcnn_pytorch_master.src import detect_faces, show_bboxes
from PIL import Image
import random
import os


def pp_save_clean_dataset():
    directory = '/Users/guillaume/Documents/Soft/Project_Insight/Data/GOT'
    dirname_ext = '_raw'
    ftrain = open(directory+'/train.txt','w')
    fval = open(directory+'/val.txt',"w")
    ftest = open(directory+'/test.txt','w')
    ##file.write(“Hello World”)
    #file.write(“This is our new text file”)
    #file.write(“and this is another line.”)
    #file.write(“Why? Because we can.”)
    #file.close()
    jclass=0
    for filedir in os.listdir(directory):
        if dirname_ext in filedir:
            jclass+=1
            i = 0
            print(filedir)
            for filename in os.listdir(directory+'/'+filedir):
                #print(directory+'/'+filedir+'/'+filename)
                if filename[0] != '.':
                    rnd = random.random()
                    # 70% train set, 20% val set, 10% test set.
                    if rnd<.7:
                        ftrain.write(directory+'/'+filedir+'/'+filename+' '+str(jclass)+'\n')
                    elif rnd<.9:
                        fval.write(directory+'/'+filedir+'/'+filename+' '+str(jclass)+'\n')
                    else:
                        ftest.write(directory+'/'+filedir+'/'+filename+' '+str(jclass)+'\n')
                    i+=1
    ftrain.close()
    fval.close()
    ftest.close()

def pp_detect_faces():
    hsize = 144
    wsize = 144
    directory = '/Users/guillaume/Documents/Soft/Project_Insight/Data/GOT'
    dirname_ext = '_unprocessed'
    for filedir in os.listdir(directory):
        if dirname_ext in filedir:
            i = 0
            print(filedir)
            for filename in os.listdir(directory+'/'+filedir):
                #print(directory+'/'+filedir+'/'+filename)
                if filename[0] != '.':
                    image = Image.open(directory+'/'+filedir+'/'+filename)
                    bounding_boxes, landmarks = detect_faces(image)
                    #print(len(bounding_boxes))
                    #print('---')
                    #print(bounding_boxes)
                    for ind in range(len(bounding_boxes)):
                        imgcopy = image.copy()
                        imgcopy = imgcopy.crop((bounding_boxes[ind,0], bounding_boxes[ind,1], bounding_boxes[ind,2], bounding_boxes[ind,3]))#.save(...)
                        imgcopy = imgcopy.resize((wsize,hsize))
                        imgcopy = imgcopy.convert("L")
                        outdir = filedir.replace(dirname_ext, '')
                        imgcopy.save(directory+'/'+outdir+'_raw'+'/'+str(i)+'.jpeg', "JPEG")
                    i+=1
                    #if i % 10 ==0:
                    #print(filename)

pp_detect_faces()

for filename in os.listdir(directory+'/'+filedir):
    print(filename)
    if i>10:
        break
    i+=1
###
# image = Image.open('./mtcnn_pytorch_master/images/example.png')
# bounding_boxes, landmarks = detect_faces(image)
# a = show_bboxes(image, bounding_boxes, landmarks)
# a.show()
###

#import prunhild

#from config import parser
#from utils import get_parameter_stats, print_parameter_stats

