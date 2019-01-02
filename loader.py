#matplotlib inline
import numpy as np
import torch
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
import sys
#import sys.path
sys.path.append(r'C:\Users\pasan\Documents\Motivation\cocoapi-master\PythonAPI')

import pycocotools
from pycocotools.coco import COCO



def call():

    dataDir='..'
    dataType='val2017'
    annFile='C:/Users/pasan/Documents/Motivation/cocoapi-master/PythonAPI/annotations/instances_train2017.json'.format(dataDir,dataType)
    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    # initialize COCO api for caption annotations
    annFile = 'C:/Users/pasan/Documents/Motivation/cocoapi-master/PythonAPI/annotations/captions_train2017.json'.format(dataDir,dataType)
    coco_caps=COCO(annFile)


    # load and display caption annotations
    catIds = coco.getCatIds(catNms=['person']);
    imgIds = coco.getImgIds(catIds=catIds );
    print('No. of images, category : people - ', len(imgIds))
    # j = 0
    ffann=[]
    for imgId in imgIds:
        # if(j==2):      To avoid loading all the annotations,  just for testing purposes
        #     break
        fann = []
        i = 0
        img = coco.loadImgs(imgId)
        img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
        I = io.imread(img['coco_url'])#'C:/Users/pasan/Documents/Motivation/cocoapi-master/PythonAPI/train2017/'+img['file_name']
        K = torch.from_numpy(I)
        print('\n\nImage Id: ', imgId, '\n')
        print( 'image tensor shape: ', K.shape)
        annIds = coco_caps.getAnnIds(imgId)
        print(type(annIds))
        for annId in annIds :
            if(i< 3):
                fann.append(annId)
                i = i+1
        print('Annotation IDs: ')
        print(fann)
        ffann.append(fann)
        anns = coco_caps.loadAnns(fann)
        print('\n 3 top selected Annotations: ')
        anns = coco_caps.showAnns(anns)
        anns = str(anns)
        print('\nType of anotations', type(anns))
        # j = j+1

    return imgIds,ffann

# if __name__ == '__main__':
#
#     L,S = call()
#     print ('yefbkj',len(L))
#     print (S)
