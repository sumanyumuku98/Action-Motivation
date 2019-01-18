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
import preprocess
from PIL import Image

class L:
    def __init__(self):
        self.dataDir='..'
        self.dataType='val2017'
        self.annFile='C:/Users/pasan/Documents/Motivation/cocoapi-master/PythonAPI/annotations/instances_train2017.json'.format(self.dataDir,self.dataType)
        # initialize COCO api for instance annotations
        self.coco=COCO(self.annFile)

        # initialize COCO api for caption annotations
        self.annfile = 'C:/Users/pasan/Documents/Motivation/cocoapi-master/PythonAPI/annotations/captions_train2017.json'.format(self.dataDir,self.dataType)
        self.coco_caps=COCO(self.annfile)
        # load and display caption annotations
        self.catIds = self.coco.getCatIds(catNms=['person']);
        temp = self.coco.getImgIds(catIds=self.catIds );
        self.imgIds = []
        for i in range(0, 10):
            self.imgIds.append(temp[i])

    def call(self):

        print('No. of images, category : people - ', len(self.imgIds))
        # j = 0
        ffann = []
        imgMap = []
        for imgId in self.imgIds:
            # if(j==2):
            #     break     #To avoid loading all the annotations,  just for testing purposes
            fann = []
            i = 0
            img = self.coco.loadImgs(imgId)
            img = self.coco.loadImgs(self.imgIds[np.random.randint(0,len(self.imgIds))])[0]
            I = io.imread('C:/Users/pasan/Documents/Motivation/cocoapi-master/PythonAPI/train2017/'+img['file_name'])#img['coco_url']
            K = torch.from_numpy(I)
            imgMap.append({imgId : img['coco_url']})
            # print('\n\nImage Id: ', imgId, '\n')
            # print( 'image tensor shape: ', K.shape)
            annIds = self.coco_caps.getAnnIds(imgId)
            # print(type(annIds))
            for annId in annIds :
                if(i< 3):
                    fann.append(annId)
                    i = i+1
            # print('Annotation IDs: ')
            # print(fann)
            labels = self.getLabel(fann)
            ffann.append(labels)
            # anns = self.coco_caps.loadAnns(fann)
            # # print('\n 3 top selected Annotations: ')
            # anns = self.coco_caps.showAnns(anns)
            # anns = str(anns)
            # print('\nType of anotations', type(anns))
            # j = j+1
        return imgMap, ffann


    def getimage(self, id):
        img = self.coco.loadImgs(id)
        img = self.coco.loadImgs(self.imgIds[np.random.randint(0,len(self.imgIds))])[0]
        I = io.imread(img['coco_url'])
        im = Image.fromarray(I)
        X = preprocess.preprocess_img(im)
        print('type of I', type(I),'\n Shape of X :', X.shape)
        return X

    def getLabel(self, ann):
        label = []
        anns = self.coco_caps.loadAnns(ann)
        for i in range(0, 3):
            label.append(anns[i]['caption'])
        return label









# if __name__ == '__main__':
#
#     L,S = call()
#     print ('yefbkj',len(L))
#     print (S)
