# import preprocess
# if __name__ == '__main__':
#     print('ddd')
#     print(len(preprocess.chunked_list()))
import torch
from torch.utils import data
# import preprocess
import loader
import numpy as np
from tempfile import TemporaryFile
import torchvision
import Vocabuilder
import pickle
import nltk
nltk.download('punkt')

class Dataset(data.Dataset):
    def __init__(self, imgIds, labelIds, vocab):
        self.imgIds = imgIds
        self.labelIds = labelIds
        self.vocab = vocab

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, index):
        label = []#list of annotation id  for one image
        ann = []# top 3  annotation from list of annotations for one image
        padding = 20 #assumption for now, ideally max length of annotation
        print('INDEXXXX', index)
        id = self.imgIds[index]
        label = self.labelIds[index]
        print('label', label)
        x = load.getimage(id)
        for i in range(0, 3):
            ann.append(label[i])
        anns = load.getLabel(ann)
        print('anns', anns)
        annotations = []
        for ann in anns:
            tokens = nltk.tokenize.word_tokenize(str(ann).lower())
            annotation = []
            print('ddfgh', vocab('<start>'))
            annotation.append(vocab('<start>'))
            annotation.extend([vocab(token) for token in tokens])
            annotation.append(vocab('<end>'))
            if len(annotation) < padding:
                for i in range(padding - len(annotation)):
                    annotation.append(0)
            print(annotation)

            annotations.append(torch.Tensor(annotation))
            print('sdefrgthyjus',annotations)

        # temp = TemporaryFile()
        # np.save(temp, x)
        # x = torch.load(temp)
        return x, annotations


def main():
    print('çaller')
    load = loader.L()
    imageList,AnnotationList = load.call()
    threshold = 1
    vocab = Vocabuilder.buildVocab(load, AnnotationList, threshold)
    vocab_path = r'C:/Users/pasan/Documents/Motivation/cocoapi-master/PythonAPI/vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print('Vocab size: ', len(vocab))

    with open(vocab_path, 'rb') as f:
        vocab2 = pickle.load(f)
    print(len(imageList), len(AnnotationList))
    # Dataset(imageList, AnnotationList)
    # image = l.getimage()
    train__dataset = Dataset(imageList, AnnotationList, vocab2)
    print(train__dataset)
    train_loader = torch.utils.data.DataLoader(train__dataset, batch_size = 5, shuffle = True)
    return train_loader;
    #     print(type(train_loader))
#     for image, annVecList in train_loader:
#         for annVec in annVecList:
#             print(len(image), len(annVec))
