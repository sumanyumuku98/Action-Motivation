# import preprocess
# if __name__ == '__main__':
#     print('ddd')
#     print(len(preprocess.chunked_list()))
import torch
from torch.utils import data
# import preprocess
import loader
import skimage.io as io
import numpy as np
from tempfile import TemporaryFile
import torchvision
import Vocabuilder
import pickle
import nltk
import preprocess
from PIL import Image

nltk.download('punkt')


class Dataset(data.Dataset):
    def __init__(self, imgMap, labelIds, vocab):
        self.imgMap = imgMap
        self.labelIds = labelIds
        self.vocab = vocab

    def __len__(self):
        return len(self.imgMap)

    def __getitem__(self, index):
        label = []  # list of annotation id  for one image
        ann = []  # top 3  annotation from list of annotations for one image
        padding = 20
        print('INDEXXXX', index)
        img = self.imgMap[index]
        anns = self.labelIds[index]
        print(img)
        for key in img:
            x = io.imread(img.get(key))
        x = Image.fromarray(x.astype('uint8'), 'RGB')
        x = preprocess.preprocess_img(x)

        annotations = []
        for ann in anns:
            tokens = nltk.tokenize.word_tokenize(str(ann).lower())
            annotation = []
            annotation.append(self.vocab('<start>'))
            annotation.extend([self.vocab(token) for token in tokens])
            annotation.append(self.vocab('<end>'))
            # if len(annotation) < padding:
            #     for i in range(padding - len(annotation)):
            #         annotation.append(0)

            annotations.append(torch.Tensor(annotation))
        print('Annotations', annotations)

        # temp = TemporaryFile()
        # np.save(temp, x)
        # x = torch.load(temp)
        return x, annotations


#
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).

    print('ýyoy')
    images, captions = zip(*data)
    print('STUCKK', captions)

    # # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    print('Images', images)
    #
    # # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = []
    for caps in captions:
        length = [len(cap) for cap in caps]
        lengths.append(length)

    targets = []
    temp = []
    for i in lengths:
        temp.append(max(i))
    maxLength = max(temp)

    # print('sdfgh', len(captions*3), maxLength)
    targets = torch.zeros(len(captions) * 3, maxLength).long()

    i = 0
    for capList in captions:
        for cap in capList:
            # print('ert', i, len(cap))
            targets[i, :len(cap)] = cap[:len(cap)]
            i += 1

    return images, targets, lengths


def main():
    print('çaller')
    load = loader.L()
    imageMap, AnnotationList = load.call()
    print('length check', len(imageMap), len(AnnotationList))
    threshold = 1
    vocab = Vocabuilder.buildVocab(load, AnnotationList, threshold)
    vocab_size = len(vocab)
    vocab_path = r'C:/Users/pasan/Documents/Motivation/cocoapi-master/PythonAPI/vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print('Vocab size: ', len(vocab))

    with open(vocab_path, 'rb') as f:
        vocab2 = pickle.load(f)

    # Dataset(imageList, AnnotationList)
    # image = l.getimage()
    train__dataset = Dataset(imageMap, AnnotationList, vocab2)
    print(train__dataset)
    train_loader = torch.utils.data.DataLoader(train__dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
    return train_loader, vocab_size

# if __name__ == '__main__':
#     train_loader = main()
#     print(type(train_loader))
#     for image, annVecList in train_loader:
#         for annVec in annVecList:
#             print(len(image), len(annVec))
