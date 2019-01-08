import torchvision
from torchvision import transforms

def preprocess_img(image):
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
    return preprocess(image)












# import loader
#
#
# def divide_chunks(list , batch_size):
#     for i in range(0, len(list), batch_size):
#         yield list[i: i + batch_size]
#
#
# def chunked_list():
#     image_chunked_list = []
#     label_chunked_list = []
#     batch_size = 32
#
#     imageList,AnnotationList = loader.call()
#     image_chunked_list = list(divide_chunks(imageList, batch_size))
#     label_chunked_list = list(divide_chunks(AnnotationList, batch_size))
#
#     return image_chunked_list, label_chunked_list
#
# def getimage(id, imageList):





# if __name__ == '__main__':
#     chunked_list = []
#     batch_size = 32
#
#     imageList,AnnotationList = loader.call()
#     print(len(imageList), len(AnnotationList))
#
#
#     chunked_list = list(divide_chunks(imageList, batch_size))
#     print(len(chunked_list))
#     print(chunked_list[0])
