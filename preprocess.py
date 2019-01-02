import loader

def divide_chunks(list , batch_size):
    for i in range(0, len(list), batch_size):
        yield list[i: i + batch_size]


if __name__ == '__main__':
    chunked_list = []
    batch_size = 32

    imageList,AnnotationList = loader.call()
    print(len(imageList), len(AnnotationList))


    chunked_list = list(divide_chunks(imageList, batch_size))
    print(len(chunked_list))
    print(chunked_list[0])
