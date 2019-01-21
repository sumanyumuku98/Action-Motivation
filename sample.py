import numpy as np 
import matplotlib.pyplot as plt
import os
import pickle as pkl
from torch
from torchvision import transforms
from Vocabuilder import Vocabulary
from PIL import Image
from MN_NET import MN_NET,preprocess_img

IMAGE_PATH = ''

def load_image(image_path, transform= None):
    image = Image.open(image_path)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
return image

VOCAB_PATH=''

MODEL_PATH=''

def main():
    with open(VOCAB_PATH,'rb') as f:
        vocab = pkl.load(f)

    embed_size = 1 # USE THE EMBED SIZE OF THE MODEL YOU TRAINED
    hidden_size = 1 # USE THE HIDDEN SIZE OF THE MODEL YOU TRAINED
    max_seq_len = 1 #USE THE VALUE USED WHILE TRAINING THE MODEL

    model = MN_NET(seq_len= max_seq_len hidden_size= hidden_size, embed_size= embed_size, vocab_size= len(vocab))

    state = torch.load(MODEL_PATH)
    model.load_state_dict(state['state_dict'])
    
    image = load_image(IMAGE_PATH,preprocess_img)
    
    features = model.get_embed_features(image)
    sample_ids = model.sample(features)

    sample_ids = sample_ids[0].cpu().numpy()         
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sample_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    print(sentence)
    image_display = Image.open(IMAGE_PATH)
    plt.imshow(np.asarray(image_display))

if __name__ == '__main__':
    main()

