'''
import data here and have utility functions that could help
'''
import numpy as np
import pickle

import matplotlib.pyplot as plt
import librosa.display
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

from keras.applications.vgg16 import VGG16 
from keras.models import Model

def audio_transform(file):
    samples, sample_rate = librosa.load(file, sr=None)

    # Short-time Fourier transform
    sgram = librosa.stft(samples)
    # librosa.display.specshow(sgram)
    magnitude, phase = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=magnitude, sr=sample_rate, n_mels=128, fmax=8000)
    # mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    mel_sgram = librosa.power_to_db(mel_scale_sgram, ref=np.min)
    # Image size
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = float(mel_sgram.shape[1]) / float(100)
    fig_size[1] = float(mel_sgram.shape[0]) / float(100)
    plt.rcParams['figure.figsize'] = fig_size
    plt.axis('off')
    librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='hz')
    plt.savefig('Spectogram_Images_2/'+'file'+'.png', bbox_inches=None, pad_inches=0)
    plt.close()
    return 

        
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

def pca_transform(features):
    pca_load = pickle.load(open('model/pca.pkl','rb'))
    pca_components = pca_load.transform(features)
    return pca_components


# if __name__ == '__main__':

