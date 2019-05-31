import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist, fashion_mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.models import model_from_json

import logging
import sys
import argparse

def test(epoch):

    #load architecture
    json_file = open('models_colin/dc_generator_arch.json', 'r')
    loaded_generator_json = json_file.read()
    json_file.close()
    loaded_generator = model_from_json(loaded_generator_json)
    logging.info('loading arch')

    # load weights into new model
    loaded_generator.load_weights('models_colin/dcgan_generator_epoch_%d.h5' % epoch)
    logging.info("loaded model")
     
    #generate digits
    examples = 100
    figsize = (10, 10)
    dim = (10, 10)
    randomDim = 100
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = loaded_generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)    


# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models_colin/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models_colin/dcgan_discriminator_epoch_%d.h5' % epoch)

def train(epochs, batchSize=128):
    batchCount = X_train.shape[0] / batchSize

    logging.info('epochs %d batch size %d batches per epoch %d' % (epochs, batchSize, batchCount))

    for e in range(1, epochs+1):
        logging.info('epoch %d' % e)
        for _ in range(int(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 5 == 0:
            #plotGeneratedImages(e)
            saveModels(e)

    # Plot losses from every epoch
    #plotLoss(e)


def main(args):

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join('logs', 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not args.train:
        test(args.epochs)
        return None

    
    K.set_image_dim_ordering('th')

    # Deterministic output.
    # Tired of seeing the same results every time? Remove the line below.
    np.random.seed(1000)

    # The results are a little better when the dimensionality of the random vector is only 10.
    # The dimensionality has been left at 100 for consistency with other GAN implementations.
    randomDim = 100

    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, np.newaxis, :, :]

    # Optimizer
    adam = Adam(lr=0.0002, beta_1=0.5)

    # Generator
    generator = Sequential()
    generator.add(Dense(128*7*7, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Reshape((128, 7, 7)))
    generator.add(UpSampling2D(size=(2, 2)))
    generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
    generator.add(LeakyReLU(0.2))
    generator.add(UpSampling2D(size=(2, 2)))
    generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=adam)

    #set up the model
    # serialize model to JSON
    model_json = generator.to_json()
    with open("models_colin/dc_generator_arch.json", "w") as json_file:
        json_file.write(model_json)

    # Discriminator
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(1, 28, 28), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)

    # Combined network
    discriminator.trainable = False
    ganInput = Input(shape=(randomDim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=adam)

    dLosses = []
    gLosses = []
    logging.info('set up models')

    train(50,128)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for SHA with weight sharing')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)    
    parser.add_argument('--train', dest='train', type=int, default=0)   

    args = parser.parse_args()

    main(args)

