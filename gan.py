import logging
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist, fashion_mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.models import model_from_json


class GAN(object):

    def __init__(self, learning_rate, random_dim):
        self.random_dim = random_dim
        self.optimizer = Adam(lr=learning_rate, beta_1=0.5)
        self.random_dim = random_dim

        self.gan, self.generator, self.discriminator = self.build_gan()
        self.gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def train_model(self, xtrain, ytrain, valdata, batch_size, epochs):

        batch_count = xtrain.shape[0] / batch_size
        d_losses = []
        g_losses = []

        for e in range(epochs):
            if e == 1 or e % 20 == 0:
                logging.info('epoch %d' % e)
                
            for _ in range(int(batch_count)):
                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                image_batch = xtrain[np.random.randint(0, xtrain.shape[0], size=batch_size)]

                # Generate fake images
                gen_images = self.generator.predict(noise)
                images = np.concatenate([image_batch, gen_images])

                # Labels for generated and real data
                y_dis = np.zeros(2*batch_size)
                # One-sided label smoothing
                y_dis[:batch_size] = 0.9

                # Train discriminator
                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch(images, y_dis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                y_gen = np.ones(batch_size)
                self.discriminator.trainable = False
                g_loss = self.gan.train_on_batch(noise, y_gen)

            # Store loss of most recent batch from this epoch
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            if e % 10 == 0:
                self.save_models(e)
                self.generate_images(e)
                self.plot_loss(e, d_losses=d_losses, g_losses=g_losses)

                logging.info('epoch: {}, discriminator loss: {}, generator loss: {}'
                                .format(e, d_losses[-1], g_losses[-1]))

    def generate_images(self, epoch):

        json_file = open("models/generator_arch.json", 'r')
        loaded_generator_json = json_file.read()
        json_file.close()
        loaded_generator = model_from_json(loaded_generator_json)
        logging.info('loading arch')

        # load weights into new model
        loaded_generator.load_weights('models/gan_generator_epoch_{}.h5'.format(epoch))
         
        examples = 100
        figsize = (10, 10)
        dim = (10, 10)
        noise = np.random.normal(0, 1, size=[examples, self.random_dim])
        gen_images = loaded_generator.predict(noise)
        gen_images = gen_images.reshape(examples, 28, 28)

        plt.figure(figsize=figsize)
        for i in range(gen_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(gen_images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('images/gan_generated_image_epoch_{}.png'.format(epoch))

    def build_gan(self):

        generator = Sequential()
        generator.add(Dense(256, input_dim=self.random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(784, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        # Save the architecture
        model_json = generator.to_json()
        with open("models/generator_arch.json", "w") as json_file:
            json_file.write(model_json)

        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        discriminator.trainable = False
        gan_input = Input(shape=(self.random_dim,))
        x = generator(gan_input)
        gan_output = discriminator(x)

        return Model(inputs=gan_input, outputs=gan_output), generator, discriminator

    def plot_loss(self, epoch, d_losses, g_losses):
        plt.figure(figsize=(10, 8))
        plt.plot(d_losses, label='Discriminitive loss')
        plt.plot(g_losses, label='Generative loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('images/gan_loss_epoch_{}.png'.format(epoch))
        logging.info('saved plots')

    def save_models(self, epoch):
        self.generator.save('models/gan_generator_epoch_{}.h5'.format(epoch))
        self.discriminator.save('models/gan_discriminator_epoch_{}.h5'.format(epoch))
        logging.info('saved weights {}'.format(epoch))


def main(args):
    batch_size = args.batch_size
    epochs = args.epochs
    dataset = args.dataset
    learning_rate = args.learning_rate
    random_dim = args.random_dim
    gen_only = args.gen_only
    gen_from_epoch = args.gen_from_epoch

    K.set_image_dim_ordering('th')
    np.random.seed(1000)

    # Load MNIST data
    if args.dataset == 'mnist':
        (xtrain, ytrain), (xtest, ytest) = mnist.load_data()        
    elif args.dataset == 'fashion_mnist':
        (xtrain, ytrain), (xtest, ytest) = fashion_mnist.load_data()                
    else:
        print('invalid dataset')
        sys.exit()

    xtrain = (xtrain.astype(np.float32) - 127.5)/127.5
    xtrain = xtrain.reshape(60000, 784)

    gan = GAN(learning_rate=learning_rate, random_dim=random_dim)
    if not gen_only:
        gan.train_model(xtrain, ytrain, (xtest, ytest), batch_size=batch_size, epochs=epochs)
    gan.generate_images(epoch=gen_from_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for GAN')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--dataset', type=str, default='mnist', help='mnist or fashion_mnist')
    parser.add_argument('--random_dim', type=int, default=100, help='size of random vector')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate for training')
    parser.add_argument('--gen_only', type=bool, default=False, help='whether or not to skip training')
    parser.add_argument('--gen_from_epoch', type=int, default=200, help='if gen_only, which epoch to generate from')


    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join('logs', 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(args)
    main(args)
