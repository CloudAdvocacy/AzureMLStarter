# KeraGAN trainer script

import argparse
import keragan
import keras
import os
import glob
from azureml.core.run import Run
import matplotlib.pyplot as plt

print("KeraGAN Trainer, version {}".format(keragan.__version__))

run = Run.get_context()

parser = argparse.ArgumentParser(description="KeraGAN Trainer")

parser.add_argument("--path",help="Azure ML Datastore dir")
parser.add_argument("--dataset",help="Dataset name within datastore")
parser.add_argument("--size",help="Image size to use", default=512, type=int)
parser.add_argument("--aspect_variance",help="Allowed aspect variance", default=0.5, type=float)
parser.add_argument("--model_path",help="Path to use for saving models", default='models')
parser.add_argument("--samples_path",help="Path to use for saving samples", default='samples')
parser.add_argument("--save_npy_path",help="Filename to save cached dataset for faster loading")
parser.add_argument("--limit",help="Limit # of images to use",type=int,default=None)
parser.add_argument("--batch_size",help="Minbatch size to use",type=int,default=128)
parser.add_argument("--save_interval",help="Epochs between saving models",type=int,default=100)
parser.add_argument("--save_img_interval",help="Epochs between generating image samples",type=int,default=100)
parser.add_argument("--print_interval",help="Epochs between printing",type=int,default=10)
parser.add_argument("--sample_images",help="View image sample",action='store_const',default=False,const=True)
parser.add_argument("--no_samples",help="Number of sample images to generate during training",type=int,default=10)
parser.add_argument("--latent_dim",help="Dimension of latent space",type=int,default=256)
parser.add_argument("--ignore_smaller",help="Ignore images smaller than required size",action='store_const',default=False,const=True)
parser.add_argument("--crop",help="Crop images to desired aspect ratio",action='store_const',default=False,const=True)
parser.add_argument("--epochs",help="Number of epochs to train",type=int,default=100)
parser.add_argument("--lr",help="Learning rate",type=float,default=0.0001)

args = parser.parse_args()

args.height = args.size
args.width = args.size
args.optimizer = keras.optimizers.Adam(lr=args.lr, beta_1=0.5, beta_2=0.999, decay=0.0, amsgrad=False)
args.path = os.path.join(args.path,args.dataset)

gan = keragan.DCGAN(args)
imsrc = keragan.ImageDataset(args)
imsrc.load()
train = keragan.GANTrainer(image_dataset=imsrc,gan=gan,args=args)

def callbk(tr):
    if tr.gan.epoch % 20 == 0:
        res = tr.gan.sample_images(n=3)
        fig,ax = plt.subplots(1,len(res))
        for i,v in enumerate(res):
            ax[i].imshow(v[0])
        run.log_image("Sample",plot=plt)

train.train(callbk)
