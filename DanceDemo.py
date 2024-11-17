import numpy as np
import cv2
import os
import pickle
import sys

from VideoSkeleton import VideoSkeleton
from VideoSkeleton import combineTwoImages
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenNearest import GenNeirest
from GenVanillaNN import *
from GenGAN import *


class DanceDemo:
    """ class that run a demo of the dance.
        The animation/posture from self.source is applied to character define self.target using self.gen
    """
    def __init__(self, filename_src, typeOfGen=4, nb_epochs=20, load=True, train=False, save_after_train=False, save_path="saved_model.pkl "):
        self.target = VideoSkeleton( "data\\taichi1.mp4" )
        self.source = VideoReader(filename_src)
        if typeOfGen==1:           # Nearest
            print("Generator: GenNeirest")
            self.generator = GenNeirest(self.target)
        elif typeOfGen==2:         # VanillaNN
            print("Generator: GenSimpleNN with skeleton vector as input")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=1)
            if load:
                self.generator.netG.load_state_dict(torch.load("GenVanillaNN1.pkl"))
            if train:
                self.generator.train(nb_epochs)
                if save_after_train:
                    torch.save(self.generator.netG.state_dict(), save_path+".pkl")
        elif typeOfGen==3:         # VanillaNN
            print("Generator: GenSimpleNN with skeleton image as input")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=2)
            if load:
                self.generator.netG.load_state_dict(torch.load("GenVanillaNN2.pkl"))
            if train:
                self.generator.train(nb_epochs)
                if save_after_train:
                    torch.save(self.generator.netG.state_dict(), save_path+".pkl")
        elif typeOfGen==4:         # GAN
            print("Generator: GenSimpleNN with GAN (discriminator)")
            self.generator = GenGAN(self.target, loadFromFile=True)
            if load:
                self.generator.netD.load_state_dict(torch.load("GAN_discriminator.pkl"))
                self.generator.netG.load_state_dict(torch.load("GAN_generator.pkl"))
            if train:
                self.generator.train(nb_epochs)
                if save_after_train:
                    torch.save(self.generator.netD.state_dict(), save_path+"_discriminator.pkl")
                    torch.save(self.generator.netG.state_dict(), save_path+"_generator.pkl")
        else:
            print("DanceDemo: typeOfGen error!!!")


    def draw(self):
        ske = Skeleton()
        image_err = np.zeros((256, 256, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)  # (B, G, R)
        for i in range(self.source.getTotalFrames()):
            image_src = self.source.readFrame()
            if i%5 == 0:
                isSke, image_src, ske = self.target.cropAndSke(image_src, ske)
                if isSke:
                    ske.draw(image_src)
                    image_tgt = self.generator.generate(ske)            # GENERATOR !!!
                    image_tgt = image_tgt*255
                    image_tgt = cv2.resize(image_tgt, (128, 128))
                else:
                    image_tgt = image_err
                image_combined = combineTwoImages(image_src, image_tgt)
                image_combined = cv2.resize(image_combined, (512, 256))
                cv2.imshow('Image', image_combined)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                if key & 0xFF == ord('n'):
                    self.source.readNFrames( 100 )
        cv2.destroyAllWindows()



if __name__ == '__main__':
    # NEAREST = 1
    # VANILLA_NN_SKE = 2
    # VANILLA_NN_Image = 3
    # GAN = 4
    gen = int(input("What strategy do you want to use? (1: Nearest image, 2: Vanilla NN with skeleton vector as input, 3: Vanilla NN with skeleton image as input, 4: GAN)\n"))
    if gen not in range(1,5):
        print("Invalid option.")
        exit()
    else:
        if gen == 1:
            nb_ep = 0
            load = False
            train = False
            save_after_train = False
            save_path = ""
        if gen in range(2,5):
            rep = input('Do you want to load the pre-trained model or train it yourself? Answer with "load" or "train"\n')
            if rep not in ["load", "train"]:
                print("Invalid option.")
                exit()
            else:
                if rep == "load":
                    nb_ep = 0
                    load = True
                    train = False
                    save_after_train = False
                    save_path = ""
                if rep == "train":
                    nb_ep = int(input("On how many epochs do you want to train it?\n"))
                    load = False
                    train = True
                    save_rep = input("Do you want to save the model after training? (y/n)\n")
                    if save_rep not in ["y", "yes", "Yes", "Y", "n", "no", "No", "N"]:
                        print("Invalid option.")
                        exit()
                    else:
                        if save_rep in ["y", "yes", "Yes", "Y"]:
                            save_after_train = True
                            save_path = input("Under what filename?\n")
                        if save_rep in ["n", "no", "No", "N"]:
                            save_after_train = False
                            save_path = ""
        
    GEN_TYPE = gen
    ddemo = DanceDemo("data\\taichi2_full.mp4", GEN_TYPE, nb_ep, load, train, save_after_train, save_path)
    ddemo.draw()