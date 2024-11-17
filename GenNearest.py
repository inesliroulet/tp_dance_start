
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it selects the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        """ generator of image from skeleton """
        # TP-TODO
        closest_distance = float('inf')
        closest_image = None
        
        for i in range(self.videoSkeletonTarget.skeCount()):
            target_ske = self.videoSkeletonTarget.ske[i]
            
            distance = ske.distance(target_ske)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_image = self.videoSkeletonTarget.readImage(i)
                
        return closest_image
        
        #empty = np.ones((64,64, 3), dtype=np.uint8)
        #return empty
