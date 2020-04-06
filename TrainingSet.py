from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from random import Random
import torch

class TrainingSet(Dataset):
    def __init__(self,paths,total_number,h=120,w=160):
        self.h = h
        self.w = w
        self.length = total_number
        self.hog = cv2.HOGDescriptor((16, 32), (16,16), (16,16), (8,8), 8,1)
        self.imgs = paths
        self.r = Random(0)
        
    def __len__(self):
        return self.length
        
    def loader(self,path):
        img = cv2.imread(path,0)
        img = cv2.resize(img, (self.w, self.h), interpolation = cv2.INTER_CUBIC)
        #img = cv2.resize(img,(self.h,self.w))
        #img_tensor = torch.from_numpy(img).to(device)
        return img
    
    def randPerspectiveWarp(self,im):
        w = self.w
        h = self.h
        r = self.r
        minsx = [ 0, 3*w/4 ]
        maxsx = [ w/4, w ]
        minsy= [ 0, 3*h/4 ]
        maxsy = [ h/4, h ]


        pts_orig = np.zeros((4, 2), dtype=np.float32)
        pts_warp = np.zeros((4, 2), dtype=np.float32) 
        pts_orig[0, 0] = 0
        pts_orig[0, 1] = 0
    
        pts_orig[1, 0] = 0
        pts_orig[1, 1] = h

        pts_orig[2, 0] = w
        pts_orig[2, 1] = 0

        pts_orig[3, 0] = w
        pts_orig[3, 1] = h
        pts_warp[0, 0] = r.uniform(minsx[0], maxsx[0])
        pts_warp[0, 1] = r.uniform(minsy[0], maxsy[0])
    
        pts_warp[1, 0] = r.uniform(minsx[0], maxsx[0])
        pts_warp[1, 1] = r.uniform(minsy[1], maxsy[1])

        pts_warp[2, 0] = r.uniform(minsx[1], maxsx[1])
        pts_warp[2, 1] = r.uniform(minsy[0], maxsy[0])

        pts_warp[3, 0] = r.uniform(minsx[1], maxsx[1])
        pts_warp[3, 1] = r.uniform(minsy[1], maxsy[1])

    # compute the 3x3 transform matrix based on the two planes of interest
        T = cv2.getPerspectiveTransform(pts_warp, pts_orig)

    # apply the perspective transormation to the image, causing an automated change in viewpoint for the net's dual input
        im_warp = cv2.warpPerspective(im, T, (w, h))
        return im_warp
    
    def __getitem__(self,index):
        path = self.imgs[index]
        #print(path)
        ori_img = self.loader(path)
        img_warp = self.randPerspectiveWarp(ori_img)
        #ori_img = cv2.cvtColor(ori_img,cv2.IMREAD_GRAYSCALE)
        #img_warp = cv2.cvtColor(img_warp,cv2.IMREAD_GRAYSCALE)
        #plt.imshow(ori_img)
        #plt.axis('off')
        #plt.show()
        #plt.imshow(img_warp)
        #plt.axis('off')
        #plt.show()
        self.r.seed(index) # adds extra randomness, but is still reproduceable with the same dataset
        switchFlag = self.r.randint(0,1)
        if switchFlag:
            img = img_warp
            des = self.hog.compute(cv2.resize(ori_img, (160,120)))
            #print(des)
        else:
            img = ori_img
            des = self.hog.compute(cv2.resize(img_warp, (160,120)))
        #des_test = self.hog.compute(cv2.resize(img_warp, (160,120)))
        #for ii in range(3648):
            #print(des[ii],des_test[ii])
        #img = torch.from_numpy(img).to(device)
        #des = torch.from_numpy(des).to(device)
        #print(img.dtype)
        #print(des.dtype)
        img = img.reshape(1,self.h,self.w)
        des = des.reshape(3648,)
        #print('mean:',np.mean(des))
        #print('var:',np.var(des))
        return img,des