# from
from decalib.utils import util
import torch
import torch.nn as nn
import numpy as np
import face_alignment
from scipy.spatial import ConvexHull
from skimage import draw


class CoarseLoss():
    def __init__(self, device='cuda'):
        self.device = device
        self.criterion = nn.MSELoss()

    def lmkLoss(self, origLandmarks, predictedLandmarks):
        """
            Difference between ground truth 2d landmarks
            And corresponding landmarks in FLAME's surface projected by
            estimated camera model
        """
        origLandmarks = torch.squeeze(origLandmarks)
        predictedLandmarks = torch.squeeze(predictedLandmarks)
        loss = origLandmarks-predictedLandmarks
        # print('-----------------After taking difference----------------------')
        # print(loss)
        loss = torch.square(loss)
        # print('---------------After squaring------------------')
        # print(loss)
        loss = torch.sum(loss, dim=1)
        # print('---------------After Summing------------------')
        # print(loss)
        loss = torch.sqrt(loss)
        # print('---------------After Square Rooting------------------')
        # print(loss)
        loss = torch.sum(loss)
        # print('---------------After Summing All Distances------------------')
        # print(loss)
        loss = torch.squeeze(loss)/68

        # loss = torch.squeeze(torch.sum(torch.sqrt(torch.sum(torch.square(origLandmarks-predictedLandmarks),dim=1))))
        # print("Landmarks loss ", loss)

        return loss

    def eyeLoss(self, flameLandmarks, groundtruthLandmarks):
        """
            Computes relative offset 
                1.) b/w landmarks on upper and lower eyelid
                2.) b/w corresponding landmarks in FLAME's surface
            Finally calculates the difference b/w the calculated offsets
            coords of eyes
            left  = 37,38,40,41
            right = 43,44,46,47
        """
        flameLandmarks = torch.squeeze(flameLandmarks)
        groundtruthLandmarks = torch.squeeze(groundtruthLandmarks)
        flameEye = [flameLandmarks[37],
                    flameLandmarks[38],
                    flameLandmarks[41],
                    flameLandmarks[40],
                    flameLandmarks[43],
                    flameLandmarks[44],
                    flameLandmarks[47],
                    flameLandmarks[46]]

        grndEye = [groundtruthLandmarks[37],
                   groundtruthLandmarks[38],
                   groundtruthLandmarks[41],
                   groundtruthLandmarks[40],
                   groundtruthLandmarks[43],
                   groundtruthLandmarks[44],
                   groundtruthLandmarks[47],
                   groundtruthLandmarks[46]]
        origDiff, flameDiff = 0, 0

        origDiff = torch.sqrt(((flameEye[2]-flameEye[0])**2 + (flameEye[3]-flameEye[1])
                              ** 2)/2 + ((flameEye[6]-flameEye[4])**2 + (flameEye[7]-flameEye[5])**2)/2)
        flameDiff = torch.sqrt(((grndEye[2]-grndEye[0])**2 + (grndEye[3]-grndEye[1])
                               ** 2)/2 + ((grndEye[6]-grndEye[4])**2 + (grndEye[7]-grndEye[5])**2)/2)

        totalDiff = torch.abs(torch.squeeze(
            torch.sum(origDiff-flameDiff))).to(device=self.device)
        return totalDiff

    def photometricLoss(self, image, renderedImage, landmarks):
        """
            Computes the difference b/w input and rendered_image
            and multiplies the difference with mask value
        """
        landmarks = torch.squeeze(landmarks)
        landmarks = landmarks.cpu().detach()
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        image = np.transpose(torch.squeeze(
            image).cpu().detach().numpy(), (1, 2, 0))
        renderedImage = np.transpose(torch.squeeze(
            renderedImage).cpu().detach().numpy(), (1, 2, 0))

        vertices = ConvexHull(outline).vertices
        Y, X = draw.polygon(outline[vertices, 1], outline[vertices, 0])
        mask = np.zeros(image.shape)
        mask[Y, X] = (255, 255, 255)
        mask = mask/255
        # print('-------------------------Mask-----------------------------')
        # print(mask)
        # print(mask.shape)
        phLoss = torch.abs(torch.sum(torch.tensor(np.multiply(
            mask, image-renderedImage)))).to(device=self.device)
        # print(phLoss)
        return phLoss/(image.shape[0]*image.shape[1]*image.shape[2])

    def identityLoss(self, inputImage, renderedImage):
        """
            Use face recognition network to output feature embedding
            for rendered and input images
            return Cosine similarity b/w two embeddings
        """
        inputImageEmbedding = util.featureEmbedding(inputImage)
        renderedImageEmbedding = util.featureEmbedding(renderedImage)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        idLoss = (1-cos(inputImageEmbedding, renderedImageEmbedding)
                  ).to(device=self.device)
        # print("Identity loss ", idLoss)
        return idLoss

    def shapeConsistencyLoss(self):
        pass

    def regularization(self, codedict):
        beta = codedict['shape']
        si = codedict['exp']
        reg = torch.sum(torch.squeeze(beta)**2) + \
            torch.sum(torch.squeeze(si)**2).to(device=self.device)
        # print("Regularizaiton Val ", reg)
        return reg


"""Uncomment below code to check loss functions working properly or not"""

"""lossCheck = CoarseLoss()

landmrk = torch.rand(68,2)

print('----------------lmk Loss Check------------------')
print(lossCheck.lmkLoss(landmrk,landmrk))

print('-----------------Eye loss-----------------------')
print(lossCheck.eyeLoss(landmrk,landmrk))

imgTemp = torch.rand(3,224,224)
print('----------------Identity loss-------------------')
print(lossCheck.identityLoss(imgTemp,imgTemp))
"""
