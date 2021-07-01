from decalib.datasets.detectors import MTCNN
from numpy.matrixlib.defmatrix import matrix
from skimage import transform
from decalib.datasets import datasets
from decalib.trainFromscratch.Loss import CoarseLoss
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import util
from decalib.deca import DECA
import os
import sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from torch.nn.functional import interpolate
from tqdm import tqdm
import pandas as pd
import torch
import face_alignment
from skimage.transform import matrix_transform, warp
from skimage import io
from facenet_pytorch import MTCNN as mtcnn
from PIL import Image
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)
    loss = CoarseLoss()
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, face_detector='sfd', device='cuda')
    dataFile = pd.read_csv(
        '/home/nandwalritik/3DFace/decalib/datasets/data.csv')
    # load test images
    args.inputpath = list(dataFile.loc[:, 'image_name'].values)
    dictMap = {}
    # for i in range(len(dataFile)):
    #     inputImg = io.imread("/home/nandwalritik/3DFace/decalib/datasets/300W_LP" + '/'+dataFile.iloc[i]['image_name'])
    #     arr = fa.get_landmarks(inputImg)
    #     # print(arr)
    #     dictMap[dataFile.iloc[i]['image_name']] = torch.tensor(arr[0]).to(device='cuda')
    testdata = datasets.TestData(
        args.inputpath, iscrop=args.iscrop, face_detector=args.detector)

    # print('-----------------------------------------DictMap-----------------------------------------')
    # print(dictMap)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca = DECA(config=deca_cfg, device=device)
    # for i in range(len(testdata)):
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None, ...]
        tform = testdata[i]['tform']

        img = np.transpose(torch.squeeze(
            images).cpu().detach().numpy(), (1, 2, 0))
        arr = fa.get_landmarks(img*255)
        dictMap[name] = torch.tensor(arr[0]).to(device='cuda')
        # util.showImage(torch.squeeze(images),"Image")
        # Dont multiply pixel value by 255 for getting correctly plotted this image with landmarks
        # util.show_landmarks(torch.squeeze(images),dictMap[name])
        # print('-------------------------------------NumberOfImages------------------------------------')
        # print(len(images))
        # print(images.shape)
        # print(name)
        codedict = deca.encode(images)

        # print('------------------------------------Printing Codedict----------------------------------')
        # print(codedict.keys())
        # print(codedict['shape'].shape,codedict['exp'].shape)

        opdict, visdict = deca.decode(codedict)  # tensor
        print('----------------------------------Testing Custom Losses--------------------------------')
        print(images.shape)
        lmkloss = loss.lmkLoss(dictMap[name], torch.squeeze(
            opdict['landmarks2d']))
        print('------------------------------------Lmkloss--------------------------------------------')
        print(lmkloss)
        eyeLoss = loss.eyeLoss(torch.squeeze(
            opdict['landmarks2d']), dictMap[name])
        print('------------------------------------Eyeloss--------------------------------------------')
        print(eyeLoss)
        idenLoss = loss.identityLoss(
            visdict['inputs'], visdict['shape_images'])
        print('------------------------------------Identity Loss--------------------------------------------')
        print(idenLoss)
        util.show_comp_landmarks(torch.squeeze(
            images), dictMap[name], opdict['landmarks2d'])

        # print('------------------------------------Printing opDict----------------------------------')
        # print(visdict.keys())
        # print(util.showImage(torch.squeeze(visdict['landmarks2d'])))

        # print('------------------------------------Printing visDict----------------------------------')
        # print(codedict.keys())
        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        if args.saveDepth:
            depth_image = deca.render.render_depth(
                opdict['transformed_vertices']).repeat(1, 3, 1, 1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, name +
                        '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name, name +
                       '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, name +
                       '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            deca.save_obj(os.path.join(
                savefolder, name, name + '.obj'), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'),
                        deca.visualize(visdict))
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                if vis_name not in visdict.keys():
                    continue
                image = util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(savefolder, name, name + '_' +
                            vis_name + '.jpg'), util.tensor2image(visdict[vis_name][0]))
    print(f'-- please check the results in {savefolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    main(parser.parse_args())
