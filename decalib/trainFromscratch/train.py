import face_alignment
from pytorch_lightning import loggers
from pytorch_lightning import callbacks
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset, random_split
from .Loss import CoarseLoss
import torch.nn.functional as F
from ..models.encoders import ResnetEncoder
from ..models.FLAME import FLAME
import pytorch_lightning as pl
from ..utils.config import cfg
from ..utils.renderer import SRenderY
from ..utils import util
import numpy as np
from torchvision import transforms
from skimage.io import imread
import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger
import os
from skimage.transform import warp, estimate_transform
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.optim as optim
import cv2
import re

from torch._six import container_abcs, string_classes, int_classes

from ..datasets import detectors
# It is used whenever the size of the inputs is same
torch.backends.cudnn.benchmark = True

_use_shared_memory = False
np_str_obj_array_pattern = re.compile(r'[SaUO]')

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.cfg = cfg
        # self.device = cfg.device
        self.image_size = self.cfg.dataset.image_size

        self.coarseLoss = CoarseLoss()

        # log hyperparameters
        # self.save_hyperparameters()
        # self.train_acc = pl.metrics.Accuracy()
        # self.valid_acc = pl.metrics.Accuracy()
        # self.test_acc = pl.metrics.Accuracy()
        self.landmarkDetector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, face_detector='sfd', device='cpu')

        # initializing encoder and decoder
        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

    def _create_model(self, model_cfg):
        self.n_param = model_cfg.n_shape+model_cfg.n_tex+model_cfg.n_exp + \
            model_cfg.n_pose+model_cfg.n_cam+model_cfg.n_light
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp,
                         model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i: model_cfg.get(
            'n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)

        # decoder
        self.flame = FLAME(model_cfg).to(self.device)

    def _setup_renderer(self, model_cfg):
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path,
                               uv_size=model_cfg.uv_size).to(self.device)
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32)/255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_eye_mask = F.interpolate(
            mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32)/255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(
            mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32)/255.
        mean_texture = torch.from_numpy(mean_texture.transpose(2, 0, 1))[
            None, :, :, :].contiguous()
        self.mean_texture = F.interpolate(
            mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(
            model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(
                    code_dict[key].shape[0], 9, 3)
        return code_dict

    def encode(self, images):
        parameters = self.E_flame(images)

        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images
        return codedict

    def decode(self, codedict):
        vertices, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])

        # projection
        landmarks2d = util.batch_orth_proj(
            landmarks2d, codedict['cam'])[:, :, :2]
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]
        landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])
        landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]
        landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        trans_vertices = util.batch_orth_proj(vertices, codedict['cam'])
        trans_vertices[:, :, 1:] = -trans_vertices[:, :, 1:]

        # render shape
        shape_image = self.render.render_shape(vertices, trans_vertices)

        outputDict = {
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'trans_vertices': trans_vertices,
            'vertices': vertices,
            'shape_image': shape_image,
            'image': codedict['images']
        }
        return outputDict

    def forward(self, images):
        codedict = self.encode(images)
        decodedDict = self.decode(codedict)
        # print('---------------------Parameters---------------------------')
        # print(self.parameters())

        return codedict, decodedDict

    def loss(self, codedict, decodedDict, landmarksOrig):

        # landmark regularization loss
        # print(landmarksOrig.shape)
        # print(decodedDict['landmarks3d'].shape)
        lmk = self.coarseLoss.lmkLoss(
            landmarksOrig, decodedDict['landmarks2d'])

        # eye loss
        eyeLoss = self.coarseLoss.eyeLoss(
            landmarksOrig, decodedDict['landmarks2d'])

        # photometric loss
        phLoss = self.coarseLoss.photometricLoss(
            decodedDict['image'], decodedDict['shape_image'], landmarksOrig)

        #  identity loss
        idLoss = self.coarseLoss.identityLoss(
            decodedDict['image'], decodedDict['shape_image'])

        """
            Add shape consistency
        """
        reg = self.coarseLoss.regularization(codedict)

        lossTotal = lmk+eyeLoss+idLoss+phLoss+reg
        return lossTotal

    def training_step(self, batch, batch_idx):
        images = batch['image']
        # print(len(images))

        trainLoss = 0
        for i in range(len(images)):
            codedict, decodedDict = self.forward(
                images[i].view(-1, 3, 224, 224).float())
            img = np.transpose(torch.squeeze(
                images[i]).cpu().detach().numpy(), (1, 2, 0))
            try:
                grndLandmarks = self.landmarkDetector.get_landmarks(img*255)
                grndLandmarks = grndLandmarks[0]
                grndLandmarks = torch.tensor(grndLandmarks).to(device='cuda')
                grndLandmarks = grndLandmarks.view(-1, 68, 2)
            except Exception as e:
                print(e)
                continue
            trainLoss += self.loss(codedict, decodedDict, grndLandmarks)
        self.log('train_loss', trainLoss, on_epoch=True)
        trainLoss = trainLoss/len(images)
        return trainLoss
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        validLoss = 0
        for i in range(len(images)):
            codedict, decodedDict = self.forward(
                images[i].view(-1, 3, 224, 224).float())
            img = np.transpose(torch.squeeze(
                images[i]).cpu().detach().numpy(), (1, 2, 0))
            try:
                grndLandmarks = self.landmarkDetector.get_landmarks(img*255)
                grndLandmarks = grndLandmarks[0]
                grndLandmarks = torch.tensor(grndLandmarks).to(device='cuda')
                grndLandmarks = grndLandmarks.view(-1, 68, 2)
            except Exception as e:
                print(e)
                continue
            validLoss += self.loss(codedict, decodedDict, grndLandmarks)
        validLoss=validLoss/len(images)
        self.log('valid_loss', validLoss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # print(self.parameters)
        return optim.Adam(self.parameters(), lr=1e-4)


class MyDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 4):
        super().__init__()
        self.dataset = Dataset_3D(csv_file=os.getcwd()+'/3d-face/decalib/datasets/data.csv',
                                  root_dir=os.getcwd()+'/3d-face/decalib/datasets/300W_LP',
                                  transform=transforms.Compose([
                                      Preprocess()
                                  ]))
        self.batch_size = batch_size
        # update size of random split
        l = len(self.dataset)
        valLen = l-int(0.3*l)
        # print(valLen)
        self.train, self.val = random_split(self.dataset, [valLen, l-valLen])

    def collate_fn(self, batch):
        '''
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
            This function refers to touch's default_collate function, which is also the default proofreading method of DataLoader. When the batch contains data such as None,
            The default default_collate squad method will get an error
            One solution is:
            Determine whether the image in the batch is None. If it is None, it is cleared in the original batch, so that it can avoid errors in the iteration.
        :param batch:
        :return:
        '''
        r"""Puts each data field into a tensor with outer dimension batch size"""
        # Add here: Determine whether image is None. If it is None, it will be cleared in the original batch, so you can avoid errors in iteration.
        # print(type(batch))
        # print(batch)
        if isinstance(batch, list):
            batch = [(image) for (image) in batch if image is not None]
        if batch == []:
            return (None)

        elem_type = type(batch[0])
        if isinstance(batch[0], torch.Tensor):
            out = None
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(error_msg_fmt.format(elem.dtype))

                return self.collate_fn([torch.from_numpy(b) for b in batch])
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0], float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(batch[0], int_classes):
            return torch.tensor(batch)
        elif isinstance(batch[0], string_classes):
            return batch
        elif isinstance(batch[0], container_abcs.Mapping):
            return {key: self.collate_fn([d[key] for d in batch]) for key in batch[0]}
        elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
            return type(batch[0])(*(self.collate_fn(samples) for samples in zip(*batch)))
        elif isinstance(batch[0], container_abcs.Sequence):
            transposed = zip(*batch)  # ok
            return [self.collate_fn(samples) for samples in transposed]

        raise TypeError((error_msg_fmt.format(type(batch[0]))))

    def train_dataloader(self):
        return DataLoader(dataset=self.train, batch_size=self.batch_size, num_workers=2, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val, batch_size=self.batch_size, num_workers=2, collate_fn=self.collate_fn)

    def test_dataloader(self):
        pass


class Preprocess(object):
    def __init__(self, iscrop=True, crop_size=224, scale=1.25, face_detector='fan'):
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.face_detector = face_detector
        if(face_detector == 'fan'):
            self.face_detector = detectors.FAN()

    def __call__(self, sample):
        image = sample['image']
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape

        if self.iscrop:
            bbox, bbox_type = self.face_detector.run(image)
            # print('---------------------------------Bounding box and It type-------------------------')
            # print(bbox,bbox_type)
            if len(bbox) < 4:
                print('no face detected! run original image')
                left = 0
                right = h-1
                top = 0
                bottom = w-1
            else:
                left = bbox[0]
                right = bbox[2]
                top = bbox[1]
                bottom = bbox[3]
            old_size, center = self.bbox2point(
                left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] -
                               size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])

        DST_PTS = np.array(
            [[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(
            self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2, 0, 1)

        return {'image': torch.tensor(dst_image).float()}

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        """
        bbox from detector and landmarks are different
        """
        if type == 'kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0,
                              bottom - (bottom - top) / 2.0])
        elif type == 'bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0,
                              bottom - (bottom - top) / 2.0 + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center


class Dataset_3D(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        img_name = os.path.join(
            self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        sample = {'image': image}

        if self.transform:
            try:
                sample = self.transform(sample)
                return sample
            except Exception as e:
                print(e)
    def __len__(self):
        return len(self.landmarks_frame)


if __name__ == "__main__":
    # ---------------Uncomment this to run training------------#
    # torch.multiprocessing.set_start_method('spawn')
    model = LitAutoEncoder()
    dm = MyDataModule()
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        dirpath='savedModel/',
        filename='decaCoarse-{epoch:02d}-{valid_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    wandb_logger = WandbLogger()
    wandb.init()
    trainer = pl.Trainer(gpus=1, logger=wandb_logger,callbacks=[checkpoint_callback])
    trainer.fit(model, dm)

    # face_dataset = Dataset_3D(csv_file='/home/nandwalritik/3DFace/decalib/datasets/data.csv',
    #                           root_dir='/home/nandwalritik/3DFace/decalib/datasets/300W_LP',
    #                           transform=transforms.Compose([
    #                               Rescale(224),
    #                               Normalize(),
    #                               ToTensor()
    #                           ]))

    # for i in range(len(face_dataset)):
    #     if i == 10:
    #         break
    #     sample = face_dataset[i]
    #     print(i, sample['image'].size(), sample['landmarks'].size())
    #     util.show_landmarks(sample['image'], sample['landmarks'])
