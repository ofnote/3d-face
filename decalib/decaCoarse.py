
import torch
import torch.nn.functional as F
from decalib.trainFromscratch.train import LitAutoEncoder
from .utils.config import cfg
from .utils.renderer import SRenderY
from .utils import util
import cv2
from skimage.io import imread
import numpy as np


class DecaCoarse():
    def __init__(self, path, config=None, device='cuda'):
        super().__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.ckpt = LitAutoEncoder.load_from_checkpoint(path)
        self.device = device
        self.image_size = self.cfg.dataset.image_size
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
        self.E_flame = self.ckpt.E_flame.to(device=self.device)
        self.flame = self.ckpt.flame.to(device=self.device)

    def _setup_renderer(self, model_cfg):
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path,
                               uv_size=model_cfg.uv_size).to(self.device)
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
    @torch.no_grad()
    def encode(self, images):
        parameters = self.E_flame(images)

        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images
        return codedict
    @torch.no_grad()
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

    
    # def forward(self,images):

