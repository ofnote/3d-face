## Getting Started
Clone the repo:
```bash
    git clone https://github.com/ofnote/3d-face
    cd 3d-face
```
### Requirements
* Python 3.7 (numpy, skimage, scipy, opencv)  
* PyTorch >= 1.6 (pytorch3d)  
* face-alignment (Optional for detecting face)  
  You can run 
  ```bash
  pip install -r requirements.txt
  ```
  Then follow the instruction to install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md).
### Training
* To generate CSV file containing images paths, first specify the directory inside preprocess.py containing training images
  Run
  ```bash
  python3 -m decalib.datasets.preprocess
  ```
* Specify the dataset foldername inside train file in line 267.
* To start training
  Run
  ```bash
  python3 -m decalib.trainFromscratch.train
  ```

## Acknowledgements
- [DECA](https://github.com/yadiraF/DECA/) - Used their coarse reconstruction part for reconstructing 3d face.
