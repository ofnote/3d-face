from ..trainFromscratch.train import LitAutoEncoder

ckpt = LitAutoEncoder.load_from_checkpoint('savedModel/sample-mnist-epoch=37-valid_loss=370.48.ckpt')

print(type(ckpt))
print(ckpt)







# import face_alignment
# from skimage import io
# import numpy as np
# import cv2
# from scipy.spatial import ConvexHull
# from skimage import draw
# fa = face_alignment.FaceAlignment(
#     face_alignment.LandmarksType._2D, flip_input=False)

# inputImg = io.imread('./image02673.png')

# preds = fa.get_landmarks(inputImg)
# inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)

# arr = np.array(preds[0])
# outline = arr[[*range(17), *range(26, 16, -1)]]
# vertices = ConvexHull(outline).vertices
# Y, X = draw.polygon(outline[vertices, 1], outline[vertices, 0])
# cropped_img = np.zeros(inputImg.shape, dtype=np.uint8)
# cropped_img[Y, X] = (255, 255, 255)
# cropped_img = cropped_img/255
# cv2.imshow("image", cropped_img)
# cv2.imshow("orig", inputImg)
# cv2.waitKey(0)
# print(Y, X)
# print(cropped_img[Y, X])
