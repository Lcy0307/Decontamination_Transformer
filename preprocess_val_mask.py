import numpy as np
import cv2
from pathlib import Path


class VC_mask_generator():
    def __init__(self):
        self.parts=8
        self.maxBrushWidth=20
        self.maxLength=80
        self.maxVertex=16
        self.maxAngle=360

    def np_free_form_mask(self,maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
        mask = np.zeros((h, w, 1), np.float32)
        numVertex = np.random.randint(maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(maxLength + 1)
            brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)

            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(int)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(int)

            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        return mask

    def generate(self, H, W):
        mask = np.zeros((H, W, 1), np.float32)
        for i in range(self.parts):
            p = self.np_free_form_mask(self.maxVertex, self.maxLength, self.maxBrushWidth, self.maxAngle, H, W)
            mask = mask + p
        mask = np.minimum(mask, 1.0)
        mask = (mask*255.).astype(np.uint8)
        mask[mask>127] = 255
        mask[mask<127] = 0
        return np.reshape(mask, (H, W, 1))


output_dir = Path("val_mask")
output_dir.mkdir(exist_ok=True)
mask_gen = VC_mask_generator()
for i in range(1000):
    mask = mask_gen.generate(256,256)
    cv2.imwrite("val_mask/binary_mask_{:03d}.png".format(i), mask)




