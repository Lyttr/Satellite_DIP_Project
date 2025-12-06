import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

def unsharp_rgb(img_pil, k=0.6, sigma=1.0):
    img = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, 1.0 + float(k), blur, -float(k), 0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
def laplacian_rgb(img_pil, alpha=0.3, ksize=3):
    img = np.array(img_pil.convert("RGB"), dtype=np.float32)
    ksize = int(ksize)
    lap = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=ksize)
    out = img - float(alpha) * lap
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
def clahe_rgb(img_pil, clip=2.0, tile=8):

    img = np.array(img_pil.convert("RGB"), dtype=np.uint8)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=float(clip),
                             tileGridSize=(int(tile), int(tile)))
    y = clahe.apply(y)

    ycrcb = cv2.merge([y, cr, cb])
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(out)
in_path = "example.png"
out_path = "output_unsharp.png"
img = Image.open(in_path)
tfm = transforms.Lambda(lambda im: unsharp_rgb(im, k=1, sigma=6))
img_sharp = tfm(img)
img_sharp.save(out_path)
clahe=transforms.Lambda(lambda im: clahe_rgb(im, clip=1, tile=8))
lap=transforms.Lambda(lambda im: laplacian_rgb(im,alpha=0.3, ksize=1))
img_clahe=clahe(img)
img_lap=lap(img_clahe)
out_path = "output_lap.png"
img_lap.save(out_path)