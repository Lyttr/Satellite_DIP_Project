import cv2
import numpy as np
from PIL import Image 
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
def bilateral_rgb(img_pil, d=5, sigma_color=25, sigma_space=25):

    img = np.array(img_pil.convert("RGB"), dtype=np.uint8)

    out = cv2.bilateralFilter(
        img,
        int(d),
        float(sigma_color),
        float(sigma_space)
    )
    return Image.fromarray(out)
def unsharp_rgb(img_pil, k=0.6, sigma=1.0):
    img = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, 1.0 + float(k), blur, -float(k), 0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)