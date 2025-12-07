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
def guided_filter_np(I, p, r, eps):
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    kernel = (2*r+1, 2*r+1)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, kernel)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, kernel)
    mean_Ip = cv2.boxFilter(I*p, cv2.CV_32F, kernel)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I*I, cv2.CV_32F, kernel)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, kernel)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, kernel)
    return mean_a * I + mean_b


def dark_channel_rgb(img_pil, krnl_ratio=0.01, min_atmos_light=220.0, eps=1e-3):
    img = np.array(img_pil.convert("RGB"), dtype=np.float32)
    H, W, _ = img.shape
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    dc = np.minimum(np.minimum(r, g), b)
    krnl_sz = int(max(max(H * krnl_ratio, W * krnl_ratio), 3.0))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (krnl_sz, krnl_sz))
    dc_eroded = cv2.erode(dc, kernel)
    t = 1.0 - dc_eroded / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    radius = krnl_sz * 4
    t_refined = guided_filter_np(gray, t, radius, eps)
    A = min(min_atmos_light, float(dc_eroded.max()))
    t_clip = np.maximum(t_refined, 0.05)
    t_clip = t_clip[..., None]
    J = (img - (1.0 - t_clip) * A) / t_clip
    J = np.clip(J, 0, 255).astype(np.uint8)
    return Image.fromarray(J)
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
out_path = "output_DCP.png"
img = Image.open(in_path)
dcp = transforms.Lambda(lambda im: dark_channel_rgb(im, krnl_ratio=0.04, min_atmos_light=200))
img_dcp = dcp(img)
img_dcp.save(out_path)
clahe=transforms.Lambda(lambda im: clahe_rgb(im, clip=1, tile=8))
lap=transforms.Lambda(lambda im: laplacian_rgb(im,alpha=0.3, ksize=1))
img_clahe=clahe(img)
img_lap=lap(img_clahe)
out_path = "output_lap.png"
img_lap.save(out_path)