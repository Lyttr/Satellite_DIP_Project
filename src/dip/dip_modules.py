import cv2
import numpy as np
from PIL import Image 
import torch
import torch.nn as nn
from torchvision import datasets, transforms
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
def highpass_rgb(img_pil, alpha=1.0, ksize=3):
    img = np.array(img_pil.convert("RGB"), dtype=np.float32)
    ksize = int(ksize)
    kernel = -1.0 * np.ones((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, ksize // 2] = (ksize * ksize) - 1.0 
    high = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    out = img + float(alpha) * high
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
def laplacian_rgb(img_pil, alpha=0.3, ksize=3):
    img = np.array(img_pil.convert("RGB"), dtype=np.float32)
    ksize = int(ksize)
    lap = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=ksize)
    out = img - float(alpha) * lap
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
class DCT2D(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        self.N = N

        C = torch.zeros(N, N, dtype=torch.float32)
        for k in range(N):
            for n in range(N):
                alpha = (1.0 / N) ** 0.5 if k == 0 else (2.0 / N) ** 0.5
                theta = torch.tensor(torch.pi * (n + 0.5) * k / N, dtype=torch.float32)
                C[k, n] = alpha * torch.cos(theta)

        self.register_buffer("C", C)  # [N, N]

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        C = self.C
        return C @ y @ C.t()


class RGBDCTTransform:
    def __init__(self, img_size: int):
        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.ToTensor()
        self.dct2d = DCT2D(img_size)

    def __call__(self, img_pil: Image.Image):
        img_pil = img_pil.convert("RGB")
        rgb = self.to_tensor(self.resize(img_pil))  

        
        r, g, b = rgb[0], rgb[1], rgb[2]
        y = 0.299 * r + 0.587 * g + 0.114 * b    

        dct_y = self.dct2d(y)                   
        dct_y = dct_y.abs()
        dct_y = torch.log1p(dct_y)               

        dct_y = dct_y - dct_y.min()
        dct_y = dct_y / (dct_y.max() + 1e-8)
        dct = dct_y.unsqueeze(0)                 

        return rgb, dct
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