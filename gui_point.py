# -*- coding: utf-8 -*-
import sys
import time
from PyQt5.QtGui import (
    QPainter,
    QPixmap,
    QKeySequence,
    QImage,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QShortcut,
)

import numpy as np
from skimage import transform, io
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from echo_sam import echo_sam_model_registry

# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
EchoSAM_CKPT_PATH = "work_dir/Echo_SAM/Echo_SAM_02100126.pth"
EchoSAM_IMG_INPUT_SIZE = 256

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def echosam_inference(echosam_model, img_embed, points_256, height, width):
    points_torch = torch.as_tensor(points_256, dtype=torch.float, device=img_embed.device)
    if len(points_torch.shape) == 2:
        points_torch = points_torch[:, None, :]  # (1, 1, 2) for points
    points = (torch.as_tensor(points_torch, dtype=torch.float32, device=img_embed.device), torch.as_tensor([[1]], dtype=torch.float32, device=img_embed.device)) # ((1, 1, 2), (1, 1))
    
    sparse_embeddings, dense_embeddings = echosam_model.prompt_encoder(
        points=points,  # use points here
        boxes=None,  # no boxes
        masks=None,
    )
    low_res_logits, _ = echosam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 16, 16)
        image_pe=echosam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 16, 16)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 16, 16)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    echosam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return echosam_seg


print("Loading EchoSAM model, a sec.")
tic = time.perf_counter()

# set up model
echosam_model = echo_sam_model_registry["vit_b"](checkpoint=EchoSAM_CKPT_PATH).to(device)
echosam_model.eval()

print(f"Done, took {time.perf_counter() - tic}")


def np2pixmap(np_img):
    height, width, channel = np_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (192, 192, 192),
    (64, 64, 64),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 127),
    (192, 0, 192),
]


class Window(QWidget):
    def __init__(self):
        super().__init__()

        # configs
        self.half_point_size = 5  # radius of bbox starting and ending points

        # app stats
        self.image_path = None
        self.color_idx = 0
        self.bg_img = None
        self.is_mouse_down = False
        self.rect = None
        self.point_size = self.half_point_size * 2
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.embedding = None
        self.prev_mask = None

        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)

        pixmap = self.load_image()

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.view)

        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Mask")

        hbox = QHBoxLayout(self)
        hbox.addWidget(load_button)
        hbox.addWidget(save_button)

        vbox.addLayout(hbox)

        self.setLayout(vbox)

        # keyboard shortcuts
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(lambda: quit())

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)

        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_mask)

    def undo(self):
        if self.prev_mask is None:
            print("No previous mask record")
            return

        self.color_idx -= 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.prev_mask.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

        self.mask_c = self.prev_mask
        self.prev_mask = None

    def load_image(self):
        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp)"
        )

        if file_path is None or len(file_path) == 0:
            print("No image path specified, plz select an image")
            exit()

        img_np = io.imread(file_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        self.img_3c = img_3c
        self.image_path = file_path
        self.get_embeddings()
        pixmap = np2pixmap(self.img_3c)

        H, W, _ = self.img_3c.shape

        self.scene = QGraphicsScene(0, 0, W, H)
        self.end_point = None
        self.rect = None
        self.bg_img = self.scene.addPixmap(pixmap)
        self.bg_img.setPos(0, 0)
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")
        self.view.setScene(self.scene)

        # event
        self.scene.mouseReleaseEvent = self.mouse_release

    def mouse_release(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = False

        # Only a single point as prompt now
        H, W, _ = self.img_3c.shape
        point_np = np.array([[x, y]])  # single point (x, y)
        point_256 = point_np / np.array([W, H]) * 256  # normalize to 256x256 size

        sam_mask = echosam_inference(echosam_model, self.embedding, point_256, H, W)

        self.prev_mask = self.mask_c.copy()
        self.mask_c[sam_mask != 0] = colors[self.color_idx % len(colors)]
        self.color_idx += 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.mask_c.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

    def save_mask(self):
        out_path = f"{self.image_path.split('.')[0]}_mask.png"
        io.imsave(out_path, self.mask_c)

    @torch.no_grad()
    def get_embeddings(self):
        print("Calculating embedding, gui may be unresponsive.")
        img_256 = transform.resize(
            self.img_3c, (256, 256), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_256 = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_256_tensor = (
            torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        # if self.embedding is None:
        with torch.no_grad():
            self.embedding = echosam_model.image_encoder(
                img_256_tensor
            )  # (1, 256, 16, 16)
        print("Done.")


app = QApplication(sys.argv)

w = Window()
w.show()

app.exec()
