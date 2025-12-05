import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from pathlib import Path
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np

# --- 🌟 Custom Transform: Smart Resize with Padding 🌟 ---
class SmartResize:
    def __init__(self, target_size, fill=0, is_ocr=False):
        """
        target_size: (height, width) for OCR, (height, width) for Province
        is_ocr: ถ้าเป็น OCR เราจะ fix height แล้วปล่อย width
        """
        self.target_size = target_size # (H, W)
        self.fill = fill
        self.is_ocr = is_ocr

    def __call__(self, img):
        # img is PIL Image
        tgt_h, tgt_w = self.target_size
        w, h = img.size

        # 1. คำนวณ Scale เพื่อรักษาสัดส่วน
        if self.is_ocr:
            # สำหรับ OCR: ยึดความสูงเป็นหลัก (Height=64), ความกว้างปรับตาม
            scale = tgt_h / h
            new_h = tgt_h
            new_w = int(w * scale)
            # แต่ถ้า new_w เกิน tgt_w ให้ยึดความกว้างแทน
            if new_w > tgt_w:
                scale = tgt_w / w
                new_w = tgt_w
                new_h = int(h * scale)
        else:
            # สำหรับ Province (Square): ยึดด้านที่ยาวที่สุด
            scale = min(tgt_h / h, tgt_w / w)
            new_h = int(h * scale)
            new_w = int(w * scale)

        # 2. Resize ด้วย BICUBIC (ดีที่สุดสำหรับภาพเล็กไปใหญ่)
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)

        # 3. Create Background & Paste (Padding)
        # สร้างภาพพื้นหลังสีดำ (หรือสีเทาค่า 0)
        # ถ้าภาพเดิมเป็น L (Gray) พื้นหลังก็ L, ถ้า RGB พื้นหลังก็ RGB
        new_img = Image.new(img.mode, (tgt_w, tgt_h), self.fill)
        
        # คำนวณตำแหน่งวางตรงกลาง
        paste_x = (tgt_w - new_w) // 2
        paste_y = (tgt_h - new_h) // 2
        
        new_img.paste(img, (paste_x, paste_y))
        return new_img

# --- Transforms Config ---
def get_ocr_transforms(is_train=True):
    # OCR Target: Height=64, Width=256 (ตามโมเดล CRNN)
    base_transforms = [
        # ใช้ Smart Resize แทน T.Resize((64, 256)) เดิม
        SmartResize((64, 256), is_ocr=True),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ]
    
    if is_train:
        # Augmentation สำหรับ OCR
        augments = [
            T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.3),
            # ตัด ColorJitter ออกเพราะเป็น Grayscale
            T.RandomAffine(degrees=2, translate=(0.02, 0.05), shear=5, fill=0),
        ]
        return T.Compose(augments + base_transforms)
    else:
        return T.Compose(base_transforms)

def get_prov_transforms(is_train=True):
    # Province Target: 224x224 (หรือลดเหลือ 128x128 ก็ได้ถ้าภาพแตกมาก)
    # แต่ MobileNetV2 ปกติรับ 224
    
    # 🌟 FIX: รับ Grayscale แต่ทำเป็น 3 Channels (Fake RGB) ในนี้เลย
    # เพื่อแก้ปัญหา Channel Mismatch โดยไม่ต้องแก้ Dataset
    
    ops = []
    
    # 1. Smart Resize (สำคัญมากสำหรับภาพเล็ก 41x12)
    ops.append(SmartResize((224, 224), is_ocr=False))
    
    # 2. Augmentation (Train only)
    if is_train:
        ops.append(T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.1), shear=10))
        ops.append(T.RandomPerspective(distortion_scale=0.2, p=0.3))
    
    # 3. Convert to Tensor & Normalize
    ops.append(T.ToTensor())
    
    # 4. 🌟 สำคัญ: ถ้าภาพมาเป็น 1 Channel (Gray) ให้ก๊อปปี้เป็น 3 Channels
    # Lambda function นี้จะเช็คว่าถ้าเป็น 1 channel ให้ทำซ้ำ
    ops.append(T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))
    
    # Normalize (ค่ามาตรฐาน ImageNet)
    ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return T.Compose(ops)

# --- Datasets (ไม่ต้องแก้ Logic มาก แค่เรียกใช้) ---
class OCRDataset(Dataset):
    def __init__(self, df, root, char_to_int, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.cti = char_to_int
        self.transform = transform

    def encode(self, txt):
        txt = str(txt) if txt is not None else ""
        return torch.tensor([self.cti[c] for c in txt if c in self.cti], dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["image"]
        try:
            # โหลดเป็น Grayscale (L) ตามที่คุณต้องการ
            img = Image.open(img_path).convert("L")
        except:
            img = Image.new('L', (256, 64))

        if self.transform:
            img = self.transform(img)

        target = self.encode(row["gt_plate"])
        return img, target, len(target), row["gt_plate"], str(img_path)

    def __len__(self): return len(self.df)

class ProvinceDataset(Dataset):
    def __init__(self, df, root, class_map=None, training=True):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.training = training
        
        if class_map is not None:
            self.p2i = class_map
            self.i2p = {i:p for p,i in self.p2i.items()}
        else:
            self.provs = sorted(df["gt_province"].unique())
            self.p2i = {p:i for i,p in enumerate(self.provs)}
            self.i2p = {i:p for p,i in self.p2i.items()}

        self.transform = get_prov_transforms(training)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_plate = row["image"]
        # Logic แปลง path ไปหาภาพจังหวัด
        img_rel_prov = img_rel_plate.replace("/plates/", "/provs/").replace("__plate", "__prov")
        img_path = self.root / img_rel_prov

        try:
            # โหลดเป็น Grayscale (L) เหมือนเดิม แล้วให้ Transform จัดการเป็น Fake RGB
            img = Image.open(img_path).convert("L") 
        except:
            img = Image.new("L", (224, 224))

        img = self.transform(img)
        label = self.p2i.get(row["gt_province"], 0)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self): return len(self.df)

def ocr_collate(batch):
    imgs, tg, lens, texts, names = zip(*batch)
    return torch.stack(imgs), torch.cat(tg), torch.tensor(lens, dtype=torch.long), None, texts, names