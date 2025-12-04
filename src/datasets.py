import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as T

# --- Transforms ---
def get_ocr_transforms(is_train=True):
    if is_train:
        return T.Compose([
            T.Resize((64, 256)),
            T.RandomAffine(degrees=2, translate=(0.05, 0.05), shear=5),
            T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.3),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
            T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return T.Compose([
            T.Resize((64, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

def get_prov_transforms(is_train=True):
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            T.RandomPerspective(distortion_scale=0.2, p=0.4),
            T.ToTensor(),
            T.RandomErasing(p=0.1, scale=(0.02, 0.15)),
            norm
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            norm
        ])

# --- Datasets ---
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
            img = Image.open(img_path).convert("L")
        except:
            img = Image.new('L', (256, 64)) # Blank image fallback

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
        
        # Mapping Logic
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
        # Convert path from plate to province crop
        img_rel_prov = img_rel_plate.replace("/plates/", "/provs/").replace("__plate", "__prov")
        img_path = self.root / img_rel_prov

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))

        img = self.transform(img)
        label = self.p2i.get(row["gt_province"], 0)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self): return len(self.df)

def ocr_collate(batch):
    imgs, tg, lens, texts, names = zip(*batch)
    return torch.stack(imgs), torch.cat(tg), torch.tensor(lens, dtype=torch.long), None, texts, names