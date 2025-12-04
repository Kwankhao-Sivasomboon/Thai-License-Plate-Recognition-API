import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import json
import editdistance
from sklearn.metrics import f1_score

# Import จากไฟล์ที่เราสร้าง
from models import ResNetCRNN, ProvinceClassifier
from datasets import OCRDataset, ProvinceDataset, ocr_collate, get_ocr_transforms
from utils import best_path_decode

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROPS_ROOT = Path("crops_all")
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOP = 15
# Config สำหรับ Windows: ถ้าใช้ GPU ตั้งเป็น 0 หรือ 4 ได้ แต่ถ้า error ให้แก้เป็น 0
NUM_WORKERS = 0 

def filter_existing_provinces(df, root):
    valid_rows = []
    print("Filtering dataset (keeping only existing province crops)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_rel_plate = row["image"]
        img_rel_prov = img_rel_plate.replace("/plates/", "/provs/").replace("__plate", "__prov")
        if (root / img_rel_prov).exists():
            valid_rows.append(row)
    return pd.DataFrame(valid_rows)

def train_province_model():
    print("\n--- Start Training Province Model ---")
    # 1. Prepare Data
    if not Path("train_unified.csv").exists():
        print("Error: train_unified.csv not found. Run preprocess.py first.")
        return

    train_df_raw = pd.read_csv("train_unified.csv").fillna("")
    val_df_raw = pd.read_csv("val_unified.csv").fillna("")
    
    train_df = filter_existing_provinces(train_df_raw, CROPS_ROOT)
    val_df = filter_existing_provinces(val_df_raw, CROPS_ROOT)

    print(f"Train samples: {len(train_df_raw)} -> {len(train_df)}")

    train_ds = ProvinceDataset(train_df, CROPS_ROOT, training=True)
    val_ds = ProvinceDataset(val_df, CROPS_ROOT, class_map=train_ds.p2i, training=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 2. Model Setup
    model = ProvinceClassifier(len(train_ds.p2i)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    # 3. Training Loop
    best_f1 = 0.0
    patience_counter = 0
    
    for ep in range(EPOCHS):
        model.train()
        train_ds.training = True

        loss_sum = 0.0; correct = 0; total = 0

        pbar = tqdm(train_loader, desc=f"Ep {ep+1}/{EPOCHS} [Prov]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                out = model(imgs)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.2%}")

        # Validation
        model.eval()
        val_ds.training = False
        all_preds, all_labels_val = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                    out = model(imgs)
                preds = out.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_val.extend(labels.numpy())

        val_f1 = f1_score(all_labels_val, all_preds, average='macro')
        print(f"   └── Val F1: {val_f1:.4f} (Best: {best_f1:.4f})")

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "class_map": train_ds.i2p}, "province_best.pth")
            print("       Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print("Early Stopping.")
                break

def train_ocr_model():
    print("\n--- Start Training OCR Model ---")
    
    # 🌟 แก้ไข Path ให้ตรงกับ Tree ของคุณ
    json_path = Path("ocr_minimal/int_to_char.json")
    
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        int_to_char = json.load(f)
    char_to_int = {v:k for k,v in int_to_char.items()}

    # 1. Prepare Data
    if not Path("train_unified.csv").exists():
        print("Error: train_unified.csv not found.")
        return

    train_df = pd.read_csv("train_unified.csv").fillna("")
    val_df = pd.read_csv("val_unified.csv").fillna("")
    
    train_ds = OCRDataset(train_df, CROPS_ROOT, char_to_int, transform=get_ocr_transforms(True))
    val_ds = OCRDataset(val_df, CROPS_ROOT, char_to_int, transform=get_ocr_transforms(False))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ocr_collate, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=ocr_collate, num_workers=NUM_WORKERS)

    # 2. Model
    model = ResNetCRNN(1, len(int_to_char)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # 3. Training Loop
    for ep in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"OCR Ep {ep+1}")
        for batch in pbar:
            imgs, tg, tg_lens, _, texts, names = batch
            imgs, tg = imgs.to(DEVICE), tg.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(imgs) # [B, T, C]
            log_probs = out.log_softmax(-1).permute(1, 0, 2) # [T, B, C]
            
            input_lengths = torch.full((imgs.size(0),), out.size(1), dtype=torch.long).to(DEVICE)
            loss = criterion(log_probs, tg, input_lengths, tg_lens)
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
        
        # Validation Logic (Simplified Save)
        Path("ocr_train_out").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "ocr_train_out/best_model.pth")
        print("   Model saved.")

if __name__ == "__main__":
    # เลือกได้ว่าจะรันอะไร
    # train_province_model()
    train_ocr_model()