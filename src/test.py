import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import json
import editdistance
from tqdm.auto import tqdm

# Import Class โมเดลและฟังก์ชันช่วยจากไฟล์ของเรา
from models import ResNetCRNN, ProvinceClassifier
from utils import beam_search_decode 

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CROPS_ROOT = Path("crops_all")
TEST_CSV_PATH = CROPS_ROOT / "test" / "test_unified.csv"

# 🌟 Path โมเดล (ปรับตามที่คุณขอ)
OCR_MODEL_PATH = Path("ocr_minimal/best_model.pth")
PROV_MODEL_PATH = Path("ocr_minimal/province_best.pth")
CHAR_MAP_PATH = Path("ocr_minimal/int_to_char.json")

# --- Transforms (สำหรับการ Test ไม่ต้อง Augment) ---
tf_ocr_eval = T.Compose([
    T.Resize((64, 256)), 
    T.ToTensor()
])

tf_prov_eval = T.Compose([
    T.Resize((224, 224)), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def find_image_file(filename):
    """ฟังก์ชันช่วยหาไฟล์รูปภาพในโฟลเดอร์ crops_all"""
    if not filename: return None
    # แปลง / เป็น \ สำหรับ Windows
    filename = str(filename).replace("\\", "/")
    
    candidates = [
        CROPS_ROOT / filename,
        CROPS_ROOT / Path(filename).name,
    ]
    for p in candidates:
        if p.exists(): return p
    return None

def main():
    # 1. Load Char Map (แก้ Encoding utf-8)
    if not CHAR_MAP_PATH.exists():
        print(f"Error: {CHAR_MAP_PATH} not found.")
        return

    with open(CHAR_MAP_PATH, 'r', encoding='utf-8') as f:
        int_to_char = json.load(f)
    
    # 2. Load Models
    print("Loading models...")
    
    # --- OCR Model ---
    # num_classes ต้องเท่ากับ len(int_to_char)
    ocr_model = ResNetCRNN(1, len(int_to_char), hidden_size=256, num_rnn_layers=2).to(DEVICE)
    
    if OCR_MODEL_PATH.exists():
        print(f" Loading OCR model from {OCR_MODEL_PATH}...")
        try:
            ckpt = torch.load(OCR_MODEL_PATH, map_location=DEVICE)
            # รองรับทั้งแบบที่ save state_dict โดยตรง หรือ save เป็น dict ใหญ่
            if "model_state_dict" in ckpt:
                ocr_model.load_state_dict(ckpt["model_state_dict"])
            else:
                ocr_model.load_state_dict(ckpt)
            ocr_model.eval()
        except Exception as e:
            print(f"Failed to load OCR model: {e}")
            return
    else:
        print(f"Error: OCR model not found at {OCR_MODEL_PATH}")
        return

    # --- Province Model ---
    prov_idx2prov = {} # ตัวแปรเก็บ Map เลข -> ชื่อจังหวัด
    
    if PROV_MODEL_PATH.exists():
        print(f" Loading Province model from {PROV_MODEL_PATH}...")
        try:
            ckpt = torch.load(PROV_MODEL_PATH, map_location=DEVICE)
            
            # ดึง Class Map ออกมาจาก Checkpoint
            if "class_map" in ckpt:
                prov_idx2prov = ckpt["class_map"]
                # แปลง Key จาก int เป็น str เพื่อความชัวร์ (หรือกลับกันตามการใช้งาน)
                prov_idx2prov = {int(k):v for k,v in prov_idx2prov.items()}
            else:
                print("Warning: 'class_map' not found in province checkpoint.")
                return

            # Init Model ด้วยจำนวน Class ที่ถูกต้อง
            prov_model = ProvinceClassifier(len(prov_idx2prov)).to(DEVICE)
            
            if "model_state" in ckpt:
                prov_model.load_state_dict(ckpt["model_state"])
            else:
                prov_model.load_state_dict(ckpt)
            prov_model.eval()
            
        except Exception as e:
            print(f"Failed to load Province model: {e}")
            return
    else:
        print(f"Error: Province model not found at {PROV_MODEL_PATH}")
        return

    # 3. Load Test Data
    if not TEST_CSV_PATH.exists():
        print(f"Error: Test CSV not found at {TEST_CSV_PATH}. Run preprocess.py first.")
        return
        
    test_df = pd.read_csv(TEST_CSV_PATH, dtype=str).fillna("")
    print(f"Starting Inference on {len(test_df)} images...")

    results = []
    
    # 4. Inference Loop
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            img_rel_path = row.get("image")
            img_path = find_image_file(img_rel_path)
            
            if img_path is None:
                # print(f"Image not found: {img_rel_path}")
                continue

            # --- A. OCR Prediction ---
            pred_plate = ""
            try:
                pil_gray = Image.open(img_path).convert("L")
                ts_ocr = tf_ocr_eval(pil_gray).unsqueeze(0).to(DEVICE)
                
                out_ocr = ocr_model(ts_ocr)
                log_probs = out_ocr[0].log_softmax(-1)
                
                # ใช้ Beam Search (จาก utils.py)
                pred_plate = beam_search_decode(log_probs, int_to_char, beam_width=3)
            except Exception as e:
                print(f"OCR Error on {img_path.name}: {e}")

            # --- B. Province Prediction ---
            pred_prov = ""
            # หาไฟล์รูปจังหวัด (เปลี่ยนชื่อจาก __plate เป็น __prov)
            prov_name = img_path.name.replace("__plate", "__prov")
            prov_path = img_path.parent.parent / "provs" / prov_name # คาดเดา path
            
            # ถ้าหาไม่เจอ ให้ลองหาแบบ recursive
            if not prov_path.exists():
                 prov_path = find_image_file(prov_name)

            if prov_path and prov_path.exists():
                try:
                    pil_rgb = Image.open(prov_path).convert("RGB")
                    ts_prov = tf_prov_eval(pil_rgb).unsqueeze(0).to(DEVICE)
                    
                    out_prov = prov_model(ts_prov)
                    idx = out_prov.argmax(1).item()
                    
                    # Map Index กลับเป็นชื่อจังหวัด
                    pred_prov = prov_idx2prov.get(idx, str(idx))
                except Exception as e:
                    print(f"Province Error on {prov_name}: {e}")
            
            # --- C. Calculate Metrics ---
            gt_plate = row.get("gt_plate", "")
            gt_prov = row.get("gt_province", "")
            
            cer = 0.0
            if gt_plate:
                cer = editdistance.eval(pred_plate, gt_plate) / max(1, len(gt_plate))
            
            acc = 0
            if gt_prov:
                acc = 1 if pred_prov == gt_prov else 0

            results.append({
                "image": img_path.name,
                "gt_plate": gt_plate,
                "pred_plate": pred_plate,
                "cer": cer,
                "gt_province": gt_prov,
                "pred_province": pred_prov,
                "acc": acc
            })

    # 5. Save Results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv("final_results.csv", index=False, encoding="utf-8-sig")
        
        avg_cer = res_df["cer"].mean()
        avg_acc = res_df["acc"].mean()
        
        print(f"\nDone! Saved to final_results.csv")
        print(f"Average CER: {avg_cer:.4f}")
        print(f"Province Accuracy: {avg_acc:.4%}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()