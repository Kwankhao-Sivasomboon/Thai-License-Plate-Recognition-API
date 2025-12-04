import os
from pathlib import Path
import json
from PIL import Image
import tqdm
import pandas as pd
import re

# --- CONFIG ---
SOURCE_ROOT = Path("OCR-Car-Plate_dataset") 
CROPS_DIR = Path("crops_all") 

img_extension = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def main():
    # 1. ข้ามการ Unzip (เพราะมีโฟลเดอร์ OCR-Car-Plate_dataset อยู่แล้ว)
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ {SOURCE_ROOT} กรุณาตรวจสอบว่าวางโฟลเดอร์ Dataset ไว้หน้าแรกสุด")
    
    # 2. Crop Images
    for tvt in ["train","valid","test"]:
        img_dir = SOURCE_ROOT / tvt / "Images"
        anno_dir = SOURCE_ROOT / tvt / "_annotations.coco.json"

        if not img_dir.exists():
            print(tvt, "no images")
            continue

        annotation = {}
        id_img = {}
        id_anno = {}
        id_cate = {}
        if anno_dir.exists():
            annotation = json.load(open(anno_dir,'r',encoding='utf-8'))
            id_img = {im['id']:im for im in annotation.get('images',[])}
            id_cate = {c["id"]: c["name"] for c in annotation.get("categories", [])}
            for a in annotation.get("annotations", []):
                id_anno.setdefault(a["image_id"], []).append(a)

        plate_out = CROPS_DIR / tvt / "plates"; plate_out.mkdir(exist_ok=True, parents=True)
        province_out  = CROPS_DIR / tvt / "provs";  province_out.mkdir(exist_ok=True, parents=True)
        images = [p for p in img_dir.rglob("*") if p.suffix.lower() in img_extension]
        print(f"[{tvt}] images:", len(images))

        # --- เริ่ม Loop Crop ---
        for p in tqdm.tqdm(images, desc=f'Crop {tvt}'):
            try:
                img = Image.open(p).convert('RGB')
                W,H = img.size
                
                # Logic การหา ID ภาพ
                id_samename = []
                for k_id,v_data in id_img.items():
                    filename = Path(v_data.get("file_name","")).name
                    if filename == p.name or str(v_data.get("file_name","")).endswith(p.name):
                        id_samename.append(k_id)

                img_id = id_samename[0] if id_samename else None
                plate_bbox = None; province_bbox = None 

                if img_id and img_id in id_anno:
                    for a in id_anno[img_id]:
                        cname = id_cate.get(a["category_id"], "").lower()
                        if "plate" in cname or "ทะเบียน" in cname or "license" in cname:
                            plate_bbox = a["bbox"]
                        if "prov" in cname or "จังหวัด" in cname or "province" in cname:
                            province_bbox = a["bbox"]

                # Save Crop Image (Plate)
                saved_plate = None
                if plate_bbox:
                    x,y,w,h = plate_bbox
                    left,top = int(max(0,x)),int(max(0,y))
                    right,bottom = int(min(W, x+w)), int(min(H, y+h))
                    if right-left>2 and bottom-top>2:
                        crop = img.crop((left,top,right,bottom)).convert("L")
                        out_name = f"{tvt}__{p.stem}__plate{p.suffix}"
                        crop.save(plate_out / out_name)
                        saved_plate = str((tvt + "/plates/" + out_name))

                # Save Crop Image (Province)
                saved_prov = None
                if province_bbox:
                    x,y,w,h = province_bbox
                    left,top = int(max(0,x)), int(max(0,y))
                    right,bottom = int(min(W, x+w)), int(min(H, y+h))
                    if right-left>2 and bottom-top>2:
                        crop = img.crop((left,top,right,bottom)).convert("RGB")
                        out_name = f"{tvt}__{p.stem}__prov{p.suffix}"
                        crop.save(province_out / out_name)
                        saved_prov = str((tvt + "/provs/" + out_name))

                # Fallback
                if not saved_plate:
                    cw,ch = int(W*0.45), int(H*0.14)
                    x0=(W-cw)//2; y0=int(H*0.7)
                    crop = img.crop((x0,y0,x0+cw,y0+ch)).convert("L")
                    out_name = f"{tvt}__{p.stem}__plate_fallback{p.suffix}"
                    crop.save(plate_out / out_name)
                    saved_plate = str((tvt + "/plates/" + out_name))
            except Exception as e:
                print(f"Error processing {p}: {e}")
                continue

    # 3. Match CSV
    # แก้ไข Path ให้ชี้ไปที่ SOURCE_ROOT ที่ถูกต้อง
    DATA_SETS = [
        (SOURCE_ROOT/"train"/"train_data.csv", "train_unified.csv", "train"),
        (SOURCE_ROOT/"valid"/"valid_data.csv", "val_unified.csv",   "valid"),
        (SOURCE_ROOT/"test"/"test_data.csv",   "test_unified.csv",  "test")
    ]

    print("\nGenerating CSVs...")
    for inp, out, split in DATA_SETS:
        process_mapping(inp, out, split)

# Helper functions (คงเดิม)
def extract_core_stem(fname):
    if not fname: return ""
    s = str(fname).strip()
    if "_jpg.rf." in s: return s.split("_jpg.rf.")[0]
    m = re.match(r'^(.+?)-f[0-9a-zA-Z_]*_wm_jpeg', s, flags=re.IGNORECASE)
    if m: return re.sub(r'\.(jpg|jpeg|png|bmp|tif|tiff)$','', m.group(1), flags=re.IGNORECASE)
    if ".rf." in s: return s.split(".rf.")[0].split(".")[0]
    s2 = re.sub(r'\.(jpg|jpeg|png|bmp|tif|tiff)$', '', s, flags=re.IGNORECASE)
    return re.sub(r'(_wm_jpeg|-wm|_thumb|[^0-9A-Za-zก-๙])+', '', s2)

def norm_key(s): return re.sub(r'[^0-9a-zก-๙]', '', str(s).lower())

def process_mapping(csv_path, out_name, split_folder):
    if not csv_path.exists(): 
        print(f"CSV Not found: {csv_path}")
        return

    print(f"Processing: {csv_path.name} -> {out_name}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    target_dir = CROPS_DIR / split_folder / "plates"
    plate_files = list(target_dir.rglob("*__plate*"))
    print(f"  Found {len(plate_files)} crops in {target_dir}")

    plate_by_basename = {p.name.lower():p for p in plate_files}
    plate_by_stem = {}
    for p in plate_files: plate_by_stem.setdefault(p.stem.lower(), []).append(p)
    norm_map = {norm_key(p.name): p for p in plate_files}

    cols_map = {c.lower():c for c in df.columns}
    fname_col = next((cols_map[c] for c in cols_map if any(x in c for x in ["file","image","name"])), None)
    plate_col = next((cols_map[c] for c in cols_map if any(x in c for x in ["plate","label","gt"])), None)
    prov_col  = next((cols_map[c] for c in cols_map if any(x in c for x in ["prov","จังหวัด"])), None)

    rows, missed = [], []
    for i, r in df.iterrows():
        orig = str(r[fname_col]).strip()
        gt_plate = str(r[plate_col]).strip()
        gt_prov  = str(r[prov_col]).strip()

        matched = plate_by_basename.get(orig.lower())
        if not matched:
            core = extract_core_stem(orig).lower()
            if core in plate_by_stem: matched = plate_by_stem[core][0]
            if not matched:
                nk = norm_key(core)
                matched = norm_map.get(nk)

        if matched:
            # 🌟 แก้ไข: ใช้ relative_to(CROPS_DIR) เพื่อให้ path ใน CSV สั้นและถูกต้องสำหรับ Dataset Class
            rows.append({"image": str(matched.relative_to(CROPS_DIR)), "gt_plate": gt_plate, "gt_province": gt_prov})
        else:
            missed.append(orig)

    pd.DataFrame(rows).to_csv(out_name, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(rows)} rows (Missed: {len(missed)})")

if __name__ == "__main__":
    main()