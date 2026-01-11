# Thai License Plate Recognition API

A high-performance microservice for Thai license plate detection and recognition. This system utilizes a multi-stage deep learning pipeline to extract plate numbers and province names from images.

## Features
- Plate Detection and Segmentation using YOLO11.
- Custom OCR (CRNN) specialized for Thai characters.
- Province Classification using MobileNetV2.
- Ready-to-deploy FastAPI backend.
- Dockerized for cloud deployment.

## Tech Stack
- AI/ML: PyTorch, Ultralytics (YOLO11), OpenCV.
- Backend: FastAPI, Uvicorn.
- Infrastructure: Docker, GCP Cloud Run.

## Architecture
1. Detection: Locate the license plate in the image.
![val_batch2_pred](https://github.com/user-attachments/assets/74c61b4c-eb9b-498e-bde5-82208c755cb5)

2. Segmentation: Separate the plate number and province area.
![val_batch1_pred](https://github.com/user-attachments/assets/25f84e37-ec9d-4b36-9509-ac83e88542de)

3. Recognition: OCR for characters and Classification for province.

![WorKflow-LPR](https://github.com/user-attachments/assets/8cdd4102-4634-4643-9d15-3ad4c0c2c7a0)


## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API:
   ```bash
   uvicorn src.api_server:app --reload
   ```

## Docker & GCP Deployment
```bash
gcloud projects create [PROJECT_NAME]
gcloud config set project [PROJECT_NAME]
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud artifacts repositories create [REPO_NAME] [CONFIG_REPO]
gcloud auth configure-docker asia-southeast1-docker.pkg.dev
docker build -t asia-southeast1-docker.pkg.dev/[PROJECT_ID]/[REPO_NAME]/[IMAGE_NAME] .
docker tag [OLD_TAG] [NEW_TAG]
docker push [TAG]
gcloud run deploy [IMAGE_NAME] [CONFIG_SERVER]
```
