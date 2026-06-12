
# 🔬 VLM MLOps Pipeline

> **Fine-tuning PaliGemma 2 on ScienceQA with full production MLOps**

A production-grade pipeline for fine-tuning a Vision-Language Model (VLM) on domain-specific multimodal tasks, with automated evaluation gating, experiment tracking, and CI/CD deployment.

![Architecture](architecture.png)

## 🎬 Demo

![Demo GIF](demo.gif)

---

## 🏗️ What This Project Does

| Stage | What Happens |
|-------|-------------|
| **Training** | Fine-tunes PaliGemma 2 (3B) using QLoRA on ScienceQA dataset |
| **Tracking** | Every experiment logged to MLflow (LR, LoRA rank, accuracy) |
| **Evaluation** | Automated harness runs 200 questions; blocks deployment if accuracy drops >2% |
| **Serving** | FastAPI endpoint accepts image + question, returns structured JSON |
| **Deployment** | Docker multi-stage build + GitHub Actions CI/CD to Docker Hub |

---

## 🚀 Quick Start

### Run the inference server

```bash
# Pull and run the latest image
docker pull yourusername/vlm-inference:latest

docker run -p 8000:8000 \
  -v /path/to/your/merged_model:/app/model \
  yourusername/vlm-inference:latest
```

### Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@science_question.jpg" \
  -F "question=What type of cell is shown?" \
  -F "choices=Prokaryote,Eukaryote,Virus,Bacteriophage"
```

Response:
```json
{
  "answer": "B",
  "confidence": "high",
  "model_version": "paligemma2-3b-scienceqa-v1",
  "inference_time_ms": 342.5
}
```

---

## 🧪 Training Your Own Model

1. Open `notebooks/colab_training.ipynb` in [Google Colab](https://colab.research.google.com)
2. Select **Runtime → Change runtime type → A100 GPU**
3. Run all cells top to bottom
4. Merged model saves to your Google Drive automatically

---

## 📊 Experiment Tracking

All runs logged to MLflow:

| Hyperparameter | Value Tried |
|---------------|-------------|
| Learning Rate | 1e-4, 2e-4, 5e-4 |
| LoRA Rank | 8, 16, 32 |
| Batch Size | 4, 8 |

Launch the MLflow UI:
```bash
mlflow ui --backend-store-uri file:///path/to/mlruns
```

---

## 🛡️ Automated Evaluation Gate

Before any model ships, `evaluation/eval_harness.py` runs 200 benchmark questions.

- ✅ Accuracy within 2% of baseline → **deployment approved**
- ❌ Accuracy drops >2% → **deployment blocked**, pipeline exits with code 1

---

## 📁 Project Structure

```
vlm-mlops-pipeline/
├── training/          # Fine-tuning code + Dockerfile
├── inference/         # FastAPI server + Dockerfile
├── evaluation/        # Eval harness + deployment gate
├── notebooks/         # Colab training notebook
├── tests/             # Unit tests
└── .github/workflows/ # CI/CD pipeline
```

---

## 🔧 Tech Stack

- **Model**: PaliGemma 2 (3B) — Google's vision-language model
- **Fine-tuning**: QLoRA via PEFT (trains only ~0.3% of parameters)
- **Experiment tracking**: MLflow
- **Serving**: FastAPI + Uvicorn
- **Containerization**: Docker (multi-stage builds)
- **CI/CD**: GitHub Actions → Docker Hub
- **Hardware**: Google Colab Pro A100

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Base model accuracy (ScienceQA) | ~65% |
| Fine-tuned accuracy | ~73% |
| Inference latency (A100) | ~340ms |
| Trainable parameters | ~10M / 3B (0.3%) |
