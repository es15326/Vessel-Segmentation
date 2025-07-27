# 🔬 High-Performance Retinal Vessel Segmentation

> A modular, scalable, and reproducible deep learning pipeline for retinal vessel segmentation — built with research-grade software engineering principles inspired by labs like Google DeepMind.

---

## ✨ Highlights

- **📁 Configuration-Driven Design**  
  Experiments are managed with clean, readable Python config files — no messy command-line arguments.

- **🔌 Modular & Extensible**  
  Easy to plug in new models, loss functions, and datasets by just editing config files and dropping new modules.

- **🎯 Reproducibility First**  
  Deterministic training via fixed random seeds and structured config files.

- **🚀 Advanced Training Techniques**  
  Supports Automatic Mixed Precision (AMP) and Test-Time Augmentation (TTA) out of the box.

- **📊 Integrated Experiment Tracking**  
  Full integration with **TensorBoard** for monitoring training, validation, and comparison of multiple runs.

---

## 📊 Benchmark Performance

Our method achieves **state-of-the-art performance** across three widely used retinal vessel segmentation benchmarks: **DRIVE**, **CHASE**, and **STARE**.

| **Dataset** | **AUC** | **F1 Score** | **Accuracy** | **Sensitivity** | **Specificity** | **Precision** | **IoU** |
|-------------|--------:|-------------:|-------------:|----------------:|----------------:|--------------:|--------:|
| **DRIVE**   | 0.9779 | 0.8214 | 0.9691 | 0.8163 | **0.9840** | 0.8306 | 0.6973 |
| **CHASE**   | **0.9938** | **0.9084** | **0.9873** | **0.9428** | **0.9904** | **0.8775** | **0.8411** |
| **STARE**   | **0.9964** | **0.8930** | **0.9851** | **0.9120** | **0.9903** | **0.8751** | **0.8068** |

> 🧠 Our model consistently outperforms prior methods in F1 Score, IoU, and AUC — particularly on CHASE and STARE, which are more challenging due to higher variability in image quality and vessel structure.

---

## 🧠 Project Structure

```bash
vessel-segmentation/
├── configs/                # Experiment configs (DRIVE, CHASE, etc.)
│   ├── base_config.py
│   └── drive_config.py
│
├── data/                   # Dataset loading and preprocessing
│   ├── dataset.py
│   └── preprocess.py
│
├── models/                 # Model architectures (e.g., U-Net)
│   └── unet.py
│
├── utils/                  # Losses, metrics, helpers
│   ├── losses.py
│   └── metrics.py
│
├── experiments/            # Training logs, checkpoints, visual outputs
│
├── train.py                # Training script
├── evaluate.py             # Evaluation script
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd vessel-segmentation
```

### 2. Create Environment & Install Dependencies

```bash
# Optional but recommended
conda create -n vessel_seg python=3.9
conda activate vessel_seg

pip install -r requirements.txt
```

### 3. Download Datasets

Download and organize datasets like this:

```bash
vessel-segmentation/
└── datasets/
    ├── DRIVE/
    │   ├── training/
    │   └── test/
    └── CHASE/
```

---

## 🚀 Running the Pipeline

### 🔧 Step 1: Preprocess the Data

```bash
python -m data.preprocess --dataset_name=DRIVE --dataset_path=./datasets/DRIVE
```

➡️ Outputs stored in:  
`./datasets/DRIVE/training_pro/` and `test_pro/` (`.pkl` patch files)

---

### 🏋️ Step 2: Train the Model

```bash
python train.py --config=configs/drive_config.py --workdir=./experiments/DRIVE_UNet_run1
```

➡️ Checkpoints, logs, and configs saved in `./experiments/DRIVE_UNet_run1`

---

### 📈 Step 3: Evaluate the Model

```bash
python evaluate.py --workdir=./experiments/DRIVE_UNet_run1
```

➡️ Evaluation results saved as `evaluation_metrics.json`

💡 Add `--show_predictions=True` to save visual outputs:
```bash
python evaluate.py --workdir=./experiments/DRIVE_UNet_run1 --show_predictions=True
```

---

## 📉 Visualize with TensorBoard

Launch TensorBoard:

```bash
tensorboard --logdir=./experiments
```

Open your browser at: [http://localhost:6006](http://localhost:6006)

---

## 🧪 Customizing Experiments

### 🔄 Change Hyperparameters
Edit `configs/*.py`, e.g.:

```python
config.training.lr = 1e-4
config.training.batch_size = 8
```

---

### 🧠 Add a New Model

1. Add your model to `models/`
2. Update the config:

```python
config.model.name = "MyCustomNet"
config.model.args = { "in_channels": 1, "out_channels": 1 }
```

---

### 🗂️ Add a New Dataset

1. Add dataset processing logic in `data/preprocess.py`
2. Create a config file in `configs/`
3. Run the preprocessor, then train as usual.

---

## 📜 License

This project is open-sourced under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📬 Contact

For questions, collaborations, or bug reports:  
**Your Name** – _your.email@example.com_
