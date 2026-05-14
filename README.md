🧠 EEG Seizure Detection — Ensemble Machine Learning
> Real-time EEG signal analysis and automated seizure detection using a voting ensemble of machine learning classifiers.
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![ML](https://img.shields.io/badge/ML-Ensemble%20Classifier-orange?style=flat-square)
![EEG](https://img.shields.io/badge/Domain-Neuroscience%20%2F%20BCI-purple?style=flat-square)

---

## 📌 Overview
Real-time EEG monitoring and seizure detection using ensemble machine learning and signal processing techniques.
---

## ✨ Features

| Feature | Details |
|--------|--------|
| 📡 Real-time streaming | 18-channel EEG — live file playback or synthetic demo |
| 🤖 Ensemble classifier | SVM + Random Forest + Gradient Boosting + XGBoost |
| 📊 Feature extraction | 40 features: time-domain, spectral, Hjorth, entropy, LZ complexity |
| 🎛️ Filter pipeline | Butterworth, FIR (Hamming), Chebyshev I, Notch (50 Hz) |
| 📈 Band power visualization | Delta / Theta / Alpha / Beta / Gamma |
| 🖥️ Interactive UI | PyQt5 desktop app with Light/Dark theme |
| 💾 Model persistence | Save & load trained models (.pkl) |

---

## 📊 Model Performance

| Model | AUC |
|------|------|
| SVM | 0.967 |
| CNN (baseline) | 0.981 |
| Random Forest | 0.958 |
| KNN | 0.931 |

> All models significantly outperform random baseline (AUC = 0.5)

---

## 🔬 Signal Processing Pipeline

- Raw EEG  
- Bandpass Filtering  
- Feature Extraction (40 features)  
- Robust Scaling  
- PCA (95% variance)  
- Recursive Feature Elimination (RFE)  
- Voting Ensemble Classifier  
- Seizure Probability Estimation  
- Final Decision (threshold-based)

---
## 🖼️ Screenshots

### Real-Time Monitoring
![Real-Time Monitoring](real_time_monitoring.png)

### Classification Results
![Classification Results](classification_results_dashboard.png)
---

**🚀 Installation**
```bash
# Clone the repository
git clone https://github.com/wessamessam119/eeg-seizure-detection.git
cd eeg-seizure-detection

# Install dependencies
pip install PyQt5 pyqtgraph numpy scipy scikit-learn mne matplotlib xgboost
```
Optional (for XGBoost support)
```bash
pip install xgboost
```
---
▶️ Usage
```bash
python final.py
```

---
🛠️ Tech Stack  
`Python 3.8+` · `PyQt5` · `pyqtgraph` · `scikit-learn` · `NumPy` · `SciPy` · `MNE` · `matplotlib` · `XGBoost`
---
📄 License  
This project is licensed under the MIT License — see LICENSE for details.
---
**🙋‍♀️ Author**  
**Eng. Wessam Essam**  
https://www.linkedin.com/in/wessam-aboalwafa-7a83b7320
