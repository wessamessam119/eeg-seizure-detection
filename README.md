# 🧠 EEG Seizure Detection — Ensemble ML v4

Real-time EEG signal analysis and seizure detection using a voting ensemble 
of machine learning classifiers.

## Features
- 📡 Real-time 18-channel EEG streaming (file or synthetic demo)
- 🤖 Ensemble ML: SVM + Random Forest + Gradient Boosting + XGBoost
- 📊 40 features: time-domain, spectral, Hjorth, entropy, complexity
- 🎛️ Filter pipeline: Butterworth, FIR, Chebyshev, Notch
- 📈 Live band power visualization (Delta/Theta/Alpha/Beta/Gamma)
- 🌗 Light/Dark theme

## Installation
```bash
pip install PyQt5 pyqtgraph numpy scipy scikit-learn mne xgboost matplotlib
python eeg_tool_v4_patched.py
```

## Tech Stack
`Python` `PyQt5` `scikit-learn` `NumPy` `SciPy` `MNE` `pyqtgraph` `matplotlib`