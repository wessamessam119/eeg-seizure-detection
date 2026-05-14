"""
EEG Signal Analysis Tool  ─  Enhanced ML 
=====================================================

  UI OVERHAUL
  ① Light mode color palette (warm off-white background, dark ink text)
  ② Feature table dialog: reduced minimum width (320px → was 380px)
  ③ Live band power section: animated bar-graph per band (delta/theta/alpha/beta/gamma)
     updated every tick with real welch PSD values from the ring buffer

  ALL PRIOR FIXES RETAINED
  (see v3 header for full changelog)

PATCH v4.1 — BandPowerWidget.update_from_segment
─────────────────────────────────────────────────
  Previously: bar fill represented where the peak frequency sat inside the
              band range (a positional proxy, not actual power).
  Now:
    • Real Welch-PSD band power (µV²) computed for every band each tick.
    • Bar fill  = smooth_power / session_peak  (auto-scales to session max).
    • Val label = live µV² power (auto-formatted via _fmt).
    • Peak label= dominant frequency inside each band (⏶ X.X Hz).
    • Exponential smoothing (α = 0.35) reduces frame-to-frame jitter.
  visualize_band_power_spectrum() unchanged — already correct.

Dependencies:
    pip install PyQt5 pyqtgraph numpy scipy scikit-learn mne
    pip install xgboost   (optional — falls back to GB if missing)
"""

import sys, os, pickle, warnings, traceback, glob
import numpy as np
from scipy import signal as sp_signal
from scipy.io import loadmat
from scipy.stats import skew, kurtosis as sp_kurtosis

warnings.filterwarnings("ignore")

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDialog, QCheckBox, QGroupBox,
    QFrame, QMessageBox, QProgressBar, QDoubleSpinBox, QSizePolicy,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressDialog,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui  import QFont, QColor, QPalette

import pyqtgraph as pg
pg.setConfigOptions(antialias=True, useOpenGL=False)

from sklearn.svm             import SVC
from sklearn.preprocessing   import RobustScaler
from sklearn.pipeline        import Pipeline
from sklearn.decomposition   import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate)
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, roc_auc_score)
from sklearn.inspection      import permutation_importance
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     VotingClassifier)
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.base            import clone


# =============================================================================
#  CONSTANTS
# =============================================================================

CHANNEL_LABELS = [
    "FP1-F7","F7-T3","T3-T5","T5-O1",
    "FP2-F8","F8-T4","T4-T6","T6-O2",
    "FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2",
    "FZ-CZ", "CZ-PZ",
]

DEFAULT_FS       = 256
WINDOW_SEC       = 2.0
WINDOW_STEP_SEC  = 0.5
DISPLAY_SECONDS  = 10
CHANNEL_SPACING  = 150
TIMER_MS         = 40
SAMPLES_PER_TICK = 6
SEIZURE_THR      = 0.50

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0,30.0),
    "gamma": (30.0,45.0),
}

FEATURE_NAMES = [
    "mean", "std", "variance", "rms", "skewness", "kurtosis", "line_length",
    "bp_delta", "bp_theta", "bp_alpha", "bp_beta", "bp_gamma",
    "bp_total",
    "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
    "ratio_alpha_beta", "ratio_theta_alpha", "ratio_delta_total",
    "ratio_alpha_theta", "ratio_beta_delta", "ratio_gamma_beta",
    "dominant_freq", "spectral_centroid", "spectral_flatness", "sef90",
    "zero_crossing_rate",
    "shannon_entropy", "hjorth_mobility", "hjorth_complexity", "teager_energy",
    "petrosian_fd", "lziv_complexity",
    "mean_corr",
    "median_amp", "ptp_amp",
    "sef50",
    "power_asymmetry",
]
N_FEATURES = len(FEATURE_NAMES)   # 40

# ── THEME PALETTES ──────────────────────────────────────────────────────────
_LIGHT = dict(
    BG      = "#F5F2EE",
    PANEL   = "#FFFFFF",
    BORDER  = "#D8D2CA",
    ACCENT  = "#1A6FBF",
    GREEN   = "#1A8C4E",
    RED     = "#C0392B",
    YELLOW  = "#B07D12",
    TEXT    = "#1C1A18",
    SUBTEXT = "#6B6560",
    CHAN    = "#2563A8",
    EEG_BG  = "#FAFAF8",
)
_DARK = dict(
    BG      = "#12151C",
    PANEL   = "#1A1E2B",
    BORDER  = "#2C3247",
    ACCENT  = "#4FC3F7",
    GREEN   = "#00E676",
    RED     = "#FF5252",
    YELLOW  = "#FFD740",
    TEXT    = "#E8ECF4",
    SUBTEXT = "#7B8299",
    CHAN    = "#4FC3F7",
    EEG_BG  = "#0D1017",
)

class _Theme:
    """Module-level theme singleton. C_xxx attributes always return the
    current palette value, so f-strings and style sheets pick up changes
    immediately without any rebuild."""
    def __init__(self):
        self._p = dict(_LIGHT)
        self.dark = False
    def toggle(self):
        self.dark = not self.dark
        self._p = dict(_DARK if self.dark else _LIGHT)
    def __getattr__(self, name):
        try:
            return self._p[name]
        except KeyError:
            raise AttributeError(name)

_T = _Theme()   # singleton

class _ColorProxy(str):
    __slots__ = ("_key",)
    def __new__(cls, key):
        obj = super().__new__(cls, _T._p[key])
        obj._key = key
        return obj
    def __str__(self):        return _T._p[self._key]
    def __repr__(self):       return _T._p[self._key]
    def __format__(self, s):  return format(_T._p[self._key], s)
    def __add__(self, o):     return _T._p[self._key] + o
    def __radd__(self, o):    return o + _T._p[self._key]
    def current(self):        return _T._p[self._key]

C_BG      = _ColorProxy("BG")
C_PANEL   = _ColorProxy("PANEL")
C_BORDER  = _ColorProxy("BORDER")
C_ACCENT  = _ColorProxy("ACCENT")
C_GREEN   = _ColorProxy("GREEN")
C_RED     = _ColorProxy("RED")
C_YELLOW  = _ColorProxy("YELLOW")
C_TEXT    = _ColorProxy("TEXT")
C_SUBTEXT = _ColorProxy("SUBTEXT")
C_CHAN    = _ColorProxy("CHAN")
C_EEG_BG  = _ColorProxy("EEG_BG")

# Band colours — same in both themes
BAND_COLORS = {
    "delta": "#6C63FF",
    "theta": "#3AA8C1",
    "alpha": "#2ECC71",
    "beta":  "#E67E22",
    "gamma": "#E74C3C",
}

# Per-channel colours used in LIGHT MODE only (18 distinct, ink-friendly)
CHANNEL_COLORS_LIGHT = [
    "#C0392B",  # FP1-F7  crimson
    "#E67E22",  # F7-T3   orange
    "#D4AC0D",  # T3-T5   gold
    "#1E8449",  # T5-O1   forest green
    "#1A5276",  # FP2-F8  dark navy
    "#8E44AD",  # F8-T4   purple
    "#17A589",  # T4-T6   teal
    "#B03A2E",  # T6-O2   dark red
    "#1F618D",  # FP1-F3  steel blue
    "#CA6F1E",  # F3-C3   burnt orange
    "#117A65",  # C3-P3   dark teal
    "#6C3483",  # P3-O1   dark violet
    "#1A5276",  # FP2-F4  navy
    "#A93226",  # F4-C4   dark crimson
    "#1D8348",  # C4-P4   green
    "#7D6608",  # P4-O2   dark gold
    "#154360",  # FZ-CZ   deep blue
    "#4A235A",  # CZ-PZ   deep purple
]

# Font sizes
FS_BAND_LABEL  = 8
FS_BAND_VALUE  = 8
FS_FEAT_TABLE  = 11
FS_FEAT_HDR    = 11
FS_LIVE_LABEL  = 8
FS_LIVE_VALUE  = 8

_PREDICT_CHUNK = 500


# =============================================================================
#  SIGNAL PROCESSING
# =============================================================================

def _safe_filtfilt(b, a, data):
    """Apply zero-phase filter with variance preservation."""
    padlen = 3 * (max(len(a), len(b)) - 1)
    if data.shape[-1] <= padlen:
        return data
    input_var = np.var(data)
    filtered = sp_signal.filtfilt(b, a, data, axis=-1)
    output_var = np.var(filtered)
    if output_var > 1e-12:
        filtered = filtered * np.sqrt(input_var / output_var)
    return filtered


def butter_bandpass(data, lowcut, highcut, fs, order=4):
    nyq  = fs / 2
    low  = max(lowcut  / nyq, 1e-4)
    high = min(highcut / nyq, 0.9999)
    b, a = sp_signal.butter(order, [low, high], btype="band")
    return _safe_filtfilt(b, a, data)


def butter_highpass(data, cutoff, fs, order=4):
    b, a = sp_signal.butter(order, max(cutoff/(fs/2), 1e-4), btype="high")
    return _safe_filtfilt(b, a, data)


def butter_lowpass(data, cutoff, fs, order=4):
    norm = min(cutoff / (fs / 2), 0.9999)
    b, a = sp_signal.butter(order, norm, btype="low")
    return _safe_filtfilt(b, a, data)


def notch_filter(data, freq, fs, q=30.0):
    b, a = sp_signal.iirnotch(freq, q, fs)
    return _safe_filtfilt(b, a, data)


def fir_bandpass(data, lowcut, highcut, fs, numtaps=101):
    from scipy.signal import firwin
    nyq  = fs / 2.0
    low  = max(lowcut  / nyq, 1e-4)
    high = min(highcut / nyq, 0.9999)
    if low >= high:
        return data
    try:
        b = firwin(numtaps, [low, high], pass_zero=False, window="hamming")
        return _safe_filtfilt(b, [1.0], data)
    except Exception:
        return data


def chebyshev_bandpass(data, lowcut, highcut, fs, order=4, rp=1.0):
    from scipy.signal import cheby1
    nyq  = fs / 2.0
    low  = max(lowcut  / nyq, 1e-4)
    high = min(highcut / nyq, 0.9999)
    if low >= high:
        return data
    try:
        b, a = cheby1(order, rp, [low, high], btype="band")
        return _safe_filtfilt(b, a, data)
    except Exception:
        return data


def apply_filters(data, fs, do_bp=False, do_hp=False,
                  do_lp=False, do_notch=False,
                  do_fir=False, do_cheby=False):
    out = data.copy().astype(np.float64)
    if do_hp:    out = butter_highpass(out, 0.5,  fs)
    if do_lp:    out = butter_lowpass( out, 40.0, fs)
    if do_bp:    out = butter_bandpass(out, 0.5, 40.0, fs)
    if do_notch: out = notch_filter(   out, 50.0, fs)
    if do_fir:   out = fir_bandpass(   out, 0.5, 40.0, fs)
    if do_cheby: out = chebyshev_bandpass(out, 0.5, 40.0, fs)
    return out.astype(np.float32)


# =============================================================================
#  FEATURE EXTRACTION  (40 features)
# =============================================================================

def _band_power_welch(freqs, psd, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not idx.any():
        return 0.0
    return float(_trapz(psd[idx], freqs[idx]))


def _welch_safe(data, fs):
    nperseg = min(len(data), max(4, int(fs * 2)))
    try:
        return sp_signal.welch(data, fs=fs, nperseg=nperseg)
    except Exception:
        return np.array([0.0]), np.array([0.0])


def _hjorth(data):
    d1   = np.diff(data)
    d2   = np.diff(d1)
    var0 = max(np.var(data), 1e-12)
    var1 = max(np.var(d1),   1e-12)
    var2 = max(np.var(d2),   1e-12)
    mob  = float(np.sqrt(var1 / var0))
    cpx  = float(np.sqrt(var2 / var1) / (mob + 1e-12))
    return mob, cpx


def _shannon_entropy(data, n_bins=64):
    counts, _ = np.histogram(data, bins=n_bins)
    p = counts / (counts.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _teager(data):
    if len(data) < 3:
        return 0.0
    energy = np.mean(data[1:-1]**2 - data[:-2] * data[2:])
    amp = np.std(data) + 1e-12
    return float(energy / (amp ** 2))


def _petrosian_fd(data):
    n    = len(data)
    diff = np.diff(data)
    nzc  = np.sum(diff[:-1] * diff[1:] < 0)
    if nzc == 0 or n < 2:
        return 1.0
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * nzc))))


def _lziv_complexity(data):
    threshold = np.median(data)
    binary    = (data > threshold).astype(np.int8)
    s = ''.join(map(str, binary))
    n = len(s)
    if n == 0:
        return 0.0
    i, k, l, c = 0, 1, 1, 1
    while True:
        if s[i + k - 1] not in s[:l]:
            c += 1; l = i + k
            if l + 1 > n:
                break
            i, k = 0, 1
        else:
            k += 1
            if i + k > n:
                c += 1; break
    return float(c * np.log2(n + 1e-9) / n) if n > 1 else 0.0


def _spectral_edge_freq(freqs, psd, fraction=0.90):
    total = psd.sum()
    if total == 0:
        return 0.0
    cumsum = np.cumsum(psd) / total
    idx    = np.searchsorted(cumsum, fraction)
    return float(freqs[min(idx, len(freqs) - 1)])


def _spectral_centroid(freqs, psd):
    s = psd.sum()
    return float(np.sum(freqs * psd) / s) if s > 0 else 0.0


def _spectral_flatness(psd):
    psd  = psd + 1e-12
    geom = np.exp(np.mean(np.log(psd)))
    return float(geom / np.mean(psd))


def band_power(data, fs, fmin, fmax):
    freqs, psd = _welch_safe(data, fs)
    return _band_power_welch(freqs, psd, fmin, fmax)


def extract_features(segment: np.ndarray, fs: float) -> np.ndarray:
    if segment.ndim == 1:
        segment = segment.reshape(1, -1)
    if segment.ndim != 2 or segment.shape[1] < 4:
        return np.zeros(N_FEATURES, dtype=np.float32)

    ch0    = segment[0].astype(np.float64)
    all_ch = segment.astype(np.float64)
    n_ch   = all_ch.shape[0]

    freqs, psd = _welch_safe(ch0, fs)

    def _bp(flo, fhi):
        return _band_power_welch(freqs, psd, flo, fhi)

    mean_v   = float(np.mean(all_ch))
    std_v    = float(np.mean(np.std(all_ch, axis=1)))
    var_v    = float(np.mean(np.var(all_ch, axis=1)))
    rms_v    = float(np.mean(np.sqrt(np.mean(all_ch**2, axis=1))))
    skw_v    = float(np.mean([skew(all_ch[i])        for i in range(n_ch)]))
    krt_v    = float(np.mean([sp_kurtosis(all_ch[i]) for i in range(n_ch)]))

    amp_norm = float(np.std(all_ch[0])) + 1e-12
    ll_v     = float(np.mean([np.sum(np.abs(np.diff(all_ch[i]))) / amp_norm
                               for i in range(n_ch)]))
    median_v = float(np.median(np.abs(all_ch)))
    ptp_v    = float(np.mean([np.ptp(all_ch[i]) for i in range(n_ch)]))

    bp_delta = _bp(0.5,  4.0)
    bp_theta = _bp(4.0,  8.0)
    bp_alpha = _bp(8.0, 13.0)
    bp_beta  = _bp(13.0,30.0)
    bp_gamma = _bp(30.0,45.0)
    bp_total = bp_delta + bp_theta + bp_alpha + bp_beta + bp_gamma + 1e-12

    rel_delta = bp_delta / bp_total
    rel_theta = bp_theta / bp_total
    rel_alpha = bp_alpha / bp_total
    rel_beta  = bp_beta  / bp_total
    rel_gamma = bp_gamma / bp_total

    ratio_ab = bp_alpha / (bp_beta  + 1e-12)
    ratio_ta = bp_theta / (bp_alpha + 1e-12)
    ratio_dt = bp_delta / bp_total
    ratio_at = bp_alpha / (bp_theta + 1e-12)
    ratio_bd = bp_beta  / (bp_delta + 1e-12)
    ratio_gb = bp_gamma / (bp_beta  + 1e-12)

    dom_freq = float(freqs[np.argmax(psd)]) if len(psd) > 0 else 0.0
    centroid = _spectral_centroid(freqs, psd)
    flatness = _spectral_flatness(psd)
    sef90    = _spectral_edge_freq(freqs, psd, 0.90)
    sef50    = _spectral_edge_freq(freqs, psd, 0.50)

    zcr = float(np.mean([
        np.sum(np.diff(np.sign(all_ch[i] - np.mean(all_ch[i]))) != 0)
        / max(all_ch.shape[1] - 1, 1)
        for i in range(n_ch)
    ]))

    s_ent       = _shannon_entropy(ch0)
    h_mob, h_cx = _hjorth(ch0)
    t_eng       = _teager(ch0)

    pet_fd = _petrosian_fd(ch0)
    lziv   = _lziv_complexity(ch0)

    if n_ch > 1:
        corr_mat  = np.corrcoef(all_ch)
        idx_u     = np.triu_indices(n_ch, k=1)
        mean_corr = float(np.nanmean(np.abs(corr_mat[idx_u])))
    else:
        mean_corr = 1.0

    if n_ch >= 8:
        left_pwr  = np.mean(np.var(all_ch[:4],  axis=1))
        right_pwr = np.mean(np.var(all_ch[4:8], axis=1))
        power_asym = float((left_pwr - right_pwr) /
                           (left_pwr + right_pwr + 1e-12))
    else:
        power_asym = 0.0

    feat = np.array([
        mean_v, std_v, var_v, rms_v, skw_v, krt_v, ll_v,
        bp_delta, bp_theta, bp_alpha, bp_beta, bp_gamma, bp_total,
        rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma,
        ratio_ab, ratio_ta, ratio_dt,
        ratio_at, ratio_bd, ratio_gb,
        dom_freq, centroid, flatness, sef90,
        zcr,
        s_ent, h_mob, h_cx, t_eng,
        pet_fd, lziv,
        mean_corr,
        median_v, ptp_v,
        sef50,
        power_asym,
    ], dtype=np.float32)

    if len(feat) < N_FEATURES:
        feat = np.pad(feat, (0, N_FEATURES - len(feat)))
    else:
        feat = feat[:N_FEATURES]

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


# =============================================================================
#  BAND POWER VISUALIZATION  (matplotlib — unchanged from v4)
# =============================================================================

def visualize_band_power_spectrum(segment: np.ndarray, fs: float,
                                  title: str = "EEG Band Power and Peak Frequency Analysis",
                                  output_path: str = None, show: bool = True):
    """
    Generate professional bar chart with band power values (µV²) and peak
    frequencies overlaid as dot markers.

    Args:
        segment   : EEG data (1-D or 2-D; if 2-D uses first channel)
        fs        : Sampling frequency in Hz
        title     : Chart title
        output_path: Optional file path to save the chart (.png)
        show      : Whether to display the chart interactively

    Returns:
        dict with band_names, powers (µV²), peak_frequencies (Hz),
        frequency_ranges
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping visualization.")
        return None

    # Convert to 1-D
    if segment.ndim == 2:
        segment = segment[0]
    segment = np.asarray(segment, dtype=np.float64).ravel()

    # Compute PSD
    freqs, psd = _welch_safe(segment, fs)

    # Per-band: power (µV²) and peak frequency (Hz)
    band_names  = list(BANDS.keys())
    powers      = []
    peak_freqs  = []

    for band in band_names:
        flo, fhi = BANDS[band]
        idx = (freqs >= flo) & (freqs <= fhi)

        if idx.any():
            # Band power: integrate PSD over the frequency range
            power = float(_trapz(psd[idx], freqs[idx]))
            powers.append(power)

            # Peak frequency: bin with highest PSD inside the band
            peak_bin  = np.argmax(psd[idx])
            peak_freq = float(freqs[np.where(idx)[0][peak_bin]])
            peak_freqs.append(peak_freq)
        else:
            powers.append(0.0)
            peak_freqs.append((flo + fhi) / 2.0)

    # ── Professional matplotlib visualization ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    bar_color         = "#4A90E2"   # professional blue
    peak_marker_color = "#E85D75"   # contrasting red/pink
    grid_color        = "#E8E8E8"
    text_color        = "#2C2C2C"

    x_pos     = np.arange(len(band_names))
    bar_width = 0.6

    # ── Bars ─────────────────────────────────────────────────────────────
    bars = ax.bar(x_pos, powers, bar_width,
                  color=bar_color, edgecolor="#2E5C8A",
                  linewidth=2.5, alpha=0.85)

    # ── Peak-frequency dot markers ────────────────────────────────────────
    ax.scatter(x_pos, powers, s=180,
               color=peak_marker_color,
               marker='o', edgecolors='white', linewidths=2.5,
               zorder=10, label='Peak Frequency')

    # ── Peak-frequency value labels (above each dot) ──────────────────────
    offset_y = max(powers) * 0.05 if any(p > 0 for p in powers) else 1.0
    for x, power, peak_freq in zip(x_pos, powers, peak_freqs):
        ax.text(x, power + offset_y,
                f'{peak_freq:.1f} Hz',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                color=peak_marker_color, family='sans-serif')

    # ── x-axis: band name + frequency range ──────────────────────────────
    freq_ranges = [f"{BANDS[b][0]:.1f}–{BANDS[b][1]:.0f} Hz"
                   for b in band_names]
    x_labels = [f"{name.upper()}\n({freq})"
                for name, freq in zip(band_names, freq_ranges)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='500',
                       color=text_color)

    # ── y-axis: Power (µV² or dB) ─────────────────────────────────────────
    ax.set_ylabel('Power (µV² or dB)', fontsize=14, fontweight='bold',
                  color=text_color, labelpad=15)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis='y', labelsize=11, colors=text_color)

    # ── Grid & spines ─────────────────────────────────────────────────────
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1,
            color=grid_color)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color(text_color)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color(text_color)

    # ── Title & subtitle ──────────────────────────────────────────────────
    ax.set_title(title, fontsize=18, fontweight='bold',
                 color=text_color, pad=25)
    ax.text(0.5, 1.02, "Real-time Brain Activity Monitoring",
            transform=ax.transAxes,
            ha='center', fontsize=11, style='italic', color='#666666')

    # ── Legend ────────────────────────────────────────────────────────────
    bar_patch  = mpatches.Patch(color=bar_color,
                                label='Band Power (µV²)', alpha=0.85)
    peak_patch = mpatches.Patch(color=peak_marker_color,
                                label='Peak Frequency Marker')
    ax.legend(handles=[bar_patch, peak_patch],
              loc='upper right', fontsize=11,
              framealpha=0.95, edgecolor=text_color, fancybox=True)

    # ── Stats box ─────────────────────────────────────────────────────────
    total_power = sum(powers)
    mean_power  = float(np.mean(powers))
    stats_text  = (f"Total Power: {total_power:.2f} µV²\n"
                   f"Mean Band Power: {mean_power:.2f} µV²")
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9,
                      edgecolor=text_color, linewidth=1.5),
            family='monospace', color=text_color)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Band power chart saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return {
        "band_names":       band_names,
        "powers":           powers,
        "peak_frequencies": peak_freqs,
        "frequency_ranges": freq_ranges,
    }


# =============================================================================
#  ENHANCED ENSEMBLE ML MODEL
# =============================================================================

class SeizureModel:
    def __init__(self):
        self.pipeline       = None
        self.threshold      = SEIZURE_THR
        self.threshold_mode = True
        self.energy_thr     = 5e6
        self.cv_score       = None
        self.cv_auc         = None
        self.conf_matrix    = None
        self.selected_feat_idx = None

    def build(self):
        svm = CalibratedClassifierCV(
            SVC(kernel="rbf", C=10, gamma="scale",
                class_weight="balanced", probability=False),
            cv=3, method="isotonic")

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42)

        gb = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42)

        estimators = [("svm", svm), ("rf", rf), ("gb", gb)]
        weights    = [1, 2, 2]

        if _HAS_XGB:
            xgb = XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
                eval_metric="logloss", verbosity=0, random_state=42)
            estimators.append(("xgb", xgb))
            weights.append(2)

        voting = VotingClassifier(
            estimators=estimators, voting="soft", weights=weights)

        rfe_estimator = RandomForestClassifier(
            n_estimators=50, n_jobs=-1, random_state=42)
        rfe = RFE(estimator=rfe_estimator,
                  n_features_to_select=min(30, N_FEATURES), step=2)

        self.pipeline = Pipeline([
            ("scaler",   RobustScaler()),
            ("pca",      PCA(n_components=0.95, whiten=True, random_state=42)),
            ("rfe",      rfe),
            ("ensemble", voting),
        ])

    def train(self, X: np.ndarray, y: np.ndarray, val_fraction: float = 0.20):
        self.build()
        X_bal, y_bal = self._balance(X, y)

        try:
            cv_res = cross_validate(
                clone(self.pipeline), X_bal, y_bal,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring={"f1": "f1", "auc": "roc_auc"}, n_jobs=-1)
            self.cv_score = float(cv_res["test_f1"].mean())
            self.cv_auc   = float(cv_res["test_auc"].mean())
        except Exception:
            self.cv_score = None; self.cv_auc = None

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_bal, y_bal, test_size=val_fraction,
            random_state=42, stratify=y_bal)

        self.pipeline.fit(X_tr, y_tr)
        y_pred_val = self.pipeline.predict(X_val)
        self.conf_matrix = confusion_matrix(
            y_val, y_pred_val, labels=[0, 1]).tolist()

        try:
            proba_val = self.pipeline.predict_proba(X_val)[:, 1]
            self.threshold = self._best_threshold(proba_val, y_val)
        except Exception:
            self.threshold = SEIZURE_THR

        self.threshold_mode = False

    def predict_proba(self, feat: np.ndarray) -> float:
        if feat.shape[0] != N_FEATURES:
            return self._threshold_heuristic(feat)
        if self.pipeline is None:
            return self._threshold_heuristic(feat)
        try:
            p_arr = self.pipeline.predict_proba(feat.reshape(1, -1))
            cls   = list(self.pipeline.named_steps["ensemble"].classes_)
            if 1 in cls:
                return float(p_arr[0, cls.index(1)])
            return 0.0
        except Exception:
            return self._threshold_heuristic(feat)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            return np.array([self._threshold_heuristic(x) for x in X],
                            dtype=np.float32)
        try:
            cls     = list(self.pipeline.named_steps["ensemble"].classes_)
            abn_idx = cls.index(1) if 1 in cls else 1
            results = []
            for start in range(0, len(X), _PREDICT_CHUNK):
                chunk = X[start: start + _PREDICT_CHUNK]
                p = self.pipeline.predict_proba(chunk)[:, abn_idx]
                results.extend(p.tolist())
            return np.array(results, dtype=np.float32)
        except Exception:
            return np.array([self._threshold_heuristic(x) for x in X],
                            dtype=np.float32)

    def classify(self, feat: np.ndarray) -> str:
        return "Abnormal" if self.predict_proba(feat) >= self.threshold else "Normal"

    @staticmethod
    def _balance(X, y):
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X, y
        n_maj = counts.max(); n_min = counts.min()
        if n_maj == n_min:
            return X, y
        min_cls = classes[np.argmin(counts)]
        X_min   = X[y == min_cls]
        deficit = n_maj - n_min
        rng     = np.random.default_rng(42)
        idx     = rng.integers(0, n_min, deficit)
        feat_std = X_min.std(axis=0) + 1e-9
        noise    = (rng.standard_normal((deficit, X.shape[1]))
                    * feat_std * 0.05).astype(np.float32)
        X_aug = X_min[idx] + noise
        y_aug = np.full(deficit, min_cls, dtype=y.dtype)
        return np.vstack([X, X_aug]), np.concatenate([y, y_aug])

    @staticmethod
    def _best_threshold(proba: np.ndarray, y_true: np.ndarray) -> float:
        best_thr, best_f1 = 0.50, 0.0
        for thr in np.linspace(0.05, 0.95, 181):
            pred = (proba >= thr).astype(int)
            f1   = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        return best_thr

    def _threshold_heuristic(self, feat: np.ndarray) -> float:
        energy = float(np.mean(feat ** 2))
        return float(np.clip(energy / (self.energy_thr + 1e-9), 0, 1))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline":  self.pipeline,
                "threshold": self.threshold,
                "cv_score":  self.cv_score,
                "cv_auc":    self.cv_auc,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            self.pipeline  = payload.get("pipeline")
            self.threshold = payload.get("threshold", SEIZURE_THR)
            self.cv_score  = payload.get("cv_score")
            self.cv_auc    = payload.get("cv_auc")
        else:
            self.pipeline  = payload
            self.threshold = SEIZURE_THR
        self.threshold_mode = False


# =============================================================================
#  SYNTHETIC EEG
# =============================================================================

class SyntheticEEG:
    def __init__(self, n_channels=18, fs=DEFAULT_FS):
        self.n_channels = n_channels; self.fs = fs
        self._t = 0.0; self._sz_start = -9999.0; self._sz_dur = 0.0
        self._rng = np.random.default_rng(42)

    def next_samples(self, n: int) -> np.ndarray:
        t = self._t + np.arange(n) / self.fs
        self._t += n / self.fs
        if self._t > self._sz_start + self._sz_dur + 8.0:
            if self._rng.random() < 0.003:
                self._sz_start = self._t
                self._sz_dur   = self._rng.uniform(3, 8)
        in_sz = (t >= self._sz_start) & (t <= self._sz_start + self._sz_dur)
        out   = np.zeros((self.n_channels, n), dtype=np.float32)
        for ch in range(self.n_channels):
            ph = ch * 0.4
            bg = (30 * np.sin(2*np.pi*10*t + ph)
                  + 15 * np.sin(2*np.pi* 3*t + ph)
                  + self._rng.standard_normal(n) * 20)
            sz = np.where(in_sz,
                          200 * np.sin(2*np.pi*5*t + ph)
                          + self._rng.standard_normal(n) * 50, 0)
            out[ch] = (bg + sz).astype(np.float32)
        return out


# =============================================================================
#  FILTER DIALOG
# =============================================================================

class FilterDialog(QDialog):
    filters_applied = pyqtSignal(bool, bool, bool, bool, bool, bool)

    def __init__(self, parent=None, init_bp=False, init_hp=False,
                 init_lp=False, init_notch=False,
                 init_fir=False, init_cheby=False):
        super().__init__(parent)
        self.setWindowTitle("Filter Settings")
        self.setModal(True); self.setFixedWidth(360)
        lay = QVBoxLayout(self)
        lay.setSpacing(10); lay.setContentsMargins(18, 18, 18, 18)

        title = QLabel("Filter Configuration")
        title.setFont(QFont("Courier New", 12, QFont.Bold))
        title.setStyleSheet(f"color:{C_ACCENT};"); lay.addWidget(title)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{C_BORDER};"); lay.addWidget(sep)

        note = QLabel("IIR filters: Butterworth order 4  |  FIR: Hamming 101-tap\n"
                      "Chebyshev I: order 4, 1 dB ripple  |  Applied sequentially")
        note.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px;"); lay.addWidget(note)

        box  = QGroupBox("Active Filters")
        box.setStyleSheet(f"""
            QGroupBox {{ color:{C_TEXT}; border:1px solid {C_BORDER};
                border-radius:4px; margin-top:8px; padding-top:8px; font-size:12px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:8px; }}
        """)
        blay = QVBoxLayout(box); blay.setSpacing(8)

        self.cb_bp    = QCheckBox("Butterworth Bandpass  (0.5 – 40 Hz)"); self.cb_bp.setChecked(init_bp)
        self.cb_hp    = QCheckBox("Butterworth High-pass (0.5 Hz)");       self.cb_hp.setChecked(init_hp)
        self.cb_lp    = QCheckBox("Butterworth Low-pass  (40 Hz)");        self.cb_lp.setChecked(init_lp)
        self.cb_notch = QCheckBox("Notch Filter          (50 Hz)");        self.cb_notch.setChecked(init_notch)

        sep_iir = QFrame(); sep_iir.setFrameShape(QFrame.HLine)
        sep_iir.setStyleSheet(f"color:{C_BORDER};")

        self.cb_fir   = QCheckBox("FIR Bandpass          (0.5 – 40 Hz, 101-tap)"); self.cb_fir.setChecked(init_fir)
        self.cb_cheby = QCheckBox("Chebyshev I Bandpass  (0.5 – 40 Hz, 1 dB)");   self.cb_cheby.setChecked(init_cheby)

        sep_extra = QFrame(); sep_extra.setFrameShape(QFrame.HLine)
        sep_extra.setStyleSheet(f"color:{C_BORDER};")

        self.cb_all   = QCheckBox("Select All")

        for w in [self.cb_bp, self.cb_hp, self.cb_lp, self.cb_notch,
                  sep_iir, self.cb_fir, self.cb_cheby, sep_extra, self.cb_all]:
            blay.addWidget(w)
        lay.addWidget(box)

        self._all_cbs = [self.cb_bp, self.cb_hp, self.cb_lp,
                         self.cb_notch, self.cb_fir, self.cb_cheby]
        self.cb_all.stateChanged.connect(self._toggle_all)
        for cb in self._all_cbs:
            cb.stateChanged.connect(self._sync_all)
        self._sync_all()

        btn_apply = QPushButton("Apply")
        btn_apply.setFixedHeight(36); btn_apply.setCursor(Qt.PointingHandCursor)
        btn_apply.clicked.connect(self._emit)
        btn_apply.setStyleSheet(f"""
            QPushButton {{ background:{C_ACCENT}; color:#FFFFFF;
                border:none; border-radius:4px; font-weight:bold; font-size:13px; }}
            QPushButton:hover {{ background:#2980D9; }}
        """)
        lay.addWidget(btn_apply)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedHeight(28); btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        btn_cancel.setStyleSheet(f"""
            QPushButton {{ background:transparent; color:{C_SUBTEXT};
                border:1px solid {C_BORDER}; border-radius:4px; }}
            QPushButton:hover {{ color:{C_TEXT}; }}
        """)
        lay.addWidget(btn_cancel)
        self.setStyleSheet(f"""
            QDialog {{ background:{C_PANEL}; color:{C_TEXT}; }}
            QCheckBox {{ color:{C_TEXT}; font-size:12px; spacing:8px; }}
            QCheckBox::indicator {{ width:16px; height:16px;
                border:1px solid {C_BORDER}; border-radius:3px; background:{C_BG}; }}
            QCheckBox::indicator:checked {{
                background:{C_ACCENT}; border-color:{C_ACCENT}; }}
        """)

    def _toggle_all(self, state):
        checked = (state == Qt.Checked)
        for cb in self._all_cbs:
            cb.blockSignals(True); cb.setChecked(checked); cb.blockSignals(False)

    def _sync_all(self):
        all_on = all(c.isChecked() for c in self._all_cbs)
        self.cb_all.blockSignals(True); self.cb_all.setChecked(all_on)
        self.cb_all.blockSignals(False)

    def _emit(self):
        self.filters_applied.emit(
            self.cb_bp.isChecked(),    self.cb_hp.isChecked(),
            self.cb_lp.isChecked(),    self.cb_notch.isChecked(),
            self.cb_fir.isChecked(),   self.cb_cheby.isChecked())
        self.accept()


# =============================================================================
#  FEATURE TABLE DIALOG
# =============================================================================

class FeatureTableDialog(QDialog):
    def __init__(self, feat_before: np.ndarray,
                 feat_after: np.ndarray = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Extracted Features ({N_FEATURES})")
        self.setMinimumSize(320, 460)
        self.resize(360, 520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(6)

        title = QLabel(f"Feature Results  ({N_FEATURES})")
        title.setFont(QFont("Courier New", 12, QFont.Bold))
        title.setStyleSheet(f"color:{C_ACCENT};"); lay.addWidget(title)

        sub = QLabel(f"Window: {WINDOW_SEC}s  |  ch0 spectral + all-ch time-domain")
        sub.setStyleSheet(f"color:{C_SUBTEXT}; font-size:{FS_FEAT_TABLE - 1}px;")
        lay.addWidget(sub)

        has_after = feat_after is not None and len(feat_after) == N_FEATURES
        cols = ["Feature", "Before", "After"] if has_after else ["Feature", "Value"]

        table = QTableWidget(N_FEATURES, len(cols))
        table.setHorizontalHeaderLabels(cols)
        hdr_font = QFont("Courier New", FS_FEAT_HDR, QFont.Bold)
        table.horizontalHeader().setFont(hdr_font)
        table.horizontalHeader().setMinimumSectionSize(55)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(cols)):
            table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        table.verticalHeader().setDefaultSectionSize(20)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(True)

        row_font = QFont("Courier New", FS_FEAT_TABLE)
        table.setFont(row_font)

        feat_units = {
            "mean": "µV", "std": "µV", "variance": "µV²", "rms": "µV",
            "skewness": "", "kurtosis": "", "line_length": "",
            "bp_delta": "µV²", "bp_theta": "µV²", "bp_alpha": "µV²",
            "bp_beta": "µV²", "bp_gamma": "µV²", "bp_total": "µV²",
            "rel_delta": "%", "rel_theta": "%", "rel_alpha": "%",
            "rel_beta": "%", "rel_gamma": "%",
            "ratio_alpha_beta": "", "ratio_theta_alpha": "",
            "ratio_delta_total": "", "ratio_alpha_theta": "",
            "ratio_beta_delta": "", "ratio_gamma_beta": "",
            "dominant_freq": "Hz", "spectral_centroid": "Hz",
            "spectral_flatness": "", "sef90": "Hz",
            "zero_crossing_rate": "",
            "shannon_entropy": "", "hjorth_mobility": "",
            "hjorth_complexity": "", "teager_energy": "µV²",
            "petrosian_fd": "", "lziv_complexity": "",
            "mean_corr": "", "median_amp": "µV", "ptp_amp": "µV",
            "sef50": "Hz", "power_asymmetry": "",
        }

        group_ranges = [
            (range(0,  7),  "#1A3A5C",  "Time Domain"),
            (range(7,  12), "#185A8C",  "Abs Band Power"),
            (range(12, 13), "#1A6FBF",  "Total Power"),
            (range(13, 18), "#1A7A4A",  "Rel Band Power"),
            (range(18, 24), "#7D5A00",  "Spectral Ratios"),
            (range(24, 29), "#1A6FBF",  "Spectral Shape"),
            (range(29, 33), "#8B2252",  "Nonlinear/Hjorth"),
            (range(33, 35), "#6B1E8C",  "Complexity"),
            (range(35, 40), "#2E5E6E",  "Cross-ch/Asym"),
        ]
        color_map = {}
        for grng, col, _ in group_ranges:
            for i in grng:
                color_map[i] = col

        for row in range(N_FEATURES):
            col_hex = color_map.get(row, C_TEXT)
            fname   = FEATURE_NAMES[row]
            unit    = feat_units.get(fname, "")
            feat_label = f"{fname} ({unit})" if unit else fname
            name_item = QTableWidgetItem(feat_label)
            name_item.setForeground(QColor(col_hex))
            name_item.setFont(row_font)
            table.setItem(row, 0, name_item)

            val_str  = f"{feat_before[row]:.6g}" if row < len(feat_before) else "—"
            val_item = QTableWidgetItem(val_str)
            val_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_item.setFont(row_font)
            table.setItem(row, 1, val_item)

            if has_after:
                val2_str  = f"{feat_after[row]:.6g}" if row < len(feat_after) else "—"
                val2_item = QTableWidgetItem(val2_str)
                val2_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                val2_item.setFont(row_font)
                diff = abs(float(feat_after[row]) - float(feat_before[row]))
                if diff > 1e-6:
                    val2_item.setForeground(QColor(str(C_GREEN)))
                table.setItem(row, 2, val2_item)

        lay.addWidget(table)

        legend_lay = QHBoxLayout()
        leg_font = QFont("Courier New", 8)
        for _, col, label in group_ranges:
            dot = QLabel(f"● {label}"); dot.setFont(leg_font)
            dot.setStyleSheet(f"color:{col};")
            legend_lay.addWidget(dot)
        legend_lay.addStretch(); lay.addLayout(legend_lay)

        close_btn = QPushButton("Close")
        close_btn.setFont(QFont("Courier New", 11))
        close_btn.setFixedHeight(30)
        close_btn.clicked.connect(self.accept)
        lay.addWidget(close_btn)

        self.setStyleSheet(f"""
            QDialog {{ background:{C_PANEL}; color:{C_TEXT}; }}
            QTableWidget {{ background:{C_BG}; color:{C_TEXT};
                gridline-color:{C_BORDER}; border:1px solid {C_BORDER}; }}
            QHeaderView::section {{ background:{C_BORDER}; color:{C_ACCENT};
                font-size:{FS_FEAT_HDR}px; font-weight:bold;
                padding:4px; border:none; }}
            QTableWidget::item {{ padding:3px 6px; }}
            QTableWidget {{ alternate-background-color:#EAE6E0; }}
            QPushButton {{ background:{C_BG}; color:{C_TEXT};
                border:1px solid {C_BORDER}; border-radius:3px; padding:3px 14px; }}
            QPushButton:hover {{ background:{C_ACCENT}; color:#FFFFFF; }}
        """)


# =============================================================================
#  EEG PLOT WIDGET
# =============================================================================

class EEGPlotWidget(QWidget):
    def __init__(self, n_channels=18, fs=DEFAULT_FS, parent=None):
        super().__init__(parent)
        self.n_channels = n_channels; self.fs = fs
        self.gain = 1.0; self.disp_secs = DISPLAY_SECONDS
        self.ch_spacing = CHANNEL_SPACING
        self._n_disp  = int(self.disp_secs * self.fs)
        self._buf     = np.zeros((n_channels, self._n_disp), dtype=np.float32)
        self._sz_mask = np.zeros(self._n_disp, dtype=bool)
        self._ptr     = 0
        self._x       = np.linspace(0.0, float(self.disp_secs), self._n_disp)
        self._curves  = []; self._sz_curves = []; self._labels = []; self._plot = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setBackground(C_EEG_BG)
        self._gw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._gw)
        self._build()

    def _build(self):
        self._gw.clear(); self._curves = []; self._sz_curves = []; self._labels = []
        self._plot = self._gw.addPlot(row=0, col=0)
        self._plot.setMenuEnabled(False); self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideButtons(); self._plot.showGrid(x=False, y=False)
        self._plot.setXRange(0.0, float(self.disp_secs), padding=0)

        sp = self.ch_spacing
        self._plot.setYRange(-sp * 1.2, self.n_channels * sp, padding=0)
        la = self._plot.getAxis("left")
        la.setStyle(tickLength=0, showValues=False)
        la.setPen(pg.mkPen(str(C_BORDER)))
        ba = self._plot.getAxis("bottom")
        ba.setPen(pg.mkPen(str(C_BORDER))); ba.setLabel("Time (s)")
        for ax in ("top", "right"):
            self._plot.showAxis(ax, False)

        for i in range(self.n_channels):
            offset = float((self.n_channels - 1 - i) * sp)
            if not _T.dark:
                ch_col = CHANNEL_COLORS_LIGHT[i % len(CHANNEL_COLORS_LIGHT)]
            else:
                ch_col = str(C_CHAN)
            c = self._plot.plot(
                self._x, np.full(self._n_disp, offset, dtype=np.float32),
                pen=pg.mkPen(ch_col, width=1))
            self._curves.append(c)
            sz = self._plot.plot(
                self._x, np.full(self._n_disp, np.nan, dtype=np.float32),
                pen=pg.mkPen(str(C_RED), width=2))
            self._sz_curves.append(sz)
            txt = CHANNEL_LABELS[i] if i < len(CHANNEL_LABELS) else f"CH{i+1}"
            lbl = pg.TextItem(text=txt, color=ch_col, anchor=(0.0, 0.5))
            lbl.setFont(QFont("Courier New", 7))
            lbl.setPos(0.0, offset)
            self._plot.addItem(lbl); self._labels.append(lbl)

        for i in range(self.n_channels + 1):
            y = i * sp - sp / 2
            self._plot.addItem(pg.InfiniteLine(
                pos=y, angle=0, pen=pg.mkPen(str(C_BORDER), width=1)))

    def push_samples(self, samples: np.ndarray, seizure_flags=None):
        if samples is None or samples.size == 0:
            return
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        n_new = samples.shape[1]
        if n_new > self._n_disp:
            samples = samples[:, :self._n_disp]; n_new = self._n_disp
        amp = float(np.max(np.abs(samples)))
        if 0.0 < amp < 0.005:
            samples = samples.astype(np.float64) * 1e6
        samples = samples.astype(np.float32)
        if seizure_flags is None:
            seizure_flags = np.zeros(n_new, dtype=bool)
        elif len(seizure_flags) != n_new:
            seizure_flags = np.pad(seizure_flags, (0, max(0, n_new - len(seizure_flags))))
        n_ch = min(samples.shape[0], self.n_channels)
        for k in range(n_new):
            col = self._ptr % self._n_disp
            self._buf[:n_ch, col] = samples[:n_ch, k] * self.gain
            self._sz_mask[col]    = bool(seizure_flags[k] if k < len(seizure_flags) else False)
            self._ptr += 1

    def refresh(self):
        if self._ptr == 0 or not self._curves:
            return
        ptr  = self._ptr % self._n_disp
        tail = self._n_disp - ptr
        order = np.empty(self._n_disp, dtype=np.int32)
        order[:tail] = np.arange(ptr, self._n_disp)
        order[tail:] = np.arange(0,   ptr)
        sz_ord = self._sz_mask[order]
        sp = self.ch_spacing
        for i in range(self.n_channels):
            offset = float((self.n_channels - 1 - i) * sp)
            y      = self._buf[i, order] + offset
            self._curves[i].setData(self._x, y, skipFiniteCheck=True)
            sz_y = np.where(sz_ord, y, np.nan).astype(np.float32)
            self._sz_curves[i].setData(self._x, sz_y, skipFiniteCheck=True)

    def set_gain(self, gain: float):
        self.gain = max(0.01, float(gain))

    def set_amplitude_scale(self, data: np.ndarray):
        try:
            n = min(self.n_channels, data.shape[0])
            stds = np.std(data[:n], axis=1)
            med_std = float(np.median(stds[stds > 0])) if np.any(stds > 0) else 0
            self.ch_spacing = max(20.0, med_std * 3.0) if med_std > 0 else CHANNEL_SPACING
        except Exception:
            self.ch_spacing = CHANNEL_SPACING
        self._build()

    def reset(self, n_channels: int, fs: float):
        self.n_channels = n_channels; self.fs = fs
        self._n_disp  = int(self.disp_secs * self.fs)
        self._buf     = np.zeros((n_channels, self._n_disp), dtype=np.float32)
        self._sz_mask = np.zeros(self._n_disp, dtype=bool)
        self._ptr     = 0
        self._x       = np.linspace(0.0, float(self.disp_secs), self._n_disp)
        self.ch_spacing = CHANNEL_SPACING
        self._build()


# =============================================================================
#  SPECTRAL WIDGET
# =============================================================================

class SpectralWidget(QWidget):
    def __init__(self, fs=DEFAULT_FS, parent=None):
        super().__init__(parent)
        self.fs    = fs
        self._data = None
        self._mode = "fft"

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(4)

        bar = QHBoxLayout()
        lbl = QLabel("Spectral View")
        lbl.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px;")
        bar.addWidget(lbl); bar.addStretch()

        self.btn_fft  = QPushButton("FFT")
        self.btn_spec = QPushButton("Spectrogram")
        for btn in (self.btn_fft, self.btn_spec):
            btn.setCheckable(True); btn.setFixedHeight(22)
            btn.setStyleSheet(f"""
                QPushButton {{ background:{C_BG}; color:{C_TEXT};
                    border:1px solid {C_BORDER}; border-radius:3px;
                    padding:2px 10px; font-size:11px; }}
                QPushButton:checked {{ background:{C_ACCENT}; color:#FFFFFF;
                    font-weight:bold; border-color:{C_ACCENT}; }}
            """)
            bar.addWidget(btn)
        self.btn_fft.setChecked(True)
        lay.addLayout(bar)

        self.gw = pg.GraphicsLayoutWidget()
        self.gw.setBackground(C_EEG_BG)
        lay.addWidget(self.gw)

        self.plot = self.gw.addPlot()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.getAxis("bottom").setPen(pg.mkPen(str(C_BORDER)))
        self.plot.getAxis("left").setPen(pg.mkPen(str(C_BORDER)))
        for ax in ("top", "right"):
            self.plot.showAxis(ax, False)

        self._fft_curve = self.plot.plot(pen=pg.mkPen(str(C_ACCENT), width=1.5))

        self._img_item = pg.ImageItem()
        self._img_item.setVisible(False)
        self.plot.addItem(self._img_item)

        self._cmap = None
        for name in ("viridis", "plasma", "inferno", "CET-L9"):
            try:
                self._cmap = pg.colormap.get(name, source="matplotlib", skipCache=True)
                break
            except Exception:
                pass
        if self._cmap is None:
            try:
                self._cmap = pg.colormap.get("viridis")
            except Exception:
                pass

        self._set_fft_labels()
        self.btn_fft.clicked.connect(lambda: self._set_mode("fft"))
        self.btn_spec.clicked.connect(lambda: self._set_mode("spec"))

    def _set_fft_labels(self):
        self.plot.setLabel("bottom", "Frequency (Hz)")
        self.plot.setLabel("left",   "PSD (µV²/Hz)")
        self.plot.setXRange(0, 60, padding=0)

    def _set_spec_labels(self):
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.setLabel("left",   "Frequency (Hz)")

    def _set_mode(self, mode: str):
        self._mode = mode
        self.btn_fft.setChecked(mode == "fft")
        self.btn_spec.setChecked(mode == "spec")
        self._refresh()

    def update_data(self, ch_data: np.ndarray, _time_arr=None):
        self._data = np.asarray(ch_data, dtype=np.float64).ravel()
        self._refresh()

    def _refresh(self):
        if self._data is None or len(self._data) < 8:
            return
        d = self._data.copy()
        if 0.0 < float(np.max(np.abs(d))) < 0.005:
            d *= 1e6

        if self._mode == "fft":
            self._img_item.setVisible(False)
            self._fft_curve.setVisible(True)
            freqs, psd = _welch_safe(d, self.fs)
            mask = freqs <= 60.0
            self._fft_curve.setData(freqs[mask], psd[mask])
            self._set_fft_labels()
            mx = float(np.max(psd[mask])) if mask.any() else 1.0
            self.plot.setYRange(0, mx * 1.1 + 1e-9, padding=0)
        else:
            self._fft_curve.setVisible(False)
            self._fft_curve.setData([], [])
            nperseg  = max(4, min(int(self.fs * 2.0), len(d)))
            noverlap = min(int(nperseg * 0.75), nperseg - 1)
            try:
                f, t, Sxx = sp_signal.spectrogram(
                    d, fs=self.fs, nperseg=nperseg, noverlap=noverlap,
                    window="hann", scaling="density", mode="psd")
            except Exception:
                return
            Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
            try:
                from scipy.ndimage import gaussian_filter
                Sxx_db = gaussian_filter(Sxx_db, sigma=0.8)
            except Exception:
                pass
            fmask    = f <= 40.0
            f_disp   = f[fmask]
            Sxx_disp = Sxx_db[fmask, :]
            if Sxx_disp.size == 0 or len(t) < 2 or len(f_disp) < 2:
                return
            img_data = Sxx_disp.T.astype(np.float32)
            vmin = float(np.percentile(img_data, 2))
            vmax = float(np.percentile(img_data, 98))
            if vmax <= vmin:
                vmax = vmin + 1.0
            self._img_item.setImage(img_data, autoLevels=False, levels=(vmin, vmax))
            if self._cmap is not None:
                self._img_item.setColorMap(self._cmap)
            n_time, n_freq = img_data.shape
            dt = (float(t[-1]) - float(t[0])) / max(n_time - 1, 1)
            df = (float(f_disp[-1]) - float(f_disp[0])) / max(n_freq - 1, 1)
            tr = pg.QtGui.QTransform()
            tr.translate(float(t[0]), float(f_disp[0]))
            tr.scale(dt, df)
            self._img_item.setTransform(tr)
            self._img_item.setVisible(True)
            self._set_spec_labels()
            self.plot.setXRange(float(t[0]) - dt * 0.5,
                                float(t[-1]) + dt * 0.5, padding=0)
            self.plot.setYRange(float(f_disp[0]) - df * 0.5,
                                float(f_disp[-1]) + df * 0.5, padding=0)


# =============================================================================
#  LIVE BAND POWER WIDGET  — horizontal bars, real µV² power + peak Hz
# =============================================================================

class BandPowerWidget(QWidget):
    """
    Five horizontal progress-bar rows — one per EEG band.

    Each row shows:
      • Band name label (left, fixed 40 px)
      • QProgressBar  (middle, auto-scaled to session-peak power per band)
      • Live µV² value (right, formatted via _fmt)
      • ⏶ Peak-frequency label in Hz (far right)

    Session-peak power is tracked independently per band so that quiet bands
    are not permanently flattened by a louder neighbour.
    """

    _BANDS  = ["delta", "theta", "alpha", "beta", "gamma"]
    _RANGES = {
        "delta": (0.5,  4.0),
        "theta": (4.0,  8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        # Per-band smoothed power and session-peak power trackers
        self._smooth = {b: 0.0 for b in self._BANDS}
        self._peak   = {b: 0.0 for b in self._BANDS}

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(2)

        self._bars      = {}   # band → QProgressBar
        self._val_lbls  = {}   # band → QLabel  (live µV²)
        self._peak_lbls = {}   # band → QLabel  (⏶ peak Hz)
        self._name_lbls = []   # all name QLabels (for theme repaint)

        bar_ss = (
            "QProgressBar {"
            f"  border:1px solid {C_BORDER};"
            "  border-radius:3px;"
            f"  background:{C_BG};"
            "  height:12px;"
            "}"
            "QProgressBar::chunk {"
            "  background:#0066CC;"
            "  border-radius:2px;"
            "}"
        )

        for band in self._BANDS:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            # Band name
            name_lbl = QLabel(band.upper())
            name_lbl.setFont(QFont("Courier New", 8, QFont.Bold))
            name_lbl.setFixedWidth(40)
            name_lbl.setStyleSheet(f"color:{C_TEXT};")
            row.addWidget(name_lbl)
            self._name_lbls.append(name_lbl)

            # Progress bar (auto-scaled to session peak)
            pb = QProgressBar()
            pb.setRange(0, 10000)
            pb.setValue(0)
            pb.setFixedHeight(14)
            pb.setTextVisible(False)
            pb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            pb.setStyleSheet(bar_ss)
            row.addWidget(pb, stretch=1)

            # Current value label (µV²)
            val_lbl = QLabel("—")
            val_lbl.setFont(QFont("Courier New", 8))
            val_lbl.setFixedWidth(62)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet(f"color:{C_TEXT};")
            row.addWidget(val_lbl)

            # Peak-frequency label (Hz)
            peak_lbl = QLabel("")
            peak_lbl.setFont(QFont("Courier New", 7))
            peak_lbl.setFixedWidth(58)
            peak_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            peak_lbl.setStyleSheet(f"color:{C_SUBTEXT};")
            row.addWidget(peak_lbl)

            main_lay.addLayout(row)
            self._bars[band]      = pb
            self._val_lbls[band]  = val_lbl
            self._peak_lbls[band] = peak_lbl

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(v: float) -> str:
        """Auto-format a µV² value to a compact string."""
        if v >= 1e6:  return f"{v/1e6:.2f}M"
        if v >= 1e3:  return f"{v/1e3:.1f}k"
        if v >= 10.0: return f"{v:.0f}"
        if v >= 1.0:  return f"{v:.1f}"
        return f"{v:.3f}"

    def update_theme(self):
        """Refresh widget styles after a theme toggle."""
        bar_ss = (
            "QProgressBar {"
            f"  border:1px solid {C_BORDER};"
            "  border-radius:3px;"
            f"  background:{C_BG};"
            "  height:12px;"
            "}"
            "QProgressBar::chunk {"
            "  background:#0066CC;"
            "  border-radius:2px;"
            "}"
        )
        t = "background:transparent;"
        for band in self._BANDS:
            self._bars[band].setStyleSheet(bar_ss)
            self._val_lbls[band] .setStyleSheet(f"color:{C_TEXT};  {t}")
            self._peak_lbls[band].setStyleSheet(f"color:{C_SUBTEXT}; {t}")
        for lbl in self._name_lbls:
            lbl.setStyleSheet(f"color:{C_TEXT}; {t}")

    # ── Core update — called every 10 ticks (~400 ms) ────────────────────────

    def update_from_segment(self, segment: np.ndarray, fs: float):
        """
        Compute real Welch-PSD band power (µV²) for each EEG band and refresh
        the live bar widget.

        Pipeline per band
        ─────────────────
        1. Welch PSD of ch0 (shared across all five bands — computed once)
        2. Integrate PSD over the band frequency range  →  power in µV²
        3. Find the PSD peak bin inside the band        →  peak_freq in Hz
        4. Exponential smoothing (α = 0.35) on raw power to reduce jitter
        5. Bar fill  = smooth_power / session_peak  (0–10 000, auto-scales)
        6. Val label = smooth_power formatted as µV²
        7. Peak label= ⏶ peak_freq in Hz
        """
        if segment is None or segment.size < 8:
            return

        ch0 = segment[0] if segment.ndim == 2 else segment
        ch0 = np.asarray(ch0, dtype=np.float64).ravel()

        # ── Single Welch PSD — reused for all five bands ──────────────────
        freqs, psd = _welch_safe(ch0, fs)

        for band in self._BANDS:
            flo, fhi = self._RANGES[band]
            idx = (freqs >= flo) & (freqs <= fhi)

            if idx.any():
                # Band power: area under PSD curve in this frequency range (µV²)
                power = float(_trapz(psd[idx], freqs[idx]))

                # Peak frequency: PSD maximum bin within the band
                peak_bin  = int(np.argmax(psd[idx]))
                peak_freq = float(freqs[np.where(idx)[0][peak_bin]])
            else:
                power     = 0.0
                peak_freq = (flo + fhi) / 2.0

            # ── Exponential smoothing (α = 0.35) ──────────────────────────
            alpha              = 0.35
            self._smooth[band] = (alpha * power
                                  + (1.0 - alpha) * self._smooth[band])
            smooth_power = self._smooth[band]

            # ── Auto-scale: track session-peak power per band ─────────────
            if smooth_power > self._peak[band]:
                self._peak[band] = smooth_power
            session_peak = max(self._peak[band], 1e-12)

            # ── Progress bar: proportion of session-peak (0–10 000) ───────
            fill = int(min(smooth_power / session_peak, 1.0) * 10_000)
            self._bars[band].setValue(fill)

            # ── Value label: live band power in µV² ───────────────────────
            self._val_lbls[band].setText(self._fmt(smooth_power) + "µV²")

            # ── Peak-frequency label: dominant Hz inside the band ─────────
            self._peak_lbls[band].setText(f"\u23f6 {peak_freq:.1f} Hz")


# =============================================================================
#  WORKER THREADS
# =============================================================================

class TrainWorker(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, normal_folder, abnormal_folder, fs, model,
                 max_segments=3000):
        super().__init__()
        self.normal_folder   = normal_folder
        self.abnormal_folder = abnormal_folder
        self.fs              = fs
        self.model           = model
        self.max_segments    = max_segments

    def run(self):
        try:
            self.progress.emit(5, "Loading normal samples…")
            normal_files = self._get_edf_files(self.normal_folder)
            X_n, y_n = self._load_folder(normal_files, 0, 5, 30)

            self.progress.emit(30, "Loading abnormal samples…")
            abnormal_files = self._get_edf_files(self.abnormal_folder)
            X_a, y_a = self._load_folder(abnormal_files, 1, 30, 55)

            X = np.vstack([X_n, X_a])
            y = np.hstack([y_n, y_a])

            self.progress.emit(57, f"Training ensemble on {len(X)} segments…")
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y)

            self.model.train(X_tr, y_tr)

            self.progress.emit(82, "Evaluating on hold-out test set…")
            proba_te = self.model.predict_proba_batch(X_te)
            y_pred   = (proba_te >= self.model.threshold).astype(int)

            acc  = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec  = recall_score(y_te, y_pred, zero_division=0)
            f1   = f1_score(y_te, y_pred, zero_division=0)
            cm   = confusion_matrix(y_te, y_pred, labels=[0, 1]).tolist()
            try:
                auc = roc_auc_score(y_te, proba_te)
            except Exception:
                auc = 0.0

            self.progress.emit(92, "Computing feature importance…")
            importance = self._feat_importance(self.model.pipeline, X_te, y_te)
            top_idx    = np.argsort(importance)[-10:][::-1]
            top_feat   = [(FEATURE_NAMES[i], float(importance[i]))
                          for i in top_idx if i < len(FEATURE_NAMES)]

            self.progress.emit(100, "Done!")
            self.finished.emit({
                "accuracy": float(acc), "precision": float(prec),
                "recall": float(rec), "f1": float(f1), "auc": float(auc),
                "cv_score": self.model.cv_score, "cv_auc": self.model.cv_auc,
                "threshold": self.model.threshold,
                "conf_matrix": cm,
                "n_train": len(X_tr), "n_test": len(X_te),
                "top_features": top_feat,
                "has_xgb": _HAS_XGB,
            })
        except Exception as e:
            self.error.emit(f"Training error: {e}\n{traceback.format_exc()}")

    def _get_edf_files(self, folder):
        if not os.path.isdir(folder):
            raise ValueError(f"Folder not found: {folder}")
        files = glob.glob(os.path.join(folder, "*.edf"))
        if not files:
            raise ValueError(f"No EDF files in {folder}")
        return sorted(files)

    def _load_folder(self, files, label, prog_start, prog_end):
        import mne
        X, y = [], []
        for i, fpath in enumerate(files):
            try:
                if len(X) >= self.max_segments:
                    break
                raw  = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                raw.pick_types(eeg=True)
                data = raw.get_data() * 1e6
                fs   = float(raw.info["sfreq"])
                win  = int(WINDOW_SEC * fs)
                step = int(WINDOW_STEP_SEC * fs)
                for start in range(0, data.shape[1] - win + 1, step):
                    if len(X) >= self.max_segments:
                        break
                    feat = extract_features(data[:, start:start+win], fs)
                    X.append(feat); y.append(label)
                pct = int(prog_start + (i+1)/len(files) * (prog_end - prog_start))
                self.progress.emit(pct, f"Processed {i+1}/{len(files)} files")
            except Exception as e:
                print(f"[WARN] Skipped {fpath}: {e}")
        if not X:
            raise ValueError("No segments extracted — check folder content")
        return np.array(X, dtype=np.float32), np.array(y)

    def _feat_importance(self, pipeline, X_te, y_te):
        try:
            res = permutation_importance(
                pipeline, X_te, y_te, n_repeats=5,
                random_state=42, n_jobs=-1)
            return res.importances_mean
        except Exception:
            return np.std(X_te, axis=0)


# =============================================================================
#  CLASSIFY WORKER
# =============================================================================

class ClassifyWorker(QObject):
    finished  = pyqtSignal(list)
    error     = pyqtSignal(str)
    progress  = pyqtSignal(int, str)

    def __init__(self, data, fs, model):
        super().__init__()
        self.data       = data
        self.fs         = fs
        self.model      = model
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            win   = int(WINDOW_SEC * self.fs)
            step  = int(WINDOW_STEP_SEC * self.fs)
            total = max(1, (self.data.shape[1] - win) // step + 1)

            features_list = []
            for i in range(total):
                if self._cancelled:
                    self.error.emit("Cancelled by user.")
                    return

                seg = self.data[:, i * step: i * step + win]
                if seg.shape[1] < win:
                    break
                features_list.append(extract_features(seg, self.fs))

                pct = int(50 * (i + 1) / total)
                self.progress.emit(pct, f"Extracting features  {i + 1}/{total}")

            if not features_list:
                self.error.emit("Not enough data to extract features.")
                return

            X_all = np.array(features_list, dtype=np.float32)
            n_win = len(X_all)
            results = []

            if self.model.pipeline is not None:
                try:
                    cls_list    = list(
                        self.model.pipeline.named_steps["ensemble"].classes_)
                    abn_idx     = cls_list.index(1) if 1 in cls_list else 1
                    n_chunks    = max(1, (n_win + _PREDICT_CHUNK - 1)
                                      // _PREDICT_CHUNK)

                    for ci, start in enumerate(range(0, n_win, _PREDICT_CHUNK)):
                        if self._cancelled:
                            self.error.emit("Cancelled by user.")
                            return

                        chunk = X_all[start: start + _PREDICT_CHUNK]
                        p     = self.model.pipeline.predict_proba(chunk)
                        results.extend(float(v) for v in p[:, abn_idx])

                        pct = 50 + int(50 * (ci + 1) / n_chunks)
                        self.progress.emit(pct, f"Classifying chunk {ci + 1}/{n_chunks}")

                except Exception:
                    results = [self.model._threshold_heuristic(f) for f in features_list]
            else:
                for i, f in enumerate(features_list):
                    if self._cancelled:
                        self.error.emit("Cancelled by user.")
                        return
                    results.append(self.model._threshold_heuristic(f))
                    pct = 50 + int(50 * (i + 1) / n_win)
                    self.progress.emit(pct, f"Heuristic {i + 1}/{n_win}")

            self.progress.emit(100, "Done")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"Classification error: {e}\n{traceback.format_exc()}")


# =============================================================================
#  CLASSIFICATION RESULT DIALOG
# =============================================================================

class ClassificationResultDialog(QDialog):
    def __init__(self, results: list, threshold: float,
                 model_name: str = "Ensemble", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Classification Results")
        self.setMinimumSize(640, 560)
        self._results    = np.array(results, dtype=np.float32)
        self._threshold  = threshold
        self._model_name = model_name
        self._build()
        self.setStyleSheet(f"""
            QDialog {{ background:{C_PANEL}; color:{C_TEXT}; }}
            QLabel  {{ color:{C_TEXT}; }}
            QPushButton {{
                background:{C_BG}; color:{C_TEXT};
                border:1px solid {C_BORDER}; border-radius:4px;
                padding:5px 20px; font-size:12px;
            }}
            QPushButton:hover {{ background:{C_ACCENT}; color:#FFFFFF; }}
        """)

    def _build(self):
        arr = self._results; thr = self._threshold
        n_total    = len(arr)
        n_abnormal = int(np.sum(arr >= thr))
        n_normal   = n_total - n_abnormal
        pct_abn    = 100.0 * n_abnormal / max(n_total, 1)
        pct_nor    = 100.0 * n_normal   / max(n_total, 1)
        mean_p     = float(arr.mean())
        max_p      = float(arr.max())
        min_p      = float(arr.min())
        verdict    = mean_p >= thr

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14); lay.setSpacing(10)

        banner = QLabel("⚠  ABNORMAL  (SEIZURE DETECTED)" if verdict else "✓  NORMAL")
        banner.setFont(QFont("Courier New", 15, QFont.Bold))
        banner.setAlignment(Qt.AlignCenter)
        banner.setStyleSheet(
            f"color:{C_RED}; padding:8px; border:2px solid {C_RED};"
            f"border-radius:6px; background:#FFF0EE;"
            if verdict else
            f"color:{C_GREEN}; padding:8px; border:2px solid {C_GREEN};"
            f"border-radius:6px; background:#EDFAF3;")
        lay.addWidget(banner)

        pct_frame = QFrame()
        pct_frame.setStyleSheet(
            f"QFrame {{ border:1px solid {C_BORDER}; border-radius:6px;"
            f" background:{C_BG}; }}")
        pct_h = QHBoxLayout(pct_frame)
        pct_h.setContentsMargins(12, 10, 12, 10); pct_h.setSpacing(0)

        for label_txt, pct_val, col in [
                ("NORMAL",   pct_nor, str(C_GREEN)),
                ("ABNORMAL", pct_abn, str(C_RED))]:
            cell = QVBoxLayout()
            big = QLabel(f"{pct_val:.1f}%")
            big.setFont(QFont("Courier New", 26, QFont.Bold))
            big.setAlignment(Qt.AlignCenter)
            big.setStyleSheet(f"color:{col}; border:none;")
            sub = QLabel(f"{label_txt} windows")
            sub.setFont(QFont("Courier New", 9))
            sub.setAlignment(Qt.AlignCenter)
            sub.setStyleSheet(f"color:{C_SUBTEXT}; border:none;")
            n_lbl = QLabel(f"({n_normal if label_txt=='NORMAL' else n_abnormal} / {n_total})")
            n_lbl.setFont(QFont("Courier New", 9))
            n_lbl.setAlignment(Qt.AlignCenter)
            n_lbl.setStyleSheet(f"color:{C_SUBTEXT}; border:none;")
            cell.addWidget(big); cell.addWidget(sub); cell.addWidget(n_lbl)
            pct_h.addLayout(cell, stretch=1)
            if label_txt == "NORMAL":
                div = QFrame(); div.setFrameShape(QFrame.VLine)
                div.setStyleSheet(f"color:{C_BORDER};")
                pct_h.addWidget(div)

        lay.addWidget(pct_frame)

        gw_tl = pg.GraphicsLayoutWidget()
        gw_tl.setBackground(str(C_EEG_BG)); gw_tl.setFixedHeight(100)
        lay.addWidget(gw_tl)

        p_tl = gw_tl.addPlot()
        p_tl.setMenuEnabled(False); p_tl.hideButtons()
        p_tl.setMouseEnabled(x=False, y=False)
        p_tl.showGrid(x=True, y=True, alpha=0.15)
        for ax in ("top", "right"):
            p_tl.showAxis(ax, False)
        p_tl.getAxis("bottom").setPen(pg.mkPen(str(C_BORDER)))
        p_tl.getAxis("bottom").setLabel("Window index")
        p_tl.getAxis("left").setPen(pg.mkPen(str(C_BORDER)))
        p_tl.getAxis("left").setLabel("P(Abnormal)")
        p_tl.setYRange(0, 1.05, padding=0)

        x_wins  = np.arange(len(arr), dtype=np.float32)
        colours = np.array(
            [[192, 57, 43, 220] if v >= thr else [26, 140, 78, 200]
             for v in arr], dtype=np.uint8)
        scatter = pg.ScatterPlotItem(
            x=x_wins, y=arr, size=5,
            brush=[pg.mkBrush(*c) for c in colours],
            pen=pg.mkPen(None))
        p_tl.addItem(scatter)
        p_tl.addItem(pg.InfiniteLine(
            pos=thr, angle=0,
            pen=pg.mkPen(str(C_YELLOW), width=1, style=Qt.DashLine)))

        stats_frame = QFrame()
        stats_frame.setStyleSheet(
            f"QFrame {{ border:1px solid {C_BORDER}; border-radius:4px; background:{C_BG}; }}")
        sf_lay = QVBoxLayout(stats_frame)
        sf_lay.setContentsMargins(12, 8, 12, 8); sf_lay.setSpacing(3)

        stats = [
            ("Total windows",       f"{n_total}",                   str(C_TEXT)),
            ("Normal windows",      f"{n_normal}  ({pct_nor:.1f}%)", str(C_GREEN)),
            ("Abnormal windows",    f"{n_abnormal}  ({pct_abn:.1f}%)", str(C_RED)),
            ("Mean P(Abnormal)",    f"{mean_p*100:.1f}%",           str(C_TEXT)),
            ("Peak P(Abnormal)",    f"{max_p*100:.1f}%",            str(C_TEXT)),
            ("Decision threshold",  f"{thr:.3f}",                   str(C_TEXT)),
            ("Model",               self._model_name,               str(C_TEXT)),
        ]
        sf = QFont("Courier New", 10)
        sfb = QFont("Courier New", 10, QFont.Bold)
        for key, val, col in stats:
            row_lay = QHBoxLayout(); row_lay.setSpacing(0)
            k_lbl = QLabel(key + ":"); k_lbl.setFont(sf)
            k_lbl.setStyleSheet(f"color:{C_SUBTEXT}; border:none;")
            v_lbl = QLabel(val); v_lbl.setFont(sfb)
            v_lbl.setStyleSheet(f"color:{col}; border:none;")
            v_lbl.setAlignment(Qt.AlignRight)
            row_lay.addWidget(k_lbl); row_lay.addStretch()
            row_lay.addWidget(v_lbl); sf_lay.addLayout(row_lay)
        lay.addWidget(stats_frame)

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(34); close_btn.clicked.connect(self.accept)
        lay.addWidget(close_btn)


# =============================================================================
#  MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Signal Analysis Tool  –  Ensemble ML v4")
        self.setMinimumSize(1240, 780)

        self.fs            = DEFAULT_FS
        self.raw_data      = None
        self.filtered_data = None
        self.n_channels    = 18
        self.file_path     = ""
        self.is_streaming  = False
        self.filt_bp = self.filt_hp = self.filt_lp = self.filt_notch = False
        self.filt_fir = self.filt_cheby = False

        self._stream_ptr = 0; self._spec_tick = 0

        self.model = SeizureModel()
        self.synth = SyntheticEEG(n_channels=self.n_channels, fs=self.fs)

        self._feat_buf_size  = int(WINDOW_SEC * self.fs)
        self._feat_buf       = None
        self._feat_ptr       = 0
        self._last_proba     = 0.0
        self._last_feat      = None
        self.model_trained   = False
        self._has_classification = False

        self.train_thread    = None
        self.train_worker    = None
        self.classify_thread = None
        self.classify_worker = None
        self._classify_running = False
        self._train_running    = False
        self._progress_dlg: QProgressDialog = None

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._apply_theme()
        self._start_streaming()

    # =========================================================================
    #  UI BUILD
    # =========================================================================

    def _build_ui(self):
        rw = QWidget(); self.setCentralWidget(rw)
        rv = QVBoxLayout(rw)
        rv.setContentsMargins(0,0,0,0); rv.setSpacing(0)
        rv.addWidget(self._make_header())
        rv.addWidget(self._make_toolbar())
        rv.addWidget(self._make_body(), stretch=1)
        rv.addWidget(self._make_statusbar())

    def _make_header(self):
        w = QWidget(); w.setFixedHeight(48); w.setObjectName("header")
        h = QHBoxLayout(w); h.setContentsMargins(16,0,16,0)
        self._hdr_dot = QLabel("●")
        self._hdr_dot.setStyleSheet(f"color:{C_ACCENT}; font-size:18px;")
        h.addWidget(self._hdr_dot)
        self._hdr_title = QLabel("EEG Signal Analysis Tool  –  Ensemble ML")
        self._hdr_title.setFont(QFont("Courier New", 13, QFont.Bold))
        self._hdr_title.setStyleSheet(f"color:{C_TEXT}; background:transparent;")
        h.addWidget(self._hdr_title)
        h.addStretch()
        self.lbl_file = QLabel("No file loaded – Demo mode")
        self.lbl_file.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px;")
        h.addWidget(self.lbl_file)
        return w

    def _make_toolbar(self):
        w = QWidget(); w.setFixedHeight(50); w.setObjectName("toolbar")
        h = QHBoxLayout(w); h.setContentsMargins(12,6,12,6); h.setSpacing(8)
        defs = [
            ("Load File",     self._load_file,       C_ACCENT),
            ("Filters",       self._open_filters,    C_ACCENT),
            ("Analyze",       self._analyze,         C_ACCENT),
            ("Show Features", self._show_features,   C_GREEN),
            ("Classify",      self._classify,        C_ACCENT),
            ("Train Model",   self._train_model,     C_YELLOW),
            ("Save Model",    self._save_model,      C_SUBTEXT),
            ("Load Model",    self._load_model,      C_SUBTEXT),
        ]
        for label, slot, color in defs:
            btn = QPushButton(label)
            btn.setFixedHeight(36); btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(slot)
            btn.setStyleSheet(self._tbtn_ss(color))
            h.addWidget(btn)
        h.addStretch()
        self.btn_theme = QPushButton("🌙 Dark")
        self.btn_theme.setFixedHeight(36); self.btn_theme.setCursor(Qt.PointingHandCursor)
        self.btn_theme.setCheckable(True)
        self.btn_theme.clicked.connect(self._toggle_theme)
        self.btn_theme.setStyleSheet(self._tbtn_ss(C_ACCENT))
        h.addWidget(self.btn_theme)
        h.addWidget(QLabel("Gain:"))
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.1, 20.0); self.spin_gain.setSingleStep(0.1)
        self.spin_gain.setValue(1.0); self.spin_gain.setFixedWidth(72)
        self.spin_gain.valueChanged.connect(lambda v: self.eeg_plot.set_gain(v))
        h.addWidget(self.spin_gain)
        self.btn_play = QPushButton("Pause")
        self.btn_play.setFixedHeight(36); self.btn_play.setCursor(Qt.PointingHandCursor)
        self.btn_play.clicked.connect(self._toggle_stream)
        self.btn_play.setStyleSheet(self._tbtn_ss(C_GREEN))
        h.addWidget(self.btn_play)
        return w

    def _tbtn_ss(self, accent):
        return f"""
            QPushButton {{
                background:{C_BG}; color:{accent};
                border:1px solid {C_BORDER}; border-radius:4px;
                font-size:12px; padding:0 12px;
            }}
            QPushButton:hover {{ background:{C_BORDER}; border-color:{accent}; }}
            QPushButton:pressed {{ background:{C_EEG_BG}; }}
        """

    def _make_body(self):
        w  = QWidget()
        hh = QHBoxLayout(w); hh.setContentsMargins(0,0,0,0); hh.setSpacing(0)
        left = QWidget()
        lv   = QVBoxLayout(left); lv.setContentsMargins(0,0,0,0); lv.setSpacing(0)
        self.eeg_plot = EEGPlotWidget(n_channels=self.n_channels, fs=self.fs)
        lv.addWidget(self.eeg_plot, stretch=3)
        self.spectral = SpectralWidget(fs=self.fs)
        self.spectral.setMaximumHeight(175)
        lv.addWidget(self.spectral, stretch=1)
        hh.addWidget(left, stretch=4)
        hh.addWidget(self._make_info_panel(), stretch=0)
        return w

    def _make_info_panel(self):
        panel = QWidget(); panel.setFixedWidth(340); panel.setObjectName("infoPanel")
        v = QVBoxLayout(panel); v.setContentsMargins(12,12,12,12); v.setSpacing(8)

        v.addWidget(self._sec("STATUS"))
        self.lbl_status = QLabel("ANALYSING…\n(HEURISTIC)")
        self.lbl_status.setFont(QFont("Courier New", 13, QFont.Bold))
        self.lbl_status.setStyleSheet(f"color:{C_YELLOW};")
        self.lbl_status.setAlignment(Qt.AlignCenter); self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)

        self.prob_bar = QProgressBar()
        self.prob_bar.setRange(0,100); self.prob_bar.setValue(0)
        self.prob_bar.setFixedHeight(10); self.prob_bar.setTextVisible(False)
        self.prob_bar.setStyleSheet(f"""
            QProgressBar {{ border:1px solid {C_BORDER}; border-radius:5px; background:{C_BG}; }}
            QProgressBar::chunk {{ background:{C_RED}; border-radius:5px; }}
        """)
        v.addWidget(self.prob_bar)
        self.lbl_proba = QLabel("Abnormal probability: 0%")
        self.lbl_proba.setStyleSheet(f"color:{C_SUBTEXT}; font-size:10px;")
        self.lbl_proba.setAlignment(Qt.AlignCenter)
        v.addWidget(self.lbl_proba)
        self.lbl_accuracy  = QLabel("Mode: Heuristic")
        self.lbl_accuracy.setStyleSheet(f"color:{C_YELLOW}; font-size:10px; font-weight:bold;")
        self.lbl_accuracy.setAlignment(Qt.AlignCenter)
        v.addWidget(self.lbl_accuracy)
        self.lbl_threshold = QLabel(f"Threshold: {SEIZURE_THR:.2f}")
        self.lbl_threshold.setStyleSheet(f"color:{C_SUBTEXT}; font-size:10px;")
        self.lbl_threshold.setAlignment(Qt.AlignCenter)
        v.addWidget(self.lbl_threshold)

        v.addWidget(self._hsep())
        v.addWidget(self._sec("SIGNAL INFO"))
        self.info_labels = {}
        self._info_key_lbls = []
        for key in ["Channels", "Sample rate", "Duration", "Filters"]:
            row = QHBoxLayout()
            k   = QLabel(key+":"); k.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px;")
            val = QLabel("---"); val.setStyleSheet(f"color:{C_TEXT}; font-size:11px;")
            val.setAlignment(Qt.AlignRight)
            row.addWidget(k); row.addWidget(val)
            v.addLayout(row)
            self.info_labels[key] = val
            self._info_key_lbls.append(k)
        self._update_info()

        v.addWidget(self._hsep())

        # ── BAND POWER (real-time bars, µV² + peak Hz) ───────────────────
        v.addWidget(self._sec("BAND POWER  (ch0  µV²)"))
        self.band_power_widget = BandPowerWidget()
        v.addWidget(self.band_power_widget)

        v.addWidget(self._hsep())
        v.addWidget(self._sec("LIVE FEATURES"))
        self._feat_summary_labels = {}
        self._feat_key_lbls_list  = []
        feat_units = {"mean": "(µV)", "std": "(µV)", "dominant_freq": "(Hz)",
                      "shannon_entropy": "", "hjorth_mobility": ""}
        key_feats = ["mean","std","dominant_freq","shannon_entropy","hjorth_mobility"]
        for fname in key_feats:
            row = QHBoxLayout()
            unit = feat_units.get(fname, "")
            label_text = fname[:14] + (" " + unit if unit else "") + ":"
            k   = QLabel(label_text); k.setFont(QFont("Courier New", FS_LIVE_LABEL))
            k.setStyleSheet(f"color:{C_SUBTEXT};")
            val = QLabel("---"); val.setFont(QFont("Courier New", FS_LIVE_VALUE, QFont.Bold))
            val.setStyleSheet(f"color:{C_TEXT};"); val.setAlignment(Qt.AlignRight)
            row.addWidget(k); row.addWidget(val)
            v.addLayout(row)
            self._feat_summary_labels[fname] = val
            self._feat_key_lbls_list.append(k)

        v.addStretch()
        v.addWidget(self._hsep())
        self.lbl_model = QLabel("Model: heuristic threshold")
        self.lbl_model.setStyleSheet(f"color:{C_SUBTEXT}; font-size:10px;")
        self.lbl_model.setWordWrap(True); v.addWidget(self.lbl_model)
        return panel

    def _sec(self, text):
        l = QLabel(text); l.setFont(QFont("Courier New", 10, QFont.Bold))
        l.setStyleSheet(f"color:{C_ACCENT}; letter-spacing:2px; background:transparent;")
        if not hasattr(self, "_sec_labels"):
            self._sec_labels = []
        self._sec_labels.append(l)
        return l

    def _hsep(self):
        s = QFrame(); s.setFrameShape(QFrame.HLine)
        s.setStyleSheet(f"color:{C_BORDER};"); return s

    def _make_statusbar(self):
        w = QWidget(); w.setFixedHeight(24); w.setObjectName("statusBar")
        h = QHBoxLayout(w); h.setContentsMargins(12,0,12,0)
        self.lbl_stream = QLabel("STREAMING (DEMO)")
        self.lbl_stream.setStyleSheet(f"color:{C_GREEN}; font-size:10px;")
        h.addWidget(self.lbl_stream); h.addStretch()
        self.lbl_time = QLabel("t = 0.0 s")
        self.lbl_time.setStyleSheet(f"color:{C_SUBTEXT}; font-size:10px;")
        h.addWidget(self.lbl_time)
        return w

    def _toggle_theme(self):
        _T.toggle()
        self.btn_theme.setText("☀️ Light" if _T.dark else "🌙 Dark")
        self._apply_theme()

    def _apply_theme(self):
        bg      = str(C_BG);     panel  = str(C_PANEL)
        border  = str(C_BORDER); accent = str(C_ACCENT)
        text    = str(C_TEXT);   sub    = str(C_SUBTEXT)
        eeg_bg  = str(C_EEG_BG)
        red     = str(C_RED);    green  = str(C_GREEN)
        yellow  = str(C_YELLOW)

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{bg}; color:{text}; }}
            #header    {{ background:{panel}; border-bottom:1px solid {border}; }}
            #toolbar   {{ background:{panel}; border-bottom:1px solid {border}; }}
            #infoPanel {{ background:{panel}; border-left:1px solid {border}; }}
            #statusBar {{ background:{panel}; border-top:1px solid {border}; }}
            QLabel     {{ color:{text}; }}
            QDoubleSpinBox {{
                background:{bg}; color:{text};
                border:1px solid {border}; border-radius:3px; padding:2px;
            }}
            QScrollBar:vertical  {{ background:{bg}; width:6px; border:none; }}
            QScrollBar::handle:vertical {{ background:{border}; border-radius:3px; }}
        """)

        self._hdr_dot.setStyleSheet(f"color:{accent}; font-size:18px; background:transparent;")
        self._hdr_title.setStyleSheet(
            f"color:{text}; font-size:13px; font-weight:bold; background:transparent;")
        self.lbl_file.setStyleSheet(f"color:{sub}; font-size:11px; background:transparent;")

        btn_defs = [
            ("Load File",     accent),
            ("Filters",       accent),
            ("Analyze",       accent),
            ("Show Features", green),
            ("Classify",      accent),
            ("Train Model",   yellow),
            ("Save Model",    sub),
            ("Load Model",    sub),
        ]
        toolbar = self.findChild(QWidget, "toolbar")
        if toolbar:
            btns = [b for b in toolbar.findChildren(QPushButton)
                    if b is not self.btn_play and b is not self.btn_theme]
            for btn, (_, col) in zip(btns, btn_defs):
                btn.setStyleSheet(self._tbtn_ss(col))
        self.btn_play.setStyleSheet(self._tbtn_ss(green))
        self.btn_theme.setStyleSheet(self._tbtn_ss(accent))

        for lbl in getattr(self, "_sec_labels", []):
            lbl.setStyleSheet(
                f"color:{accent}; letter-spacing:2px; background:transparent;")

        self.prob_bar.setStyleSheet(f"""
            QProgressBar {{ border:1px solid {border}; border-radius:5px;
                background:{bg}; }}
            QProgressBar::chunk {{ background:{red}; border-radius:5px; }}
        """)

        t = f"background:transparent;"
        self.lbl_proba    .setStyleSheet(f"color:{text}; font-size:10px; {t}")
        self.lbl_threshold.setStyleSheet(f"color:{sub};  font-size:10px; {t}")
        self.lbl_model    .setStyleSheet(f"color:{sub};  font-size:10px; {t}")
        self.lbl_file     .setStyleSheet(f"color:{sub};  font-size:11px; {t}")
        self.lbl_time     .setStyleSheet(f"color:{sub};  font-size:10px; {t}")
        self.lbl_stream   .setStyleSheet(f"color:{green};font-size:10px; {t}")

        acc_ss = self.lbl_accuracy.styleSheet()
        if "background" not in acc_ss:
            self.lbl_accuracy.setStyleSheet(acc_ss + f" {t}")

        st_ss = self.lbl_status.styleSheet()
        if "background" not in st_ss:
            self.lbl_status.setStyleSheet(st_ss + f" {t}")

        for key, val_lbl in self.info_labels.items():
            val_lbl.setStyleSheet(f"color:{text}; font-size:11px; {t}")
        for lbl in getattr(self, "_info_key_lbls", []):
            lbl.setStyleSheet(f"color:{sub}; font-size:11px; {t}")

        for val_lbl in self._feat_summary_labels.values():
            val_lbl.setStyleSheet(f"color:{text}; {t}")
        for lbl in getattr(self, "_feat_key_lbls_list", []):
            lbl.setStyleSheet(f"color:{sub}; {t}")

        self.band_power_widget.update_theme()

        self.eeg_plot._gw.setBackground(eeg_bg)
        self.eeg_plot._build()
        for sep_line in self.eeg_plot._plot.items:
            if isinstance(sep_line, pg.InfiniteLine) and sep_line.angle == 0:
                sep_line.setPen(pg.mkPen(border, width=1))

        self.spectral.gw.setBackground(eeg_bg)
        self.spectral.plot.getAxis("bottom").setPen(pg.mkPen(border))
        self.spectral.plot.getAxis("left").setPen(pg.mkPen(border))
        self.spectral._fft_curve.setPen(pg.mkPen(accent, width=1.5))

    # =========================================================================
    #  STREAMING
    # =========================================================================

    def _start_streaming(self):
        self.is_streaming = True; self._timer.start(TIMER_MS)
        self.btn_play.setText("Pause")
        self.lbl_stream.setText(
            "STREAMING (FILE)" if self.raw_data is not None else "STREAMING (DEMO)")
        self.lbl_stream.setStyleSheet(f"color:{C_GREEN}; font-size:10px;")

    def _stop_streaming(self):
        self.is_streaming = False; self._timer.stop()
        self.btn_play.setText("Resume")
        self.lbl_stream.setText("PAUSED")
        self.lbl_stream.setStyleSheet(f"color:{C_YELLOW}; font-size:10px;")

    def _toggle_stream(self):
        if self.is_streaming:
            self._stop_streaming()
        else:
            self._start_streaming()

    # =========================================================================
    #  MAIN TICK
    # =========================================================================

    def _tick(self):
        n = SAMPLES_PER_TICK
        if self.raw_data is not None:
            src = self.filtered_data if self.filtered_data is not None \
                  else self.raw_data
            end = self._stream_ptr + n
            if end > src.shape[1]:
                self._stream_ptr = 0; end = n
            n_ch  = min(self.n_channels, src.shape[0])
            chunk = src[:n_ch, self._stream_ptr:end].copy()
            self._stream_ptr = end
            t_sec = self._stream_ptr / self.fs
            if chunk.shape[1] == 0:
                return
        else:
            chunk = self.synth.next_samples(n)
            t_sec = self.synth._t

        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.shape[0] > self.n_channels:
            chunk = chunk[:self.n_channels, :]

        sz_flags = self._detect(chunk)
        if len(sz_flags) != chunk.shape[1]:
            sz_flags = np.zeros(chunk.shape[1], dtype=bool)

        self.eeg_plot.push_samples(chunk, sz_flags)
        self.eeg_plot.refresh()

        self._spec_tick += 1
        if self._spec_tick % 10 == 0:
            nd  = self.eeg_plot._n_disp; buf = self.eeg_plot._buf[0]
            ptr = self.eeg_plot._ptr % nd; tail = nd - ptr
            order = np.empty(nd, dtype=np.int32)
            order[:tail] = np.arange(ptr, nd); order[tail:] = np.arange(0, ptr)
            ordered_buf = buf[order]
            self.spectral.update_data(ordered_buf)

            # Update band power bars every 10 ticks (~400 ms)
            seg_for_bands = self.eeg_plot._buf[:, order]  # (n_ch, n_disp)
            self.band_power_widget.update_from_segment(seg_for_bands, self.fs)

        if self._has_classification:
            p = int(self._last_proba * 100)
            self.prob_bar.setValue(p)
            self.lbl_proba.setText(f"Abnormal probability: {p}%")
        else:
            self.prob_bar.setValue(0)
            self.lbl_proba.setText("Abnormal probability: —")

        if not self._has_classification:
            mode = "ML MODEL" if self.model_trained else "HEURISTIC"
            self.lbl_status.setText(f"ANALYSING…\n({mode})")
            self.lbl_status.setStyleSheet(
                f"color:{C_ACCENT}; font-weight:bold; font-size:11px;")
        else:
            thr = self.model.threshold
            if self._last_proba >= thr:
                mode = "ML" if self.model_trained else "HEUR"
                self.lbl_status.setText(f"⚠  ABNORMAL\n[{mode}] LIVE")
                self.lbl_status.setStyleSheet(
                    f"color:{C_RED}; font-weight:bold; font-size:13px;")
            else:
                mode = "ML" if self.model_trained else "HEUR"
                self.lbl_status.setText(f"✓  NORMAL\n[{mode}] LIVE")
                self.lbl_status.setStyleSheet(
                    f"color:{C_GREEN}; font-weight:bold; font-size:13px;")

        self.lbl_time.setText(f"t = {t_sec:.1f} s")

    # =========================================================================
    #  DETECTION
    # =========================================================================

    def _detect(self, chunk: np.ndarray) -> np.ndarray:
        n = chunk.shape[1]
        if self._feat_buf is None:
            self._feat_buf = np.zeros(
                (self.n_channels, self._feat_buf_size), dtype=np.float32)
            self._feat_ptr = 0

        for k in range(n):
            col = self._feat_ptr % self._feat_buf_size
            self._feat_buf[:, col] = chunk[:, k]
            self._feat_ptr += 1

        sz = np.zeros(n, dtype=bool)
        if self._feat_ptr >= self._feat_buf_size:
            start = self._feat_ptr % self._feat_buf_size
            tail  = self._feat_buf_size - start
            order = np.empty(self._feat_buf_size, dtype=np.int32)
            order[:tail] = np.arange(start, self._feat_buf_size)
            order[tail:] = np.arange(0, start)
            seg  = self._feat_buf[:, order]
            feat = extract_features(seg, self.fs)
            self._last_feat = feat
            self._update_feat_summary(feat)

            proba = self.model.predict_proba(feat)
            self._last_proba = proba; self._has_classification = True
            if proba >= self.model.threshold:
                sz[:] = True
        return sz

    def _update_feat_summary(self, feat: np.ndarray):
        key_feats = ["mean","std","dominant_freq","shannon_entropy","hjorth_mobility"]
        for fname in key_feats:
            if fname in FEATURE_NAMES:
                idx = FEATURE_NAMES.index(fname)
                val = float(feat[idx]) if idx < len(feat) else 0.0
                self._feat_summary_labels[fname].setText(f"{val:.3g}")

    # =========================================================================
    #  TOOLBAR ACTIONS
    # =========================================================================

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load EEG File", "",
            "EEG Files (*.edf *.mat *.npy *.csv *.txt);;All Files (*)")
        if not path: return
        self._stop_streaming()
        try:
            data, fs = self._read_file(path)
            self.raw_data = data; self.fs = fs
            self.n_channels    = min(data.shape[0], len(CHANNEL_LABELS))
            self.filtered_data = None; self._stream_ptr = 0
            self._feat_buf = None; self._feat_ptr = 0
            self._last_proba = 0.0; self._last_feat = None
            self._has_classification = False
            self._feat_buf_size = int(WINDOW_SEC * self.fs)
            self.eeg_plot.reset(self.n_channels, self.fs)
            self.eeg_plot.set_amplitude_scale(data)
            self.spectral.fs = self.fs
            self.synth = SyntheticEEG(n_channels=self.n_channels, fs=self.fs)
            self.file_path = path
            self.lbl_file.setText(os.path.basename(path))
            self._update_info(); self._start_streaming()
            self._show_msg(
                f"Loaded: {os.path.basename(path)}\n"
                f"{self.n_channels} ch  |  {fs:.0f} Hz  |  "
                f"{data.shape[1]/fs:.1f} s")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            self._start_streaming()

    def _read_file(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            d = np.load(path); fs = DEFAULT_FS
        elif ext == ".mat":
            mat  = loadmat(path); keys = [k for k in mat if not k.startswith("_")]
            d = mat[keys[0]]; fs = DEFAULT_FS
        elif ext in (".csv", ".txt"):
            d = np.loadtxt(path, delimiter=","); fs = DEFAULT_FS
        elif ext == ".edf":
            try:
                import mne
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
                raw.pick_types(eeg=True)
                d = raw.get_data() * 1e6; fs = float(raw.info["sfreq"])
            except ImportError:
                raise RuntimeError("MNE not installed.  pip install mne")
        else:
            raise ValueError(f"Unsupported format: {ext}")
        if d.ndim == 1: d = d.reshape(1, -1)
        if d.shape[0] > d.shape[1]: d = d.T
        d = d.astype(np.float32)
        if 0.0 < float(np.max(np.abs(d))) < 0.005:
            d = d * 1e6
        return d, fs

    def _open_filters(self):
        dlg = FilterDialog(self, self.filt_bp, self.filt_hp,
                           self.filt_lp, self.filt_notch,
                           self.filt_fir, self.filt_cheby)
        dlg.filters_applied.connect(self._apply_filters); dlg.exec_()

    def _apply_filters(self, bp, hp, lp, notch, fir=False, cheby=False):
        self.filt_bp = bp; self.filt_hp = hp
        self.filt_lp = lp; self.filt_notch = notch
        self.filt_fir = fir; self.filt_cheby = cheby
        if self.raw_data is not None:
            self.filtered_data = apply_filters(
                self.raw_data[:self.n_channels], self.fs,
                bp, hp, lp, notch, fir, cheby)
            self.eeg_plot.set_amplitude_scale(self.filtered_data)
        self._update_info()
        active = [nm for nm, f in [("BP",bp),("HP",hp),("LP",lp),
                                    ("N50",notch),("FIR",fir),("CHEBY",cheby)] if f]
        self._show_msg("Filters: " + (", ".join(active) if active else "None"))

    def _analyze(self):
        d = self._cur_data()
        if d is None: self._show_msg("Load a file first."); return
        dur = d.shape[1] / self.fs; amp = float(np.mean(np.abs(d)))
        smp = min(d.shape[1], int(10 * self.fs))
        powers = {b: band_power(d[0, :smp], self.fs, flo, fhi)
                  for b, (flo, fhi) in BANDS.items()}
        win = int(WINDOW_SEC * self.fs)
        seg = d[:, :win] if d.shape[1] >= win else \
              np.pad(d, ((0,0),(0, win-d.shape[1])), mode='edge')
        feat = extract_features(seg, self.fs); self._last_feat = feat
        txt = (f"Signal Analysis\n{'─'*38}\n"
               f"Channels    : {d.shape[0]}\nDuration    : {dur:.2f} s\n"
               f"Sample rate : {self.fs:.0f} Hz\nMean |amp|  : {amp:.2f} µV\n"
               f"{'─'*38}\nBand Powers (first 10 s, ch0):\n"
               + "\n".join(f"  {b:6s} : {p:.2f}" for b, p in powers.items())
               + f"\n{'─'*38}\n{N_FEATURES} features extracted. → 'Show Features'\n\n"
               + "Generating band power visualization…")
        self._show_msg(txt)

        try:
            viz_result = visualize_band_power_spectrum(
                seg, self.fs,
                title="EEG Band Power and Peak Frequency Analysis",
                output_path=None, show=True)
            if viz_result:
                self._show_msg("✓ Band power chart generated successfully!\n"
                             "Close the chart window to continue.")
        except Exception as e:
            print(f"[WARN] Visualization failed: {e}")

    def _show_features(self):
        d = self._cur_data()
        if d is None and self._last_feat is None:
            self._show_msg("No features yet. Load a file first."); return
        feat_before = self._last_feat
        if feat_before is None:
            win = int(WINDOW_SEC * self.fs); n = d.shape[1]
            seg = d[:, :win] if n >= win else \
                  np.pad(d, ((0,0),(0, win-n)), mode='edge')
            feat_before = extract_features(seg, self.fs)
        feat_after = None
        if self.filtered_data is not None:
            win = int(WINDOW_SEC * self.fs); n = self.filtered_data.shape[1]
            seg = self.filtered_data[:, :win] if n >= win else \
                  np.pad(self.filtered_data, ((0,0),(0, win-n)), mode='edge')
            feat_after = extract_features(seg, self.fs)
        FeatureTableDialog(feat_before, feat_after, parent=self).exec_()

    def _classify(self):
        d = self._cur_data()
        if d is None:
            self._show_msg("No data loaded."); return
        if not self.model_trained:
            self._show_msg("No model trained.\nUse 'Train Model' or 'Load Model'.")
            return
        if self._classify_running:
            self._show_msg("Classification already running…"); return

        self._classify_running = True
        self.classify_worker = ClassifyWorker(d, self.fs, self.model)
        self.classify_thread = QThread(self)

        self.classify_worker.moveToThread(self.classify_thread)
        self.classify_thread.started.connect(
            self.classify_worker.run, Qt.QueuedConnection)
        self.classify_worker.finished.connect(
            self._on_classify_finished, Qt.QueuedConnection)
        self.classify_worker.error.connect(
            self._on_classify_error, Qt.QueuedConnection)
        self.classify_worker.progress.connect(
            self._on_classify_progress, Qt.QueuedConnection)
        self.classify_worker.finished.connect(
            self.classify_thread.quit, Qt.QueuedConnection)
        self.classify_worker.error.connect(
            self.classify_thread.quit, Qt.QueuedConnection)
        self.classify_thread.finished.connect(
            self._on_classify_thread_finished, Qt.QueuedConnection)

        self._progress_dlg = QProgressDialog(
            "Classifying EEG…", "Cancel", 0, 100, self)
        self._progress_dlg.setWindowTitle("Classification")
        self._progress_dlg.setWindowModality(Qt.WindowModal)
        self._progress_dlg.setMinimumDuration(0)
        self._progress_dlg.setValue(0)
        self._progress_dlg.setStyleSheet(f"""
            QProgressDialog {{ background:{C_PANEL}; color:{C_TEXT}; }}
            QLabel           {{ color:{C_TEXT}; }}
            QPushButton      {{ background:{C_BG}; color:{C_TEXT};
                                border:1px solid {C_BORDER}; border-radius:3px;
                                padding:4px 16px; }}
        """)
        self._progress_dlg.canceled.connect(self._cancel_classify)
        self.classify_thread.start()

    def _cancel_classify(self):
        if self.classify_worker is not None:
            self.classify_worker.cancel()

    def _on_classify_progress(self, pct: int, msg: str):
        if self._progress_dlg is not None:
            self._progress_dlg.setValue(pct)
            self._progress_dlg.setLabelText(msg)

    def _on_classify_thread_finished(self):
        self._classify_running = False
        if self._progress_dlg is not None:
            self._progress_dlg.close()
            self._progress_dlg = None
        self.classify_worker = None
        self.classify_thread = None

    def _on_classify_finished(self, results):
        if not results:
            self._show_msg("No classification results."); return
        arr = np.array(results, dtype=np.float32)
        thr = self.model.threshold
        self._last_proba         = float(arr.mean())
        self._has_classification = True
        model_name = ("Ensemble (SVM+RF+GB)" + ("+XGB" if _HAS_XGB else "")
                      if not self.model.threshold_mode else "Heuristic")
        dlg = ClassificationResultDialog(
            results, thr, model_name=model_name, parent=self)
        dlg.exec_()

    def _on_classify_error(self, e):
        self._show_msg(f"Classification Error:\n{e}")

    def _train_model(self):
        nf = QFileDialog.getExistingDirectory(
            self, "Select Folder with NORMAL EEG files (.edf)")
        if not nf: return
        af = QFileDialog.getExistingDirectory(
            self, "Select Folder with ABNORMAL/SEIZURE EEG files (.edf)")
        if not af: return
        if self._train_running:
            self._show_msg("Training already running…"); return

        self._train_running = True
        xgb_note = ("+ XGBoost" if _HAS_XGB else
                    "(XGBoost not installed — using SVM+RF+GB)")
        self.train_worker = TrainWorker(nf, af, self.fs, self.model,
                                        max_segments=3000)
        self.train_thread = QThread(self)
        self.train_worker.moveToThread(self.train_thread)
        self.train_thread.started.connect(
            self.train_worker.run, Qt.QueuedConnection)
        self.train_worker.finished.connect(
            self._on_train_finished, Qt.QueuedConnection)
        self.train_worker.error.connect(
            self._on_train_error, Qt.QueuedConnection)
        self.train_worker.finished.connect(
            self.train_thread.quit, Qt.QueuedConnection)
        self.train_worker.error.connect(
            self.train_thread.quit, Qt.QueuedConnection)
        self.train_thread.finished.connect(
            self._on_train_thread_finished, Qt.QueuedConnection)

        self._show_msg(
            f"Ensemble training started…\n"
            f"SVM + Random Forest + Gradient Boosting {xgb_note}\n"
            f"Pipeline: RobustScaler → PCA(95%) → RFE(30 feats) → VotingClassifier\n"
            f"Running in background — this may take a few minutes.")
        self.train_thread.start()

    def _on_train_thread_finished(self):
        self._train_running  = False
        self.train_worker    = None
        self.train_thread    = None

    def _on_train_finished(self, r):
        self.model_trained = True
        self.lbl_model.setText("Model: SVM+RF+GB" + ("+XGB" if r.get("has_xgb") else ""))
        cv_f1  = r.get("cv_score"); cv_auc = r.get("cv_auc")
        thr    = r.get("threshold", SEIZURE_THR)
        cv_f1_s  = f"{cv_f1*100:.1f}%"  if cv_f1  is not None else "N/A"
        cv_auc_s = f"{cv_auc*100:.1f}%" if cv_auc is not None else "N/A"
        self.lbl_accuracy.setText(f"Mode: ML  F1:{cv_f1_s} AUC:{cv_auc_s}")
        self.lbl_accuracy.setStyleSheet(f"color:{C_GREEN}; font-size:10px; font-weight:bold;")
        self.lbl_threshold.setText(f"Threshold: {thr:.2f}")
        msg = (
            f"TRAINING COMPLETE\n{'='*54}\n"
            f"Model: SVM + RF + GB" + (" + XGB\n" if r.get("has_xgb") else "\n") +
            f"Pipeline: RobustScaler → PCA(95%) → RFE → Ensemble\n"
            f"Features: {N_FEATURES}  |  Balanced: Yes (SMOTE-lite)\n\n"
            f"PERFORMANCE (hold-out test set):\n"
            f"  Accuracy    : {r['accuracy']*100:.1f}%\n"
            f"  Precision   : {r['precision']*100:.1f}%\n"
            f"  Recall/Sens : {r['recall']*100:.1f}%\n"
        )
        cm = r.get("conf_matrix", [[0,0],[0,0]])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        spec = tn / max(tn + fp, 1)
        msg += (
            f"  Specificity : {spec*100:.1f}%\n"
            f"  F1-Score    : {r['f1']*100:.1f}%\n"
            f"  ROC-AUC     : {r['auc']*100:.1f}%\n\n"
            f"5-FOLD CV (balanced set):\n"
            f"  F1          : {cv_f1_s}\n  AUC         : {cv_auc_s}\n\n"
            f"CONFUSION MATRIX (test):\n"
            f"  TN={tn}  FP={fp}\n  FN={fn}  TP={tp}\n\n"
            f"AUTO-TUNED THRESHOLD: {thr:.3f}\n"
            f"Dataset — Train: {r['n_train']}  Test: {r['n_test']}\n\n"
            f"TOP 10 FEATURES BY IMPORTANCE:\n"
        )
        for i, (fn2, imp) in enumerate(r.get("top_features", [])[:10], 1):
            msg += f"  {i:2d}. {fn2:22s}  {imp:.4f}\n"
        msg += f"\n{'='*54}\nLive detection active. Classify for full analysis."
        self._show_msg(msg)

    def _on_train_error(self, e):
        self._show_msg(f"Training Error:\n{e}")

    def _save_model(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "eeg_model.pkl", "Pickle (*.pkl)")
        if path:
            self.model.save(path)
            self._show_msg(f"Saved: {os.path.basename(path)}")

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Pickle (*.pkl)")
        if path:
            try:
                self.model.load(path); self.model_trained = True
                self.lbl_model.setText(f"Model: {os.path.basename(path)}")
                cv_f1_s  = (f"{self.model.cv_score*100:.1f}%"
                            if self.model.cv_score is not None else "N/A")
                cv_auc_s = (f"{self.model.cv_auc*100:.1f}%"
                            if self.model.cv_auc is not None else "N/A")
                self.lbl_accuracy.setText(f"Mode: ML  F1:{cv_f1_s} AUC:{cv_auc_s}")
                self.lbl_accuracy.setStyleSheet(
                    f"color:{C_GREEN}; font-size:10px; font-weight:bold;")
                self.lbl_threshold.setText(f"Threshold: {self.model.threshold:.2f}")
                self._show_msg(
                    f"Loaded: {os.path.basename(path)}\n"
                    f"Threshold: {self.model.threshold:.2f}\n"
                    f"CV F1: {cv_f1_s}  AUC: {cv_auc_s}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def _cur_data(self):
        return self.filtered_data if self.filtered_data is not None \
               else self.raw_data

    def _update_info(self):
        dur = (f"{self.raw_data.shape[1]/self.fs:.1f} s"
               if self.raw_data is not None else "∞ (demo)")
        active = [nm for nm, f in [("BP",self.filt_bp),("HP",self.filt_hp),
                                    ("LP",self.filt_lp),("N50",self.filt_notch),
                                    ("FIR",self.filt_fir),("CHEBY",self.filt_cheby)] if f]
        self.info_labels["Channels"].setText(str(self.n_channels))
        self.info_labels["Sample rate"].setText(f"{self.fs:.0f} Hz")
        self.info_labels["Duration"].setText(dur)
        self.info_labels["Filters"].setText(
            ", ".join(active) if active else "None")

    def _show_msg(self, text: str):
        d = QMessageBox(self); d.setWindowTitle("EEG Tool"); d.setText(text)
        d.setStyleSheet(f"""
            QMessageBox {{ background:{C_PANEL}; color:{C_TEXT}; }}
            QLabel {{ color:{C_TEXT}; }}
            QPushButton {{ background:{C_BG}; color:{C_TEXT};
                border:1px solid {C_BORDER}; border-radius:3px;
                padding:4px 16px; }}
            QPushButton:hover {{ background:{C_ACCENT}; color:#FFFFFF; }}
        """)
        d.exec_()


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    app = QApplication(sys.argv); app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(str(C_BG)))
    pal.setColor(QPalette.WindowText,      QColor(str(C_TEXT)))
    pal.setColor(QPalette.Base,            QColor(str(C_PANEL)))
    pal.setColor(QPalette.AlternateBase,   QColor("#EAE6E0"))
    pal.setColor(QPalette.Text,            QColor(str(C_TEXT)))
    pal.setColor(QPalette.Button,          QColor(str(C_PANEL)))
    pal.setColor(QPalette.ButtonText,      QColor(str(C_TEXT)))
    pal.setColor(QPalette.Highlight,       QColor(str(C_ACCENT)))
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(pal)
    win = MainWindow(); win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()