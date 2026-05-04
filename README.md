# Bulbar-Onset ALS / Dysarthria Detection — STM32F407VET6

> **Real-time speech-based motor speech disorder detection on a bare-metal ARM Cortex-M4 microcontroller — no OS, no cloud, no host-PC dependency.**

This is an ongoing embedded systems research project extending prior Parkinson's disease detection work toward bulbar-onset ALS and dysarthria — a broader class of motor speech disorders that cause progressive degradation of articulation, voicing, and speech rhythm.

---

## Project Status

**Active Development** — Feature extraction pipelines complete. GSP integration and NanoEdge AI inference integration in progress.

---

## Hardware

| Component | Details |
|---|---|
| MCU | STM32F407VET6 (ARM Cortex-M4F @ 168 MHz) |
| SRAM | 192 kB |
| Flash | 512 kB |
| Microphone | INMP441 Digital MEMS (I2S interface) |
| Display (planned) | Waveshare 3.2" 320×240 Touch LCD — ILI9341 + XPT2046, driven via FSMC 8080 16-bit parallel |

---

## What It Does

The system captures speech audio in real time, runs a dual-path feature extraction pipeline entirely on-chip, and (when complete) performs AI-based dysarthria classification — all in bare-metal embedded C.

### Signal Processing Pipeline

```
INMP441 (I2S)
      │
      ▼
 Audio Frame (1024 samples @ 16 kHz)
      │
      ├──────────────────────────────────────────────┐
      │                                              │
      ▼                                              ▼
  PATH A: GFCC                               PATH B: STFT + NNMF
  ─────────────                              ──────────────────────
  DC removal                                 Hamming window
  Hamming window                             1024-pt real FFT (CMSIS-DSP)
  24-ch ERB Cochlear Filterbank              Spectrogram (30 frames × 512 bins)
  (cascaded 2nd-order IIR resonators)        Rank-1 NMF (multiplicative updates)
  Peak-energy envelope detection             W vector → spectral basis
  Log compression                            H vector → temporal activations
  DCT-II → 13 GFCCs
      │                                              │
      └──────────────┬───────────────────────────────┘
                     │
                     ▼
           Graph Signal Processing (GSP)
           ──────────────────────────────
           Chain graphs over: freq bins, cepstral coeffs, time frames
           Graph Laplacian (L = D − A)
           Graph Smoothness (S = xᵀLx)
           Graph Fourier Transform (GFT: x_G = Uᵀx)
           Applied across: Spectrogram, NNMF W, NNMF H, GFCC matrix
                     │
                     ▼
           Feature Vector → NanoEdge AI Classifier (planned)
                     │
                     ▼
           ILI9341 Touch Display — waveform + classification output (planned)
```

---

## Why GFCC Over MFCC?

Standard MFCCs use a triangular Mel filterbank — a linear approximation of human hearing. GFCCs replace this with a **biologically accurate cochlear model**:

- **ERB-spaced (Equivalent Rectangular Bandwidth) filterbank** — matches how the basilar membrane of the human ear responds to frequency
- **Cascaded 2nd-order IIR resonators** — each channel approximates a gammatone bandpass filter
- **Peak-energy envelope detection** — captures transient speech events better than frame averaging
- **Better sensitivity to irregular voicing and consonant degradation** — the hallmarks of dysarthric speech

---

## Why Graph Signal Processing?

Frame-by-frame cepstral methods treat each feature vector independently. GSP models the **relationships between features**:

| Graph Structure | Signal | What It Captures |
|---|---|---|
| Frequency chain graph | Spectrogram frame | Spectral smoothness / irregularities |
| Time chain graph | Frame energy sequence | Rhythm stability, articulation timing |
| Joint time-frequency graph | Full spectrogram | Formant movement and dynamics |
| Frequency graph | NNMF W (spectral basis) | Vocal tract resonance stability |
| Time graph | NNMF H (temporal activations) | Speech production energy regularity |
| Cepstral chain graph | GFCC frame vector | Formant distortion, articulation patterns |

Dysarthric speech causes measurable disruptions in **all three domains** — spectral, temporal, and cepstral. GSP features quantify these disruptions as graph energy values that feed the classifier.

---

## File Structure

```
├── Core/
│   ├── Src/
│   │   ├── main.c            — system init, pipeline orchestration
│   │   ├── gfcc.c            — GFCC: cochlear filterbank → DCT
│   │   ├── cochlea.c         — ERB IIR filterbank, envelope detection
│   │   ├── mfcc.c            — reference MFCC pipeline (parallel path)
│   │   ├── spectrogram.c     — STFT spectrogram construction
│   │   └── nnmf.c            — rank-1 NMF with multiplicative updates
│   └── Inc/
│       ├── gfcc.h
│       ├── cochlea.h
│       ├── mfcc.h
│       ├── spectrogram.h
│       └── nnmf.h
└── docs/
    └── gsp_feature_table.md  — full GSP feature extraction design table
```

---

## Dependencies

- **CMSIS-DSP** (`arm_math.h`) — FFT, cos, vector ops via hardware FPU
- **STM32 HAL** — I2S, GPIO, UART
- **NanoEdge AI Studio** — KNN classifier library (to be integrated)
- **tiny-AES-c** — planned for encrypted UART feature logging

---

## Related Project

**[Parkinson's Disease Detection — STM32F401RE](link-to-other-repo)**
The predecessor project: full MFCC + rank-1 NMF pipeline + NanoEdge AI KNN inference on a Cortex-M4 @ 84 MHz, 96 kB SRAM. First known bare-metal implementation of this pipeline on a Cortex-M4 device.

---

## Author

**Rohan Kumar Singh** — B.Tech ECE, Babasaheb Bhimrao Ambedkar University, Lucknow
Embedded Systems Intern, BIT Mesra (2025)
[LinkedIn](https://linkedin.com/in/your-link) | rohankumar17362@gmail.com
