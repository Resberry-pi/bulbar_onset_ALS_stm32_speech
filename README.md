<p align="center">
  <img src="https://img.shields.io/badge/MCU-STM32F407VET6-03234B?style=for-the-badge&logo=stmicroelectronics&logoColor=white" alt="MCU"/>
  <img src="https://img.shields.io/badge/Core-ARM%20Cortex--M4F-00979D?style=for-the-badge" alt="Core"/>
  <img src="https://img.shields.io/badge/DSP-CMSIS--DSP-blue?style=for-the-badge" alt="DSP"/>
  <img src="https://img.shields.io/badge/AI-NanoEdge%20AI-8A2BE2?style=for-the-badge" alt="AI"/>
  <img src="https://img.shields.io/badge/Language-Embedded%20C-yellow?style=for-the-badge" alt="Lang"/>
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge" alt="Status"/>
</p>

<h1 align="center">Bulbar-Onset ALS / Dysarthria Detection</h1>
<h3 align="center">STM32F407VET6 — Bare-Metal Real-Time Motor Speech Analysis</h3>

<p align="center">
  <em>Real-time speech-based motor speech disorder detection on a bare-metal ARM Cortex-M4 microcontroller — no OS, no cloud, no host-PC dependency.</em>
</p>

---

## Overview

This is an ongoing embedded systems research project extending prior Parkinson's disease detection work toward bulbar-onset ALS and dysarthria — a broader class of motor speech disorders that cause progressive degradation of articulation, voicing, and speech rhythm.

The system captures speech audio in real time, runs a dual-path feature extraction pipeline entirely on-chip, and performs AI-based dysarthria classification — all in bare-metal embedded C.

---

## Hardware

| Component | Details |
|---|---|
| **MCU** | STM32F407VET6 — ARM Cortex-M4F @ 168 MHz |
| **SRAM** | 192 kB |
| **Flash** | 512 kB |
| **Microphone** | INMP441 Digital MEMS — I2S interface |
| **Display** (planned) | Waveshare 3.2" 320x240 Touch LCD — ILI9341 + XPT2046, FSMC 8080 16-bit parallel |

<!--
  TIP: Add a hardware photo here once assembled.
  ![Hardware Setup](docs/images/hardware_setup.jpg)
-->

---

## Signal Processing Pipeline

```
INMP441 (I2S)
      |
      v
 Audio Frame (1024 samples @ 16 kHz)
      |
      +----------------------------------------------+
      |                                              |
      v                                              v
  PATH A: GFCC                               PATH B: STFT + NNMF
  ------------                               --------------------
  DC removal                                 Hamming window
  Hamming window                             1024-pt real FFT (CMSIS-DSP)
  24-ch ERB Cochlear Filterbank              Spectrogram (30 frames x 512 bins)
  (cascaded 2nd-order IIR resonators)        Rank-1 NMF (multiplicative updates)
  Peak-energy envelope detection             W vector -- spectral basis
  Log compression                            H vector -- temporal activations
  DCT-II -> 13 GFCCs
      |                                              |
      +-----------------+----------------------------+
                        |
                        v
           Graph Signal Processing (GSP)
           ------------------------------
           Chain graphs over: freq bins, cepstral coeffs, time frames
           Graph Laplacian  (L = D - A)
           Graph Smoothness (S = x^T L x)
           Graph Fourier Transform (GFT: x_G = U^T x)
           Applied across: Spectrogram, NNMF W, NNMF H, GFCC matrix
                        |
                        v
           Feature Vector -> NanoEdge AI Classifier (planned)
                        |
                        v
           ILI9341 Touch Display -- waveform + classification output (planned)
```

---

## Why GFCC Over MFCC?

Standard MFCCs use a triangular Mel filterbank — a linear approximation of human hearing. GFCCs replace this with a biologically accurate cochlear model:

| MFCC | GFCC |
|---|---|
| Triangular Mel filterbank | ERB-spaced cochlear filterbank |
| Linear frequency approximation | Matches basilar membrane response |
| Frame energy averaging | Peak-energy envelope detection |
| Weaker on consonants | Better sensitivity to consonant degradation |

- **ERB-spaced filterbank** — matches how the basilar membrane of the human ear responds to frequency
- **Cascaded 2nd-order IIR resonators** — each channel approximates a gammatone bandpass filter
- **Peak-energy envelope detection** — captures transient speech events missed by frame averaging
- **Better sensitivity to irregular voicing and consonant degradation** — the hallmarks of dysarthric speech

---

## Why Graph Signal Processing?

Frame-by-frame cepstral methods treat each feature vector independently. GSP models the relationships between features across time, frequency, and cepstral dimensions simultaneously.

| Graph Structure | Signal | What It Captures |
|---|---|---|
| Frequency chain graph | Spectrogram frame | Spectral smoothness / irregularities |
| Time chain graph | Frame energy sequence | Rhythm stability, articulation timing |
| Joint time-frequency graph | Full spectrogram | Formant movement and dynamics |
| Frequency graph | NNMF W — spectral basis | Vocal tract resonance stability |
| Time graph | NNMF H — temporal activations | Speech production energy regularity |
| Cepstral chain graph | GFCC frame vector | Formant distortion, articulation patterns |

Dysarthric speech causes measurable disruptions in all three domains — spectral, temporal, and cepstral. GSP features quantify these disruptions as graph energy values that feed the classifier.

---

## Project Status

**Active Development** — Feature extraction pipelines complete. GSP integration and NanoEdge AI inference integration in progress.

| Module | Status |
|---|---|
| I2S audio capture (INMP441) | Complete |
| GFCC — cochlear filterbank + DCT | Complete |
| STFT spectrogram | Complete |
| Rank-1 NNMF | Complete |
| Graph Signal Processing (GSP) | In Progress |
| NanoEdge AI classifier integration | Planned |
| ILI9341 display output | Planned |

---

## File Structure

```
.
|-- Core/
|   |-- Src/
|   |   |-- main.c            -- system init, pipeline orchestration
|   |   |-- gfcc.c            -- GFCC: cochlear filterbank -> DCT
|   |   |-- cochlea.c         -- ERB IIR filterbank, envelope detection
|   |   |-- mfcc.c            -- reference MFCC pipeline (parallel path)
|   |   |-- spectrogram.c     -- STFT spectrogram construction
|   |   +-- nnmf.c            -- rank-1 NMF with multiplicative updates
|   +-- Inc/
|       |-- gfcc.h
|       |-- cochlea.h
|       |-- mfcc.h
|       |-- spectrogram.h
|       +-- nnmf.h
+-- docs/
    +-- gsp_feature_table.md  -- full GSP feature extraction design table
```

---

## Dependencies

| Library | Purpose |
|---|---|
| **CMSIS-DSP** (`arm_math.h`) | FFT, cosine, vector ops via hardware FPU |
| **STM32 HAL** | I2S, GPIO, UART peripheral drivers |
| **NanoEdge AI Studio** | KNN classifier library (to be integrated) |
| **tiny-AES-c** | Planned — encrypted UART feature logging |

---

## Related Project

**[Parkinson's Disease Detection — STM32F401RE](link-to-other-repo)**

The predecessor project: full MFCC + rank-1 NMF pipeline + NanoEdge AI KNN inference on a Cortex-M4 @ 84 MHz, 96 kB SRAM. First known bare-metal implementation of this pipeline on a Cortex-M4 device.

---

## Author

**Rohan Kumar Singh**
B.Tech ECE — Babasaheb Bhimrao Ambedkar University, Lucknow
Embedded Systems Intern — BIT Mesra (2025)

<p>
  <a href="https://linkedin.com/in/YOUR_LINKEDIN_ID">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin" alt="LinkedIn"/>
  </a>
  &nbsp;
  <a href="mailto:rohankumar17362@gmail.com">
    <img src="https://img.shields.io/badge/Email-rohankumar17362@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/>
  </a>
</p>
