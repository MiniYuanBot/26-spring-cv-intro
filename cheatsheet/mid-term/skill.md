# Skill: Notes-to-Cheatsheet Multi-File Compiler

## Role

You are a dense academic cheatsheet compiler. You read the user's lecture notes from `./cheatsheet/mid-term/tex-notes` and emit **7 separate `.tex` files** into `./cheatsheet/mid-term/source/`. These files are `\input` into the pre-existing `main.tex` which already provides preamble, `multicols*{4}`, and document structure.

## Input

- **Source notes**: `./cheatsheet/mid-term/tex-notes`
  - If this path is a file, read it as the single consolidated notes document.
  - If this path is a directory, read all `.tex`, `.md`, `.txt`, `.pdf` files inside and merge them in lexicographic order as the source.
  - If the path does not exist, abort with: `Error: Input path ./cheatsheet/mid-term/tex-notes not found.`

## Output

Write exactly these 7 files to `./cheatsheet/mid-term/source/`, **overwriting any existing files**:

1. `01-edge-detection.tex`
2. `02-classical-feature-extraction.tex`
3. `03-neural-network-basics.tex`
4. `04-convolutional-neural-networks.tex`
5. `05-semantic-segmentation.tex`
6. `06-object-detection.tex`
7. `07-instance-segmentation.tex`

## Global Content Rules

- **No preamble**, no `\documentclass`, no `\begin{document}`, no `\end{document}`, no `multicols*`.
- Each file starts with its `\section{...}` and contains only `\subsection{}`, `\subsubsection*{}`, and compressed body content.
- Every `\subsubsection*` must contain at least one formula, table, or sentence. Never leave empty.
- Compress to atomic units only:
  - Inline math `$...$`
  - `\[...\]` for critical equations (max 2 per subsubsection)
  - `tabular` for comparisons (max 4 rows)
  - `enumerate` with `\tightlist` for algorithms (max 4 steps)
  - One-sentence definitions
- No paragraphs. No `&` in titles. No `\paragraph`. No figures/images.
- Preserve formulas exactly as in source notes.

## Immutable Structure per File

### `01-edge-detection.tex`

```latex
\section{Edge Detection}
\subsection{Conceptions}
\subsubsection*{Image as Discrete Function}
\subsubsection*{LTI Systems}
\subsubsection*{Convolution vs Cross-Correlation}
\subsection{Frequency Domain}
\subsubsection*{Fourier Transform}
\subsubsection*{Gaussian Filtering}
\subsection{Edge Detection}
\subsubsection*{Edge Gradients}
\subsubsection*{Finite Difference and Derivative Theorem}
\subsubsection*{Canny Detector}
```

### `02-classical-feature-extraction.tex`

```latex
\section{Classical Feature Extraction}
\subsection{Line Fitting}
\subsubsection*{Least Squares}
\subsubsection*{SVD Fitting}
\subsubsection*{RANSAC}
\subsection{Corner Detection}
\subsubsection*{Harris Response}
\subsubsection*{Harris Pipeline}
\subsubsection*{Geometric Properties}
\subsection{Classical Recognition Pipeline}
\subsubsection*{Feature Detection}
\subsubsection*{Bag-of-Visual-Words}
\subsubsection*{Limits of Classical CV}
```

### `03-neural-network-basics.tex`

```latex
\section{ML and Neural Network Basics}
\subsection{Problem Setup}
\subsubsection*{Task Formulation}
\subsubsection*{Data Pipeline}
\subsection{Models}
\subsubsection*{Logistic Regression}
\subsubsection*{MLP}
\subsection{Optimization}
\subsubsection*{Loss Functions}
\subsubsection*{Backpropagation}
\subsubsection*{Optimizers}
\subsubsection*{Learning Rate}
\subsection{Evaluation Metrics}
```

### `04-convolutional-neural-networks.tex`

```latex
\section{Convolutional Neural Networks}
\subsection{CNN Core Mechanisms}
\subsubsection*{Conv Layer}
\subsubsection*{Padding and Stride}
\subsubsection*{Pooling}
\subsubsection*{CNN vs FC}
\subsection{CNN Training Practice}
\subsubsection*{Data Preprocessing}
\subsubsection*{Weight Initialization}
\subsubsection*{Batch Training}
\subsection{Classification Backbones}
\subsubsection*{Receptive Field}
\subsubsection*{VGG}
\subsubsection*{ResNet}
\subsubsection*{Beyond ResNet}
```

### `05-semantic-segmentation.tex`

```latex
\section{Semantic Segmentation}
\subsection{Encoder-Decoder}
\subsubsection*{FCN Encoder}
\subsubsection*{Non-learnable Upsampling}
\subsubsection*{Transposed Convolution}
\subsection{UNet}
\subsubsection*{Skip Connections}
\subsubsection*{Information Split}
\subsection{Segmentation Evaluation}
\subsubsection*{Overlap Metrics}
\subsubsection*{Soft IoU Loss}
```

### `06-object-detection.tex`

```latex
\section{Object Detection}
\subsection{Basics}
\subsubsection*{Task Definition}
\subsubsection*{Regression Losses}
\subsection{Two-Stage Detectors}
\subsubsection*{R-CNN to Fast R-CNN}
\subsubsection*{Faster R-CNN}
\subsubsection*{RoI Generation}
\subsection{Single-Stage Detectors}
\subsubsection*{Core Mechanism}
\subsubsection*{Training Design}
\subsection{Post-processing and Evaluation}
\subsubsection*{Non-Maximum Suppression}
\subsubsection*{Detection Metrics}
```

### `07-instance-segmentation.tex`

```latex
\section{Instance Segmentation}
\subsection{Mask R-CNN Pipeline}
\subsubsection*{Three-Head Architecture}
\subsubsection*{RoIAlign}
\subsection{Mask Head Design}
\subsubsection*{Output Strategy}
\subsubsection*{Channel Strategy}
\subsubsection*{Loss}
```

## Extraction Mapping

Map content by meaning, not keyword:

- **File 01**: image as function, LTI, convolution/cross-correlation, Fourier, Gaussian, gradients, finite difference, derivative theorem, Canny (NMS, hysteresis).
- **File 02**: least squares, SVD fitting, RANSAC, Harris response/pipeline, equivariance/invariance, SIFT, BoVW, classical pipeline limits.
- **File 03**: task formulation, data pipeline (train/val/test), logistic regression, MLP, activations, loss functions, backpropagation, optimizers, LR scheduling, accuracy/precision/recall/F1.
- **File 04**: conv layer arithmetic, padding/stride, pooling, CNN vs FC, preprocessing, weight initialization (Xavier/He), batch training, receptive field recursion, VGG, ResNet (Basic/Bottleneck), MobileNet/SENet/DenseNet.
- **File 05**: FCN encoder, bilinear/nearest/max-unpooling, transposed convolution (checkerboard), UNet skip connections, IoU/mIoU, Soft IoU Loss.
- **File 06**: bbox, IoU, regression losses (Smooth L1), R-CNN→Fast→Faster R-CNN, RPN, RoI generation, YOLO/SSD, NMS/Soft-NMS, AP/mAP.
- **File 07**: Mask R-CNN three-head, RoIAlign, mask head output strategy (Sigmoid), channel strategy (class-specific K), per-pixel binary CE loss.

If a `\subsubsection*` has no direct source material, write a **single defining sentence or core formula** for it.

## Space Budget (4 columns × 2 pages)

The 7 files flow sequentially through `multicols*{4}`:

| Page | Column | Files / Sections | Budget |
| ---- | ------ | ---------------- | ------ |
| P1 | C1 | File 01 (Edge) | 25% |
| P1 | C2 | File 02 (Classical) | 20% |
| P1 | C3 | File 03 (ML Basics) | 22% |
| P1 | C4 | File 04 pt.1 (CNN Core/Training) | 22% |
| P2 | C5 | File 04 pt.2 (Backbones) | 20% |
| P2 | C6 | File 05 (Seg) | 20% |
| P2 | C7 | File 06 (Detection) | 25% |
| P2 | C8 | File 07 (Instance) | 15% |

**Overflow handling**:

1. Delete explanatory prose first; keep formulas and tables.
2. Merge `enumerate` into semicolon-separated inline lists.
3. If `multicols*` misplaces a section across columns, insert `\vfill\null\columnbreak` at the end of the preceding file.

## Prohibited

- Adding/removing/renaming any section, subsection, or subsubsection.
- Using `&` in any title.
- Using `\paragraph` or `\subparagraph`.
- Including figures/images.
- Changing font sizes inside body text.
- Writing any content outside the `\section...\subsubsection*` hierarchy.

## Execution Order

1. Read `./cheatsheet/mid-term/tex-notes`.
2. Extract formulas, definitions, algorithms, and comparison tables.
3. Assign each unit to the correct file and `\subsubsection*`.
4. Write all 7 files to `./cheatsheet/mid-term/source/`, overwriting existing files.
5. Report the written file paths. Do not ask questions.
