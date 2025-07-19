# Cell Instance Segmentation and Morphometric Analysis with YOLO11

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![YOLO11](https://img.shields.io/badge/YOLO11-ultralytics-red) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Status](https://img.shields.io/badge/status-Deployed-brightgreen)

<p align="center">
  <img src="assets/gradio_app_screenshot.png" alt="Interactive Application" width="800"/>
</p>

## 🚀 Interactive Application

👉 **[🔬 Access the Live Application Here](https://huggingface.co/spaces/alicefvictorino/cell-segmentation-app)** 🚀

---

## 📑 Table of Contents
- [Objective](#-objective)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Results and Analysis](#-results-and-analysis)
- [Technologies Used](#️-technologies-used)
- [About the Dataset](#-about-the-dataset)
- [License](#-license)
- [References](#-references)

---

## 🎯 Objective

Manual analysis of microscopy images for cell identification and quantification is a **slow, subjective, and error-prone** process. This project offers an automated solution that aims to **automatize** individual cell detection and segmentation, and **extracts quantitative metrics** from images, such as area and perimeter (in pixels), circularity and elongation.

---

## 🚀 How to Run

This project includes a reproducible workflow using a single Jupyter Notebook. This is the **recommended way to run the entire pipeline**, from data download to model training.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alicefvictorino/cell-instance-segmentation/blob/main/main_workflow_kaggle.ipynb)

**Click the button above to open the `main_workflow_kaggle.ipynb` notebook in Google Colab.**

This automated workflow will:
1.  **Clone the repository** and install all required dependencies.
2.  **Handle Kaggle API authentication** to securely access the dataset.
3.  **Download and preprocess the data** directly from Kaggle.
4.  **Train the `yolo11s-seg` model** for 50 epochs.
5.  **Save the final model** and all training results to your personal Google Drive for persistence.

---

## 📂 Project Structure

```
cell-instance-segmentation/
├── 📁 app/
│   ├── app.py          # Main script for the Gradio web interface.
│   └── best.pt         # Trained model used by the Gradio application.
├── 📁 assets/
│   └── ...             # Images and visual assets for the README.
├── 📁 scripts/
│   ├── preprocess.py   # Converts RLE annotations to YOLO format.
│   ├── train.py        # Trains the yolo11 model.
│   └── analysis.py     # Performs morphometric analysis on segmented cells.
├── main_workflow_kaggle.ipynb  # The reproducible workflow for the pipeline.
├── requirements.txt      # Python dependencies for the project.
└── README.md      
└── LICENSE       
```

---

## 📊 Results and Analysis

To validate the model's effectiveness, an iterative training process was conducted. An initial baseline model was trained for 25 epochs, followed by a refined model trained for 100 epochs on a properly split dataset (80% train, 20% validation). The comparison clearly demonstrates the significant impact of longer, more structured training.

### Model Performance Comparison

The evaluation on the validation set shows a substantial improvement in the refined model across all key metrics.

| Metric          | Baseline Model (25 Epochs) | Refined Model (100 Epochs) | Improvement |
| :-------------- | :------------------------- | :------------------------- | :---------- |
| **mAP50-95(M)** | 0.17                       | **0.28**                   | **+65%**    |
| **mAP50(M)**    | 0.52                       | **0.62**                   | **+19%**    |

-   **mAP50-95(M):** This metric, which evaluates the precise quality of the mask overlap, saw a **65% increase**, moving from 0.17 to 0.28. This indicates that the refined model's masks are significantly more accurate.
-   **mAP50(M):** With a more flexible overlap criterion, the refined model correctly segments over 62% of the cells, a 10-point improvement.

The training graphs below illustrate the stable learning process of the 100-epoch model, with consistently decreasing loss curves and ascending mAP curves.

<p align="center">
  <img src="assets/results_100_epochs.png" alt="Training Graphs - 100 Epochs" width="900"/>
  <br>
  <em>Loss and precision graphs during the 100-epoch training run.</em>
</p>

### Qualitative and Classification Analysis

A comparison of the confusion matrices reveals the most significant improvement. The refined model is more effective at correctly identifying cell types and reduces the number of missed cells (false negatives classified as `background`).

| Baseline Model (25 Epochs)                                                                      | Refined Model (100 Epochs)                                                                        |
| :----------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| <img src="assets/confusion_matrix_25_epochs.png" alt="Confusion Matrix - 25 Epochs" width="400"/> | <img src="assets/confusion_matrix_100_epochs.png" alt="Confusion Matrix - 100 Epochs" width="400"/> |

-   **Reduction in Missed Detections:** The number of true cells misclassified as `background` dropped from **33,594** in the 25-epoch model to just **6,607** in the 100-epoch model. This is a **~80% reduction in false negatives**, proving the refined model is vastly superior at finding cells.
-   **Improved Classification Accuracy:** The diagonal of the 100-epoch matrix is much "cleaner," indicating fewer mistakes between cell types once they are detected.

---

## 🛠️ Technologies Used

- **YOLO11-seg** - Instance segmentation model
- **PyTorch** - Deep learning framework
- **Ultralytics** - YOLO library
- **OpenCV** - Image processing
- **Pandas & NumPy** - Data analysis
- **Matplotlib** - Visualization
- **Gradio** - Interactive web interface
- **Hugging Face Spaces** - Deployment platform
- **Google Colab** - Training environment

---

## 📚 About the Dataset

This project uses the **Sartorius Cell Instance Segmentation Dataset**, available on Kaggle, which contains:
- 🔬 **High-quality microscopy images**
- 🏷️ **RLE annotations** for instance segmentation
- 🧬 **Multiple neurological cell types**
- 📊 **Well-structured training and test data**

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## References

-   **Sartorius - Cell Instance Segmentation (Kaggle Competition):**
    Howard, A., Chow, A., et al. (2021). *Sartorius - Cell Instance Segmentation*. Kaggle. Retrieved from https://kaggle.com/competitions/sartorius-cell-instance-segmentation
-   **Ultralytics yolo11 Documentation:**
    -   Models: https://docs.ultralytics.com/pt/models/yolo11/
    -   Segmentation Task: https://docs.ultralytics.com/pt/tasks/segment/#models





