# 👁️ VisionTag - Enterprise YOLOv8 OBB Annotator

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-lightgrey.svg)
![YOLOv8](https://img.shields.io/badge/AI-Ultralytics_YOLOv8-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**VisionTag** is a professional-grade, locally-hosted desktop application designed for annotating complex, overlapping objects in images (specifically optimized for X-Ray imagery) using **Oriented Bounding Boxes (OBB)**. 

Built with Python and Tkinter, it provides an enterprise-level workflow for Data Operations and Machine Learning Engineers to prepare high-quality datasets for training object detection models.

---

## ✨ Key Features

* **🪄 AI Auto-Detect (Pre-labeling):** Integrated with Ultralytics YOLOv8. Load your trained `.pt` model and let the AI predict and draw the OBBs automatically. You only need to review and adjust the generated bounding boxes!
* **🎛️ Data Augmentation Generator:** Multiply your dataset instantly. Automatically generate 90°, 180°, and 270° rotated copies of your images along with precisely recalculated OBB coordinates.
* **📊 Analytics Dashboard:** A real-time, interactive bar chart showing your dataset distribution to help you identify and prevent data imbalance before training your models.
* **📂 Bulk Actions & Synchronization:** Rename classes across thousands of `.txt` and `.json` files instantly without breaking the dataset directory structure.
* **🔄 Enterprise QoL (Quality of Life):**
  * **Z-Order Management:** Click on overlapping boxes to dynamically bring them to the front.
  * **Unlimited Undo/Redo:** Full state-history stack (`Ctrl+Z` / `Ctrl+Y`) for seamless error corrections.
  * **Dynamic Neon Hashing:** Auto-generates high-contrast colors for each unique class using MD5 hashing.
  * **Modern HUD:** Real-time fading toast notifications and image parameter overlays (Zoom, Brightness, Contrast).

---

## 📁 Directory Structure

The project follows a clean, maintainable, and modular architecture:

```bash
VisionTag_Project/
│
├── utils/
│   ├── __init__.py          # Marks the directory as a Python package
│   ├── geometry.py          # Core math logic for OBB translations and rotations
│   └── theme.py             # UI utilities, Dynamic Color Generator, and Fade Animations
│
├── main.py                  # Main application entry point and Tkinter UI controller
├── requirements.txt         # Project dependencies
├── .gitignore               # Ignored files (Cache, Large AI Models, Local Datasets)
└── README.md                # Project documentation
```

## 🛠️ Installation & Setup
**Clone the repository**:
```Bash
git clone [https://github.com/yourusername/VisionTag.git](https://github.com/yourusername/VisionTag.git)
cd VisionTag
```
**Install dependencies**:
It is highly recommended to use a virtual environment.
```Bash
pip install -r requirements.txt
```
**Run the application**:
```Bash
python main.py
```

## 🎮 Keyboard & Mouse Controls

| Action | Control / Shortcut |
| :--- | :--- |
| **Draw Box** | `Left-Click` (Empty space) & Drag |
| **Move Box** | `Left-Click` (Center of box) & Drag |
| **Resize Box** | `Left-Click` (Edge of box) & Drag |
| **Bring Box to Front** | `Left-Click` on any overlapping box |
| **Rotate Box** | `Ctrl` + `Mouse Scroll` |
| **Delete Specific Box** | `Shift` + `Left-Click` on Box |
| **Pan Image** | `Right-Click` & Drag |
| **Zoom In/Out** | `Mouse Scroll` |
| **Undo / Redo** | `Ctrl + Z` / `Ctrl + Y` |
| **Copy / Paste Box** | `Ctrl + C` / `Ctrl + V` |
| **Adjust Brightness** | `-` / `=` or `+` |
| **Adjust Contrast** | `[` / `]` |
| **Toggle Visibility** | `V` |
| **Focus Search Bar** | `Ctrl + F` |
| **Confirm & Next Image**| `Enter` |
| **Next / Prev Image** | `Arrow Right` / `Arrow Left` |

---

## 📁 Output Format

VisionTag natively exports annotations in the standard **YOLOv8 OBB format** using normalized coordinates (values between 0.0 and 1.0). The output text file will be structured as follows:

```text
[class_id] [x1] [y1] [x2] [y2] [x3] [y3] [x4] [y4]
```

Note: The application automatically handles the conversion from display pixels back to normalized image resolution ratios upon exporting.
