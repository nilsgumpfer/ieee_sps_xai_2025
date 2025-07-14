# From Saliency to Semantics: XAI for ECG Time Series Analysis

### Tutorial at IEEE SPS Cycle 2 Seasonal School on Explainable AI and Applications to Biometric Signal Processing, 16-20 July 2025, IIIT Allahabad, Prayagraj, India

## 📁 Structure

- `data/` – Sample datasets (or download scripts)
- `plots/` – Directory where plots will be saved
- `scripts/` – Python scripts for demonstrations or assignments
- `requirements.txt` – List of required Python packages

## ⚙️ Setup Instructions

You can set up your environment using either `venv` (standard Python virtual environments) or `conda/miniconda`. Choose one of the following:

---


### 🐍 Option 1: Using `venv` (Python 3.10 recommended)

1. **Create virtual environment**:
   ```bash
   python3 -m venv ml-env
   ```

2. **Activate the environment**:
   - On macOS/Linux:
     ```bash
     source ml-env/bin/activate
     ```
   - On Windows:
     ```bash
     .\ml-env\Scripts\activate
     ```

3. **Install requirements**:
   ```bash
   pip3 install --upgrade pip
   pip3 install -r requirements.txt
   ```
   
---

### 🧪 Option 2: Using `conda` / `miniconda`

1. **Create a new conda environment**:
   ```bash
   conda create -n ml-env python=3.10
   ```

2. **Activate the environment**:
   ```bash
   conda activate ml-env
   ```

3. **Install requirements**:
   ```bash
   pip3 install -r requirements.txt
   ```

## 📝 Notes

- This repository assumes basic familiarity with Python and Shell comands.
- GPU acceleration is not required but may speed up certain examples if available (you can use https://colab.google/ for that purpose).

---

## 📚 License

This repository is intended for educational purposes. Content is provided under the [MIT License](LICENSE) unless otherwise noted.
