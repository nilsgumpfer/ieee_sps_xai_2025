# From Saliency to Semantics: XAI for ECG Time Series Analysis

### Tutorial at IEEE SPS Cycle 2 Seasonal School on Explainable AI and Applications to Biometric Signal Processing, 16-20 July 2025, IIIT Allahabad, Prayagraj, India

## 📁 Structure

- `data/` – Sample data (or download scripts)
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

## ⚙️ Data Initialization

To download all necessary data,
run the following scripts inside the data folder:

```bash
   cd  data
   bash download_alexnet.sh
   bash download_ecg_models.sh
   bash download_ecgs.sh
   bash download_images.sh
   ```

## 📝 Notes

- This repository assumes basic familiarity with Python and Shell comands.
- GPU acceleration is not required but may speed up certain examples if available (you can use https://colab.google/ for that purpose).

## 📚 Citations

Citation for ECG Models:

```bibtex
@InProceedings{Gumpfer2024,
  author    = {Gumpfer, Nils and Borislav Dinov and Samuel Sossalla and Michael Guckert and Jennifer Hannig},
  booktitle = {22nd International Conference on Artificial Intelligence in Medicine, AIME 2024, Salt Lake City, UT, USA, July 9 - 12, 2024, Proceedings},
  title     = {{Towards Trustworthy {AI} in Cardiology: A Comparative Analysis of Explainable {AI} Methods for Electrocardiogram Interpretation}},
  chapter   = {36},
  doi       = {10.1007/978-3-031-66535-6_36},
  editor    = {Finkelstein, Josef and Moskovitch, Robert and Parimbelli, Enea},
  pages     = {350--361},
  publisher = {Springer Nature Switzerland AG},
  series    = {Lecture Notes in Computer Science},
  volume    = {14845},
  month     = {07},
  year      = {2024},
}
```

## 📚 License

This repository is intended for educational purposes. Content is provided under the [MIT License](LICENSE) unless otherwise noted.
