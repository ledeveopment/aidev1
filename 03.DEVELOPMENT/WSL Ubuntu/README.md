# SETUP WSL for Tensorflow GPU 
User: aiubuntu01
Password: LamKhue?2

### ✅ **1. Voraussetzungen**

*   **WSL2** aktiviert und Ubuntu installiert.
*   **NVIDIA-Treiber auf Windows** mit WSL-Unterstützung (kein CUDA Toolkit auf Windows nötig).
*   Prüfe GPU-Zugriff in WSL:
	```bash
	nvidia-smi
	```
	→ Wenn die GPU angezeigt wird, ist alles korrekt.

***

### ✅ **2. Anaconda installieren (falls nicht vorhanden)**

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-Linux-x86_64.sh
bash Anaconda3-2024.06-Linux-x86_64.sh
source ~/.bashrc
conda --version
```

***

### ✅ **3. Neue Umgebung erstellen**

```bash
conda create --name tf-gpu python=3.12
conda activate tf-gpu
```

***

### ✅ **4. CUDA 12.3 und cuDNN installieren**

Für TensorFlow GPU mit CUDA 12.3:

```bash
conda install -c conda-forge cudatoolkit=12.3 cudnn=8.9
```

Setze die Umgebungsvariablen:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

***

### ✅ **5. TensorFlow GPU installieren**

TensorFlow 2.15 unterstützt CUDA 11.8 offiziell, aber für CUDA 12.3 musst du die neue Variante nutzen:

```bash
pip install tensorflow[and-cuda]
```

Diese Version installiert automatisch die passenden CUDA-Bibliotheken für TensorFlow (ab TF 2.15).

***

### ✅ **6. Testen**

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Wenn die GPU angezeigt wird, ist alles korrekt.

***

#### 🔍 **Hinweise**

*   Mit `tensorflow[and-cuda]` brauchst du keine manuelle CUDA/cuDNN-Installation, aber wenn du explizit CUDA 12.3 nutzen willst, ist der obige Conda-Weg richtig.
*   Prüfe Kompatibilität: `nvcc --version` und `nvidia-smi`.
*   Für maximale Performance: `conda install -c conda-forge mamba` und dann Pakete mit `mamba` installieren.

***
