#!/bin/bash

# === BACKUP mit conda-pack ===
echo "📦 Backup der Umgebung 'tf-gpu' als tar.gz wird erstellt..."
conda activate tf-gpu
conda install -y -c conda-forge conda-pack
conda-pack -n tf-gpu -o tf-gpu.tar.gz

echo "✅ Backup gespeichert als 'tf-gpu.tar.gz'."

# === RESTORE auf anderem Rechner ===
echo "📥 Wiederherstellung auf anderem Rechner:"
echo "1. Kopiere 'tf-gpu.tar.gz' auf den Zielrechner."
echo "2. Entpacke mit:"
echo "   mkdir -p ~/tf-gpu"
echo "   tar -xzf tf-gpu.tar.gz -C ~/tf-gpu"
echo "3. Aktiviere mit:"
echo "   source ~/tf-gpu/bin/activate"
