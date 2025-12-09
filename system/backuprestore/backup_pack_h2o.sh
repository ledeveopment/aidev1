#!/bin/bash

# === BACKUP mit conda-pack ===
echo "📦 Backup der Umgebung 'h2o' als tar.gz wird erstellt..."
conda activate h2o
conda install -y -c conda-forge conda-pack
conda-pack -n h2o -o h2o.tar.gz

echo "✅ Backup gespeichert als 'h2o.tar.gz'."

# === RESTORE auf anderem Rechner ===
echo "📥 Wiederherstellung auf anderem Rechner:"
echo "1. Kopiere 'h2o.tar.gz' auf den Zielrechner."
echo "2. Entpacke mit:"
echo "   mkdir -p ~/h2o"
echo "   tar -xzf h2o.tar.gz -C ~/h2o"
echo "3. Aktiviere mit:"
echo "   source ~/h2o/bin/activate"
