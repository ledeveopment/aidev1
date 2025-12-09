#!/bin/bash

# === BACKUP mit conda-pack ===
echo "📦 Backup der Umgebung 'fbprophet' als tar.gz wird erstellt..."
conda init 
conda activate fbprophet
conda install -y -c conda-forge conda-pack
conda-pack -n fbprophet -o fbprophet.tar.gz

echo "✅ Backup gespeichert als 'fbprophet.tar.gz'."

# === RESTORE auf anderem Rechner ===
echo "📥 Wiederherstellung auf anderem Rechner:"
echo "1. Kopiere 'fbprophet.tar.gz' auf den Zielrechner."
echo "2. Entpacke mit:"
echo "   mkdir -p ~/fbprophet"
echo "   tar -xzf fbprophet.tar.gz -C ~/fbprophet"
echo "3. Aktiviere mit:"
echo "   source ~/fbprophet/bin/activate"
