#!/bin/bash

# === BACKUP mit conda-pack ===
echo "📦 Backup der Umgebung 'autogluon' als tar.gz wird erstellt..."
conda activate autogluon
conda install -y -c conda-forge conda-pack
conda-pack -n autogluon -o autogluon.tar.gz

echo "✅ Backup gespeichert als 'autogluon.tar.gz'."

# === RESTORE auf anderem Rechner ===
echo "📥 Wiederherstellung auf anderem Rechner:"
echo "1. Kopiere 'autogluon.tar.gz' auf den Zielrechner."
echo "2. Entpacke mit:"
echo "   mkdir -p ~/autogluon"
echo "   tar -xzf autogluon.tar.gz -C ~/autogluon"
echo "3. Aktiviere mit:"
echo "   source ~/autogluon/bin/activate"
