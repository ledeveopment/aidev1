#!/bin/bash

# === BACKUP mit conda-pack ===
echo "📦 Backup der Umgebung 'py3.8-news' als tar.gz wird erstellt..."
conda activate py3.8-news
conda install -y -c conda-forge conda-pack
conda-pack -n py3.8-news -o py3.8-news.tar.gz

echo "✅ Backup gespeichert als 'py3.8-news.tar.gz'."

# === RESTORE auf anderem Rechner ===
echo "📥 Wiederherstellung auf anderem Rechner:"
echo "1. Kopiere 'py3.8-news.tar.gz' auf den Zielrechner."
echo "2. Entpacke mit:"
echo "   mkdir -p ~/py3.8-news"
echo "   tar -xzf py3.8-news.tar.gz -C ~/py3.8-news"
echo "3. Aktiviere mit:"
echo "   source ~/py3.8-news/bin/activate"
