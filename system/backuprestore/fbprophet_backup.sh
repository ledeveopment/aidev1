#!/bin/bash

# === BACKUP ===
echo "🔄 Backup der Umgebung 'fbprophet' wird erstellt..."
conda activate fbprophet
conda env export > fbprophet_env.yml
echo "✅ Backup gespeichert als 'fbprophet_env.yml'."

