# PERFORMANCE TUNNING Tensorflow GPU  ******* 
# ------

import tensorflow as tf
import time
from tensorflow.keras import mixed_precision

# GPU-Optimierungen aktivieren
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("✅ Memory Growth aktiviert")

mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed Precision aktiviert")

tf.config.optimizer.set_jit(True)
print("✅ XLA aktiviert")

# Benchmark-Test
print("\n🚀 Starte GPU-Benchmark...")
with tf.device('/GPU:0'):
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    start = time.time()
    c = tf.matmul(a, b)
    # Synchronisation erfolgt automatisch, wenn wir das Ergebnis auswerten
    _ = c.numpy()  # zwingt TensorFlow, die Berechnung abzuschließen
    print("✅ Dauer für 10k x 10k Matrix-Multiplikation:", round(time.time() - start, 2), "Sekunden")