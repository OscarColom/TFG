import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Rutas de las carpetas
base_path = os.path.dirname(os.path.abspath(__file__))

# train_path = os.path.join(base_path, "DSA-Self-real-L1A-dataset/train")
# val_path = os.path.join(base_path, "vaDSA-Self-real-L1A-dataset/val")

train_path = "/data/upftfg04/ocolom/DSA-Self-real-L1A-dataset/train"
val_path = "/data/upftfg04/ocolom/DSA-Self-real-L1A-dataset/val"

# Crear la carpeta de validación si no existe
os.makedirs(val_path, exist_ok=True)

# Obtener la lista de archivos en la carpeta de entrenamiento
all_files = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]

# Separar los archivos en entrenamiento y validación (90% - 10%)
train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=42)

# Mover los archivos de validación a la carpeta de validación
for file in val_files:
    shutil.move(os.path.join(train_path, file), os.path.join(val_path, file))

print(f"Se han movido {len(val_files)} archivos a la carpeta de validación.")