import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def extract_identifier(filename, prefix):
    """ Extrae la parte común del nombre del archivo después de 'output_' o 'target_'. """
    return filename[len(prefix):] if filename.startswith(prefix) else None

def visualize_tiff_pairs(folder_path):
    """
    Visualiza automáticamente los pares de imágenes TIFF (output, target) en la carpeta dada.
    
    Parámetros:
        folder_path (str): Ruta de la carpeta que contiene las imágenes TIFF.
    """
    # Obtener lista de archivos TIFF en la carpeta
    tiff_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".tiff")])

    # Diccionario para almacenar pares {identificador: (output_path, target_path)}
    image_pairs = {}

    # Recorrer archivos y emparejarlos por su identificador
    for filename in tiff_files:
        if filename.startswith("output_"):
            identifier = extract_identifier(filename, "output_")
            if identifier:
                image_pairs.setdefault(identifier, [None, None])[0] = os.path.join(folder_path, filename)
        elif filename.startswith("target_"):
            identifier = extract_identifier(filename, "target_")
            if identifier:
                image_pairs.setdefault(identifier, [None, None])[1] = os.path.join(folder_path, filename)

    # Filtrar solo los pares completos
    image_pairs = {k: v for k, v in image_pairs.items() if v[0] and v[1]}
    image_keys = sorted(image_pairs.keys())  # Lista ordenada de identificadores

    if not image_pairs:
        print("No se encontraron pares de imágenes TIFF en la carpeta.")
        return
    
    num_pairs = len(image_keys)
    current_pair_idx = 0

    def show_pair(index):
        """Muestra el par de imágenes correspondiente al índice dado."""
        identifier = image_keys[index]
        output_path, target_path = image_pairs[identifier]

        output_image_np = tifffile.imread(output_path)
        target_image_np = tifffile.imread(target_path)

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(output_image_np, cmap='gray')
        plt.title(f"Output")

        plt.subplot(1, 2, 2)
        plt.imshow(target_image_np, cmap='gray')
        plt.title(f"Target:")

        plt.suptitle(f"Par {index + 1} de {num_pairs}")
        plt.draw()

    # Mostrar el primer par
    plt.figure(figsize=(10, 5))
    show_pair(current_pair_idx)

    def on_key(event):
        nonlocal current_pair_idx
        if event.key == 'right' and current_pair_idx < num_pairs - 1:
            current_pair_idx += 1
        elif event.key == 'left' and current_pair_idx > 0:
            current_pair_idx -= 1
        else:
            return
        show_pair(current_pair_idx)

    plt.connect('key_press_event', on_key)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualizar pares de imágenes TIFF en una carpeta.")
    parser.add_argument("folder_path", type=str, help="Ruta de la carpeta con imágenes TIFF.")
    args = parser.parse_args()
    visualize_tiff_pairs(args.folder_path)
