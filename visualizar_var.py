import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def extract_identifier(filename, prefix):
    """ Extrae la parte común del nombre del archivo después del prefijo especificado. """
    return filename[len(prefix):] if filename.startswith(prefix) else None

def visualize_tiff_sets(folder_path):
    """
    Visualiza automáticamente los conjuntos de imágenes TIFF (output, target, input, var) en la carpeta dada.
    
    Parámetros:
        folder_path (str): Ruta de la carpeta que contiene las imágenes TIFF.
    """
    tiff_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".tiff")])
    
    image_sets = {}  # Diccionario {identificador: (output, target, input, var)}

    for filename in tiff_files:
        for prefix in ["output_", "target_", "input_", "var_"]:
            identifier = extract_identifier(filename, prefix)
            if identifier:
                if identifier not in image_sets:
                    image_sets[identifier] = [None, None, None, None]
                idx = ["output_", "target_", "input_", "var_"].index(prefix)
                image_sets[identifier][idx] = os.path.join(folder_path, filename)
                break

    image_sets = {k: v for k, v in image_sets.items() if all(v)}  # Filtrar conjuntos completos
    image_keys = sorted(image_sets.keys())
    
    if not image_sets:
        print("No se encontraron conjuntos completos de imágenes TIFF en la carpeta.")
        return
    
    num_sets = len(image_keys)
    current_set_idx = 0

    def show_set(index):
        """Muestra el conjunto de imágenes correspondiente al índice dado."""
        identifier = image_keys[index]
        output_path, target_path, input_path, var_path = image_sets[identifier]

        output_image_np = tifffile.imread(output_path)
        target_image_np = tifffile.imread(target_path)
        input_image_np = tifffile.imread(input_path)
        var_image_np = tifffile.imread(var_path)

        # Si input_image_np es un stack de frames, seleccionamos el primer frame
        if len(input_image_np.shape) > 2:
            input_image_np = input_image_np[0]  # Tomamos solo el primer frame

        plt.clf()
        plt.subplot(2, 2, 1)
        #plt.imshow(output_image_np, cmap='gray', vmin=0, vmax=4751) 
        plt.imshow(output_image_np, cmap='gray') 
        plt.title("Output")
        
        plt.subplot(2, 2, 2)
        #plt.imshow(target_image_np, cmap='gray', vmin=0, vmax=4751)             
        plt.imshow(target_image_np, cmap='gray')
        plt.title("Target")
        
        plt.subplot(2, 2, 3)
        #plt.imshow(input_image_np, cmap='gray', vmin=0, vmax=4751)             
        plt.imshow(input_image_np, cmap='gray')
        plt.title("Input (First Frame)")
        
        plt.subplot(2, 2, 4)
        plt.imshow(var_image_np, cmap='gray', vmin=0, vmax=1)           
        #plt.imshow(var_image_np, cmap='gray')
        plt.title("Variance")
        
        plt.suptitle(f"Conjunto {index + 1} de {num_sets}")
        plt.draw()

    plt.figure(figsize=(10, 10))
    show_set(current_set_idx)

    def on_key(event):
        nonlocal current_set_idx
        if event.key == 'right' and current_set_idx < num_sets - 1:
            current_set_idx += 1
        elif event.key == 'left' and current_set_idx > 0:
            current_set_idx -= 1
        else:
            return
        show_set(current_set_idx)

    plt.connect('key_press_event', on_key)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualizar conjuntos de imágenes TIFF en una carpeta.")
    parser.add_argument("folder_path", type=str, help="Ruta de la carpeta con imágenes TIFF.")
    args = parser.parse_args()
    visualize_tiff_sets(args.folder_path)
