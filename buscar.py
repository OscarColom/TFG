import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torch.autograd import Variable
from torchvision.transforms.functional import to_pil_image
import tifffile
import random  # Importar random para usar random.choice
from datetime import datetime
import argparse
from matplotlib.widgets import Slider

# Obtener la ruta del directorio donde está el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta relativa a los datos
folder_path = os.path.join(script_dir, "DSA-Self-real-L1A-dataset/test")

# PROCESAR EL DATASET
class DSASuperResolutionDataset(Dataset):
    def __init__(self, folder_path, augment=False):
        self.file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(".npy")]
        self.transform = T.Normalize(mean=0.5, std=0.5)  # map [0, 1] into [-1, 1]
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def preprocess_frame(self, frame, to_rgb=True):
        frame = frame.unsqueeze(0)  # Añadir canal de profundidad (1, H, W) para escala de grises
        if to_rgb:
            frame = frame.expand(3, frame.shape[1], frame.shape[2])  # Convertir a RGB
        frame = frame / 4751
        return self.transform(frame)

    def augment_frames(self, frames):
        # Aplicar las mismas transformaciones a todos los frames
        if self.augment:
            # Generar parámetros aleatorios comunes
            hflip = random.random() > 0.5
            vflip = random.random() > 0.5
            angle = random.choice([0, 90])

            # Aplicar transformaciones
            augmented_frames = []
            for frame in frames:
                if hflip:
                    frame = TF.hflip(frame)
                if vflip:
                    frame = TF.vflip(frame)
                if angle != 0:
                    frame = TF.rotate(frame, angle)
                augmented_frames.append(frame)
            return augmented_frames
        return frames

    def __getitem__(self, idx):
        stack = np.load(self.file_paths[idx])  # (15, 256, 256)
        stack_tensor = torch.tensor(stack, dtype=torch.float32)

        target_frame = stack_tensor[0]
        reference_frame = stack_tensor[1]
        input_frames = stack_tensor[1:]

        # Preprocesar todos los frames
        target_frame = self.preprocess_frame(target_frame, to_rgb=True)
        reference_frame = self.preprocess_frame(reference_frame, to_rgb=True)
        input_frames = [self.preprocess_frame(frame, to_rgb=False) for frame in input_frames]

        # Aplicar aumentos de manera consistente
        if self.augment:
            # Aplicar las mismas transformaciones a todos los frames relacionados
            frames_to_augment = [target_frame, reference_frame] + input_frames
            augmented_frames = self.augment_frames(frames_to_augment)

            # Separar los frames aumentados
            target_frame = augmented_frames[0]
            reference_frame = augmented_frames[1]
            input_frames = augmented_frames[2:]

        input_frames = torch.stack(input_frames)

        return target_frame, reference_frame, input_frames



#CARGAR DATASET
# Crear datasets para cluster
#train_dataset = DSASuperResolutionDataset(folder_path="/data/upftfg04/ocolom/DSA-Self-real-L1A-dataset/train", augment = True)
test_dataset = DSASuperResolutionDataset(folder_path=folder_path)
#val_dataset = DSASuperResolutionDataset(folder_path="/data/upftfg04/ocolom/DSA-Self-real-L1A-dataset/val")


# Crear DataLoaders
#train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)





def visualize_stacks_from_dataloader(dataloader, dataset):
    """
    Visualiza los stacks de imágenes del DataLoader, mostrando el nombre del archivo.
    
    Parámetros:
        dataloader (DataLoader): DataLoader con batch_size=1 que contiene los stacks de imágenes.
        dataset (Dataset): Dataset para acceder a los nombres de archivo originales.
    """
    stacks = list(dataloader)  # Obtener todos los datos
    num_stacks = len(stacks)

    if num_stacks == 0:
        print("El DataLoader está vacío.")
        return
    
    current_stack_idx = 0
    input_stack = stacks[current_stack_idx][0].squeeze(0).numpy()  # Shape: [num_frames, H, W]
    num_frames = input_stack.shape[0]
    
    # Obtener el nombre del archivo actual
    stack_filename = os.path.basename(dataset.file_paths[current_stack_idx])

    # Crear la figura y la subtrama para mostrar el frame
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    im = ax.imshow(input_stack[0], cmap='gray')
    plt.title(f"{stack_filename} - Stack {current_stack_idx + 1}/{num_stacks} - Frame 1/{num_frames}")

    # Slider para cambiar de frame dentro de un stack
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
    
    def update_frame(val):
        frame_idx = int(slider.val)
        im.set_data(input_stack[frame_idx])
        plt.title(f"{stack_filename} - Stack {current_stack_idx + 1}/{num_stacks} - Frame {frame_idx + 1}/{num_frames}")
        fig.canvas.draw_idle()
    
    slider.on_changed(update_frame)
    
    def on_key(event):
        nonlocal current_stack_idx, input_stack, num_frames, stack_filename
        if event.key == 'right':  # Avanzar al siguiente stack
            if current_stack_idx < num_stacks - 1:
                current_stack_idx += 1
        elif event.key == 'left':  # Retroceder al stack anterior
            if current_stack_idx > 0:
                current_stack_idx -= 1
        else:
            return

        # Cargar el nuevo stack y resetear el slider
        input_stack = stacks[current_stack_idx][0].squeeze(0).numpy()
        num_frames = input_stack.shape[0]
        stack_filename = os.path.basename(dataset.file_paths[current_stack_idx])  # Actualizar nombre del archivo
        
        slider.valmax = num_frames - 1
        slider.set_val(0)
        im.set_data(input_stack[0])
        plt.title(f"{stack_filename} - Stack {current_stack_idx + 1}/{num_stacks} - Frame 1/{num_frames}")
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# Ejemplo de uso:
visualize_stacks_from_dataloader(test_loader, test_dataset)