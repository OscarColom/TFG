import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import os
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
import torch.nn.functional as F
from torchvision.utils import flow_to_image
from torchvision.transforms.functional import to_pil_image
from torch.autograd import Variable



# Función para visualizar las imágenes en el batch
def plot_images(images):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
    for idx, img in enumerate(images):
        img = to_pil_image(img)  # Convertir tensor a imagen PIL
        axes[idx].imshow(img)
        axes[idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show() 

base_path = os.path.dirname(os.path.abspath(__file__))

# Cargar la imagen
im1 = np.load(os.path.join(base_path, "DSA-Self-real-L1A-dataset/test/20201122_101804_ssc16d3_0020_basic_panchromatic_dn.tif_x2278_y0311.npy"))
print(im1[0].shape)

#print(im1[0, :255, :255])  # Mostrar una pequeña parte de la imagen
print("Valor mínimo:", im1.min())
print("Valor máximo:", im1.max())

# Convertir a tensor y agregar dimensión de canal
#im1_tensor = torch.tensor(im1, dtype=torch.float32).unsqueeze(0) / 4751  # Agregar dimensión de canal

# Definir la transformación
transform = T.Compose([
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=[0.5], std=[0.5])  # Asegúrate de que el tensor tiene 1 canal
])

# Aplicar la transformación
#im1_tensor_normalized = transform(im1_tensor)

# Mostrar una pequeña parte de la imagen
#print(im1_tensor_normalized[0, :255, :255])

# Obtener los valores mínimo y máximo de los píxeles
#print("Valor mínimo:", im1_tensor_normalized.min().item())
#print("Valor máximo:", im1_tensor_normalized.max().item())


####################################################

# folder_path = (os.path.join(base_path, "DSA-Self-real-L1A-dataset/test"))
# min_val = float('inf')
# max_val = float('-inf')

# # Iterar sobre todos los archivos .npy en la carpeta
# for file_name in os.listdir(folder_path):
#     if file_name.endswith(".npy"):
#         file_path = os.path.join(folder_path, file_name)
#         data = np.load(file_path)
#         min_val = min(min_val, data.min())
#         max_val = max(max_val, data.max())

# print("Valor mínimo en todo el dataset:", min_val)
# print("Valor máximo en todo el dataset:", max_val)



# folder_path = (os.path.join(base_path, "DSA-Self-real-L1A-dataset/train"))
# min_val = float('inf')
# max_val = float('-inf')

# # Iterar sobre todos los archivos .npy en la carpeta
# for file_name in os.listdir(folder_path):
#     if file_name.endswith(".npy"):
#         file_path = os.path.join(folder_path, file_name)
#         data = np.load(file_path)
#         min_val = min(min_val, data.min())
#         max_val = max(max_val, data.max())

# print("Valor mínimo en todo el dataset:", min_val)
# print("Valor máximo en todo el dataset:", max_val)

class DSASuperResolutionDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(".npy")]
        self.transform = T.Normalize(mean=0.5, std=0.5)  # map [0, 1] into [-1, 1]

    def __len__(self):
        return len(self.file_paths)

    def preprocess_frame(self, frame, to_rgb=True):
        
        frame = frame.unsqueeze(0)  # Añadir canal de profundidad (1, H, W) para escala de grises, frame tien (256,256)
        if to_rgb:
            frame = frame.expand(3, frame.shape[1], frame.shape[2])  # Convertir a RGB, copiar el canal para los 3
        #max_val_frame = frame.max() # par apasar cada iamgen de [0,1], pense en hacer frame = frame /4571 que es el pixel con mas valor del dataset pero quedaban algunas imagenes muy oscuras
        #frame = frame / max_val_frame
        frame = frame / 4751
        return self.transform(frame) #SE aplica la normalizacion [0,1]->[-1,1]

    def __getitem__(self, idx):
        # Cargar el stack de imágenes
        stack = np.load(self.file_paths[idx])  # (15, 256, 256)

        # Asegurar formato tensor adecuado
        stack_tensor = torch.tensor(stack, dtype=torch.float32)  # Aqui aun tienen valores [0, 255]

        # Separar frames: target, referencia, input
        target_frame = stack_tensor[0]  # Frame 0
        reference_frame = stack_tensor[1]  # Frame 1
        input_frames = stack_tensor[1:]  # Frames del 1 al 14

        # Normalizar target y referencia
        target_frame = self.preprocess_frame(target_frame, to_rgb=True)
        reference_frame = self.preprocess_frame(reference_frame, to_rgb=True)

        # Normalizar cada frame de los input_frames
        input_frames = torch.stack([self.preprocess_frame(frame, to_rgb=False) for frame in input_frames])


        return target_frame, reference_frame, input_frames


base_path = os.path.dirname(os.path.abspath(__file__))

train_dataset = DSASuperResolutionDataset(folder_path=os.path.join(base_path, "DSA-Self-real-L1A-dataset/train"))
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

test_dataset = DSASuperResolutionDataset(folder_path=os.path.join(base_path, "DSA-Self-real-L1A-dataset/test"))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Cargar modelo RAFT preentrenado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raft_model = raft_large(pretrained=True, progress=False).to(device)
raft_model.eval()

def compute_optical_flow(batch1, batch2, model):
    """Calcula el flujo óptico entre dos lotes de imágenes usando RAFT."""
    flows = model(batch1, batch2)
    return flows[-1]  # Devolvemos el flujo de la última iteración






def upscale_flow(lr_flow):
    # lr_flow es de la forma [8, 2, H, W] (batch de 8 imágenes, 2 canales para X e Y)
    sr_flow = F.interpolate(lr_flow, scale_factor=2, mode='bilinear', align_corners=False)
    return sr_flow*2



def warp(x, flo, device = None):
    """
    Alinea una imagen/tensor de acuerdo con el flujo óptico.
    - output: imagen alineada
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.sum(flo * flo) == 0:
        return x
    
    B, C, H, W = x.size()

    # Crear una malla de coordenadas
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(device)

    # Añadir el flujo a la malla
    vgrid = Variable(grid) + flo.to(device)

    # Normalizar la malla a [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)

    # Realizar interpolación bilineal usando grid_sample
    output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return output








# Extraer el primer batch del train_loader
for target_frame, reference_frame, input_frames in train_loader:
    # Asegurarse de que los datos estén en el dispositivo correcto
    target_frame = target_frame.to(device)
    reference_frame = reference_frame.to(device)


    print("Max value of target_frame:", target_frame.max())
    print("Min value of target_frame:", target_frame.min())

    print("Shape of target_frame:", target_frame.shape)
    print("Shape of reference_frame:", reference_frame.shape)


    print("Shape of input_frames:", input_frames.shape)
    input_concatenado_ = input_frames.view(input_frames.shape[0], -1, input_frames.shape[-2], input_frames.shape[-1])
    print("Shape of input_framesconactenados :", input_concatenado_.shape)

    # Calcular el flujo óptico
    shift_amount = 20
    shifted_target_frame =  torch.roll(target_frame, shifts=shift_amount, dims=-1)     
    optical_flow = compute_optical_flow(target_frame, shifted_target_frame, raft_model)
    #optical_flow = compute_optical_flow(target_frame, reference_frame, raft_model)
    prueba = warp(shifted_target_frame,optical_flow )

    # Mostrar la forma del resultado
    print("Shape of optical_flow:", optical_flow.shape)
    
    sr_flow = upscale_flow(optical_flow)
    print("Shape of sr_optical_flow:", sr_flow.shape)

    
    refrence_sr = F.interpolate(reference_frame, scale_factor=2, mode='bilinear', align_corners=False)
    print("Shape of refrence_sr:", refrence_sr.shape)

    sr_reference_aligned = warp(refrence_sr, sr_flow) #usamos otra para n probar l ared
    print("Shape of sr_reference_aligned:", sr_reference_aligned.shape)
    print("max value of sr_reference_aligned:", sr_reference_aligned.max())
    print("min value of sr_reference_aligned:", sr_reference_aligned.min())
    
    #sr_reference_aligned_batch = [(img1 + 1) / 2 for img1 in sr_reference_aligned]
    # prueba = [(img1 + 1) / 2 for img1 in prueba]
    # shifted_target_frame = [(img1 + 1) / 2 for img1 in shifted_target_frame]

    #target_frame= [(img1 + 1) / 2 for img1 in target_frame]
    prueba = (prueba + 1) / 2
    shifted_target_frame = (shifted_target_frame + 1) / 2
    target_frame = (target_frame + 1) / 2

    prueba_recortada = prueba[:, :, :, :-20]
    target_frame_recortado = target_frame[:, :, :, :-20]

    resta = (target_frame_recortado - prueba_recortada)
    resta_mean = torch.abs(target_frame_recortado - prueba_recortada)
    print(torch.mean(resta_mean))
    #resta = hacer resta 

    #torch.mean( (im - warped_im)**2 
    #torch.mean( torch.abs(im - warped_im) )
    plot_images(resta)

    plot_images(resta_mean)

    plot_images(prueba)
    #plot_images(target_frame)
    # print("Shape of target_image_gary:", target_frame.shape)
    # target_image = target_frame[0].detach().cpu().clamp(0, 1)
    # target_frame_grayscale = target_frame[:, 0, :, :]
    # print("Shape of target_image_gary:", target_image.shape)
    # plot_images(target_image)


    #Visualizacion
    low_imgs = flow_to_image(optical_flow)
    img1_batch = [(img1 + 1) / 2 for img1 in target_frame]
    grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, low_imgs)]
    plot(grid)

    break  # Solo procesar el primer batch para comprobación



#PAAR CROPS

# import torch
# import torchvision.transforms.functional as TF
# import random

# class DSASuperResolutionDataset(Dataset):
#     def __init__(self, folder_path, augment=False, crop_size=128, num_crops=4):
#         self.file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(".npy")]
#         self.transform = T.Normalize(mean=0.5, std=0.5)  # map [0, 1] into [-1, 1]
#         self.augment = augment
#         self.crop_size = crop_size  # Tamaño de los recortes
#         self.num_crops = num_crops  # Número de recortes por imagen

#     def __len__(self):
#         return len(self.file_paths) * self.num_crops  # Aumentamos el tamaño del dataset

#     def preprocess_frame(self, frame, to_rgb=True):
#         frame = frame.unsqueeze(0)  # Añadir canal de profundidad (1, H, W) para escala de grises
#         if to_rgb:
#             frame = frame.expand(3, frame.shape[1], frame.shape[2])  # Convertir a RGB
#         frame = frame / 4751
#         return self.transform(frame)

#     def random_crop(self, frame):
#         """Genera un recorte aleatorio de la imagen."""
#         _, h, w = frame.shape
#         if h < self.crop_size or w < self.crop_size:
#             raise ValueError("El tamaño del recorte es mayor que la imagen original.")
#         top = random.randint(0, h - self.crop_size)
#         left = random.randint(0, w - self.crop_size)
#         return TF.crop(frame, top, left, self.crop_size, self.crop_size)

#     def __getitem__(self, idx):
#         # Calcular el índice de la imagen original y el índice del recorte
#         img_idx = idx // self.num_crops
#         crop_idx = idx % self.num_crops

#         # Cargar la imagen original
#         stack = np.load(self.file_paths[img_idx])  # (15, 256, 256)
#         stack_tensor = torch.tensor(stack, dtype=torch.float32)

#         # Obtener los frames
#         target_frame = stack_tensor[0]
#         reference_frame = stack_tensor[1]
#         input_frames = stack_tensor[1:]

#         # Preprocesar todos los frames
#         target_frame = self.preprocess_frame(target_frame, to_rgb=True)
#         reference_frame = self.preprocess_frame(reference_frame, to_rgb=True)
#         input_frames = [self.preprocess_frame(frame, to_rgb=False) for frame in input_frames]

#         # Aplicar recortes (crops)
#         if self.augment:
#             # Aplicar el mismo recorte a todos los frames relacionados
#             top = random.randint(0, target_frame.shape[1] - self.crop_size)
#             left = random.randint(0, target_frame.shape[2] - self.crop_size)

#             target_frame = TF.crop(target_frame, top, left, self.crop_size, self.crop_size)
#             reference_frame = TF.crop(reference_frame, top, left, self.crop_size, self.crop_size)
#             input_frames = [TF.crop(frame, top, left, self.crop_size, self.crop_size) for frame in input_frames]

#         input_frames = torch.stack(input_frames)

#         return target_frame, reference_frame, input_frames



# train_dataset = DSASuperResolutionDataset(
#     folder_path="/data/upftfg04/ocolom/DSA-Self-real-L1A-dataset/train",
#     augment=True,
#     crop_size=128,
#     num_crops=4
# )