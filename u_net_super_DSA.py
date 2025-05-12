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


import wandb

wandb.login(key="877815d8768c4344b4047cf25b96a41ee3c8bcdc")

# Inicializar W&B
wandb.init(
    project="superresolucion-imagenes",
    config={
        "learning_rate": 1e-4,
        "batch_size": 8,
        "epochs": 2500,
        "model_architecture": "U-Net",
        "dataset": "DSA-Self-real-L1A-dataset"
    },
    mode="disabled"
    #settings=wandb.Settings(init_timeout=600, mode="offline")  # Modo offline y timeout de 600 segundos
)
# Función para guardar mensajes en un archivo de texto
def log_message(message, log_file):
    """
    Guarda un mensaje en un archivo de texto y opcionalmente lo imprime en la consola.
    
    Parámetros:
        message (str): El mensaje a guardar.
        log_file (str): Ruta del archivo de texto donde se guardará el mensaje.
    """
    with open(log_file, "a") as f:  # Abre el archivo en modo 'append' (añadir)
        f.write(message + "\n")     # Escribe el mensaje en el archivo
    print(message)  # Opcional: Imprime el mensaje en la consola





# Configurar el parser de argumentos
parser = argparse.ArgumentParser(description="Entrenamiento de U-Net para superresolución.")
parser.add_argument("--data_dir", type=str, required=True, help="Ruta del directorio de datos.")
parser.add_argument("--output_dir", type=str, required=True, help="Ruta del directorio de salida.")
parser.add_argument("--run_name", type=str, required=True, help="Nombre de la ejecución.")
parser.add_argument("--learning_rate", type=float, required=True, help="Tasa de aprendizaje.")
parser.add_argument("--batch_size", type=int, required=True, help="Tamaño del batch.")
parser.add_argument("--epochs", type=int, required=True, help="Número de épocas.")
parser.add_argument("--resume", type=lambda x: x.lower() == 'true', required=True, help="Indica si se debe reanudar el entrenamiento.")
parser.add_argument("--resume_checkpoint_path", type=str, required=True, help="Ruta del checkpoint para reanudar el entrenamiento.")
parser.add_argument("--criterion", type=str, required=True, help="Función de pérdida.")
parser.add_argument("--optimizer", type=str, required=True, help="Optimizador.")
args = parser.parse_args()

# Crear la carpeta de salida
os.makedirs(args.output_dir, exist_ok=True)

# Definir el archivo de log después de que output_dir esté definido
log_file = os.path.join(args.output_dir, "training_log.txt")

# Mensaje de confirmación
print(f"Datos cargados desde: {args.data_dir}")
print(f"Resultados guardados en: {args.output_dir}")




# Crear el archivo de log
log_file = os.path.join(args.output_dir, "training_log.txt")




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
train_dataset = DSASuperResolutionDataset(folder_path="/data/upftfg04/ocolom/DSA-Self-real-L1A-dataset/train", augment = True)
test_dataset = DSASuperResolutionDataset(folder_path="/data/upftfg04/ocolom/DSA-Self-real-L1A-dataset/test")
val_dataset = DSASuperResolutionDataset(folder_path="/data/upftfg04/ocolom/DSA-Self-real-L1A-dataset/val")


# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)



# Ruta al archivo de pesos descargado
weights_path = "./raft_large_C_T_SKHT_V2-ff5fadd5.pth"

# CARGAR MODELO RAFT PREENTRENADO LOCALMENTE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raft_model = raft_large(weights=None).to(device)
raft_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
raft_model.eval()




def compute_optical_flow(batch1, batch2, model):
    """Calcula el flujo óptico entre dos lotes de imágenes usando RAFT."""
    flows = model(batch1, batch2)
    return flows[-1]  # Devolvemos el flujo de la última iteración (es el mas preciso)


def upscale_flow(lr_flow):
    "Augmentar resolucion de sr_flow [8, 2, 256, 256] -> [8, 2, 512, 512]"
    # lr_flow es de la forma [8, 2, H, W] (batch de 8 imágenes, 2 canales para X e Y)
    sr_flow = F.interpolate(lr_flow, scale_factor=2, mode='bilinear', align_corners=False)
    return sr_flow * 2  #Los vectores tienen que ser mas grandes y que la imagen lo es


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




def moving_average(data, window_size):
    """
    Aplica un promedio móvil centrado a una lista de datos.
    
    Parámetros:
        data (list): Lista de valores a suavizar.
        window_size (int): Tamaño de la ventana (número de puntos a promediar).
    
    Retorna:
        smoothed_data (numpy array): Datos suavizados.
    """
    # Convertir la lista a un array de NumPy
    data = np.array(data)
    
    # Crear un kernel para el promedio móvil
    kernel = np.ones(window_size) / window_size
    
    # Aplicar convolución para calcular el promedio móvil
    smoothed_data = np.convolve(data, kernel, mode='same')
    
    return smoothed_data


# DEFINICIÓN DE LA ARQUITECTURA U-NET

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2) #de momento no se usa
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]  #mirar que tan diferentes so nlas dimensiones de las imagenes
        diffY = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffY // 2, diffY - diffY // 2, #Rellenar x1 para que coincidan dimensiones
                                    diffX // 2, diffX - diffX // 2))
        return self.conv(torch.cat([x2, x1], dim=1)) #Concatenar las dos imagenes, los canales (skip conection)



class final_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #igual que up pero sin skip conection
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)

        return self.conv(x)




class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



#U-net
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = double_conv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.final_up = final_up(64, 32)# poenr tamaño al de la original
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        #x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])  # concatenar los canales [8, 14, 3, 256, 256] -> [8, 42, 256, 256]
        x = x.squeeze(2)  # concatenar los canales [8, 14, 1, 256, 256] -> [8, 14, 256, 256]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final_up(x)  ##
        return self.outc(x)




def train_model(model, train_loader, val_loader, num_epochs=15, lr=1e-3, output_dir="outputs", checkpoint_path="checkpoint.pth", resume=False, resume_checkpoint_path=None, criterion="L1Loss", optimizer="Adam"):
    # Crear la ruta completa del checkpoint
    checkpoint_path = os.path.join(output_dir, checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Configurar la función de pérdida
    if criterion == "L1Loss":
        criterion = nn.L1Loss()
    elif criterion == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Criterion no soportado: {criterion}")

    # Configurar el optimizador
    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        raise ValueError(f"Optimizador no soportado: {optimizer}")


    # #Parametros del Scheduler del lr
    # patience = 5  # Número de épocas para esperar antes de reducir el LR
    # factor = 0.5  # Factor de reducción del LR
    # min_lr = 1e-6  # LR mínimo permitido
    # best_val_loss = float('inf')  # Mejor pérdida de validación registrada
    # epochs_without_improvement = 0  # Contador de épocas sin mejora


    train_losses = []  # Guardará las pérdidas de entrenamiento por iteración
    train_epoch_losses = []  # Pérdidas de entrenamiento promedio por época
    val_losses = []  # Pérdidas de validación por época

    # 1. Definir sistema de checkpoint
    start_epoch = 0
    # Cargar checkpoint existente si se especifica
    if resume:
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            train_epoch_losses = checkpoint.get('train_epoch_losses', [])
            log_message(f"Resumiendo entrenamiento desde epoch {start_epoch}", log_file)
            for epoch in range(len(train_epoch_losses)):
                log_message(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_losses[epoch]:.4f}, Val Loss: {val_losses[epoch]:.4f}", log_file)
            log_message(f"Continuacion del entrenamiento", log_file)
        else:
            log_message("No se encontró el checkpoint. Iniciando desde epoch 0.", log_file)


    # Guardar información sobre el optimizador y la función de pérdida
    log_message(f"Configuración del entrenamiento:", log_file)
    log_message(f"- Función de pérdida: {criterion.__class__.__name__}", log_file)
    log_message(f"- Optimizador: {optimizer.__class__.__name__}", log_file)
    log_message(f"- Learning rate: {lr}", log_file)
    log_message(f"- Weight decay: {optimizer.param_groups[0]['weight_decay']}", log_file)
    log_message(f"- Número de épocas: {num_epochs}", log_file)
    log_message(f"- Carpeta de salida: {output_dir}", log_file)
    log_message(f"Iniciando entrenamiento en {datetime.now()}", log_file)

    for epoch in range(start_epoch, num_epochs):
        # Modo entrenamiento
        model.train()
        train_loss_epoch = 0
        for target_frame, reference_frame, input_frames in train_loader:
            target_frame, reference_frame, input_frames = target_frame.to(device), reference_frame.to(device), input_frames.to(device)

            #target_frame(shape) --> (3,256, 256)
            #reference_frame(shape) --> (3,256, 256)
            #input_frames(shape) --> (14,1,256, 256)

            optimizer.zero_grad()
            output = model(input_frames) #Dentro la U-net conatena el input  (14,1,256, 256) --> (14 ,256, 256)
            #output(shape) --> (1,512, 512)

            # Calculamos el optical flow entre el target y la de referencia    lr_flow(shape) --> (N,2,256,256)
            lr_flow = compute_optical_flow(target_frame,reference_frame, raft_model)

            # Augmentamos resolucion del optical flow         sr_flow (shape) --> (N,2,512, 512)
            sr_flow = upscale_flow(lr_flow)  # (usa interpolacion)

            # Alineamoes el output de la U-net con el target      sr_reference_aligned(shape) --> (N,3,512, 512)
            sr_reference_aligned = warp(output, sr_flow)

            # Bajamos resolucion de la imagen alineada  lr_reference_aligned(shape) -->  (N,1,256,256)
            #lr_reference_aligned = F.interpolate(sr_reference_aligned, scale_factor=0.5, mode='bilinear', align_corners=True)
            lr_reference_aligned = sr_reference_aligned[:, :, ::2, ::2]  #Seleccionar píxeles pares en ambas dimensiones


            #Pasar el target a escala de grises target_frame_grayscale = (1,256,256)
            target_frame_grayscale = target_frame[:, 0, :, :]
            target_frame_grayscale = target_frame_grayscale.unsqueeze(1)

            # Calculamos la loss
            loss = criterion(lr_reference_aligned, target_frame_grayscale)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())  # Guardar la pérdida de la iteración
            train_loss_epoch += loss.item()  # Acumular para el promedio por época

            # Registrar la pérdida de entrenamiento por iteración
            wandb.log({"train_loss_iter": loss.item()})

        train_loss_epoch /= len(train_loader)
        train_epoch_losses.append(train_loss_epoch)

        # Modo evaluación
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for target_frame, reference_frame, input_frames in val_loader:
                target_frame, reference_frame, input_frames = target_frame.to(device), reference_frame.to(device), input_frames.to(device)
                output = model(input_frames)

                lr_flow = compute_optical_flow(target_frame,reference_frame, raft_model)
                sr_flow = upscale_flow(lr_flow)  # (usa interpolacion)
                sr_reference_aligned = warp(output, sr_flow)
                # lr_reference_aligned = F.interpolate(sr_reference_aligned, scale_factor=0.5, mode='bilinear', align_corners=True)
                # loss = criterion(lr_reference_aligned, target_frame)

                lr_reference_aligned = sr_reference_aligned[:, :, ::2, ::2]  #Seleccionar píxeles pares en ambas dimensiones
                #Pasar el target a escala de grises target_frame_grayscale = (1,256,256)
                target_frame_grayscale = target_frame[:, 0, :, :]
                target_frame_grayscale = target_frame_grayscale.unsqueeze(1)
                # Calculamos la loss
                loss = criterion(lr_reference_aligned, target_frame_grayscale)

                val_loss_epoch += loss.item()

        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)

        # Registrar métricas por época
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_epoch": train_loss_epoch,
            "val_loss_epoch": val_loss_epoch
        })

        # # Ajustar el learning rate si no hay mejora
        # if val_loss_epoch < best_val_loss:
        #     best_val_loss = val_loss_epoch
        #     epochs_without_improvement = 0
        # else:
        #     epochs_without_improvement += 1
        #     if epochs_without_improvement >= patience:
        #         # Reducir el LR
        #         for param_group in optimizer.param_groups:
        #             new_lr = max(param_group['lr'] * factor, min_lr)
        #             param_group['lr'] = new_lr
        #         log_message(f"Reduciendo LR a {new_lr}",log_file)
        #         epochs_without_improvement = 0  # Reiniciar el contador


        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_epoch_losses': train_epoch_losses,
                'val_losses': val_losses,
            }, checkpoint_path)

        # Imprimir métricas
        log_message(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}",log_file)

    return train_losses, train_epoch_losses, val_losses



# ENTRENAMIENTO
unet_model = UNet(n_channels=14, n_classes=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model.to(device)


train_losses, train_epoch_losses, val_losses = train_model(unet_model, train_loader, val_loader, num_epochs = args.epochs, lr = args.learning_rate ,output_dir = args.output_dir, resume = args.resume, resume_checkpoint_path=args.resume_checkpoint_path, criterion= args.criterion, optimizer= args.optimizer)



#Crear plot de la Loss
iterations_per_epoch = len(train_loader) # Crear el eje X para las loss
train_x = [i / iterations_per_epoch + 1 for i in range(len(train_losses))]  # Ajustar a escala de épocas
val_x = list(range(1, len(val_losses) + 1)) # Crear el eje X para las loss val

# Calcular el promedio móvil de la pérdida de entrenamiento
#window_size = 11  # 5 puntos anteriores + 5 posteriores + el punto actual
window_size = max(5, len(train_losses) // 200)  
smoothed_train_losses = moving_average(train_losses, window_size)

# Aplicar suavizado a la validación (con ventana más grande que en entrenamiento)
window_size_val = max(5, len(val_losses) // 50)  # Ajustar según la cantidad de épocas
smoothed_val_losses = moving_average(val_losses, window_size_val)


# Graficar la curva de loss
plt.figure(figsize=(10, 6))
# Train loss (suavizada)
plt.semilogy(train_x, smoothed_train_losses, label='Train Loss (smoothed)', color='blue', alpha=0.8, linewidth=1.5, linestyle='-')
# Train loss (original, más transparente)
plt.semilogy(train_x, train_losses, label='Train Loss (original)', color='blue', alpha=0.05, linewidth=1, linestyle='-')
# Validation loss (suavizada, sin marcadores)
plt.semilogy(val_x, smoothed_val_losses, label='Validation Loss (smoothed)', color='red', alpha=0.8, linewidth=1.5, linestyle='-')
# Validation loss (original, más transparente, sin marcadores)
plt.semilogy(val_x, val_losses, label='Validation Loss (original)', color='orange', alpha=0.05, linewidth=1, linestyle='-')

# Configuración del gráfico
plt.xticks(val_x[::max(1, len(val_x) // 10)])  # Mostrar menos ticks en el eje X
plt.legend()
plt.title("Loss Curve (Log Scale): Training (Smoothed) and Validation (Smoothed)")
plt.xlabel("Epochs")
plt.ylabel("Loss (Log Scale)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Guardar la gráfica
plot_path = os.path.join(args.output_dir, "loss_curve_smoothed.png")
plt.savefig(plot_path, dpi=300)
log_message(f"Loss curve saved at {plot_path}", log_file)
plt.close()



#Guardar paar ver resultado

def process_and_save_selected_stacks_by_name(dataset, model, device, output_dir, selected_filenames):
    """
    Procesa los stacks seleccionados a través del modelo y guarda los pares (output, target),
    asegurando que los stacks procesados coincidan con los nombres de archivo dados.
    
    Parámetros:
        dataset (Dataset): Dataset de prueba.
        model (torch.nn.Module): Modelo de superresolución.
        device (torch.device): Dispositivo en el que se ejecuta el modelo.
        output_dir (str): Carpeta donde se guardarán las imágenes.
        selected_filenames (list): Lista de nombres de archivos .npy a procesar.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar los índices de los stacks que coinciden con los nombres deseados
    selected_indices = [i for i, path in enumerate(dataset.file_paths) if os.path.basename(path) in selected_filenames]

    if not selected_indices:
        print("⚠ No se encontraron stacks con los nombres proporcionados.")
        return
    
    print(f"📂 Procesando {len(selected_indices)} imágenes seleccionadas...")

    for stack_idx in selected_indices:
        stack_filename = os.path.basename(dataset.file_paths[stack_idx])
        target_frame, _, input_frames = dataset[stack_idx]  # Cargar datos
        target_frame, input_frames = target_frame.to(device), input_frames.to(device)
        
        # Predicción del modelo
        output = model(input_frames.unsqueeze(0))  # Añadir dimensión de batch
        
        # Convertir tensores a imágenes
        output_image = (((output[0].detach().cpu()) + 1) / 2) * 4751
        target_image = (((target_frame.detach().cpu()) + 1) / 2) * 4751
        
        # Convertir target_frame a escala de grises
        target_image = target_image.mean(dim=0, keepdim=True)  

        output_image_wandb = (output_image / 4751) * 255
        target_imageoutput_image_wandb = (target_image / 4751) * 255

        # Registrar imágenes en W&B
        wandb.log({
            "output_image": wandb.Image(output_image_wandb.squeeze().numpy()),
            "target_image": wandb.Image(target_imageoutput_image_wandb.squeeze().numpy())
        })
        
        # Convertir tensores a arrays de NumPy
        output_image_np = output_image.numpy()
        target_image_np = target_image.numpy()
        
        # Guardar imágenes en formato TIFF, usando el nombre original del archivo
        output_path = os.path.join(output_dir, f"output_{stack_filename.replace('.npy', '.tiff')}")
        target_path = os.path.join(output_dir, f"target_{stack_filename.replace('.npy', '.tiff')}")

        tifffile.imwrite(output_path, output_image_np.squeeze().astype(np.uint16))
        tifffile.imwrite(target_path, target_image_np.squeeze().astype(np.uint16))
        
        print(f"✅ Guardado: {stack_filename} -> {output_path}, {target_path}")

    print(f"\n🎉 ¡Procesamiento completado! Las imágenes están en {output_dir}")

# Lista de archivos a procesar
selected_filenames = [
    "20200809_060638_ssc14d1_0004_basic_panchromatic_dn.tif_x0960_y0440.npy",
    "20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.npy",
    "20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0164_y0428.npy",
    "20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0293_y0368.npy",
    "20201122_101804_ssc16d3_0020_basic_panchromatic_dn.tif_x2278_y0311.npy"
]

# Llamar a la función con nombres en lugar de índices
process_and_save_selected_stacks_by_name(test_dataset, unet_model, device, args.output_dir, selected_filenames)

# Finalizar W&B
wandb.finish()