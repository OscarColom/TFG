import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot
import os 
import torch
from torchvision.models.optical_flow import raft_large
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image
from torchvision.transforms.functional import to_pil_image

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


# 1. Cargar imágenes TIFF
#base_path = r"X:\outputs\run_prueba_denosing_and_var_lr0.001_bs8_epochs200_resumeFalse_criterionL1Loss_optimizerAdam"
base_path = r"X:\outputs\run_prueba_super_y_var_lr0.0005_bs8_epochs2000_resumeTrue_criterionL1Loss_optimizerAdam"

y_pred = tiff.imread(os.path.join(base_path, "output_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
sigma = tiff.imread(os.path.join(base_path, "var_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
y_true = tiff.imread(os.path.join(base_path, "target_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
sigma = np.sqrt(sigma)
input = tiff.imread(os.path.join(base_path, "input_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
reference_frame = input[1]

# 2. Downsample (usar solo píxeles pares)
y_pred_ds = y_pred[::2, ::2]
sigma_ds = sigma[::2, ::2]
#y_pred_ds = y_pred
#sigma_ds = sigma
# print(y_pred_ds.shape)
# print(sigma_ds.shape)
# print(y_true.shape)
#print(sigma_ds)



# # # Convertir imágenes a tensores (suponiendo que son imágenes en escala de grises)
# y_true_for_flow = torch.from_numpy(y_true).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
# y_true_for_flow = y_true_for_flow.repeat(1, 3, 1, 1)  # (1, 3, H, W)
# y_true_for_flow = y_true_for_flow / 4751
# y_true_for_flow = (y_true_for_flow - 0.5) / 0.5  # Esto equivale a Normalize(mean=0.5, std=0.5)

# reference_frame = torch.from_numpy(reference_frame).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
# reference_frame = reference_frame.repeat(1, 3, 1, 1)  # (1, 3, H, W)
# reference_frame = reference_frame = reference_frame / 4751
# reference_frame = (reference_frame - 0.5) / 0.5 

# y_pred_for_flow = torch.from_numpy(y_pred_ds).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
# y_pred_for_flow = y_pred_for_flow.repeat(1, 3, 1, 1) 
# y_pred_for_flow = y_pred_for_flow = y_true / 4751
# y_pred_for_flow = (y_pred_for_flow - 0.5) / 0.5 


# shift_amount = 20
# shifted_target_frame =  torch.roll(y_true_for_flow, shifts=shift_amount, dims=-1) 


# flow = compute_optical_flow(y_true_for_flow,reference_frame, raft_model)
# #flow = compute_optical_flow(y_true_for_flow,y_pred_for_flow, raft_model)

# # flow_rgb = flow_to_image(flow[0])
# # plt.figure(figsize=(6, 6))
# # plt.imshow(to_pil_image(flow_rgb))  # Convierte el tensor [3, H, W] en imagen RGB
# # plt.title("Optical Flow (via torchvision)")
# # plt.axis('off')
# # plt.show()


# # # Convertir y_pred_ds y sigma_ds a tensores antes del warp
# y_pred_ds = torch.from_numpy(y_pred_ds).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
# sigma_ds = torch.from_numpy(sigma_ds).float().unsqueeze(0).unsqueeze(0).to(device)


# y_pred_ds = warp(y_pred_ds, flow).detach().cpu().numpy()
# sigma_ds = warp(sigma_ds, flow).detach().cpu().numpy()
# # #y_true = np.squeeze(y_true.numpy()) # Esto asegura que y_true quede con la forma (256, 256)


# y_pred_ds = np.squeeze(y_pred_ds)
# sigma_ds = np.squeeze(sigma_ds)


# 3. Calcular intervalo de confianza
alphas = np.linspace(0.01, 0.99, 20)
empirical_coverage = []

for alpha in alphas:
    Z_alpha = norm.ppf(0.5 + alpha / 2)
    lower = y_pred_ds - Z_alpha * sigma_ds
    upper = y_pred_ds + Z_alpha * sigma_ds
    within = (y_true >= lower) & (y_true <= upper)
    alpha_emp = np.mean(within)
    empirical_coverage.append(alpha_emp)

# Graficar curva de calibración
plt.plot(alphas, empirical_coverage, label="Cobertura empírica")
plt.plot(alphas, alphas, '--', label="Ideal")  # Línea identidad
plt.xlabel("Alpha teórica (nivel de confianza)")
plt.ylabel("Alpha empírica (cobertura observada)")
plt.title("Calibración de incertidumbre")
plt.grid(True)
plt.legend()
plt.show()