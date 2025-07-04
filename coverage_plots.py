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
#base_path = r"C:\Users\oscar\Desktop\TFG\Memoria\run_prueba_denosing_and_var_lr0.00001_bs8_epochs4500_resumeTrue_criterionL1Loss_optimizerAdam"
#base_path = r"X:\outputs\run_prueba_denosing_and_var_lr0.001_bs8_epochs1000_resumeFalse_criterionL1Loss_optimizerAdam"
#base_path = r"X:\outputs\run_prueba_denosing_and_var_lr0.001_bs8_epochs200_resumeFalse_criterionL1Loss_optimizerAdam"
base_path = r"C:\Users\oscar\Desktop\TFG\Memoria\\run_prueba_super_y_var_lr0.0005_bs8_epochs3000_resumeTrue_criterionL1Loss_optimizerAdam"

y_pred = tiff.imread(os.path.join(base_path, "output_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
sigma = tiff.imread(os.path.join(base_path, "var_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
y_true = tiff.imread(os.path.join(base_path, "target_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
sigma = np.sqrt(sigma)
input = tiff.imread(os.path.join(base_path, "input_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
reference_frame = input[0]

# 2. Downsample (usar solo píxeles pares)
y_pred_ds = y_pred[::2, ::2]
sigma_ds = sigma[::2, ::2]
# y_pred_ds = y_pred
# sigma_ds = sigma
# print(y_pred_ds.shape)
# print(sigma_ds.shape)
# print(y_true.shape)
#print(sigma_ds)




def evaluate_uncertainty(y_true, y_pred_ds, sigma_ds, reference_frame=None, raft_model=None, with_flow=False):
    """
    Evalúa la calibración de incertidumbre usando cobertura empírica.
    
    Parámetros:
    - y_true: Ground truth (H, W) - numpy array
    - y_pred_ds: Predicción (H, W) - numpy array
    - sigma_ds: Desviación estándar predicha (H, W) - numpy array
    - reference_frame: Imagen de referencia (H, W) - numpy array, necesario si with_flow=True
    - raft_model: Modelo RAFT cargado, necesario si with_flow=True
    - device: 'cuda' o 'cpu'
    - with_flow: Si se desea aplicar flujo óptico antes de calcular la calibración
    """
    
    def preprocess_for_flow(img):
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
        tensor = tensor.repeat(1, 3, 1, 1)  # (1, 3, H, W)
        tensor = tensor / 4751
        tensor = (tensor - 0.5) / 0.5  # Normalize(mean=0.5, std=0.5)
        return tensor

    if with_flow:
        if reference_frame is None or raft_model is None:
            raise ValueError("reference_frame y raft_model son requeridos si with_flow=True")

        # Preprocesamiento para el flujo óptico
        y_true_tensor = preprocess_for_flow(y_true)
        reference_tensor = preprocess_for_flow(reference_frame)

        # Calcular flujo óptico
        flow = compute_optical_flow(y_true_tensor, reference_tensor, raft_model)

        # Warp de las predicciones y sigma
        y_pred_ds_tensor = torch.from_numpy(y_pred_ds).float().unsqueeze(0).unsqueeze(0).to(device)
        sigma_ds_tensor = torch.from_numpy(sigma_ds).float().unsqueeze(0).unsqueeze(0).to(device)

        y_pred_ds = warp(y_pred_ds_tensor, flow).detach().cpu().numpy()
        sigma_ds = warp(sigma_ds_tensor, flow).detach().cpu().numpy()

        y_pred_ds = np.squeeze(y_pred_ds)
        sigma_ds = np.squeeze(sigma_ds)

    # Calcular intervalo de confianza y cobertura empírica
    alphas = np.linspace(0.01, 0.99, 20)
    empirical_coverage = []

    for alpha in alphas:
        Z_alpha = norm.ppf(0.5 + alpha / 2)
        lower = y_pred_ds - Z_alpha * sigma_ds
        upper = y_pred_ds + Z_alpha * sigma_ds
        within = (y_true >= lower) & (y_true <= upper)
        alpha_emp = np.mean(within)
        empirical_coverage.append(alpha_emp)

    # Graficar la curva de calibración
    plt.plot(alphas, empirical_coverage, label="Empirical Coverage")
    plt.plot(alphas, alphas, '--', label="Ideal")  # Identity line
    plt.xlabel("Theoretical Alpha (confidence level)")
    plt.ylabel("Empirical Alpha (observed coverage)")
    plt.title("Uncertainty Calibration" + (" (with flow)" if with_flow else ""))
    plt.grid(True)
    plt.legend()
    plt.show()

    return alphas, empirical_coverage

#evaluate_uncertainty(y_true, y_pred_ds, sigma_ds, with_flow=False)
#evaluate_uncertainty(y_true, y_pred_ds, sigma_ds, reference_frame, raft_model, with_flow=True)



def plot_error_vs_variance(y_true, y_pred_ds, sigma_ds, reference_frame=None, raft_model=None, with_flow=False):

    def preprocess_for_flow(img):
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
        tensor = tensor.repeat(1, 3, 1, 1)  # (1, 3, H, W)
        tensor = tensor / 4751
        tensor = (tensor - 0.5) / 0.5  # Normalize(mean=0.5, std=0.5)
        return tensor

    if with_flow:
        if reference_frame is None or raft_model is None:
            raise ValueError("reference_frame y raft_model son requeridos si with_flow=True")

        # Preprocesamiento para el flujo óptico
        y_true_tensor = preprocess_for_flow(y_true)
        reference_tensor = preprocess_for_flow(reference_frame)

        # Calcular flujo óptico
        flow = compute_optical_flow(y_true_tensor, reference_tensor, raft_model)

        # Warp de las predicciones y sigma
        y_pred_ds_tensor = torch.from_numpy(y_pred_ds).float().unsqueeze(0).unsqueeze(0).to(device)
        sigma_ds_tensor = torch.from_numpy(sigma_ds).float().unsqueeze(0).unsqueeze(0).to(device)

        y_pred_ds = warp(y_pred_ds_tensor, flow).detach().cpu().numpy()
        sigma_ds = warp(sigma_ds_tensor, flow).detach().cpu().numpy()

        y_pred_ds = np.squeeze(y_pred_ds)
        sigma_ds = np.squeeze(sigma_ds)

    target = y_true.astype(np.float32)
    prediction = y_pred_ds.astype(np.float32)
    std_dev = sigma_ds.astype(np.float32)  # Predicted standard deviation

    error = np.abs(target - prediction)  # Absolute error per pixel

    plt.figure(figsize=(10, 4))  # Más ancho, menos alto

    # Subplot 1: Absolute Error
    plt.subplot(1, 2, 1)
    plt.imshow(error, cmap='plasma', vmin=0, vmax=50)
    plt.colorbar( shrink=0.5)
    plt.title('Absolute Error Map')
    plt.axis('off')

    # Subplot 2: Predicted Standard Deviation
    plt.subplot(1, 2, 2)
    plt.imshow(std_dev, cmap='plasma', vmin=0, vmax=50)
    plt.colorbar( shrink=0.5)
    plt.title('Predicted Standard Deviation Map')
    plt.axis('off')

    plt.suptitle('Error vs. Predicted Standard Deviation', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuste para el título
    plt.show()


plot_error_vs_variance(y_true, y_pred_ds, sigma_ds, reference_frame, raft_model, with_flow=True)