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
from matplotlib.widgets import Button
import math

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
#base_path = r"X:\outputs\run_prueba_super_y_var_lr0.0005_bs8_epochs2000_resumeTrue_criterionL1Loss_optimizerAdam"
base_path = r"X:\outputs\run_prueba_denosing_and_var_lr0.001_bs8_epochs1000_resumeFalse_criterionL1Loss_optimizerAdam"

y_pred = tiff.imread(os.path.join(base_path, "output_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
sigma = tiff.imread(os.path.join(base_path, "var_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
y_true = tiff.imread(os.path.join(base_path, "target_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
sigma = np.sqrt(sigma)
input = tiff.imread(os.path.join(base_path, "input_20201122_101804_ssc16d2_0020_basic_panchromatic_dn.tif_x0078_y0416.tiff"))
reference_frame = input[0]

# 2. Downsample (usar solo píxeles pares)
# y_pred_ds = y_pred[::2, ::2]
# sigma_ds = sigma[::2, ::2]


def visualize_alignment_toggle(img1, img2, title1='Imagen 1', title2='Imagen 2', cmap='gray'):
    """
    Visualiza dos imágenes alternando entre ellas al hacer clic en un botón.
    
    Args:
        img1 (np.ndarray): Primera imagen.
        img2 (np.ndarray): Segunda imagen.
        title1 (str): Título para la primera imagen.
        title2 (str): Título para la segunda imagen.
        cmap (str): Mapa de color para la visualización.
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # Mostrar primera imagen por defecto
    img_display = ax.imshow(img1, cmap=cmap)
    ax.set_title(title1)
    ax.axis('off')

    # Estado de alternancia
    state = {'toggle': True}

    def toggle(event):
        if state['toggle']:
            img_display.set_data(img2)
            ax.set_title(title2)
        else:
            img_display.set_data(img1)
            ax.set_title(title1)
        state['toggle'] = not state['toggle']
        fig.canvas.draw_idle()

    # Botón
    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
    button = Button(ax_button, 'Alternar')
    button.on_clicked(toggle)

    plt.show()

def visualize_alignment_toggle_3(img1, img2, img3, 
                                 title1='Imagen 1', 
                                 title2='Imagen 2', 
                                 title3='Imagen 3', 
                                 cmap='gray'):
    """
    Visualiza tres imágenes alternando entre ellas al hacer clic en un botón.
    
    Args:
        img1, img2, img3 (np.ndarray): Imágenes a mostrar.
        title1, title2, title3 (str): Títulos de cada imagen.
        cmap (str): Mapa de color para la visualización.
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    images = [img1, img2, img3]
    titles = [title1, title2, title3]
    index = {'current': 0}

    # Mostrar primera imagen
    img_display = ax.imshow(images[0], cmap=cmap)
    ax.set_title(titles[0])
    ax.axis('off')

    def toggle(event):
        index['current'] = (index['current'] + 1) % 3
        img_display.set_data(images[index['current']])
        ax.set_title(titles[index['current']])
        fig.canvas.draw_idle()

    # Botón
    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
    button = Button(ax_button, 'Alternar')
    button.on_clicked(toggle)

    plt.show()

def visualize_sigma_and_error(pred, target, sigma,reference_frame, apply_flow=False, cmap='viridis'):
    """
    Muestra la imagen de sigma (desviación estándar) y el error cuadrático lado a lado,
    con opción de alinear la predicción y sigma mediante flujo óptico.

    Args:
        pred (np.ndarray): Imagen predicha.
        target (np.ndarray): Imagen ground truth.
        sigma (np.ndarray): Desviación estándar de la predicción.
        apply_flow (bool): Si True, aplica flujo óptico para alinear pred y sigma.
        cmap (str): Mapa de colores para mostrar las imágenes.
    """
    assert pred.shape == target.shape == sigma.shape, "Las imágenes deben tener las mismas dimensiones"

    if apply_flow:
        # Preprocesamiento de entrada para flujo óptico
        y_true_for_flow = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0).to(device)
        y_true_for_flow = y_true_for_flow.repeat(1, 3, 1, 1) / 4751
        y_true_for_flow = (y_true_for_flow - 0.5) / 0.5

        reference = torch.from_numpy(reference_frame).float().unsqueeze(0).unsqueeze(0).to(device)
        reference = reference.repeat(1, 3, 1, 1) / 4751
        reference = (reference - 0.5) / 0.5

        # Calcular flujo
        flow = compute_optical_flow(y_true_for_flow, reference, raft_model)

        # Alinear predicción y sigma
        pred_tensor = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0).to(device)
        sigma_tensor = torch.from_numpy(sigma).float().unsqueeze(0).unsqueeze(0).to(device)
        reference_tensor = torch.from_numpy(reference_frame).float().unsqueeze(0).unsqueeze(0).to(device)
        
        pred = warp(pred_tensor, flow).detach().cpu().numpy().squeeze()
        sigma = warp(sigma_tensor, flow).detach().cpu().numpy().squeeze()
        reference_tensor_aligned = warp(reference_tensor, flow).detach().cpu().numpy().squeeze()
        
    # Calcular error
    error = np.sqrt((pred - reference_frame)**2)
    sigma = np.sqrt(sigma)
    # Calcular estadísticas
    sigma_mean = np.mean(sigma)
    error_mean = np.mean(error)

    # Mostrar
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(sigma, cmap=cmap)
    axes[0].set_title("Desviación Estándar (σ)")
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].text(0.5, -0.1, f"Media σ: {sigma_mean:.4f}", ha='center', va='top', transform=axes[0].transAxes)

    im2 = axes[1].imshow(error, cmap=cmap)
    axes[1].set_title("Error Cuadrático (pred - gt)²")
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].text(0.5, -0.1, f"Media error: {error_mean:.4f}", ha='center', va='top', transform=axes[1].transAxes)

    plt.tight_layout()
    plt.show()


# visualize_alignment_toggle(y_pred_ds, reference_frame)

def plot_error_vs_variance(target, prediction, variance):
    """
    Parameters:
    - target: numpy array (H x W), ground truth image
    - prediction: numpy array (H x W), model prediction
    - variance: numpy array (H x W), predicted variance by the model
    """
    import numpy as np
    import matplotlib.pyplot as plt

    target = target.astype(np.float32)
    prediction = prediction.astype(np.float32)
    variance = variance.astype(np.float32)

    error = np.abs(target - prediction)  # Absolute error
    # error = np.sqrt(error)  # Optional: square root of error
    
    i, j = 110, 75  # Example pixel coordinates

    # --- Figure 1: 4 maps ---
    plt.figure(figsize=(16, 10))

    # Subplot 1: Ground Truth
    plt.subplot(2, 2, 1)
    plt.imshow(target, cmap='viridis')
    plt.colorbar(label='Value', shrink=0.7)
    plt.title('Ground Truth')
    plt.axis('off')

    # Subplot 2: Prediction
    plt.subplot(2, 2, 2)
    plt.imshow(prediction, cmap='viridis')
    plt.colorbar(label='Value', shrink=0.7)
    plt.title('Model Prediction')
    plt.axis('off')

    # Subplot 3: Absolute Error
    plt.subplot(2, 2, 3)
    plt.imshow(error, cmap='viridis')
    plt.colorbar(label='Error', shrink=0.7)
    plt.title('Absolute Error')
    plt.axis('off')
    print(f"  Pixel ({i},{j}) Absolute Error: {error[i,j]:.2f}")

    # Subplot 4: Variance
    plt.subplot(2, 2, 4)
    plt.imshow(variance, cmap='plasma', vmin=0, vmax=500)
    plt.colorbar(label='Predicted Variance (σ²)', shrink=0.7)
    plt.title('Variance Map')
    plt.axis('off')

    plt.suptitle('Comparison: Ground Truth, Prediction, Error and Variance', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # --- Figure 2: Scatter plot ---
    plt.figure(figsize=(8, 8))
    plt.scatter(variance.flatten(), error.flatten(), alpha=0.3, s=5, label='Pixels')
    plt.plot([0, 2000], [0, 2000], 'r--', label='Error = σ (ideal)')
    plt.xlabel('Predicted Variance (σ²)')
    plt.ylabel('Absolute Error')
    plt.title('Error vs. Predicted Variance')
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.legend()
    plt.grid(True)
    plt.show()


plot_error_vs_variance(y_true, y_pred, sigma)