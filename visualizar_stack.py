import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_input_frames_interactive(input_stack_path):
    """
    Visualiza los frames de un stack de input uno a uno usando un slider interactivo.
    
    Parámetros:
        input_stack_path (str): Ruta del archivo TIFF que contiene el stack de input.
    """
    # Cargar el stack de input desde el archivo TIFF
    input_stack_np = tifffile.imread(input_stack_path)  # Shape: [num_frames, height, width]

    # Verificar la forma del stack
    print(f"Forma del stack de input: {input_stack_np.shape}")

    # Crear la figura y la subtrama para mostrar el frame
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)  # Ajustar el espacio para el slider

    # Mostrar el primer frame inicialmente
    frame_idx = 0
    im = ax.imshow(input_stack_np[frame_idx], cmap='gray')
    plt.title(f"Frame {frame_idx + 1} / {input_stack_np.shape[0]}")

    # Crear un slider para navegar entre los frames
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Posición y tamaño del slider
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=input_stack_np.shape[0] - 1,
        valinit=0,
        valstep=1
    )

    # Función para actualizar la imagen cuando se mueve el slider
    def update(val):
        frame_idx = int(slider.val)
        im.set_data(input_stack_np[frame_idx])
        plt.title(f"Frame {frame_idx + 1} / {input_stack_np.shape[0]}")
        fig.canvas.draw_idle()

    # Conectar el slider a la función de actualización
    slider.on_changed(update)

    # Mostrar la ventana interactiva
    plt.show()

if __name__ == "__main__":
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description="Visualizar los frames de input uno a uno con un slider.")
    parser.add_argument("input_stack", type=str, help="Ruta del archivo TIFF que contiene el stack de input.")
    
    # Obtener los argumentos
    args = parser.parse_args()
    
    # Llamar a la función para visualizar los frames de input
    visualize_input_frames_interactive(args.input_stack)