import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.patches as patches

def load_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    return np.array(image)

def extract_cells(image_np: np.ndarray, cell_size: int, cell_count: int) -> list:
    img_height, img_width, _ = image_np.shape
    cells = []
    extracted_cells = 0

    for y in range(0, img_height, cell_size):
        for x in range(0, img_width, cell_size):
            if extracted_cells >= cell_count:
                break

            cell = image_np[y:y + cell_size, x:x + cell_size]
            if cell.mean() < 240:  # порог для фильтрации пустых участков
                cells.append(cell)
                extracted_cells += 1

    return cells

def display_extracted_cells(image_np: np.ndarray, cells: list, cell_size: int):

    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    for idx, cell in enumerate(cells):
        y = (idx // (image_np.shape[1] // cell_size)) * cell_size
        x = (idx % (image_np.shape[1] // cell_size)) * cell_size
        rect = patches.Rectangle((x, y), cell_size, cell_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.title("Исходное изображение с выделенными ячейками")
    plt.show()

def create_mosaic(cells: list, cell_size: int, padding: int = 10) -> np.ndarray:
    if len(cells) > 16:
        cells = cells[:16]
    elif len(cells) < 16:
        cells = cells * (16 // len(cells)) + cells[:16 % len(cells)]

    grid_size = 4
    mosaic_size = grid_size * cell_size + (grid_size - 1) * padding
    mosaic = np.ones((mosaic_size, mosaic_size, 3), dtype=np.uint8) * 255  # фон белого цвета

    for idx, cell in enumerate(cells):
        i = idx // grid_size
        j = idx % grid_size
        y_start = i * (cell_size + padding)
        x_start = j * (cell_size + padding)
        mosaic[y_start:y_start + cell_size, x_start:x_start + cell_size] = cell

    return mosaic

def save_image(image: np.ndarray, file_path: str):
    Image.fromarray(image).save(file_path)

def extract_and_display_cells(image_path: str, cell_size: int, cell_count: int, padding: int = 10):
    image_np = load_image(image_path)
    cells = extract_cells(image_np, cell_size, cell_count)
    display_extracted_cells(image_np, cells, cell_size)
    mosaic = create_mosaic(cells, cell_size, padding)

    plt.figure(figsize=(8, 8))
    plt.imshow(mosaic)
    plt.axis('off')
    plt.title("Мозаика")
    plt.show()

    # Сохранение мозаики
    save_image(mosaic, 'HW_3/cells/mosaic.jpg')

# Пример использования
extract_and_display_cells("HW_3/cells/train1_1.jpeg", cell_size=256, cell_count=56)
