import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage import measure
import os
import matplotlib.pyplot as plt

def load_images_and_masks(image_dir: str, mask_dir: str) -> tuple:
    """
    Загрузить изображения и маски из указанных директорий.

    :param image_dir: директория с изображениями
    :param mask_dir: директория с масками
    :return: списки изображений и масок
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Директория не найдена: {image_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Директория не найдена: {mask_dir}")

    images = []
    masks = []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    if len(image_files) != len(mask_files):
        print(f"Файлы в {image_dir}: {image_files}")
        print(f"Файлы в {mask_dir}: {mask_files}")
        raise ValueError("Количество изображений и масок должно быть одинаковым.")

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Файл не найден: {img_path}")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Файл не найден: {mask_path}")

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        if mask is None:
            raise ValueError(f"Не удалось загрузить маску: {mask_path}")

        images.append(image)
        masks.append(mask)

    return images, masks

def extract_tail(image: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Извлечь хвост из изображения на основе маски.

    :param image: изображение
    :param mask: маска
    :return: извлеченный хвост и его маска
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    tail = image[y:y+h, x:x+w]
    tail_mask = mask[y:y+h, x:x+w]

    return tail, tail_mask

def normalize_tail(tail: np.ndarray, tail_mask: np.ndarray, target_size: tuple = (128, 128)) -> np.ndarray:
    """
    Нормализовать извлеченный хвост.

    :param tail: извлеченный хвост
    :param tail_mask: маска извлеченного хвоста
    :param target_size: целевой размер для нормализации
    :return: нормализованный хвост
    """
    contours, _ = cv2.findContours(tail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)

    rows, cols, _ = tail.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_tail = cv2.warpAffine(tail, M, (cols, rows))
    rotated_mask = cv2.warpAffine(tail_mask, M, (cols, rows))

    contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_tail = rotated_tail[y:y+h, x:x+w]

    normalized_tail = cv2.resize(cropped_tail, target_size)

    return normalized_tail

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Вычислить IoU (Intersection over Union) для двух масок.

    :param mask1: первая маска
    :param mask2: вторая маска
    :return: значение IoU
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def visualize_results(image: np.ndarray, mask: np.ndarray, tail: np.ndarray, tail_mask: np.ndarray, normalized_tail: np.ndarray, gt_mask: np.ndarray, iou_score: float, index: int):
    """
    Визуализировать результаты обработки изображения.

    :param image: исходное изображение
    :param mask: исходная маска
    :param tail: извлеченный хвост
    :param tail_mask: маска извлеченного хвоста
    :param normalized_tail: нормализованный хвост
    :param gt_mask: истинная маска
    :param iou_score: значение IoU
    :param index: индекс изображения
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Исходное изображение')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Исходная маска')
    axes[0, 1].axis('off')

    if tail is not None and tail_mask is not None:
        axes[0, 2].imshow(cv2.cvtColor(tail, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Вырезанный хвост')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(tail_mask, cmap='gray')
        axes[1, 0].set_title('Маска вырезанного хвоста')
        axes[1, 0].axis('off')
    else:
        axes[0, 2].set_title('Вырезанный хвост (None)')
        axes[0, 2].axis('off')

        axes[1, 0].set_title('Маска вырезанного хвоста (None)')
        axes[1, 0].axis('off')

    axes[1, 1].imshow(normalized_tail)
    axes[1, 1].set_title('Нормализованный хвост')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(gt_mask, cmap='gray')
    axes[1, 2].set_title(f'Истинная маска\nIoU: {iou_score:.2f}')
    axes[1, 2].axis('off')

    plt.suptitle(f'Изображение {index}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    image_dir = 'HW_4/whale_tail/images'
    mask_dir = 'HW_4/whale_tail/ground_truth'

    images, masks = load_images_and_masks(image_dir, mask_dir)

    target_size = (128, 128)
    iou_scores = []

    if not os.path.exists('normalized_tails'):
        os.makedirs('normalized_tails')

    for i, (image, mask) in enumerate(zip(images, masks)):
        tail, tail_mask = extract_tail(image, mask)
        if tail is None or tail_mask is None:
            iou_scores.append(0)
            continue

        normalized_tail = normalize_tail(tail, tail_mask, target_size)

        gt_mask = cv2.resize(mask, target_size)
        gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)[1]

        iou_score = calculate_iou(gt_mask, cv2.threshold(normalized_tail[:, :, 0], 127, 255, cv2.THRESH_BINARY)[1])
        iou_scores.append(iou_score)

        cv2.imwrite(f'normalized_tails/tail_{i}.png', normalized_tail)

        visualize_results(image, mask, tail, tail_mask, normalized_tail, gt_mask, iou_score, i)

    print(f'IoU scores: {iou_scores}')
    print(f'Average IoU: {np.mean(iou_scores)}')

if __name__ == '__main__':
    main()
