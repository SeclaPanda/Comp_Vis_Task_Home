import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

def compute_hog(image: np.ndarray, pixels_per_cell: tuple = (8, 8), cells_per_block: tuple = (2, 2), visualize: bool = False) -> tuple:
    """
    Вычислить HOG признаки для изображения.

    :param image: входное изображение
    :param pixels_per_cell: количество пикселей на ячейку
    :param cells_per_block: количество ячеек на блок
    :param visualize: флаг для визуализации HOG признаков
    :return: HOG признаки, HOG изображение, изображение в оттенках серого, HOG изображение до улучшения контраста
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(gray_image, orientations=9, pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block, visualize=visualize)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd, hog_image_rescaled, gray_image, hog_image

def cross_correlation(hog_image: np.ndarray, hog_template: np.ndarray) -> tuple:
    """
    Вычислить кросс-корреляцию между HOG изображением и HOG шаблоном.

    :param hog_image: HOG изображение
    :param hog_template: HOG шаблон
    :return: координаты максимального совпадения и значение максимального совпадения
    """
    hog_image = np.float32(hog_image)
    hog_template = np.float32(hog_template)
    result = cv2.matchTemplate(hog_image, hog_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val

def visualize_template_transforms(template: np.ndarray, gray_template: np.ndarray, hog_template: np.ndarray, hog_template_rescaled: np.ndarray):
    """
    Визуализировать преобразования шаблона.

    :param template: исходный шаблон
    :param gray_template: шаблон в оттенках серого
    :param hog_template: HOG признаки шаблона (до контраста)
    :param hog_template_rescaled: HOG признаки шаблона (после контраста)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Исходный шаблон')
    axes[0].axis('off')

    axes[1].imshow(gray_template, cmap='gray')
    axes[1].set_title('Шаблон в серых тонах')
    axes[1].axis('off')

    axes[2].imshow(hog_template, cmap='gray')
    axes[2].set_title('HOG признаки (до контраста)')
    axes[2].axis('off')

    axes[3].imshow(hog_template_rescaled, cmap='gray')
    axes[3].set_title('HOG признаки (после контраста)')
    axes[3].axis('off')

    plt.show()

def visualize_results(image: np.ndarray, template: np.ndarray, max_loc: tuple, max_val: float, index: int, hog_image: np.ndarray = None, hog_template: np.ndarray = None, gray_image: np.ndarray = None, gray_template: np.ndarray = None):
    """
    Визуализировать результаты поиска шаблона на изображении.

    :param image: исходное изображение
    :param template: шаблон
    :param max_loc: координаты максимального совпадения
    :param max_val: значение максимального совпадения
    :param index: индекс изображения
    :param hog_image: HOG изображение
    :param hog_template: HOG шаблон
    :param gray_image: изображение в оттенках серого
    :param gray_template: шаблон в оттенках серого
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')

    axes[1].imshow(gray_image, cmap='gray')
    axes[1].set_title('Изображение в серых тонах')
    axes[1].axis('off')

    axes[2].imshow(hog_image, cmap='gray')
    axes[2].set_title('HOG признаки (до контраста)')
    axes[2].axis('off')

    axes[3].imshow(hog_image, cmap='gray')
    axes[3].set_title('HOG признаки (после контраста)')
    axes[3].axis('off')

    axes[4].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[4].set_title('Исходный шаблон')
    axes[4].axis('off')

    h, w = template.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Найден шаблон с точностью {max_val:.2f}')
    plt.axis('off')
    plt.show()

def save_image(image: np.ndarray, file_path: str):
    """
    Сохранить изображение в указанный файл.

    :param image: изображение для сохранения
    :param file_path: путь к файлу для сохранения
    """
    cv2.imwrite(file_path, image)

def main():
    template_path = 'HW_4/hog/eye.png'
    image_paths = ['HW_4/hog/eyes.jpg', 'HW_4/hog/eyes_dark.jpg', 'HW_4/hog/woman.jpg']

    template = cv2.imread(template_path)
    if template is None:
        print(f"Не удалось загрузить шаблон: {template_path}")
        return

    _, hog_template, gray_template, hog_template_rescaled = compute_hog(template, visualize=True)
    visualize_template_transforms(template, gray_template, hog_template, hog_template_rescaled)

    for index, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            continue

        _, hog_image, gray_image, _ = compute_hog(image, visualize=True)
        max_loc, max_val = cross_correlation(hog_image, hog_template)

        visualize_results(image, template, max_loc, max_val, index, hog_image=hog_image,
                          hog_template=hog_template, gray_image=gray_image, gray_template=gray_template)

        # Сохранение промежуточных изображений
        save_image(gray_image, f'HW_4/hog/gray_image_{index}.png')
        save_image(hog_image, f'HW_4/hog/hog_image_{index}.png')
        save_image(image, f'HW_4/hog/result_image_{index}.png')

if __name__ == '__main__':
    main()
