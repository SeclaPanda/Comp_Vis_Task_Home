import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)

def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def create_masks(hsv_image: np.ndarray) -> tuple:
    forest_lower = np.array([35, 25, 25])
    forest_upper = np.array([85, 255, 255])
    desert_lower = np.array([15, 25, 25])
    desert_upper = np.array([35, 255, 255])

    forest_mask = cv2.inRange(hsv_image, forest_lower, forest_upper)
    desert_mask = cv2.inRange(hsv_image, desert_lower, desert_upper)

    return forest_mask, desert_mask

def count_pixels(mask: np.ndarray) -> int:
    return cv2.countNonZero(mask)

def classify_image(image_path: str) -> tuple:
    image = load_image(image_path)
    hsv_image = convert_to_hsv(image)
    forest_mask, desert_mask = create_masks(hsv_image)

    forest_count = count_pixels(forest_mask)
    desert_count = count_pixels(desert_mask)

    classification = "Forest" if forest_count > desert_count else "Desert"

    return classification, image, forest_mask, desert_mask

def show_images(image: np.ndarray, forest_mask: np.ndarray, desert_mask: np.ndarray, classification: str, title: str):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'{title} - Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(forest_mask, cmap='gray')
    plt.title('Forest Mask')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(desert_mask, cmap='gray')
    plt.title('Desert Mask')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Classification: {classification}')
    plt.axis('off')

    plt.show()

def save_image(image: np.ndarray, file_path: str):
    cv2.imwrite(file_path, image)

# Тестовые изображения
image_path_forest = 'HW_3/desert_forest/test_image_00.jpg'
classification_forest, image_forest, forest_mask_forest, desert_mask_forest = classify_image(image_path_forest)
print(f"Classification: {classification_forest}")

image_path_desert = 'HW_3/desert_forest/test_image_04.jpg'
classification_desert, image_desert, forest_mask_desert, desert_mask_desert = classify_image(image_path_desert)
print(f"Classification: {classification_desert}")

# Отображение изображений
show_images(image_forest, forest_mask_forest, desert_mask_forest, classification_forest, 'Test Image 00')
show_images(image_desert, forest_mask_desert, desert_mask_desert, classification_desert, 'Test Image 04')

# Сохранение изображений
save_image(image_forest, 'HW_3/desert_forest/test_image_00_classified.jpg')
save_image(image_desert, 'HW_3/desert_forest/test_image_04_classified.jpg')
