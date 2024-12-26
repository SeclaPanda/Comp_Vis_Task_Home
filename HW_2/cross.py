import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

def zero_pad(image: np.ndarray, pad_height: int, pad_width: int) -> np.ndarray:
    """
    Добавить нулевую обводку к изображению.

    :param image: входное изображение
    :param pad_height: высота обводки
    :param pad_width: ширина обводки
    :return: изображение с обводкой
    """
    return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

def cross_correlation(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Кросс-корреляция между изображением и шаблоном.

    :param f: изображение
    :param g: шаблон
    :return: результат кросс-корреляции
    """
    tg = g.astype(np.float64)
    ig = f.astype(np.float64)

    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    img_pad = zero_pad(ig, Hk // 2, Wk // 2)
    sum_sq_tg = np.sum(tg ** 2)

    for i in range(Hi):
        for j in range(Wi):
            img_slice = img_pad[i:i + Hk, j:j + Wk]
            koeff = np.sqrt(sum_sq_tg * np.sum(img_slice ** 2))
            out[i, j] = np.sum(img_slice * tg) / koeff

    return out

def zero_mean_cross_correlation(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Кросс-корреляция с нулевым средним значением между изображением и шаблоном.

    :param f: изображение
    :param g: шаблон
    :return: результат кросс-корреляции с нулевым средним значением
    """
    temp = g - np.mean(g)
    return cross_correlation(f, temp)

def normalized_cross_correlation(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Нормализованная кросс-корреляция между изображением и шаблоном.

    :param f: изображение
    :param g: шаблон
    :return: результат нормализованной кросс-корреляции
    """
    tg = g.astype(np.float64)
    ig = f.astype(np.float64)

    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    img_pad = zero_pad(ig, Hk // 2, Wk // 2)

    sigma = np.std(tg)
    mid = np.mean(tg)
    norm_tg = (tg - mid) / sigma
    sum_sq_tg = np.sum(tg ** 2)

    for i in range(Hi):
        for j in range(Wi):
            img_slice = img_pad[i:i + Hk, j:j + Wk]
            koeff = np.sqrt(sum_sq_tg * np.sum(img_slice ** 2))
            out[i, j] = np.sum(((img_slice - np.mean(img_slice)) / np.std(img_slice)) * norm_tg) / koeff

    return out

def check_product_on_shelf(shelf: np.ndarray, product: np.ndarray):
    """
    Проверить наличие продукта на полке.

    :param shelf: изображение полки
    :param product: изображение продукта
    """
    out = zero_mean_cross_correlation(shelf, product)
    out = out / float(product.shape[0] * product.shape[1])
    out = out > 0.025

    if np.sum(out) > 0:
        print('The product is on the shelf')
    else:
        print('The product is not on the shelf')

def save_image(image: np.ndarray, file_path: str):
    """
    Сохранить изображение в указанный файл.

    :param image: изображение для сохранения
    :param file_path: путь к файлу для сохранения
    """
    cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Загрузка изображений
img = cv2.imread('HW_2\\img\\shelf.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
temp = cv2.imread('HW_2\\img\\template.jpg')
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
temp_grey = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

# Кросс-корреляция между изображением и шаблоном
out = cross_correlation(img_grey, temp_grey)
y, x = np.unravel_index(out.argmax(), out.shape)

plt.figure(figsize=(25, 20))
plt.subplot(311), plt.imshow(temp), plt.title('Template'), plt.axis('off')
plt.subplot(312), plt.imshow(img), plt.title('Result (blue marker on the detected location)'), plt.axis('off')
plt.subplot(313), plt.imshow(out), plt.title('Cross-correlation (white means more correlated)'), plt.axis('off')
plt.plot(x, y, 'bx', ms=40, mew=10)
plt.show()

# Сохранение результата
save_image(img, 'HW_2\\img\\shelf_result.jpg')

# Кросс-корреляция с нулевым средним значением
out = zero_mean_cross_correlation(img_grey, temp_grey)
y, x = np.unravel_index(out.argmax(), out.shape)

plt.figure(figsize=(30, 20))
plt.subplot(311), plt.imshow(temp), plt.title('Template'), plt.axis('off')
plt.subplot(312), plt.imshow(img), plt.title('Result (blue marker on the detected location)'), plt.axis('off')
plt.subplot(313), plt.imshow(out), plt.title('Cross-correlation (white means more correlated)'), plt.axis('off')
plt.plot(x, y, 'bx', ms=40, mew=10)
plt.show()

# Сохранение результата
save_image(img, 'HW_2\\img\\shelf_zero_mean_result.jpg')

# Проверка наличия продукта на полке
img2 = cv2.imread('HW_2\\img\\shelf_soldout.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_grey = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(10, 5))
plt.imshow(img), plt.axis('off'), plt.show()
check_product_on_shelf(img_grey, temp_grey)

plt.figure(figsize=(10, 5))
plt.imshow(img2), plt.axis('off'), plt.show()
check_product_on_shelf(img2_grey, temp_grey)

# Нормализованная кросс-корреляция
img_dark = cv2.imread('HW_2\\img\\shelf_dark.jpg')
img_dark = cv2.cvtColor(img_dark, cv2.COLOR_BGR2RGB)
img_dark_grey = cv2.cvtColor(img_dark, cv2.COLOR_RGB2GRAY)

out = zero_mean_cross_correlation(img_dark_grey, temp_grey)
y, x = np.unravel_index(out.argmax(), out.shape)

plt.figure(figsize=(10, 5))
plt.imshow(img_dark), plt.title('Result (red marker on the detected location)'), plt.axis('off')
plt.plot(x, y, 'rx', ms=25, mew=5)
plt.show()

# Сохранение результата
save_image(img_dark, 'HW_2\\img\\shelf_dark_result.jpg')

# Нормализованная кросс-корреляция с использованием внешней функции

out = normalized_cross_correlation(img_dark_grey, temp_grey)
y, x = np.unravel_index(out.argmax(), out.shape)

plt.figure(figsize=(10, 5))
plt.imshow(img_dark), plt.title('Result (red marker on the detected location)'), plt.axis('off')
plt.plot(x, y, 'rx', ms=25, mew=5)
plt.show()

# Сохранение результата
save_image(img_dark, 'HW_2\\img\\shelf_dark_normalized_result.jpg')
