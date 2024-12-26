import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

# Настройки matplotlib
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # Установить размер изображений по умолчанию
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def load_and_show_image(image_path: str):
    img = cv2.imread(image_path, 0)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return img

def conv_nested(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image = np.pad(image, Hk // 2)

    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for l in range(Wk):
                    out[i, j] += image[i + k, j + l] * kernel[Hk - 1 - k, Wk - 1 - l]

    return out

def zero_pad(image: np.ndarray, pad_height: int, pad_width: int) -> np.ndarray:
    H, W = image.shape
    out = np.zeros_like(image)

    if pad_width > pad_height:
        out = np.pad(image, pad_height)
        h, w = out.shape
        sup = pad_width - pad_height
        out = np.insert(out, [0] * sup, np.zeros(h), axis=1)
        out = np.insert(out, [-1] * sup, np.zeros(h), axis=1)
    else:
        out = np.pad(image, pad_width)
        h, w = out.shape
        sup = pad_height - pad_width
        out = np.insert(out, [0] * sup, np.zeros(w), axis=0)
        out = np.insert(out, [-1] * sup, np.zeros(w), axis=0)

    return out

def conv_fast(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    img_pad = zero_pad(image, Hk // 2, Wk // 2)
    kernel = np.flip(kernel)

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(img_pad[i:i + Hk, j:j + Wk] * kernel)

    return out

def save_image(image: np.ndarray, file_path: str):
    cv2.imwrite(file_path, image)

# Загрузка и отображение исходного изображения
image_path = 'HW_2/img/dog.jpg'
img = load_and_show_image(image_path)

# Простое ядро свертки
kernel = np.array([
    [1, 0, 1],
    [0, 0, 0],
    [1, 0, 0]
])

# Создание тестового изображения: белый квадрат в центре
test_img = np.zeros((9, 9))
test_img[3:6, 3:6] = 1

# Применение свертки к тестовому изображению
test_output = conv_nested(test_img, kernel)

# Построение ожидаемого выхода
expected_output = np.zeros((9, 9))
expected_output[2:7, 2:7] = 1
expected_output[5:, 5:] = 0
expected_output[4, 2:5] = 2
expected_output[2:5, 4] = 2
expected_output[4, 4] = 3

# Отображение тестового изображения, результата свертки и ожидаемого выхода
plt.subplot(1, 3, 1)
plt.imshow(test_img)
plt.title('Test image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(test_output)
plt.title('Convolution')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(expected_output)
plt.title('Expected output')
plt.axis('off')
plt.show()

# Применение свертки к исходному изображению
kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

out = conv_nested(img, kernel)

# Отображение исходного изображения, результата свертки и ожидаемого выхода
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(out)
plt.title('Convolution')
plt.axis('off')

solution_img = cv2.imread('HW_2/img/convoluted_dog.jpg', 0)
plt.subplot(2, 2, 4)
plt.imshow(solution_img)
plt.title('What you should get')
plt.axis('off')
plt.show()

# Сохранение результата свертки
save_image(out, 'HW_2/img/convoluted_dog_output.jpg')

# Добавление нулевой обводки к изображению
pad_width = 20
pad_height = 40
padded_img = zero_pad(img, pad_height, pad_width)
print(padded_img.shape)

# Отображение изображения с обводкой и ожидаемого выхода
plt.subplot(1, 2, 1)
plt.imshow(padded_img)
plt.title('Padded dog')
plt.axis('off')

solution_img = cv2.imread('HW_2/img/padded_dog.jpg', 0)
plt.subplot(1, 2, 2)
plt.imshow(solution_img)
plt.title('What you should get')
plt.axis('off')
plt.show()

# Сохранение изображения с обводкой
save_image(padded_img, 'HW_2/img/padded_dog_output.jpg')

# Сравнение времени выполнения двух реализаций свертки

t0 = time()
out_fast = conv_fast(img, kernel)
t1 = time()
out_nested = conv_nested(img, kernel)
t2 = time()

print("conv_nested: took %f seconds." % (t2 - t1))
print("conv_fast: took %f seconds." % (t1 - t0))

# Отображение результатов свертки
plt.subplot(1, 2, 1)
plt.imshow(out_nested)
plt.title('conv_nested')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(out_fast)
plt.title('conv_fast')
plt.axis('off')
plt.show()

# Сохранение результатов свертки
save_image(out_nested, 'HW_2/img/conv_nested_output.jpg')
save_image(out_fast, 'HW_2/img/conv_fast_output.jpg')
