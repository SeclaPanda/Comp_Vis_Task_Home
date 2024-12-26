import matplotlib.pyplot as plt
from collections import deque
import cv2
import numpy as np

import cv2
import numpy as np

def create_color_masks(hsv: np.ndarray) -> tuple:
    #Создать маски для синего и красного цветов.
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask_red = mask_red_1 | mask_red_2

    return mask_blue, mask_red

def find_contours(mask: np.ndarray) -> list:
    #Найти контуры на маске.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_lanes_with_obstacles(contours: list, width: int, num_lanes: int) -> list:
    #Определить полосы с препятствиями.
    lane_width = width // num_lanes
    lanes_with_obstacles = [False] * num_lanes

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        lane_index = x // lane_width
        if lane_index < num_lanes:
            lanes_with_obstacles[lane_index] = True

    return lanes_with_obstacles

def find_road_number(image: np.ndarray) -> int:
    #Найти номер полосы, на которой нет препятствия в конце пути.
    # Преобразование изображения в цветовое пространство HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Создание масок для синего и красного цветов
    _, mask_red = create_color_masks(hsv)

    # Нахождение контуров красного цвета
    contours_red = find_contours(mask_red)

    # Определение размеров изображения и количества полос
    height, width, _ = image.shape
    num_lanes = 5

    # Определение полос с препятствиями
    lanes_with_obstacles = detect_lanes_with_obstacles(contours_red, width, num_lanes)

    # Нахождение первой свободной полосы
    for i in range(num_lanes):
        if not lanes_with_obstacles[i]:
            return i

    return -1

# Загрузка тестового изображения
test_image = cv2.imread('HW_1/task_2/image_00.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# Нахождение номера полосы без препятствия
road_number = find_road_number(test_image)

# Вывод результата
print(f'Нужно перестроиться на дорогу номер {road_number}')
