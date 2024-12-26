import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

def plot_one_image(image: np.ndarray) -> None:
    fig, axs = plt.subplots(1, 1, figsize=(8, 7))
    axs.imshow(image)
    axs.axis('off')
    plt.show()

def find_way_from_maze(image: np.ndarray) -> tuple:
    # Преобразование изображения в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    height, width = binary.shape

    # Найти начальную и конечную точки
    start = (0, np.where(binary[0] == 255)[0][0])
    end = (height - 1, np.where(binary[-1] == 255)[0][0])

    # Возможные движения (вверх, вниз, влево, вправо)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue = deque([start])
    visited = set([start])
    prev = {start: None}

    while queue:
        current = queue.popleft()

        if current == end:
            break

        for move in moves:
            neighbor = (current[0] + move[0], current[1] + move[1])

            if (0 <= neighbor[0] < height) and (0 <= neighbor[1] < width) and binary[neighbor] == 255 and neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                prev[neighbor] = current

    path = []
    if end in prev:
        step = end
        while step:
            path.append(step)
            step = prev[step]

    if path:
        path = path[::-1]
        x_coords, y_coords = zip(*path)
        return np.array(x_coords), np.array(y_coords)
    else:
        return None

def plot_maze_path(image: np.ndarray, coords: tuple) -> np.ndarray:
    img_wpath = image.copy()
    if coords:
        x, y = coords
        img_wpath[x, y, :] = [0, 0, 255]  # Красный цвет для пути

    return img_wpath

# Загрузка тестового изображения
test_image = cv2.imread("HW_1/task_1/25 by 22 orthogonal maze.png")

# Отображение тестового изображения
plot_one_image(test_image)

# Вычисление координат пути через лабиринт
way_coords = find_way_from_maze(test_image)

# Нарисовать путь на изображении
image_with_way = plot_maze_path(test_image, way_coords)

# Сохранение изображения с путем
cv2.imwrite('HW_1/task_1/maze_solution.jpg', image_with_way)
