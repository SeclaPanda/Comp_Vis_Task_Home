import cv2
import numpy as np

def calculate_new_borders(M, h: int, w: int) -> tuple:
    """
    Находим крайние точки после преобразования и из них считаем новые высоту и ширину картинки.

    :param M: матрица преобразования
    :param h: высота изображения
    :param w: ширина изображения
    :return: новые границы и размеры изображения
    """
    corners = np.array([
        [0, 0, 1],
        [0, h, 1],
        [w, 0, 1],
        [w, h, 1]
    ])

    transformed_corners = M @ corners.T
    low_w = np.min(transformed_corners[0])
    high_w = np.max(transformed_corners[0])
    new_w = int(np.round(high_w - low_w))

    low_h = np.min(transformed_corners[1])
    high_h = np.max(transformed_corners[1])
    new_h = int(np.round(high_h - low_h))

    return (low_w, low_h), (new_w, new_h)

def rotate_image(image: np.ndarray, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    h, w, _ = image.shape
    M_rotate = cv2.getRotationMatrix2D(point, angle, scale=1.0)
    low, new_shp = calculate_new_borders(M_rotate, h, w)
    M_rotate[0, 2] -= low[0]
    M_rotate[1, 2] -= low[1]
    return cv2.warpAffine(image, M_rotate, (new_shp[0], new_shp[1]))

def find_contours(image: np.ndarray) -> tuple:
    """
    Находит контуры на изображении.

    :param image: входное изображение
    :return: изображение с наложенными контурами и сами контуры
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 10, 10)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)  # Рисуем контуры красным цветом
    return contour_image, contours

def apply_perspective_transform(image: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Применить перспективное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image: исходное изображение
    :param points1: начальные точки
    :param points2: конечные точки
    :return: преобразованное изображение
    """
    transformation_matrix = cv2.getPerspectiveTransform(points1.astype(np.float32), points2.astype(np.float32))
    width = int(max(np.linalg.norm(points1[0] - points1[1]), np.linalg.norm(points1[2] - points1[3])))
    height = int(max(np.linalg.norm(points1[0] - points1[3]), np.linalg.norm(points1[1] - points1[2])))
    return cv2.warpPerspective(image, transformation_matrix, (width, height))

# Загрузка тестового изображения
test_image = cv2.imread('HW_1/task_3/lk.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# Поворот изображения
test_point = (200, 200)
test_angle = 15
transformed_image = rotate_image(test_image, test_point, test_angle)

# Сохранение изображения в ту же папку
output_path = 'HW_1/task_3/transformed_lk.jpg'
cv2.imwrite(output_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
