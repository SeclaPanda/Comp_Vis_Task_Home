import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_haar_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('HW_3/haar/haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier('HW_3/haar/haarcascade_mcs_mouth.xml')

    if face_cascade.empty() or eye_cascade.empty() or nose_cascade.empty() or mouth_cascade.empty():
        raise ValueError("Не удалось загрузить один из каскадов Хаара. Проверьте пути к файлам каскадов.")

    return face_cascade, eye_cascade, nose_cascade, mouth_cascade

def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение по пути {image_path}")
    return img

def convert_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_face_parts(img_path: str, face_cascade, eye_cascade, nose_cascade, mouth_cascade) -> tuple:
    img = load_image(img_path)
    gray = convert_to_gray(img)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    mask = np.zeros_like(gray)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.ellipse(mask, ((ex + ew//2 + x), (ey + eh//2 + y)), (ew//2, eh//2), 0, 0, 360, 255, -1)

        nose = nose_cascade.detectMultiScale(roi_gray)
        for (nx, ny, nw, nh) in nose:
            cv2.ellipse(mask, ((nx + nw//2 + x), (ny + nh//2 + y)), (nw//2, nh//2), 0, 0, 360, 255, -1)

        mouth = mouth_cascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouth:
            cv2.ellipse(mask, ((mx + mw//2 + x), (my + mh//2 + y)), (mw//2, mh//2), 0, 0, 360, 255, -1)

    return img, mask

def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height))

def apply_seamless_cloning(src_img: np.ndarray, dst_img: np.ndarray, src_mask: np.ndarray, center: tuple) -> np.ndarray:
    return cv2.seamlessClone(src_img, dst_img, src_mask, center, cv2.NORMAL_CLONE)

def show_images(src_img: np.ndarray, dst_img: np.ndarray, src_resized: np.ndarray, seamless_img: np.ndarray):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
    plt.title('Source Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB))
    plt.title('Destination Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(src_resized, cv2.COLOR_BGR2RGB))
    plt.title('Resized Source Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(seamless_img, cv2.COLOR_BGR2RGB))
    plt.title('Seamless Cloned Image')
    plt.axis('off')

    plt.show()

def save_image(image: np.ndarray, file_path: str):
    cv2.imwrite(file_path, image)

# Загрузка каскадов Хаара
face_cascade, eye_cascade, nose_cascade, mouth_cascade = load_haar_cascades()

# Загрузка изображений и обнаружение частей лица
src_img, src_mask = detect_face_parts("HW_3/faces/face_1.jpg", face_cascade, eye_cascade, nose_cascade, mouth_cascade)
dst_img, dst_mask = detect_face_parts("HW_3/faces/face_2.jpg", face_cascade, eye_cascade, nose_cascade, mouth_cascade)

# Приведение изображений к одному размеру
height = dst_img.shape[0]
width = dst_img.shape[1]
src_resized = resize_image(src_img, width, height)
src_mask_resized = resize_image(src_mask, width, height)

# Центр для применения метода seamless cloning
center = (width // 2, height // 2)

# Применение метода seamless cloning
seamless_img = apply_seamless_cloning(src_resized, dst_img, src_mask_resized, center)

# Отображение изображений
show_images(src_img, dst_img, src_resized, seamless_img)

# Сохранение результата
save_image(seamless_img, 'HW_3/faces/seamless_cloned_image.jpg')
