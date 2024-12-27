# CV_course_work

# В данном репозитории находится Курсовая работа по курсу CV

# Выполнил студент 2 курса магистратуры, Московского Авиационного Института (НИУ МАИ), группы 209М, Баклашкин Алексей Андреевич.

Работа представляет собой приложение по классификации самолетов на фото и видео.

Для реализации поставленной задачи выбрана предобученная модель YOLO - yolov10b

# Запуск и развёртывание

1. Сколинть данный репозиторий.
2. Установить необходимые зависимости (requirements.txt) (pip install ultralytics fastapi uvicorn python-multipart opencv-python).
3. Прописать в терминале команду python -m uvicorn app:app --reload  - запуск сервера.
4. В терминале появится ссылка на http://127.0.0.1:8000/ перейти по ней и далее открывается окно для загрузки фото и видео, после загрузки выводится результат классификации.

## Пример использования программы:

# *Интерфейс*

![главный экран](https://github.com/SeclaPanda/Comp_Vis_Task_Home/blob/main/term_paper/screenshot/main_page.png)


# *Детекция на видео*
![видео](https://github.com/SeclaPanda/Comp_Vis_Task_Home/blob/main/term_paper/screenshot/video.png)

# *Детекция на фото*

![пингвин](https://github.com/SeclaPanda/Comp_Vis_Task_Home/blob/main/term_paper/screenshot/ready.png)

# *Детекция на фото с слишком большим кол-вом объектов (шумным)*

![пингвин](https://github.com/SeclaPanda/Comp_Vis_Task_Home/blob/main/term_paper/screenshot/noise.png)


