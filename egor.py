from __future__ import annotations
from abc import ABC, abstractmethod
from random import randrange
from typing import List
from models import *
from easyocr import Reader as Rd
from utils import *
import yaml
# results = reader.readtext(image)
# cv2.imwrite("/home/ea/projects/SIRIUS21/ocr.pytorch/test_images/tt.png", warped_img)
# warped_img = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
# warped_img = enlarge_img(warped_img, 200)
# print(pytesseract.image_to_string(warped_img, config = r'--psm 8'))
# show_image(warped_img
#----------------------------------------------------------------------------------------------
#######################################################################################################################
"""
img = "/home/home/projects/neural_backend/samples/problem1.png"

osr = Rd(["en"], gpu=True, verbose=False)
results = osr.readtext(img, adjust_contrast = 0.7)

for (bbox, text, prob) in results:
    print(text, ' and', prob)


'''self, img, min_size = 20, text_threshold = 0.7, low_text = 0.4,\
   link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,\
   slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
   width_ths = 0.5, add_margin = 0.1, reformat=True, optimal_num_chars=None'''
"""
#######################################################################################################################
# old--------------------------------------------------------------------------------------------------------------
#hile True:

#   _, img = cap.read()

#   warped_img = img
#   # warped_img = warp_image(img, np.array(eval(str([(215, 686), (788, 468),(789, 682), (290, 953)])),dtype="float32"))

#   results = model.predict(warped_img)
#   for (bbox, text, prob) in results:

#       (tl, tr, br, bl) = bbox
#       tl = (int(tl[0]), int(tl[1]))
#       tr = (int(tr[0]), int(tr[1]))
#       br = (int(br[0]), int(br[1]))
#       bl = (int(bl[0]), int(bl[1]))

#       cv2.rectangle(warped_img, tl, br, (0, 255, 0), 2)
#       cv2.putText(warped_img, text, (tl[0], tl[1] - 10),
#       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#   show_image(warped_img)

#######################################################################################################################
"""
save = {0: 1}
ded = OCRReader("/home/home/projects/neural_backend/samples/numberOCR.jpg", type='img', all_info=save)
while True:
    ded.image_run(watch=False)
"""
#print(os.getpid())
######################################################################################################################
'''


class Subject(ABC):
    """
    Интферфейс издателя объявляет набор методов для управлениями подписчиками.
    """

    @abstractmethod
    def attach(self, observer: Observer) -> None:
        """
        Присоединяет наблюдателя к издателю.
        """
        pass

    @abstractmethod
    def detach(self, observer: Observer) -> None:
        """
        Отсоединяет наблюдателя от издателя.
        """
        pass

    @abstractmethod
    def notify(self) -> None:
        """
        Уведомляет всех наблюдателей о событии.
        """
        pass


class ConcreteSubject(Subject):
    """
    Издатель владеет некоторым важным состоянием и оповещает наблюдателей о его
    изменениях.
    """

    _state: int = None
    """
    Для удобства в этой переменной хранится состояние Издателя, необходимое всем
    подписчикам.
    """

    _observers: List[Observer] = []
    """
    Список подписчиков. В реальной жизни список подписчиков может храниться в
    более подробном виде (классифицируется по типу события и т.д.)
    """

    def attach(self, observer: Observer) -> None:
        print("Subject: Attached an observer.")
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    """
    Методы управления подпиской.
    """

    def notify(self) -> None:
        """
        Запуск обновления в каждом подписчике.
        """

        print("Subject: Notifying observers...")
        for observer in self._observers:
            observer.update(self)

    def some_business_logic(self) -> None:
        """
        Обычно логика подписки – только часть того, что делает Издатель.
        Издатели часто содержат некоторую важную бизнес-логику, которая
        запускает метод уведомления всякий раз, когда должно произойти что-то
        важное (или после этого).
        """

        print("\nSubject: I'm doing something important.")
        self._state = randrange(0, 10)

        print(f"Subject: My state has just changed to: {self._state}")
        self.notify()


class Observer(ABC):
    """
    Интерфейс Наблюдателя объявляет метод уведомления, который издатели
    используют для оповещения своих подписчиков.
    """

    @abstractmethod
    def update(self, subject: Subject) -> None:
        """
        Получить обновление от субъекта.
        """
        pass


"""
Конкретные Наблюдатели реагируют на обновления, выпущенные Издателем, к которому
они прикреплены.
"""


class ConcreteObserverA(Observer):
    def update(self, subject: Subject) -> None:
        if subject._state < 3:
            print("ConcreteObserverA: Reacted to the event")


class ConcreteObserverB(Observer):
    def update(self, subject: Subject) -> None:
        if subject._state == 0 or subject._state >= 2:
            print("ConcreteObserverB: Reacted to the event")


if __name__ == "__main__":
    # Клиентский код.

    subject = ConcreteSubject()

    observer_a = ConcreteObserverA()
    subject.attach(observer_a)

    observer_b = ConcreteObserverB()
    subject.attach(observer_b)

    subject.some_business_logic()
    subject.some_business_logic()

    subject.detach(observer_a)

    subject.some_business_logic()
'''
"""
cameras = {

    # "bot1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:554",
    # "bot2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:555",
    # "mid1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:556",
    # "mid2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:557",
    # "top": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:558"

    "mid1": "./data/backend_processor_tests/mid_test_main.mp4",
    "top": "./data/backend_processor_tests/top_test_main.mp4"

    # "mid1": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo.mp4",
    # "top": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo1.mp4"

    # "mid1": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo2.mp4",
    # "top": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo3.mp4"

    # "mid1": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo4.mp4",
    # "top": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo5.mp4"

}

opt_param = {
    'threshold_magnitude': 9,
    'size_accumulation': 6,
    'opticflow_param': {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 100,
        'iterations': 4,
        'poly_n': 5,
        'poly_sigma': 1.1,
        'flags': cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
    }
}

NN_full_empty_cfg = {

    "device": "cpu",  # "cpu" or "cuda:0" for gpu
    "input_shape": (3, 128, 128),  # ch, h, w
    "classes": ['empty', 'fill'],
    "pathToWeights": "./weight/fill_classifier.pt"

}

NN_train_cfg = {
    "device": "cpu",  # "cpu" or "cuda:0" for gpu
    "input_shape": (3, 128, 128),  # ch, h, w
    "classes": ['None', 'Train'],
    "pathToWeights": "./weight/TrainOrNone.pt"
}

ocr_type = "rtsp"
ocr_gpu = False
source = "file"
max_wait_iteration = 4
cut_cord_mid1 = [(0, 249), (1296, 249), (1296, 1065), (0, 1065)]
do_imshow = False
do_save_results = True
dir_for_save = './data/results_of_backend/'
flag_4img = 0
##########--text decoration--###############################################################
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 255, 0)
thickness = 2
"""
cfg = Config('config.yaml')
print(cfg.cfg)
