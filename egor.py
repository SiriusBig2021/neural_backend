# results = reader.readtext(image)
# cv2.imwrite("/home/ea/projects/SIRIUS21/ocr.pytorch/test_images/tt.png", warped_img)
# warped_img = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
# warped_img = enlarge_img(warped_img, 200)
# print(pytesseract.image_to_string(warped_img, config = r'--psm 8'))
# show_image(warped_img
#----------------------------------------------------------------------------------------------
#######################################################################################################################
"""
from easyocr import Reader as Rd
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
from __future__ import annotations

import time

"""
from models import OCRReader
save = {0: 1}
ded = OCRReader("/home/home/projects/neural_backend/samples/numberOCR.jpg", type='img', all_info=save)
while True:
    ded.image_run(watch=False)
"""
######################################################################################################################
# from abc import ABC, abstractmethod
# from random import randrange
# from typing import List
#
#
# class Subject(ABC):
#     """
#     Интферфейс издателя объявляет набор методов для управлениями подписчиками.
#     """
#
#     @abstractmethod
#     def attach(self, observer: Observer) -> None:
#         """
#         Присоединяет наблюдателя к издателю.
#         """
#         pass
#
#     @abstractmethod
#     def detach(self, observer: Observer) -> None:
#         """
#         Отсоединяет наблюдателя от издателя.
#         """
#         pass
#
#     @abstractmethod
#     def notify(self) -> None:
#         """
#         Уведомляет всех наблюдателей о событии.
#         """
#         pass
#
#
# class ConcreteSubject(Subject):
#     """
#     Издатель владеет некоторым важным состоянием и оповещает наблюдателей о его
#     изменениях.
#     """
#
#     _state: int = None
#     """
#     Для удобства в этой переменной хранится состояние Издателя, необходимое всем
#     подписчикам.
#     """
#
#     _observers: List[Observer] = []
#     """
#     Список подписчиков. В реальной жизни список подписчиков может храниться в
#     более подробном виде (классифицируется по типу события и т.д.)
#     """
#
#     def attach(self, observer: Observer) -> None:
#         print("Subject: Attached an observer.")
#         self._observers.append(observer)
#
#     def detach(self, observer: Observer) -> None:
#         self._observers.remove(observer)
#
#     """
#     Методы управления подпиской.
#     """
#
#     def notify(self) -> None:
#         """
#         Запуск обновления в каждом подписчике.
#         """
#
#         print("Subject: Notifying observers...")
#         for observer in self._observers:
#             observer.update(self)
#
#     def some_business_logic(self) -> None:
#         """
#         Обычно логика подписки – только часть того, что делает Издатель.
#         Издатели часто содержат некоторую важную бизнес-логику, которая
#         запускает метод уведомления всякий раз, когда должно произойти что-то
#         важное (или после этого).
#         """
#
#         print("\nSubject: I'm doing something important.")
#         self._state = randrange(0, 10)
#
#         print(f"Subject: My state has just changed to: {self._state}")
#         self.notify()
#
#
# class Observer(ABC):
#     """
#     Интерфейс Наблюдателя объявляет метод уведомления, который издатели
#     используют для оповещения своих подписчиков.
#     """
#
#     @abstractmethod
#     def update(self, subject: Subject) -> None:
#         """
#         Получить обновление от субъекта.
#         """
#         pass
#
#
# """
# Конкретные Наблюдатели реагируют на обновления, выпущенные Издателем, к которому
# они прикреплены.
# """
#
#
# class ConcreteObserverA(Observer):
#     def update(self, subject: Subject) -> None:
#         if subject._state < 3:
#             print("ConcreteObserverA: Reacted to the event")
#
#
# class ConcreteObserverB(Observer):
#     def update(self, subject: Subject) -> None:
#         if subject._state == 0 or subject._state >= 2:
#             print("ConcreteObserverB: Reacted to the event")
#
#
# if __name__ == "__main__":
#     # Клиентский код.
#
#     subject = ConcreteSubject()
#
#     observer_a = ConcreteObserverA()
#     subject.attach(observer_a)
#
#     observer_b = ConcreteObserverB()
#     subject.attach(observer_b)
#
#     subject.some_business_logic()
#     subject.some_business_logic()
#
#     subject.detach(observer_a)
#
#     subject.some_business_logic()
########################################################################################
# import time
# t = time.ctime(time.time())
# print(t)
########################################################################################
def find_time(func):
    def wrap():
        first = time.time()
        func()
        second = time.time()
        print(second - first, " - time of func")
    return wrap()
########################################################################################
"""
/home/home/projects/neural_backend/data/archive1/archive2/mid1_4.mp4" 
/home/home/projects/neural_backend/data/archive1/archive2/top_4.mp4

mid1 - 10sec start  30sec - timeline
top - 50sec  start  30sec - timeline

"""
print(time.ctime())
