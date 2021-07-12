from DatasetUtils.DatasetCreator import DatasetCreator
from DatasetUtils.VideoCutter import VideoCutter
from DatasetUtils.DatasetSplitter import DatasetSplitter

import cv2

opt_param = {
    # Не лезь если ты не Женя
    'threshold_magnitude': 6,  # пороговая амплитуда движения
    'size_accumulation': 6,  # кол-во накапливаемых изображения
    # Не лезть
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

########## Параметры ##########

'''ВЕЗДЕ УКАЗЫВАТЬ АБСОЛЮТНЫЙ ПУТЬ'''

datasetPath = ''  # путь до папки куда будут сохраненны нарезанные изображения
datasetMetaFilePath = ''  # путь до файла с мета-данными о нарезанных изображениях
trainMetaFilePath = ''  # путь до файла с мета-данными о нарезанных изображениях для Train выборки
testMetaFilePath = ''  # путь до файла с мета-данными о нарезанных изображениях для Test выборки
nbInPercentOnTrain = 0.7  # процент данных, который пойдет в Train выборку от общего кол-во данных в datasetMetaFilePath

splitter = DatasetSplitter(datasetMetaFilePath, nbInPercentOnTrain, trainMetaFilePath, testMetaFilePath)

### Обязательно заполнить перед запуском ###

# Тип объекта(class) в видео(Train/None)
# Статус(status) вагона(Fill(полный)/Empty(пустой))

labels = {
    'class': 'Train',
    'status': 'Empty'
}

############################################

pathToVideo = ''  # путь до видео, которое булет нарезанно на кадры
pathToMetaFile = ''  # путь до мета-файла нарезанных изображений

dsmaker = DatasetCreator(pathToVideo, datasetPath, pathToMetaFile, labels, splitter)
# dsmaker.create()
