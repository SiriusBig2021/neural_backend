Пример кода в файле ForTest

VideoCutter:
    Класс для нарезки видео.В указанную директорию сохраняет куски, на которых обнаруженно движение
    
    HotKeys:
        e - проматывает видео на 30 кадров вперед
        q - проматывает видео на 30 кадров назад
        s - перейти в к слудующему видео в папке

    __init__()
        *param outputPath - папка куда будут сохранены куски видео с движением
        *param opt_param - настройки для DenseOpticalFlow
        *param inputPath(Необязательно) - пака с которой будут взяты видео для нарезки

   Устанавливает текущую директорию с видео для нарезки
   setHandlerFolder()
        *param absolutePath - асбсолютный путь до папки

   Добавляет видео в в очередь для обработки
   addVideoInHandlerFolder()
        *param absolutePathToVideo - абсолютный путь до видео

   Начинает нарезку
   execute()

DatasetSplitter:
    Класс для формирования Train и Test наборов

    __init__()
        *param datasetMetaFile - путь до общего для всего датасеты мета-файла
        *param percentTrainImg - кол-во(0.7 == 70% , 0,14 == 14% и тд) изображений для Train выборки
        *param outTrainMetaFile - путь до выходного мета-файла для Train выборки
        (если файл отсутсвует,то он будет создан по указнном пути)
        *param outTestMetaFile - путь до выходного мета-файла для Test выборки
        (если файл отсутсвует,то он будет создан по указнном пути)

    Начинает разделение датасета(Перезаписывает данные в outTrainMetaFile и outTestMetaFile)
    split()


DatasetCreator:
    Класс для нарезки видео на кадры и создания анотации к ним.

    __init__()
        *param absoluteVideoPath - абсолютный путь до видео
        *param absolutePathOutFolder - абсолютный путь до директории,в которую будут сохранены изображения
        *param absolutePathToMetaFile - абсолютный путь до мета-файла, куда пишется вся информация для о изображениях
        *param labels - информация о состояние вагонов на видео
        *param splitter - экземпляр класса DatasetSplitter.Необходим для разделения датасета

    Создает датасет.
    create()
