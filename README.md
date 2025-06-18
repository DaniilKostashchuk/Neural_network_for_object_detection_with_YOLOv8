# Neural_network_for_object_detection_with_YOLOv8
Этот проект реализует полный пайплайн для обучения модели YOLOv8 (You Only Look Once версии 8) на пользовательских данных. YOLOv8 - это современная архитектура нейронных сетей для задач детекции объектов, обеспечивающая высокую точность при хорошей скорости работы.<p>
Проект выполнен совместно с https://github.com/Alex-ipgg
## Архитектура решения
1. Подготовка данных
   - Организация структуры директорий
   - Разделение данных на train/val/test
   - Копирование изображений и аннотаций
2. Конфигурация
   - Создание YAML-файла с путями и классами объектов
3. Конфигураци
   - Инициализация модели YOLOv8
   - Настройка гиперпараметров
   - Процесс обучения с сохранением чекпоинтов
4. Визуализация и оценка
   - Построение графиков метрик обучения
   - Валидация лучшей модели
## Ключевые компоненты
```python
def prepare_dataset(data_root='data', split_ratios=(0.7, 0.2, 0.1), random_state=42):
    """Подготавливает структуру данных и разделяет на train/val/test"""
    data_root = os.path.abspath(data_root)
    images_dir = os.path.join(data_root, 'raw_images')
    labels_dir = os.path.join(data_root, 'raw_labels')
    
    # Создаем структуру директорий
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(data_root, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(data_root, 'labels', subset), exist_ok=True)
    
    # Получаем список файлов
    image_files = [f.replace('.jpg', '') for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    # Разделяем данные
    train_val, test = train_test_split(image_files, test_size=split_ratios[2], random_state=random_state)
    train, val = train_test_split(train_val, test_size=split_ratios[1]/(1-split_ratios[2]), random_state=random_state)
    
    def copy_files(files, subset):
        for file in files:
            shutil.copy(
                os.path.join(images_dir, file + '.jpg'),
                os.path.join(data_root, 'images', subset, file + '.jpg')
            )
            label_file = os.path.join(labels_dir, file + '.txt')
            if os.path.exists(label_file):
                shutil.copy(
                    label_file,
                    os.path.join(data_root, 'labels', subset, file + '.txt')
                )
    
    copy_files(train, 'train')
    copy_files(val, 'val')
    copy_files(test, 'test')
    
    print(f"Dataset prepared with {len(train)} train, {len(val)} val, {len(test)} test samples")
```
```python
def create_data_yaml(data_root='data', class_names=['object']):
    """Создает data.yaml файл с абсолютными путями"""
    data_root = os.path.abspath(data_root)
    data_yaml = f"""
path: {data_root}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    with open(os.path.join(data_root, 'data.yaml'), 'w') as f:
        f.write(data_yaml)
    print(f"Created data.yaml at {os.path.join(data_root, 'data.yaml')}")
```
```python
def train_yolov8(data_root='data', class_names=['object'], epochs=100):
    """Обучает модель YOLOv8 на пользовательских данных"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_yaml_path = os.path.abspath(os.path.join(data_root, 'data.yaml'))
    print(f"Using data.yaml at: {data_yaml_path}")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
    
    # Загрузка модели
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')
    
    # Обучение модели
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=8,
        imgsz=640,
        device=device,
        workers=4,
        lr0=0.01,
        weight_decay=0.0005,
        project='runs/train',
        name='exp',
        exist_ok=True,
        patience=50,
        box=0.05,
        cls=0.5,
        kobj=1.0,
        iou=0.2,
        seed=42
    )
    
    # Путь к лучшей модели
    best_model_path = os.path.join('runs', 'train', 'exp', 'weights', 'best.pt')
    print(f"Best model should be at: {best_model_path}")
    
    return model, best_model_path
```
```python
def plot_metrics(metrics_path='runs/train/exp/results.csv'):
    """Визуализация метрик обучения из CSV файла"""
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}")
        return
    
    try:
        # Читаем CSV файл с метриками
        metrics = pd.read_csv(metrics_path)
        
        plt.figure(figsize=(15, 10))
        
        # Проверяем доступные столбцы
        available_columns = metrics.columns.tolist()
        print("Available columns in results.csv:", available_columns)
        
        # Графики потерь (с проверкой наличия столбцов)
        plt.subplot(2, 2, 1)
        if 'train/box_loss' in available_columns:
            plt.plot(metrics['epoch'], metrics['train/box_loss'], label='Train Box Loss')
        if 'val/box_loss' in available_columns:
            plt.plot(metrics['epoch'], metrics['val/box_loss'], label='Val Box Loss')
        plt.title('Box Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        if 'train/cls_loss' in available_columns:
            plt.plot(metrics['epoch'], metrics['train/cls_loss'], label='Train Class Loss')
        if 'val/cls_loss' in available_columns:
            plt.plot(metrics['epoch'], metrics['val/cls_loss'], label='Val Class Loss')
        plt.title('Class Loss')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        if 'train/obj_loss' in metrics.columns:
            plt.plot(metrics['epoch'], metrics['train/obj_loss'], label='Train Obj Loss')
        if 'val/obj_loss' in metrics.columns:
            plt.plot(metrics['epoch'], metrics['val/obj_loss'], label='Val Obj Loss')
        elif 'train/dfl_loss' in metrics.columns:
            plt.plot(metrics['epoch'], metrics['train/dfl_loss'], label='Train DFL Loss')
        elif 'val/dfl_loss' in metrics.columns:
            plt.plot(metrics['epoch'], metrics['val/dfl_loss'], label='Val DFL Loss')
        else:
            print("Object Loss/DFL Loss data not found in results.csv")
        plt.title('Object Loss')
        plt.legend()
        
        # Графики метрик
        plt.subplot(2, 2, 4)
        if 'metrics/precision(B)' in available_columns:
            plt.plot(metrics['epoch'], metrics['metrics/precision(B)'], label='Precision')
        if 'metrics/recall(B)' in available_columns:
            plt.plot(metrics['epoch'], metrics['metrics/recall(B)'], label='Recall')
        if 'metrics/mAP50(B)' in available_columns:
            plt.plot(metrics['epoch'], metrics['metrics/mAP50(B)'], label='mAP@0.5')
        if 'metrics/mAP50-95(B)' in available_columns:
            plt.plot(metrics['epoch'], metrics['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        plt.title('Metrics')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
        
    except Exception as e:
        print(f"Error plotting metrics: {e}")
```
## Особенности реализации
1. Гибкость конфигурации:
   - Поддержка пользовательских путей к данным
   - Настройка соотношения train/val/test
   - Возможность указания произвольных классов объектов
2. Автоматизация:
   - Автоматическое создание структуры директорий
   - Генерация конфигурационного файла
   - Сохранение лучшей модели
3. Визуализация:
   - Графики потерь (box loss, class loss, obj loss)
   - Графики метрик (precision, recall, mAP)
4. Оптимизации:
   - Поддержка GPU (CUDA) при наличии
   - Настройка гиперпараметров обучения
   - Ранняя остановка (patience=50)
## Использование
1. Подготовьте данные в формате:
```
data
├── raw_images # исходные изображения (.jpg)
└── raw_labels # аннотации в формате YOLO (.txt)
```
2. Запустите пайплайн:
```python
if __name__ == '__main__':
    DATA_ROOT = 'data'
    CLASS_NAMES = ['object']  # Замените на ваши классы
    
    # 1. Подготовка данных (раскомментируйте при первом запуске)
    prepare_dataset(DATA_ROOT)
    
    # 2. Создание data.yaml (раскомментируйте при первом запуске)
    create_data_yaml(DATA_ROOT, CLASS_NAMES)
    
    # 3. Обучение модели
    model, best_model_path = train_yolov8(DATA_ROOT, CLASS_NAMES, epochs=10)
    
    # 4. Визуализация метрик
    plot_metrics()
    
    # 5. Проверка лучшей модели
    if os.path.exists(best_model_path):
        print(f"\nSuccess! Best model saved at: {best_model_path}")
        # Загрузка лучшей модели для проверки
        best_model = YOLO(best_model_path)
        metrics = best_model.val()
        print("\nValidation metrics:")
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    else:
        print("\nWarning: Best model not found at expected path!")
```
## Полная структура папок для проекта детекции объектов с использованием YOLOv8:
```
project_root
│
├── data                                  # Основная директория с данными
│   ├── raw_images                        # Исходные изображения
│   │   ├── image1.jpg                    # Пример изображения
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   ├── raw_labels                        # Аннотации в формате YOLO
│   │   ├── image1.txt                    # Пример аннотации
│   │   ├── image2.txt
│   │   └── ...
│   │
│   ├── images                            # Автоматически создается prepare_dataset()
│   │   ├── train                         # Обучающая выборка (70%)
│   │   ├── val                           # Валидационная выборка (20%)
│   │   └── test                          # Тестовая выборка (10%)
│   │
│   ├── labels                            # Автоматически создается prepare_dataset()
│   │   ├── train                         # Аннотации для train
│   │   ├── val                           # Аннотации для val
│   │   └── test                          # Аннотации для test
│   │
│   └── data.yaml                         # Конфигурационный файл (создается create_data_yaml())
│
├── runs                                  # Директория с результатами обучения
│   └── train
│       └── exp                           # Папка эксперимента
│           ├── weights                   # Веса модели
│           │   ├── best.pt               # Лучшая модель
│           │   └── last.pt               # Последняя модель
│           │
│           ├── results.csv               # Метрики обучения
│           └── ...                       # Другие файлы логирования
│
├── training_metrics.png                  # Графики метрик (создается plot_metrics())
│
└── yolov8.py                             # Основной скрипт
```
### Формат аннотаций (YOLO format)
Каждый .txt файл в raw_labels соответствует изображению и содержит:
```
<class_id> <x_center> <y_center> <width> <height>
```
где координаты нормализованы относительно размеров изображения (0-1).
### Пример содержимого data.yaml
```
path: /absolute/path/to/data
train: images/train
val: images/val
test: images/test

nc: 3
names: ['person', 'car', 'dog']
```
### Рекомендации по организации данных:
1. Все изображения должны быть в одном формате (желательно `.jpg` или `.png`)
2. Каждое изображение должно иметь соответствующий файл аннотаций
3. Имена файлов изображений и аннотаций должны совпадать (например: `image1.jpg` и `image1.txt`)
## Метрики 
### Основные метрики детекции
1. _Precision (Точность)_
   - __Формула:__ `TP / (TP + FP)`
   - __Что измеряет:__ Доля правильно обнаруженных объектов среди всех обнаружений
   - __Интерпретация:__
      * Высокая precision → мало ложных срабатываний (FP)
      * Низкая precision → много ложных обнаружений
2. _Recall (Полнота)_
   - __Формула:__ `TP / (TP + FN)`
   - __Что измеряет:__ Долb реальных объектов, которые модель смогла обнаружить
   - __Интерпретация:__
      * Высокий recall → модель пропускает мало объектов
      * Низкий recall → много пропущенных объектов (FN)
3. _mAP (mean Average Precision)_
   - __Основная метрика для оценки YOLO-моделей__
   - __Две версии:__
      * __mAP@0.5:__ При IoU threshold = 0.5
      * __mAP@0.5:0.95:__ Среднее значение mAP для IoU от 0.5 до 0.95 с шагом 0.05
   - __Как вычисляется:__
      1) Строится precision-recall кривая для каждого класса
      2) Вычисляется площадь под кривой (Average Precision)
      3) Усредняется по всем классам
### Метрики потерь (Losses)
1. _Box Loss_
2. _Class Loss_
3. _Object Loss_

## Примеры использования
### Детекция медоедов 


### Детекция палеофауны

## Ссылки на данные
https://www.kaggle.com/datasets/bolg4rin/honey-badger-dataset <p>
https://universe.roboflow.com/aleksandr-2oaxq/test_it
## Лицензия
Этот проект распространяется под лицензией MIT. Подробности см. в файле `LICENSE`.
