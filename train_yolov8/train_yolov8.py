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
    data=data_yaml_path,    # конфиг датасета
    epochs=epochs,          # количество эпох
    batch=8,            # количество изображений в батче
    imgsz=512,          # размер изображения 512x512
    device=device,          # обучение на GPU 0
    workers=4,           # 4 потока для загрузки данных
    lr0=0.01,            # начальный learning rate
    weight_decay=0.0005, # L2-регуляризация
    project="runs/train",# папка для сохранения
    name="exp",          # имя эксперимента
    exist_ok=True,       # перезаписывать папку, если она есть
    patience=50,         # ранняя остановка после 50 эпох без улучшений
    box=0.05,            # вес лосса для боксов
    cls=0.5,             # вес лосса для классификации
    kobj=1.0,            # вес лосса для уверенности в объекте
    iou=0.2,             # порог IoU для NMS
    seed=42              # фиксированный seed
    )
    
    # Путь к лучшей модели
    best_model_path = os.path.join('runs', 'train', 'exp', 'weights', 'best.pt')
    print(f"Best model should be at: {best_model_path}")
    
    return model, best_model_path
