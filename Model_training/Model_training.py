if __name__ == '__main__':
    DATA_ROOT = 'data'
    CLASS_NAMES = ['Honey Badger']  # Замените на ваши классы
    
    # 1. Подготовка данных (раскомментируйте при первом запуске)
    # prepare_dataset(DATA_ROOT)
    
    # 2. Создание data.yaml (раскомментируйте при первом запуске)
    # create_data_yaml(DATA_ROOT, CLASS_NAMES)
    
    # 3. Обучение модели
    model, best_model_path = train_yolov8(DATA_ROOT, CLASS_NAMES, epochs=500)
    
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
