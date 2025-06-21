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
