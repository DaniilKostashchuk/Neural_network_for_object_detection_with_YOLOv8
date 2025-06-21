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
