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
