import os

# Получаем список всех файлов в папке
def get_all_files(directory):
    return os.listdir(directory)

# Оставляем только те, что подходят под шаблон: checkpoints_XXX.pth
def filter_checkpoint_files(files):
    checkpoints = []
    for file in files:
        if file.startswith("checkpoint_") and file.endswith(".pth"):
            checkpoints.append(file)

    return checkpoints

# Извлекаем число из имени файла
def extract_version_number(filename):
    # Например: "checkpoints_150.pth" -> 150
    return int(filename.split("_")[1].split(".")[0])

# Находим файл с самой большой версией
def find_latest_checkpoint(files):
    return max(files, key=extract_version_number)


def get_last_checkpoint_version():
    directory = "checkpoints"

    all_files = get_all_files(directory)
    checkpoint_files = filter_checkpoint_files(all_files)

    if not checkpoint_files:
        print("Чекпойнты не найдены.")
        return None

    latest_file = find_latest_checkpoint(checkpoint_files)
    return latest_file

