# Загрузка данных
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_paths = [os.path.join(lr_dir, f) for f in os.listdir(lr_dir)]
        self.hr_paths = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_paths[idx]).convert('RGB')
        hr_img = Image.open(self.hr_paths[idx]).convert('RGB')

        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img