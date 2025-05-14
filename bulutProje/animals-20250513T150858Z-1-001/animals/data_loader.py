
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Veri seti yolu
data_dir = "/content/drive/MyDrive/animals/raw-img"

# Görüntü boyutu ve augmentasyon işlemleri
img_size = 224

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Yatayda çevirme
    transforms.RandomRotation(10),  # 10 dereceye kadar döndürme
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # Resmi rastgele yeniden boyutlandırıp kırpma
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Renk değişiklikleri
    transforms.Resize((img_size, img_size)),  # Boyutlandırma
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Görüntülerin normalize edilmesi
])

# Dataset'i oluşturma
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Eğitim/Test ayrımı
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# Sınıf isimlerini yazdır
class_names = dataset.classes
print("Sınıflar:", class_names)
print("Eğitim veri sayısı:", len(train_dataset))
print("Test veri sayısı:", len(test_dataset))
