
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import precision_score, recall_score  

# data_loader.py dosyasından veri çek
from data_loader import train_loader, test_loader, class_names

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# ResNet18 modelini yükle
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim fonksiyonu
def train_model(model, epochs=5):
    for epoch in range(epochs):
        print(f"{epoch+1}. epoch başlatılıyor...")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "/content/drive/MyDrive/animals/animal_classifier.pth")
    print("Model kaydedildi: animal_classifier.pth")

# Test fonksiyonu (Precision ve Recall dahil)
def test_model():
    model.eval()
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    acc = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')

    print(f"Test Doğruluğu (Accuracy): {acc:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# Modeli eğit ve test et
print("Eğitim başlatılıyor...")
train_model(model, epochs=5)
print("Eğitim bitti. Test başlatılıyor...")
test_model()
