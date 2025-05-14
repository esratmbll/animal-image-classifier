
import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Modeli yükle
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("/content/drive/MyDrive/animals/animal_classifier.pth", map_location=torch.device("cpu")))
model.eval()

# Sınıf isimleri
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# Görüntü ön işleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Tahmin fonksiyonu
def predict(image):
    img = Image.fromarray(image).convert("RGB")
    image_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Gradio arayüzü
interface = gr.Interface(fn=predict,
                         inputs=gr.Image(type="numpy"),
                         outputs="text",
                         title="Hayvan Sınıflandırıcı",
                         description="Yüklediğiniz resme göre hayvan sınıfını tahmin eder.")

interface.launch(share=True)
