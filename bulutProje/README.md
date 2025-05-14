
# Hayvan Görüntü Sınıflandırıcı

Bu proje, bir görüntü sınıflandırma modeli geliştirerek belirli bir kategoriye ait görselleri tanıyabilen bir yapay zeka uygulamasıdır. 
Model, eğitildiği veri kümesine dayanarak, yüklenen görselin hangi hayvana ait olduğunu tahmin eder.
Bu proje, kullanıcıların basit bir arayüz ile görsellerini yükleyip tahmin alabilecekleri bir sistem oluşturmayı amaçlamıştır.

# Proje İçeriği

- Model: PyTorch kullanarak eğitilen bir ResNet18 modelini içerir.
- Veri Kümesi: 10 farklı hayvan sınıfından oluşan bir görüntü veri kümesi.
- Kütüphaneler: PyTorch, Gradio
- Kullanıcı Arayüzü: Gradio üzerinden kullanıcı dostu bir arayüz.
- Eğitim ve Test Süreçleri:Modelin eğitim ve test aşamaları için Python kodu içerir.

# Gereksinimler

- Python 3.6 veya daha yüksek
- PyTorch (modelin eğitimi ve tahmin için)
- Gradio (kullanıcı arayüzü için)
- torchvision  (görüntü işleme için)
- PIL (görüntü işleme için)

# Gereksinimlerin Kurulumu

Projenin çalışması için aşağıdaki komutlarla gerekli kütüphaneleri kurabilirsiniz:

"pip install torch torchvision gradio"


# Google Colab Üzerinde Çalıştırma

### 1. Projeyi Google Colab'a Yükleme
- Google Colab'ı açın.
- Yeni bir notebook oluşturun.
- Aşağıdaki komut ile GitHub reposunu Colab ortamına klonlayın:

```bash
!git clone https://github.com/kullanici_adiniz/proje_adi.git
```

> Bu komut, projenizi Colab’a indirir.

### 2. Google Drive’ı Bağlama ve Model Dosyasını Ekleme
- Eğer eğitilmiş modeliniz Google Drive’da ise aşağıdaki kod ile Drive'ınızı bağlayın:

```python
from google.colab import drive
drive.mount('/content/drive')
```

- Daha sonra model dosyasını Python kodunuzda bu şekilde kullanabilirsiniz:

```python
model_path = '/content/drive/MyDrive/animals/animal_classifier.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
```

> Not: Eğer modeli GPU’da kullanacaksanız `map_location` kısmını `cuda` yapabilirsiniz.

### 3. Kodları Çalıştırma
- Aşağıdaki komutlarla eğitim ve veri yükleme dosyalarını çalıştırın:

```bash
!python /content/drive/MyDrive/animals/model.py
!python /content/drive/MyDrive/animals/data_loader.py
```

### 4. Gradio Arayüzünü Başlatma
- Arayüzü başlatmak için:

```bash
!python /content/drive/MyDrive/animals/app.py
```

> Çalıştırıldıktan sonra Gradio size bir bağlantı verecek ve bu bağlantı aracılığıyla modelinizi test edebileceksiniz.

# Kullanım

1. Arayüzde "Resim yükleyin" kısmına tıklayarak bir resim seçin.
2. "Gönder" butonuna tıklayın ve modelin tahminini görün.

# Eğitim ve Test Süreci

Model, ResNet-18 mimarisine dayalı olarak eğitildi. Eğitim sırasında doğruluk ve kayıp hesaplanarak modelin başarımı ölçüldü.

Test setinde doğruluk: **87.89%**  
Precision: **0.8759**  
Recall: **0.8605**


# Dosyalar
- raw-img: Eğitim verisi olarak kullanılan, sınıflandırılacak hayvan görüntülerinin bulunduğu klasör.
- app.py: Gradio arayüzü ile kullanıcıya modelin tahminlerini sunan web uygulaması dosyası.
- data_loader.py: Veri setini yükleyip, gerekli ön işleme işlemlerini yaparak eğitim ve test veri setlerine ayıran Python dosyası.
- model.py: Modelin oluşturulup, eğitilip test edilen ve sonuçların hesaplandığı Python dosyası.