# Otonom Arac Navigasyon Sistemi - Maksimum Marjinli Guvenlik Modulu

Bu proje, iki dogrusal ayrilabilir engel sinifini en genis guvenlik koridoru ile ayiran hard-margin SVM tabanli bir modeli uygular.

## Klasor Yapisi
- `models/point2d.py`: Nokta veri sinifi
- `dataset.py`: Veri uretimi ve donusum
- `linear_svm.py`: Hard-margin optimizasyonu
- `visualizer.py`: Grafik cizimi
- `main.py`: Uygulama giris noktasi

## Calistirma
```bash
pip install -r requirements.txt
python main.py
```

## Beklenen Cikti
- Ayirici dogru denklemi
- Marjin genisligi
- Support vector sayisi
- `outputs/demo.png` gorseli

## Matematiksel Model
Amaç: 
min 1/2 ||w||^2
koşuluyla y_i (w^T x_i + b) >= 1

Bu model marjini maksimize ederek en guvenli ayirici siniri uretir.
