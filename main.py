from dataset import Dataset
from linear_svm import LinearSVM
from visualizer import Visualizer


def main():
    dataset = Dataset.generate_sample()
    x, y = dataset.to_numpy()

    model = LinearSVM().fit(x, y)
    predictions = model.predict(x)
    accuracy = (predictions == y).mean()

    print('=== Otonom Araç Güvenlik Modülü / Maksimum Marjin Demo ===')
    print('Ayırıcı denklem:', model.separator_equation())
    print(f'Marjin genişliği: {model.get_margin():.4f}')
    print(f'Eğitim doğruluğu: {accuracy * 100:.2f}%')
    print('Support vector sayısı:', len(model.support_vectors))

    Visualizer.plot(x, y, model)
    print('Grafik kaydedildi: outputs/demo.png')


if __name__ == '__main__':
    main()
