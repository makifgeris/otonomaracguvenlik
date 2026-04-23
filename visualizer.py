import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot(x, y, model, output_path='outputs/demo.png'):
        x1 = x[y == -1]
        x2 = x[y == 1]
        plt.figure(figsize=(7, 6), dpi=160)
        plt.scatter(x1[:, 0], x1[:, 1], s=100, label='Engel Sınıfı A')
        plt.scatter(x2[:, 0], x2[:, 1], s=100, marker='s', label='Engel Sınıfı B')

        xx = np.linspace(0, 8.5, 300)
        w = model.w
        b = model.b
        yy = -(w[0] * xx + b) / w[1]
        yy1 = -(w[0] * xx + b - 1) / w[1]
        yy2 = -(w[0] * xx + b + 1) / w[1]
        plt.plot(xx, yy, linewidth=2.5, label='Karar sınırı')
        plt.plot(xx, yy1, '--', linewidth=1.4, label='Marjin')
        plt.plot(xx, yy2, '--', linewidth=1.4)

        sv = model.support_vectors
        plt.scatter(sv[:, 0], sv[:, 1], s=220, facecolors='none', edgecolors='black', linewidths=2.2, label='Support Vector')
        plt.grid(True, alpha=0.25)
        plt.xlim(0, 8.5)
        plt.ylim(0, 8.5)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Maksimum Marjinli Ayırıcı')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
