import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt


class MnistDataloader:
    def __init__(self, train_img, train_lbl, test_img, test_lbl):
        self.train_img = train_img
        self.train_lbl = train_lbl
        self.test_img = test_img
        self.test_lbl = test_lbl

    def read_images_labels(self, img_path, lbl_path):
        # Read labels
        with open(lbl_path, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            labels = array("B", f.read())

        # Read images
        with open(img_path, "rb") as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            img_data = array("B", f.read())

        images = []
        for i in range(size):
            start = i * rows * cols
            end = (i + 1) * rows * cols
            img = np.array(img_data[start:end], dtype=np.uint8).reshape(28, 28)
            images.append(img)

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.train_img, self.train_lbl)
        x_test, y_test = self.read_images_labels(self.test_img, self.test_lbl)
        return (x_train, y_train), (x_test, y_test)


# ============================================================
# Load MNIST as (N, 784)


def load_mnist():
    loader = MnistDataloader(
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte"
    )

    (train_imgs, train_lbls), (test_imgs, test_lbls) = loader.load_data()

    X_train = np.array(train_imgs).reshape(len(train_imgs), -1).astype(np.float32) / 255.0
    X_test = np.array(test_imgs).reshape(len(test_imgs), -1).astype(np.float32) / 255.0
    y_train = np.array(train_lbls, dtype=int)
    y_test = np.array(test_lbls, dtype=int)

    return X_train, y_train, X_test, y_test


# ============================================================
# Template-based Classifier


def compute_templates(X, y, mode="mean"):
    templates = []
    for digit in range(10):
        Xc = X[y == digit]
        if mode == "mean":
            templates.append(Xc.mean(axis=0))
        else:
            templates.append(np.median(Xc, axis=0))
    return np.array(templates)


def predict(X, templates, metric="euclidean"):
    diff = X[:, None, :] - templates[None, :, :]
    if metric == "euclidean":
        dists = np.sqrt((diff ** 2).sum(axis=2))
    else:
        dists = np.abs(diff).sum(axis=2)
    return np.argmin(dists, axis=1)


# ============================================================
# Evaluation Helpers


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def per_class_accuracy(cm):
    acc = []
    for d in range(10):
        total = cm[d].sum()
        correct = cm[d, d]
        acc.append(correct / total if total > 0 else 0.0)
    return acc


# ============================================================
# Plotting Helpers


def show_templates(templates, title, filename):
    plt.figure(figsize=(8, 3))
    for d in range(10):
        plt.subplot(2, 5, d + 1)
        plt.imshow(templates[d].reshape(28, 28), cmap="gray")
        plt.title(str(d))
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, title="Confusion Matrix", filename="confusion.png"):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    ticks = np.arange(10)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, str(val), ha="center", va="center", fontsize=6)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_per_class_accuracy(per_digit, overall_acc, mode, metric, filename):
    digits = np.arange(10)
    percentages = [p * 100 for p in per_digit]  
    avg_percent = overall_acc * 100

    plt.figure(figsize=(8, 5))
    bars = plt.bar(digits, percentages, color="cornflowerblue", edgecolor="black")

    plt.xticks(digits, digits)
    plt.ylim(0, 110)  # leaves space above bars
    plt.xlabel("Digit")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Per-Class Accuracy ({mode.capitalize()} + {metric.capitalize()})")

    plt.axhline(avg_percent, color="red", linestyle="--", linewidth=2, alpha=0.7)

   
    plt.text(
        9.2,                     # far-right placement
        avg_percent + 2,        # 2% above the line for clarity
        f"Avg = {avg_percent:.2f}%",
        color="red",
        fontsize=11,
        ha="right",
        va="bottom",
        fontweight="bold"
    )

  
    for i, v in enumerate(percentages):
        plt.text(i, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()




# ============================================================
# Main experiment
# ============================================================

def run_experiment(X_train, y_train, X_test, y_test, mode, metric):
    print("\n=======================================")
    print(f"{mode.capitalize()} + {metric.capitalize()}")

    templates = compute_templates(X_train, y_train, mode)
    y_pred = predict(X_test, templates, metric)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy(y_test, y_pred)
    per_digit = per_class_accuracy(cm)

    # Convert to percentages
    per_digit_percent = [round(a * 100, 2) for a in per_digit]
    overall_percent = round(acc * 100, 2)

    print(f"Overall accuracy: {overall_percent}%")
    print(f"Per-class accuracy (%): {per_digit_percent}")
    print("Confusion matrix:\n", cm)

    # Save bar chart in % format
    bar_filename = f"per_class_{mode}_{metric}.png"
    plot_per_class_accuracy(per_digit, acc, mode, metric, bar_filename)

    return templates, cm, acc, per_digit



# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    # Compute templates for visualization
    mean_templates = compute_templates(X_train, y_train, mode="mean")
    median_templates = compute_templates(X_train, y_train, mode="median")
    
    print("Digit | Train Count | Test Count")
    for digit in range(10):
        train_count = np.sum(y_train == digit)
        test_count = np.sum(y_test == digit)
        print(f"{digit} | {train_count} | {test_count}")
        
    # Save template images
    show_templates(mean_templates, "Mean Templates", "mean_templates.png")
    show_templates(median_templates, "Median Templates", "median_templates.png")

    # Run 4 experiments
    configs = [
        ("mean", "euclidean"),
        ("mean", "manhattan"),
        ("median", "euclidean"),
        ("median", "manhattan")
    ]

    for mode, metric in configs:
        templates, cm, acc, per_digit = run_experiment(
            X_train, y_train, X_test, y_test,
            mode=mode, metric=metric
        )

        # Only plot confusion matrix for one case 
        if mode == "mean" and metric == "euclidean":
            plot_confusion_matrix(
                cm,
                "Confusion Matrix (Mean + Euclidean)",
                "cm_mean_euclidean.png"
            )


if __name__ == "__main__":
    main()
