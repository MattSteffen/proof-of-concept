"""
Computer Vision Baseline: Transfer Learning with Pre-trained CNN
================================================================

This script demonstrates why Transfer Learning is an excellent baseline for computer vision:
1. Leverages pre-trained features from large datasets (ImageNet)
2. Much faster than training from scratch
3. Often surprisingly effective even with simple classifiers
4. Requires minimal data and computational resources
5. Easy to implement with modern frameworks

Dataset: Fashion-MNIST (clothing classification)
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def load_and_explore_data():
    """Load Fashion-MNIST dataset and perform initial exploration"""
    print("=" * 60)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("=" * 60)

    print("Loading Fashion-MNIST dataset...")
    print("Fashion-MNIST: 70,000 grayscale images of clothing items")
    print(
        "10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Boot"
    )

    # Transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load Fashion-MNIST
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    # Class names
    class_names = train_dataset.classes

    print(f"\nDataset shapes:")
    print(f"Training images: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}")
    print(f"Test labels: {y_test.shape}")

    print(f"\nImage properties:")
    print(f"Image size: {x_train.shape[1]}x{x_train.shape[2]} pixels")
    print(f"Color channels: {1} (grayscale)")
    print(f"Pixel value range: {x_train.min()} to {x_train.max()}")

    print(f"\nClass distribution (training):")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, (cls, count) in enumerate(zip(unique, counts)):
        print(f"  {class_names[cls]}: {count} samples")

    # Visualize sample images
    print(f"\nVisualizing sample images...")
    plt.figure(figsize=(12, 8))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title(f"{class_names[y_train[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("fashion_mnist_samples.png", dpi=100, bbox_inches="tight")
    print("Sample images saved as 'fashion_mnist_samples.png'")
    plt.close()

    return (
        (x_train, y_train),
        (x_test, y_test),
        class_names,
        train_dataset,
        test_dataset,
    )


def establish_naive_baselines(y_train):
    """Establish naive baselines"""
    print("\n" + "=" * 60)
    print("STEP 2: NAIVE BASELINES")
    print("=" * 60)

    # Most frequent class baseline
    unique, counts = np.unique(y_train, return_counts=True)
    most_frequent_accuracy = max(counts) / len(y_train)
    print(f"Most Frequent Class Baseline: {most_frequent_accuracy:.3f}")
    print("(Always predict the most common class)")

    # Random baseline
    random_accuracy = 1.0 / len(unique)  # 10-class classification
    print(f"Random Baseline: {random_accuracy:.3f}")
    print("(Random guessing among 10 classes)")

    print(f"\nOur goal: Beat {most_frequent_accuracy:.3f} significantly!")

    return most_frequent_accuracy


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def simple_cnn_baseline(train_dataset, test_dataset):
    """Simple CNN baseline from scratch"""
    print("\n" + "=" * 60)
    print("STEP 3: SIMPLE CNN FROM SCRATCH (Comparison Baseline)")
    print("=" * 60)

    print("Training a simple CNN from scratch for comparison...")
    print("Architecture: Conv2D -> MaxPool -> Conv2D -> MaxPool -> Dense -> Output")

    # Use subset for faster training
    subset_size = 10000
    train_subset = Subset(train_dataset, range(subset_size))
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"Using subset of {subset_size} images for faster training...")

    # Build simple CNN
    model_simple = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_simple.parameters())

    print(
        f"Model parameters: {sum(p.numel() for p in model_simple.parameters() if p.requires_grad):,}"
    )

    # Train model
    print("Training simple CNN (this will take a moment)...")
    model_simple.train()
    for epoch in range(5):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")
        for images, labels in pbar:
            optimizer.zero_grad()
            outputs = model_simple(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    # Evaluate
    model_simple.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_simple(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    simple_cnn_accuracy = correct / total
    print(f"Simple CNN Test Accuracy: {simple_cnn_accuracy:.3f}")

    return simple_cnn_accuracy


def extract_traditional_features(images):
    """Extract traditional computer vision features"""
    print("\n" + "=" * 60)
    print("STEP 4: TRADITIONAL FEATURE EXTRACTION")
    print("=" * 60)

    print("Extracting traditional computer vision features:")
    print("✓ Pixel intensities (flattened)")
    print("✓ Basic statistics (mean, std, min, max)")
    print("✓ Edge detection responses")

    # Flatten images to vectors
    flattened = images.reshape(len(images), -1).astype("float32") / 255.0

    # Basic statistical features
    means = np.mean(flattened, axis=1, keepdims=True)
    stds = np.std(flattened, axis=1, keepdims=True)
    mins = np.min(flattened, axis=1, keepdims=True)
    maxs = np.max(flattened, axis=1, keepdims=True)

    # Simple edge detection (difference between adjacent pixels)
    edges_h = (
        np.abs(images[:, :, 1:] - images[:, :, :-1]).mean(axis=(1, 2)).reshape(-1, 1)
    )
    edges_v = (
        np.abs(images[:, 1:, :] - images[:, :-1, :]).mean(axis=(1, 2)).reshape(-1, 1)
    )

    # Combine all features
    features = np.concatenate(
        [
            flattened,  # Raw pixels
            means,
            stds,
            mins,
            maxs,  # Statistics
            edges_h,
            edges_v,  # Edge features
        ],
        axis=1,
    )

    print(f"Feature vector size: {features.shape[1]}")
    print(f"Raw pixels: {flattened.shape[1]}")
    print(
        f"Statistical features: {means.shape[1] + stds.shape[1] + mins.shape[1] + maxs.shape[1]}"
    )
    print(f"Edge features: {edges_h.shape[1] + edges_v.shape[1]}")

    return features


def traditional_ml_baseline(x_train, y_train, x_test, y_test):
    """Traditional ML baseline with hand-crafted features"""
    print("\n" + "=" * 60)
    print("STEP 5: TRADITIONAL ML BASELINE")
    print("=" * 60)

    print("Testing traditional ML approaches:")
    print("1. Logistic Regression on raw pixels")
    print("2. Random Forest on extracted features")

    # Extract features
    print("\nExtracting features...")
    train_features = extract_traditional_features(x_train)
    test_features = extract_traditional_features(x_test)

    # Use subset for faster training
    subset_size = 20000
    train_features_subset = train_features[:subset_size]
    y_train_subset = y_train[:subset_size]

    print(f"\nUsing subset of {subset_size} samples for faster training...")

    # 1. Logistic Regression on raw pixels
    print("\n1. Logistic Regression Baseline:")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features_subset)
    test_scaled = scaler.transform(test_features)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(train_scaled, y_train_subset)

    log_reg_accuracy = log_reg.score(test_scaled, y_test)
    print(f"   Test accuracy: {log_reg_accuracy:.3f}")

    # 2. Random Forest
    print("\n2. Random Forest Baseline:")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(train_features_subset, y_train_subset)

    rf_accuracy = rf.score(test_features, y_test)
    print(f"   Test accuracy: {rf_accuracy:.3f}")

    return max(log_reg_accuracy, rf_accuracy)


def create_transfer_learning_model():
    """Create transfer learning model using pre-trained features"""
    print("\n" + "=" * 60)
    print("STEP 6: TRANSFER LEARNING BASELINE")
    print("=" * 60)

    print("Why Transfer Learning for computer vision baseline?")
    print("✓ Leverages features learned from millions of ImageNet images")
    print(
        "✓ Pre-trained CNNs capture low-level (edges) to high-level (shapes) features"
    )
    print("✓ Much faster than training from scratch")
    print("✓ Often works well even with different domains")
    print("✓ Can use simple classifiers on top of extracted features")

    # Load pre-trained ResNet50 (without top classification layer)
    print("\nLoading pre-trained ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all the parameters in the network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # 10 classes for Fashion-MNIST

    print(f"Base model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Pre-trained on ImageNet (1.4M images, 1000 classes)")

    return model


def train_transfer_learning_model(train_dataset, test_dataset):
    """Train transfer learning model"""
    print("\nPreparing data for transfer learning...")

    # Use subset for demonstration
    subset_size = 15000
    train_subset = Subset(train_dataset, range(subset_size))
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Using {subset_size} training samples...")

    # Create model
    model = create_transfer_learning_model()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    print(f"\nTotal model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print("(Most parameters are frozen in the pre-trained base)")

    # Train model
    print("\nTraining transfer learning model...")
    print("Note: Only training the classification head, base CNN is frozen")

    model.train()
    for epoch in range(5):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for images, labels in pbar:
            # Reshape for ResNet
            images = images.repeat(1, 3, 1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    # Evaluate
    print("\nEvaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.repeat(1, 3, 1, 1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())

    test_accuracy = correct / total
    print(f"Transfer Learning Test Accuracy: {test_accuracy:.3f}")

    return model, test_accuracy, np.array(all_preds)


def feature_extraction_approach(train_dataset, test_dataset, y_test):
    """Alternative: Extract features and use traditional ML"""
    print("\n" + "=" * 60)
    print("STEP 7: FEATURE EXTRACTION + TRADITIONAL ML")
    print("=" * 60)

    print("Alternative approach: Extract CNN features, then use simpler classifiers")
    print("✓ Extract features using pre-trained CNN")
    print("✓ Train fast traditional ML classifier on extracted features")
    print("✓ Often nearly as good as end-to-end fine-tuning")
    print("✓ Much faster training and prediction")

    # Load feature extractor
    feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.eval()

    def extract_features(loader, desc="Extracting features"):
        features = []
        labels_list = []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=desc):
                images = images.repeat(1, 3, 1, 1)
                outputs = feature_extractor(images).squeeze()
                features.append(outputs.numpy())
                labels_list.append(labels.numpy())
        return np.concatenate(features), np.concatenate(labels_list)

    print("\nExtracting features using pre-trained ResNet50...")
    train_loader = DataLoader(Subset(train_dataset, range(20000)), batch_size=1000)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    train_features, y_train_subset = extract_features(
        train_loader, desc="Extracting train features"
    )
    test_features, _ = extract_features(test_loader, desc="Extracting test features")

    print(f"Extracted feature dimension: {train_features.shape[1]}")
    print(f"Training samples: {train_features.shape[0]}")
    print(f"Test samples: {test_features.shape[0]}")

    # Train simple classifiers on extracted features
    print("\nTraining classifiers on extracted features:")

    # 1. Logistic Regression
    print("\n1. Logistic Regression on CNN features:")
    log_reg_cnn = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_cnn.fit(train_features, y_train_subset)
    log_reg_cnn_accuracy = log_reg_cnn.score(test_features, y_test)
    print(f"   Test accuracy: {log_reg_cnn_accuracy:.3f}")

    # 2. Random Forest
    print("\n2. Random Forest on CNN features:")
    rf_cnn = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_cnn.fit(train_features, y_train_subset)
    rf_cnn_accuracy = rf_cnn.score(test_features, y_test)
    print(f"   Test accuracy: {rf_cnn_accuracy:.3f}")

    best_feature_accuracy = max(log_reg_cnn_accuracy, rf_cnn_accuracy)

    print(f"\nBest feature extraction approach: {best_feature_accuracy:.3f}")

    return best_feature_accuracy


def compare_all_results(
    naive_acc, simple_cnn_acc, traditional_ml_acc, transfer_acc, feature_acc
):
    """Compare all approaches"""
    print("\n" + "=" * 60)
    print("STEP 8: COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 60)

    print("Performance Summary:")
    print(f"Naive Baseline (most frequent):     {naive_acc:.3f}")
    print(f"Traditional ML (hand-crafted):      {traditional_ml_acc:.3f}")
    print(f"Simple CNN from scratch:            {simple_cnn_acc:.3f}")
    print(f"Feature Extraction + ML:            {feature_acc:.3f}")
    print(f"Transfer Learning (fine-tuning):    {transfer_acc:.3f}")

    improvements = {
        "Traditional ML": traditional_ml_acc - naive_acc,
        "Simple CNN": simple_cnn_acc - naive_acc,
        "Feature Extraction": feature_acc - naive_acc,
        "Transfer Learning": transfer_acc - naive_acc,
    }

    print(f"\nImprovements over naive baseline:")
    for method, improvement in improvements.items():
        percentage = (improvement / naive_acc) * 100 if naive_acc > 0 else 0
        print(f"{method:20}: +{improvement:.3f} ({percentage:.1f}%)")

    best_method = max(improvements.items(), key=lambda x: x[1])
    print(f"\nBest approach: {best_method[0]} with {best_method[1]:.3f} improvement")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("✓ Transfer learning provides excellent baseline for computer vision")
    print("✓ Pre-trained features work well even on different domains")
    print("✓ Feature extraction + traditional ML often nearly as good")
    print("✓ Much faster than training CNNs from scratch")
    print("✓ Requires minimal computational resources")
    print("✓ Good starting point for further improvements")


def main():
    """Main execution function"""
    print("TRANSFER LEARNING BASELINE FOR COMPUTER VISION")
    print("Fashion-MNIST: Clothing Classification")

    # Step 1: Load and explore data
    (x_train, y_train), (x_test, y_test), class_names, train_dataset, test_dataset = (
        load_and_explore_data()
    )

    # Step 2: Establish naive baselines
    naive_accuracy = establish_naive_baselines(y_train)

    # Step 3: Simple CNN baseline (for comparison)
    simple_cnn_accuracy = simple_cnn_baseline(train_dataset, test_dataset)

    # Step 4 & 5: Traditional ML baseline
    traditional_ml_accuracy = traditional_ml_baseline(x_train, y_train, x_test, y_test)

    # Step 6: Transfer learning baseline
    transfer_model, transfer_accuracy, predictions = train_transfer_learning_model(
        train_dataset, test_dataset
    )

    # Step 7: Feature extraction approach
    feature_accuracy = feature_extraction_approach(train_dataset, test_dataset, y_test)

    # Step 8: Compare all results
    compare_all_results(
        naive_accuracy,
        simple_cnn_accuracy,
        traditional_ml_accuracy,
        transfer_accuracy,
        feature_accuracy,
    )


if __name__ == "__main__":
    main()
