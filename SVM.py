import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from skimage.io import imread
from skimage.transform import resize
import warnings

warnings.filterwarnings("ignore")

class SVMImageClassifier:
    def __init__(self, image_dir, image_size=(64, 64), use_pca=False, n_components=50):
        """
        Initialize the SVM Image Classifier.
        
        :param image_dir: Path to the dataset directory. Subfolders represent classes.
        :param image_size: Tuple for resizing images (default: 64x64).
        :param use_pca: Whether to use PCA for dimensionality reduction.
        :param n_components: Number of PCA components to retain (if use_pca=True).
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.use_pca = use_pca
        self.n_components = n_components
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if use_pca else None

    def load_data(self):
        """
        Load and preprocess image data from the dataset directory.
        :return: Tuple of (X, y) where X is feature matrix and y is labels.
        """
        X = []
        y = []
        classes = os.listdir(self.image_dir)
        self.class_names = classes  # Save class names for interpretation
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(self.image_dir, class_name)
            for image_file in os.listdir(class_path):
                try:
                    # Read and resize the image
                    image = imread(os.path.join(class_path, image_file))
                    image_resized = resize(image, self.image_size, anti_aliasing=True).flatten()
                    X.append(image_resized)
                    y.append(class_idx)
                except Exception as e:
                    print(f"Error reading {image_file}: {e}")
        
        return np.array(X), np.array(y)

    def train(self, X_train, y_train):
        """
        Train the SVM model.
        :param X_train: Training features.
        :param y_train: Training labels.
        """
        if self.use_pca:
            X_train = self.pca.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions with the trained SVM model.
        :param X_test: Test features.
        :return: Predicted labels.
        """
        if self.use_pca:
            X_test = self.pca.transform(X_test)
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set.
        :param X_test: Test features.
        :param y_test: True labels.
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=self.class_names)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n", report)

    def run_pipeline(self):
        """
        Run the complete pipeline: Load data, preprocess, split, train, and evaluate.
        """
        print("Loading data...")
        X, y = self.load_data()
        print("Scaling data...")
        X = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Training SVM model...")
        self.train(X_train, y_train)
        print("Evaluating model...")
        self.evaluate(X_test, y_test)


if __name__ == '__main__':
    # Update 'your_dataset_directory' with the path to your dataset
    classifier = SVMImageClassifier(image_dir='your_dataset_directory', image_size=(64, 64), use_pca=True, n_components=100)
    classifier.run_pipeline()
