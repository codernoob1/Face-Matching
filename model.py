import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve, f1_score
from tqdm import tqdm
from collections import defaultdict
import random
from insightface.app import FaceAnalysis

# Configuration
config = {
    "train_path": "/home/madhusudanxsoul/Downloads/Comys_Hackathon5/Task_B/train",
    "val_path": "/home/madhusudanxsoul/Downloads/Comys_Hackathon5/Task_B/val",
    "image_size": (112, 112),
    "match_threshold": 0.6
}

# Initialize ArcFace
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def get_arcface_embedding(img_array):
    faces = face_app.get(img_array)
    return faces[0].embedding if faces else None

def load_image(path):
    img = load_img(path, target_size=config["image_size"])
    img = img_to_array(img).astype(np.uint8)
    return img

class FaceDataset:
    def __init__(self, root_path):
        self.root_path = root_path
        self.identities = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        self.identity_images = self._map_identities()

    def _map_identities(self):
        identity_map = defaultdict(dict)
        for identity in self.identities:
            identity_dir = os.path.join(self.root_path, identity)
            clean_images = [f for f in os.listdir(identity_dir) if not f.startswith('.') and not os.path.isdir(os.path.join(identity_dir, f))]
            identity_map[identity]['clean'] = [os.path.join(identity_dir, img) for img in clean_images]

            distortion_dir = os.path.join(identity_dir, 'distortion')
            if os.path.exists(distortion_dir):
                distorted_images = os.listdir(distortion_dir)
                identity_map[identity]['distorted'] = [os.path.join(distortion_dir, img) for img in distorted_images]
        return identity_map

class FaceVerifier:
    def __init__(self):
        self.reference_embeddings = {}
        self.val_embeddings = defaultdict(list)

    def create_reference(self, identity_folder):
        embeddings = []
        for img_name in os.listdir(identity_folder):
            if img_name.startswith('.') or os.path.isdir(os.path.join(identity_folder, img_name)):
                continue
            img_path = os.path.join(identity_folder, img_name)
            img = load_image(img_path)
            embedding = get_arcface_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
        if embeddings:
            self.reference_embeddings[os.path.basename(identity_folder)] = np.mean(embeddings, axis=0)

    def compute_val_embeddings(self, val_dataset):
        print("\nComputing embeddings for ALL validation samples...")
        for identity in tqdm(val_dataset.identities):
            identity_folder = os.path.join(config["val_path"], identity)
            for img_name in os.listdir(identity_folder):
                if img_name.startswith('.') or os.path.isdir(os.path.join(identity_folder, img_name)):
                    continue
                img_path = os.path.join(identity_folder, img_name)
                img = load_image(img_path)
                emb = get_arcface_embedding(img)
                if emb is not None:
                    self.val_embeddings[identity].append(('clean', emb))

            distortion_dir = os.path.join(identity_folder, 'distortion')
            if os.path.exists(distortion_dir):
                for img_name in os.listdir(distortion_dir):
                    img_path = os.path.join(distortion_dir, img_name)
                    img = load_image(img_path)
                    emb = get_arcface_embedding(img)
                    if emb is not None:
                        self.val_embeddings[identity].append(('distorted', emb))

    def evaluate_full(self):
        if not self.val_embeddings:
            self.compute_val_embeddings(FaceDataset(config["val_path"]))

        true_labels = []
        similarity_scores = []

        print("\nEvaluating positive matches (same identity)...")
        for identity, embeddings in tqdm(self.val_embeddings.items()):
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    emb1 = embeddings[i][1]
                    emb2 = embeddings[j][1]
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    true_labels.append(1)
                    similarity_scores.append(sim)

        print("\nEvaluating negative matches (different identities)...")
        identities = list(self.val_embeddings.keys())
        for i in tqdm(range(len(identities))):
            for j in range(i+1, len(identities)):
                id1, id2 = identities[i], identities[j]
                emb1 = random.choice(self.val_embeddings[id1])[1]
                emb2 = random.choice(self.val_embeddings[id2])[1]
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                true_labels.append(0)
                similarity_scores.append(sim)

        return np.array(true_labels), np.array(similarity_scores)

    def visualize_results(self, true_labels, similarity_scores):
        fpr, tpr, thresholds_roc = roc_curve(true_labels, similarity_scores)
        roc_auc = auc(fpr, tpr)

        precision, recall, thresholds_pr = precision_recall_curve(true_labels, similarity_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds_pr[best_idx]
        config["match_threshold"] = best_threshold

        predictions = (similarity_scores >= best_threshold).astype(int)
        acc = accuracy_score(true_labels, predictions)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='orange', label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(similarity_scores[true_labels == 1], bins=50, alpha=0.5, label='Positive Pairs')
        plt.hist(similarity_scores[true_labels == 0], bins=50, alpha=0.5, label='Negative Pairs')
        plt.axvline(best_threshold, color='red', linestyle='--', label=f'Threshold: {best_threshold:.4f}')
        plt.title(f"Accuracy at threshold: {acc:.4f}")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"\nOptimal Threshold (F1-based): {best_threshold:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
    def predict_image(self, img_path):
        """Predict identity match for a given image"""
        img = load_image(img_path)  # Use the existing load_image function
        query_embedding = get_arcface_embedding(img)  # Use ArcFace for consistency
        
        if query_embedding is None:
            print("No face detected in the image")
            return None, 0.0, 0

        best_match = None
        best_similarity = -1
        for ref_id, ref_embedding in self.reference_embeddings.items():
            similarity = np.dot(query_embedding, ref_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(ref_embedding))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = ref_id

        label = 1 if best_similarity >= config["match_threshold"] else 0
        print(f"\nPredicted Match: {best_match}")
        print(f"Similarity Score: {best_similarity:.4f}")
        print(f"Label: {'Positive' if label == 1 else 'Negative'}")
        return best_match, best_similarity, label

# Run Evaluation
verifier = FaceVerifier()
train_dataset = FaceDataset(config["train_path"])

print("\nCreating reference embeddings from training set...")
for identity in tqdm(train_dataset.identities):
    folder = os.path.join(config["train_path"], identity)
    verifier.create_reference(folder)

true_labels, similarity_scores = verifier.evaluate_full()
verifier.visualize_results(true_labels, similarity_scores)
