import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from relic.aug import get_relic_aug_inference

import warnings

warnings.filterwarnings("ignore")


def linear_regression(embeddings, labels, embeddings_val, labels_val):
    X_train, X_test = embeddings, embeddings_val
    y_train, y_test = labels, labels_val
    
    clf = LogisticRegression(max_iter=100)
    clf = CalibratedClassifierCV(clf)
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy STL10: ", acc)


class STL10Eval:

    def __init__(self, image_size=96):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = get_relic_aug_inference(image_size=(image_size, image_size))
        train_ds = torchvision.datasets.STL10("data/",
                                        split='train',
                                        transform=transform,
                                        download=True)
        val_ds = torchvision.datasets.STL10("data/",
                                        split='test',
                                        transform=transform,
                                        download=True)
    

        self.train_loader = DataLoader(train_ds,
                                batch_size=64,
                                num_workers=4)
        self.val_loader = DataLoader(val_ds,
                            batch_size=64,
                            num_workers=4)

    @torch.inference_mode
    def evaluate(self, relic_model):
        relic_model.eval()
        model = relic_model.target_encoder[0]
        with torch.no_grad():            
            embeddings, labels = self._get_image_embs_labels(model, self.train_loader)
            embeddings_val, labels_val = self._get_image_embs_labels(model, self.val_loader)
            
            linear_regression(embeddings, labels, embeddings_val, labels_val)

    def _get_image_embs_labels(self, model, dataloader):
        embs, labels = [], []
        for idx, (images, targets) in enumerate(dataloader):
            with torch.no_grad():
                images = images.to(self.device)
                out = model(images)
                features = out.cpu().detach().tolist()
                embs.extend(features)
                labels.extend(targets.cpu().detach().tolist())
        return np.array(embs), np.array(labels)
    
    def _get_text_embs(self, model):
        input_ids = torch.tensor(self.encoded_texts["input_ids"]).to(self.device)
        attention_mask = torch.tensor(self.encoded_texts["attention_mask"]).to(self.device)
        return model.extract_text_features(input_ids, attention_mask)
