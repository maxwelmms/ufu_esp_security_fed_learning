import flwr as fl
from src.model import create_model
from src.utils import preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import numpy as np

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self):
        """
        Retorna os pesos do modelo para o servidor.
        """
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        Treina o modelo localmente com os pesos recebidos do servidor.
        """
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """
        Avalia o modelo local e retorna métricas.
        """
        self.model.set_weights(parameters)
        y_pred = (self.model.predict(self.X_test) > 0.5).astype(int)
        acc = accuracy_score(self.y_test, y_pred)
        return float(acc), len(self.X_test), {"accuracy": acc}

def evaluate_model(model, X_test, y_test):
    """
    Função auxiliar para calcular métricas detalhadas.
    """
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred)
    }
    return metrics
