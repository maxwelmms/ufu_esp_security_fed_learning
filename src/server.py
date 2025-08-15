import flwr as fl
from src.model import create_model
from src.utils import load_data, preprocess_data
from src.client import FlowerClient, evaluate_model
from src.poison_attack import label_poisoning

# Carregar e preparar dados
X, y = load_data("data/ERENO-2.0-100K.csv")
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Avaliação antes do ataque
print("Métricas antes do ataque:")
from src.client import create_model
model = create_model(X_train.shape[1])
metrics_before = evaluate_model(model, X_test, y_test)
print(metrics_before)

# Aplicar label poisoning para teste
X_poisoned, y_poisoned = label_poisoning(X_train, y_train, poison_fraction=0.1)

# Criar e iniciar Flower server
def client_fn(cid: str):
    """
    Função que retorna cada cliente.
    """
    return FlowerClient(model, X_poisoned, y_poisoned, X_test, y_test)

strategy = fl.server.strategy.FedAvg(min_fit_clients=1, min_available_clients=1)

fl.server.start_server(
    server_address="localhost:8080",
    config={"num_rounds": 3},
    strategy=strategy,
)
