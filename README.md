# UFU ESP Security Federated Learning

Repositório do projeto de aprendizado federado usando Flower Framework com dataset **ERENO-2.0-100K** e módulo de teste de label poisoning para projeto de TCC - Segurança em Dispositivos IoT: Teste e Relato de Vulnerabilidades

## Estrutura

``
ufu_esp_security_fed_learning/
├─ data/
│  └─ ERENO-2.0-100K.csv
├─ src/
│  ├─ client.py
│  ├─ model.py
│  ├─ poison_attack.py
│  ├─ server.py
│  └─ utils.py
├─ scripts/
│  └─ run_project.sh
├─ requirements.txt
└─ README.md
 ``

## Como rodar

1. Clone o projeto
2. Copie o dataset `ERENO-2.0-100K.csv` para `data/`
3. Execute o script:

```bash
bash scripts/run_project.sh

```

## Funcionalidades

* Treinamento federado com Flower.

* Avaliação de métricas (accuracy, precision, recall, f1-score, AUC).

* Teste de label poisoning (envenenamento de labels) com comparação das métricas antes e depois do ataque.

## Dependências

* Python 3.10+

* TensorFlow

* Flower (flwr)

* scikit-learn

* pandas

* numpy

* matplotlib



---

✅ Esse setup permite:

1. Carregar o CSV.
2. Treinar um modelo local simples.
3. Executar aprendizado federado via Flower.
4. Aplicar ataque de **label poisoning**.
5. Comparar métricas antes e depois do ataque.

---

