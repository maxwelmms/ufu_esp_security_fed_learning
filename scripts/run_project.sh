#!/bin/bash
# Script para executar o projeto

# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Executar servidor federado
python3 src/server.py
