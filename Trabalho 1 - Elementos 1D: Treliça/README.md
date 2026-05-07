# Análise Dinâmica de Treliças via MEF

Este projeto apresenta o desenvolvimento de um solver customizado em Python para a **Análise Modal e Harmônica de Treliças 3D** utilizando o Método dos Elementos Finitos (MEF). O trabalho faz parte de uma disciplina da Universidade de Brasília.

## 🌐 Acesso Online (Recomendado)

A documentação completa e o Notebook Virtual interativo estão disponíveis online através do GitHub Pages. Esta é a forma mais simples de interagir com a simulação sem a necessidade de configurar um ambiente local:

>**[Acesse a Documentação Interativa aqui](https://caiobraga.github.io/seu-repositorio/trabalho1)**

---

## 📂 Estrutura do Projeto

* **`site/`**: Documentação técnica gerada via MkDocs.
* **`notebook.ipynb`**: Versão Jupyter Notebook tradicional com a sequência de cálculos e validações.
* **`notebook_main.py`**: Código-fonte do notebook interativo desenvolvido em **Marimo** (WASM).
* **`FEM_classes_3D.py`**: Implementação das classes de núcleo: `Node`, `TrussElement` e `FEMSystem`.
* **`truss_config.py`**: Parâmetros de geometria e propriedades de material da treliça.
* **`plot_functions.py`**: Scripts auxiliares para visualização de modos e resposta em frequência (FRF).
* **`script_validação.apdl`**: Script para validação cruzada no **Ansys Mechanical APDL**.
* **`requirements.txt`**: Dependências necessárias para reproduzir o ambiente local.

## 🛠️ Execução Local

### 1. Instalação de Dependências
Recomenda-se o uso do Python 3.11 ou superior:

```bash
pip install -r requirements.txt
```

### 2. Servidor Local (Para visualização offline)
Caso prefira rodar o site localmente, é necessário um servidor para que o WebAssembly (WASM) funcione corretamente devido às políticas de segurança dos navegadores:

1. Abra o terminal na pasta raiz.
2. Execute:
    ```bash
    python -m http.server 8010
    ```
3. Acesse: `http://localhost:8010/site/`

### 3. Desenvolvimento Interativo (Marimo)
Para editar ou rodar o solver em tempo real no seu computador:
```bash
marimo edit notebook_main.py
```
Ou utilize a extensão para o github


