name: IA - Inteligência Artificial
description: >
  Repositório com códigos, notebooks e estudos desenvolvidos para aprendizado
  e prática de Inteligência Artificial, Machine Learning e Ciência de Dados.

repository:
  url: https://github.com/VitorSetragni/IA
  structure:
    - Algoritimos/
    - Base_de_Dados/
    - Bibliotecas/
  files:
    Algoritimos:
      - Apriori.ipynb
      - Arvore_de_decisao_restaurante_codifica_e_treina.ipynb
      - Bagging.ipynb
      - Codifica_imputa_e_balanceia.ipynb
      - CrossValidationAD_RF.ipynb
    Base_de_Dados:
      - Iris.csv
      - JogarTenis.csv
      - MercadoSim.csv
      - PaoeManteiga_Sim.csv
      - PaoeManteiga_SimNao.csv
    Bibliotecas:
      - arvores_decisao_do_zero.py
      - Pratica/relatorio_arvores.ipynb

content:
  Algoritimos:
    description: >
      Notebooks de implementação e prática de algoritmos de aprendizado de máquina.
    topics:
      - Apriori: "Algoritmo de associação de regras (Market Basket Analysis)"
      - Arvore_de_decisao_restaurante_codifica_e_treina: "Exemplo de treinamento e codificação de árvore de decisão"
      - Bagging: "Ensemble learning com Bagging e Random Forest"
      - Codifica_imputa_e_balanceia: "Pré-processamento: codificação, imputação e balanceamento"
      - CrossValidationAD_RF: "Validação cruzada em Árvore de Decisão e Random Forest"
  Base_de_Dados:
    description: "Conjunto de bases utilizadas nos notebooks."
    datasets:
      - Iris.csv: "Dataset clássico de classificação"
      - JogarTenis.csv: "Exemplo didático de regras de decisão"
      - MercadoSim.csv: "Base simulada para o Apriori"
      - PaoeManteiga_SimNao.csv: "Base binária para análise de regras de associação"
  Bibliotecas:
    description: "Scripts e módulos criados do zero."
    scripts:
      - arvores_decisao_do_zero.py: "Implementação manual de árvore de decisão"
      - relatorio_arvores.ipynb: "Relatório e análise dos resultados"

technologies:
  language: Python 3.11+
  environment: Jupyter Notebook
  libraries:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
    - mlxtend

execution:
  steps:
    - step: "Clonar o repositório"
      command: git clone https://github.com/VitorSetragni/IA.git
    - step: "Criar e ativar ambiente virtual"
      commands:
        - python -m venv venv
        - source venv/bin/activate   # Linux/Mac
        - venv\Scripts\activate      # Windows
    - step: "Instalar dependências"
      command: pip install -r requirements.txt
    - step: "Abrir notebooks"
      command: jupyter notebook

objective: >
  Este repositório tem caráter educacional e experimental, com o intuito de:
  - Consolidar fundamentos de IA e ML;
  - Demonstrar o funcionamento interno dos algoritmos;
  - Criar um portfólio de notebooks comentados e didáticos.

author:
  name: Vitor Setragni
  email: vitorsetragni@gmail.com
  location: Brasil

license:
  type: Educacional e Livre
  note: >
    Este projeto é de uso educacional e livre.
    Sinta-se à vontade para estudar, modificar e reutilizar o código.

quote: "🌱 O aprendizado de máquina começa pela curiosidade de entender como a máquina aprende."
