# 🤖 IA — Inteligência Artificial

Repositório com projetos, códigos e notebooks desenvolvidos para estudo e prática de **Inteligência Artificial (IA)**, **Aprendizado de Máquina (Machine Learning)** e **Ciência de Dados**.

---

## 🧩 Estrutura do Repositório

IA-main/
├── Algoritimos/
│ ├── Apriori.ipynb
│ ├── Árvore_de_decisão_restaurante_codifica_e_treina.ipynb
│ ├── Bagging.ipynb
│ ├── Codifica imputa e balanceia.ipynb
│ └── CrossValidationAD_RF.ipynb
│
├── Base_de_Dados/
│ ├── Iris.csv
│ ├── JogarTénis.csv
│ ├── MercadoSim.csv
│ ├── PãoeManteiga Sim.csv
│ └── PãoeManteiga SimNao.csv
│
└── Bibliotecas/
├── arvores_decisao_do_zero.py
└── Pratica/
└── relatorio_arvores.ipynb

---

## 🧠 Conteúdo

### 📘 Algoritimos
Implementações e experimentos com diferentes técnicas de aprendizado de máquina:
- **Apriori** → Regras de associação (Market Basket Analysis)  
- **Árvore de Decisão** → Codificação, treino e avaliação  
- **Bagging** → Ensemble learning (Random Forest)  
- **Pré-processamento** → Codificação, imputação e balanceamento de dados  
- **Cross Validation** → Avaliação cruzada com Árvore de Decisão e Random Forest  

### 📂 Base de Dados
Conjuntos de dados usados nos experimentos:
- `Iris.csv` — dataset clássico de classificação  
- `JogarTénis.csv` — exemplo de regras de decisão  
- `MercadoSim.csv`, `PãoeManteiga.csv` — bases simuladas para o Apriori  

### 🧰 Bibliotecas
Implementações manuais e práticas complementares:
- `arvores_decisao_do_zero.py` — árvore de decisão criada do zero  
- `Pratica/relatorio_arvores.ipynb` — análise de resultados e desempenho  

---

## 🧪 Tecnologias Utilizadas

- **Python 3.11+**
- **Jupyter Notebook**
- **Principais bibliotecas:**
  - pandas  
  - numpy  
  - scikit-learn  
  - matplotlib  
  - seaborn  
  - mlxtend (Apriori)

---

## 🚀 Como Executar

1. **Clone o repositório**
   ```bash
   git clone https://github.com/VitorSetragni/IA.git
   cd IA
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
jupyter notebook
