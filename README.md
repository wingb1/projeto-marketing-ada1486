# Predição de Aceitação de Campanhas de Marketing

**Aluno:** Wingson Souza  
**Turma:** 1486 – AdaTech  
**Trabalho Final ML 1**

---

## Objetivo

O objetivo do projeto foi desenvolver um modelo capaz de prever a probabilidade de um cliente aceitar uma nova oferta de produto durante uma campanha de marketing direto.
Trata-se de um problema de classificação binária, em que o modelo estima se o cliente responderá positivamente à campanha, ou seja, se realizará uma compra após o contato promocional.

---

## Base de Dados

A base utilizada foi o **Marketing Campaign Data Set**, que contém informações sobre o perfil e comportamento de clientes.  
Entre as variáveis disponíveis estão:

- Dados demográficos (idade, escolaridade, estado civil, número de filhos e dependentes)  
- Informações financeiras (renda familiar, despesas, tempo desde o cadastro)  
- Histórico de compras (quantidade gasta com carne, vinho, frutas, produtos doces etc.)  
- Histórico de campanhas anteriores (colunas `AcceptedCmp1` até `AcceptedCmp5`)  
- A variável alvo indica se o cliente **aceitou ou não** a nova campanha.

---

## Metodologia

O trabalho seguiu as seguintes etapas principais:

1. **Análise exploratória (EDA):** identificação de correlações, dados ausentes e balanceamento da variável alvo.  
2. **Pré-processamento:** tratamento de variáveis numéricas e categóricas com `ColumnTransformer`, normalização, codificação One-Hot e balanceamento com **SMOTE**.  
3. **Treinamento:** foram testados os modelos **Logistic Regression**, **Random Forest**, **SVC (RBF)** e **XGBoost**.  
4. **Validação:** uso de *holdout* e *cross-validation*, com métricas de **Accuracy**, **F1-score** e **ROC-AUC**.  
5. **Ajuste de hiperparâmetros:** feito via Grid Search e Random Search (principalmente para RF e XGB).  
6. **Interpretação:** análise de importância das variáveis com **SHAP**.  
7. **Implantação:** serialização do modelo com `joblib` e criação de uma API usando **FastAPI** com o endpoint `/prever`.

---

## Resultados

| Modelo              | Acc   | F1     | ROC-AUC |
|----------------------|-------|--------|---------|
| Logistic Regression  | 0.822 | 0.579  | 0.910   |
| Random Forest        | 0.883 | 0.426  | 0.874   |
| SVC (RBF)            | 0.836 | 0.569  | 0.901   |
| **XGBoost Tunado**   | **0.842** | **0.553** | **0.903** |

O modelo com melhor equilíbrio entre desempenho e interpretabilidade foi o **XGBoost**, escolhido como modelo final para implantação.

As variáveis que mais influenciaram o resultado (segundo o SHAP) foram:  
`Recency` (baixo aumenta chance), `DaysSinceEnroll` (alto aumenta),  
`MntMeatProducts`, `NumStorePurchases` e `AcceptedCmp3`.

---

##  Como rodar a API
```bash
uvicorn app:app --reload
# depois abra http://127.0.0.1:8000/docs
