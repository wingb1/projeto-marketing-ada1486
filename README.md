# Predição de Aceitação de Campanhas de Marketing

**Aluno:** Wingson Souza  
**Turma:** 1486 - AdaTech  
**Trabalho Final ML 1**

##  Objetivo
Prever a **probabilidade de um cliente aceitar** uma nova campanha (classificação binária).

##  Metodologia (resumo)
1. EDA: balanceamento, correlação, distribuições, ausentes.
2. Pré-processamento (ColumnTransformer): num/cat, escala, One-Hot; SMOTE; (PCA opcional).
3. Modelagem: Logistic Regression, Random Forest, SVC e XGBoost.
4. Validação: holdout + cross-validation, métricas (Accuracy, F1, ROC-AUC).
5. Tuning: Grid/Random Search (RF/XGB).
6. Explicabilidade: SHAP (summary + exemplo local).
7. Produtização: pipeline serializada (`joblib`) + API FastAPI (`/prever`).

##  Resultados no teste
| Modelo | Acc | F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.822 | 0.579 | 0.910 |
| Random Forest | 0.883 | 0.426 | 0.874 |
| SVC (RBF) | 0.836 | 0.569 | 0.901 |
| **XGBoost Tunado** | **0.842** | **0.553** | **0.903** |

**Top influências (SHAP):** `Recency` (baixo ↑), `DaysSinceEnroll` (alto ↑), `MntMeatProducts` (alto ↑), `NumStorePurchases` (alto ↑), `AcceptedCmp3` (1 ↑).

##  Como rodar a API
```bash
uvicorn app:app --reload
# depois abra http://127.0.0.1:8000/docs
