import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_excel('Enem_2024_Amostra_Perfeita.xlsx')

cols = {
    "Y": "NOTA_MT_MATEMATICA",
    "CN": "NOTA_CN_CIENCIAS_DA_NATUREZA",
    "CH": "NOTA_CH_CIENCIAS_HUMANAS",
    "LC": "NOTA_LC_LINGUAGENS_E_CODIGOS",
}

data = (
    df[[cols["Y"], cols["CN"], cols["CH"], cols["LC"]]]
    .apply(pd.to_numeric, errors="coerce")
    .dropna()
)

st.title("📈 Análise de Regressão - Notas ENEM 2024")
st.success(f"Dados carregados: {len(data)} observações")

# -------------------------------------------------
# 1. Correlação
# -------------------------------------------------
st.header("1️⃣ Análise de Correlação")

V = data.values
n, m = V.shape

corr_manual = np.zeros((m, m))

for i in range(m):
    for j in range(m):
        xi = V[:, i]
        xj = V[:, j]
        corr_manual[i, j] = np.corrcoef(xi, xj)[0, 1]

corr_df = pd.DataFrame(corr_manual, columns=data.columns, index=data.columns)

st.subheader("Matriz de Correlação (Manual)")
st.dataframe(
    corr_df.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1)
)


st.subheader("Heatmap de Correlação")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)

# Dispersões
st.subheader("Gráficos de Dispersão")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, var in zip(axes, [cols["CN"], cols["CH"], cols["LC"]]):
    ax.scatter(data[var], data[cols["Y"]], alpha=0.3)
    ax.set_xlabel(var)
    ax.set_ylabel("Matemática")
    ax.set_title(f"Corr: {data[cols['Y']].corr(data[var]):.3f}")

st.pyplot(fig)

# -------------------------------------------------
# 2. Seleção de Variáveis
# -------------------------------------------------
st.header("2️⃣ Seleção de Variáveis")

y = data[cols["Y"]]
X_vars = data[[cols["CN"], cols["CH"], cols["LC"]]]

def backward_elimination(X, y, sig=0.05):
    cols = list(X.columns)
    while len(cols) > 0:
        Xc = sm.add_constant(X[cols])
        model = sm.OLS(y, Xc).fit()
        pvals = model.pvalues.iloc[1:]
        worst = pvals.idxmax()
        if pvals.max() > sig:
            cols.remove(worst)
        else:
            break
        if len(cols) == 0:
            break
    return cols

def forward_selection(X, y, sig=0.05):
    selected = []
    remaining = list(X.columns)
    while remaining:
        best_p = 1.0
        best_col = None
        for col in remaining:
            cols_test = selected + [col]
            Xc = sm.add_constant(X[cols_test])
            model = sm.OLS(y, Xc).fit()
            p = model.pvalues[col]
            if p < best_p:
                best_p = p
                best_col = col
        if best_p < sig:
            selected.append(best_col)
            remaining.remove(best_col)
        else:
            break
    return selected

back_vars = backward_elimination(X_vars, y)
fwd_vars = forward_selection(X_vars, y)

col1, col2 = st.columns(2)
with col1:
    st.write("**Backward:**", back_vars or "Nenhuma variável significativa")
with col2:
    st.write("**Forward:**", fwd_vars or "Nenhuma variável significativa")

# -------------------------------------------------
# 3. Diagnóstico de Regressão
# -------------------------------------------------
st.header("3️⃣ Diagnóstico dos Modelos")

def diagnostico(model, nome):
    st.subheader(f"Modelo — {nome}")

    yhat = model.fittedvalues
    resid = model.resid
    X = model.model.exog
    var_names = model.model.exog_names[1:]

    # Coeficientes
    col1, col2 = st.columns(2)
    with col1:
        coef_df = pd.DataFrame({
            "Variável": model.params.index,
            "Beta": model.params.values,
            "P-valor": model.pvalues.values
        })
        st.dataframe(coef_df)

    with col2:
        st.write(f"R²: {model.rsquared:.4f}")
        st.write(f"R² Ajustado: {model.rsquared_adj:.4f}")
        st.write(f"F: {model.fvalue:.2f}")
        st.write(f"Prob(F): {model.f_pvalue:.4e}")
        st.write(f"AIC: {model.aic:.2f}")
        st.write(f"BIC: {model.bic:.2f}")

    # Linearidade
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(yhat, resid, alpha=0.35)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_title("Resíduos vs Preditos")
    ax.set_xlabel("Valores Preditos")
    ax.set_ylabel("Resíduos")
    st.pyplot(fig)

    # Durbin-Watson
    dw = np.sum((resid[1:] - resid[:-1]) ** 2) / np.sum(resid**2)
    st.write(f"Durbin-Watson: {dw:.4f}")

    # Breusch-Pagan
    u2 = resid**2
    Xc = model.model.exog
    beta_u = np.linalg.solve(Xc.T @ Xc, Xc.T @ u2)
    u2hat = Xc @ beta_u
    R2_aux = 1 - np.sum((u2 - u2hat)**2) / np.sum((u2 - u2.mean())**2)
    LM = len(y) * R2_aux
    p_bp = stats.chi2.sf(LM, Xc.shape[1] - 1)
    st.write(f"Breusch-Pagan: LM={LM:.2f}, p={p_bp:.4f}")

    # QQ-plot
    fig, ax = plt.subplots(figsize=(8, 4))
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("QQ-plot dos Resíduos")
    st.pyplot(fig)

    # VIF (se houver variáveis além da constante)
    if X.shape[1] > 1:
        vifs = [variance_inflation_factor(X, j) for j in range(1, X.shape[1])]
        vif_df = pd.DataFrame({"Variável": var_names, "VIF": vifs})
        st.dataframe(vif_df)

# Criar modelos
X1 = sm.add_constant(X_vars)
model1 = sm.OLS(y, X1).fit()

# Verificar se há variáveis no modelo backward
if back_vars:
    X2 = sm.add_constant(X_vars[back_vars])
    model2 = sm.OLS(y, X2).fit()
else:
    # Modelo apenas com constante se nenhuma variável for significativa
    X2 = pd.DataFrame({'const': [1]*len(y)})
    model2 = sm.OLS(y, X2).fit()

# Diagnóstico
diagnostico(model1, "Modelo Completo")
diagnostico(model2, "Modelo Backward")

# -------------------------------------------------
# 4. Métricas dos modelos (RMSE)
# -------------------------------------------------
st.header("4️⃣ Métricas dos Modelos")

def rmse_calc(model):
    resid = model.resid
    return np.sqrt(np.mean(resid**2))

rmse1 = rmse_calc(model1)
rmse2 = rmse_calc(model2)

st.write(f"RMSE Modelo Completo: {rmse1:.4f}")
st.write(f"RMSE Modelo Backward: {rmse2:.4f}")

# -------------------------------------------------
# 5. Comparação e validação cruzada (K-Fold)
# -------------------------------------------------
st.header("5️⃣ Validação Cruzada com K-Fold")

def kfold_evaluation(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmses = []
    r2s = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        model = sm.OLS(y_train, X_train_const).fit()
        y_pred = model.predict(X_test_const)
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))
    return np.mean(rmses), np.mean(r2s)

rmse1_cv, r2_1_cv = kfold_evaluation(X_vars, y)

# Para o modelo backward, verificar se há variáveis
if back_vars:
    rmse2_cv, r2_2_cv = kfold_evaluation(X_vars[back_vars], y)
else:
    # Se não há variáveis, modelo apenas com constante
    X_empty = pd.DataFrame(index=X_vars.index)
    rmse2_cv, r2_2_cv = kfold_evaluation(X_empty, y)

st.write(f"K-Fold (k=5) RMSE Modelo Completo: {rmse1_cv:.4f}")
st.write(f"K-Fold (k=5) R² Modelo Completo: {r2_1_cv:.4f}")
st.write(f"K-Fold (k=5) RMSE Modelo Backward: {rmse2_cv:.4f}")
st.write(f"K-Fold (k=5) R² Modelo Backward: {r2_2_cv:.4f}")