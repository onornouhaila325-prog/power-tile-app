import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dalles Piézoélectriques",
    page_icon="⚡",
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }
  .main { background-color: #0d1117; }

  h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
  }

  /* Header hero */
  .hero {
    background: linear-gradient(135deg, #00d4aa15, #0066ff15);
    border: 1px solid #00d4aa40;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
  }
  .hero h1 { font-size: 2.4rem; font-weight: 800; color: #00d4aa; margin: 0; }
  .hero p  { color: #8b949e; font-size: 1rem; margin-top: 0.5rem; }

  /* KPI cards */
  .kpi-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #00d4aa; }
  .kpi-label { font-size: 0.8rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }

  /* Section titles */
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #e6edf3;
    border-left: 3px solid #00d4aa;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] .css-1d391kg { background-color: #161b22; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background-color: #161b22;
    border-radius: 8px;
    border: 1px solid #30363d;
    color: #8b949e;
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background-color: #00d4aa20 !important;
    border-color: #00d4aa !important;
    color: #00d4aa !important;
  }

  /* Metric overrides */
  [data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem;
  }
  [data-testid="stMetricValue"] { color: #00d4aa !important; font-family: 'Syne', sans-serif; }

  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PALETTE MATPLOTLIB
# ─────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
ACCENT    = "#00d4aa"
ACCENT2   = "#0066ff"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
PALETTE   = [ACCENT, ACCENT2, "#ff6b6b", "#ffd166"]

plt.rcParams.update({
    "figure.facecolor": CARD_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  TEXT,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
})

# ─────────────────────────────────────────────
# SIDEBAR — UPLOAD
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Power Tile")
    st.markdown("---")
    st.markdown("### 📂 Charger les données")
    uploaded = st.file_uploader("Déposez votre fichier CSV", type=["csv"])
    st.markdown("---")
    st.markdown("### 🔍 Filtres")

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df[(df['voltage(v)'] > 0) & (df['current(uA)'] > 0)].copy()
    df['resistance_Ohm'] = (df['voltage(v)'] / (df['current(uA)'] * 1e-6)).round(2)
    return df

if uploaded is None:
    st.markdown("""
    <div class="hero">
      <h1>⚡ Dalles Piézoélectriques</h1>
      <p>Application d'analyse et visualisation — Power Tile Data</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Commencez par charger votre fichier **power_tile_data.csv** dans la barre latérale.")
    st.stop()

df = load_data(uploaded)

# ─────────────────────────────────────────────
# FILTRES SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    locations = st.multiselect(
        "Zone de pression",
        options=df['step_location'].unique().tolist(),
        default=df['step_location'].unique().tolist()
    )
    w_min, w_max = int(df['weight(kgs)'].min()), int(df['weight(kgs)'].max())
    weight_range = st.slider("Poids (kgs)", w_min, w_max, (w_min, w_max))

df_f = df[
    (df['step_location'].isin(locations)) &
    (df['weight(kgs)'] >= weight_range[0]) &
    (df['weight(kgs)'] <= weight_range[1])
]

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚡ Dalles Piézoélectriques</h1>
  <p>Analyse complète de la génération d'énergie — Power Tile Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, f"{len(df_f)}", "Mesures"),
    (k2, f"{df_f['Power(mW)'].mean():.3f}", "Power moyen (mW)"),
    (k3, f"{df_f['Power(mW)'].max():.3f}", "Power max (mW)"),
    (k4, f"{df_f['voltage(v)'].mean():.1f}", "Tension moy. (V)"),
    (k5, f"{df_f['weight(kgs)'].mean():.1f}", "Poids moyen (kg)"),
]
for col, val, label in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">{val}</div>
          <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Distributions",
    "🔗 Corrélations",
    "📍 Par zone",
    "🤖 Modèle ML"
])

# ══════════════════════════════════════════════
# TAB 1 — DISTRIBUTIONS
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Distribution des variables</div>', unsafe_allow_html=True)

    cols_num = ['voltage(v)', 'current(uA)', 'weight(kgs)', 'Power(mW)']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=CARD_BG)
    fig.suptitle("Histogrammes des variables numériques", color=TEXT, fontsize=13, fontweight='bold')

    for ax, col, color in zip(axes.flat, cols_num, PALETTE):
        ax.hist(df_f[col], bins=15, color=color, edgecolor=CARD_BG, alpha=0.9)
        ax.set_title(col, color=TEXT, fontsize=10, fontweight='bold')
        ax.set_xlabel(col, color=MUTED, fontsize=8)
        ax.set_ylabel("Fréquence", color=MUTED, fontsize=8)
        ax.grid(axis='y', alpha=0.4)
        ax.axvline(df_f[col].mean(), color='white', linestyle='--', linewidth=1.2, label='Moyenne')
        ax.legend(fontsize=7, labelcolor=TEXT, facecolor=CARD_BG, edgecolor='#30363d')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-title">Statistiques descriptives</div>', unsafe_allow_html=True)
    st.dataframe(df_f[cols_num].describe().round(3), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — CORRÉLATIONS
# ══════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Matrice de corrélation</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor=CARD_BG)
        corr = df_f[['voltage(v)', 'current(uA)', 'weight(kgs)', 'Power(mW)', 'resistance_Ohm']].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    ax=ax2, linewidths=0.5, linecolor='#0d1117',
                    annot_kws={"size": 9}, mask=mask,
                    cbar_kws={"shrink": 0.8})
        ax2.set_title("Corrélations entre variables", color=TEXT, fontsize=11)
        ax2.tick_params(colors=MUTED, labelsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

    with col_b:
        st.markdown('<div class="section-title">Scatter : Voltage vs Power</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 5), facecolor=CARD_BG)
        colors_map = {"Center": ACCENT, "Edge": ACCENT2, "Corner": "#ff6b6b"}
        for loc in df_f['step_location'].unique():
            sub = df_f[df_f['step_location'] == loc]
            ax3.scatter(sub['voltage(v)'], sub['Power(mW)'],
                        label=loc, color=colors_map.get(loc, ACCENT),
                        alpha=0.75, edgecolors='none', s=55)
        ax3.set_xlabel("Voltage (V)", color=MUTED)
        ax3.set_ylabel("Power (mW)", color=MUTED)
        ax3.set_title("Tension vs Puissance par zone", color=TEXT, fontsize=11)
        ax3.legend(facecolor=CARD_BG, edgecolor='#30363d', labelcolor=TEXT, fontsize=9)
        ax3.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown('<div class="section-title">Scatter : Poids vs Power</div>', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(10, 4), facecolor=CARD_BG)
    for loc in df_f['step_location'].unique():
        sub = df_f[df_f['step_location'] == loc]
        ax4.scatter(sub['weight(kgs)'], sub['Power(mW)'],
                    label=loc, color=colors_map.get(loc, ACCENT),
                    alpha=0.75, edgecolors='none', s=55)
    ax4.set_xlabel("Poids (kg)", color=MUTED)
    ax4.set_ylabel("Power (mW)", color=MUTED)
    ax4.set_title("Poids appliqué vs Puissance générée", color=TEXT, fontsize=11)
    ax4.legend(facecolor=CARD_BG, edgecolor='#30363d', labelcolor=TEXT, fontsize=9)
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)

# ══════════════════════════════════════════════
# TAB 3 — PAR ZONE
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Comparaison par zone de pression</div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig5, ax5 = plt.subplots(figsize=(6, 4.5), facecolor=CARD_BG)
        grp = df_f.groupby('step_location')['Power(mW)'].mean().sort_values(ascending=False)
        bars = ax5.bar(grp.index, grp.values, color=PALETTE[:len(grp)], edgecolor=CARD_BG, width=0.5)
        for bar, val in zip(bars, grp.values):
            ax5.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                     f"{val:.3f}", ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
        ax5.set_title("Power moyen par zone", color=TEXT, fontsize=11)
        ax5.set_ylabel("Power moyen (mW)", color=MUTED)
        ax5.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig5)

    with col_d:
        fig6, ax6 = plt.subplots(figsize=(6, 4.5), facecolor=CARD_BG)
        data_box = [df_f[df_f['step_location'] == loc]['Power(mW)'].values
                    for loc in df_f['step_location'].unique()]
        bp = ax6.boxplot(data_box, labels=df_f['step_location'].unique(),
                         patch_artist=True, notch=False,
                         medianprops=dict(color='white', linewidth=2),
                         whiskerprops=dict(color=MUTED),
                         capprops=dict(color=MUTED),
                         flierprops=dict(marker='o', color=ACCENT, markersize=5))
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax6.set_title("Boxplot Power par zone", color=TEXT, fontsize=11)
        ax6.set_ylabel("Power (mW)", color=MUTED)
        ax6.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig6)

    st.markdown('<div class="section-title">Statistiques par zone</div>', unsafe_allow_html=True)
    st.dataframe(
        df_f.groupby('step_location')[['voltage(v)', 'current(uA)', 'weight(kgs)', 'Power(mW)']]
            .agg(['mean', 'min', 'max']).round(3),
        use_container_width=True
    )

# ══════════════════════════════════════════════
# TAB 4 — MODÈLE ML
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Modèle Random Forest — Prédiction de Power(mW)</div>', unsafe_allow_html=True)

    @st.cache_resource
    def train_model(data_hash):
        df_m = df_f.copy()
        le = LabelEncoder()
        df_m['loc_enc'] = le.fit_transform(df_m['step_location'])
        feats = ['voltage(v)', 'current(uA)', 'weight(kgs)', 'loc_enc', 'resistance_Ohm']
        X = df_m[feats]
        y = df_m['Power(mW)']
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        return rf, sc, le, feats, y_te, y_pred

    rf, sc, le, feats, y_te, y_pred = train_model(len(df_f))

    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score",  f"{r2:.4f}",   help="Plus proche de 1 = meilleur")
    m2.metric("RMSE (mW)", f"{rmse:.4f}", help="Erreur quadratique moyenne")
    m3.metric("MAE (mW)",  f"{mae:.4f}",  help="Erreur absolue moyenne")

    col_e, col_f = st.columns(2)

    with col_e:
        fig7, ax7 = plt.subplots(figsize=(6, 4.5), facecolor=CARD_BG)
        ax7.scatter(y_te, y_pred, color=ACCENT, alpha=0.7, edgecolors='none', s=55)
        lim = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
        ax7.plot(lim, lim, 'w--', linewidth=1.5)
        ax7.set_xlabel("Valeurs réelles (mW)", color=MUTED)
        ax7.set_ylabel("Valeurs prédites (mW)", color=MUTED)
        ax7.set_title("Réel vs Prédit", color=TEXT, fontsize=11)
        ax7.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig7)

    with col_f:
        fig8, ax8 = plt.subplots(figsize=(6, 4.5), facecolor=CARD_BG)
        fi = pd.Series(rf.feature_importances_, index=feats).sort_values()
        colors_fi = [ACCENT if v == fi.max() else ACCENT2 for v in fi.values]
        ax8.barh(fi.index, fi.values, color=colors_fi, edgecolor=CARD_BG)
        ax8.set_title("Importance des features", color=TEXT, fontsize=11)
        ax8.set_xlabel("Importance", color=MUTED)
        ax8.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig8)

    # ── Données brutes ──
    st.markdown('<div class="section-title">Données filtrées</div>', unsafe_allow_html=True)
    st.dataframe(df_f.reset_index(drop=True), use_container_width=True, height=300)
