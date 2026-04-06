import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Foot Traffic Analytics",
    page_icon="👣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f1e;
    color: #dce8ff;
  }
  .main { background-color: #0a0f1e; }

  h1, h2, h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.04em; }

  /* HERO */
  .hero {
    background: linear-gradient(120deg, #0d1f4e 0%, #0a0f1e 60%);
    border: 1px solid #1e3a7a;
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "👣";
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.07;
  }
  .hero h1 { font-size: 2.8rem; color: #4fa3ff; margin: 0; line-height: 1; }
  .hero p  { color: #7a9fd4; font-size: 0.95rem; margin-top: 0.4rem; }
  .hero .badge {
    display: inline-block;
    background: #4fa3ff20;
    border: 1px solid #4fa3ff50;
    color: #4fa3ff;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    margin-right: 6px;
    margin-top: 8px;
  }

  /* KPI CARDS */
  .kpi-grid { display: flex; gap: 12px; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .kpi {
    flex: 1; min-width: 140px;
    background: #0d1a35;
    border: 1px solid #1e3a7a;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    text-align: center;
  }
  .kpi-val { font-family: 'Bebas Neue', sans-serif; font-size: 2.1rem; color: #4fa3ff; line-height: 1; }
  .kpi-lbl { font-size: 0.72rem; color: #7a9fd4; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.07em; }
  .kpi-delta { font-size: 0.75rem; margin-top: 4px; }
  .up   { color: #3dd68c; }
  .down { color: #ff6b6b; }

  /* SECTION TITLE */
  .stitle {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.25rem;
    color: #dce8ff;
    border-left: 3px solid #4fa3ff;
    padding-left: 12px;
    margin: 1.4rem 0 0.9rem 0;
    letter-spacing: 0.05em;
  }

  /* ALERT BADGE */
  .peak-badge {
    background: #ff6b6b18;
    border: 1px solid #ff6b6b50;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin: 4px 0;
    font-size: 0.88rem;
    color: #ffb3b3;
  }
  .peak-badge strong { color: #ff6b6b; }

  /* SIDEBAR */
  [data-testid="stSidebar"] { background-color: #0d1a35; border-right: 1px solid #1e3a7a; }

  /* TABS */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    background-color: #0d1a35;
    border-radius: 8px;
    border: 1px solid #1e3a7a;
    color: #7a9fd4;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    padding: 7px 18px;
  }
  .stTabs [aria-selected="true"] {
    background-color: #4fa3ff22 !important;
    border-color: #4fa3ff !important;
    color: #4fa3ff !important;
  }

  [data-testid="metric-container"] {
    background: #0d1a35;
    border: 1px solid #1e3a7a;
    border-radius: 12px;
    padding: 1rem;
  }
  [data-testid="stMetricValue"] { color: #4fa3ff !important; font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────
BG    = "#0a0f1e"
CARD  = "#0d1a35"
BLUE  = "#4fa3ff"
GREEN = "#3dd68c"
RED   = "#ff6b6b"
AMBER = "#ffb347"
MUTED = "#7a9fd4"
TEXT  = "#dce8ff"
PAL   = [BLUE, GREEN, RED, AMBER, "#b39dff", "#ff9dd4"]

plt.rcParams.update({
    "figure.facecolor": CARD, "axes.facecolor": CARD,
    "axes.edgecolor": "#1e3a7a", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": "#1e3a7a", "grid.linewidth": 0.5,
})

# ─────────────────────────────────────────────
# DATA GENERATOR (simulé depuis power tile)
# ─────────────────────────────────────────────
@st.cache_data
def generate_data(seed=42, days=7, venue="Centre Commercial"):
    np.random.seed(seed)
    zones_mall  = ["Entrée Principale", "Galerie A", "Galerie B", "Food Court", "Parking"]
    zones_gare  = ["Hall Central", "Quai 1-2", "Quai 3-4", "Boutiques", "Sortie Métro"]
    zones = zones_mall if venue == "Centre Commercial" else zones_gare

    rows = []
    base_date = datetime(2024, 1, 1)

    for day in range(days):
        date = base_date + timedelta(days=day)
        dow  = date.weekday()  # 0=lundi, 6=dimanche
        is_weekend = dow >= 5

        for hour in range(6, 23):
            for zone in zones:
                # Profil horaire réaliste
                if venue == "Centre Commercial":
                    if 10 <= hour <= 13:   base_steps = 420
                    elif 14 <= hour <= 18: base_steps = 510
                    elif hour == 19:       base_steps = 380
                    elif hour < 10:        base_steps = 80
                    else:                  base_steps = 150
                else:  # Gare
                    if 7 <= hour <= 9:     base_steps = 600
                    elif 17 <= hour <= 19: base_steps = 580
                    elif 12 <= hour <= 13: base_steps = 300
                    else:                  base_steps = 120

                # Multiplicateurs par zone
                zone_mult = {
                    "Entrée Principale": 1.4, "Hall Central": 1.5,
                    "Galerie A": 1.1,         "Quai 1-2": 1.2,
                    "Food Court": 1.3,        "Boutiques": 0.9,
                    "Parking": 0.7,           "Sortie Métro": 1.3,
                    "Galerie B": 0.95,        "Quai 3-4": 0.85,
                }.get(zone, 1.0)

                if is_weekend: base_steps *= 1.35

                steps = int(base_steps * zone_mult * np.random.uniform(0.75, 1.25))

                # Lien avec piézoélectrique : énergie générée
                voltage   = round(np.random.uniform(5, 45) * (steps / 500), 2)
                current   = round(np.random.uniform(40, 55), 2)
                power_mw  = round((voltage * current * 1e-6) * 1000, 4)

                rows.append({
                    "date":       date.strftime("%Y-%m-%d"),
                    "jour":       ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"][dow],
                    "heure":      hour,
                    "zone":       zone,
                    "pas":        steps,
                    "voltage_v":  voltage,
                    "current_uA": current,
                    "power_mw":   power_mw,
                    "weekend":    is_weekend,
                })

    return pd.DataFrame(rows), zones

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👣 Foot Traffic")
    st.markdown("---")
    venue = st.selectbox("🏢 Type de lieu", ["Centre Commercial", "Gare"])
    days  = st.slider("📅 Nombre de jours", 3, 14, 7)
    st.markdown("---")
    st.markdown("### 🔍 Filtres")

df, zones = generate_data(days=days, venue=venue)

with st.sidebar:
    sel_zones = st.multiselect("Zones", zones, default=zones)
    sel_days  = st.multiselect("Jours", df['jour'].unique().tolist(), default=df['jour'].unique().tolist())
    h_min, h_max = st.slider("Plage horaire", 6, 22, (6, 22))
    st.markdown("---")
    st.markdown("### 📥 Données")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Télécharger CSV", csv, "foot_traffic.csv", "text/csv")

# FILTRE
dff = df[
    df['zone'].isin(sel_zones) &
    df['jour'].isin(sel_days)  &
    df['heure'].between(h_min, h_max)
]

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>Foot Traffic Analytics</h1>
  <p>Analyse du flux piétonnier & énergie piézoélectrique — <strong>{venue}</strong></p>
  <span class="badge">👣 Flux piétons</span>
  <span class="badge">⚡ Énergie dalle</span>
  <span class="badge">📊 {days} jours</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
total_steps    = dff['pas'].sum()
avg_hour       = dff.groupby('heure')['pas'].mean().mean()
peak_hour      = dff.groupby('heure')['pas'].sum().idxmax()
busiest_zone   = dff.groupby('zone')['pas'].sum().idxmax()
total_energy   = dff['power_mw'].sum()
avg_power      = dff['power_mw'].mean()

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpis_data = [
    (c1, f"{total_steps:,}",        "Total Pas",         ""),
    (c2, f"{int(avg_hour):,}",      "Moy. Pas/Heure",    ""),
    (c3, f"{peak_hour}h",           "Heure de Pointe",   "🔴"),
    (c4, busiest_zone.split()[0],   "Zone + Fréquentée", "🏆"),
    (c5, f"{total_energy:.1f}",     "Énergie Tot. (mW)", ""),
    (c6, f"{avg_power:.3f}",        "Power Moy. (mW)",   ""),
]
for col, val, lbl, icon in kpis_data:
    with col:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-val">{icon} {val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Flux par Heure",
    "🗺️ Zones",
    "📅 Vue Hebdo",
    "⚡ Énergie Piézo",
    "🔥 Heatmap"
])

# ══════════════════════════════════════════════
# TAB 1 — FLUX PAR HEURE
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="stitle">Nombre de pas par heure</div>', unsafe_allow_html=True)

    hourly = dff.groupby('heure')['pas'].agg(['mean','sum','std']).reset_index()
    hourly.columns = ['heure','moy','total','std']

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), facecolor=CARD)

    # Courbe moyenne avec zone d'incertitude
    ax = axes[0]
    ax.fill_between(hourly['heure'],
                    hourly['moy'] - hourly['std'],
                    hourly['moy'] + hourly['std'],
                    alpha=0.2, color=BLUE)
    ax.plot(hourly['heure'], hourly['moy'], color=BLUE, linewidth=2.5, marker='o', markersize=5)
    peak_val = hourly.loc[hourly['moy'].idxmax()]
    ax.axvline(peak_val['heure'], color=RED, linestyle='--', linewidth=1.2, alpha=0.8)
    ax.annotate(f"⏰ Pointe : {int(peak_val['heure'])}h\n{int(peak_val['moy'])} pas/h",
                xy=(peak_val['heure'], peak_val['moy']),
                xytext=(peak_val['heure']+0.8, peak_val['moy']*0.88),
                color=RED, fontsize=8.5,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
    ax.set_title("Moyenne des pas par heure (toutes zones)", color=TEXT, fontsize=11, fontweight='bold')
    ax.set_xlabel("Heure de la journée", color=MUTED, fontsize=9)
    ax.set_ylabel("Pas moyens", color=MUTED, fontsize=9)
    ax.set_xticks(range(6, 23))
    ax.grid(alpha=0.3)

    # Barres par zone
    ax2 = axes[1]
    zone_hour = dff.groupby(['heure','zone'])['pas'].mean().reset_index()
    for i, zone in enumerate(sel_zones[:5]):
        sub = zone_hour[zone_hour['zone'] == zone]
        ax2.plot(sub['heure'], sub['pas'], color=PAL[i % len(PAL)],
                 linewidth=1.8, marker='o', markersize=4, label=zone, alpha=0.85)
    ax2.set_title("Flux par heure et par zone", color=TEXT, fontsize=11, fontweight='bold')
    ax2.set_xlabel("Heure", color=MUTED, fontsize=9)
    ax2.set_ylabel("Pas moyens", color=MUTED, fontsize=9)
    ax2.set_xticks(range(6, 23))
    ax2.legend(facecolor=CARD, edgecolor='#1e3a7a', labelcolor=TEXT, fontsize=8, loc='upper left')
    ax2.grid(alpha=0.3)

    plt.tight_layout(pad=2)
    st.pyplot(fig)

    # Alertes heures de pointe
    st.markdown('<div class="stitle">⚠️ Alertes heures de pointe</div>', unsafe_allow_html=True)
    top3 = hourly.nlargest(3, 'moy')
    for _, row in top3.iterrows():
        st.markdown(f"""<div class="peak-badge">
        🔴 <strong>{int(row['heure'])}h00</strong> — Moyenne de <strong>{int(row['moy'])} pas/heure</strong>
        — Total cumulé : {int(row['total']):,} pas
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — ZONES
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="stitle">Analyse par zone</div>', unsafe_allow_html=True)

    zone_stats = dff.groupby('zone').agg(
        total_pas=('pas','sum'),
        moy_pas=('pas','mean'),
        max_pas=('pas','max'),
        energy=('power_mw','sum')
    ).sort_values('total_pas', ascending=False).reset_index()

    col_a, col_b = st.columns(2)

    with col_a:
        fig2, ax = plt.subplots(figsize=(6.5, 5), facecolor=CARD)
        bars = ax.barh(zone_stats['zone'], zone_stats['total_pas'],
                       color=PAL[:len(zone_stats)], edgecolor=CARD, height=0.6)
        for bar, val in zip(bars, zone_stats['total_pas']):
            ax.text(val + zone_stats['total_pas'].max()*0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:,}", va='center', color=TEXT, fontsize=8.5, fontweight='bold')
        ax.set_title("Total pas par zone", color=TEXT, fontsize=11, fontweight='bold')
        ax.set_xlabel("Total pas", color=MUTED, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    with col_b:
        fig3, ax = plt.subplots(figsize=(6.5, 5), facecolor=CARD)
        wedges, texts, autotexts = ax.pie(
            zone_stats['total_pas'],
            labels=zone_stats['zone'],
            autopct='%1.1f%%',
            colors=PAL[:len(zone_stats)],
            pctdistance=0.78,
            startangle=90,
            wedgeprops=dict(edgecolor=CARD, linewidth=2)
        )
        for t in texts:     t.set_color(MUTED); t.set_fontsize(8)
        for t in autotexts: t.set_color(TEXT);  t.set_fontsize(8); t.set_fontweight('bold')
        ax.set_title("Répartition du flux par zone", color=TEXT, fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown('<div class="stitle">Tableau de bord des zones</div>', unsafe_allow_html=True)
    zone_stats_disp = zone_stats.copy()
    zone_stats_disp.columns = ['Zone','Total Pas','Moy. Pas/h','Pic Pas/h','Énergie (mW)']
    zone_stats_disp = zone_stats_disp.set_index('Zone')
    zone_stats_disp = zone_stats_disp.round(2)
    st.dataframe(zone_stats_disp, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — VUE HEBDO
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="stitle">Vue hebdomadaire du flux piétonnier</div>', unsafe_allow_html=True)

    order_days = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
    daily = dff.groupby('jour')['pas'].sum().reindex(
        [d for d in order_days if d in dff['jour'].unique()]
    ).reset_index()
    daily.columns = ['jour','total']

    col_c, col_d = st.columns([2, 1])

    with col_c:
        fig4, ax = plt.subplots(figsize=(8, 4.5), facecolor=CARD)
        bar_colors = [GREEN if j in ['Sam','Dim'] else BLUE for j in daily['jour']]
        bars = ax.bar(daily['jour'], daily['total'], color=bar_colors, edgecolor=CARD, width=0.6)
        for bar, val in zip(bars, daily['total']):
            ax.text(bar.get_x() + bar.get_width()/2, val + daily['total'].max()*0.01,
                    f"{val:,}", ha='center', color=TEXT, fontsize=8.5, fontweight='bold')
        ax.set_title("Total pas par jour de la semaine", color=TEXT, fontsize=11, fontweight='bold')
        ax.set_ylabel("Total pas", color=MUTED)
        ax.grid(axis='y', alpha=0.3)
        p1 = mpatches.Patch(color=BLUE, label='Semaine')
        p2 = mpatches.Patch(color=GREEN, label='Week-end')
        ax.legend(handles=[p1,p2], facecolor=CARD, edgecolor='#1e3a7a', labelcolor=TEXT, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig4)

    with col_d:
        st.markdown('<div class="stitle">Stats journalières</div>', unsafe_allow_html=True)
        for _, row in daily.iterrows():
            is_we = row['jour'] in ['Sam','Dim']
            color = GREEN if is_we else BLUE
            st.markdown(f"""
            <div style="background:#0d1a35; border:1px solid #1e3a7a; border-left: 3px solid {color};
                        border-radius:8px; padding:0.6rem 1rem; margin-bottom:6px;">
              <span style="font-family:'Bebas Neue',sans-serif; color:{color}; font-size:1.1rem;">{row['jour']}</span>
              <span style="float:right; color:{TEXT}; font-weight:600;">{int(row['total']):,} pas</span>
            </div>""", unsafe_allow_html=True)

    # Flux moyen par heure × jour
    st.markdown('<div class="stitle">Flux moyen par heure selon le jour</div>', unsafe_allow_html=True)
    day_hour = dff.groupby(['jour','heure'])['pas'].mean().reset_index()

    fig5, ax = plt.subplots(figsize=(13, 4.5), facecolor=CARD)
    present_days = [d for d in order_days if d in dff['jour'].unique()]
    for i, day in enumerate(present_days):
        sub = day_hour[day_hour['jour'] == day]
        lw   = 2.5 if day in ['Sam','Dim'] else 1.5
        alpha = 1.0 if day in ['Sam','Dim'] else 0.65
        ax.plot(sub['heure'], sub['pas'], color=PAL[i % len(PAL)],
                linewidth=lw, alpha=alpha, label=day, marker='o', markersize=3.5)
    ax.set_xticks(range(6, 23))
    ax.set_xlabel("Heure", color=MUTED)
    ax.set_ylabel("Pas moyens", color=MUTED)
    ax.legend(facecolor=CARD, edgecolor='#1e3a7a', labelcolor=TEXT, fontsize=8.5, ncol=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig5)

# ══════════════════════════════════════════════
# TAB 4 — ÉNERGIE PIÉZOÉLECTRIQUE
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="stitle">⚡ Énergie générée par les dalles piézoélectriques</div>', unsafe_allow_html=True)

    st.info("💡 Chaque pas sur une dalle piézoélectrique génère une énergie mesurée en milliwatts (mW). "
            "Plus le flux est élevé, plus l'énergie récupérée est importante.")

    e1, e2, e3 = st.columns(3)
    e1.metric("⚡ Énergie Totale",     f"{dff['power_mw'].sum():.2f} mW")
    e2.metric("📊 Puissance Moyenne",  f"{dff['power_mw'].mean():.4f} mW")
    e3.metric("🏆 Pic de Puissance",   f"{dff['power_mw'].max():.4f} mW")

    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<div class="stitle">Énergie vs Pas par heure</div>', unsafe_allow_html=True)
        fig6, ax1 = plt.subplots(figsize=(7, 4.5), facecolor=CARD)
        ax2 = ax1.twinx()
        h_energy = dff.groupby('heure')[['pas','power_mw']].mean().reset_index()
        ax1.bar(h_energy['heure'], h_energy['pas'], color=BLUE, alpha=0.45, width=0.6, label='Pas')
        ax2.plot(h_energy['heure'], h_energy['power_mw'], color=AMBER, linewidth=2.5,
                 marker='D', markersize=5, label='Énergie (mW)')
        ax1.set_xlabel("Heure", color=MUTED)
        ax1.set_ylabel("Pas moyens", color=BLUE)
        ax2.set_ylabel("Power moyen (mW)", color=AMBER)
        ax1.tick_params(axis='y', colors=BLUE)
        ax2.tick_params(axis='y', colors=AMBER)
        ax1.set_title("Flux piéton & énergie générée", color=TEXT, fontsize=10, fontweight='bold')
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labs1+labs2, facecolor=CARD, edgecolor='#1e3a7a', labelcolor=TEXT, fontsize=8)
        ax1.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig6)

    with col_f:
        st.markdown('<div class="stitle">Énergie par zone</div>', unsafe_allow_html=True)
        fig7, ax = plt.subplots(figsize=(7, 4.5), facecolor=CARD)
        z_en = dff.groupby('zone')['power_mw'].sum().sort_values(ascending=True)
        colors_e = [GREEN if v == z_en.max() else AMBER for v in z_en.values]
        ax.barh(z_en.index, z_en.values, color=colors_e, edgecolor=CARD, height=0.55)
        for i, (val, zone) in enumerate(zip(z_en.values, z_en.index)):
            ax.text(val + z_en.max()*0.01, i, f"{val:.1f} mW",
                    va='center', color=TEXT, fontsize=8.5)
        ax.set_title("Énergie piézo cumulée par zone", color=TEXT, fontsize=10, fontweight='bold')
        ax.set_xlabel("Énergie (mW)", color=MUTED)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig7)

    st.markdown('<div class="stitle">Corrélation Pas ↔ Énergie</div>', unsafe_allow_html=True)
    fig8, ax = plt.subplots(figsize=(12, 3.5), facecolor=CARD)
    for i, zone in enumerate(sel_zones[:5]):
        sub = dff[dff['zone'] == zone]
        ax.scatter(sub['pas'], sub['power_mw'], color=PAL[i%len(PAL)],
                   alpha=0.5, s=25, label=zone, edgecolors='none')
    ax.set_xlabel("Nombre de pas", color=MUTED)
    ax.set_ylabel("Power (mW)", color=MUTED)
    ax.set_title("Relation entre flux piétonnier et énergie récupérée", color=TEXT, fontsize=10, fontweight='bold')
    ax.legend(facecolor=CARD, edgecolor='#1e3a7a', labelcolor=TEXT, fontsize=8, ncol=5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig8)

# ══════════════════════════════════════════════
# TAB 5 — HEATMAP
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="stitle">🔥 Heatmap : Heure × Zone</div>', unsafe_allow_html=True)

    pivot = dff.pivot_table(values='pas', index='zone', columns='heure', aggfunc='mean')

    fig9, ax = plt.subplots(figsize=(14, 5), facecolor=CARD)
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd",
                annot=True, fmt=".0f", annot_kws={"size": 7.5},
                linewidths=0.4, linecolor=CARD,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Flux moyen (pas) par zone et par heure", color=TEXT, fontsize=12, fontweight='bold')
    ax.set_xlabel("Heure de la journée", color=MUTED)
    ax.set_ylabel("Zone", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig9)

    st.markdown('<div class="stitle">🔥 Heatmap : Heure × Jour</div>', unsafe_allow_html=True)
    order_days = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
    present = [d for d in order_days if d in dff['jour'].unique()]
    pivot2 = dff.pivot_table(values='pas', index='jour', columns='heure', aggfunc='mean')
    pivot2 = pivot2.reindex([d for d in present if d in pivot2.index])

    fig10, ax = plt.subplots(figsize=(14, 4), facecolor=CARD)
    sns.heatmap(pivot2, ax=ax, cmap="Blues",
                annot=True, fmt=".0f", annot_kws={"size": 7.5},
                linewidths=0.4, linecolor=CARD,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Flux moyen (pas) par jour et par heure", color=TEXT, fontsize=12, fontweight='bold')
    ax.set_xlabel("Heure", color=MUTED)
    ax.set_ylabel("Jour", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig10)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7a9fd4; font-size:0.8rem; padding: 1rem 0;">
  👣 <strong>Foot Traffic Analytics</strong> — Dalles Piézoélectriques &amp; Analyse de flux
  &nbsp;|&nbsp; Données simulées à partir du modèle Power Tile
</div>
""", unsafe_allow_html=True)
