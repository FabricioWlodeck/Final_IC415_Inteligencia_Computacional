#!/usr/bin/env python
# coding: utf-8

# # EDA — Dataset de mandioca · Clasificación de enfermedades (5 clases)
# 
# **Trabajo Final Integrador — IC522 · Inteligencia Computacional** · WLODECK, Fabricio Joaquín
# 
# Objetivo: analizar la integridad, composición y calidad del dataset de imágenes de hojas de
# mandioca para preparar el pipeline de clasificación de la consigna.
# 
# **Diccionario de clases (usado en todas las salidas):**
# 
# | Código | Nombre legible | Descripción |
# |---|---|---|
# | 0 | **CBB** | Cassava Brown Streak Disease (bacteriosis / CFMV) |
# | 1 | **CBSD** | Cassava Brown Streak Disease (rayas marrones) |
# | 2 | **CGM** | Cassava Green Mosaic (moteado verde) |
# | 3 | **CMD** | Cassava Mosaic Disease (mosaico) |
# | 4 | **Saludable** | Hoja sin síntomas |
# 
# **Cómo ejecutar**
# 
# * **VS Code:** abrir la carpeta del proyecto, instalar Python y Jupyter, confiar en la carpeta
#   (Workspace Trust), abrir el `.ipynb` y *Run All* (o ejecutar con nbconvert, ver al final).
# * **Google Colab:** *Archivo → Subir a Google Drive* o subir el notebook; subir `labels.csv`
#   (y opcionalmente `mandioca.zip` para las celdas de imágenes, que se desactivan solas si no
#   encuentran la carpeta).
# 
# **Entregables:** `01_EDA_mandioca.ipynb` (ejecutado) + `.html` + `.pdf`.
# 
# > Las celdas de **imágenes** (Fase 5) se autodesactivan si no existe la carpeta de imágenes.

# In[1]:


import os, re, time, random, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NOMBRE = {
    0: "CBB (bacteriosis)",
    1: "CBSD (rayas marrones)",
    2: "CGM (moteado verde)",
    3: "CMD (mosaico)",
    4: "Saludable",
}
PALETA = {0: "#D73027", 1: "#FC8D59", 2: "#FCC163", 3: "#4575B4", 4: "#1A9850"}
ORDEN = sorted(NOMBRE)

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 60)
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

def pct(s):
    return (100 * s / s.sum()).round(2)

print("Semilla:", SEED, "| pandas", pd.__version__, "| numpy", np.__version__)


# ## Fase 0 — Entorno y carga de datos
# 
# Detección de entorno (Colab / local) y ubicación de `labels.csv` con rutas candidatas + respaldo absoluto.

# In[2]:


def detectar_entorno():
    try:
        import google.colab  # noqa: F401
        return "colab"
    except Exception:
        return "local"

ENTORNO = detectar_entorno()
print("Entorno detectado:", ENTORNO)

CANDIDATOS = [
    Path("datos/dataset original/labels.csv"),
    Path("dataset original/labels.csv"),
    Path("datos/labels.csv"),
    Path("../datos/dataset original/labels.csv"),
    Path(r"D:\final_inteligencia\datos\dataset original\labels.csv"),
]

if ENTORNO == "colab":
    print("En Colab subi 'labels.csv' por el panel de archivos (o ejecuta files.upload()).")
    try:
        from google.colab import files
        if not any(p.exists() for p in CANDIDATOS):
            uploaded = files.upload()
            CANDIDATOS.insert(0, Path(list(uploaded)[0]))
    except Exception as e:
        print("No se pudo abrir files.upload():", e)

RUTA_CSV = next((p for p in CANDIDATOS if p.exists()), None)
assert RUTA_CSV is not None, "No se encontro labels.csv en ninguna ruta candidata"
df = pd.read_csv(RUTA_CSV)
print("CSV:", RUTA_CSV.resolve())
print("Shape:", df.shape)


# ## Fase 1 — Identidad del dataset y diccionario
# 
# Primer contacto: dimensiones, tipos, nulos y muestra de filas.

# In[3]:


print("Forma:", df.shape)
print("\nColumnas:", list(df.columns))
display(df.head())
display(df.tail())

tabla_diccionario = pd.DataFrame({
    "columna": ["image_id", "label", "source"],
    "tipo": [str(df.image_id.dtype), str(df.label.dtype), str(df.source.dtype)],
    "descripcion": [
        "nombre del archivo .jpg en la carpeta images/",
        "clase entera 0..4 (mapear con NOMBRE de la celda 1)",
        "fuente/campana de captura (2019 o 2020)",
    ],
    "n_unicos": [df.image_id.nunique(), df.label.nunique(), df.source.nunique()],
})
display(tabla_diccionario)


# ## Fase 2 — Integridad y calidad (antes de mirar distribuciones)
# 
# Chequeos de nulos, duplicados de id, cruce CSV ↔ disco, duplicados exactos (SHA-1 completo),
# legibilidad de las imágenes y coherencia **nombre del archivo ↔ etiqueta**.

# In[4]:


print("Nulos por columna:\n", df.isna().sum().to_string())
print("\nimage_id duplicados:", int(df.image_id.duplicated().sum()))
print("Valores de label:", sorted(df.label.unique()))
print("Valores de source:", sorted(df.source.unique()))
print("rango de label valido [0,4]:", bool(df.label.between(0, 4).all()))
print("extensiones presentes:", Counter(Path(x).suffix for x in df.image_id))


# In[5]:


def buscar_carpeta_imagenes():
    cands = [
        Path("datos/dataset original/images"),
        Path("dataset original/images"),
        Path("datos/images"),
        Path("images"),
        Path(r"D:\final_inteligencia\datos\dataset original\images"),
    ]
    return next((p for p in cands if p.exists()), None)

RAIZ_IMG = buscar_carpeta_imagenes()
if RAIZ_IMG is None:
    print("AVISO: no se encontro la carpeta de imagenes.")
    print("Las fases de imagenes se omitiran (en Colab: subir mandioca.zip y descomprimir).")
else:
    disco = set(os.listdir(RAIZ_IMG))
    en_csv = set(df.image_id)
    print("Carpeta:", RAIZ_IMG.resolve())
    print("Archivos en disco:", len(disco), "| image_id en CSV:", len(en_csv))
    print("Faltan en disco:", len(en_csv - disco), "| Sobran en disco:", len(disco - en_csv))
    print("Extensiones no .jpg:", {e for e in disco if not e.lower().endswith(".jpg")} or "ninguna")
    print("Total GB:", round(sum(os.path.getsize(RAIZ_IMG / f) for f in disco) / 1024**3, 2))


# ### 2.1 Duplicados exactos (SHA-1 completo)
# 
# Lectura de **todos** los archivos en chunks de 1 MB (≈30 s en este equipo para 3.09 GiB). Se reportan:
# archivos, hashes únicos, grupos duplicados, archivos involucrados y excedentes.

# In[6]:


if RAIZ_IMG is None:
    print("Omitido: no hay carpeta de imagenes.")
else:
    t0 = time.time()
    hashes = defaultdict(list)
    files_all = sorted(f for f in os.listdir(RAIZ_IMG))
    for i, f in enumerate(files_all, 1):
        h = hashlib.sha1()
        with open(RAIZ_IMG / f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        hashes[h.hexdigest()].append(f)
        if i % 5000 == 0:
            print(f"  {i}/{len(files_all)} archivos hasheados...")

    dups = {k: v for k, v in hashes.items() if len(v) > 1}
    print(f"Archivos procesados: {len(files_all)}")
    print(f"Hashes unicos:       {len(hashes)}")
    print(f"Grupos duplicados:   {len(dups)}")
    print(f"Archivos en duplicados: {sum(len(v) for v in dups.values())}")
    print(f"Excedentes (a borrar si hubiera): {sum(len(v) - 1 for v in dups.values())}")
    for k, v in list(dups.items())[:10]:
        print("  dup:", v)
    print(f"Tiempo: {time.time() - t0:.1f} s")


# ### 2.2 Legibilidad de las imágenes (muestra aleatoria, semilla fija)

# In[7]:


try:
    from PIL import Image
    PIL_OK = True
except Exception as e:
    PIL_OK = False
    print("Pillow no disponible:", e)

if RAIZ_IMG is None or not PIL_OK:
    print("Omitido: sin carpeta de imagenes o sin Pillow.")
else:
    t0 = time.time()
    files_all = [f for f in os.listdir(RAIZ_IMG) if f.lower().endswith(".jpg")]
    muestra = random.sample(files_all, min(3000, len(files_all)))
    modos, rotas = Counter(), []
    for f in muestra:
        try:
            with Image.open(RAIZ_IMG / f) as im:
                modo = im.mode
            with Image.open(RAIZ_IMG / f) as im:
                im.verify()
            modos[modo] += 1
        except Exception:
            rotas.append(f)
    print("Muestra verificada:", len(muestra))
    print("Ilegibles/corruptas:", len(rotas), rotas[:10])
    print("Modos de color:", dict(modos))
    print(f"Tiempo: {time.time() - t0:.1f} s")


# ### 2.3 Coherencia nombre del archivo ↔ etiqueta (chequeo de filtrado de la etiqueta)
# 
# El subconjunto `source = 2019` tiene nombres `train-<tipo>-NNN.jpg`, que **contienen la clase**.
# Si el crosstab es diagonal, el nombre filtra la etiqueta: jamás debe usarse como feature.

# In[8]:


tipo = df.image_id.str.extract(r"^train-([a-z]+)-\d+\.jpg$")[0]
con_patron = tipo.notna()
print("Archivos con nombre descriptivo:", int(con_patron.sum()), "| numericos:", int((~con_patron).sum()))

ct = pd.crosstab(tipo.fillna("(numerico 2020)"), df.label, margins=True)
ct.rename(columns=NOMBRE, inplace=True)
display(ct)

MAPEO_NOMBRE = {"cbb": 0, "cbsd": 1, "cgm": 2, "cmd": 3, "healthy": 4}
sub = df[con_patron].copy()
sub["label_esperado"] = sub.image_id.str.extract(r"^train-([a-z]+)-\d+\.jpg$")[0].map(MAPEO_NOMBRE)
coherencia = float((sub.label == sub.label_esperado).mean())
print(f"Coherencia nombre->label en los descriptivos: {100 * coherencia:.2f} %")
print("Todos los descriptivos pertenecen a source 2019:",
      bool((df.loc[con_patron, "source"] == 2019).all()))


# ## Fase 3 — Composición y desbalance de clases
# 
# **Normalización:** todos los % de esta sección son **sobre el total de 26,337 registros**, salvo que se indique otra base (se aclara celda por celda).

# In[9]:


t_clase = df.label.value_counts().sort_index().rename_axis("label").to_frame("n")
t_clase["%"] = pct(t_clase.n)
t_clase.index = [NOMBRE[i] for i in t_clase.index]
display(t_clase)

ratio = t_clase.n.max() / t_clase.n.min()
print(f"Clase mayoritaria: {t_clase.n.idxmax()} | minoritaria: {t_clase.n.idxmin()}")
print(f"Ratio desbalance max/min: {ratio:.2f} : 1")

fig, ax = plt.subplots(figsize=(8, 4.2))
etiquetas = [NOMBRE[i] for i in ORDEN]
valores = [int(t_clase.loc[e, "n"]) for e in etiquetas]
ax.bar(etiquetas, valores, color=[PALETA[i] for i in ORDEN], edgecolor="black", linewidth=0.6)
for x, v in zip(range(len(valores)), valores):
    ax.text(x, v + 200, f"{100 * v / sum(valores):.2f} %\nn = {v:,}", ha="center", fontsize=9)
ax.set_title("Distribución de clases (n sobre el total)")
ax.set_ylabel("imágenes")
ax.set_ylim(0, max(valores) * 1.22)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7.5, 7.5))
etiquetas_pie = [f"{e}\nn = {v:,}" for e, v in zip(etiquetas, valores)]
wedges, textos, autopcts = ax.pie(
    valores,
    labels=etiquetas_pie,
    colors=[PALETA[i] for i in ORDEN],
    autopct=lambda p: f"{p:.2f} %",
    startangle=90,
    counterclock=False,
    wedgeprops={"edgecolor": "black", "linewidth": 0.8},
    pctdistance=0.72,
    labeldistance=1.08,
    textprops={"fontsize": 9},
)
for t in autopcts:
    t.set_color("black")
    t.set_fontsize(9)
    t.set_fontweight("bold")
ax.set_title("Distribución de clases — gráfico de torta\n(% sobre el total de 26,337 registros)", fontsize=11)
plt.tight_layout()
plt.show()


# **Implicancias → desbalance y métricas** *(desarrollo completo en `reportes\00_resumen_consigna_y_avance.md`)*
# 
# CMD = 58.71 % vs CBB = 5.67 % → el accuracy engaña: reportar **macro-F1, balanced accuracy,
# ROC-AUC/PR-AUC y matriz de confusión normalizada por fila**, penalizando más el error
# **saludable ↔ enferma** (falsos negativos = el costo más alto para la toma de decisiones).

# In[10]:


ct_ns = pd.crosstab(df.label, df.source, margins=True, margins_name="Total")
ct_ns.rename(index=NOMBRE, inplace=True)
print("Cruce label x source — n (absolutos):")
display(ct_ns)

ct_col = (pd.crosstab(df.label, df.source, normalize="columns") * 100).round(2)
ct_col.rename(index=NOMBRE, inplace=True)
print("Mismo cruce — % normalizado POR COLUMNA (distribucion de clases dentro de cada fuente):")
display(ct_col)

ct_row = (pd.crosstab(df.label, df.source, normalize="index") * 100).round(2)
ct_row.rename(index=NOMBRE, inplace=True)
print("Mismo cruce — % normalizado POR FILA (como se reparte cada clase entre las fuentes):")
display(ct_row)


# ## Fase 4 — Fuentes de captura y sesgo de dominio
# 
# `source` separa dos campañas/años. Las de **2019** traen nombres descriptivos y las de **2020**
# numéricos, lo que sugiere que son datasets originales distintos (cámara, fondo e iluminación
# diferentes). Al dividir train/val/test hay que **estratificar también por `source`** para no
# mezclar dominios dentro de cada split.

# In[11]:


t_fuente = df.source.value_counts().sort_index().rename_axis("source").to_frame("n")
t_fuente["%"] = pct(t_fuente.n)
display(t_fuente)

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.bar([str(s) for s in t_fuente.index], t_fuente.n, color=["#8C6BB1", "#225EA8"],
       edgecolor="black", linewidth=0.6)
for i, (s, row) in enumerate(t_fuente.iterrows()):
    ax.text(i, row.n + 300, f"{row['%']:.2f} %\nn = {int(row.n):,}", ha="center", fontsize=9)
ax.set_title("Registros por fuente (source)")
ax.set_ylabel("registros")
plt.tight_layout()
plt.show()

presentes = pd.crosstab(df.label, df.source) > 0
print("¿Toda clase esta presente en ambas fuentes?:", bool(presentes.all().all()))
print("Mínimo de registros por (clase, fuente):", int(pd.crosstab(df.label, df.source).min().min()))


# ## Fase 5 — Análisis de las imágenes (núcleo del EDA)
# 
# Esta fase **se autodesactiva** si no hay carpeta de imágenes (en Colab: subir `mandioca.zip` y
# descomprimir). Analiza dimensiones, aspecto, brillo/contraste por clase y montajes visuales
# por clase.

# In[12]:


disp_img = RAIZ_IMG is not None and PIL_OK
if not disp_img:
    print("OMITIDO: sin carpeta de imagenes o sin Pillow.")
else:
    files_all = [f for f in os.listdir(RAIZ_IMG) if f.lower().endswith(".jpg")]
    muestra = random.sample(files_all, min(3000, len(files_all)))
    dims, anchos, altos, kbs = Counter(), [], [], []
    for i, f in enumerate(muestra, 1):
        with Image.open(RAIZ_IMG / f) as im:
            dims[(im.width, im.height)] += 1
            anchos.append(im.width)
            altos.append(im.height)
        kbs.append(os.path.getsize(RAIZ_IMG / f) / 1024)
        if i % 1000 == 0:
            print(f"  {i}/{len(muestra)}")
    tot = sum(dims.values())
    top = dims.most_common(6)
    print("Muestra:", tot, "imagenes")
    print("Tamaños distintos en la muestra:", len(dims))
    print("Mas frecuentes:", [(f"{w}x{h}", f"{100 * c / tot:.1f} %") for (w, h), c in top])
    print(f"Mediana ancho x alto: {np.median(anchos):.0f} x {np.median(altos):.0f}")
    print(f"Mediana de peso: {np.median(kbs):.1f} KB | min {min(kbs):.1f} | max {max(kbs):.1f} KB")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(anchos, bins=40, color="#4575B4", edgecolor="black", linewidth=0.4)
    axes[0].set_title("Ancho de las imágenes (px)")
    axes[0].set_xlabel("px")
    axes[1].hist(kbs, bins=40, color="#1A9850", edgecolor="black", linewidth=0.4)
    axes[1].set_title("Peso de archivo (KB)")
    axes[1].set_xlabel("KB")
    plt.tight_layout()
    plt.show()


# **Conclusión práctica (Fase 5.1–5.2):** la mayoría son 800×600 → para la red conviene
# **redimensionar a 224×224**. Como solo 3.4 % son casi cuadradas, un `resize` directo *deforma* la hoja;
# mejor `Resize(256)` con proporción + `CenterCrop(224)` (o *letterbox* con pad).

# In[13]:


if not disp_img:
    print("OMITIDO.")
else:
    ratios = [w / h for w, h in zip(anchos, altos)]
    casi_cuadradas = np.mean([(0.9 <= r <= 1.1) for r in ratios])
    print(f"Proporción media (alto/ancho invertido->w/h): {np.mean(ratios):.3f}")
    print(f"Imágenes casi cuadradas (0.9 <= w/h <= 1.1): {100 * casi_cuadradas:.1f} %")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(ratios, bins=50, color="#91BFDB", edgecolor="black", linewidth=0.4)
    ax.axvline(1.0, color="red", linestyle="--", label="1:1 (cuadrada)")
    ax.set_title("Distribución de relación de aspecto (ancho/alto)")
    ax.set_xlabel("w/h")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ### 5.3 Brillo (media) y contraste (desviación estándar) por clase
# 
# n = 150 imágenes por clase, semilla fija. Sirve para detectar sesgo de captura
# (iluminación o cámara distintas **entre clases**, lo cual sería un sesgo grave).

# In[14]:


if not disp_img:
    print("OMITIDO.")
else:
    t0 = time.time()
    filas = []
    for lab in ORDEN:
        g = df.loc[df.label == lab, "image_id"]
        for f in random.sample(list(g), min(150, len(g))):
            with Image.open(RAIZ_IMG / f) as im:
                arr = np.asarray(im.convert("L"), dtype=np.float32)
            filas.append({"label": lab, "archivo": f, "brillo": arr.mean(),
                          "contraste": arr.std()})
    img_stats = pd.DataFrame(filas)
    resumen = img_stats.groupby("label")[["brillo", "contraste"]].agg(["mean", "std"]).round(2)
    resumen.index = [NOMBRE[i] for i in resumen.index]
    display(resumen)
    print(f"n total: {len(img_stats)} | tiempo {time.time() - t0:.1f} s")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    datos_b = [img_stats.loc[img_stats.label == lab, "brillo"] for lab in ORDEN]
    datos_c = [img_stats.loc[img_stats.label == lab, "contraste"] for lab in ORDEN]
    for ax, datos, titulo in ((axes[0], datos_b, "Brillo (media de gris)"),
                              (axes[1], datos_c, "Contraste (desv. estándar de gris)")):
        ax.boxplot(datos)
        ax.set_xticklabels([NOMBRE[i].split(" ")[0] for i in ORDEN])
        ax.set_title(titulo)
    plt.tight_layout()
    plt.show()
    print("Rango de brillo medio entre clases:",
          f"{img_stats.groupby('label').brillo.mean().min():.1f}",
          "-", f"{img_stats.groupby('label').brillo.mean().max():.1f}",
          "| Rango de contraste:", f"{img_stats.groupby('label').contraste.mean().min():.1f}",
          "-", f"{img_stats.groupby('label').contraste.mean().max():.1f}")


# ### 5.4 Montajes visuales por clase (obligatorio)
# 
# Respuesta a la consigna: *«determinar si sirve toda la información de las imágenes»* —
# se muestran muestras aleatorias con **hoja completa, fondo/tallo y bordes**, para decidir qué
# regiones aportan síntoma y cuáles son ruido (al momento de elegir *augmentations* y ROI).

# In[15]:


if not disp_img:
    print("OMITIDO.")
else:
    fig, axes = plt.subplots(len(ORDEN), 5, figsize=(17, 4.2 * len(ORDEN)))
    for r, lab in enumerate(ORDEN):
        muestras = random.sample(list(df.loc[df.label == lab, "image_id"]), 5)
        for c, f in enumerate(muestras):
            with Image.open(RAIZ_IMG / f) as im:
                axes[r, c].imshow(im)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"\n{NOMBRE[lab]}", fontsize=9, color=PALETA[lab])
            if c == 0:
                axes[r, c].text(-0.15, 0.5, NOMBRE[lab], rotation=90,
                                transform=axes[r, c].transAxes, va="center",
                                fontsize=10, color=PALETA[lab], fontweight="bold")
        for c in range(len(muestras), 5):
            axes[r, c].axis("off")
    fig.suptitle("Muestras aleatorias por clase (semilla 42)", y=1.005, fontsize=13)
    plt.tight_layout()
    plt.show()


# ### 5.6 Experimento — hojas vs "no-hojas" (bitácora)
# 
# **Objetivo:** el trabajo se apoya en imágenes de **hojas**; lo que no sea hoja (tierra, piedras,
# fondo dominante) no aporta. Regla: **no se borra nada del dataset original** — se mide, se
# documenta y se **copia** lo afectado a una carpeta aparte (`datos/cambios 01 - no hojas/`).
# 
# **Experimento 1 — métricas de color/fondo (HSV + textura + cromático), sin dependencias nuevas**
# 1. HSV con PIL; máscaras: *verde/amarillo hoja* (H 30–110°, S ≥ 50, V ≥ 60 — llega hasta 110°
#    para **no** descartar amarillamientos patológicos = síntoma), *neutro* (S ≤ 30 y V ≤ 60 ó ≥ 210),
#    *suelo* (H 5–35°, S 40–140, V 60–190) y *plano* (bloques 32×32 con desv. estándar < 8).
# 2. `score_hoja = pct_verde − 0.5·pct_plano − 0.3·pct_neutro` (exploratorio, no sirve solo).
# 3. El criterio final usa **greenness cromático** a 160×120 (no depende del brillo):
#    * `verde_suave` = fracción de píxeles con **G − max(R,B) > 4 y G > 20**
#    * `celdas_verdes` = nº de celdas 20×20 (de 48) con **> 20 %** de píxeles verdes
# 4. Salidas: histograma, reparto por clase y **montaje de las 20 de menor score** →
#    **primero se mira, después se fija el umbral** (regla anti-divergencia).

# In[16]:


if not disp_img:
    print("OMITIDO: sin carpeta de imagenes.")
else:
    import shutil
    t0 = time.time()
    RAIZ_PROY = RUTA_CSV.resolve().parents[2]
    CAMBIOS1 = RAIZ_PROY / "datos" / "cambios 01 - no hojas"
    (CAMBIOS1 / "imagenes").mkdir(parents=True, exist_ok=True)

    def metricas_hoja(path):
        with Image.open(path) as im:
            hsv = np.asarray(im.convert("HSV"), dtype=np.int16)
            gris = np.asarray(im.convert("L"), dtype=np.float32)
            rgb120 = np.asarray(im.convert("RGB").resize((160, 120)), dtype=np.int16)
        Hdeg = hsv[..., 0] * 360.0 / 255.0
        S, V = hsv[..., 1], hsv[..., 2]
        verde = ((Hdeg >= 30) & (Hdeg <= 110) & (S >= 50) & (V >= 60)).mean()
        neutro = ((S <= 30) & ((V <= 60) | (V >= 210))).mean()
        suelo = ((Hdeg >= 5) & (Hdeg <= 35) & (S >= 40) & (S <= 140) & (V >= 60) & (V <= 190)).mean()
        h32 = gris.shape[0] // 32 * 32
        w32 = gris.shape[1] // 32 * 32
        bloques = (gris[:h32, :w32].reshape(h32 // 32, 32, w32 // 32, 32)
                     .transpose(0, 2, 1, 3).reshape(-1, 32 * 32))
        pct_plano = float((bloques.std(axis=1) < 8).mean())
        R120, G120, B120 = rgb120[..., 0], rgb120[..., 1], rgb120[..., 2]
        verde_cromo = float(((G120 > R120 + 8) & (G120 > B120 + 8) & (G120 > 35)).mean())
        m_suave = (G120 - np.maximum(R120, B120) > 4) & (G120 > 20)
        verde_suave = float(m_suave.mean())
        celdas = m_suave.reshape(6, 20, 8, 20).transpose(0, 2, 1, 3).reshape(48, 400).mean(axis=1)
        celdas_verdes = int((celdas > 0.20).sum())
        return verde, neutro, suelo, pct_plano, verde_cromo, verde_suave, celdas_verdes

    files_all = sorted(f for f in os.listdir(RAIZ_IMG) if f.lower().endswith(".jpg"))
    filas = []
    for i, f in enumerate(files_all, 1):
        verde, neutro, suelo, plano, vc, vs, cv = metricas_hoja(RAIZ_IMG / f)
        filas.append({"image_id": f, "pct_verde": verde, "pct_neutro": neutro,
                      "pct_suelo": suelo, "pct_plano": plano,
                      "score_hoja": verde - 0.5 * plano - 0.3 * neutro,
                      "verde_cromo": vc, "verde_suave": vs, "celdas_verdes": cv})
        if i % 5000 == 0:
            print(f"  {i}/{len(files_all)} procesadas...")
    met = pd.DataFrame(filas).merge(df, on="image_id", how="left")
    print(f"Imagenes procesadas: {len(met)} | tiempo {time.time() - t0:.1f} s")
    display(met[["pct_verde", "pct_neutro", "pct_suelo", "pct_plano", "score_hoja",
                 "verde_cromo", "verde_suave", "celdas_verdes"]]
            .describe().round(3))
    etapa1 = met.pct_verde < 0.15
    etapa2 = etapa1 & (met.verde_suave < 0.035) & (met.celdas_verdes < 3)
    print(f"Etapa 1 - candidatas con pocas hojas HSV (pct_verde < 0.15): {int(etapa1.sum())}")
    print(f"Etapa 2 - heuristica no-hoja (verde_suave < 0.035 y celdas_verdes < 3): "
          f"{int(etapa2.sum())} (+ 8 manuales = 33 total)")
    print(f"Rescatadas como HOJA dentro de la etapa 1: {int((etapa1 & ~etapa2).sum())}")


# In[17]:


if not disp_img or 'met' not in dir():
    print("OMITIDO.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].hist(met.score_hoja, bins=60, color="#4575B4", edgecolor="black", linewidth=0.4)
    axes[0].set_title("Distribucion de score_hoja")
    axes[0].set_xlabel("score_hoja")
    datos = [met.loc[met.label == lab, "score_hoja"] for lab in ORDEN]
    axes[1].boxplot(datos)
    axes[1].set_xticklabels([NOMBRE[i].split(" ")[0] for i in ORDEN], fontsize=8)
    axes[1].set_title("score_hoja por clase")
    plt.tight_layout()
    plt.show()

    res_clase = met.groupby("label").score_hoja.agg(["mean", "median", "min"]).round(3)
    res_clase.index = [NOMBRE[i] for i in res_clase.index]
    display(res_clase)

    print("Las 20 de menor score (validacion visual ANTES de fijar umbral):")
    worst = met.nsmallest(20, "score_hoja").reset_index(drop=True)
    fig, axes = plt.subplots(4, 5, figsize=(19, 14))
    for k, row in worst.iterrows():
        ax = axes.ravel()[k]
        with Image.open(RAIZ_IMG / row.image_id) as im:
            ax.imshow(im)
        ax.set_title(f"score {row.score_hoja:.3f} | verd {row.pct_verde:.2f} | plan {row.pct_plano:.2f}",
                     fontsize=8)
        ax.set_xlabel(row.image_id, fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("20 imágenes con menor score_hoja (candidatas a no-hoja)", fontsize=14)
    plt.tight_layout()
    plt.show()


# **Experimento 2 — criterio fijado y copia de las afectadas**
# 
# Calibración (mostrar → decidir → recién copiar):
# 
# | Intento | Regla | n | Problema detectado al mirar |
# |---|---|---|---|
# | preliminar | `score_hoja < 0.35` | 9,222 (35 %) | falsos masivos: hojas oscuras sobre tierra |
# | intento 1 | `pct_verde < 0.08 y (plano > 0.30 ó neutro > 0.25)` | 25 | 19 falsos (hojas oscuras) |
# | intento 2 | + `verde_cromo < 0.05` | 44 | 11 falsos (hojas desaturadas) |
# | intento 3 | `pct_verde < 0.15 y verde_cromo < 0.015` | 24 | rescataba bien; se cambió a `verde_suave` + celdas |
# | **automática** | **`pct_verde < 0.15 y verde_suave < 0.035 y celdas_verdes < 3`** | **25** | **25/25: tubérculos, raíces, palos** |
# | **+ manual** | **lista de 8 raíces/palos con pasto de fondo** | **33** | **validación final 33/33** |
# 
# Las 8 manuales son raíces o palos donde el pasto de fondo "engaña" al verde cromático
# (1004389140, 2139839273, 4203623611, 746746526, 2698282165, 1905119159, 274726002,
# 3058561440). Control adicional: muestra aleatoria de 48 rescatadas con verde alto →
# **48/48 son hojas** (no se esconden raíces/palos en el resto de rescatadas).
# 
# Salida en `datos/cambios 01 - no hojas/` (todo sobre **copias**, el original intacto):
# 
# | Carpeta/archivo | Contenido |
# |---|---|
# | `outliers/` | las **33** no-hoja (raíces, tubérculos cortados, palos) |
# | `imagenes filtradas/` | las **26,304** del dataset **sin** las 33 intrusas |
# | `labels_filtrado.csv` | 26,304 filas (`image_id`, `label`, `source`) → lista limpia para entrenar |
# | `labels_cambios.csv` | métricas + `flag_no_hoja` + `origen_flag` de las 26,337 |
# | `experimento.md` | bitácora completa del experimento |
# 
# Las copias de corridas previas se **borran** antes de regenerar.

# In[18]:


if not disp_img or 'met' not in dir():
    print("OMITIDO.")
else:
    VERDE_ETAPA1, UMBRAL_SUAVE, CELDAS_MIN = 0.15, 0.035, 3
    RAICES_MANUALES = {
        "1004389140.jpg", "2139839273.jpg", "4203623611.jpg", "746746526.jpg",
        "2698282165.jpg", "1905119159.jpg", "274726002.jpg", "3058561440.jpg",
    }
    heuristica = ((met.pct_verde < VERDE_ETAPA1) & (met.verde_suave < UMBRAL_SUAVE)
                  & (met.celdas_verdes < CELDAS_MIN))
    manual = met.image_id.isin(RAICES_MANUALES) & ~heuristica

    met_out = met.copy()
    met_out["flag_no_hoja"] = heuristica | manual
    met_out["origen_flag"] = np.where(~met_out.flag_no_hoja, "",
                                      np.where(manual, "manual", "heuristica"))
    no_hoja = met_out[met_out.flag_no_hoja].sort_values("verde_suave")

    if (CAMBIOS1 / "imagenes").exists():          # carpeta legada de corridas previas
        shutil.rmtree(CAMBIOS1 / "imagenes")
    OUTLIERS = CAMBIOS1 / "outliers"
    FILTRADAS = CAMBIOS1 / "imagenes filtradas"
    for d in (OUTLIERS, FILTRADAS):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    ids_no_hoja = set(no_hoja.image_id)
    t0c = time.time()
    for i, f in enumerate(files_all, 1):
        shutil.copy(RAIZ_IMG / f, (OUTLIERS if f in ids_no_hoja else FILTRADAS) / f)
        if i % 5000 == 0:
            print(f"  {i}/{len(files_all)} copiadas...")

    met_out.to_csv(CAMBIOS1 / "labels_cambios.csv", index=False)
    labels_filtrado = met_out.loc[~met_out.flag_no_hoja, ["image_id", "label", "source"]]
    labels_filtrado.to_csv(CAMBIOS1 / "labels_filtrado.csv", index=False)

    print(f"Regla: pct_verde<{VERDE_ETAPA1} y verde_suave<{UMBRAL_SUAVE} "
          f"y celdas_verdes<{CELDAS_MIN} | +{int(manual.sum())} manuales")
    print(f"outliers: {len(no_hoja)} ({100 * len(no_hoja) / len(met_out):.2f} %) -> {OUTLIERS}")
    print(f"imagenes filtradas: {len(labels_filtrado)} -> {FILTRADAS}")
    print(f"labels_filtrado.csv: {len(labels_filtrado)} filas")
    print(f"Copia de imagenes: {time.time() - t0c:.1f} s")
    t_no = no_hoja.groupby("label").size().rename("n_no_hoja").to_frame()
    t_no["% de la clase"] = (100 * t_no.n_no_hoja / df.label.value_counts().sort_index()).round(2)
    t_no.index = [NOMBRE[i] for i in t_no.index]
    display(t_no)
    display(labels_filtrado.groupby("label").size().rename("n_filtrado").to_frame())

    n = len(no_hoja)
    ncols = 7
    nfilas = -(-n // ncols)
    fig, axes = plt.subplots(nfilas, ncols, figsize=(23, 3.2 * nfilas))
    axes = axes.ravel()
    for k, (idx, row) in enumerate(no_hoja.iterrows()):
        ax = axes[k]
        with Image.open(RAIZ_IMG / row.image_id) as im:
            ax.imshow(im)
        etiqueta = "M" if row.origen_flag == "manual" else "H"
        ax.set_title(f"[{etiqueta}] suave {row.verde_suave:.3f} | cel {row.celdas_verdes}",
                     fontsize=8)
        ax.set_xlabel(row.image_id, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Las {n} imagenes NO-hoja [H]=heuristica [M]=manual (validacion visual final)",
                 fontsize=14)
    plt.tight_layout()
    plt.show()

    experimento = f"""# Experimento 01 — hojas vs no-hojas (criterio de color/fondo)

* **Fecha:** 28/09/2026
* **Dataset base:** datos/dataset original (intacto, 26,337 imagenes)
* **Metodo:** PIL/numpy sin dependencias nuevas; metricas por imagen en labels_cambios.csv
  * pct_verde: mascara HSV hoja (H 30-110, S>=50, V>=60)
  * pct_plano: bloques 32x32 con std<8; pct_neutro: S<=30 y V<=60 o >=210
  * score_hoja: pct_verde - 0.5*pct_plano - 0.3*pct_neutro (exploratorio, insuficiente)
  * verde_suave: (G - max(R,B) > 4 y G > 20) a 160x120 — verde cromatico robusto a brillo
  * celdas_verdes: celdas 20x20 (de 48) con >20 % de pixeles verdes — hoja con estructura
* **Calibracion (mostrar -> decidir):**
  1. score_hoja<0.35 -> 9,222 (35 %): falsos masivos (hojas oscuras). DESCARTADO.
  2. pct_verde<0.08 y (plano>0.30 o neutro>0.25) -> 25 con 19 falsos positivos (hojas oscuras).
  3. + verde_cromo<0.05 -> 44 con 11 falsos (hojas desaturadas con pasto al fondo).
  4. Etapa automatica final: pct_verde<0.15 y verde_suave<0.035 y celdas_verdes<3
     -> 25 imagenes, 25/25 validadas: tuberculos cortados, raices, palos, lena.
  5. + lista manual de 8 raices/palo con pasto de fondo (verde cromatico las rescataba
     por error): total {len(no_hoja)} imagenes no-hoja.
  6. Control: muestra aleatoria de 48 rescatadas con verde alto -> 48/48 hojas:
     no hay raices/palos escondidos en el resto de rescatadas.
* **Resultado:** {len(no_hoja)} imagenes no-hoja ({100 * len(no_hoja) / len(met_out):.2f} %)
* **Por clase:**
"""
    for idx, row in t_no.iterrows():
        experimento += f"* {idx}: {int(row.n_no_hoja)} ({row['% de la clase']} % de la clase)\n"
    experimento += "* **Limites:** una raiz sobre cesped muy verde (pct_verde>=0.15) podria\n"
    experimento += "  quedar fuera; si aparece en validacion, agregarla a RAICES_MANUALES.\n"
    experimento += "  Color no separa tierra vs lesion marron, ni hoja vs pasto (luz intensa).\n"
    experimento += "* **Salidas en esta carpeta:**\n"
    experimento += f"  * outliers/ -> {len(no_hoja)} copias (las no-hoja)\n"
    experimento += (f"  * imagenes filtradas/ -> {len(labels_filtrado)} copias "
                    f"(dataset completo sin las {len(no_hoja)})\n")
    experimento += (f"  * labels_filtrado.csv -> {len(labels_filtrado)} filas "
                    f"(image_id,label,source) para entrenamiento\n")
    experimento += "  * labels_cambios.csv -> metricas + flag_no_hoja + origen_flag (26,337)\n"
    experimento += "  * experimento.md -> esta bitacora\n"
    with open(CAMBIOS1 / "experimento.md", "w", encoding="utf-8") as fh:
        fh.write(experimento)
    print("\nexperimento.md escrito en", CAMBIOS1)


# ## Fase 6 — Nota metodológica (el EDA **no** hace split)
# 
# **Decisión:** este notebook no divide train/val/test; solo deja el método para la fase de
# entrenamiento:
# 
# * **Subset rápido = 20 % del dataset original**, estratificado por clase (misma proporción por
#   clase, semilla 42) → probar transfer learning y estrategias de balanceo con pruebas rápidas
#   antes de correr el 100 % *(se genera en la fase de entrenamiento, no acá)*.
# * Al dividir, estratificar **también por `source`** (81/19 %) para no mezclar dominios 2019/2020.
# * Anti-fuga: 0 duplicados exactos (Fase 2.1); `image_id` y el nombre `train-<clase>-…` jamás
#   como feature — solo píxeles.
# * Preprocesamiento: `Resize(256) + CenterCrop(224)` (el `resize(224,224)` directo deforma),
#   normalización ImageNet, augmentations acotadas: flip, rotación ±15°, brillo/contraste ±10 %,
#   **sin** cambios fuertes de color (los síntomas son de color).
# * Balanceo (`class_weight` o oversampling) **solo** dentro de entrenamiento.
# 
# *Desarrollo completo en `reportes/00_resumen_consigna_y_avance.md`.*

# ## Fase 7 — Conclusiones
# 
# **Las conclusiones completas** (tablas, números verificados contra las salidas y decisiones)
# están en **`reportes/00_resumen_consigna_y_avance.md` → sección 4**.
# 
# **Resumen corto:**
# * Integridad: 26,337 CSV = 26,337 imágenes · **0 duplicados exactos** (SHA-1) · 0 corruptas.
# * Desbalance **10.36 : 1** (CMD 58.71 % vs CBB 5.67 %) → macro-F1 / balanced accuracy /
#   matriz normalizada; el error más caro es **saludable ↔ enferma**.
# * Las 4,940 de 2019 filtran la clase por el nombre del archivo → **jamás usar `image_id` como feature**.
# * 800×600 dominante (80.4 %) y solo 3.4 % casi cuadradas → `Resize(256) + CenterCrop(224)`.
# * Sin sesgo de captura por clase (brillo 110.6–118.0 / contraste 49.5–53.6 entre clases).
# * Experimento 5.6: **33 imágenes no-hoja** (0.13 % — raíces, tubérculos cortados, palos;
#   25 automáticas + 8 manuales validadas) → `datos/cambios 01 - no hojas/` con
#   `outliers/` (las 33), `imagenes filtradas/` (26,304) y `labels_filtrado.csv`
#   (lista limpia para entrenar). El original queda intacto.
# 
# **Pendientes:** near-duplicates (pHash) · subset 20 % y split → *fase de entrenamiento*.

# ## Fase 8 — Reproducibilidad y entregables
# 
# **Entregable: solo código** — `01_EDA_mandioca.ipynb` (ejecutado) y `01_EDA_mandioca.py` en la
# raíz del proyecto. **No se generan PDF ni HTML** salvo pedido explícito. Las conclusiones viven
# en `reportes/00_resumen_consigna_y_avance.md` (Markdown, sección 4).
# 
# **Estructura de datos:**
# 
# * `datos/dataset original/` → `images/` + `labels.csv` (**intacto**, nunca se modifica).
# * `datos/cambios 01 - no hojas/` → `outliers/` (las 33 no-hoja), `imagenes filtradas/`
#   (26,304 = dataset sin las 33), `labels_filtrado.csv` (lista limpia para entrenar),
#   `labels_cambios.csv` (métricas + flags de las 26,337) y `experimento.md`.
#   Cada experimento futuro = una carpeta `cambios 02 - …`.
# 
# **Google Colab:** subir `labels.csv` (celda 2 lo detecta); para las celdas de imágenes subir
# `mandioca.zip` y ejecutar la celda siguiente.
# 
# **Gate de calidad:**
# 
# ```bash
# python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 01_EDA_mandioca.ipynb
# python -m nbconvert --to script 01_EDA_mandioca.ipynb
# ```

# In[19]:


# Colab: descomentar para subir el dataset de imagenes y descomprimirlo.
# import zipfile
# from google.colab import files
# upload = files.upload()            # subir mandioca.zip
# with zipfile.ZipFile(list(upload)[0]) as z:
#     z.extractall("datos")
# print("listo")


# In[20]:


import nbformat
_nbk = nbformat.read(str(Path(r"D:\final_inteligencia\01_EDA_mandioca.ipynb")), 4)
print("Celdas:", len(_nbk.cells),
      "| markdown:", sum(c.cell_type == "markdown" for c in _nbk.cells),
      "| código:", sum(c.cell_type == "code" for c in _nbk.cells))

