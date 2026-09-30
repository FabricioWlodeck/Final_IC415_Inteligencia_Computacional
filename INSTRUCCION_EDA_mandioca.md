# INSTRUCCIÓN DE TRABAJO — EDA del dataset de mandioca (clasificación de enfermedades)

**Proyecto:** Inteligencia Computacional — Trabajo Final Integrador
**Alumno:** WLODECK, Fabricio Joaquín
**Carpeta del proyecto:** `D:\final_inteligencia\`
**Datos:** `D:\final_inteligencia\datos\dataset original\` (`labels.csv` + `images/` con 26,337 `.jpg` — **intacto, nunca se modifica**); experimentos sobre copias en `datos\cambios 0N - …`; `mandioca.zip` queda en `datos\`
**Consigna:** clasificar imágenes de mandioca en 5 clases: `0 = CBB`, `1 = CBSD`, `2 = CGM`, `3 = CMD`, `4 = Saludable`, con imágenes capturadas con dispositivos de bajo costo en condiciones no controladas.

## Entregable

**Regla de carpetas:** el código (`.ipynb`, `.py`) vive en la **raíz**; los resúmenes/conclusiones van en Markdown a **`reportes\00_resumen_consigna_y_avance.md`**; **entregable = solo código** — ningún PDF/HTML salvo pedido explícito (los comandos quedan abajo por si acaso).

1. `01_EDA_mandioca.ipynb` en la raíz del proyecto, **ejecutado** (salidas y figuras embebidas), en español.
2. `01_EDA_mandioca.py` en la raíz → `python -m nbconvert --to script 01_EDA_mandioca.ipynb`
3. `reportes\01_EDA_mandioca.html` → `python -m nbconvert --to html --embed-images --output-dir reportes 01_EDA_mandioca.ipynb` *(solo si se pide)*
4. `reportes\01_EDA_mandioca.pdf` → Edge headless sobre el HTML (no hay pandoc):
   `"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless --disable-gpu --user-data-dir="%TEMP%\edge_pdf" --no-pdf-header-footer --print-to-pdf="...\reportes\01_EDA_mandioca.pdf" "file:///.../reportes/01_EDA_mandioca.html"`
5. El notebook debe poder subirse a **Google Colab** sin cambios (carga de CSV con detección de entorno).

**Gate de calidad obligatorio antes de entregar:**
`python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 01_EDA_mandioca.ipynb` → **0 errores** (la Fase 5.6 tarda ~16 min sobre las 26,337 imágenes), verificar conteo de figuras y acentos UTF-8.

---

## Datos ya verificados (NO re-investigar, usar como control de coherencia)

| Hallazgo | Valor |
|---|---|
| Filas CSV | 26,337 × 3 columnas (`image_id, label, source`), sin nulos, sin `image_id` duplicados |
| Imágenes | 26,337 `.jpg` en carpeta plana, 1:1 con el CSV (0 faltantes, 0 sobrantes), 3.09 GiB |
| Clases | CBB(0) 1,492 = 5.67 % · CBSD(1) 3,476 = 13.20 % · CGM(2) 3,017 = 11.46 % · CMD(3) 15,462 = 58.71 % · Saludable(4) 2,890 = 10.97 % · ratio máx/mín = 10.36 : 1 |
| `source` | 2020: 21,397 = 81.24 % · 2019: 4,940 = 18.76 % |
| Filtrado de clase por nombre | Las 4,940 de 2019 se llaman `train-<clase>-NNN.jpg` (crosstab 100 % coherente con label); las de 2020 son numéricas |
| Duplicados exactos | 0 (SHA-1 completo, ~30 s en este equipo) |
| Legibilidad | 0 rotas en muestra de 3,000; todas modo RGB |
| Dimensiones | dominante 800×600 (80.4 % del muestreo), 132 tamaños distintos, mediana 119.5 KB |
| Aspecto | media w/h = 1.30, solo 3.4 % casi cuadradas |
| Brillo/contraste por clase | casi idénticos (brillo 110.6–118.0 / contraste 49.5–53.6) → sin sesgo de captura evidente |

---

## Fase 0 — Entorno
1. Primera celda: detectar entorno (`import google.colab` → Colab; si falla → local). En local: rutas candidatas `datos/labels.csv`, `labels.csv` + **ruta absoluta de respaldo**. En Colab: `files.upload()` para el CSV.
2. Dependencias: pandas, numpy, matplotlib, seaborn, **Pillow**, nbconvert (Colab ya las trae).
3. Trampas del equipo si falla algo: alias de Microsoft Store tumba a `python` (Settings → App execution aliases → desactivar); **Workspace Trust deshabilita la extensión Python** → confiar la carpeta; VS Code debe abrirse después de instalar Python.
4. Ejecutar **siempre** con nbconvert al final; nunca dar las celdas por buenas "a ojo".

## Fase 1 — Identidad y diccionario
5. `shape`, `head`, `dtypes` del CSV; listar `images/` y comparar conteos.
6. Diccionario con mapeo **a nombres legibles** en TODAS las salidas: `0 → CBB (bacteriosis)`, `1 → CBSD (rayas marrones)`, `2 → CGM (moteado verde)`, `3 → CMD (mosaico)`, `4 → Saludable`.
7. Citar la consigna: imágenes de dispositivos no profesionales, condiciones de campo variables, síntomas tempranos/avanzados, costo asimétrico de errores.

## Fase 2 — Integridad y calidad (antes de mirar distribuciones)
8. Nulos, `image_id` duplicados, cruce CSV ↔ disco (esperado 0/0).
9. **Duplicados exactos**: escaneo completo con SHA-1 de los 26,337 archivos (lectura por chunks de 1 MB, ~30 s en este equipo). Reportar: total de archivos, hashes únicos, grupos duplicados, archivos involucrados y excedentes.
10. **Legibilidad**: `PIL.Image.verify()` sobre muestra aleatoria (n = 3,000, `random.seed(42)`) + conteo de excepciones; reportar modos de color encontrados.
11. **Coherencia nombre ↔ etiqueta**: crosstab del patrón `train-<tipo>-*.jpg` vs `label`. Debe mostrar que el nombre **revela la clase** en las 4,940 de 2019. Aclarar: el modelo solo ve píxeles, pero el nombre jamás puede usarse como feature y las dos fuentes probablemente sean datasets originales distintos (posible sesgo de dominio).

## Fase 3 — Composición y desbalance
12. Tabla n + % por `label` **aclaramos siempre la normalización** (% sobre el total) + gráfica de barras con etiquetas de % y `n`.
13. Calcular el ratio de desbalance y discutir consecuencias según la consigna: accuracy engaña → proponer **macro-F1, balanced accuracy, matriz de confusión normalizada por fila**, y destacar que el error más grave es confundir saludable ↔ enferma (o perder una enfermedad).
14. Cruce `label × source` en `n` y `%` (normalize por columna **y** por fila, etiquetando cuál es cuál).

## Fase 4 — Fuentes / sesgo de dominio
15. `value_counts` de `source` con % + gráfica; nota de que 2019 trae nombres descriptivos y 2020 numéricos → riesgo de sesgo de dominio (fondo/iluminación/cámara distintos por año) si el split no estratifica también por `source`.
16. Chequeo explícito: ¿algún label aparece solo en una fuente? (no, pero debe quedar el test).

## Fase 5 — Análisis de las imágenes (núcleo del EDA)
17. **Dimensiones**: muestra de 3,000 con semilla → top de tamaños, nº de tamaños distintos, mediana de ancho/alto/KBytes. Conclusión práctica: proponer redimensión a **224×224** para el pipeline.
18. **Aspecto**: % casi cuadradas (3.4 %) → avisar que `resize` directo deforma; sugerir `Resize(256) + CenterCrop(224)` o resize con proporción + pad.
19. **Brillo (media) y contraste (desv. estándar) por clase** (n = 150/clase, semilla) + distribuciones superpuestas → confirmar/invalidar sesgo de captura por clase.
20. **Montajes visuales obligatorios**: grilla con 3–5 muestras aleatorias por clase (semilla fija), con el nombre legible de la clase. Incluir deliberadamente casos que muestren **hoja, tallo/fondo y raíz** para responder a la consigna ("determinar si sirve toda la información") y anotar si aportan síntoma o solo ruido.
21. **Experimento 5.6 — hojas vs "no-hojas"** (bitácora en `datos\cambios 0N - …\experimento.md`):
     medir por imagen (HSV + textura + greenness cromático), **primero mirar montajes, después fijar el criterio**, y solo entonces **copiar** las afectadas a la carpeta de cambios (el original intacto). Criterio final verificado: `pct_verde < 0.15 y verde_suave < 0.035 y celdas_verdes < 3` **+ lista manual de raíces/palos con pasto de fondo** → 33 imágenes (0.13 %), validadas 33/33. Las hojas oscuras/desenfocadas se **rescatan** (no se descartan). Salidas de `datos\cambios 01 - no hojas\`: `outliers\` (las 33), `imagenes filtradas\` (26,304 = dataset sin las 33), `labels_filtrado.csv` (26,304 filas, lista para entrenar), `labels_cambios.csv` (métricas + flags) y `experimento.md`.

## Fase 6 — Nota metodológica de split (el EDA **no** parte los datos)
22. **Decisión del alumno:** este notebook **no** hace split train/val/test; solo deja el método escrito para la fase de entrenamiento: **subset rápido = 20 % del dataset original, estratificado por clase** (misma proporción, semilla 42), generado recién ahí; al partir, estratificar también por `source` (81/19 %).
23. Reglas anti-fuga: cero duplicados entre splits (verificado en Fase 2), nunca usar `image_id` como feature, y si en el futuro aparecen near-duplicates, agruparlas antes de partir.
24. Dejar impresos los parámetros propuestos: tamaño de entrada, normalización ImageNet (mean/std), augmentations candidatas (flip, rotación, brillo/contraste — coherentes con lo visto en Fase 5.19).

## Fase 7 — Conclusiones
25. Las conclusiones completas (tablas, números verificados contra las salidas y decisiones) se escriben en **`reportes\00_resumen_consigna_y_avance.md` → sección 4**, con subtítulos por fase; el notebook solo deja un resumen corto con puntero.
26. **Regla anti-divergencia**: todo número escrito en markdown debe coincidir con la salida ejecutada (en EDA anteriores fallé dos veces: puse 61.02 % cuando era 59.95 %, y χ² = 4.14 cuando scipy con Yates daba 3.91). Recalcular en código o re-validar después de ejecutar.
27. Sección "Pendientes / lo que falta": near-duplicates, subset 20 % y split → *fase de entrenamiento*.

## Fase 8 — Entregables y verificación
28. **Entregable = solo código** (`.ipynb` ejecutado + `.py`). Generar `.html`/`.pdf` **únicamente si se pide explícitamente** (comandos arriba).
29. Instrucciones Colab dentro del notebook: subir `labels.csv`; para la parte de imágenes subir `mandioca.zip` y `!unzip` (la sección de imágenes debe **autodesactivarse** si no hay carpeta, con `if os.path.exists(...)`).
30. Verificación final: nº de celdas, `errores == 0`, nº de figuras > 0, acentos UTF-8 correctos, coherencia markdown ↔ outputs.
