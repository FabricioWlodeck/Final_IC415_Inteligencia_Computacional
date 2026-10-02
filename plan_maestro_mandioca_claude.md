# Plan maestro — Notebook v3 (recall enfermo) — Mandioca IC522

**Objetivo del archivo:** que cualquier sesión (este modelo u otro más barato) pueda retomar el trabajo
sin perder contexto ni decisiones ya tomadas. Actualizar la sección "PROGRESO" cada vez que se complete
una fase.

---

## 1. Contexto y decisión de métrica (ya resuelto, NO volver a discutir)

**Pregunta del usuario:** su prioridad es no confundir ninguna de las 4 clases enfermas (CBB, CBSD,
CGM, CMD) con la clase sana — es decir, minimizar falsos negativos (enfermo→sano). Preguntó si
macro-F1 (usado en `02_transfer_learning_mandioca.ipynb`, v1) es la métrica correcta.

**Conclusión (fundamentada, ver respuesta completa en el historial de chat):**
- Macro-F1 NO es la métrica correcta: pesa igual todos los errores entre 5 clases y no fija ningún
  límite a los falsos negativos.
- F2 binario (enfermo vs sano) tampoco sirve: el dataset es ~80-89% enfermo, un clasificador trivial
  que diga "siempre enfermo" ya saca F2≈0.95-0.98 (casi degenerado).
- **Métrica elegida:** sensibilidad (recall) de la clase agregada "enfermo" con restricción dura
  **≥ 0.97**, ajustada con un umbral τ sobre `p(sano)` en **validación**. Entre los modelos que
  cumplen la restricción, se selecciona por **macro-F1** (para no perder la distinción entre las 5
  clases). Se reportan además métricas clásicas (accuracy, balanced acc, precision/recall/F1 macro y
  weighted, por clase) solo a título informativo, NO para seleccionar modelos.
- Regla de decisión: `pred = "sano"` si `p(sano) ≥ τ`, si no gana el argmax de las 4 enfermedades.
  τ se busca en val como el **mayor** τ tal que sensibilidad(enfermo) ≥ 0.97 (si ninguno alcanza,
  τ=0, nunca predecir sano).

## 2. Decisiones del usuario (respondidas vía tool `question`, NO volver a preguntar)

| Pregunta | Respuesta elegida |
|---|---|
| Esquema de métrica | Recall enfermo + restricción (0.97) + macro-F1 como desempate |
| Objetivo de recall | 0.97 |
| Outliers a excluir (además de las 33 base) | **Red 2** (298 total, archivo `labels_outliers_prueba.csv`, columna `flag_base \| flag_red2`) |
| Cómo armar splits 20%/25% | Reusar manifest v1 (`manifest_20pct.csv`): 20% = v1 menos outliers red2 (reponiendo test a 60/clase desde test_pool_rest si falta); 25% = mismo val/test + train ampliado con 5% extra estratificado desde out_subset |
| LIME: interpretación "4 por clase" | **4 aciertos + 4 errores** por clase (no 2+2) |
| LIME: qué modelo define acierto/error | **Ambos** (antes y después del FT), mostrar explícitamente qué dijo cada uno con probabilidades |
| Alcance deep learning | Cabeza MV2+B7 sobre embeddings CACHEADOS (backbone congelado + sin aug ⇒ equivalente matemático a pasar la red, pero mucho más rápido) en 20% y 25%; fine-tuning completo SOLO en MobileNetV2 (30% backbone descongelado) en 20% y 25% |

## 3. Archivo de salida

- **Notebook:** `D:\final_inteligencia\02_transfer_learning_mandioca_v3_recall_enfermo.ipynb`
- **NO modifica** `02_transfer_learning_mandioca.ipynb` (v1) ni `02b_transfer_learning_mandioca_v2.ipynb` (v2).
- Artefactos nuevos en `modelos\v3\` y `reportes\v3\` (checkpoints, embeddings .npz, *_res.json, CSV, MD, PNG LIME).
- Splits nuevos: `datos\splits\manifest_v3_20pct.csv` y `manifest_v3_25pct.csv`.

## 4. Método de construcción

El notebook se genera con un **script Python generador** (usa `nbformat`), NO se edita el .ipynb a mano.
Script generador: `C:\Users\Fabricio\AppData\Local\Temp\opencode\build_nb_v3.py`
Pickle intermedio de celdas: `C:\Users\Fabricio\AppData\Local\Temp\opencode\nb_cells.pkl`

**Flujo para retomar en otra sesión:**
1. Leer este archivo completo.
2. Leer `build_nb_v3.py` para ver exactamente qué fases ya están escritas (buscar comentarios
   `# FASE N -`).
3. Seguir agregando fases seguidas con el patrón `edit` tool: localizar el bloque final
   `import pickle ... print("Fase X lista:", ...)` y reemplazarlo insertando el código nuevo ANTES del
   guardado del pickle, dejando el guardado + print al final con el número de fase incrementado.
4. Cada fase se verifica ejecutando:
   `& "D:\final_inteligencia\.venv\Scripts\python.exe" "C:\Users\Fabricio\AppData\Local\Temp\opencode\build_nb_v3.py"`
   y debe imprimir "Fase N lista: M celdas" sin errores de sintaxis.
5. **IMPORTANTE sobre comillas:** las celdas se agregan con `code(r"""...""")`. Si el código Python
   embebido necesita docstrings triple-comillas, usar `code(r'''...''')` (comillas simples triples) para
   el wrapper exterior y `"""..."""` para los docstrings internos (evita conflicto de delimitadores).
6. ~~Paso pendiente~~ — **YA EJECUTADO:** el generador termina con `nb["cells"] = cells;
   nbf.write(nb, ...)` y escribió el notebook (46 celdas). Si se modifica el generador, basta con
   correrlo de nuevo (reescribe el .ipynb completo).
7. Smoke test (ya PASÓ, exit 0 — reejecutar solo si se cambia el generador):
   ```powershell
   $env:SMOKE="1"
   & "D:\final_inteligencia\.venv\Scripts\python.exe" -m nbconvert --to notebook --execute `
     --output "02_v3_smoke_out.ipynb" --ExecutePreprocessor.timeout=1500 `
     "D:\final_inteligencia\02_transfer_learning_mandioca_v3_recall_enfermo.ipynb"
   ```
   **OJO:** usar SIEMPRE `--output "02_v3_smoke_out.ipynb"` (archivo aparte) — así el notebook
   principal nunca queda con outputs de smoke y no hace falta "restaurarlo" después. Si los hay
   errores, corregir el generador (NO editar el .ipynb a mano) y regenerar. Antes de un re-smoke tras
   cambios de lógica, borrar los `*_smoke.*` afectados de `modelos\v3\` (si no, la caché devuelve
   resultados viejos).
8. Entregado: el notebook principal está limpio (46 celdas, 0 outputs) y el usuario lo corre completo
   en VS Code con SMOKE sin definir (ver "Cómo correr la corrida completa" en la sección 7).

## 5. Estructura completa de fases (plan de contenido)

### Fase 0 — Entorno y configuración ✅ ESCRITA
- Imports, semillas, `SMOKE`/`TAG`, clase `Cronometro` + lista global `TIEMPOS` (registra etapa,
  detalle, backbone, fracción, n_imgs, segundos, ms_por_img — para la Fase 8).
- Paths: `BASE = Path("D:/final_inteligencia")`, `DIR_IMG`, `DIR_LABELS`, `DIR_OUTLIERS2`, `DIR_SPLIT`,
  `MANIFEST_V1`, `DIR_MODEL`, `DIR_V3 = DIR_MODEL/"v3"`, `DIR_REPORT = BASE/"reportes"/"v3"`,
  `DIR_LIME = DIR_REPORT/"lime"`.
- Constantes: `CLASS_NAMES`, `NUM_CLASSES=5`, `HEALTHY_IDX=4`, `MEAN/STD` ImageNet, `RECALL_OBJ=0.97`,
  `MAX_TEST_PC=60`, `FRAC_25_EXTRA=0.05`.
- `BACKBONES` dict con 7 entradas (clave → nombre timm + tamaño input):
  - mobilenetv2 → `mobilenetv2_100` @224
  - efficientnet_b7 → `tf_efficientnet_b7.ra_in1k` @600
  - resnet50 → `resnet50.a1_in1k` @224
  - densenet121 → `densenet121.ra_in1k` @224
  - vgg16 → `vgg16.tv_in1k` @224
  - efficientnet_b0 → `efficientnet_b0.ra_in1k` @224
  - mobilenetv3_small → `mobilenetv3_small_100.lamb_in1k` @224
  (Verificado que existen en timm 1.0.30 instalado en el venv.)
- `CONFIG_TL` (épocas/lr/patience/batch para TL de cabeza de MV2 y B7), `FT_CFG` (fine-tuning MV2:
  epochs=12, lr=1e-4, patience=3, unfreeze=0.30, batch=32, size=224).
- `LIME_SAMPLES=800`, `LIME_ACIERTOS=4`, `LIME_ERRORES=4`. En SMOKE se reducen todos los parámetros.

### Fase 1 — Datos: exclusión red2 + splits 20%/25% ✅ ESCRITA
- Carga `labels_filtrado.csv` (26,304 filas, ya sin las 33 outliers base).
- Carga `labels_outliers_prueba.csv`, calcula `ids_red2 = flag_base | flag_red2` (298 ids).
- **Guardado de manifiestos solo si `not SMOKE`** (en smoke quedan en memoria).
- **Celda extra SMOKE (sesión 2):** si `SMOKE=1`, reduce en memoria a 8 train / 4 val / 4 test por
  clase (ambas fracciones), con asserts de no-fuga. Sin esta reducción, el smoke entrenaba con el
  dataset completo y el SVM del_timeout (ver sección 7).
- Carga `manifest_20pct.csv` (v1), pasa a `out_subset` las filas cuyo `image_id` está en `ids_red2`.
- Repone el test balanceado a 60/clase si alguna clase quedó corta, tomando de `test_pool_rest` (nunca
  de red2). Asserts de no-fuga train/val/test y de que ningún split activo contiene ids de red2.
- Guarda `manifest_v3_20pct.csv`.
- Rama 25%: parte de la 20% ya corregida, agrega a train un 5% adicional del dataset total,
  muestreado estratificado (clase×source) desde `out_subset` (excluyendo red2). val/test quedan
  idénticos entre ambas fracciones (verificado con assert de comparación de arrays ordenados).
  Guarda `manifest_v3_25pct.csv`.
- `MANIFESTS = {"20pct": man, "25pct": man25}` queda como variable global usada en todas las fases
  siguientes.
- Gráfico de barras comparando distribución de clases por split entre las dos fracciones.

### Fase 2 — Métricas: τ + métricas completas ✅ ESCRITA
Funciones definidas (ya no se vuelven a tocar, solo se usan):
- `buscar_umbral_tau(y_true, prob_sano, objetivo)` → mayor τ con sensibilidad(enfermo) ≥ objetivo.
- `aplicar_tau(prob, tau)` → y_pred con la regla asimétrica.
- `metricas_enfermo_sano(y_true, y_pred)` → sensibilidad, especificidad, VPN, TP/FN/TN/FP,
  `fn_por_clase_enferma` (dict por clase enferma).
- `metricas_completas(y_true, y_pred, y_prob)` → accuracy, balanced_acc, f1/precision/recall
  macro+weighted+por_clase, + llama a `metricas_enfermo_sano`.
- `evaluar_con_tau(y_val, prob_val, y_test, prob_test, objetivo)` → ajusta τ en val, devuelve
  dict con `tau`, `val`, `test`, `val_argmax`, `test_argmax` (estos últimos dos para comparar contra
  la regla simple sin τ, a título informativo).
- `cumple_objetivo(res, objetivo)` → bool.
- `elegir_mejor(resultados: dict[id, res], objetivo)` → `(mejor_id, hubo_alguno_que_cumple: bool)`.
  Filtra por sensibilidad val≥objetivo, entre esos toma el de mayor f1_macro val; si ninguno cumple,
  usa el universo completo (ningún otro criterio adicional, queda documentado en el notebook que es
  un caso degenerado a vigilar).

### Fase 3 — Embeddings cacheados (7 backbones) ✅ ESCRITA
- `MandiocaDataset` (devuelve también `image_id`, no solo x,y — necesario para LIME y para alinear
  embeddings por id).
- `build_transforms(size, aug)` — Resize proporcional + Crop + Normalize, sin aumento salvo si aug=True
  (no se usa aug=True en ningún lado de momento, pero queda la función genérica).
- `Featurizador` (backbone timm congelado, `num_classes=0` = ya da el vector pooled GAP).
- `todas_activas` = unión de image_ids train+val+test de las dos fracciones (sin duplicados) — se
  featuriza UNA sola vez por backbone, se reutiliza para 20% y 25%.
- `extraer_embeddings_backbone(bb_key, bb_cfg, df_imgs)` — cachea en `modelos/v3/emb_{bb_key}.npz`
  (X, y, ids). Usa `Cronometro` para registrar tiempo de extracción por backbone.
- `EMB_RAW[bb_key] = (X, y, ids)` para las 7 claves.
- `splits_de_embeddings(bb_key, manifest_df)` — alinea por image_id a train/val/test de un manifest
  dado.
- `EMB[frac][bb_key][split] = (X, y)` — estructura final usada por Fases 4 y 6.

### Fase 4 — TL cabeza sobre embeddings cacheados (MV2, B7 × 20%/25%) ✅ ESCRITA
- `entrenar_cabeza(exp_id, bb_key, frac, epochs, lr, patience, batch)`:
  - Cabeza = `Sequential(Dropout(0.3), Linear(F, 5))`, entrenada con Adam + CE con pesos de clase
    balanceados, directamente sobre tensores de embeddings (sin pasar por el backbone — ya cacheado).
  - Cada época: entrena, evalúa val y test con `evaluar_con_tau`, guarda historial.
  - Selección de mejor época: score = `(cumple_objetivo, f1_macro_val)` tipo tupla para ordenar primero
    por cumplimiento y luego por F1. Early stopping con `patience`.
  - Cachea checkpoint `.pt` (solo state_dict de la cabeza) + `_res.json` con todo (val/test métricas
    completas, tau, history, tiempos).
- Loop: `RESULTADOS_TL["TLH_{bb_key}_{frac}"]` para bb_key en {mobilenetv2, efficientnet_b7} × frac en
  {20pct, 25pct} → 4 experimentos.
- Tabla comparativa + matrices de confusión normalizadas (test) de los 4.

### Fase 5 — Fine-tuning MV2 (30% backbone) × 20%/25% ✅ ESCRITA
- `Clf` (backbone timm completo + head Sequential Dropout+Linear) — réplica de v1/v2.
- `predecir_proba(model, dataset, batch_size)` → (y, prob) vía softmax, usado para evaluar con
  imágenes reales (ya no hay embeddings cacheados porque el backbone se descongela).
- `fine_tuning_mv2(frac)`:
  - Carga `Clf("mobilenetv2_100")`, inicializa la cabeza con los pesos ya entrenados en
    `TLH_mobilenetv2_{frac}` (checkpoint "antes" de la Fase 4) — así el fine-tuning parte de una
    cabeza ya razonable, no de una cabeza nueva aleatoria.
  - Descongela backbone.parameters() excepto el primer 70% (es decir, el último 30% entrenable) +
    head siempre entrenable.
  - Entrena con imágenes reales (`MandiocaDataset` + `build_transforms(224, aug=False)`), AMP si CUDA,
    Adam lr=1e-4, CE con pesos de clase, hasta `FT_CFG["epochs"]` con early stopping `patience=3`.
  - Mismo criterio de selección de época (τ + f1_macro) que la Fase 4.
  - Guarda checkpoint completo del modelo + `_res.json` con `antes = "TLH_mobilenetv2_{frac}"` para
    poder comparar.
- `RESULTADOS_FT[frac]` para frac en {20pct, 25pct}.
- Tabla antes/después (4 filas: antes-20%, después-20%, antes-25%, después-25%) + grid 2x2 de matrices
  de confusión (filas=antes/después, columnas=20%/25%).

### Fase 6 — Clasificadores clásicos sobre 7 backbones × 2 fracciones ⬜ PENDIENTE
**A implementar.** Contenido planeado:
- `registrar_ml(exp_id, metodo, bb_key, frac, extra, res_evaluado, complejidad, t_s)` — guarda JSON en
  `modelos/v3/{exp_id}_res.json` con la misma estructura que usa `evaluar_con_tau` (val/test/tau).
- Para cada `frac` en {20pct, 25pct} y cada `bb_key` en los 7 backbones:
  - **RandomForest** (`n_estimators=500, class_weight="balanced_subsample"`) — necesita
    `predict_proba` para aplicar τ.
  - **SVM RBF** con `probability=True` (necesario para τ; más lento que v1/v2 que no lo usaban) —
    grid chico `C∈{1,10} × gamma∈{"scale",1e-3}` seleccionado por f1_macro en val (con τ aplicado
    usando el mismo set de validación, cuidado de no mezclar criterios: elegir hiperparámetros por
    el score final `elegir_mejor` sobre val, no por f1 "pelado").
  - **XGBoost** (`objective="multi:softprob"`, `sample_weight` balanceado) — tiene predict_proba
    nativo. Evaluar si usar `tree_method="hist", device="cuda"` para acelerar (xgboost 3.4.1 soporta
    device=cuda).
  - **MLP** (torch, igual arquitectura que v1: 256 ocultas, Dropout 0.3, pesos de clase) — reusar
    patrón de `entrenar_cabeza` pero con una capa oculta adicional.
  - Escalado: `StandardScaler` fit sobre train, aplicado a val/test (para SVM y MLP; RF y XGB no lo
    necesitan pero no molesta aplicar solo a los que lo requieren).
  - **IMPORTANTE:** aplicar `evaluar_con_tau` a las probabilidades de cada clasificador (todos dan
    `predict_proba`), exactamente igual que en deep learning, para que la comparación sea consistente.
  - Total: 7 backbones × 2 fracciones × 4 clasificadores = 56 corridas. Tiempos esperados (según v1):
    RF/XGB rápidos (<15s c/u salvo quizás backbones de más dimensiones), SVM con probability=True más
    lento (puede tardar 1-3 min por combinación en vez de 50-90s), MLP rápido (<5s).
  - Registrar tiempo de cada corrida en `TIEMPOS` vía `Cronometro`.
- Al final: tabla gigante (56 filas) + heatmap o barplot comparando backbones × clasificadores por
  `f1_macro_test` y por `sensibilidad_enfermo_test`, separado por fracción.
- Guardar la tabla intermedia como `tabla_clasicos_v3` (se reusa en Fase 9 para el consolidado).

### Fase 7 — LIME: 4 aciertos + 4 errores × clase × modelo (antes/después FT) ⬜ PENDIENTE
**A implementar.** Contenido planeado:
- Modelos a explicar: el "antes" = mejor `TLH_mobilenetv2_{frac_elegida}` (cabeza congelada) y
  "después" = `FT_mobilenetv2_{frac_elegida}` (fine-tuned). Decidir qué fracción usar para LIME: usar
  la que haya quedado seleccionada como mejor global entre 20%/25% (ver Fase 9) para no duplicar el
  trabajo de LIME 2 veces — PERO el usuario no restringió esto; default razonable: usar la fracción
  con mejor resultado en la Fase 5 (comparar `F1_test` con τ aplicado entre las 2 fracciones de FT).
  Documentar la elección explícitamente en una celda markdown antes de ejecutar LIME.
- Para cada clase (cbb, cbsd, cgm, cmd, healthy) y cada modelo (antes, después):
  - Usar las predicciones ya calculadas en `RESULTADOS_TL`/`RESULTADOS_FT` (test set) para identificar,
    dentro de esa clase real, cuáles filas son `y_true==y_pred` (acierto) y cuáles `y_true!=y_pred`
    (error).
  - **Para healthy:** acierto = predijo healthy bien; error = predijo healthy pero era otra cosa
    (sano→enfermo, falso positivo) — OJO: dentro de la clase real "healthy" los únicos errores
    posibles son sano clasificado como alguna enfermedad (no hay "enfermo→sano" en la clase healthy,
    ese error ocurre en las clases enfermas).
  - **Para las 4 clases enfermas:** priorizar como errores los casos `pred==healthy` (el error crítico
    enfermo→sano) antes que confusiones entre enfermedades; si no hay 4 de ese tipo, completar con
    otras confusiones entre enfermedades y aclarar en el texto cuáles son cuáles.
  - Tomar hasta 4 aciertos y 4 errores por clase y modelo (semilla 42 para el muestreo si hay más
    candidatos que 4). Si una clase/modelo tiene menos de 4 errores disponibles, usar los que haya y
    decirlo explícitamente en el texto (no fallar).
  - **Evitar explicar la misma imagen más de lo necesario:** como pueden pedirse conjuntos distintos
    para "antes" y "después" (porque los aciertos/errores de cada modelo no coinciden), lo esperado es
    que terminen siendo imágenes parcialmente distintas entre ambos modelos — está bien, es información
    relevante (muestra qué corrigió o rompió el fine-tuning).
  - Para cada imagen seleccionada, generar explicación LIME con `lime_image.LimeImageExplainer`,
    `num_samples=LIME_SAMPLES` (800, o 60 en smoke), sobre el modelo correspondiente (antes o después).
  - **Texto explícito pedido por el usuario:** por cada imagen mostrar algo como:
    > "ANTES del FT (MV2 cabeza congelada): predice **CGM** (p=0.62) — real **CBB** → ERROR (confundió
    > enfermedad, no es el error crítico) / → ERROR CRÍTICO (enfermo→sano) / → ACIERTO.
    > DESPUÉS del FT: predice **CBB** (p=0.81) → ACIERTO."
    Con tabla de las 5 probabilidades de cada modelo debajo de la imagen.
  - Guardar cada figura en `reportes/v3/lime/lime_{clase}_{acierto|error}_{idx}.png`.
  - Al final de la fase: tabla resumen con columnas
    `image_id, clase_real, tipo(acierto/error), modelo, clase_predicha, prob_predicha, es_error_critico`.
- Verificar tiempo total estimado: ~5 clases × 2 tipos × 4 ejemplos × 2 modelos = 80 explicaciones
  (un poco menos si hay escasez de errores en alguna clase) — a 800 samples cada una, estimar ritmo
  según v1 (10 imgs×2 modelos×500 samples ≈ 69s ⇒ ~3.45s/explicación a 500 samples ⇒ proporcional a
  800 samples ≈ 5.5s/explicación ⇒ 80×5.5s ≈ 7-8 min). Usar `Cronometro` para medir el tiempo real.

### Fase 8 — Tiempos e inferencia (latencia) ⬜ PENDIENTE
**A implementar.** Contenido planeado:
- Medir **latencia de inferencia** (no entrenamiento) de cada modelo deep final (las 2 cabezas TL
  elegidas + las 2 FT) en batch=1 y batch=32, promediando sobre N≈100-200 iteraciones con warmup,
  igual que hacía la v2 (`medir_latencia`). Reportar ms/imagen.
- Para los clasificadores clásicos: medir tiempo de **extracción del embedding** (ya está en
  `TIEMPOS` desde la Fase 3, reusar) + tiempo de `predict_proba` sobre 1 imagen (simulado repitiendo
  una fila N veces) — reportar ms/imagen total (extracción + clasificación) porque es lo que de
  verdad le toma al pipeline completo decidir sobre una foto nueva.
- Consolidar `TIEMPOS` (la lista global acumulada durante todo el notebook) en un DataFrame y
  exportarlo a:
  - `reportes/v3/tiempos_v3.csv`
  - `reportes/v3/tiempos_v3.md` (tabla Markdown + resumen por etapa: total de entrenamiento vs
    embeddings vs clásicos vs LIME vs inferencia, % del tiempo total de la corrida).
- Incluir un cronograma wall-clock como el de `resultados_y_tiempos_de_ejecucion.md` de la v1 (hora
  inicio/fin de cada fase) — usar `time.strftime` al principio/final de cada fase mayor.

### Fase 9 — Consolidado final ⬜ PENDIENTE
**A implementar.** Contenido planeado:
- Juntar en un solo DataFrame: `RESULTADOS_TL` (4 filas), `RESULTADOS_FT` (2 filas), la tabla de
  clásicos de la Fase 6 (56 filas) → columnas comunes: ID, Metodo, Backbone, Fraccion, Tau,
  Sens_enf_val, Sens_enf_test, F1_val, F1_test, BalAcc_test, Accuracy_test, Precision_macro_test,
  Recall_macro_test, F1_weighted_test, + f1 por clase.
- Marcar con una columna `Cumple_objetivo` (bool, sensibilidad val≥0.97) y ordenar: primero los que
  cumplen, luego por F1_test descendente.
- Exportar `reportes/v3/resultados_v3.csv`.
- Generar tabla Markdown + redactar conclusiones (mejor modelo global, efecto 20% vs 25%, efecto del
  fine-tuning, mejor clásico vs mejor red, costo de la restricción de recall en términos de
  especificidad/falsos positivos de sanas, backbones nuevos: cuál funcionó mejor en los clásicos).
- Opcional (si el usuario lo pide después): insertar un bloque en
  `reportes/00_resumen_consigna_y_avance.md` entre marcadores `<!-- INICIO-RESULTADOS-V3 -->` /
  `<!-- FIN-RESULTADOS-V3 -->`, siguiendo el patrón de v1/v2. **No implementado todavía porque no fue
  pedido explícitamente para la v3** — confirmar con el usuario antes de tocar ese archivo compartido.
- Celda final de reproducibilidad (semillas, versiones de librerías, hardware) igual que v1/v2.

## 6. Pendientes / cosas a verificar antes de la corrida completa

1. ~~**xgboost con GPU**~~ → resuelto: la celda intenta `device="cuda"` con `try/except` y cae a CPU
   automáticamente si falla (probado en smoke, funcionó).
2. ~~**SVM con `probability=True`**~~ → verificado en smoke: el grid completo (3 combinaciones × 7
   backbones × 2 fracciones) corre sin problemas con datos reducidos. En la corrida completa es el
   clasificador más lento; si resulta prohibitivo, bajar `GRID_SVM` a una sola combinación (está en
   la celda "Fase 6 parte SVM" del generador).
3. ~~**VGG16**~~ → sin problema: solo se usa para embeddings (una pasada), nunca para backward.
4. ~~**Fracción para LIME**~~ → resuelto en el código: `FRAC_LIME` se elige automáticamente por el
   mayor F1 test (con τ) de la Fase 5; la decisión queda impresa en una celda markdown.
5. ~~**Manifests v3 no pisan a v1**~~ → verificado: nombres `manifest_v3_20pct.csv` /
   `manifest_v3_25pct.csv`, distintos de `manifest_20pct.csv`.
6. ~~**Entregar .ipynb limpio tras smoke**~~ → resuelto: los outputs de smoke van a un archivo
   aparte (`02_v3_smoke_out.ipynb`); el notebook principal nunca se ejecuta con nbconvert, sale
   limpio del generador (46 celdas, 0 outputs).

### 6.1 Dependencias nuevas detectadas durante el smoke
* **`tabulate`** instalado en el venv (`pip install tabulate`, v0.10.0) — lo necesita
  `DataFrame.to_markdown()` de la Fase 8 para escribir `tiempos_v3.md`. Si se reconstruye el entorno
  desde cero, instalarlo (figura también en requirements si el usuario lleva uno).

### 6.2 Reglas de caché (importante al re-correr)
* Todo en `modelos\v3\` usa sufijo `_smoke` cuando `SMOKE=1`: las corridas smoke **no pisan** los
  checkpoints reales. Si se cambia la lógica de un experimento, **borrar su `*_smoke.*`** o quedará
  el resultado viejo por caché.
* `reportes\v3\` NO tiene sufijo: siempre se limpia antes de la corrida completa (ya limpio ahora;
  la evidencia del smoke vive en `02_v3_smoke_out.ipynb`).

## 7. PROGRESO (actualizar cada sesión)

| Fecha/hora | Fase | Estado | Notas |
|---|---|---|---|
| sesión 1 | Fases 0-5 (config, datos, métricas, embeddings, TL cabeza, FT MV2) | ✅ Escritas y verificadas | generador `build_nb_v3.py` |
| sesión 1 | Fase 6 (clásicos 7×2×4) | ✅ Escrita y verificada | RF/SVM/XGB/MLP con `predict_proba` + τ |
| sesión 1 | Fase 7 (LIME 4+4) | ✅ Escrita y verificada | FRAC_LIME automático; prioriza errores críticos |
| sesión 1 | Fase 8 (tiempos/latencia) | ✅ Escrita y verificada | Cronometro global + `tiempos_v3.md/csv` |
| sesión 1 | Fase 9 (consolidado) | ✅ Escrita y verificada | `resultados_v3.csv` + conclusiones |
| sesión 1 | Escritura del .ipynb (`nbf.write`) | ✅ HECHO | 46 celdas, 0 outputs |
| sesión 1 → 2 | Smoke intento 1 | ❌ TIMEOUT 30 min (sin error de código) | **Causa: faltaba la reducción de muestras en SMOKE** → SVM con `probability=True` entrenó con el dataset completo. Se detectó revisando `modelos/v3` en disco |
| sesión 2 | Fix: reducción SMOKE en Fase 1 | ✅ | Celda nueva: 8 train / 4 val / 4 test por clase en memoria; manifiestos NO se guardan en disco en modo smoke |
| sesión 2 | Fix: guards en Fase 7 (LIME vacío) | ✅ | Con 4 imgs/test puede faltar aciertos/errores por clase → `if len(df_lime)` + avisos |
| sesión 2 | Fix: escapes `\\n` dobles → `\n` | ✅ | 8 líneas con `\n` literal en prints/títulos |
| sesión 2 | Limpieza de artefactos smoke viejos | ✅ | `modelos/v3/*_smoke*` y `reportes/v3` borrados (eran del intento 1 con datos completos) |
| sesión 2 | Smoke intento 2 | ❌ Falló en Fase 8 | `ModuleNotFoundError: tabulate` (para `to_markdown`). Todo lo anterior (Fases 0-7, incl. LIME) OK |
| sesión 2 | Fix: `pip install tabulate` | ✅ | venv `.venv`, v0.10.0 |
| sesión 2 | **Smoke intento 3 — PASÓ (exit 0)** | ✅ | Ejecución completa de punta a punta: 46/46 celdas sin error. Salidas en `02_v3_smoke_out.ipynb` (8.6 MB). Generó: 7 embeddings, 4 TLH, 2 FT, 56 clásicos (62 filas en `resultados_v3.csv`), 18 figuras LIME (2 errores críticos detectados), `tiempos_v3.md/csv`, `info_reproducibilidad_v3.json` |
| sesión 2 | Verificación de resultados smoke | ✅ | 62 filas (14 RF + 14 SVM + 14 XGB + 14 MLP + 4 TLH + 2 FT), 24 columnas de métricas (accuracy, F1 macro/weighted, precision/recall, por clase, sensibilidad, especificidad, VPN, τ, Cumple_objetivo) |
| sesión 2 | Limpieza post-smoke de `reportes/v3` | ✅ | Borrado para que la corrida real no mezcle PNGs/CSVs de smoke (lección de la v1). Evidencia preservada en `02_v3_smoke_out.ipynb` |
| sesión 2 | **ENTREGA — notebook listo para corrida completa** | ✅ | `02_transfer_learning_mandioca_v3_recall_enfermo.ipynb` limpio, 46 celdas |

### Cómo correr la corrida completa (siguiente paso del usuario)
1. Abrir `D:\final_inteligencia\02_transfer_learning_mandioca_v3_recall_enfermo.ipynb` en VS Code
   con el kernel del venv `D:\final_inteligencia\.venv`.
2. Asegurarse de que **no** esté definida la variable de entorno `SMOKE` (o `SMOKE=0`).
3. Run All. Tiempo estimado: **1.5-2.5 h** (dominado por embeddings de B7@600 y fine-tuning MV2;
   los clásicos y LIME son rápidos).
4. Al terminar: revisar `reportes\v3\resultados_v3.csv`, `reportes\v3\tiempos_v3.md` y
   `reportes\v3\lime\*.png`. Completar la celda markdown "Conclusiones" de la Fase 9 con los
   números reales.
5. Opcional: exportar el notebook a script (`nbconvert --to script`) para tener el .py.

**"Verificada" (fases escritas):** el generador corre sin excepciones.
**"Smoke PASÓ":** el notebook ejecutó completo dentro de Jupyter (nbconvert) con `SMOKE=1`, 0 errores
de celda, todos los artefactos se generaron. **Los resultados numéricos del smoke NO significan nada**
(datos reducidos a 8 train/clase, 1-2 épocas) — solo validan el pipeline.

**"Verificada" significa:** el script generador corre sin excepciones y reporta el conteo de celdas
esperado. **NO significa** que el código ya se ejecutó dentro de un notebook real contra los datos —
eso recién se comprueba en el smoke test (pendiente, fases 6-9 deben estar completas primero porque
nbconvert ejecuta el notebook completo de punta a punta).
