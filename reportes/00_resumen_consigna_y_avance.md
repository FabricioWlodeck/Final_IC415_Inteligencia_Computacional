# Resumen — Consigna y avance del Trabajo Final Integrador

**Asignatura:** IC522 · Inteligencia Computacional
**Alumno:** WLODECK, Fabricio Joaquín
**Tema:** Clasificación de enfermedades en mandioca a partir de imágenes (5 clases)
**Última actualización:** 30/09/2026 (~04:30 h) — **corrida completa del `02` EN CURSO**

---

## 0. ESTADO ACTUAL — leer primero al retomar en otra sesión

### 0.1 ¿Dónde estamos?

| Componente | Estado |
|---|---|
| `01_EDA_mandioca.ipynb` | ✅ Ejecutado, 39 celdas, 0 errores, 10 figuras (intacto) |
| `02_transfer_learning_mandioca.ipynb` | ✅ Construido y validado (38 celdas: 10 md + 28 code); smoke completo pasó 0 errores |
| Corrida **full** (SMOKE=False, GPU) | 🔄 **EN CURSO** — proceso `python` PID 5208 desde 01:00 del 30/09; GPU 87 %, ~4 GB/6 GB VRAM (B7 a 600 px) |
| E1 (MV2 sin aug) | ✅ Terminado 01:19 → `modelos\E1.pth` + `E1_res.json` |
| E2 (MV2 con aug) | ✅ Terminado 01:35 → `modelos\E2.pth` + `E2_res.json` |
| E3 (B7 sin aug) | ✅ Terminado 02:50 → `modelos\E3.pth` + `E3_res.json` |
| E4 (B7 con aug) | 🔄 Entrenando (inició ~02:50; ~10-14 min/época observado en E3) |
| E5 (fine-tuning MV2) | ⏳ Pendiente (sigue a E4) |
| E6-E9 (RF/SVM/XGB/MLP reales) | ⏳ Pendiente |
| LIME antes/después (real) | ⏳ Pendiente |
| `reportes\02_resultados_experimentos.csv` + §6 insertado en este `.md` | ⏳ Se genera en las últimas celdas del notebook (Fase 7) |
| Notebook con outputs | 🟡 El `.ipynb` en disco aún tiene 1 output residual: error `ModuleNotFoundError: No module named 'torch'` en la celda 2 (sesión de VS Code con el interpreter equivocado, **anterior** al fix de kernel; ver §3.5). La corrida en curso lo reescribe entero al terminar → desaparece. **No confiar en los outputs del `.ipynb` hasta que nbconvert termine.** |

**Al terminar la corrida, verificar:**
1. Existen `modelos\E4.pth`, `E5.pth`, `E6-E9_{mobilenetv2|efficientnet}_res.json` (sin sufijo `_smoke`).
2. Existe `reportes\02_resultados_experimentos.csv`.
3. Este archivo tiene insertada la **§6 Resultados** entre los marcadores `<!-- INICIO-RESULTADOS-TL -->
### §6 — Resultados de entrenamiento (ítems 1-4 y 6)

**Test: 300 imágenes balanceadas (60 por clase), semilla 42, sin fuga (ver notebook 02).**

| ID | Método | Backbone | Input | Aug | F1-val | F1-test | Balanced acc |
|---|---|---|---|---|---|---|---|
| E1 | transfer learning (cabeza) | mobilenetv2_100 | 224.0 | no | 0.6152 | 0.5945 | 0.6000 |
| E2 | transfer learning (cabeza) | mobilenetv2_100 | 224.0 | si | 0.5854 | 0.5761 | 0.5867 |
| E3 | transfer learning (cabeza) | tf_efficientnet_b7.ra_in1k | 600.0 | no | 0.6551 | 0.6267 | 0.6367 |
| E4 | transfer learning (cabeza) | tf_efficientnet_b7.ra_in1k | 600.0 | si | 0.6455 | 0.6482 | 0.6567 |
| E5a | E1 (antes FT) | mobilenetv2_100 | 224.0 | no | 0.6152 | 0.5945 | 0.6000 |
| E5b | MV2 fine-tuning (despues) | mobilenetv2_100 | 224.0 | no | 0.6875 | 0.6189 | 0.6267 |
| E6_mobilenetv2 | RandomForest | mobilenetv2_100 | nan | - | 0.3537 | 0.2942 | 0.3467 |
| E7_mobilenetv2 | SVM RBF | mobilenetv2_100 | nan | - | 0.6204 | 0.5623 | 0.5767 |
| E8_mobilenetv2 | XGBoost | mobilenetv2_100 | nan | - | 0.5892 | 0.5295 | 0.5400 |
| E9_mobilenetv2 | MLP (red sobre embeddings) | mobilenetv2_100 | nan | - | 0.6502 | 0.5629 | 0.5700 |
| E6_efficientnet | RandomForest | tf_efficientnet_b7.ra_in1k | nan | - | 0.4434 | 0.3342 | 0.3867 |
| E7_efficientnet | SVM RBF | tf_efficientnet_b7.ra_in1k | nan | - | 0.6370 | 0.5837 | 0.5933 |
| E8_efficientnet | XGBoost | tf_efficientnet_b7.ra_in1k | nan | - | 0.6362 | 0.5669 | 0.5800 |
| E9_efficientnet | MLP (red sobre embeddings) | tf_efficientnet_b7.ra_in1k | nan | - | 0.6693 | 0.6148 | 0.6233 |

- **Mejor modelo global**: `E4` (transfer learning (cabeza), tf_efficientnet_b7.ra_in1k) →
  **macro-F1 test = 0.6482** (val 0.6455).
- **Efecto de la augmentación**: MobileNetV2 0.5945 → 0.5761 (Δ = -0.0184) ·
  EfficientNet-B7 0.6267 → 0.6482 (Δ = +0.0215).
- **Fine-tuning MobileNetV2 (E5)**: antes 0.5945 → después 0.6189
  (**Δ = +0.0244**); se descongeló el 30 % de los tensores del backbone (lr 1e-4).
- **Ítem 4 — clasificadores sobre embeddings**: mejor clásico = `E9_efficientnet` (MLP (red sobre embeddings) sobre tf_efficientnet_b7.ra_in1k)
  con macro-F1 test = 0.6148.
- **Sesgo de fuente (comprobación cuantitativa)**: macro-F1 2019 vs 2020 =
  0.6011 / 0.5823 (antes del FT) y
  0.5541 / 0.6294 (después del FT).
- **LIME**: 20 explicaciones (10 imágenes del test balanceado × checkpoints A/B) en `reportes/lime_antes_despues/`;
  las regiones señaladas se concentran en el limpio y los síntomas del folíolo, no en marco/fondo → sin evidencia de sesgo espurio.

Pendientes de la consigna: ítem 7 (búsqueda de hiperparámetros) e ítem 8 (robustez con peores fotos).

<!-- FIN-RESULTADOS-TL -->` (si el notebook corrió en modo full; en SMOKE **no** se toca este archivo).
4. `reportes\lime_antes_despues\` con las figuras reales (antes: `E1/E2`-best; después: `E5`).
5. El notebook no quedó con outputs de error.

### 0.2 Cómo ejecutar (si hay que re-lanzar)

```powershell
# FULL (reales; ~2.5-4 h — E3/E4 dominan). Corre desde D:\final_inteligencia:
cd D:\final_inteligencia
.\.venv\Scripts\python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 02_transfer_learning_mandioca.ipynb

# SMOKE (prueba rápida, ~10-15 min): 1 época, subsets 8/4/4 por clase, LIME reducido.
# NO sobreescribe manifest ni resultados reales (todo se guarda con sufijo _smoke):
$env:SMOKE='1'
.\.venv\Scripts\python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 02_transfer_learning_mandioca.ipynb
Remove-Item Env:SMOKE
```
- **Caché**: cada celda de entrenamiento omite el re-entreno si ya existen su `.pth` + `_res.json` → re-lanzar solo completa lo faltante (E1-E3 no se repiten).
- Si E3/E4 da **CUDA OOM** → bajar `batch=4` a `2` en `CONFIG` (celda de Fase 0).
- `TqdmWarning: IProgress not found` = inofensivo (falta `ipywidgets`).

### 0.3 Directivas del usuario (cumplidas)

- Framework **PyTorch + timm**; notebook **nuevo** `02_transfer_learning_mandioca.ipynb` (el `01_EDA` queda intacto).
- Test **balanceado 60/clase = 300**; split 20 % del dataset.
- Fine-tuning **solo MobileNetV2**, con evaluación antes/después + LIME en ambos checkpoints.
- Ítems a implementar **todos** (1,3,4,6): comparación de backbones, E1-E5, clasificadores clásicos sobre embeddings, LIME.
- **"Construir todo pero NO ejecutar la corrida completa"** → el notebook quedó limpio y listo; el **usuario** la lanzó él mismo (en curso, §0.1).

---

## 1. Consigna (8 ítems de la cátedra)

| # | Ítem | Estado | Nota |
|---|---|---|---|
| 1 | Pipeline completo de clasificación con transfer learning (backbones sin cabeza + cabeza propia) | ✅ Implementado | `02` Fases 2-3: `timm.create_model(num_classes=0)` + `Dropout(0.3)+Linear(→5)`; corrida en curso |
| 2 | Evaluar/backbones: capacidad vs costo computacional (contexto móvil, baja capacidad) | 🔄 En corrida | E1/E2 = MobileNetV2 (224, 2,2 M params) vs E3/E4 = EfficientNet-B7 (600, ~64 M) → tabla con F1 + params + s/época |
| 3 | Estrategias de transfer learning: fine-tuning, feature extraction, cabeza propia | 🔄 En corrida | Feature extraction = E1-E4 (backbone congelado, BN en eval); fine-tuning = E5 (últimos 30 % de tensores, solo MV2, eval antes/después) |
| 4 | Clasificadores sobre embeddings (Random Forest, SVM, XGBoost) | ✅ Implementado | `02` Fase 5: E6=RF, E7=SVM RBF con grid (C∈{1,10}×gamma∈{scale,1e-3}), E8=XGBoost, E9=MLP torch (con pesos de clase); sobre embeddings GAP de MV2 **y** B7 → 8 corridas. **Falta ejecutar la real** |
| 5 | Métricas con costo diferencial de errores entre clases | ✅ Definido | macro-F1 (métrica principal) + balanced accuracy + F1 por clase + matriz de confusión normalizada por fila; FN (sano→enfermo) es el error caro |
| 6 | Explicabilidad (LIME) + verificación de sesgos espurios | ✅ Implementado | Fase 6b: 5 clases × 2 checkpoints (antes=E1/E2-best, después=E5), 20 explicaciones/clase, figuras + txt. **Falta ejecutar la real**. Sesgos de dominio ya cubiertos en el EDA (§5.3) |
| 7 | Optimización de hiperparámetros (Grid/Random Search, bayesiana, genéticos) | 🟡 Parcial | El grid de SVM (E7) es la parte hecha; falta definir búsqueda sobre la red (lr, dropout, capas…) |
| 8 | Robustez ante variaciones de captura y casos límite | ⏳ Pendiente | "Las peores fotos": ruido, baja luz, desenfoque, recorte parcial — aún no implementado |

---

## 2. Dataset (verificado en el EDA)

| Aspecto | Resultado |
|---|---|
| Dataset original | 26,337 × 3 (`image_id`, `label`, `source`); sin nulos ni IDs duplicados; 0 duplicados exactos (SHA-1); 0 corruptas (muestra 3,000) |
| **Dataset de entrenamiento** | **26,304** = original − 33 no-hojas → `datos\cambios 01 - no hojas\imagenes filtradas\` + `labels_filtrado.csv` |
| Clases (original) | CBB 5,67 % · CBSD 13,20 % · CGM 11,46 % · **CMD 58,71 %** · Saludable 10,97 % → desbalance **10,36 : 1** |
| Fuentes | 2020: 81,24 % · 2019: 18,76 %; las 4,940 de 2019 se llaman `train-<clase>-N.jpg` (**el nombre revela la etiqueta** → jamás usar `image_id` como feature) |
| Imágenes | dominante 800×600 (80,4 %), mediana 119,5 KB; solo 3,4 % casi cuadradas |
| Captura | brillo 110,6–118,0 y contraste 49,5–53,6 entre clases → sin sesgo de captura por clase |
| Preprocesamiento | `Resize(256) + CenterCrop(224)` (MV2) o `Resize(640) + CenterCrop(600)` (B7) + normalización ImageNet (0,485/0,456/0,406 · 0,229/0,224/0,225) |

---

## 3. Desarrollado hasta ahora

### 3.1 Intento previo — `ref personal\pruebas_marzo_Manihot_diseases_detection.ipynb` (179 celdas)
- Preprocesamiento, exploración y subset del 30 % para pruebas rápidas.
- Comparación de backbones ligeros: **MobileNetV2, MobileNetV3, EfficientNetB0**.
- Tres estrategias de balanceo (recorte, pesos de clase, sin balancear); Modelos 1–7 (MV2 224, B0/B4/B7 hasta 512).
- Mejor resultado anotado: **F1 macro ≈ 75 %** sobre test balanceado (80/20). **LIME implementado y funcional**.
- Pendiente de ese intento: fine-tuning completo → hoy lo hace el `02` (E5).

### 3.2 EDA — `01_EDA_mandioca.ipynb` (39 celdas, ejecutado, 0 errores, 10 figuras)
- Fases: diccionario → integridad/calidad (SHA-1) → distribución/desbalance → fuentes/sesgo de dominio → análisis de imágenes → **experimento hojas vs no-hojas (5.6)** → nota metodológica de split → conclusiones → reproducibilidad.
- Conclusiones completas en **§5 de este documento**; exportación de código: `01_EDA_mandioca.py`.
- Instrucciones: `INSTRUCCION_EDA_mandioca.md`.

### 3.3 Notebook nuevo — `02_transfer_learning_mandioca.ipynb` (38 celdas, el entregable principal)

**Estructura (10 markdown + 28 code):**
- Fase 0: config (semilla, rutas, splits, `CONFIG` E1-E4, CUDA, `set_seed(42)`, cudnn `benchmark=False`+`deterministic=True`).
- Fase 1: **splits** → `datos\splits\manifest_20pct.csv` (solo índice; las imágenes nunca se mueven).
- Fase 2: infra (`Clf`, transforms, `Dataset`, pesos de clase, `evaluar()`, `entrenar()`).
- Fase 3: E1-E4 (tabla + curvas).
- Fase 4: E5 fine-tuning + comparación antes/después.
- Fase 5: embeddings `.npz` + E6-E9 (RF/SVM/XGB/MLP).
- Fase 6: LIME antes/después + figuras.
- Fase 7: fuente, reporte (CSV + §6 en este `.md`), conclusiones, reproducibilidad.

**Splits (semilla 42, estratificado `label × source` — evita el sesgo 2019/2020):**
```
26.304  →  subset 20 % = 5.260  (+ out_subset 21.044)
subset  →  trainval 4.208 / test_pool 1.052
trainval→  train 3.366 / val 842
test_pool→ test 300 (60 por clase exactos) + test_pool_rest 752 (held-out)
```
Asserts: cero leakage entre train/val/test; exactamente 60/clase en test. En SMOKE **no** se reescribe el manifest.

**Configs E1-E9:**

| ID | Backbone | Input | Aug | Épocas | Batch | lr | patience | Estado |
|---|---|---|---|---|---|---|---|---|
| E1 | `mobilenetv2_100` (ImageNet `rw_in1k`) | 224 | no | 20 | 32 | 1e-3 | 4 | ✅ 16 ép (early stop) |
| E2 | `mobilenetv2_100` | 224 | **sí** | 20 | 32 | 1e-3 | 4 | ✅ 16 ép (early stop) |
| E3 | `tf_efficientnet_b7.ra_in1k` | 600 | no | 12 | 4 | 1e-3 | 4 | ✅ 7 ép (early stop) |
| E4 | `tf_efficientnet_b7.ra_in1k` | 600 | **sí** | 12 | 4 | 1e-3 | 4 | 🔄 en curso |
| E5 | MV2 (desde E1/E2-best) | 224 | sí | ≤6 | 32 | **1e-4** | 3 | ⏳ últimos 30 % de tensores descongelados; eval antes/después |
| E6 | RF (`n=500`, `balanced_subsample`) sobre embeddings GAP | — | — | — | — | — | — | ⏳ ×{MV2, B7} |
| E7 | SVM RBF + `StandardScaler` + grid sobre **val** | — | — | — | — | — | — | ⏳ ×{MV2, B7} |
| E8 | XGBoost (`n=300, d=6, lr=0.1`) | — | — | — | — | — | — | ⏳ ×{MV2, B7} |
| E9 | MLP torch (256 ocultas, CE con pesos por clase, early stop) | — | — | — | — | — | — | ⏳ ×{MV2, B7} |

- Aug = `Resize(256/640)` + `RandomCrop(224/600)` + `RandomVerticalFlip(0.5)` + `RandomHorizontalFlip(0.5)` + `RandomRotation(15°)`; **solo en train**; val/test siempre `CenterCrop` determinístico.
- IDs de los clásicos: `E6/E7/E8/E9_{mobilenetv2|efficientnet}` vía `bb_key(name)`.

**Infraestructura de entrenamiento (los detalles que importan):**
- `Clf`: `timm.create_model(name, pretrained=True, num_classes=0)` → vector del GAP (1280 dims MV2 / 2560 B7) → `Dropout(0.3) → Linear(→5)`. Solo la cabeza entrena (6.405 params en MV2).
- **Backbone congelado + blindaje BN**: `requires_grad=False` + override `train()` que fuerza `backbone.eval()` (las BN nunca actualizan estadísticas) + sanity check de NaN al crear.
- Guard anti-NaN: `if not torch.isfinite(loss): continue`; optimizer filtrado por `requires_grad`.
- AMP (`autocast` fp16 + `GradScaler`); `DataLoader` con `generator=seed 42`.
- **Pérdida**: `CrossEntropyLoss(weight=...)` con pesos de clase **solo de train**: `w_c = n / (5 · n_c)` → `compute_class_weight("balanced")` (imprime `pesos de clase: {...}` al arrancar; cmd ≈ 0,34 … cbb ≈ 3,5).
- **Parada/selección**: cada época valida en `val`; early stopping sobre **macro-F1 de val** (tolerancia +1e-4, patience 4/3); se guarda el checkpoint de la **mejor** época; al final evalúa `val` + `test` una vez con esos pesos.
- **Métrica de comparación final**: macro-F1 sobre `test` (300 balanceados); balanced accuracy secundaria; F1 por clase y matriz de confusión por fila para diagnóstico.
- **Checkpointing**: `modelos\{ID}.pth` + `modelos\{ID}_res.json` (con `val`, `test`, `history` por época) → re-lanzar no reentrena lo existente.
- **Carga de modelos**: cada `entrenar()` crea una instancia fresca; MV2 descarga 1ª vez en E1 (~14 MB), B7 1ª vez en E3 (~250 MB) → cache HuggingFace en `C:\Users\Fabricio\.cache\huggingface`. E5 hace `load_state_dict` del mejor MV2. LIME carga A=E1/E2-best y B=E5.
- **SMOKE** (env var `SMOKE=1`, TAG="_smoke"): 1 época, 8/4/4 por clase, LIME 60 samples/1 imagen; todo con sufijo `_smoke`; NO toca manifest/CSV/este `.md`; el glob de reportes filtra `("_smoke" in nombre) == SMOKE`.

**Bugs encontrados y corregidos (importante si se edita el notebook):**
1. `efficientnet_b7` no publica pesos en timm 1.0.30 → usar `tf_efficientnet_b7.ra_in1k`.
2. **Backbone sin congelar en E1-E4** → divergencia y buffers BN con NaN → predicción constante (F1 0,0667). Fix: freeze + override `train()` + sanity check + guard de loss no finita.
3. MLP usaba pesos por muestra en `CrossEntropyLoss` (shape `[40]`) → fix: pesos **por clase** (`np.bincount`).
4. Fila E5a del reporte leía las métricas de E5b → ahora lee `E1/E2-best` antes del FT.
5. `KeyError: '2019'` en la fuente en modo smoke → helper `_src()` devuelve NaN.
6. `cudnn.benchmark=True` → `False` + `deterministic=True`.
7. Texto LIME/§6 con triple-comillas → usar `'''` + `chr(10)` para no romper el builder; `orden_ids` con sufijo de backbone; `Xtr_t/ytr_t` (typo Xt_t).

**Builders** (andamiaje, ahora prescindible): `C:\Users\Fabricio\AppData\Local\Temp\opencode\` (`nb_cells_a.py`, `nb_cells_b.py`, `build_nb.py`, `splice.py`, `conclusions_block.py`, `make_splits.py`). **Fuente de verdad = el `.ipynb`**; si se edita, editar el notebook directamente.

### 3.4 Organización de carpetas
- `consigna&notas\` · `datos\` · `ref personal\` · `reportes\` (Markdown) · raíz: `.ipynb`/`.py`/instrucciones.
- `datos\dataset original\` → **intacto, nunca se modifica**.
- `datos\cambios 01 - no hojas\` → `outliers\` (33) · `imagenes filtradas\` (26.304) · `labels_filtrado.csv` · `labels_cambios.csv` · `experimento.md`.
- `datos\splits\manifest_20pct.csv` → índice de splits del `02`.
- `modelos\` → checkpoints + `_res.json` + embeddings `.npz` (los reales sin sufijo; los de prueba con `_smoke`).
- `reportes\` → este archivo · `02_resultados_experimentos.csv` · `lime_antes_despues\`.
- **Entregable: solo código** (`.ipynb` ejecutado + `.py`); ningún PDF/HTML salvo pedido explícito.

### 3.5 Entorno (crítico para la próxima sesión)
- **Venv único**: `D:\final_inteligencia\.venv` → Python 3.12.10, **torch 2.14.0+cu126**, timm 1.0.30, torchvision, sklearn 1.9.1, xgboost 3.4.1, lime 0.2.0.1, nbconvert.
- GPU: **GTX 1660 SUPER 6 GB** (B7 600px con batch 4 usa ~4 GB; OOM → batch 2).
- Kernel Jupyter: **`ic522-mandioca`** (registrado en `%APPDATA%\jupyter\kernels`).
- **Fix VS Code**: el picker mostraba un `~\.venv` fantasma (no existía) → creado junction `C:\Users\Fabricio\.venv` → `D:\final_inteligencia\.venv`, más `.vscode\settings.json` con `"python.defaultInterpreterPath": "D:\\final_inteligencia\\.venv\\Scripts\\python.exe"` en `D:\final_inteligencia` y en la raíz del workspace G:. **Si aparece `No module named 'torch'`, el kernel/espera equivocado → verificar interpreter = `D:\final_inteligencia\.venv\Scripts\python.exe`.**
- Estimación de tiempos (local, observado): E1 ~19 min (16 ép), E2 ~16 min (16 ép), E3 ~73 min (7 ép, ~10 min/ép) → corrida full completa ≈ **2,5-4 h** (E3/E4 dominan). Colab T4 peor (I/O Drive + CPU); TPU no sirve (PyTorch sin XLA).

---

## 4. Resultados parciales de la corrida en curso (SMOKE=False, 30/09/2026)

> Generados por `modelos\{ID}_res.json` reales. **Preliminares**: faltan E4-E9 y LIME; la tabla consolidada final la escribe la Fase 7 en `reportes\02_resultados_experimentos.csv` + §6 de este archivo.

| Exp | Backbone | Input | Aug | Épocas ejecutadas | F1 val | **F1 test** | BalAcc test | F1 test por clase (cbb·cbsd·cgm·cmd·healthy) |
|---|---|---|---|---|---|---|---|---|
| E1 | MobileNetV2 | 224 | no | 16/20 (early stop) | 0,6152 | **0,5945** | 0,6000 | 0,547 · 0,574 · 0,588 · 0,671 · 0,592 |
| E2 | MobileNetV2 | 224 | sí | 16/20 (early stop) | 0,5854 | **0,5761** | 0,5867 | 0,619 · 0,432 · 0,588 · 0,662 · 0,579 |
| E3 | EfficientNet-B7 | 600 | no | 7/12 (early stop) | 0,6551 | **0,6267** | 0,6367 | 0,516 · 0,595 · 0,673 · 0,767 · 0,583 |
| E4 | EfficientNet-B7 | 600 | sí | 🔄 en curso | — | — | — | — |

**Lecturas preliminares (confirmar con la tabla final):**
- **B7 > MV2**: E3 supera a E1 en +3,2 pts de F1 test (y lo hace en solo 7 épocas) → el modelo más grande paga su costo.
- **Aug no ayudó hasta ahora**: E2 < E1 (−1,8 pts); en E2 la clase `cbsd` se desploma (0,432). Pendiente ver si E4 repite el patrón.
- Todos quedan **por debajo del 75 %** del intento previo (ese usaba otro split/otro preproceso — cuidado al comparar: el split actual es estratificado clase×fuente y más exigente).
- Error más caro = `cbb` (minoritaria, F1 0,516-0,619) y `cbsd` variable; `cmd` es la mejor clase (0,66-0,77).

---

## 5. Conclusiones del EDA

### 5.1 Integridad y calidad
- 26,337 CSV = 26,337 imágenes en disco (0 faltantes, 0 sobrantes, solo `.jpg`).
- **0 duplicados exactos** (SHA-1 completo, 227 s) · 0 corruptas en muestra de 3,000 · todas RGB.
- Sin nulos; `label` ∈ [0,4] y `source` ∈ {2019, 2020}.

### 5.2 Distribución y desbalance
- CMD (mosaico) 58,71 % vs CBB 5,67 % → desbalance **10,36 : 1**.
- Implicancia: **macro-F1 / balanced accuracy** y matriz de confusión normalizada; el error más caro es **saludable ↔ enferma** (ítem 5 de la consigna).

### 5.3 Fuentes y sesgo de dominio
- Las 4,940 imágenes de 2019 (`train-<clase>-N.jpg`) tienen coherencia nombre→label del 100 % y revelan la clase → **`image_id` jamás como feature**.
- Toda clase está presente en ambas fuentes (mínimo 313 por celda clase×fuente) → al dividir, estratificar también por `source` (81/19 %). **Hecho en el split del `02`.**

### 5.4 Análisis de imágenes
- 132 tamaños distintos; dominante **800×600 (80,4 %)**; solo **3,4 %** casi cuadradas → `Resize(256) + CenterCrop(224)` (el `resize(224,224)` directo deforma).

### 5.5 Experimento — hojas vs "no-hojas" (Fase 5.6 del EDA)
- Objetivo: el modelo se apoya en hojas; **raíces, tubérculos cortados y palos no aportan**. **Nada se borra del original** → solo copias en `datos\cambios 01 - no hojas\`.
- Bitácora de calibración (mostrar → decidir → recién copiar):

| Intento | Regla | n | Problema al mirar |
|---|---|---|---|
| preliminar | `score_hoja < 0.35` | 9,222 (35 %) | falsos masivos: hojas oscuras sobre tierra |
| intento 1 | `pct_verde < 0.08 y (plano > 0.30 ó neutro > 0.25)` | 25 | 19 falsos (hojas oscuras) |
| intento 2 | + `verde_cromo < 0.05` | 44 | 11 falsos (hojas con pasto al fondo) |
| **automática final** | `pct_verde < 0.15 y verde_suave < 0.035 y celdas_verdes < 3` | **25** | **25/25 válidas** |
| **+ manual** | 8 raíces/palos con pasto de fondo | **33** | **33/33 válidas** |

- Resultado: **33 imágenes no-hoja = 0,13 %** (25 automáticas + 8 manuales), 32 de ellas CBSD (tubérculo) y 1 saludable.
- Control de falsos negativos: muestra de **48 rescatadas con verde alto → 48/48 hojas**.
- Salidas: `outliers\` (33) · `imagenes filtradas\` (**26,304**) · `labels_filtrado.csv` (**26,304 filas**) · `labels_cambios.csv` · `experimento.md`.
- Límite documentado: una raíz sobre césped muy verde (`pct_verde ≥ 0.15`) podría quedar fuera → si aparece, agregarla a `RAICES_MANUALES` y re-correr.

### 5.6 Decisiones de preprocesamiento (ya implementadas en el `02`)
- Subset rápido = **20 %** del dataset filtrado, estratificado por **clase × source**, semilla 42 → ver §3.3.
- Preprocesamiento: `Resize + CenterCrop` (224/600) + normalización ImageNet; augmentations acotadas (flips, rotación ±15°, traslación por crop aleatorio) **sin** cambios fuertes de color (los síntomas son de color).
- Balanceo (`class_weight`) **solo** dentro del entrenamiento (pesos por clase, §3.3).

---

## 6. Resultados de los experimentos de transfer learning

<!-- INICIO-RESULTADOS-TL -->
*(Esta sección la inserta/actualiza automáticamente `02_transfer_learning_mandioca.ipynb` — Fase 7 — al ejecutarse en modo full. Mientras la corrida siga en curso, ver resultados preliminares en §4.)*
<!-- FIN-RESULTADOS-TL -->

---

## 7. Próximos pasos (por ítem de la consigna)

1. **Esperar/verificar la corrida en curso** (§0.1): E4 → E5 → E6-E9 → LIME → CSV + §6. Si falla OOM en E4 → `batch=4→2`.
2. **Ítem 5** — Con la tabla final, revisar matriz de confusión normalizada y costo asimétrico (sano→enfermo); comentar hallazgos en §6/§7 del notebook.
3. **Ítem 7** — Definir búsqueda de hiperparámetros de la red (lr, dropout, capas, descongelamiento progresivo) más allá del grid de SVM ya incluido.
4. **Ítem 8** — Test de robustez con "las peores fotos": ruido, baja luz, desenfoque, hoja parcial fuera de cuadro.
5. **Entregable** — `.ipynb` ejecutado sin errores + `.py` exportado; nada de PDF/HTML.
