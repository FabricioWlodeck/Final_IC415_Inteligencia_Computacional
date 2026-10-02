# Resumen — Consigna y avance del Trabajo Final Integrador

**Asignatura:** IC522 · Inteligencia Computacional
**Alumno:** WLODECK, Fabricio Joaquín
**Tema:** Clasificación de enfermedades en mandioca a partir de imágenes (5 clases)
**Última actualización:** 02/10/2026 — **`02` re-ejecutado: test = 500 (100/clase), Fase 6 (latencias) y Fase 7 (LIME nuevo) incorporadas** · 2ª corrida 13:17→13:41 con guard anti-warnings → **0 avisos**

---

## 0. ESTADO ACTUAL — leer primero al retomar en otra sesión

### 0.1 ¿Dónde estamos?

| Componente | Estado |
|---|---|
| `01_EDA_mandioca.ipynb` | ✅ Ejecutado, 39 celdas, 0 errores, 10 figuras (intacto) |
| `02_transfer_learning_mandioca.ipynb` | ✅ 41 celdas (11 md + 30 code); corrida full del **02/10/2026** con **0 errores** (21 min 42 s; 2ª corrida 13:17→13:41 = 23,5 min con guard anti-warnings, **0 avisos**) |
| Corrida full 02/10 (test=500) | ✅ Sin reentrenar: caches E1-E5 re-evalúan el test nuevo (`[cache] ... re-evaluando SIN reentrenar`); embeddings de test re-extraídos (500 filas) |
| Splits | ✅ `datos\splits\manifest_20pct_test100.csv` (test=500, 100/clase); `manifest_20pct.csv` (test=300) **intacto** para v2/v3 |
| E1-E5 (redes) | ✅ caches re-evaluadas sobre test=500 → F1-test actuales en §6 (mejor: E4 = 0,6727) |
| E6-E9 (RF/SVM/XGB/MLP) + embeddings | ✅ recalculados sobre test=500 (8 corridas) |
| Fase 6 — Latencias (nueva) | ✅ `reportes\02_latencia_deep.csv` (10 filas) + `reportes\02_latencia_clasicos.csv` (8 filas) |
| Fase 7 — LIME (nuevo diseño) | ✅ 20 imágenes (2 aciertos + 2 errores/clase) × 2 modelos auto-elegidos → **40 explicaciones**, 20 PNG en `reportes\lime_antes_despues\` |
| `reportes\02_resultados_experimentos.csv` + §6 | ✅ 14 filas (sin duplicados `_v2`); §6 insertada en la sección 6 de este `.md` |
| Notebook con outputs | ✅ Sin outputs de error |

**Verificado el 02/10 tras la corrida:**
1. ✅ Existen `modelos\E1-E5.pth` + `E6-E9_{mobilenetv2|efficientnet}_res.json` (sin sufijo `_smoke`).
2. ✅ Existe `reportes\02_resultados_experimentos.csv` con **14 filas**.
3. ✅ §6 insertada entre `<!-- INICIO-RESULTADOS-TL -->` y `<!-- FIN-RESULTADOS-TL -->` **en la sección 6** (bug de primera-ocurrencia corregido en la celda de reporte: el reemplazo se ancla en el título `## 6. ...`, así el texto de este checklist que menciona los marcadores ya no confunde al script).
4. ✅ `reportes\lime_antes_despues\` con 20 PNG (2 aciertos + 2 errores por clase; M1 = E4, M2 = E2).
5. ✅ El notebook no quedó con outputs de error.
6. ✅ **0 warnings en todo el notebook** (2ª corrida 02/10): guard anti-`sklearn.utils.parallel.delayed` en las celdas 2, 25 y 31 — la 1ª corrida spameó ~180.000 avisos en la celda de latencias (RF predict corriendo con `warnings.filters` vacío, un race intermitente del paralelo de sklearn) e inflaba el `ms_predict` de RF.

### 0.2 Cómo ejecutar (si hay que re-lanzar)

```powershell
# FULL con caches existentes: ~22-45 min (02/10: 21,7 min; 2ª corrida con guard: 23,5 min — varía con la carga del sistema). Entrenando desde cero: 2,5-4 h (E3/E4 dominan).
cd D:\final_inteligencia
.\.venv\Scripts\python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 02_transfer_learning_mandioca.ipynb

# SMOKE (prueba rápida, ~10-15 min): 1 época, subsets 8/4/4 por clase, LIME reducido.
# NO sobreescribe manifest ni resultados reales (todo se guarda con sufijo _smoke):
$env:SMOKE='1'
.\.venv\Scripts\python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 02_transfer_learning_mandioca.ipynb
Remove-Item Env:SMOKE
```
- **Caché**: cada celda de entrenamiento omite el re-entreno si ya existen su `.pth` + `_res.json` → re-lanzar solo completa lo faltante (E1-E3 no se repiten). Si cambió el tamaño del test, el cache **re-evalúa sin reentrenar** y los embeddings de test se re-extraen solos si el conteo no coincide.
- Si E3/E4 da **CUDA OOM** → bajar `batch=4` a `2` en `CONFIG` (celda de Fase 0).
- `TqdmWarning: IProgress not found` = inofensivo (falta `ipywidgets`).
- `sklearn.utils.parallel.delayed should be used with sklearn.utils.parallel.Parallel` **ya no aparece**: la guardia de sklearn lo dispara cuando `warnings.filters` queda vacío (race intermitente en el paralelo de RF) y en ese caso emite 1 aviso **por árbol** (500 × 310 iteraciones en la celda de latencias). Las celdas 2, 25 y 31 restauran los filtros y filtran ese mensaje.

### 0.3 Directivas del usuario (cumplidas)

- Framework **PyTorch + timm**; notebook **nuevo** `02_transfer_learning_mandioca.ipynb` (el `01_EDA` queda intacto).
- Test **balanceado 60/clase = 300** (directiva original; `manifest_20pct.csv` queda así para v2/v3). **Actualización 02/10 (v1): test = 100/clase = 500** con top-up (40 cbb desde `out_subset`, porque el test_pool de cbb solo tiene 60) → `manifest_20pct_test100.csv`.
- Fine-tuning **solo MobileNetV2**, con evaluación antes/después + LIME en ambos checkpoints.
- Ítems a implementar **todos** (1,3,4,6): comparación de backbones, E1-E5, clasificadores clásicos sobre embeddings, LIME.
- **"Construir todo pero NO ejecutar la corrida completa"** → el notebook quedó limpio y listo; el **usuario** la lanzó él mismo (30/09) y se **re-ejecutó completa el 02/10** con el test nuevo. **Actualización 02/10 (v1)**: medir latencia de inferencia (batch=1), matrices de confusión con **conteos absolutos**, y LIME = 2 aciertos + 2 errores por clase en los modelos que auto-selección el notebook (M1 = menos FN "enfermo predicho sano"; M2 = menor latencia).

---

## 1. Consigna (8 ítems de la cátedra)

| # | Ítem | Estado | Nota |
|---|---|---|---|
| 1 | Pipeline completo de clasificación con transfer learning (backbones sin cabeza + cabeza propia) | ✅ Ejecutado | `02` Fases 2-3: `timm.create_model(num_classes=0)` + `Dropout(0.3)+Linear(→5)`; E1-E4 entrenados y re-evaluados en test=500 |
| 2 | Evaluar/backbones: capacidad vs costo computacional (contexto móvil, baja capacidad) | ✅ Ejecutado | E1/E2 = MobileNetV2 (224, 2,2 M params) vs E3/E4 = EfficientNet-B7 (600, ~64 M) → F1 + params + s/época + **latencias batch=1** (Fase 6) |
| 3 | Estrategias de transfer learning: fine-tuning, feature extraction, cabeza propia | ✅ Ejecutado | Feature extraction = E1-E4 (backbone congelado, BN en eval); fine-tuning = E5 (últimos 30 % de tensores, solo MV2, eval antes/después en §6) |
| 4 | Clasificadores sobre embeddings (Random Forest, SVM, XGBoost) | ✅ Ejecutado | `02` Fase 5: E6=RF, E7=SVM RBF con grid (C∈{1,10}×gamma∈{scale,1e-3}), E8=XGBoost, E9=MLP torch (con pesos de clase); sobre embeddings GAP de MV2 **y** B7 → 8 corridas, test=500 (§6) |
| 5 | Métricas con costo diferencial de errores entre clases | ✅ Definido | macro-F1 (métrica principal) + balanced accuracy + F1 por clase + matriz de confusión con **conteos absolutos** (n=100/clase); FN = **enfermo predicho sano** (real ≠ healthy → predicha healthy) es el error caro |
| 6 | Explicabilidad (LIME) + verificación de sesgos espurios | ✅ Ejecutado | Fase 7 (nuevo diseño 02/10): 20 imágenes del test (2 aciertos + 2 errores por clase) × 2 modelos auto-elegidos (M1 = menos FN, M2 = más rápido) = 40 explicaciones en 20 PNG. Sesgos de dominio ya cubiertos en el EDA (§5.3) |
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

### 3.3 Notebook nuevo — `02_transfer_learning_mandioca.ipynb` (41 celdas, el entregable principal)

**Estructura (11 markdown + 30 code):**
- Fase 0: config (semilla, rutas, splits, `CONFIG` E1-E4, CUDA, `set_seed(42)`, cudnn `benchmark=False`+`deterministic=True`).
- Fase 1: **splits** → `manifest_20pct.csv` (+ `manifest_20pct_test100.csv`, v1) (solo índice; las imágenes nunca se mueven).
- Fase 2: infra (`Clf`, transforms, `Dataset`, pesos de clase, `evaluar()`, `entrenar()`).
- Fase 3: E1-E4 (tabla + curvas).
- Fase 4: E5 fine-tuning + comparación antes/después.
- Fase 5: embeddings `.npz` + E6-E9 (RF/SVM/XGB/MLP).
- Fase 6: **latencia de inferencia** (deep batch=1 / batch mayor; clásicos RF/SVM/XGB/MLP) → `02_latencia_deep.csv` + `02_latencia_clasicos.csv` *(nuevo 02/10)*.
- Fase 7: LIME con 2 modelos auto-elegidos (M1 = menos FN, M2 = más rápido) sobre 20 imágenes del test (2 aciertos + 2 errores por clase) → 40 explicaciones *(nuevo diseño 02/10)*.
- Fase 8: fuente (`f1_por_fuente`), reporte (CSV + §6 en este `.md`), conclusiones, reproducibilidad.

**Splits (semilla 42, estratificado `label × source` — evita el sesgo 2019/2020):**
```
26.304  →  subset 20 % = 5.260  (+ out_subset 21.044)
subset  →  trainval 4.208 / test_pool 1.052
trainval→  train 3.366 / val 842
test_pool→ test 300 (60/clase) + test_pool_rest 752 (held-out)   ← manifest_20pct.csv (v2/v3, intacto)
+ top-up 02/10 → test 500 (100/clase)                            ← manifest_20pct_test100.csv (v1)
    (40 cbb desde out_subset — el pool de cbb solo tiene 60; resto de clases del test_pool:
     cbsd 138, cgm 120, cmd 618, healthy 116 → 100 c/u; test_pool_rest queda 592)
```
Asserts: cero leakage entre train/val/test; v1 valida "test balanceado total = 500". En SMOKE **no** se reescribe el manifest.

**Configs E1-E9:**

| ID | Backbone | Input | Aug | Épocas | Batch | lr | patience | Estado |
|---|---|---|---|---|---|---|---|---|
| E1 | `mobilenetv2_100` (ImageNet `rw_in1k`) | 224 | no | 20 | 32 | 1e-3 | 4 | ✅ 16 ép (early stop) |
| E2 | `mobilenetv2_100` | 224 | **sí** | 20 | 32 | 1e-3 | 4 | ✅ 16 ép (early stop) |
| E3 | `tf_efficientnet_b7.ra_in1k` | 600 | no | 12 | 4 | 1e-3 | 4 | ✅ 7 ép (early stop) |
| E4 | `tf_efficientnet_b7.ra_in1k` | 600 | **sí** | 12 | 4 | 1e-3 | 4 | ✅ 12 ép (tope) |
| E5 | MV2 (desde E1/E2-best) | 224 | sí | ≤6 | 32 | **1e-4** | 3 | ✅ 6 ép (tope); antes/después en §6 |
| E6 | RF (`n=500`, `balanced_subsample`) sobre embeddings GAP | — | — | — | — | — | — | ✅ ×{MV2, B7} |
| E7 | SVM RBF + `StandardScaler` + grid sobre **val** | — | — | — | — | — | — | ✅ ×{MV2, B7} |
| E8 | XGBoost (`n=300, d=6, lr=0.1`) | — | — | — | — | — | — | ✅ ×{MV2, B7} |
| E9 | MLP torch (256 ocultas, CE con pesos por clase, early stop) | — | — | — | — | — | — | ✅ ×{MV2, B7} |

- Aug = `Resize(256/640)` + `RandomCrop(224/600)` + `RandomVerticalFlip(0.5)` + `RandomHorizontalFlip(0.5)` + `RandomRotation(15°)`; **solo en train**; val/test siempre `CenterCrop` determinístico.
- IDs de los clásicos: `E6/E7/E8/E9_{mobilenetv2|efficientnet}` vía `bb_key(name)`.

**Infraestructura de entrenamiento (los detalles que importan):**
- `Clf`: `timm.create_model(name, pretrained=True, num_classes=0)` → vector del GAP (1280 dims MV2 / 2560 B7) → `Dropout(0.3) → Linear(→5)`. Solo la cabeza entrena (6.405 params en MV2).
- **Backbone congelado + blindaje BN**: `requires_grad=False` + override `train()` que fuerza `backbone.eval()` (las BN nunca actualizan estadísticas) + sanity check de NaN al crear.
- Guard anti-NaN: `if not torch.isfinite(loss): continue`; optimizer filtrado por `requires_grad`.
- AMP (`autocast` fp16 + `GradScaler`); `DataLoader` con `generator=seed 42`.
- **Pérdida**: `CrossEntropyLoss(weight=...)` con pesos de clase **solo de train**: `w_c = n / (5 · n_c)` → `compute_class_weight("balanced")` (imprime `pesos de clase: {...}` al arrancar; cmd ≈ 0,34 … cbb ≈ 3,5).
- **Parada/selección**: cada época valida en `val`; early stopping sobre **macro-F1 de val** (tolerancia +1e-4, patience 4/3); se guarda el checkpoint de la **mejor** época; al final evalúa `val` + `test` una vez con esos pesos.
- **Métrica de comparación final**: macro-F1 sobre `test` (500 balanceados desde 02/10; 300 en la corrida original del 30/09); balanced accuracy secundaria; F1 por clase y matriz de confusión con conteos absolutos para diagnóstico.
- **Checkpointing**: `modelos\{ID}.pth` + `modelos\{ID}_res.json` (con `val`, `test`, `history` por época) → re-lanzar no reentrena lo existente.
- **Carga de modelos**: cada `entrenar()` crea una instancia fresca; MV2 descarga 1ª vez en E1 (~14 MB), B7 1ª vez en E3 (~250 MB) → cache HuggingFace en `C:\Users\Fabricio\.cache\huggingface`. E5 hace `load_state_dict` del mejor MV2. LIME (Fase 7) auto-elige **M1** = modelo con menos FN (enfermo predicho sano) y **M2** = menor latencia batch=1 entre E1-E5, y explica 2 aciertos + 2 errores por clase en ambos (02/10: M1=E4, M2=E2).
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
- `datos\splits\manifest_20pct.csv` (test 300; v2/v3) + `manifest_20pct_test100.csv` (test 500; v1) → índices de splits del `02`.
- `modelos\` → checkpoints + `_res.json` + embeddings `.npz` (los reales sin sufijo; los de prueba con `_smoke`).
- `reportes\` → este archivo · `02_resultados_experimentos.csv` · `lime_antes_despues\`.
- **Entregable: solo código** (`.ipynb` ejecutado + `.py`); ningún PDF/HTML salvo pedido explícito.

### 3.5 Entorno (crítico para la próxima sesión)
- **Venv único**: `D:\final_inteligencia\.venv` → Python 3.12.10, **torch 2.14.0+cu126**, timm 1.0.30, torchvision, sklearn 1.9.1, xgboost 3.4.1, lime 0.2.0.1, nbconvert.
- GPU: **GTX 1660 SUPER 6 GB** (B7 600px con batch 4 usa ~4 GB; OOM → batch 2).
- Kernel Jupyter: **`ic522-mandioca`** (registrado en `%APPDATA%\jupyter\kernels`).
- **Fix VS Code**: el picker mostraba un `~\.venv` fantasma (no existía) → creado junction `C:\Users\Fabricio\.venv` → `D:\final_inteligencia\.venv`, más `.vscode\settings.json` con `"python.defaultInterpreterPath": "D:\\final_inteligencia\\.venv\\Scripts\\python.exe"` en `D:\final_inteligencia` y en la raíz del workspace G:. **Si aparece `No module named 'torch'`, el kernel/espera equivocado → verificar interpreter = `D:\final_inteligencia\.venv\Scripts\python.exe`.**
- Estimación de tiempos (local, observado): E1 ~19 min (16 ép), E2 ~16 min (16 ép), E3 ~73 min (7 ép, ~10 min/ép) → corrida desde cero ≈ **2,5-4 h** (E3/E4 dominan). **Con caches, la re-ejecución completa (sin reentrenar) tardó 21 min 42 s el 02/10** (dominado por SVM/XGB grids + latencias + LIME). Colab T4 peor (I/O Drive + CPU); TPU no sirve (PyTorch sin XLA).

---

## 4. Resultados de la PRIMERA corrida (30/09/2026) — histórico (test = 300)

> ⚠️ **Histórico**: corte parcial del 30/09 con **test = 300** (E4 aún entrenando en esa captura). Los valores vigentes (test = 500, con latencias y LIME nuevo) están en **§6**. La tabla consolidada la escribe la Fase 8 en `reportes\02_resultados_experimentos.csv` + §6 de este archivo.

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
### §6 — Resultados de entrenamiento (ítems 1-4 y 6)

**Test: 500 imágenes balanceadas (100 por clase), semilla 42, sin fuga (ver notebook 02).**

| ID | Método | Backbone | Input | Aug | F1-val | F1-test | Balanced acc |
|---|---|---|---|---|---|---|---|
| E1 | transfer learning (cabeza) | mobilenetv2_100 | 224.0 | no | 0.6152 | 0.6171 | 0.6220 |
| E2 | transfer learning (cabeza) | mobilenetv2_100 | 224.0 | si | 0.5854 | 0.5957 | 0.6020 |
| E3 | transfer learning (cabeza) | tf_efficientnet_b7.ra_in1k | 600.0 | no | 0.6551 | 0.6529 | 0.6620 |
| E4 | transfer learning (cabeza) | tf_efficientnet_b7.ra_in1k | 600.0 | si | 0.6455 | 0.6727 | 0.6800 |
| E5a | E1 (antes FT) | mobilenetv2_100 | 224.0 | no | 0.6152 | 0.6171 | 0.6220 |
| E5b | MV2 fine-tuning (despues) | mobilenetv2_100 | 224.0 | no | 0.6875 | 0.6706 | 0.6760 |
| E6_mobilenetv2 | RandomForest | mobilenetv2_100 | nan | - | 0.3537 | 0.2917 | 0.3400 |
| E7_mobilenetv2 | SVM RBF | mobilenetv2_100 | nan | - | 0.6204 | 0.5622 | 0.5780 |
| E8_mobilenetv2 | XGBoost | mobilenetv2_100 | nan | - | 0.5892 | 0.5347 | 0.5480 |
| E9_mobilenetv2 | MLP (red sobre embeddings) | mobilenetv2_100 | nan | - | 0.6502 | 0.5731 | 0.5800 |
| E6_efficientnet | RandomForest | tf_efficientnet_b7.ra_in1k | nan | - | 0.4434 | 0.3506 | 0.3900 |
| E7_efficientnet | SVM RBF | tf_efficientnet_b7.ra_in1k | nan | - | 0.6370 | 0.6080 | 0.6160 |
| E8_efficientnet | XGBoost | tf_efficientnet_b7.ra_in1k | nan | - | 0.6362 | 0.5890 | 0.6000 |
| E9_efficientnet | MLP (red sobre embeddings) | tf_efficientnet_b7.ra_in1k | nan | - | 0.6693 | 0.6224 | 0.6300 |

- **Mejor modelo global**: `E4` (transfer learning (cabeza), tf_efficientnet_b7.ra_in1k) →
  **macro-F1 test = 0.6727** (val 0.6455).
- **Efecto de la augmentación**: MobileNetV2 0.6171 → 0.5957 (Δ = -0.0214) ·
  EfficientNet-B7 0.6529 → 0.6727 (Δ = +0.0197).
- **Fine-tuning MobileNetV2 (E5)**: antes 0.6171 → después 0.6706
  (**Δ = +0.0535**); se descongeló el 30 % de los tensores del backbone (lr 1e-4).
- **Ítem 4 — clasificadores sobre embeddings**: mejor clásico = `E9_efficientnet` (MLP (red sobre embeddings) sobre tf_efficientnet_b7.ra_in1k)
  con macro-F1 test = 0.6224.
- **Sesgo de fuente (comprobación cuantitativa)**: macro-F1 2019 vs 2020 =
  0.6295 / 0.6114 (antes del FT) y
  0.6482 / 0.6753 (después del FT).
- **LIME**: 40 explicaciones (20 imágenes del test: 2 aciertos + 2 errores por clase, donde
  coinciden E4 y E5) en `reportes/lime_antes_despues/`; se revisa si las regiones resaltadas son
  hoja/síntoma o marco/fondo.
- **Latencia de inferencia (batch=1, ms/imagen)**: E1 20.239 · E2 16.958 · E3 83.022 · E4 73.715 · E5 14.806 · mobilenetv2+RF 144.661 · mobilenetv2+SVM 28.831 · mobilenetv2+XGB 26.372 · mobilenetv2+MLP 25.278 · efficientnet+RF 192.3 · efficientnet+SVM 89.01 · efficientnet+XGB 78.589 · efficientnet+MLP 77.812. Tablas completas en
  `reportes/02_latencia_deep.csv` y `reportes/02_latencia_clasicos.csv`.

Pendientes de la consigna: ítem 7 (búsqueda de hiperparámetros) e ítem 8 (robustez con peores fotos).

<!-- FIN-RESULTADOS-TL -->

<!-- INICIO-RESULTADOS-V2 -->
### 6.1 — Resultados v2 (`02b_transfer_learning_mandioca_v2.ipynb`)

**Modo:** SMOKE (validación rápida) · **seed** 42 · **input** 384 px · **test** 300 imgs (60/clase) · **selección por val**.

| ID | Método | F1-val | F1-test | IC 95 % (bootstrap) | Costo/test | ms/img |
|---|---|---|---|---|---|---|
| E1 (v1) | backbone congelado, cabeza lineal @224 | 0.1300 | 0.1167 | [0.0000, 0.2206] | 1.500 | - |
| FT_noaug | FT MobileNetV2 completo (sin aug) | 0.1833 | 0.1971 | [0.0399, 0.3089] | 1.600 | 14.7 |
| FT_aug | FT MobileNetV2 completo (con aug) | 0.1127 | 0.2655 | [0.0868, 0.4134] | 1.350 | 12.8 |

- **Mejor FT (elegido por val):** `FT_noaug` → **macro-F1 test = 0.1971** (val 0.1833), costo 1.600, 14.7 ms/img, 0.88 GFLOPs, 9.2 MB.
- **vs. v1 (E1, backbone congelado):** 0.1167 → 0.1971 (Δ = +0.0805).
- **Δ bootstrap** (FT_aug -> FT_noaug): +0.0646 IC95 [-0.0598, +0.2214] → NO distinguible del ruido.
- **Δ bootstrap** (E1 (v1) -> FT_noaug): -0.0673 IC95 [-0.1828, +0.0564] → NO distinguible del ruido.
- **Calibración:** ECE test 0.0259 → 0.0314 con temperatura T = 0.48.
- **Abstención:** tau = 0.00 (elegido por val, precisión ≥ 0.85); ver tabla 07 en el CSV para cobertura/precisión en test y rechazo OOD.
- **Explicabilidad:** la hoja ocupa 58.0% del cuadro; LIME concentra 57.9% y Grad-CAM 56.6% de la importancia dentro de la hoja (clase real).
- **Robustez (peor caso medido para FT_noaug):** ruido nivel 0.2 → macro-F1 0.0600 (base 0.1971).

Detalle completo: `reportes\03_resultados_v2.csv` (secciones 01-13), informes en `reportes\informes_v2\`, LIME/Grad-CAM en `reportes\lime_gradcam_v2\`, robustez en `reportes\robustez_v2\`.

<!-- FIN-RESULTADOS-V2 -->

---

## 7. Próximos pasos (por ítem de la consigna)

1. ✅ **Corrida completa verificada** (original 30/09 + re-ejecución 02/10 con test=500, Fase 6 latencias y Fase 7 LIME nuevos; §0.1).
2. **Ítem 5** — Con la tabla final, revisar matriz de confusión normalizada y costo asimétrico (sano→enfermo); comentar hallazgos en §6/§7 del notebook.
3. **Ítem 7** — Definir búsqueda de hiperparámetros de la red (lr, dropout, capas, descongelamiento progresivo) más allá del grid de SVM ya incluido.
4. **Ítem 8** — Test de robustez con "las peores fotos": ruido, baja luz, desenfoque, hoja parcial fuera de cuadro.
5. **Entregable** — `.ipynb` ejecutado sin errores + `.py` exportado; nada de PDF/HTML.
