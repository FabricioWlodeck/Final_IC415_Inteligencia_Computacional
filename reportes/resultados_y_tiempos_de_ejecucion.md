# Resultados y tiempos de ejecución — Transfer Learning (`02_transfer_learning_mandioca.ipynb`)

**Registro de la corrida COMPLETA (SMOKE=False)** · Fecha: **30/09/2026, 01:00:04 → 05:23:10 h** · Duración total: **4 h 23 min 06 s**
**Hardware:** Windows · GTX 1660 SUPER 6 GB (VRAM pico ≈ 4 GB con B7@600 batch 4) · venv `D:\final_inteligencia\.venv` (torch 2.14.0+cu126, timm 1.0.30)
**Test:** 300 imágenes balanceadas (60/clase), semilla 42, estratificado `clase × fuente`, sin fuga.
**Fuentes de datos de este documento:** `modelos\*_res.json` (historiales por época + `tiempo_total_s`/`tiempo_ep_s`), `reportes\02_resultados_experimentos.csv` (tabla maestra), timestamps de archivos, §6 insertada en `00_resumen_consigna_y_avance.md`.

---

## 1. Resumen ejecutivo (hallazgos clave)

1. **Mejor modelo global: E4** (EfficientNet-B7 + augmentación) → **macro-F1 test = 0,6482** (val 0,6455, BalAcc 0,6567).
2. **Backbone**: B7 > MV2 en todos los cuadrantes: E3 vs E1 = +0,0322 · E4 vs E2 = +0,0721 de F1 test (a costa de ~10,5 min/época vs ~1 min, y 63,8 M params vs 2,23 M).
3. **Augmentación**: efecto opuesto por backbone — **perjudicó a MobileNetV2** (0,5945 → 0,5761, Δ = −0,0184; en cbsd cayó a 0,4318) y **ayudó a B7** (0,6267 → 0,6482, Δ = +0,0215).
4. **Fine-tuning MV2 (E5) funciona**: test 0,5945 → **0,6189** (Δ = +0,0244) y val 0,6152 → **0,6875** (Δ = +0,0723) descongelando solo el 30 % del backbone con lr 1e-4 (6 épocas, 4 min 43 s).
5. **Clásicos sobre embeddings (ítem 4)**: ninguno supera a la red; el mejor es **E9_efficientnet (MLP sobre embeddings B7) = 0,6148**. El peor es RandomForest (E6_mobilenetv2 = 0,2942, con F1 de cgm = 0,0 — prácticamente no predice esa clase).
6. **LIME**: 10 figuras (20 explicaciones) → regiones sobre el limpio/síntomas del folíolo, no sobre marco/fondo → **sin evidencia de sesgo espurio**.
7. **Sesgo de fuente (comprobación cuantitativa)**: F1 2019 vs 2020 = 0,6011/0,5823 antes del FT y 0,5541/0,6294 después (nunca se usó `image_id` como feature).
8. **Costo**: los experimentos de entrenamiento E1-E5 consumieron el **92,9 %** del tiempo; E3+E4 (B7) solos el **78,4 %**.

---

## 2. Tabla maestra de resultados (todas las corridas)

*Test = 300 imgs balanceadas. Columnas f1_* = F1 por clase (cbb, cbsd, cgm, cmd, healthy).*

| ID | Método | Backbone | Input | Aug | Épocas | Params | F1-val | **F1-test** | BalAcc | f1_cbb | f1_cbsd | f1_cgm | f1_cmd | f1_healthy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | transfer learning (cabeza) | mobilenetv2_100 | 224 | no | 16/20 | 2.230.277 | 0,6152 | **0,5945** | 0,6000 | 0,5472 | 0,5739 | 0,5882 | 0,6711 | 0,5920 |
| E2 | transfer learning (cabeza) | mobilenetv2_100 | 224 | sí | 16/20 | 2.230.277 | 0,5854 | **0,5761** | 0,5867 | 0,6195 | 0,4318 | 0,5882 | 0,6621 | 0,5789 |
| E3 | transfer learning (cabeza) | tf_efficientnet_b7.ra_in1k | 600 | no | 7/12 | 63.799.765 | 0,6551 | **0,6267** | 0,6367 | 0,5161 | 0,5950 | 0,6726 | 0,7671 | 0,5827 |
| E4 | transfer learning (cabeza) | tf_efficientnet_b7.ra_in1k | 600 | sí | 12/12 | 63.799.765 | 0,6455 | **0,6482** | 0,6567 | 0,5253 | 0,6107 | 0,7059 | 0,8029 | 0,5965 |
| E5a | E1 (antes del FT) | mobilenetv2_100 | 224 | no | 16/20 | 2.230.277 | 0,6152 | **0,5945** | 0,6000 | 0,5472 | 0,5739 | 0,5882 | 0,6711 | 0,5920 |
| E5b | MV2 fine-tuning (después) | mobilenetv2_100 | 224 | no* | 6/6 | 1.743.744† | 0,6875 | **0,6189** | 0,6267 | 0,6038 | 0,6557 | 0,5625 | 0,6962 | 0,5763 |
| E6_mv2 | RandomForest (n=500) | mobilenetv2_100 | — | — | — | 500 | 0,3537 | **0,2942** | 0,3467 | 0,2632 | 0,4396 | **0,0000** | 0,4069 | 0,3614 |
| E7_mv2 | SVM RBF (C=1, gamma=scale) | mobilenetv2_100 | — | — | — | 2.813 | 0,6204 | **0,5623** | 0,5767 | 0,4138 | 0,5645 | 0,5631 | 0,6803 | 0,5899 |
| E8_mv2 | XGBoost (n=300, d=6, lr=0.1) | mobilenetv2_100 | — | — | — | 300 | 0,5892 | **0,5295** | 0,5400 | 0,5263 | 0,5714 | 0,4835 | 0,5938 | 0,4727 |
| E9_mv2 | MLP (256 ocultas) | mobilenetv2_100 | — | — | — | 329.221 | 0,6502 | **0,5629** | 0,5700 | 0,5743 | 0,5524 | 0,5400 | 0,6395 | 0,5082 |
| E6_eff | RandomForest (n=500) | tf_efficientnet_b7 | — | — | — | 500 | 0,4434 | **0,3342** | 0,3867 | 0,3733 | 0,5000 | 0,0323 | 0,4444 | 0,3210 |
| E7_eff | SVM RBF (C=10, gamma=scale) | tf_efficientnet_b7 | — | — | — | 2.412 | 0,6370 | **0,5837** | 0,5933 | 0,5474 | 0,5254 | 0,6042 | 0,6821 | 0,5593 |
| E8_eff | XGBoost (n=300, d=6, lr=0.1) | tf_efficientnet_b7 | — | — | — | 300 | 0,6362 | **0,5669** | 0,5800 | 0,5745 | 0,5345 | 0,5111 | 0,6780 | 0,5366 |
| E9_eff | MLP (256 ocultas) | tf_efficientnet_b7 | — | — | — | 656.901 | 0,6693 | **0,6148** | 0,6233 | 0,5849 | 0,5210 | 0,6538 | 0,7763 | 0,5378 |

\* E5b hereda la augmentación de E2 (la config de datos no cambió; el `aug` del JSON quedó en `false` porque describe el bucle de FT).
† Params entrenables durante el FT (30 % del backbone + cabeza).

### Ranking por F1-test
```
1. E4  B7+aug ........ 0,6482   ← mejor global
2. E3  B7 ............. 0,6267
3. E5b MV2 FT ......... 0,6189
4. E9e MLP-B7 emb ..... 0,6148
5. E1  MV2 ............ 0,5945
6. E2  MV2+aug ........ 0,5761   ← peor de las redes
── clásicos ──
E7e 0,5837 · E8e 0,5669 · E9m 0,5629 · E7m 0,5623 · E8m 0,5295 · E6e 0,3342 · E6m 0,2942
```

### Efectos cuantificados
| Efecto | Comparación | Δ F1-test |
|---|---|---|
| Backbone (sin aug) | E3 − E1 | **+0,0322** |
| Backbone (con aug) | E4 − E2 | **+0,0721** |
| Aug en MV2 | E2 − E1 | **−0,0184** |
| Aug en B7 | E4 − E3 | **+0,0215** |
| Fine-tuning MV2 | E5b − E5a | **+0,0244** (val: **+0,0723**) |
| Mejor red vs mejor clásico | E4 − E9_eff | **+0,0334** |
| Sesgo de fuente (antes del FT) | 2019 vs 2020 | 0,6011 vs 0,5823 |
| Sesgo de fuente (después del FT) | 2019 vs 2020 | 0,5541 vs 0,6294 |

---

## 3. Historiales por época (resultados crudos para análisis)

Early stopping: sobre **macro-F1 de val** con tolerancia 1e-4 y patience 4 (E1-E4) / 3 (E5). El checkpoint guardado = mejor época.

### E1 — MV2, 224, sin aug · 16 ép · total 1077,1 s · mejor ép 12 (val 0,6152) → paró en 16 (12+4)
| ép | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | 1,3126 | 1,0733 | 0,9890 | 0,9366 | 0,9069 | 0,8654 | 0,8531 | 0,8147 | 0,8086 | 0,7868 | 0,7897 | 0,7785 | 0,7580 | 0,7400 | 0,7397 | 0,7205 |
| val-F1 | 0,5299 | 0,5709 | 0,5697 | 0,5889 | 0,5852 | 0,6021 | 0,5412 | 0,6039 | 0,5721 | 0,6002 | 0,6010 | **0,6152** | 0,5934 | 0,5890 | 0,5892 | 0,5966 |
| s/ép | 195,7* | 53,1 | 50,2 | 52,7 | 50,2 | 50,9 | 53,0 | 69,2 | 51,7 | 63,0 | 57,8 | 77,6 | 60,5 | 62,0 | 56,7 | 50,7 |

### E2 — MV2, 224, con aug · 16 ép · total 931,8 s · mejor ép 12 (val 0,5854) → paró en 16
| ép | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | 1,3609 | 1,1718 | 1,1091 | 1,0648 | 1,0518 | 0,9986 | 1,0132 | 0,9904 | 0,9720 | 0,9655 | 0,9477 | 0,9580 | 0,9612 | 0,9354 | 0,9469 | 0,9285 |
| val-F1 | 0,4763 | 0,5031 | 0,5326 | 0,5404 | 0,5123 | 0,5244 | 0,4799 | 0,5654 | 0,5123 | 0,5539 | 0,5638 | **0,5854** | 0,5503 | 0,5572 | 0,5468 | 0,5522 |
| s/ép | 52,2 | 51,7 | 62,2 | 60,9 | 57,8 | 63,1 | 57,0 | 53,4 | 56,2 | 57,8 | 58,4 | 62,2 | 57,5 | 58,6 | 53,2 | 52,9 |

### E3 — B7, 600, sin aug · 7 ép · total 4480,2 s · mejor ép 3 (val 0,6551) → paró en 7 (3+4)
| ép | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| loss | 1,1082 | 0,9015 | 0,8274 | 0,7998 | 0,7719 | 0,7261 | 0,7080 |
| val-F1 | 0,5959 | 0,6287 | **0,6551** | 0,6462 | 0,6198 | 0,5987 | 0,6106 |
| s/ép | 651,5 | 622,5 | 625,6 | 621,1 | 617,7 | 626,6 | 623,6 |

### E4 — B7, 600, con aug · 12 ép (tope) · total 7892,8 s · mejor ép 8 (val 0,6455) → paró en 12 (8+4)
| ép | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | 1,1167 | 0,9284 | 0,8698 | 0,8468 | 0,8320 | 0,7976 | 0,7882 | 0,7657 | 0,7469 | 0,7618 | 0,7480 | 0,7579 |
| val-F1 | 0,5799 | 0,6032 | 0,6416 | 0,6449 | 0,6209 | 0,6285 | 0,5804 | **0,6455** | 0,6421 | 0,5393 | 0,6073 | 0,6127 |
| s/ép | 639,2 | 689,5 | 659,9 | 623,5 | 624,8 | 629,3 | 647,6 | 663,0 | 703,3 | 670,0 | 620,2 | 631,0 |

### E5 — MV2 fine-tuning (30 % backbone, lr 1e-4) · 6 ép (tope) · total 283,3 s · mejor ép 6 (val 0,6875)
| ép | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| loss | 0,8223 | 0,6658 | 0,5484 | 0,4109 | 0,3078 | 0,2189 |
| val-F1 | 0,5911 | 0,5863 | 0,6322 | 0,6213 | 0,6610 | **0,6875** |
| s/ép | 50,3 | 44,8 | 44,6 | 44,3 | 43,9 | 44,0 |

> **Oportunidad detectada:** E5 llegó al **tope de 6 épocas con el val-F1 aún subiendo** (mejor = última época, loss cayendo sin estancarse) → subir el tope (p. ej. 10-12 con patience 3) probablemente mejore todavía más.

### Clásicos E6-E9 — hiperparámetros ganadores y tiempos
| ID | Método (config ganadora) | Embeddings | F1-val | F1-test | Tiempo |
|---|---|---|---|---|---|
| E6_mv2 | RandomForest (n=500) | MV2 (1280 d) | 0,3537 | 0,2942 | 7,1 s |
| E6_eff | RandomForest (n=500) | B7 (2560 d) | 0,4434 | 0,3342 | 14,3 s |
| E7_mv2 | SVM RBF, grid → **C=1, gamma=scale** | MV2 | 0,6204 | 0,5623 | 49,1 s |
| E7_eff | SVM RBF, grid → **C=10, gamma=scale** | B7 | 0,6370 | 0,5837 | 93,6 s |
| E8_mv2 | XGBoost (n=300, d=6, lr=0.1) | MV2 | 0,5892 | 0,5295 | 123,0 s |
| E8_eff | XGBoost (n=300, d=6, lr=0.1) | B7 | 0,6362 | 0,5669 | 219,3 s |
| E9_mv2 | MLP torch (256 ocultas, CE con pesos por clase) | MV2 | 0,6502 | 0,5629 | 2,5 s |
| E9_eff | MLP torch (256 ocultas, CE con pesos por clase) | B7 | 0,6693 | 0,6148 | 2,8 s |

*(El grid del SVM se seleccionó por F1-val sobre las combinaciones C∈{1,10} × gamma∈{scale,1e-3}.)*

---

## 4. TIEMPOS DE EJECUCIÓN

### 4.1 Cronograma completo de la corrida (wall-clock, 30/09/2026)

| Hora inicio | Hora fin | Fase | Duración |
|---|---|---|---|
| 01:00:04 | 01:02:07 | Arranque: imports, splits + manifest, carga MV2 | **2 min 03 s** |
| 01:02:07 | 01:20:04 | **E1** MV2 sin aug (16 ép) | **17 min 57 s** (1077,1 s) |
| 01:20:05 | 01:35:36 | **E2** MV2 con aug (16 ép) | **15 min 32 s** (931,8 s) |
| 01:35:47 | 02:50:22 | **E3** B7 sin aug (7 ép) | **74 min 35 s** (4480,2 s) |
| 02:50:30 | 05:02:01 | **E4** B7 con aug (12 ép) | **131 min 31 s** (7892,8 s) |
| 05:02:05 | 05:06:48 | **E5** fine-tuning MV2 (6 ép) | **4 min 43 s** (283,3 s) |
| 05:06:48 | 05:13:27 | Embeddings GAP → 6 `.npz` | **6 min 39 s** (MV2 46 s · B7 5 min 53 s) |
| 05:13:27 | 05:21:59 | Clásicos E6-E9 (8 corridas) | **8 min 32 s** (511,7 s) |
| 05:21:59 | 05:23:08 | LIME (10 figuras = 20 explicaciones) | **1 min 09 s** |
| 05:23:08 | 05:23:10 | Reporte: CSV + §6 en `00_resumen…md` + conclusiones | **~2 s** |
| **01:00:04** | **05:23:10** | **TOTAL** | **4 h 23 min 06 s** (≈ 15.786 s) |

### 4.2 Distribución del tiempo por experimento

| Exp | Épocas | s/época | **Total s** | **Total (min:ss)** | % de la corrida |
|---|---|---|---|---|---|
| E1 | 16 | 65,9 (≈54 en régimen; 1ª ép = 195,7 s con setup) | 1.077,1 | 17:57 | 6,8 % |
| E2 | 16 | 57,2 | 931,8 | 15:32 | 5,9 % |
| E3 | 7 | 626,9 | 4.480,2 | 74:40 | 28,4 % |
| E4 | 12 | 650,1 | 7.892,8 | 131:33 | **49,9 %** |
| E5 | 6 | 45,3 | 283,3 | 4:43 | 1,8 % |
| **E1-E5** | 57 | — | **14.665,2** | 244:25 | **92,9 %** |
| Embeddings | — | — | ≈399 | 6:39 | 2,5 % |
| E6-E9 (8) | — | — | 511,7 | 8:32 | 3,2 % |
| LIME | — | — | ≈69 | 1:09 | 0,4 % |
| Setup + reporte | — | — | ≈126 | 2:06 | 0,8 % |

### 4.3 Costo por modelo (lo que hay que considerar para retomar/escalar)

| Métrica | MobileNetV2 @224 | EfficientNet-B7 @600 | Relación |
|---|---|---|---|
| Params totales | 2.230.277 | 63.799.765 | **×28,6** |
| s/época (batch 32 / 4) | ≈ 51-66 | ≈ 617-703 | **×~11** |
| Costo de 1 época | ~1 min | **~10,5 min** | — |
| Costo de 10 épocas | ~10 min | **~1 h 45 min** | — |
| VRAM pico | ~1,5 GB | ~4 GB (6 GB disponibles) | — |
| Fine-tuning (30 %, lr 1e-4) | 45 s/época | no intentado (VRAM insuficiente probable) | — |

### 4.4 Ejecución SMOKE (referencia, noche del 29/09 — no altera resultados reales)

| Exp | Tiempo | | Exp | Tiempo |
|---|---|---|---|---|
| E1_smoke | 7,9 s | | E6-E9_smoke | 0,0-3,0 s c/u |
| E2_smoke | 1,4 s | | Embeddings smoke | ~15 s |
| E3_smoke | 31,1 s | | LIME smoke (5 figs) | ~12 s |
| E4_smoke | 11,3 s | | **Total smoke** | **≈ 6 min** |
| E5_smoke | 12,4 s | | | |

---

## 5. LIME — explicabilidad (ítem 6)

- **Salida:** `reportes\lime_antes_despues\` → **10 PNG** reales (2 por clase: cbb, cbsd, cgm, cmd, healthy), generados 05:22:15–05:23:08.
- **Contenido de cada PNG** (verificado): 4 paneles — `A_antes → <pred>` (overlay LIME + mapa de regiones) arriba, `B_después → <pred>` abajo; título con `image_id | real = <clase> | fuente`. **20 explicaciones en total** (10 imágenes × 2 checkpoints: A = mejor MV2 antes del FT, B = MV2 con fine-tuning).
- **Conclusión del notebook:** las regiones que sostienen la predicción se concentran en el limbo/síntomas del folíolo (manchas, clorosis), **no** en marco, fondo ni artefactos de captura → sin evidencia de sesgo espurio (coherente con el análisis de brillo/contraste por clase del EDA).
- ⚠️ **4 archivos residuales del SMOKE** (12:00 AM del 30/09) conviven en la misma carpeta y NO son de la corrida real: `lime_01_cbsd.png`, `lime_02_cgm.png`, `lime_03_cmd.png`, `lime_04_healthy.png` (timestamp 00:00). **Recomendación: borrarlos** para no confundirlos con los 10 reales (05:22).

---

## 6. Artefactos generados (dónde vive cada cosa)

| Artefacto | Ruta | Detalle |
|---|---|---|
| Checkpoints + métricas | `modelos\E1…E5.pth` + `_res.json` | Los JSON tienen `val`, `test` (F1 macro/per-clase, BalAcc, y_true/y_pred), `history` por época (loss, val_f1, s/ép) y `tiempo_total_s`/`tiempo_ep_s` |
| Embeddings | `modelos\emb_{mobilenetv2_100\|tf_efficientnet_b7.ra_in1k}_{train\|val\|test}.npz` | GAP 1280/2560 dims; train 9,8 MB (MV2) / 31,8 MB (B7) |
| Clásicos | `modelos\E6-E9_{mobilenetv2\|efficientnet}_res.json` | incluye `extra` con la config ganadora |
| Tabla maestra | `reportes\02_resultados_experimentos.csv` | 15 filas (E1-E4, E5a/E5b, E6-E9 ×2 backbones) |
| §6 en el resumen | `reportes\00_resumen_consigna_y_avance.md` | insertada entre `<!-- INICIO-RESULTADOS-TL -->` y `<!-- FIN-RESULTADOS-TL -->` (05:23:10) |
| Figuras LIME | `reportes\lime_antes_despues\*.png` | 10 reales + 4 residuales de smoke (§5) |
| Splits | `datos\splits\manifest_20pct.csv` | train 3.366 / val 842 / test 300 / test_pool_rest 752 / out 21.044 |
| Manifests smoke (no tocar) | `modelos\*_smoke.*` | vestigios de la validación; separados por sufijo |

---

## 7. Observaciones y pendientes

1. ⚠️ **El `02_transfer_learning_mandioca.ipynb` en disco NO tiene guardados los outputs de esta corrida** (último guardado 00:31, con el error viejo `No module named 'torch'` de la sesión de kernel equivocado). Los resultados viven en los JSON/CSV/§6, no en el notebook. Si el entregable pide "`.ipynb` ejecutado":
   - Opción A (barata, ~5-10 min): re-lanzar el `nbconvert --inplace` de §0.2 — la caché **salta todos los entrenamientos** (E1-E9 existen) y solo re-genera LIME, CSV y §6.
   - Opción B: abrir en VS Code con el kernel `ic522-mandioca`, Run All y **guardar** (Ctrl+S).
2. **E5 no llegó a su techo**: cortó en el tope de 6 épocas con val-F1 subiendo (0,6875) → considerar 10-12 épocas para el ítem 3.
3. **Augmentación perjudicó a MV2** (sobre todo cbsd: 0,574 → 0,432): si se refina, probar solo flip+rotación sin el traslado del RandomCrop, o reducir el rango de rotación.
4. **RandomForest es muy malo sobre estos embeddings** (cgm con F1 = 0,0 en MV2): los embeddings GAP linealmente "muestran" la clase pero RF con n=500 no las aprovecha; documentar como hallazgo, no como error de pipeline.
5. **Pendientes de consigna**: ítem 7 (búsqueda de hiperparámetros de la red) e ítem 8 (robustez con peores fotos).
6. Brecha val→test en E5b (0,6875 vs 0,6189): con val de 842 y test de 300 hay ruido ±3-4 pts; las comparaciones finas (Δ < 0,02) tomalas con pinza.
