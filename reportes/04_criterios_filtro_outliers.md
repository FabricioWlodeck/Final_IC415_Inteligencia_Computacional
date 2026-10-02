# Criterios del filtro Hoja vs "no-hojas" — primer filtro y ampliaciones de umbral

**Alumno:** WLODECK, Fabricio Joaquín · **Proyecto:** IC522 — Trabajo Final Integrador

*Fuentes: `01_EDA_mandioca.ipynb` (Fase 5.6 — Experimento "hojas vs no-hojas"), `volumen_outliers_prueba.ipynb`
(ambos ejecutados, 0 errores) y `reportes/00_resumen_consigna_y_avance.md` (sección 4).
Todos los conteos se verificaron contra las salidas ejecutadas de los notebooks.
Dataset de referencia: `datos/dataset original/` = **26,337 imágenes, 100 % intacto**.*

---

## 1. Primer filtro — criterios explícitos (Experimento 5.6, EDA 01)

Este es el filtro que se aplicó y quedó documentado: **33 imágenes no-hoja = 0,13 % del dataset**.

### 1.1 Tabla de criterios

| # | Criterio | Definición exacta | Umbral | Rol en la regla |
|---|---|---|---|---|
| 1 | `pct_verde` | fracción de píxeles de la máscara HSV "hoja": **H 30–110°, S ≥ 50, V ≥ 60** | `< 0.15` | Etapa 1 (descarte grueso) |
| 2 | `verde_suave` | fracción de píxeles con **`G − max(R,B) > 4` y `G > 20`**, medido a 160×120 (greenness cromático, no depende del brillo) | `< 0.035` | Etapa 2 (filtro fino) |
| 3 | `celdas_verdes` | nº de celdas 20×20 (de un total de **48**) con **> 20 %** de píxeles verdes (estructura de hoja) | `< 3` | Etapa 2 (filtro fino) |
| 4 | **Compuerta** | las **tres condiciones a la vez** (1 **Y** 2 **Y** 3) | — | si se cumple → imagen = **no-hoja** |
| 5 | Lista manual | raíces/palos con pasto de fondo que el verde cromático "rescataba": `1004389140`, `2139839273`, `4203623611`, `746746526`, `2698282165`, `1905119159`, `274726002`, `3058561440` | **+ 8** | se suman a la regla automática |
| — | **Resultado final** | 25 automáticas + 8 manuales | **33** | **0,13 %** de 26,337 · validadas **33/33** |

Flujo de la etapa 1 a la 2 (salida ejecutada del EDA): **1,630** candidatas con
`pct_verde < 0.15` → **25** pasan la heurística `verde_suave` + `celdas_verdes` → **+ 8 manuales
= 33**.

### 1.2 Consideraciones textuales

* **Rigidez deliberada de la compuerta AND.** Las tres condiciones se exigen a la vez para no
  descartar hojas: basta con que *algo* de verde rellene una métrica para que la imagen se
  "rescate". El costo ya estaba documentado: **cualquier poco de pasto o follaje de fondo**
  (pct_verde, celdas verdes) hace que una raíz o un palo evada el filtro.
* **La máscara HSV llega hasta H = 110°** (no 90°) para **no** descartar amarillamientos
  patológicos: el amarillo es síntoma, no ruido.
* **`verde_suave` reemplazó a `verde_cromo`** en la calibración: el cromático estricto
  (`G > R+8 y G > B+8`) rescataba hojas desaturadas; el "suave" (`G − max(R,B) > 4`) es más
  tolerante al brillo/sombra manteniendo la idea de greenness.
* **`celdas_verdes` aporta estructura**, no solo color: una hoja ocupa celdas completas; la
  tierra o una raíz dispersa no.
* **Qué NO puede separar el color** (límite explícito del EDA 01): tierra vs. lesión marrón de
  la hoja, y hoja vs. pasto con luz intensa. Por eso una "raíz sobre césped muy verde"
  (`pct_verde ≥ 0.15`) **quedaba fuera** de la regla automática → se agregó a la lista manual.
* **Las hojas oscuras/desenfocadas se rescatan, no se descartan** (las ramas A y C castigan
  la falta de verde/saturación, y una hoja sana oscura puede tener poco verde medible; por eso
  la calibración descartó los intentos que exigían `pct_verde` demasiado bajo o `plano/neutro`
  altos). Con la red ampliada este riesgo de falsos positivos **aumenta** (ver §2.3).
* **Método anti-divergencia aplicado:** primero se miran los montajes, después se fija el
  umbral; todo se hace **sobre copias** (el dataset original nunca se modifica).
* **Calibración previa (bitácora EDA 01), de la que salió el criterio final:**

| Intento | Regla probada | n | Problema detectado al mirar |
|---|---|---|---|
| preliminar | `score_hoja < 0.35` | 9,222 (35 %) | falsos masivos: hojas oscuras sobre tierra → descartado |
| intento 1 | `pct_verde < 0.08 y (plano > 0.30 ó neutro > 0.25)` | 25 | 19 falsos positivos (hojas oscuras) |
| intento 2 | + `verde_cromo < 0.05` | 44 | 11 falsos (hojas desaturadas con pasto al fondo) |
| intento 3 | `pct_verde < 0.15 y verde_cromo < 0.015` | 24 | rescataba bien; se cambió a `verde_suave` + celdas |
| **automática final** | **`pct_verde < 0.15 y verde_suave < 0.035 y celdas_verdes < 3`** | **25** | **25/25: tubérculos, raíces, palos** |
| **+ manual** | **lista de 8 raíces/palos con pasto de fondo** | **33** | **validación final 33/33** |

* **Salidas del primer filtro** (`datos/cambios 01 - no hojas/`): `outliers/` (33),
  `imagenes filtradas/` (26,304), `labels_filtrado.csv`, `labels_cambios.csv` (métricas de las
  26,337) y `experimento.md`.

---

## 2. Intentos siguientes — qué se cambió y cómo creció el conteo

Notebook: `volumen_outliers_prueba.ipynb` (prueba exploratoria, **solo cómputo y copias**;
el dataset original y el `01_EDA_mandioca.ipynb` no se tocaron). Filosofía idéntica
(mismas métricas), umbrales relajados y — sobre todo — **compuertas OR agregadas**.

### 2.1 Tabla de ampliaciones

| Intento | Qué se relajó / qué se agregó respecto al anterior | Criterio resultante | n (regla) | Unión con las 33 | % del dataset | Δ absoluto vs base | Δ proporción | Múltiplo |
|---|---|---|---|---|---|---|---|---|
| **base (EDA 01)** | — (referencia) | `pct_verde<0.15` **Y** `verde_suave<0.035` **Y** `celdas_verdes<3` + 8 manuales | 33 | 33 | 0,13 % | — | — | 1,0 × |
| **red 1 · ampliada** | Rama A: `pct_verde` 0,15 → **0,35**; `verde_suave` 0,035 → **0,08**; `celdas_verdes` 3 → **8** · **Nueva rama B (OR):** `pct_suelo > 0,40` · **Nueva rama C (OR):** `pct_neutro > 0,45` | (A) **OR** (B) **OR** (C) | 117 | **119** | **0,45 %** | **+ 86** | **+ 0,32 pp** | **3,6 ×** |
| **red 2 · muy ampliada** | Rama A: verde 0,35 → **0,45**; suave 0,08 → **0,12**; celdas 8 → **12** · Rama B: suelo 0,40 → **0,30** · Rama C: neutro 0,45 → **0,35** | (A) **OR** (B) **OR** (C) | 298 | **298** | **1,13 %** | **+ 265** | **+ 1,00 pp** | **9,0 ×** |

"Unión con las 33" = la marca de la red **más** las 33 ya confirmadas (ninguna de las conocidas
se pierde al cambiar de umbral: la base queda contenida en la red 2; 2 de las 33 caen fuera de
la red 1 y se arrastran igual). `pp` = puntos porcentuales sobre el dataset total.

### 2.2 Crecimiento por rama (por qué subió el número)

| Rama | base (EDA 01) | red 1 | red 2 |
|---|---|---|---|
| **A — verde relajado** (AND de las 3 métricas de color/estructura) | 25 | **87** | **149** |
| **B — `pct_suelo`** (color tierra/cáscara/madera, **OR**, nueva) | — | **32** | **169** |
| **C — `pct_neutro`** (blanco/gris/sombra, **OR**, nueva) | — | **1** | **4** |
| **Unión de la red** | 25 (+8 manuales = 33) | **117** (119 con base) | **298** |

### 2.3 Descripción de los cambios

**Red 1 (el primer intento de ampliación).** Dos cambios estratégicos, tal como sugirió el
análisis previo:

1. **Relajar el verde (rama A):** la regla original exigía las tres condiciones a la vez
   (`AND` rígido); al subir `pct_verde` a 0,35, `verde_suave` a 0,08 y `celdas_verdes` a 8 se
   tolera una **franja de pasto o follaje de fondo** sin llegar a la estructura de una planta
   completa. Captura el subgrupo "raíz/palo con verde de fondo".
2. **Incorporar `pct_suelo` y `pct_neutro` con compuerta OR (ramas B y C):** son métricas que
   el EDA ya calculaba pero que la regla final **no** usaba. Si la imagen es abrumadoramente
   **marrón** (tierra, cáscara de raíz, madera, pulpa) o **blanca/gris** (interior de la
   mandioca, cuadernos, sombras), no es una hoja de interés **sin importar cuánto pasto haya
   alrededor** — que es exactamente por lo que las 8 manuales se escapaban de la AND.

**Red 2 (cota superior).** Todos los umbrales se corren en la misma dirección (verde y
suelo/neutro más permissivos), manteniendo la misma estructura A **OR** B **OR** C. Sirve para
ver **hasta dónde** llega el conteo si se afloja más, no como filtro final.

**Cómo creció el volumen (absoluto y proporción):**

* **33 → 119 → 298** imágenes marcadas (0,13 % → 0,45 % → 1,13 % del dataset): la red 1
  multiplica el conteo por **3,6×** (+86 imágenes) y la red 2 por **9,0×** (+265 vs base).
* En proporción del dataset disponible: se pasaría de descartar **0,13 %** (quedan 26,304) a
  descartar **0,45 %** (quedan 26,218) con la red 1, o **1,13 %** (quedan 26,039) con la red 2.
* El salto más grande entre red 1 y red 2 lo produce la **rama B (`pct_suelo`)**: pasa de 32 a
  169 imágenes (+137) al bajar el umbral de 0,40 a 0,30; la rama A aporta +62 y la rama C +3.
* **Reparto en carpetas** (partición disjunta, cada imagen en una sola carpeta):
  `01_base_confirmadas/` = **33** · `02_red_ampliada_nuevas/` = **86** ·
  `03_red_muy_ampliada_nuevas/` = **179** · total = **298 copias** en
  `datos/cambios 02 - prueba volumen outliers/outliers_prueba/` (+ `montajes/` con 9 grillas
  JPEG, `labels_outliers_prueba.csv` con las 26,337 métricas + flags, `outliers_prueba.csv`
  con las 298 y `resumen_outliers_prueba.md`).

**Distribución por clase de la red 2 (señal de falsos positivos):**

| Clase | Marcadas (unión red 2) | % de la clase |
|---|---|---|
| CBSD (rayas marrones) | 103 de 3,476 | **2,96 %** |
| CMD (mosaico) | 142 de 15,462 | 0,92 % |
| CGM (moteado verde) | 26 de 3,017 | 0,86 % |
| Saludable | 23 de 2,890 | 0,80 % |
| CBB (bacteriosis) | 4 de 1,492 | 0,27 % |

CBSD se marca **~3× más** que el resto: es la clase con lesiones marrones, y la rama B
(`pct_suelo`, rango H 5–35° = color tierra/madera) **no distingue tierra de lesión patológica**
— el límite de color que el EDA 01 ya había documentado.

**Riesgos de falsos positivos a la hora de revisar:**

* Rama B: hojas con lesiones marrones (CBSD/CBB) o tierra debajo de la hoja.
* Rama C: sombras profundas o fondo blanco (aunque su volumen es mínimo: 1–4 imágenes).
* Rama A relajada: hojas oscuras/desenfocadas y **fotos de planta entera donde domina el
  suelo** (caso gris: ¿son "no-hoja" o solo fotos poco útiles?).

**Qué NO cambió en esta ampliación:** el dataset original sigue con 26,337 imágenes;
`01_EDA_mandioca.ipynb` no se modificó; no se descarta ni se elimina nada — solo se **cuenta y
se copia** a la carpeta aparte. Las métricas se reutilizaron de `labels_cambios.csv`
(26,337 filas, sin recalcular).

**Pendiente / próximo paso:** revisar `02_red_ampliada_nuevas/` (86) y, si se quiere afinar,
`03_red_muy_ampliada_nuevas/` (179), contar cuántas son realmente no-hoja y recién ahí fijar el
umbral definitivo — la regla rectora del EDA 01 se mantiene: **primero se mira, después se
fija el umbral**.
