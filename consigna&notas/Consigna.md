# Items a cumplir:
- [x] 1 - Desarrollar un pipeline completo de clasificación que integre extracción de características mediante transfer learning y modelos de clasificación, optimizado para operar con imágenes capturadas en condiciones no controladas.
      Eligir entre 2 o mas modelos preentrenados (MovileNet v2 y v3 y eficentNet), "cortarles" la cabeza de clasificación y ponerle cabeza propia para que clasifique las imágenes  en las categorías dadas
      
- [x] 2 - Evaluar y seleccionar arquitecturas de redes neuronales pre-entrenadas como extractores de características (backbones), considerando el balance entre capacidad representacional, costo computacional, y requerimientos de recursos para su implementación en contextos con limitaciones tecnológicas.
      Básicamente elegir entre los modelos analizados el que mejor se adapte a la situación de baja capacidad computacional, ya que se tratan de dispositivos móviles. El mas adaptado podría ser el que tiene menor numero de características o el que tiene menor tiempo de inferencia. (MovileNet v2)
      
- [x] 3 - Diseñar e implementar estrategias de transfer learning apropiadas para el dominio agrícola, explorando técnicas de fine-tuning, feature extraction, y entrenamiento de capas de clasificación personalizadas.
      No entiendo bien lo de feature extraction pero el fine tuning esta medio hecho, habria que hacerlo mas robusto.
      
- [ ] 4 - Explorar modelos de clasificación sobre embeddings, incluyendo tanto arquitecturas end-to-end (redes neuronales completas) como clasificadores clásicos de machine learning (Random Forest, SVM, XGBoost, etc.) operando sobre características extraídas por el backbone. 
	      AUN QUEDA POR IMPLEMENTAR
      
      
- [x] 5 - Definir y calcular métricas de evaluación apropiadas para el contexto del problema, considerando la importancia diferencial de los errores de clasificación y el impacto de las confusiones entre clases específicas en la toma de decisiones productivas.
	    *Básicamente considerar falsos negativos como los mas importantes*  
	
- [ ] 6 - Implementar técnicas de explicabilidad para interpretar las decisiones del modelo, verificar que no opera bajo sesgos espurios, y generar confianza en los usuarios finales del sistema. Existen varios métodos para esto, pero se recomienda utilizar LIME (ya implementado y funcional)
	    NO SE COMO IMPLEMENTAR AUN  
      
- [ ] 7 - Optimizar hiperparámetros del pipeline mediante técnicas sistemáticas (Grid Search, Random Search, optimización bayesiana, algoritmos genéticos), documentando el proceso y justificando las configuraciones finales seleccionadas.
	    NO se como hacer esto aun. 
      
- [ ] 8 - Evaluar la robustez del modelo frente a variaciones en las condiciones de captura y analizar su comportamiento ante casos límite o ambiguos.
	      Pasarle las peores fotos posibles