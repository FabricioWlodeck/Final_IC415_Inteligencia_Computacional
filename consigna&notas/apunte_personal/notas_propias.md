# Qué buscamos hacer?
Se busca desarrolllar un sistema de clasificaciones de patógenos que afectan a la madioca como cultivo. El dataset consiste en imagenes las cuales se clasifican en 5 segun el patogeno entre los cuales estan estos patógenos:
- Cassava Bacterial Blight (CBB)
- Cassava Brown Streak Disease (CBSD) 
- Cassava Green Mottle (CGM) 
- Cassava Mosaic Disease (CMD) 
- Saludable 

## Consideraciones importantes del dominio:
- Algunas enfermedades presentan síntomas visuales similares en etapas tempranas 
- La calidad de las imágenes puede variar significativamente debido al uso de dispositivos 
no profesionales 
-  Las condiciones de iluminación natural en campo pueden introducir variabilidad en el 
aspecto de los síntomas 
- La confusión entre ciertas clases puede tener implicaciones económicas diferentes (ej: 
identificar erróneamente una planta saludable como enferma vs. no detectar una planta 
enferma)

## Como se ven estos diferentes patogenos?
Algo importante a tener en cuenta al realizar todo este analisis es saber o almenos conocer a grandes rasgos como se caracterizan esta patologías. Se describen a continuación algunas de dichas caracteristicas:

### Cassava Bacterial Blight (CBB)
- Se ven manchas marrones con bordes rectos o angulares, como si estuvieran encerradas por las venitas de la hoja. Suelen tener un borde amarillo alrededor.
- Si avanza, las manchas se juntan y secan gran parte de la hoja.
- A veces se puede ver una gotita pegajosa brillante (como resina) sobre las venas.

![Ejemplo de hoja de mandioca con presencia de CBB](ejemplo%20CBB.png)


### Cassava Brown Streak Disease (CBSD)
- Se notan manchas amarillas que siguen el recorrido de las venas de la hoja.
- El detalle clave para diferenciarla es que la hoja no se deforma. Mantiene su forma plana y tamaño normal, **solo cambia de color**.

![Ejemplo de hoja de mandioca con presencia de CBSD](ejemplo%20CBSD.png)

### Cassava Green Mottle (CGM)
- La hoja parece "salpicada" o moteada con puntitos amarillos mezclados con el verde normal.\
- Afecta más a las hojas nuevas, que pueden verse un poco fruncidas, arrugadas o con los bordes levemente doblados, pero sin perder totalmente su forma.

![Ejemplo de hoja de mandioca con presencia de CGM](ejemplo%20CGM.png)

### Cassava Mosaic Disease (CMD)
- La hoja toma un patrón de "camuflaje" o mosaico con manchas irregulares verde oscuro, verde claro y blanco/amarillo pálido.
- Su característica principal es la deformación severa: las hojas se achican, se retuercen, crecen chuecas y quedan muy arrugadas (como atrofiadas).

![Ejemplo de hoja de mandioca con presencia de CMD](ejemplo%20CMD.png)



# Conclusiones en cuanto al Dataset:
Las imagenes presentes en el dataser son, en su inmensa mayoría, imagenes centradas en las hojas de la mandioca, dejando en evidencia que estas mismas serán el principal foco de analisis para determinar la presencia de determinada patología. 


# Consideraciones y cambios
-  que significa lo siguiente? :
Conclusión práctica (Fase 5.1–5.2): la mayoría son 800×600 → para la red conviene redimensionar a 224×224. Como solo 3.4 % son casi cuadradas, un resize directo deforma la hoja; mejor Resize(256) con proporción + CenterCrop(224) (o letterbox con pad).

# Refencias y Comparaciones:
1) En el video de yt (https://www.youtube.com/watch?v=dpA1ypY-zVg) mencionan que realizaron un modelo para una competencia, la cual tuvo vigencia en 2019, se obtuvieron los siguientes resultados en su desarrollo:
* **Modelos desde cero:** Obtenían precisiones entre el **58% y 72%** en el entrenamiento, pero con dificultades para generalizar.
* **MobileNet V2:** Alcanzó entre un **64% y 66%** de precisión tras corregir los errores de preprocesamiento.
* **EfficientNet (B4 / B7):** Ofreció los mejores resultados, logrando entre un **80% y 83%** de precisión en el conjunto de validación y en evaluaciones tipo competencia

2) El ganador de dicha competencia tuvo un **score de 0.93860 o 93.860%** unos capos (todos chinos, impresionante). El ganador menciona algunas cosas sobre su modelo:
- model: 5 fold se_resnext101
- data augumentaion: RandomCrop, VFlip, HFilp, RandomRotate
- 3 * TTA

3) El segundo puesto menciona:
- I used 6fold seresnext 101
- lr: 2e-4
- batch size: 16
- input size: 448x448x3
- epochs: 5
- augmentations: standard
- tta: 8x

4) el tercer puesto menciona:
- First I trained 5-folds se_resnext50_32x4d using train images. All the models were trained on kaggle kernels for a few hours.
- lr: 1e-4
- batch size: 20
- input size: 500x500x3
- augmentations: random crop, random erasing, hflip, vflip, random affine,
- tta: 10 times


=======================================================================

# Seguimiento de Consignas:
- [ ] punto 1 