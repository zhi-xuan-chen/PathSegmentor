

<!-- # PathSegmentor -->
[![header](https://capsule-render.vercel.app/api?type=waving&height=180&color=gradient&text=PathSegmentor&reversal=false&desc=Segment%20Anything%20in%20Pathology%20Images%20with%20Natural%20Language&fontSize=40&fontAlign=19&fontAlignY=22&descSize=28&descAlignY=47&descAlign=47)](https://arxiv.org/pdf/2506.20988)

<!-- [![Arxiv Page](https://img.shields.io/badge/Arxiv-2407.18449-red?style=flat-square)](https://arxiv.org/abs/2407.18449) -->
<!-- [![GitHub Stars](https://img.shields.io/github/stars/zhi-xuan-chen/PathSegmentor?style=social)](https://github.com/zhi-xuan-chen/PathSegmentor) -->
---
Este repositorio aloja la implementación oficial del artículo de investigación titulado *"Segment Anything in Pathology Images with Natural Language".* Proporciona código, modelos y recursos para apoyar la segmentación semántica guiada por lenguaje natural en imágenes patológicas.

# 🛠 Configuración del Entorno

El entorno probado utiliza Python 3.11, PyTorch 2.1.0, torchvision 0.16.0 y CUDA 12.1 en Linux con una GPU NVIDIA. Crea el entorno con Conda:

```bash
git clone https://github.com/zhi-xuan-chen/PathSegmentor.git
cd PathSegmentor
conda env create -f environment.yml
conda activate pathsegmentor
```

Instala las dependencias del modelo. Se deshabilita el aislamiento de compilación porque la versión personalizada de Detectron2 importada utiliza PyTorch durante la compilación:

```bash
python -m pip install --no-build-isolation -r requirements.txt
```

Descarga los recursos de NLTK utilizados por el codificador de lenguaje:

```bash
python -m nltk.downloader punkt averaged_perceptron_tagger
```

La extensión personalizada de Detectron2 se compila durante la instalación, por lo que se requiere un compilador de C++ y un toolkit de CUDA que proporcione `nvcc`. El controlador de NVIDIA debe ser compatible con CUDA 12.1 o una versión posterior.


# 🗂️ Conjunto de Datos PathSeg

El conjunto de datos PathSeg se construye preprocesando una colección de conjuntos de datos de segmentación de imágenes patológicas de acceso público. Para información detallada sobre los conjuntos de datos de origen, consulta la documentación correspondiente: [PathSeg Dataset](assets/readmes/Datasets_acquisition.md).

También planeamos hacer de código abierto las versiones procesadas de los conjuntos de datos que hemos curado y publicarlos en el siguiente repositorio: [PathSeg on Hugging Face](https://huggingface.co/datasets/Trusure/PathSeg/). Actualmente estamos solicitando permisos de redistribución a los autores/mantenedores originales de cada conjunto de datos. Una vez que se otorgue el permiso para un conjunto de datos, subiremos rápidamente sus imágenes y máscaras procesadas al repositorio [PathSeg](https://huggingface.co/datasets/Trusure/PathSeg/).

Proporcionamos un script de descarga en un solo clic en `pathseg_datasets/download.sh`, que puede recuperar la mayoría de los conjuntos de datos, excepto aquellos que requieren inicio de sesión de usuario o verificación de acceso adicional.

Además, `assets/readmes/Datasets_processing.md` proporciona instrucciones detalladas sobre cómo convertir los datos en bruto al formato y estructura de directorios requeridos.

Para brindar una visión general del conjunto de datos, proporcionamos un subconjunto de muestra en el directorio `pathseg_datasets/Demo`. Puedes utilizar este subconjunto para entrenar y evaluar el modelo, lo que te ayudará a entender cómo funciona el código. Además, puedes crear tu propio conjunto de datos siguiendo la misma estructura organizativa que el dataset `Demo` y realizar el ajuste fino de PathSegmentor en él. Las instrucciones detalladas se pueden encontrar en la sección [Fine-tuning](#fine-tuning-your-pathsegmentor).

# 💾 Checkpoints de PathSegmentor
Puedes descargar los puntos de control preentrenados de PathSegmentor desde este enlace de Google Drive: [PathSegmentor Checkpoints](https://drive.google.com/drive/folders/1HxFdA9KXfS6CKgMX-i6TEVfflHiO5Tmf?usp=sharing).
 
# 🚀 Cómo Utilizar PathSegmentor
El flujo de trabajo de inferencia utilizando nuestro PathSegmentor preentrenado se ha optimizado en el [cuaderno de inferencia](inference_examples.ipynb). Puedes ejecutar directamente la inferencia en las imágenes de ejemplo proporcionadas en el directorio `examples`.

>💡 **Nota**: Recuerda cambiar la ruta del punto de control `pretrained_pth` en el cuaderno de acuerdo con la configuración de tu entorno.

Proporcionamos cuatro tipos de ejemplos en el directorio `examples`: `1_irregular`, `2_tiny`, `3_dense` y `4_others`. Estos directorios pueden contener subcategorías adicionales, pero cada carpeta de nivel final siempre incluye dos archivos: una imagen y una máscara.

Para ejecutar la inferencia en una imagen específica, simplemente reemplaza las variables `image_path` y `mask_path` en el cuaderno con las rutas a tu imagen y archivo de máscara elegidos.

El `text_prompt` utilizado para la segmentación puede derivarse del final del nombre del archivo de máscara. Por ejemplo, el archivo de máscara llamado `Part_2_1211_pathology_bile+duct_nuclei+connective.png` corresponde a la prompt: “nuclei-level connective in bile duct pathology”.

>💡 **Nota:** Elimina cualquier signo `+` al construir la prompt.

Ahora, ejecuta cada celda en el cuaderno. Se generará una figura que contiene la imagen, la máscara predicha y la máscara de ground truth (GT) al final, lo que te permitirá comparar los resultados.

# Ajuste Fino de tu PathSegmentor
Para realizar el ajuste fino (fine-tuning) de PathSegmentor en tu propio conjunto de datos, sigue las instrucciones a continuación para organizar tus datos de acuerdo con el formato requerido y comenzar el proceso.

## 📁 Organización del Conjunto de Datos
Antes de comenzar, crea una carpeta llamada `pathseg_datasets` en un directorio con espacio de almacenamiento suficiente. Esta carpeta se utilizará para almacenar los conjuntos de datos preprocesados requeridos por PathSegmentor.

La estructura de cada conjunto de datos dentro de `pathseg_datasets` está organizada de la siguiente manera:
<pre>
pathseg_datasets  
├── [DATASET_NAME]  
│   ├── train  
│   ├── train_mask  
│   ├── test  
│   └── test_mask
</pre>
Las carpetas `train` y `train_mask` contienen las imágenes y los archivos de máscara correspondientes utilizados para el entrenamiento, mientras que las carpetas `test` y `test_mask` contienen las imágenes y archivos de máscara utilizados para las pruebas.

El directorio `pathseg_datasets/Demo` sirve como un ejemplo claro de cómo estructurar adecuadamente un conjunto de datos llamado `Demo`.

## ⚙️ Preprocesamiento del Conjunto de Datos
### Procesamiento de Imágenes
PathSegmentor requiere parches de imagen de `1024×1024` extraídos de imágenes de cortes enteros (WSI) a una magnificación de `40x`. Por lo tanto, asegúrate de que tus WSIs estén escalados a `40x`, luego recórtalos en parches de `1024×1024`. Guarda las imágenes resultantes en los directorios `train` o `test` según su división designada.

Asegúrate de que los nombres de los archivos de imagen cumplan con el siguiente formato:
```
[IMAGE-NAME]_pathology_[REGION].png
```
Aquí, `[IMAGE-NAME]` se refiere a un identificador único para cada imagen, mientras que `[REGION]` indica la región anatómica a la que pertenece la imagen patológica.

### Procesamiento de Máscaras
Cada imagen puede tener múltiples archivos de máscara binaria del mismo tamaño (1024×1024), con cada máscara representando un objeto distinto dentro de la imagen. En estas máscaras, los píxeles con un valor de 1 indican la región de primer plano.

Los nombres de archivo de las máscaras siguen un formato similar, pero incluyen un campo adicional:
```
[IMAGE-NAME]_pathology_[REGION]_[OBJECT].png
```
La parte inicial del nombre del archivo de máscara coincide con su imagen correspondiente. El segmento `[OBJECT]` indica el objeto específico representado por la máscara dentro de la imagen. En `[OBJECT]`, los espacios se reemplazan con signos `+` para facilitar el análisis. El primer componente de `[OBJECT]` indica la estructura histológica a la que pertenece el objeto. Por ejemplo, `nuclei+neoplastic` se refiere a un objeto `neoplastic` que pertenece a la estructura `nuclei`.

## 📝 Generación del JSON de Información del Conjunto de Datos
Después de preparar los archivos de imagen y máscara preprocesados, puedes generar los archivos JSON de información del conjunto de datos, los cuales se utilizan tanto para el entrenamiento como para las pruebas.

Puedes generar los archivos JSON para un único conjunto de datos ejecutando el siguiente comando:
```bash
cd pathseg_datasets
python create_single_dataset.py
```
>💡 **Nota:** Asegúrate de configurar el parámetro `targetpath` con la ruta de tu conjunto de datos.

Si tienes múltiples conjuntos de datos que requieren la generación de archivos JSON, ejecuta el siguiente comando:
```bash
cd pathseg_datasets
python create_multiple_dataset.py
```
>💡 **Nota:** Del mismo modo, asegúrate de actualizar la lista `dataset_paths` con las rutas de todos tus conjuntos de datos.

Una vez generados los archivos JSON, el directorio de tu conjunto de datos se verá así:
<pre>
pathseg_datasets  
├── [DATASET_NAME]  
│   ├── train  
│   ├── train_mask  
│   ├── train.json  
│   ├── test  
│   ├── test_mask  
│   └── test.json  
</pre>

## 📋 Registrar Conjunto de Datos
Para utilizar estos conjuntos de datos, debes registrarlos apropiadamente. Abre el archivo [register_pathseg_datasets.py](datasets/registration/register_pathseg_datasets.py) y añade tu `[DATASET_NAME]` a la lista `datasets`. El conjunto de datos de entrenamiento se registrará como `pathseg_[DATASET_NAME]_train`, y el conjunto de datos de prueba como `pathseg_[DATASET_NAME]_test`.

Finalmente, añade estos conjuntos de datos registrados al archivo de configuración [pathsegmentor.yaml](configs/pathsegmentor.yaml), alrededor de la línea 233.

Una vez completado, tus conjuntos de datos estarán listos para el entrenamiento y la evaluación.

## 💪 Entrenamiento
Puedes comenzar el entrenamiento ejecutando el siguiente comando:
```bash
bash scripts/train.sh
```
>💡 **Nota:** Puedes modificar las variables `SAVE_DIR` y `RESUME_FROM` en [pathsegmentor.yaml](configs/pathsegmentor.yaml) según sea necesario. Estas variables especifican la ruta de salida para los puntos de control de entrenamiento y la ruta al punto de control preentrenado, respectivamente.

## 🧪 Evaluación
Una vez completado el entrenamiento, puedes evaluar tu PathSegmentor ajustado finamente en el conjunto de prueba de tu conjunto de datos ejecutando:
```bash
bash scripts/eval.sh
```
>💡 **Nota:** Puedes personalizar el directorio de salida para los archivos JSON de resultados en [default_trainer.py](trainer/default_trainer.py), alrededor de la línea 79.

<!-- # Citation -->
