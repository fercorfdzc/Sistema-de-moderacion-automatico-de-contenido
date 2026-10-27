# Orígenes del Dataset

El conjunto de datos final (`dataset_ready_nlp.csv`) utilizado para entrenar el modelo BERT de moderación de chat ha sido construido a partir de la consolidación, limpieza y balanceo de múltiples fuentes públicas:

## 1. Spanish Hate Speech Superset
- **Enlace:** [manueltonneau/spanish-hate-speech-superset](https://huggingface.co/datasets/manueltonneau/spanish-hate-speech-superset)
- **Descripción:** Dataset en español utilizado como una de las bases principales para la detección de discursos de odio (*Hate Speech*). (Relacionado a la base `es_hf_102024.csv`).

## 2. Dataset TASS 2019
- **Enlace:** [mrm8488/tass-2019](https://huggingface.co/datasets/mrm8488/tass-2019)
- **Descripción:** Conjunto de datos especializado en análisis de sentimientos en español proveniente del Taller de Análisis Semántico en la SEPLN. Se adaptaron las etiquetas originales de sentimiento para alinearlas con una clasificación binaria.

## 3. HatEval
- **Enlace:** [valeriobasile/HatEval](https://huggingface.co/datasets/valeriobasile/HatEval)
- **Descripción:** Dataset enfocado en la detección de lenguaje de odio (ej. contra mujeres e inmigrantes en Twitter). Aporta ejemplos concretos para mejorar la robustez del modelo.

## 4. Hate Speech and Offensive Language (CrowdFlower)
- **Enlace:** [t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language)
- **Descripción:** Dataset en inglés (`labeled_data.csv`) enfocado específicamente en distinguir entre lenguaje ofensivo y discursos de odio real.

## 5. Stanford Sentiment Treebank (SST-2)
- **Enlace:** [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2)
- **Descripción:** Dataset en inglés de Hugging Face (benchmark GLUE), enfocado en análisis de sentimiento (positivo/negativo) de reseñas de películas. Ayuda a mejorar la detección de polaridad y contexto en el idioma inglés.

---

### Proceso de Preparación
Estos conjuntos de datos originales pasaron por un riguroso proceso de limpieza donde se consolidaron en un solo formato binarizado. Se removieron:
- URLs y enlaces
- Menciones a usuarios
- Caracteres especiales y ruido
- Registros duplicados

Finalmente, se aplicó una técnica de submuestreo (*undersampling*) para garantizar que existiera un balance exacto entre los mensajes etiquetados como "No Hate / Clean" (0.0) y "Hate" (1.0) en cada idioma, obteniendo un conjunto de datos robusto y equitativo para el entrenamiento del modelo.
