# News_Classification-lab
Tarea 4 Task 2

# News Classification Lab (AG News · Transformers)

## Objetivo
Implementar, en un solo notebook/archivo, un flujo reproducible para **clasificación de noticias (AG News)** con **Hugging Face Transformers**, entrenando un modelo de secuencias (p. ej., RoBERTa/DeBERTa) en un **subset configurable** y reportando **Accuracy** y **F1 Macro**; además, generar salidas comparativas y gráficos para evaluar el rendimiento. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## Secciones principales
1) **Setup del entorno y estructura**  
   - Clona `News_Classification-lab` y crea carpetas `src/`, `data/`, `notebooks/`, `outputs/`.  
   - Instala versiones fijadas de `torch`, `transformers`, `datasets`, `evaluate`, `scikit-learn`, `matplotlib`, `seaborn`.  
   :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

2) **Función de entrenamiento `train_one_small(...)`**  
   - Encapsula el flujo: carga AG News, divide 70/15/15, tokeniza, entrena con `Trainer`, y calcula `accuracy` y `f1_macro`.  
   - Firma (parámetros clave): `model_name`, `epochs`, `batch_size`, `max_length`, `train_samples`, `val_samples`, `test_samples`.  
   :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

3) **Experimentos**  
   - Entrenamientos rápidos con subsets (p. ej., 1000/500/500) para comparar modelos como **RoBERTa-base** y **DeBERTa-v3-small**.  
   :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

4) **Visualización y comparación**  
   - Genera y guarda gráficos de barras (F1/Accuracy), dispersión F1 vs Accuracy, y ranking en `outputs/plots/`.  
   :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}

---

## Dataset y tarea
- **Dataset:** AG News (Hugging Face `datasets`).  
- **Tarea:** Clasificación de 4 clases a partir del texto.  
- **Split:** 70% train / 15% valid / 15% test (a partir de un 70/30 + 50/50).  
- **Subset opcional:** selección reproducible de `train/val/test` para ejecución rápida.  
:contentReference[oaicite:14]{index=14}

---

## Cómo funciona (resumen técnico)
- **Tokenización:** `AutoTokenizer` con `truncation=True`, `padding='max_length'`, `max_length` configurable. :contentReference[oaicite:15]{index=15}  
- **Modelo:** `AutoModelForSequenceClassification(num_labels=4)`. :contentReference[oaicite:16]{index=16}  
- **Entrenamiento:** `Trainer` con `TrainingArguments` (batch sizes, epochs, weight decay, eval/save por época). :contentReference[oaicite:17]{index=17}  
- **Métricas:** `accuracy` y `f1_macro` con scikit-learn. :contentReference[oaicite:18]{index=18}  
- **Persistencia:** guarda `metrics.json` por modelo en `outputs/<modelo>/`. :contentReference[oaicite:19]{index=19}

---

## Uso rápido (Colab)
1. Ejecuta las celdas en orden (incluyen clonación/instalación/estructura). :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}  
2. La función `train_one_small` se escribe en `src/train_small.py` y se importa en tiempo de ejecución. :contentReference[oaicite:22]{index=22} :contentReference[oaicite:23]{index=23}  
3. Ejemplo mínimo:
   ```python
   from src.train_small import train_one_small
   res = train_one_small(
       model_name='FacebookAI/roberta-base',
       epochs=1,
       batch_size=8,
       max_length=64,
       train_samples=1000, val_samples=500, test_samples=500
   )
   print(res)  # {'model': ..., 'accuracy': ..., 'f1_macro': ...}
