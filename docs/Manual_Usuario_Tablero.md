# Manual de Usuario — Tablero SciText-ES

**Versión:** 1.0  \
**Audiencia:** Usuarios finales que consumen el tablero ya desplegado (sin instalación local).  \
**Alcance:** Uso del tablero para análisis de documentos científicos en español.

---

## 1. Descripción general

SciText-ES es una aplicación web para analizar textos científicos en español. Permite:

- **Tarea 1:** Segmentación retórica (INTRO, BACK, METH, RES, DISC, CONTR, LIM, CONC).
- **Tarea 2:** Detección de contribuciones científicas (binario: sí/no).
- **Comparación:** Métricas y trade-offs entre tres enfoques de modelos.

El tablero está organizado en 4 vistas principales:

1. **Entrada** — Pegar texto y seleccionar modelo/tareas.
2. **Segmentación** — Resultados por párrafo con etiquetas y confianza.
3. **Contribuciones** — Fragmentos detectados como aportes.
4. **Comparación** — Métricas y tiempo/costo estimado.

---

## 2. Acceso al tablero

Abre en tu navegador la URL oficial del tablero:

```
http://100.52.250.51:5000/
```

Recomendación: usar **Chrome** o **Firefox** actualizados.

**Imagen 1 — Pantalla de acceso**

![Pantalla de acceso](../docs/images/01_acceso.png)

---

## 2.1 Acceso vía API (opcional)

Además del tablero, la API se puede consumir directamente (por ejemplo desde **Postman**), ya sea contra un backend local o una **IP pública**.

**Base URL**
- Local: `http://localhost:5000`
- IP pública: `http://100.52.250.51:5000`

### 2.1.1 Endpoint principal — Análisis

- **Método:** `POST`
- **Ruta:** `/api/analyze`
- **Headers:** `Content-Type: application/json`

**Mock de request (Postman)**

```json
{
  "text": "En este trabajo presentamos un sistema automático para segmentar secciones retóricas en artículos científicos en español. El objetivo es facilitar la recuperación de información y la comparación de métodos.\n\nNuestra metodología combina un modelo encoder con un clasificador ligero. Evaluamos en un corpus de artículos y reportamos mejoras en F1.",
  "model": "encoder",
  "tasks": ["segmentation", "contributions"],
  "encoder_variant": "roberta"
}
```

**Imagen 1.1 — Ejemplo en Postman**

![Postman](../docs/images/postman.png)

**Mock de response (200)**

```json
{
  "id": "b8b2e9b1-3d3e-4d2d-9c85-1c2f4b7b3a21",
  "model": "encoder",
  "model_name": "Encoder (BETO/RoBERTa)",
  "segmentation": {
    "segments": [
      {
        "paragraph_index": 0,
        "text": "En este trabajo presentamos un sistema automático para segmentar secciones retóricas en artículos científicos en español. El objetivo es facilitar la recuperación de información y la comparación de métodos.",
        "label": "INTRO",
        "confidence": 0.86
      },
      {
        "paragraph_index": 1,
        "text": "Nuestra metodología combina un modelo encoder con un clasificador ligero. Evaluamos en un corpus de artículos y reportamos mejoras en F1.",
        "label": "METH",
        "confidence": 0.83
      }
    ],
    "stats": {
      "total_paragraphs": 2,
      "total_words": 58,
      "avg_confidence": 0.845,
      "time_seconds": 0.62
    }
  },
  "contributions": {
    "fragments": [
      {
        "paragraph_index": 0,
        "text": "En este trabajo presentamos un sistema automático para segmentar secciones retóricas en artículos científicos en español. El objetivo es facilitar la recuperación de información y la comparación de métodos.",
        "is_contribution": false,
        "contribution_type": null,
        "confidence": 0.31,
        "highlight": "",
        "source_label": "INTRO"
      },
      {
        "paragraph_index": 1,
        "text": "Nuestra metodología combina un modelo encoder con un clasificador ligero. Evaluamos en un corpus de artículos y reportamos mejoras en F1.",
        "is_contribution": true,
        "contribution_type": null,
        "confidence": 0.82,
        "highlight": "Nuestra metodología combina un modelo encoder",
        "source_label": "METH"
      }
    ],
    "stats": {
      "total_fragments": 2,
      "positive": 1,
      "negative": 1,
      "avg_confidence_positive": 0.82
    }
  }
}
```

### 2.1.2 Endpoint de comparación

- **Método:** `GET`
- **Ruta:** `/api/compare/{analysis_id}`
- **Nota:** Usa el `analysis_id` devuelto por `/api/analyze`.

**Mock de response (200)**

```json
{
  "analysis_id": "b8b2e9b1-3d3e-4d2d-9c85-1c2f4b7b3a21",
  "original_model": "encoder",
  "task1_metrics": {
    "encoder": { "f1": 0.84, "precision": 0.85, "recall": 0.83, "latency": 0.55 },
    "llm": { "f1": 0.80, "precision": 0.79, "recall": 0.81, "latency": 2.05 },
    "api": { "f1": 0.88, "precision": 0.89, "recall": 0.87, "latency": 1.45 }
  },
  "task2_metrics": {
    "encoder": { "f1": 0.78, "precision": 0.77, "recall": 0.79, "latency": 0.32 },
    "llm": { "f1": 0.83, "precision": 0.82, "recall": 0.84, "latency": 1.18 },
    "api": { "f1": 0.91, "precision": 0.92, "recall": 0.90, "latency": 0.86 }
  },
  "cost_per_doc": { "encoder": 0.0011, "llm": 0.0083, "api": 0.0345 },
  "total_time": { "encoder": 0.87, "llm": 3.23, "api": 2.31 }
}
```

---

## 3. Vista de Entrada (inicio)

### 3.1 Pegar texto

1. Copia el contenido del artículo científico (mínimo recomendado: 250 caracteres).
2. Pega el texto en el campo principal.
3. El sistema separa párrafos cuando hay líneas en blanco.

**Ejemplo de texto para analizar**

```text
En este trabajo presentamos un sistema automático para segmentar secciones retóricas en artículos científicos en español. El objetivo principal es facilitar la recuperación de información y la comparación de métodos en el dominio académico. La motivación surge de la dificultad de analizar manualmente grandes volúmenes de literatura.

En la literatura previa, diversos autores han propuesto técnicas basadas en reglas y modelos estadísticos para identificar introducción, metodología y resultados. Sin embargo, estos enfoques suelen degradarse cuando el estilo del artículo cambia o cuando el dominio es muy específico.

Nuestra metodología combina un modelo encoder fine-tuned con un clasificador ligero para cada párrafo. Utilizamos un corpus de artículos científicos y aplicamos normalización, limpieza y segmentación. El entrenamiento se realizó con validación cruzada y ajuste de hiperparámetros.

Los resultados obtenidos demuestran mejoras consistentes en F1 frente a un baseline. En la tabla 2 se observa que el modelo propuesto supera el rendimiento de métodos clásicos en las categorías METH y RES.

En conclusión, el sistema propuesto mejora la identificación de secciones retóricas y permite detectar aportes científicos con mayor precisión. Como trabajo futuro, se evaluará la transferencia a otros dominios y la incorporación de modelos más grandes.
```

### 3.2 Selección de modelo

- **Encoder (BETO/RoBERTa)**
  - Rápido y bajo costo.
  - Ideal para pruebas rápidas.

- **Llama 3.3 70B Instruct (OpenRouter)**
  - Alta calidad, pero sujeto a latencia y límites de API.
  - Recomendado si necesitas mejor desempeño.

- **API Comercial**
  - Máxima calidad (si está habilitada).
  - Puede requerir clave/API configurada.

### 3.3 Selección de tareas

- **Tarea 1 — Segmentación retórica** (recomendado)
- **Tarea 2 — Extracción de contribuciones** (recomendado)

Puedes ejecutar una o ambas.

### 3.4 Ejecutar análisis

Haz clic en **Analizar documento**. Aparecerá un indicador de carga.

**Imagen 2 — Vista de entrada**

![Vista de entrada](../docs/images/02_entrada.png)

---

## 4. Vista de Segmentación

En esta vista verás:

- Lista de párrafos con su **etiqueta retórica**.
- **Confianza** por párrafo (barra y porcentaje).
- Resumen con:
  - Total de párrafos
  - Total de palabras
  - Confianza promedio
  - Tiempo de análisis

### Interpretación rápida

- **INTRO**: contexto y objetivo del trabajo.
- **BACK**: antecedentes y trabajos previos.
- **METH**: metodología o procedimiento.
- **RES**: resultados y métricas.
- **DISC**: discusión e interpretación.
- **CONTR**: contribución explícita.
- **LIM**: limitaciones y trabajo futuro.
- **CONC**: conclusiones.

**Imagen 3 — Segmentación retórica**

![Segmentación retórica](../docs/images/03_segmentacion.png)

---

## 5. Vista de Contribuciones

Muestra fragmentos del texto donde el sistema detecta aportes científicos.

Para cada fragmento:

- **Is contribution:** verdadero/falso
- **Confianza** del clasificador
- **Highlight**: frase clave destacada (si aplica)

### Consejos

- Si no aparecen contribuciones, prueba con un texto más largo o más explícito.
- La sección de **Metodología** y **Resultados** suele tener más contribuciones.

**Imagen 4 — Contribuciones**

![Contribuciones](../docs/images/04_contribuciones.png)

---

## 6. Vista de Comparación

Resume métricas entre los modelos disponibles:

- **F1, Precisión, Recall** por tarea.
- **Latencia** estimada.
- **Costo por documento** (estimación comparativa).
- **Trade-offs**: pros y contras de cada modelo.

**Imagen 5 — Comparación de modelos**

![Comparación de modelos](../docs/images/05_comparacion.png)

---

## 7. Exportar reporte

En la vista de comparación, puedes usar **Exportar reporte** para guardar un JSON con:

- Resultados de segmentación
- Contribuciones detectadas
- Métricas comparativas

---

## 8. Errores frecuentes y soluciones

### 8.1 “Backend no disponible”
- El servidor no está accesible.
- Verifica conexión o consulta al administrador.

### 8.2 “Timeout llamando al backend”
- El modelo tardó mucho en responder.
- Reintenta o cambia a un modelo más rápido (Encoder).

### 8.3 “Rate limited” (OpenRouter)
- El modelo gratuito está temporalmente limitado.
- Espera y reintenta o cambia de modelo.
- **Recomendación:** deja unos minutos entre ejecuciones cuando uses el modelo Llama 3.3 70B Instruct (OpenRouter free).

---

## 9. Buenas prácticas

- Usa textos con **varios párrafos** y contenido académico real.
- Evita textos demasiado cortos o sin estructura.
- Si necesitas rapidez, usa **Encoder**.
- Para calidad, usa **Llama 3.3 70B Instruct** (cuando haya disponibilidad).

---

## 10. Glosario

- **Retórica**: estructura funcional de un texto (introducción, metodología, etc.).
- **F1**: métrica balanceada entre precisión y recall.
- **Recall**: proporción de verdaderos positivos detectados.
- **Precisión**: proporción de detecciones correctas.
- **Latencia**: tiempo de respuesta del modelo.

---

## 11. Soporte

Si tienes problemas, contacta al equipo responsable del despliegue y comparte:

- URL que estabas usando
- Texto de error completo
- Modelo seleccionado
- Fecha y hora del incidente

## 12. Referencias

- Aplicación desplegada: `http://100.52.250.51:5000/`
- Repositorio del proyecto: `https://github.com/jdyepest/entrega_2_micropryecto`

---

**Fin del manual.**
