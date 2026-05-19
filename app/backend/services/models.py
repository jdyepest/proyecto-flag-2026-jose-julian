"""
Model configuration and constants.
When real APIs are available, update MODELS with actual connection details.
"""

MODELS = {
    "encoder": {
        "name": "Encoder (SciBERT)",
        "description": "Clasificador encoder fine-tuned para español científico",
        "icon": "⚡",
        "tag": "Rápido",
        "color": "#1565C0",
        # Real model: model_id, tokenizer, device config
        "simulated_delay_s": 0.5,
        "task1_f1_base": 0.83,
        "task2_f1_base": 0.78,
        "cost_per_doc": 0.001,
    },
    "llm": {
        "name": "LLM Open-Weight (Llama 3.1 8B)",
        "description": "Modelo open-weight servido por OpenRouter/Ollama proxy",
        "icon": "🧠",
        "tag": "Balanceado",
        "color": "#7B1FA2",
        # Real model: API endpoint, model_name, inference params
        "simulated_delay_s": 2.0,
        "task1_f1_base": 0.79,
        "task2_f1_base": 0.82,
        "cost_per_doc": 0.008,
    },
    "api": {
        "name": "API Comercial (GPT/Gemini)",
        "description": "Modelo de frontera vía API externa",
        "icon": "☁️",
        "tag": "Mayor calidad",
        "color": "#E65100",
        # Real model: API key, endpoint, model id
        "simulated_delay_s": 1.5,
        "task1_f1_base": 0.87,
        "task2_f1_base": 0.91,
        "cost_per_doc": 0.035,
    },
}

RHETORICAL_LABELS = ["INTRO", "BACK", "METH", "RES", "DISC", "CONTR", "LIM", "CONC"]

CONTRIBUTION_TYPES = ["Metodológica", "Empírica", "Recurso", "Conceptual"]
