import json
import os
import sys


def main() -> None:
    """
    Smoke test para LLM local (Ollama) usando /api/analyze con model=llm.

    Requisitos:
      - Ollama corriendo (por defecto http://localhost:11434)
      - Modelo descargado:
          ollama pull llama3.3:70b-instruct
        (o configura LOCAL_LLM_MODEL)
    """
    backend_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, backend_dir)

    from main import app  # noqa: E402

    sample_text = (
        "En este trabajo presentamos un enfoque para segmentación retórica.\n\n"
        "Según trabajos previos, los transformadores han mejorado el rendimiento.\n\n"
        "Utilizamos un modelo RoBERTa fine-tuned y evaluamos en un corpus científico.\n\n"
        "Los resultados muestran una mejora significativa.\n\n"
        "Nuestra contribución principal es liberar un dataset anotado.\n\n"
        "En conclusión, el método es efectivo."
    )

    payload = {"text": sample_text, "model": "llm", "tasks": ["segmentation", "contributions"]}

    with app.test_client() as client:
        resp = client.post("/api/analyze", json=payload)
        print("status:", resp.status_code)
        data = resp.get_json()
        if resp.status_code != 200:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return
        print("model:", data.get("model"))
        seg = (data.get("segmentation") or {}).get("segments") or []
        cont = (data.get("contributions") or {}).get("fragments") or []
        print("segments:", len(seg), "contrib_fragments:", len(cont))
        if seg:
            print("seg[0]:", json.dumps(seg[0], ensure_ascii=False, indent=2))
        if cont:
            print("cont[0]:", json.dumps(cont[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Optional: make it easy to switch to 'large' variant for a quick test
    os.environ.setdefault("LOCAL_LLM_VARIANT", "small")
    main()

