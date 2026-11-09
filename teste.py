from pathlib import Path
from ultralytics import YOLO


def inferir_imagens():
    # Caminho dos melhores pesos treinados
    pesos = Path(r"runs\detect\treino_garrafas-9\weights\best.pt")

    if not pesos.exists():
        pesos = Path(r"runs\detect\treino_garrafas-9\weights\last.pt")

    modelo = YOLO(str(pesos))

    # Pasta de entrada e pasta de saida
    pasta_origem = Path(r"dataset_yolo\images\val")
    pasta_destino = Path(r"runs\inferencia_teste")

    # Executa inferencia
    resultados = modelo.predict(
        source=str(pasta_origem),
        conf=0.25,  # Confianca minima
        iou=0.45,  # Supressao nao-maxima
        imgsz=640,
        device=0,  # GPU RTX 5060
        save=True,  # Salva fotos com bounding boxes
        project=str(pasta_destino.parent),
        name=pasta_destino.name,
        exist_ok=True,
    )

    print(f"Predicoes salvas em: {pasta_destino}")


if __name__ == "__main__":
    inferir_imagens()