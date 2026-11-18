import json
from pathlib import Path
import cv2
from ultralytics import YOLO


def inferir_e_gerar_json():
    pesos = Path(r"runs\detect\treino_garrafas-11\weights\best.pt")

    if not pesos.exists():
        pesos = Path(r"runs\detect\treino_garrafas-11\weights\last.pt")

    modelo = YOLO(str(pesos))

    pasta_origem = Path(r"FotosAprovadas")
    pasta_destino = Path(r"runs\inferencia_teste")
    pasta_destino.mkdir(parents=True, exist_ok=True)

    resultados = modelo.predict(
        source=str(pasta_origem),
        conf=0.6,
        iou=0.45,
        imgsz=640,
        device=0,
        save=False,
        stream=True,
    )

    for resultado in resultados:
        img_path = Path(resultado.path)

        img = resultado.plot(labels=False, conf=False, line_width=2)
        caminho_saida = pasta_destino / img_path.name
        cv2.imwrite(str(caminho_saida), img)

        shapes = []
        for box in resultado.boxes:
            cls_id = int(box.cls[0])
            label = resultado.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            shapes.append(
                {
                    "label": label,
                    "confidence": round(conf, 4),
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None,
                }
            )

        dados_json = {
            "version": "7.9.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_path.name,
            "imageData": None,
            "imageHeight": resultado.orig_shape[0],
            "imageWidth": resultado.orig_shape[1],
        }

        caminho_json = img_path.parent / f"{img_path.stem}.json"
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(dados_json, f, indent=2, ensure_ascii=False)

    print(f"Imagens em: {pasta_destino}")
    print(f"JSONs em: {pasta_origem}")


if __name__ == "__main__":
    inferir_e_gerar_json()