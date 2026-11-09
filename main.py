import json
import os
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

ORIGEM_DIR = Path("fotosComJason/2026-10-31 - Frontal")
DATASET_DIR = Path("dataset_yolo")
CLASSES = [
    "AguaAdv5L",
    "AguaGasAdv500ml",
    "AguaTonicaMineiroLata",
    "ColaAdv2L",
    "ColaAdv600ml",
    "ColaAdvLata",
    "ColaPetAdv1,5L",
    "FardoAguaAdv",
    "FardoAguaAdv1,5L",
    "FardoAguaAdv500ml",
    "FardoAguaGasAdv1,5L",
    "FardoAguaGasAdv500ml",
    "FardoAguaGasMineiro500ml",
    "FardoAguaMineiro",
    "FardoAguaTonicaMineiraLata",
    "FardoColaAdv",
    "FardoColaAdv600ml",
    "FardoColaAdvLata",
    "FardoFrutaAdv",
    "FardoFrutaAdv1L",
    "FardoFrutaAdv2L",
    "FardoFrutaAdvLata",
    "FardoFrutaMineiro2L",
    "FardoGuaranaAdv2L",
    "FardoGuaranaAdv600ml",
    "FardoGuaranaAdvLata",
    "FardoGuaranaMineiro2L",
    "FardoGuaranaMineiro600ml",
    "FardoGuaranaMineiroBaby",
    "FardoGuaranaMineiroBabyZero",
    "FardoGuaranaMineiroLata",
    "FardoGuaranaMineiroZero2L",
    "FardoGuaranaZeroAdv2L",
    "FardoGuaranaZeroAdvLata",
    "FardoGuaranaZeroMineiroLata",
    "FardoMiniPetAdv",
    "FardoMiniPetColaAdv",
    "FardoMiniPetFrutaAdv",
    "FardoMiniPetGuaranaAdv",
    "FardoMiniPetMineiro",
    "FardoZap2L",
    "FardoZapLata",
    "FrutaAdv1L",
    "FrutaAdv2L",
    "FrutaAdvLata",
    "FrutaMineiro2L",
    "FrutaMiniPetAdv",
    "GuaranaAdv1L",
    "GuaranaAdv2L",
    "GuaranaAdv600ml",
    "GuaranaLataAdv",
    "GuaranaMineiro2L",
    "GuaranaMineiro600ml",
    "GuaranaMineiroLata",
    "GuaranaMiniPetAdv",
    "GuaranaMiniPetZeroAdv",
    "GuaranaZeroAdv1L",
    "GuaranaZeroAdv2L",
    "GuaranaZeroAdv600ml",
    "GuaranaZeroAdvLata",
    "GuaranaZeroMiniPetAdv",
    "MineiroZero2L",
    "MineiroZeroLata",
    "Zap2L",
    "ZapLata",
    "ZeroAdv2L",
    "aguaAdv1,5L",
    "aguaAdv500ml",
    "aguaGasAdv1,5L",
    "aguaGasAdv500ml",
    "aguaGasMineiro500ml",
    "mineiroBabyZero",
    "mineirobaby",
    "miniPetColaAdv",
    "miniPetGuaranaAdv",
    "águamineiro",
]
SPLIT_TREINO = 0.8



def converter_bbox(points, img_w, img_h):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Normalizacao 0 a 1
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    # Clamping de seguranca
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return x_center, y_center, w, h


def preparar_dataset():
    # Cria diretorios necessarios
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    json_files = list(ORIGEM_DIR.glob("*.json"))
    random.seed(42)
    random.shuffle(json_files)

    split_idx = int(len(json_files) * SPLIT_TREINO)
    treino_jsons = json_files[:split_idx]
    val_jsons = json_files[split_idx:]

    for split, arquivos in [("train", treino_jsons), ("val", val_jsons)]:
        for jf in arquivos:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Busca imagem correspondente
            img_nome = data.get("imagePath", jf.stem + ".jpg")
            img_path = ORIGEM_DIR / Path(img_nome).name
            if not img_path.exists():
                for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
                    temp_p = ORIGEM_DIR / (jf.stem + ext)
                    if temp_p.exists():
                        img_path = temp_p
                        break

            if not img_path.exists():
                continue

            img_w = data["imageWidth"]
            img_h = data["imageHeight"]

            # Gera linhas TXT YOLO
            linhas = []
            for shape in data["shapes"]:
                label = shape["label"]
                if label not in CLASSES:
                    continue
                cls_id = CLASSES.index(label)
                xc, yc, w, h = converter_bbox(shape["points"], img_w, img_h)
                linhas.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            # Copia imagem e grava label
            shutil.copy2(img_path, DATASET_DIR / "images" / split / img_path.name)
            txt_path = DATASET_DIR / "labels" / split / f"{jf.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(linhas)

    # Cria arquivo YAML
    yaml_conteudo = f"""path: {DATASET_DIR.resolve()}
train: images/train
val: images/val

names:
"""
    for idx, nome in enumerate(CLASSES):
        yaml_conteudo += f"  {idx}: {nome}\n"

    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_conteudo)

    return yaml_path


def treinar(yaml_path):
    modelo = YOLO("yolo11n.pt")
    yaml_path = Path("")
    modelo.train(
        data=str(yaml_path),
        epochs=200,
        imgsz=640,
        batch=8,
        workers=0,
        name="treino_garrafas",
        exist_ok=False,
        device=0,
        val=False,
        plots=False,
    )


if __name__ == "__main__":
    caminho_yaml = preparar_dataset()
    treinar(caminho_yaml)