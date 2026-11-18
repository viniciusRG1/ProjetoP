from collections import Counter
import json
import zipfile

caminho_zip = r"C:\Users\vinic\Downloads\teste.zip"  # ajuste o caminho do .zip
contagem = Counter()


def extrair_tags(obj):
    if isinstance(obj, dict):
        # Padrão LabelMe / X-AnyLabeling
        if "shapes" in obj and isinstance(obj["shapes"], list):
            for s in obj["shapes"]:
                if isinstance(s, dict) and "label" in s:
                    contagem[str(s["label"])] += 1
            return

        # Padrão COCO
        if "categories" in obj and "annotations" in obj:
            mapa = {c["id"]: c["name"] for c in obj["categories"]}
            for ann in obj["annotations"]:
                nome = mapa.get(ann.get("category_id"))
                if nome:
                    contagem[str(nome)] += 1
            return

        # Padrão VoTT / Label Studio / genérico
        for k, v in obj.items():
            if k == "tags" and isinstance(v, list):
                for t in v:
                    if isinstance(t, str):
                        contagem[t] += 1
                    elif isinstance(t, dict) and "name" in t:
                        contagem[str(t["name"])] += 1
            elif k in ("label", "tag", "class", "category") and isinstance(
                v, str
            ):
                contagem[v] += 1
            elif isinstance(v, (dict, list)):
                extrair_tags(v)

    elif isinstance(obj, list):
        for item in obj:
            extrair_tags(item)


with zipfile.ZipFile(caminho_zip, "r") as z:
    jsons = [
        f
        for f in z.namelist()
        if f.lower().endswith(".json") and not f.startswith("__MACOSX")
    ]
    for nome in jsons:
        with z.open(nome) as f:
            extrair_tags(json.load(f))

print(f"JSONs lidos: {len(jsons)}")
print(f"Tags únicas: {len(contagem)}\n")

for tag, qtd in sorted(contagem.items()):
    print(f"- {tag}: {qtd}")