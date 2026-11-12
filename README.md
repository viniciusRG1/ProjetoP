Detecção de Produtos e Auto-Rotulagem (YOLOv11 + LabelMe)

Sistema de visão computacional desenvolvido para detecção de 75 classes de produtos (garrafas, fardos e variantes de volume) em cenários de alta densidade (20+ itens por imagem), integrado a um fluxo de Active Learning para geração automática de anotações no formato LabelMe.

Tecnologias e Requisitos

Modelo: Ultralytics YOLOv11 (yolo11n / yolo11s)

Framework: PyTorch com suporte a CUDA (testado em NVIDIA RTX 5060 8GB)

Processamento de Imagem: OpenCV (cv2)

Ferramenta de Rotulagem: LabelMe

Instalação

Clone o repositório e acesse a pasta do projeto:

git clone https://github.com/viniciusRG1/ModeloProduto.git
cd ModeloProduto


Crie e ative um ambiente virtual (opcional, mas recomendado):

python -m venv .venv
.venv\Scripts\activate


Instale as dependências necessárias:

pip install ultralytics opencv-python labelme


Fluxo de Trabalho

1. Rotulagem Manual (LabelMe)

Abra a pasta de imagens carregando a lista de classes e desativando a gravação da imagem em base64 (--nodata) para manter os arquivos .json leves:

labelme "Fotos11Marco" --labels classes.txt --nodata


2. Treinamento

A estratégia recomendada segue duas etapas:

YOLOv11n (Nano): Validação rápida de pipeline e criação de linha de base.

YOLOv11s (Small): Escala para maior precisão quando o dataset atingir alta densidade de instâncias (~7.500+ caixas no total).

Para iniciar o treinamento:

python main.py


Principais hiperparâmetros utilizados:

imgsz=640 (ou 1280 para maior nitidez em itens pequenos/distantes)

epochs=250 com patience=40 (Early Stopping para evitar overfitting)

batch=8 e workers=0 (estabilidade em ambiente Windows)

mosaic=1.0 e mixup=0.1 (Data Augmentation para cenários densos)

3. Inferência e Auto-Anotação

O script de inferência realiza duas tarefas simultaneamente:

Salva imagens limpas (contendo apenas as bounding boxes, sem textos ou porcentagens) em runs/inferencia_teste/ para conferência visual.

Gera os arquivos .json compatíveis com o LabelMe diretamente na pasta de origem (Fotos11Marco/).

python inferencia.py


4. Revisão (Active Learning)

Após rodar a inferência, abra novamente a pasta Fotos11Marco no LabelMe para corrigir eventuais falsos positivos/negativos, converter para o formato YOLO e retroalimentar o treinamento.
