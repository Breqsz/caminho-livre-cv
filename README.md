# Caminho Livre CV

Detector de barreiras de acessibilidade em calçadas, em tempo real, com **YOLO11n fine-tunado**.
Global Solution FIAP 2026.2 · Engenharia de Software 4ES · disciplina **Applied Computer Vision**.

> **Status: em desenvolvimento.** Código de inferência, testes e notebook de treino prontos.
> O treino no Colab, as métricas e o relatório ainda **não** foram concluídos. Nenhum número de
> desempenho deste README é definitivo até a seção *Resultados* ser preenchida com a execução real.

## Ficha técnica

| | |
|---|---|
| Equipe | Guilherme Rocha Bianchini (RM97974) · Nikolas Rodrigues Moura dos Santos (RM551566) · Rodrigo Brasileiro (RM98952) · Thiago Jardim de Oliveira (RM551624) |
| Produto da equipe | Caminho Livre — mapa colaborativo de barreiras de acessibilidade (este módulo é **independente** do MVP) |
| Tarefa | Detecção de objetos em tempo real (webcam, vídeo ou imagens) |
| Modelo | YOLO11n (Ultralytics), fine-tuning a partir dos pesos COCO |
| Dataset | [ROD-Dataset](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO) — 24.326 imagens, 25 classes |
| Treino | Google Colab, GPU T4 — [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) |
| Inferência | CPU local, Python 3.12, OpenCV |
| Licença | AGPL-3.0 (exigida pelo Ultralytics) |

## O que o sistema faz

1. Lê frames de uma webcam, de um vídeo ou de uma pasta de imagens.
2. Detecta 25 classes de objetos urbanos com o YOLO11n treinado.
3. **Desfoca pessoas antes de exibir ou salvar** (LGPD — privacidade por padrão).
4. Traduz as classes para categorias de acessibilidade do Caminho Livre
   ([`configs/barriers.yaml`](configs/barriers.yaml), editável sem retreinar):

| Categoria | Classes |
|---|---|
| Calçada obstruída | Dustbin, Traffic Cone, Traffic Barrel, Electrical Pole, Electrical Box, Plant Pot, Bench, Chair, Bicycle Rack, Fire hydrant, Tree, Guard rail |
| Veículo — verificar se está na calçada | Car, Motorcycle, Bike |
| Possível barreira de nível | Stairs |
| Atenção ao piso | Manhole |
| Travessia identificada (contexto, não conta) | Pedestrian crosswalk |
| Pessoa — desfocada, não conta | Person |
| Ignorado | Building, Road, Traffic sign, Truck, Bus, Dog |

5. Mostra painel com contagem por categoria, FPS e o aviso
   **"Detecção automática — requer verificação humana"**.

## Como rodar

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
pytest                            # testes unitários

# inferência (precisa de models/best.pt — gerado pelo notebook de treino)
python -m caminho_livre_cv.detect --source 0                        # webcam
python -m caminho_livre_cv.detect --source rua.mp4 --save out.mp4   # vídeo
python -m caminho_livre_cv.detect --source pasta/ --save saida/ --no-show
```

Opções: `--conf` (padrão 0.35), `--no-blur` (desliga o desfoque), `--weights`, `--imgsz`. `q` ou `Esc` encerra.

## Como treinar

Abra [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) no Google Colab com GPU T4 e execute tudo.
O notebook baixa o dataset, corrige o nome da classe 20 (`Teraffic Barrel` → `Traffic Barrel`),
treina 50 épocas (`imgsz=640`, `patience=15`, `seed=0`), avalia no split **test** e salva os artefatos
no Google Drive. O notebook é gerado por `scripts/build_notebook.py`.

Scripts de medição (após o treino):

```bash
python scripts/prepare_dataset.py --root data/rod --download
python scripts/benchmark_fps.py --weights models/best.pt --images data/rod/test/images --n 200
python scripts/compare_baseline.py --images <imagens> --out docs/assets/baseline
```

## Resultados

*Pendente — será preenchido com a execução real do treino (mAP50, mAP50-95, precisão/recall por classe,
matriz de confusão, FPS em CPU e comparação com o YOLO11n COCO sem fine-tune).*

## Limitações conhecidas

- O dataset não tem a classe **rampa**; rampas de acesso não são detectadas.
- Imagens do dataset vêm de outros países (Roboflow Universe, Canadá, Irã): o desempenho em calçadas de São Paulo pode ser menor.
- O modelo não sabe se um veículo está **sobre** a calçada ou na rua — por isso a categoria pede verificação.
- A detecção é um apoio à triagem, não uma medição de conformidade legal de acessibilidade.

## Dados e licenças

- ROD-Dataset: licença MIT; as 26 coleções de origem do Roboflow Universe mantêm suas licenças originais.
  Zandi et al., *Real-Time Obstacle Detection and Distance Estimation on Smartphones for Visually Impaired
  and Distracted Pedestrians*, 2026 — [arXiv:2611.00681](https://arxiv.org/abs/2611.00681).
- Ultralytics YOLO11: AGPL-3.0.
- Nenhum dado pessoal é coletado; pessoas são desfocadas em toda saída gerada.
