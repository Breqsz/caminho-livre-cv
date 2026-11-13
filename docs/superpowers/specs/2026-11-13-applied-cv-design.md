# Caminho Livre CV — Design (Applied Computer Vision, GS 2026.2)

Status: APROVADO pelo Breq em 2026-11-13 · entrega 19/11/2026 23h55

Equipe: Guilherme Rocha Bianchini (RM97974) · Nikolas Rodrigues Moura dos Santos (RM551566) ·
Rodrigo Brasileiro (RM98952) · Thiago Jardim de Oliveira (RM551624)

## 1. Intenção e porquê

Entregar a disciplina **Applied Computer Vision** da Global Solution FIAP 2026.2 (4ES).
Rubrica (PDF da FIAP, p. 43–44): detecção de objetos em tempo real com **treinamento
customizado** de uma arquitetura de visão; entregáveis = repositório GitHub público
com código e documentação + vídeo de demonstração; nota = PDF (até 5) + vídeo (até 5).

Tema: barreiras de acessibilidade em calçadas, coerente com o produto da equipe
(**Caminho Livre**), mas como **módulo independente** — sem integração técnica com o
MVP (decisão do Breq, 2026-11-13). O MVP continua sem depender de visão computacional.

## 2. Objetivos

- Fine-tuning de **YOLO11** (Ultralytics) num dataset público de obstáculos urbanos.
- Inferência em **tempo real** (webcam, arquivo de vídeo, pasta de imagens) com overlay
  que traduz as classes detectadas para as categorias do Caminho Livre.
- Métricas reais e reproduzíveis: mAP50, mAP50-95, precisão/recall por classe, matriz de
  confusão, FPS em CPU.
- Repositório com README (ficha técnica, setup, treinar, rodar, resultados, licenças, LGPD).
- Relatório PDF no formato da rubrica e roteiro do vídeo.

## 3. Não-objetivos

Integração com API/MVP · dataset próprio ou anotação · realidade aumentada · app mobile ·
deploy em nuvem · detecção de rampas (o dataset escolhido não tem essa classe) ·
GenAI/LLM.

## 4. Anti-objetivos (falha mesmo que "funcione")

- Afirmar que o modelo mede conformidade legal ou "garante" que a calçada é acessível.
- Métrica, FPS ou exemplo inventado — tudo vem de execução real, com log.
- Usar o YOLO pré-treinado no COCO como se fosse o modelo customizado.
- Vídeo ou exemplo publicado com rosto identificável (LGPD).

## 5. Restrições

- Prazo: 6 dias (13→19/11).
- Máquina local sem GPU NVIDIA → **treino no Google Colab (T4)**; inferência em CPU local.
- Python local é 3.14; se torch/ultralytics não tiverem wheel, usar venv 3.12.
- Ultralytics é **AGPL-3.0** → repositório público sob AGPL-3.0.
- Desvio do stack default (Next/TS): justificado, pois CV com YOLO é ecossistema Python.

## 6. Abordagens consideradas

| # | Abordagem | Trade-off | Decisão |
|---|---|---|---|
| 1 | **Só dataset público** + YOLO11 fine-tune | Mais rápido e reproduzível; risco de gap de domínio (fotos não são de SP) | **Escolhida pelo Breq** |
| 2 | Só dataset próprio (filmar território, anotar 300–500 frames) | Mais autêntico; 1–2 dias de anotação, pouca imagem → mAP baixo | Descartada (prazo) |
| 3 | Híbrido (público + 150–300 frames próprios) | Melhor demo em SP; custa horas de anotação | Descartada |
| — | Realidade aumentada | Não agrega na rubrica, alto risco | Fora |

Mitigação do gap de domínio da abordagem 1: filmar um trecho de calçada de SP **só para
inferência** (demo qualitativa, sem rótulo, declarado como tal no relatório).

## 7. Dados

**Dataset:** [`Abtinzandi/Obstacle-Detection-Dataset-YOLO`](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO) (Hugging Face).
- Licença MIT do agregado; as 26 coleções de origem (Roboflow Universe) **mantêm suas
  licenças** → citar no README e no relatório.
- 24.326 imagens / 40.195 caixas / 25 classes; train 19.186 · val 3.511 · test 1.629.
- Formato YOLO TXT (normalizado) + CSV de metadados por split; ~1,54 GB.
- Classes: Bike, Building, Car, Person, Stairs, Traffic sign, Electrical Pole, Road,
  Motorcycle, Dustbin, Dog, Manhole, Tree, Guard rail, Pedestrian crosswalk, Truck, Bus,
  Bench, Traffic Cone, Fire hydrant, Traffic Barrel, Plant Pot, Electrical Box, Chair,
  Bicycle Rack.

Alternativas avaliadas e não usadas: RampNet (anotação por ponto, não caixa) · Zenodo
"Image Dataset of Accessibility Barriers" (CC BY 4.0, degraus/escadas/rampas, 8 GB em
CVAT XML; fica como extensão futura para rampas) · Mendeley (só 4 classes).

## 8. Modelo e treino

- **Treinar nas 25 classes originais** (chamada minha): remapear antes do treino deixaria
  objetos sem rótulo nas imagens e juntaria objetos visualmente diferentes numa só classe.
  A tradução para categorias de acessibilidade fica na camada de inferência.
- Modelo principal **YOLO11n** (tempo real em CPU), a partir dos pesos COCO.
  YOLO11s fica como comparação opcional, se sobrar tempo de GPU.
- Colab T4, `imgsz=640`, ~50 épocas com `patience` de early stopping; checkpoints
  salvos no Google Drive para sobreviver a desconexões.
- Avaliar no split **test** (não usado no treino); reportar artefatos do Ultralytics
  (curvas PR, matriz de confusão, `results.csv`).
- **Baseline para justificar o treino customizado:** YOLO11n COCO sem fine-tune não tem
  classes como cone, barril, poste, lixeira, faixa de pedestre ou escada. Mostrar isso
  lado a lado no relatório.
- Pesos finais (`best.pt`, ~6 MB para o n) versionados em `models/` ou numa GitHub Release.

## 9. Camada de inferência (o "produto" desta disciplina)

CLI: `python -m caminho_livre_cv.detect --source <0|video.mp4|pasta> --weights models/best.pt [--save] [--no-blur]`

Saída: janela com caixas e rótulos em PT-BR, painel com contagem por categoria, FPS e
aviso fixo "detecção automática — requer verificação humana". Com `--save`, grava mp4 ou
imagens anotadas.

Mapeamento em `configs/barriers.yaml` (editável e documentado):

| Categoria no overlay | Classes |
|---|---|
| Calçada obstruída — mobiliário/objeto | Dustbin, Traffic Cone, Traffic Barrel, Electrical Pole, Electrical Box, Plant Pot, Bench, Chair, Bicycle Rack, Fire hydrant, Tree, Guard rail |
| Veículo — verificar se está sobre a calçada | Car, Motorcycle, Bike |
| Possível barreira de nível | Stairs |
| Atenção ao piso | Manhole |
| Travessia identificada (contexto) | Pedestrian crosswalk |
| Privacidade — desfocado, não contado | Person |
| Ignorado | Building, Road, Traffic sign, Truck, Bus, Dog |

**Privacidade by design:** caixas de `Person` são desfocadas (Gaussian blur) por padrão
antes de exibir ou salvar.

Lógica pura (mapeamento, resumo por categoria, blur) separada do I/O → testável com pytest.

## 10. Estrutura do repositório

```
caminho-livre-cv/
  README.md                 ficha técnica, setup, treino, inferência, resultados, licenças, LGPD
  LICENSE                   AGPL-3.0
  pyproject.toml / requirements.txt
  configs/barriers.yaml
  notebooks/train_colab.ipynb   download → stats → treino → avaliação → export
  scripts/prepare_dataset.py    baixa do HF, gera data.yaml, estatísticas por classe
  scripts/benchmark_fps.py      FPS em CPU (registra hardware)
  scripts/compare_baseline.py   COCO vs fine-tuned nas mesmas imagens
  src/caminho_livre_cv/{detect,barrier_map,privacy,overlay,config}.py
  tests/                        mapeamento, resumo, blur
  models/best.pt
  docs/report/relatorio.md → relatorio.pdf
  docs/assets/                  gráficos e exemplos (sem rostos)
  docs/video/roteiro.md
```

## 11. Relatório PDF (rubrica)

Título · integrantes (nome + RM) · resumo (**≤250 palavras**) · introdução · objetivo
(**≤50 palavras**) · explicação técnica das ferramentas (YOLO11, Ultralytics, PyTorch,
OpenCV, Colab, dataset) · metodologia passo a passo · exemplos de resultados (métricas,
curvas, matriz, baseline vs fine-tuned, demo em SP) · conclusão com limitações.
Fonte em Markdown, convertido para PDF.

## 12. Vídeo (roteiro)

~3–5 min: problema → dataset → treino no Colab → métricas → **demo em tempo real**
(webcam + trecho de calçada de SP) → limitações → repositório. A equipe grava.

## 13. Cronograma

| Dia | Entrega |
|---|---|
| 13/11 | repo, `prepare_dataset`, notebook pronto |
| 14/11 | treino no Colab, avaliação, baseline |
| 15–16/11 | CLI de inferência, mapeamento, blur, testes, benchmark FPS |
| 17/11 | relatório PDF + README |
| 18/11 | filmagem SP, gravação do vídeo |
| 19/11 | revisão final, links em janela anônima, envio |

## 14. Riscos

- Desconexão do Colab → checkpoints no Drive + `resume=True`.
- Gap de domínio em SP → declarado; demo SP é qualitativa.
- Licenças heterogêneas das fontes → citação explícita, uso acadêmico.
- Desbalanceamento de classes → reportar AP por classe, sem esconder classes fracas.

## 15. Fica com o executor / pendências do Breq

- **Breq:** nomes e RMs dos integrantes; quem filma o trecho de SP; conta Google para o Colab; conta GitHub (Breqsz) do repositório.
- **Executor:** hiperparâmetros finos, layout do overlay, formato exato das tabelas.
