# Caminho Livre CV — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detector YOLO11n fine-tunado no ROD-Dataset, com inferência em tempo real que traduz detecções para categorias de acessibilidade do Caminho Livre, mais o relatório PDF e o roteiro do vídeo da disciplina.

**Architecture:** O treino roda no Google Colab (T4) por um notebook versionado. Um pacote Python (`caminho_livre_cv`) separa a lógica pura (mapeamento de classes para categoria, resumo, blur, overlay) do I/O (CLI com OpenCV + Ultralytics). Scripts reproduzíveis geram FPS e a comparação com o baseline COCO.

**Tech Stack:** Python 3.12 (venv) · ultralytics (YOLO11) · torch CPU · opencv-python · numpy · PyYAML · huggingface_hub · pytest · Google Colab T4.

**Spec:** `docs/superpowers/specs/2026-11-13-applied-cv-design.md`

## Global Constraints

- Dataset: `Abtinzandi/Obstacle-Detection-Dataset-YOLO` (HF, `repo_type="dataset"`); splits `train/` `valid/` `test/`, cada um com `images/` e `labels/`.
- O `data.yaml` original tem o typo `'Teraffic Barrel'` (id 20) → o nosso `data.yaml` gerado usa `'Traffic Barrel'`. Os demais 24 nomes ficam exatamente como no original.
- Treinar nas **25 classes originais**; o mapeamento para categoria fica só na inferência (`configs/barriers.yaml`).
- Modelo base `yolo11n.pt`, `imgsz=640`, `epochs=50`, `patience=15`, `seed=0`; avaliação final no split **test**.
- Aviso fixo no overlay: `Detecção automática — requer verificação humana`.
- `Person` é sempre desfocado por padrão (`--no-blur` desliga) e nunca entra nas contagens.
- Rótulos do overlay e textos de usuário em PT-BR; identificadores de código em inglês.
- Licença do repositório: AGPL-3.0. Nenhum segredo, `.env` ou dataset bruto versionado (`data/` no `.gitignore`).
- Antes do 1º commit: `git config user.email guirochabianchini@gmail.com` (conta Breqsz). Commits sem Co-Authored-By / atribuição a Claude.
- Nenhuma métrica, FPS ou imagem de exemplo nos docs que não venha de execução real registrada em `docs/assets/`.
- Equipe (relatório/README): Guilherme Rocha Bianchini (RM97974), Nikolas Rodrigues Moura dos Santos (RM551566), Rodrigo Brasileiro (RM98952), Thiago Jardim de Oliveira (RM551624).

## Review Focus

1. **Nome de classe divergente** (typo `Teraffic Barrel`, ou pesos COCO no baseline com nomes como `potted plant`) → nunca `KeyError`; nome desconhecido vira categoria `ignorado`. Teste em Task 1 (`test_unknown_class_is_ignored`) e checagem de completude (`test_every_dataset_class_is_mapped`).
2. **Caixa parcialmente fora do frame ou com largura/altura zero** → o blur faz clip e ignora a caixa vazia, sem crash. Teste em Task 2.
3. **Pessoa com confiança abaixo do limiar de exibição** → ainda é desfocada (limiar de privacidade `0.15` < limiar de exibição padrão `0.35`). Teste em Task 4 (`test_low_conf_person_still_blurred`).
4. **Webcam ausente ou caminho de vídeo inexistente** → mensagem PT-BR em stderr e `exit code 2`, sem traceback. Teste em Task 4.
5. **Webcam reporta FPS 0 ao gravar com `--save`** → o writer usa 30 FPS. Teste em Task 4 (`test_writer_fps_fallback`).

---

### Task 1: Scaffold + configuração e mapeamento de barreiras

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, `LICENSE` (texto AGPL-3.0), `configs/barriers.yaml`
- Create: `src/caminho_livre_cv/__init__.py`, `src/caminho_livre_cv/config.py`, `src/caminho_livre_cv/barrier_map.py`
- Test: `tests/test_barrier_map.py`

**Interfaces:**
- Produces:
  - `config.Category` (dataclass frozen): `key: str, label: str, color_bgr: tuple[int,int,int], classes: tuple[str,...], counted: bool`
  - `config.BarrierConfig` (dataclass frozen): `categories: tuple[Category,...]`, `privacy_classes: frozenset[str]`, método `category_for(class_name: str) -> Category | None`
  - `config.load_barrier_config(path: Path = DEFAULT_CONFIG) -> BarrierConfig`; `DEFAULT_CONFIG = <repo>/configs/barriers.yaml`
  - `barrier_map.Detection` (dataclass frozen): `class_name: str, confidence: float, xyxy: tuple[int,int,int,int]`
  - `barrier_map.IGNORED = "ignorado"`
  - `barrier_map.categorize(det: Detection, cfg: BarrierConfig) -> str` (chave da categoria; desconhecida → `IGNORED`)
  - `barrier_map.summarize(dets: list[Detection], cfg: BarrierConfig) -> dict[str, int]` (label PT-BR → contagem; só categorias `counted`, na ordem do YAML, omite zeros)

`configs/barriers.yaml` — chaves, labels e classes exatamente como na tabela da seção 9 do spec:

| key | label | counted | classes |
|---|---|---|---|
| `calcada_obstruida` | Calçada obstruída | true | Dustbin, Traffic Cone, Traffic Barrel, Electrical Pole, Electrical Box, Plant Pot, Bench, Chair, Bicycle Rack, Fire hydrant, Tree, Guard rail |
| `veiculo` | Veículo — verificar se está na calçada | true | Car, Motorcycle, Bike |
| `nivel` | Possível barreira de nível | true | Stairs |
| `piso` | Atenção ao piso | true | Manhole |
| `travessia` | Travessia identificada | false | Pedestrian crosswalk |
| `privacidade` | Pessoa (desfocada) | false | Person |
| `ignorado` | Ignorado | false | Building, Road, Traffic sign, Truck, Bus, Dog |

`privacy_classes: [Person]`. Cores BGR à escolha do executor, distintas por categoria.

- [ ] **Step 1: Ambiente.** `py -3.12 -m venv .venv` (instalar o Python 3.12 via `winget install Python.Python.3.12` se faltar); `pip install -r requirements.txt` (`ultralytics`, `opencv-python`, `numpy`, `pyyaml`, `huggingface_hub`, `pytest`; torch CPU via `--index-url https://download.pytorch.org/whl/cpu`). `pip install -e .`. Esperado: `python -c "import ultralytics, cv2"` sem erro.
- [ ] **Step 2: Testes que falham** em `tests/test_barrier_map.py`:
  - `test_every_dataset_class_is_mapped`: as 25 classes (lista do Global Constraints, com `Traffic Barrel`) → `cfg.category_for(n) is not None` para todas.
  - `test_no_class_in_two_categories`.
  - `test_categorize_known`: `Detection("Traffic Cone", .9, (0,0,10,10))` → `"calcada_obstruida"`; `"Stairs"` → `"nivel"`.
  - `test_unknown_class_is_ignored`: `"potted plant"` e `"Teraffic Barrel"` → `IGNORED`.
  - `test_summarize_counts_only_counted`: 2× Dustbin, 1× Car, 3× Person, 1× Pedestrian crosswalk → `{"Calçada obstruída": 2, "Veículo — verificar se está na calçada": 1}`.
  - `test_summarize_empty` → `{}`.
- [ ] **Step 3:** `pytest tests/test_barrier_map.py -v` → FAIL (import error).
- [ ] **Step 4:** Implementar `config.py` e `barrier_map.py` conforme as interfaces.
- [ ] **Step 5:** `pytest -v` → PASS.
- [ ] **Step 6: Commit.** `git init`, `git config user.email guirochabianchini@gmail.com`, `git add -A`, `git commit -m "feat: scaffold e mapeamento de barreiras"`.

### Task 2: Privacidade (blur)

**Files:** Create `src/caminho_livre_cv/privacy.py` · Test `tests/test_privacy.py`

**Interfaces:**
- Produces: `privacy.blur_regions(frame: np.ndarray, boxes: list[tuple[int,int,int,int]], ksize: int = 51) -> np.ndarray` (retorna cópia; não altera a entrada; faz clip das caixas ao frame; ignora caixas com área 0 depois do clip; `ksize` forçado a ímpar)

- [ ] **Step 1: Testes que falham:**
  - `test_blur_changes_region_only`: frame 100×100 com ruído aleatório (seed 0), caixa `(10,10,50,50)` → região difere do original; pixel `(80,80)` igual.
  - `test_input_not_mutated`.
  - `test_box_partially_outside_is_clipped`: `(-20,-20,30,30)` e `(90,90,200,200)` → sem exceção, shape preservado.
  - `test_zero_area_box_ignored`: `(10,10,10,40)` → saída igual à entrada.
  - `test_even_ksize_ok`: `ksize=50` não levanta erro.
- [ ] **Step 2:** `pytest tests/test_privacy.py -v` → FAIL.
- [ ] **Step 3:** Implementar com `cv2.GaussianBlur` por ROI.
- [ ] **Step 4:** `pytest -v` → PASS.
- [ ] **Step 5:** `git commit -m "feat: blur de pessoas por padrão (LGPD)"`.

### Task 3: Overlay

**Files:** Create `src/caminho_livre_cv/overlay.py` · Test `tests/test_overlay.py`

**Interfaces:**
- Consumes: `Detection`, `BarrierConfig`, `categorize`, `summarize`, `IGNORED`.
- Produces: `overlay.DISCLAIMER = "Detecção automática — requer verificação humana"`; `overlay.render(frame: np.ndarray, dets: list[Detection], cfg: BarrierConfig, fps: float | None = None) -> np.ndarray` (cópia; caixa + rótulo `"<classe PT> <conf>"` para categorias ≠ `ignorado` e ≠ `privacidade`; painel no canto superior esquerdo com `summarize`, linha `FPS: x.x` quando houver e `DISCLAIMER` no rodapé)

Acentos: `cv2.putText` não renderiza "ç/ã". Usar Pillow (`ImageFont.truetype` com fonte do sistema, fallback `ImageFont.load_default()`) para o texto. Adicionar `pillow` ao `requirements.txt`. Os nomes PT das classes ficam num dict `CLASS_LABELS_PT` em `overlay.py` (ex.: `Traffic Cone → "Cone"`, `Dustbin → "Lixeira"`, `Electrical Pole → "Poste"`); para classe sem tradução, usar o nome original.

- [ ] **Step 1: Testes que falham:**
  - `test_render_returns_same_shape_and_copy`.
  - `test_render_draws_for_counted_class`: frame preto + Dustbin em `(20,20,60,60)` → algum pixel não preto na borda da caixa.
  - `test_render_skips_ignored_class`: frame preto + só `Building` → a área da caixa (excluindo painel e rodapé) segue preta.
  - `test_render_without_fps_ok` (`fps=None`).
- [ ] **Step 2:** `pytest tests/test_overlay.py -v` → FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** `pytest -v` → PASS.
- [ ] **Step 5:** `git commit -m "feat: overlay com categorias em PT-BR"`.

### Task 4: CLI de detecção em tempo real

**Files:** Create `src/caminho_livre_cv/detect.py`, `src/caminho_livre_cv/__main__.py` (não obrigatório) · Test `tests/test_detect.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `detect.PRIVACY_CONF = 0.15`; `detect.DEFAULT_CONF = 0.35`
  - `detect.results_to_detections(result, min_conf: float) -> list[Detection]` (usa `result.names`, `result.boxes.cls/conf/xyxy`)
  - `detect.process_frame(frame, predict: Callable[[np.ndarray, float], list[Detection]], cfg, conf: float = DEFAULT_CONF, blur: bool = True, fps: float | None = None) -> tuple[np.ndarray, list[Detection]]` — chama `predict(frame, PRIVACY_CONF)` uma vez; desfoca toda caixa em `cfg.privacy_classes` (qualquer conf ≥ PRIVACY_CONF) se `blur`; filtra `conf` para render e retorno.
  - `detect.writer_fps(cap_fps: float) -> float` (≤0 ou NaN → 30.0)
  - `detect.main(argv: list[str] | None = None) -> int` — args `--source` (default `"0"`; dígito → webcam), `--weights` (default `models/best.pt`), `--conf`, `--save PATH` (opcional; pasta de imagens → salva cada imagem anotada em PATH/; vídeo/webcam → mp4 `mp4v`), `--no-blur`, `--no-show`. Fonte inválida ou pesos ausentes → mensagem PT-BR em stderr, `return 2`. `q`/ESC encerra.
  - Uso: `python -m caminho_livre_cv.detect --source video.mp4 --save out.mp4`

- [ ] **Step 1: Testes que falham** (com `predict` falso; sem carregar modelo):
  - `test_low_conf_person_still_blurred`: predict retorna `Person` conf 0.2 em `(10,10,60,60)` sobre frame com ruído; `conf=0.35` → região alterada e `Person` fora do retorno.
  - `test_no_blur_flag_keeps_person_region`: mesmo caso, `blur=False` → região da pessoa igual ao original (excluindo o painel; usar caixa fora da área do painel, ex. `(200,200,260,260)` num frame 320×320).
  - `test_conf_filter`: Dustbin 0.3 e Dustbin 0.8 → só a de 0.8 retorna.
  - `test_writer_fps_fallback`: `writer_fps(0) == 30.0`, `writer_fps(float("nan")) == 30.0`, `writer_fps(25) == 25`.
  - `test_main_missing_video_returns_2`: `main(["--source","nao_existe.mp4","--weights",<arquivo tmp vazio>,"--no-show"]) == 2` e stderr contém `"não encontrad"`.
  - `test_main_missing_weights_returns_2`.
- [ ] **Step 2:** `pytest tests/test_detect.py -v` → FAIL.
- [ ] **Step 3:** Implementar. Validar fonte e pesos **antes** de carregar o YOLO. FPS por média móvel das últimas 30 iterações.
- [ ] **Step 4:** `pytest -v` → PASS.
- [ ] **Step 5: Smoke manual com pesos COCO** (antes do treino ficar pronto): `python -m caminho_livre_cv.detect --source <imagem de rua qualquer> --weights yolo11n.pt --save docs/assets/smoke/ --no-show` → roda sem erro; classes COCO caem em `ignorado` exceto as que coincidirem (`car`, `bench`... em minúsculas → `ignorado`, esperado).
- [ ] **Step 6:** `git commit -m "feat: CLI de detecção em tempo real"`.

### Task 5: Preparação do dataset

**Files:** Create `scripts/prepare_dataset.py` · Test `tests/test_prepare_dataset.py`

**Interfaces:**
- Produces:
  - `CLASS_NAMES: list[str]` (25, com `Traffic Barrel`)
  - `write_data_yaml(root: Path, out: Path) -> Path` — escreve `path: <root absoluto>`, `train: train/images`, `val: valid/images`, `test: test/images`, `nc: 25`, `names: CLASS_NAMES`
  - `class_stats(root: Path) -> dict[str, dict[str, int]]` — conta caixas por classe por split lendo `*/labels/*.txt`
  - CLI: `python scripts/prepare_dataset.py --root data/rod [--download]` → baixa via `huggingface_hub.snapshot_download(repo_id="Abtinzandi/Obstacle-Detection-Dataset-YOLO", repo_type="dataset", local_dir=root)` quando `--download`, escreve `data/rod/data_cl.yaml` e `docs/assets/class_stats.csv`

- [ ] **Step 1: Testes que falham** (dataset falso em `tmp_path`, 2 labels):
  - `test_data_yaml_fixes_typo`: `"Traffic Barrel" in names` e `"Teraffic Barrel" not in names`, `nc == 25`.
  - `test_data_yaml_paths_absolute`.
  - `test_class_stats_counts`: label `0 .5 .5 .1 .1\n20 .5 .5 .1 .1` em train → `stats["Bike"]["train"] == 1`, `stats["Traffic Barrel"]["train"] == 1`.
- [ ] **Step 2:** FAIL → **Step 3:** implementar → **Step 4:** `pytest -v` PASS.
- [ ] **Step 5:** `git commit -m "feat: preparação do ROD-Dataset"`.

### Task 6: Notebook de treino no Colab (execução real)

**Files:** Create `notebooks/train_colab.ipynb`; artefatos em `docs/assets/training/` (`results.csv`, `results.png`, `confusion_matrix_normalized.png`, `PR_curve.png`, `test_metrics.json`, `val_batch0_pred.jpg`); `models/best.pt`

Células, em ordem:
1. GPU check (`!nvidia-smi`) e `pip install ultralytics huggingface_hub`.
2. Montar o Drive; `RUN_DIR=/content/drive/MyDrive/caminho-livre-cv/runs`.
3. Clonar o repo (ou subir `scripts/prepare_dataset.py`) e rodar `prepare_dataset.py --root /content/rod --download`.
4. Treino: `YOLO("yolo11n.pt").train(data=".../data_cl.yaml", epochs=50, imgsz=640, patience=15, seed=0, project=RUN_DIR, name="y11n", exist_ok=True)`; célula alternativa de retomada: `YOLO(f"{RUN_DIR}/y11n/weights/last.pt").train(resume=True)`.
5. Avaliação: `model.val(data=..., split="test")` → salvar `map50`, `map50_95`, precisão/recall médios e AP50 por classe em `test_metrics.json`.
6. Baixar `best.pt` + artefatos listados acima.

- [ ] **Step 1:** Escrever o notebook (sem saídas) e commitar.
- [ ] **Step 2:** Rodar no Colab do Breq (Chrome) com runtime T4. Esperado: treino conclui ou para por early stopping; `results.csv` com ≥1 linha por época.
- [ ] **Step 3:** Rodar avaliação no test; conferir que `test_metrics.json` tem 25 entradas por classe.
- [ ] **Step 4:** Copiar artefatos para `docs/assets/training/` e `best.pt` para `models/`. Commit: `git commit -m "feat: treino YOLO11n e métricas no split test"`.

### Task 7: Benchmark de FPS e comparação com baseline

**Files:** Create `scripts/benchmark_fps.py`, `scripts/compare_baseline.py`; saídas `docs/assets/fps.json`, `docs/assets/baseline/*.jpg`, `docs/assets/baseline/summary.md`

**Interfaces:**
- Consumes: `models/best.pt`, `data/rod/test/images`, `results_to_detections`, `render`, `blur_regions`.
- `benchmark_fps.py --weights models/best.pt --images data/rod/test/images --n 200` → descarta 10 de aquecimento; grava `{"hardware": platform.processor()/cpu_count, "imgsz": 640, "n": 200, "mean_ms":…, "fps":…}`.
- `compare_baseline.py --images <6 imagens do test escolhidas por conterem Traffic Cone, Dustbin, Electrical Pole, Pedestrian crosswalk, Stairs, Traffic Barrel> --out docs/assets/baseline` → para cada imagem, lado a lado `yolo11n.pt` (COCO) × `best.pt`, com pessoas desfocadas; `summary.md` com a contagem de detecções por modelo.

- [ ] **Step 1:** Implementar os dois scripts (scripts de medição; sem TDD, validação pela saída).
- [ ] **Step 2:** Rodar ambos. Esperado: `fps.json` existe com `fps > 0`; 6 imagens lado a lado; nenhum rosto nítido nas imagens (inspeção visual).
- [ ] **Step 3:** Commit `docs: FPS em CPU e baseline COCO vs fine-tuned`.

### Task 8: Demo em vídeo de SP + README + relatório + roteiro

**Files:** Create/Modify `README.md`, `docs/report/relatorio.md`, `docs/report/relatorio.pdf`, `docs/video/roteiro.md`, `docs/assets/demo/`

- [ ] **Step 1: Demo SP** (quando a equipe tiver o clipe): `python -m caminho_livre_cv.detect --source <clipe_sp.mp4> --save docs/assets/demo/demo_sp.mp4 --no-show`; extrair 3 frames para `docs/assets/demo/`. Sem clipe → usar vídeo de teste montado com imagens do split test e declarar isso.
- [ ] **Step 2: README**: ficha técnica (equipe + RMs, disciplina, dataset com citação e nota de licenças, modelo, hardware), setup do venv, `pytest`, como treinar (link do notebook), como rodar a CLI, tabela de resultados lida de `test_metrics.json` e `fps.json`, mapeamento de categorias, LGPD (blur por padrão, nenhum dado pessoal coletado), limitações (sem rampas, gap de domínio, veículos não localizados em relação à calçada), licença AGPL-3.0.
- [ ] **Step 3: Relatório** `relatorio.md` com as seções da rubrica: título, integrantes, resumo (**≤250 palavras**, verificar com contagem), introdução, objetivo (**≤50 palavras**, verificar), ferramentas, metodologia passo a passo, resultados (tabela de métricas, curvas, matriz de confusão, baseline, FPS, demo SP qualitativa), conclusão e limitações, referências. Converter para PDF (skill `document-skills:pdf` ou pandoc).
- [ ] **Step 4: Roteiro do vídeo** `roteiro.md`, 3–5 min, com tempos: problema → dataset → treino no Colab → métricas → demo ao vivo (webcam + clipe SP) → limitações → repositório.
- [ ] **Step 5:** Verificação final: `pytest -v` PASS; checar no `git ls-files` que não há `.env`, `data/` nem imagem com rosto; os números do relatório batem com os JSON. Commit `docs: README, relatório e roteiro`.
- [ ] **Step 6:** **Pedir confirmação ao Breq** antes de criar o repo público no GitHub (Breqsz) e dar push.
