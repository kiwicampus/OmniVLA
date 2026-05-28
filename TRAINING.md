# Entrenamiento OmniVLA

Esta es la unica ruta soportada de entrenamiento en este repo. La idea es que local y Vertex AI usen el mismo trainer y el mismo YAML:

- `vla-scripts/train_omnivla.py`: ejecuta el fine-tuning.
- `config_nav/train_omnivla.yaml`: define dataset, modelo, optimizacion, checkpoints y logging.
- `scripts/omnivla_training.sh`: unico helper local con subcomandos `setup`, `check`, `doctor`, `inspect`, `smoke` y `train`.
- `vla-scripts/inspect_training_dataset.py`: construye el dataset desde el mismo YAML y muestra un sample.
- `omnivla_training/vertex/submit_training_job.py`: sube el YAML y lanza el mismo trainer dentro del contenedor de Vertex.

Los scripts antiguos `vla-scripts/train_omnivla_dataset.py`, `vla-scripts/train_omnivla_single_dataset.py` y los configs MBRA/multi-dataset fueron eliminados para que no haya dos maneras incompatibles de entrenar.

## Setup rapido

En una maquina nueva:

```bash
bash scripts/omnivla_training.sh setup
conda activate omnivla
bash scripts/omnivla_training.sh doctor
```

El setup instala PyTorch 2.7.0 con CUDA 12.8 por defecto, para soportar GPUs
Blackwell/RTX 50 como la RTX 5090. Para un entorno legacy con CUDA 12.1:

```bash
TORCH_VERSION=2.2.0 \
TORCHVISION_VERSION=0.17.0 \
TORCHAUDIO_VERSION=2.2.0 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
bash scripts/omnivla_training.sh setup
```

El helper unico reemplaza los wrappers sueltos:

```bash
bash scripts/omnivla_training.sh check
bash scripts/omnivla_training.sh inspect
bash scripts/omnivla_training.sh smoke
bash scripts/omnivla_training.sh train
```

## Ruta local en 6 pasos

1. Crea el entorno:

```bash
bash scripts/omnivla_training.sh setup
conda activate omnivla
```

2. Edita `config_nav/train_omnivla.yaml`. Para un dataset nuevo cambia solo
   `dataset.name`, `dataset.root` o `dataset.repo_id`, y `dataset.action_key`.

3. Valida la maquina:

```bash
bash scripts/omnivla_training.sh doctor
```

4. Inspecciona un sample real:

```bash
bash scripts/omnivla_training.sh inspect
```

5. Corre un paso de entrenamiento:

```bash
bash scripts/omnivla_training.sh smoke
```

6. Lanza el run:

```bash
bash scripts/omnivla_training.sh train
```

## Ruta Vertex en 6 pasos

1. Crea un entorno solo para enviar jobs:

```bash
conda create -n omnivla-vertex-submit python=3.10 -y
conda activate omnivla-vertex-submit
pip install -r omnivla_training/requirements_vertex_submit.txt
```

2. Autentica Google Cloud:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <GCP_PROJECT_ID>
```

3. Define proyecto, buckets e imagen:

```bash
export VERTEX_PROJECT_ID=<GCP_PROJECT_ID>
export VERTEX_OMNIVLA_IMAGE_URI=gcr.io/$VERTEX_PROJECT_ID/omnivla-training:latest
export VERTEX_OMNIVLA_STAGING_BUCKET=gs://<STAGING_BUCKET>/omnivla
export VERTEX_OMNIVLA_TRAIN_BUCKET=gs://<TRAINING_BUCKET>/omnivla
```

4. Construye la imagen:

```bash
gcloud builds submit \
  --config omnivla_training/cloudbuild.yaml \
  --substitutions=_IMAGE_URI=$VERTEX_OMNIVLA_IMAGE_URI \
  .
```

5. Lanza un smoke test:

```bash
python omnivla_training/vertex/submit_training_job.py \
  --config-path config_nav/train_omnivla.yaml \
  --smoke-test \
  --sync
```

6. Lanza el run completo:

```bash
python omnivla_training/vertex/submit_training_job.py \
  --config-path config_nav/train_omnivla.yaml \
  --steps 12000 \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --save-freq 500
```

No pongas secretos en archivos. Para Hugging Face o W&B, usa variables de
entorno (`HF_TOKEN`, `WANDB_API_KEY`) o Secret Manager.

## Seguridad del repo

Este repo ignora por defecto:

- `.env`, `.env.*`, `.claude/`, claves `*.pem`, `*.key`, `*.p12`
- `runs/`, `checkpoints/`, `wandb/`, `lerobot_models/`
- datasets locales, caches y manifests generados en `config_nav/episode_manifests/*.yaml`

Los archivos de Vertex no contienen project IDs, buckets ni nombres de secretos
reales. El submitter exige que los definas por variables de entorno antes de
lanzar un job.

## Que se entrena

El fine-tuning parte de `model.vla_path`, por defecto `NHirose/omnivla-original`.

Esto significa que el checkpoint base por defecto es OmniVLA, no OpenVLA puro. Si quieres una ablacion desde OpenVLA, cambia explicitamente:

```yaml
model:
  vla_path: openvla/openvla-7b
```

La ruta recomendada para navegacion es mantener `NHirose/omnivla-original`, porque ya trae preentrenamiento de navegacion. OpenVLA puro es un punto de partida mas general y normalmente requiere mas datos para llegar al mismo comportamiento.

Se entrenan tres cosas:

- LoRA adapters insertados en las capas lineales del VLA. Los pesos base del modelo quedan congelados y los parametros nuevos de LoRA capturan la adaptacion al dominio.
- `pose_projector`: MLP que proyecta `goal_pose` al espacio del LLM cuando hay pose de objetivo disponible.
- `action_head`: MLP que lee hidden states de los tokens de accion y predice acciones continuas.

No se entrena MBRA. Tampoco se entrena una mezcla de datasets. La ruta canonica toma un dataset LeRobot por run.

## Que YAML se usa

OmniVLA usa solo este YAML:

```text
config_nav/train_omnivla.yaml
```

El trainer lee estas secciones:

- `dataset`
- `model`
- `training`
- `checkpoint`
- `logging`

No lee configs de ACT/LeRobot con secciones como `policy`, `job`, `optimizer`, `num_workers` top-level o `wandb` top-level. Esos YAMLs son de otra pipeline. Si pegas un YAML de ACT en este trainer, la mayoria de campos no tienen significado para OmniVLA.

Los otros YAMLs que quedan tienen otro proposito:

- `omnivla_training/cloudbuild.yaml`: build del contenedor de Vertex.
- `prismatic/vla/datasets/data_config.yaml`: config heredada del paquete Prismatic/RLDS, no es la ruta canonica de entrenamiento.

No se versionan manifests de entrenamientos viejos. Si un dataset nuevo necesita
filtrar episodios corruptos, genera un manifest nuevo con
`omnivla_training/training_scripts/curate_episodes.py` y apunta
`dataset.episodes_file` a ese archivo.

## Que predice el modelo

El dataset entrega acciones crudas de robot, normalmente velocidad lineal y velocidad angular. El adaptador las convierte a un chunk de `NUM_ACTIONS_CHUNK` waypoints. En esta repo el chunk esperado es de 8 pasos y cada accion tiene 4 dimensiones:

```text
[x_normalized, y_normalized, cos(theta), sin(theta)]
```

`x` e `y` se normalizan con `dataset.metric_waypoint_spacing`. Por defecto `0.1` significa que una unidad equivale a 10 cm.

El modelo recibe:

- imagen actual
- imagen objetivo
- prompt de lenguaje: `What action should the robot take to <instruction>?`
- opcionalmente `goal_pose`, si el dataset trae waypoints GPS/UTM suficientes

El modelo produce hidden states para los tokens de accion. `action_head` convierte esos hidden states en acciones continuas y la perdida compara contra los waypoints del dataset.

## Perdidas

La perdida principal es MSE entre accion predicha y accion objetivo:

```text
loss = action_loss
     + smoothness_loss_weight * smoothness_loss
     + object_loss_weight * object_loss
```

- `action_loss`: MSE del chunk completo de acciones.
- `smoothness_loss`: penaliza cambios bruscos entre acciones consecutivas predichas. Esta activo si `training.smoothness_loss_weight > 0`.
- `object_loss`: supervision opcional del ultimo waypoint contra `obj_pose_norm`. Solo se aplica cuando el sample tiene pose valida y el objetivo cae dentro del horizonte del chunk.

La configuracion canonica deja `object_loss_weight: 0.0` porque no todos los datasets tienen pose confiable, y usa `smoothness_loss_weight: 0.02`.

## Contrato del dataset

El trainer espera un dataset en formato LeRobot. Debe poder construirse con `LeRobotDataset("", root, episodes=[...], video_backend="pyav")`.

Campos esperados:

- `observation.image.main` o equivalente parseado como imagen RGB principal.
- `action`, con al menos velocidad lineal y angular.
- `episode_index` y `frame_index`.
- `language_instruction`, opcional. Si falta, se usa `dataset.default_language_instruction`.
- `observation.state.waypoints`, opcional. Si existe, se usa para construir `goal_pose`.

La imagen objetivo es el ultimo frame del episodio. Esto hace que el entrenamiento sea goal-conditioned: el modelo aprende a moverse desde el frame actual hacia el estado visual final del episodio.

Si hay waypoints validos, el sample usa `modality_id_with_pose` (por defecto `8`). Si no hay pose, usa `modality_id_without_pose` (por defecto `6`) y entrena con imagen objetivo + lenguaje.

## Episodios limpios

`dataset.episodes_file` apunta a un manifest compacto de episodios. El formato recomendado es YAML por rangos:

```yaml
kind: omnivla_episode_manifest
schema_version: 1
repo_id: mi_org/mi_dataset
video_key: observation.image.main
summary:
  total_episode_count: 12
  clean_episode_count: 10
  excluded_episode_count: 2
clean_episode_ranges:
  - [0, 4]
  - [7, 11]
excluded_episode_ranges:
  - [5, 6]
```

El trainer tambien sigue aceptando formatos simples para casos pequenos:

```json
[0, 1, 2, 3]
```

```json
{"clean_episodes": [0, 1, 2, 3]}
```

Si se define `episodes_file`, el trainer usa solo esos episodios. Si ademas `val_ratio > 0`, separa una fraccion como holdout. Importante: hoy ese holdout no se evalua durante entrenamiento; solo queda fuera de optimizacion. Para entrenar con todo el subset limpio, deja `val_ratio: 0.0`.

## Curar episodios corruptos

La unica herramienta soportada para curar episodios es:

```bash
python omnivla_training/training_scripts/curate_episodes.py
```

Hace tres cosas en una sola pasada:

- lee `meta/episodes/**/*.parquet`
- aplica exclusiones manuales de episodios, rangos o archivos de video
- opcionalmente abre los MP4 y excluye episodios cuyo `to_timestamp` cae fuera de la duracion real del video

Ejemplo para generar un manifest nuevo desde un dataset remoto:

```bash
python omnivla_training/training_scripts/curate_episodes.py \
  --repo-id mi_org/mi_dataset \
  --check-video-duration \
  --overflow-threshold-s 0.5 \
  --output config_nav/episode_manifests/mi_dataset.yaml
```

Ejemplo con un dataset local y exclusiones manuales:

```bash
python omnivla_training/training_scripts/curate_episodes.py \
  --root /mnt/datasets/mi_dataset \
  --exclude-episode-range 720 829 \
  --exclude-video chunk-000:33 \
  --output config_nav/episode_manifests/mi_dataset.yaml
```

Para una pasada rapida que no descargue/abra videos, omite `--check-video-duration`; en ese modo solo se aplican exclusiones manuales. Para datasets remotos, `--check-video-duration` puede descargar muchos MP4, asi que conviene correrlo una vez, revisar el manifest y versionar solo el YAML compacto.

## Config canonica

Edita `config_nav/train_omnivla.yaml`. Ese archivo esta documentado campo por campo con comentarios, opciones validas y presets comentados para casos comunes como `robotcom/single_waypoints`, dataset local, curacion de episodios y ablacion desde OpenVLA.

Las secciones efectivas son:

```yaml
dataset:
  root: null
  repo_id: robotcom/single_waypoints

model:
  vla_path: NHirose/omnivla-original

training:
  batch_size: 1
  grad_accumulation_steps: 8

checkpoint:
  run_root_dir: runs/omnivla_train

logging:
  wandb:
    enable: false
```

Usa una de estas dos formas para el dataset:

- `dataset.root: /ruta/local/al/snapshot`
- `dataset.root: null` y `dataset.repo_id: owner/dataset`

Para datasets privados en Hugging Face, exporta `HF_TOKEN` antes de entrenar.

No agregues secciones de ACT como `policy`, `job`, `optimizer` o `wandb` top-level: pertenecen a otra pipeline y este trainer no las lee.

## Inspeccionar antes de entrenar

Siempre inspecciona el dataset con el mismo YAML que vas a entrenar:

```bash
bash scripts/omnivla_training.sh inspect
```

Esto descarga o resuelve el dataset, aplica el subset de episodios, carga processor/tokenizer y muestra shapes de un sample. Si esto falla, el entrenamiento tambien va a fallar.

Para inspeccionar otro sample:

```bash
bash scripts/omnivla_training.sh inspect --sample-idx 100
```

## Smoke test local

Antes de intentar entrenar, corre el doctor:

```bash
bash scripts/omnivla_training.sh doctor
```

Ese comando revisa config, manifest, imports principales, `torchrun`, `lerobot` y CUDA. Si quieres revisar solo archivos/config sin exigir GPU:

```bash
bash scripts/omnivla_training.sh doctor --static-only
```

Ejecuta un paso de optimizacion:

```bash
bash scripts/omnivla_training.sh smoke
```

El smoke test necesita CUDA. Si no hay GPU, el trainer falla temprano con `CUDA is required for OmniVLA training`.

Para RTX 50 / Blackwell, usa PyTorch con CUDA 12.8 o mas nuevo. El setup
canonico ya usa `torch==2.7.0` con `cu128` por defecto. Wheels antiguos como
`torch==2.2.0+cu121` pueden decir que CUDA existe, pero fallan al ejecutar
kernels en `sm_120`.

Los tests unitarios estaticos del repo se corren con:

```bash
bash scripts/omnivla_training.sh check
```

## Entrenamiento local completo

Un GPU:

```bash
bash scripts/omnivla_training.sh train
```

Varios GPUs en la misma maquina:

```bash
NPROC_PER_NODE=2 bash scripts/omnivla_training.sh train
```

El batch efectivo es:

```text
training.batch_size * numero_de_gpus * training.grad_accumulation_steps
```

Con los defaults `1 * 1 * 8 = 8`.

## Logging con W&B

Activa W&B en el YAML:

```yaml
logging:
  wandb:
    enable: true
    entity: tu_entity
    project: omnivla-training
```

Luego:

```bash
export WANDB_API_KEY=...
```

Si W&B esta desactivado, el trainer imprime metricas en stdout.

## Checkpoints

Los checkpoints se guardan bajo `checkpoint.run_root_dir` con un nombre derivado de modelo, dataset, batch y learning rate.

Cada checkpoint contiene:

- `lora_adapter/`: pesos LoRA y `adapter_config.json`.
- `action_head--{step}_checkpoint.pt`: head continuo.
- `pose_projector--{step}_checkpoint.pt`: projector de pose.
- `resolved_config.yaml`: config final, incluyendo el modelo base pedido y el path resuelto.
- archivos del tokenizer/processor.

Por defecto `merge_lora_during_training: false`. Esto evita un merge caro durante training. La inferencia fusiona LoRA al cargar.

## Inferencia desde un checkpoint

Configura `inference/run_omnivla.py`:

```python
class InferenceConfig:
    base_model_path: Optional[str] = None
    checkpoint_dir: str = "./checkpoints/<checkpoint_dir>"
    resume_step: int = 23500
```

Con `base_model_path = None`, inferencia busca el modelo base en `resolved_config.yaml` y luego en `lora_adapter/adapter_config.json`.

## Vertex AI

Vertex AI no tiene un trainer distinto. El submitter solo prepara el job y el wrapper ejecuta:

```bash
python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node N \
  /app/vla-scripts/train_omnivla.py \
  --config /tmp/omnivla_output/runtime_config.yaml
```

### Entorno de submit

Usa un entorno separado para enviar jobs; no necesitas instalar CUDA aqui:

```bash
conda create -n omnivla-vertex-submit python=3.10 -y
conda activate omnivla-vertex-submit
pip install -r omnivla_training/requirements_vertex_submit.txt

gcloud auth login
gcloud auth application-default login
gcloud config set project <GCP_PROJECT_ID>
```

### Variables obligatorias

El repo no trae project IDs, buckets ni nombres de secretos reales. Define todo
por variables de entorno:

```bash
export VERTEX_PROJECT_ID=<GCP_PROJECT_ID>
export VERTEX_REGION=us-central1
export VERTEX_OMNIVLA_IMAGE_URI=gcr.io/$VERTEX_PROJECT_ID/omnivla-training:latest
export VERTEX_OMNIVLA_STAGING_BUCKET=gs://<STAGING_BUCKET>/omnivla
export VERTEX_OMNIVLA_TRAIN_BUCKET=gs://<TRAINING_BUCKET>/omnivla
export VERTEX_OMNIVLA_MACHINE_TYPE=a2-highgpu-1g
export VERTEX_OMNIVLA_ACCELERATOR_TYPE=NVIDIA_TESLA_A100
export VERTEX_OMNIVLA_ACCELERATOR_COUNT=1
export VERTEX_OMNIVLA_BOOT_DISK_SIZE_GB=500
```

`submit_training_job.py` falla temprano si falta alguna de estas variables
obligatorias:

- `VERTEX_PROJECT_ID`
- `VERTEX_OMNIVLA_IMAGE_URI`
- `VERTEX_OMNIVLA_STAGING_BUCKET`
- `VERTEX_OMNIVLA_TRAIN_BUCKET`

### Tokens y secretos

No guardes tokens en el repo. Para datasets/modelos privados, usa una de estas
dos rutas:

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
```

o guarda los valores en Secret Manager y pasa solo los nombres:

```bash
export VERTEX_HF_TOKEN_SECRET=<secret-name>
export VERTEX_HF_TOKEN_SECRET_VERSION=latest
export VERTEX_WANDB_SECRET=<secret-name>
export VERTEX_WANDB_SECRET_VERSION=latest
```

Si no usas W&B, deja `logging.wandb.enable: false` en el YAML.

Para `us-east4`, usa A100 80GB:

```bash
export VERTEX_REGION=us-east4
export VERTEX_OMNIVLA_MACHINE_TYPE=a2-ultragpu-1g
export VERTEX_OMNIVLA_ACCELERATOR_TYPE=NVIDIA_A100_80GB
export VERTEX_OMNIVLA_ACCELERATOR_COUNT=1
```

### Construir imagen

```bash
gcloud builds submit --config omnivla_training/cloudbuild.yaml .
```

La imagen usa PyTorch 2.7.0 + CUDA 12.8 por defecto. Si tu proyecto necesita
otra version, pasala al build:

```bash
gcloud builds submit \
  --config omnivla_training/cloudbuild.yaml \
  --substitutions=_IMAGE_URI=$VERTEX_OMNIVLA_IMAGE_URI,_TORCH_VERSION=2.7.0,_TORCHVISION_VERSION=0.22.0,_TORCHAUDIO_VERSION=2.7.0,_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  .
```

### Smoke test en Vertex

```bash
python omnivla_training/vertex/submit_training_job.py \
  --config-path config_nav/train_omnivla.yaml \
  --smoke-test \
  --sync
```

### Run completo en Vertex

```bash
python omnivla_training/vertex/submit_training_job.py \
  --config-path config_nav/train_omnivla.yaml \
  --steps 12000 \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --save-freq 500
```

El wrapper descarga datasets/modelos desde `gs://` si los pasas como override, escribe outputs en disco local y sincroniza periodicamente a `VERTEX_OMNIVLA_TRAIN_BUCKET`.

### Checklist Vertex

Antes de lanzar un run largo:

1. `bash scripts/omnivla_training.sh check`
2. Edita `config_nav/train_omnivla.yaml`.
3. Define las variables `VERTEX_*`.
4. Construye la imagen con `gcloud builds submit`.
5. Lanza primero `--smoke-test --sync`.
6. Lanza el run completo cuando el smoke pase.

## Cambiar de dataset

Para entrenar otro dataset, cambia solo el YAML:

```yaml
dataset:
  name: mi_dataset
  root: /mnt/datasets/mi_dataset
  repo_id: null
  episodes_file: null
```

o:

```yaml
dataset:
  name: mi_dataset
  root: null
  repo_id: mi_org/mi_dataset
  revision: main
  episodes_file: null
```

Si necesitas excluir episodios corruptos, genera un manifest para ese dataset y
cambia `episodes_file` a ese path. Despues corre inspeccion y smoke test antes
del run largo.

## Ejemplo: `robotcom/single_waypoints`

El YAML de ACT que contiene `policy.type: act`, `chunk_size: 100`, `vision_backbone: resnet18`, etc. no sirve directamente para OmniVLA. Para entrenar OmniVLA con ese dataset, usa el preset comentado al final de `config_nav/train_omnivla.yaml` o edita el bloque `dataset` asi:

```yaml
dataset:
  name: single_waypoints
  root: null
  repo_id: robotcom/single_waypoints
  revision: main
  local_files_only: false
  video_backend: pyav
  tolerance_s: 0.001
  max_decode_retries: 8
  episodes_file: null
  val_ratio: 0.0
  split_seed: 7
  num_workers: 4
  context_size: 0
  action_spacing: 1
  action_key: observation.state
  aux_image_size: [96, 96]
  metric_waypoint_spacing: 0.1
  default_language_instruction: reach the goal image
  modality_id_without_pose: 6
  modality_id_with_pose: 8
```

Mantén el bloque `model` como:

```yaml
model:
  vla_path: NHirose/omnivla-original
  num_images_in_input: 2
  attn_implementation: sdpa
  use_lora: true
  lora_rank: 32
  lora_dropout: 0.0
```

Para una prueba pequena, baja pasos y workers:

```yaml
training:
  batch_size: 1
  grad_accumulation_steps: 8
  max_steps: 1000
```

Luego:

```bash
bash scripts/omnivla_training.sh inspect
bash scripts/omnivla_training.sh smoke
```

Importante: ese dataset puede ser tiny, pero OmniVLA sigue siendo un modelo 7B. El comentario del YAML de ACT sobre 8 GB VRAM aplica a ACT/ResNet18, no a OmniVLA. Para OmniVLA espera una GPU grande tipo A100/L4 grande o Vertex.

Tambien es importante `action_key: observation.state` para `single_waypoints`: en ese dataset el campo `action` esta definido como `t1_latitude/t1_longitude`, mientras que `observation.state` contiene `twist.twist.linear.x` y `twist.twist.angular.z`. El trainer necesita velocidades `[linear, angular]` para convertirlas al chunk de waypoints OmniVLA.

## Ajustes importantes

- `training.vla_learning_rate`: LR para LoRA. Mantener bajo si partes de `NHirose/omnivla-original`.
- `training.head_learning_rate`: LR para `action_head` y `pose_projector`, que empiezan desde cero.
- `training.max_steps`: numero de pasos de optimizador, no numero de samples.
- `checkpoint.save_freq`: frecuencia de checkpoints por pasos de optimizador.
- `dataset.num_workers`: subir en maquinas grandes; bajar si hay problemas de memoria o video decoding.
- `dataset.context_size`: cuantos frames historicos se cargan para tensores auxiliares `cur_image`.
- `dataset.action_spacing`: separacion entre acciones futuras dentro del chunk.
- `dataset.tolerance_s`: tolerancia de LeRobot/video timestamps. Si el decoder salta errores, revisa o cura episodios.

## Problemas comunes

`Unable to import LeRobot dataset classes`

Instala LeRobot compatible:

```bash
pip install --no-deps lerobot==0.4.3
```

`CUDA is required for OmniVLA training`

El trainer no tiene modo CPU. Usa una GPU local o Vertex.

Errores de video decoding o timestamp tolerance

Primero baja `dataset.num_workers` para diagnosticar. Si el episodio esta corrupto, genera un nuevo manifest con `omnivla_training/training_scripts/curate_episodes.py` y apunta `dataset.episodes_file` a ese YAML.

OOM

Baja `training.batch_size` a `1`, sube `grad_accumulation_steps` para mantener batch efectivo, reduce `dataset.num_workers`, o usa una GPU con mas memoria. Para 7B, A100 es el target practico.

W&B pide login durante un run no interactivo

Deja `logging.wandb.enable: false` o exporta `WANDB_API_KEY`.

## Estado actual de validacion

El trainer registra metricas de entrenamiento: `loss`, `action_loss`, `action_l1`, `final_xy_l1`, `smoothness_loss`, `object_loss`, `grad_norm` y learning rates. Todavia no ejecuta evaluacion periodica sobre holdout. Si necesitas comparar checkpoints, usa un split fijo de episodios y evalua offline con inferencia o un script de rollout.
