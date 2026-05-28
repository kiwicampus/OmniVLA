# OmniVLA

OmniVLA es un modelo Vision-Language-Action para navegacion robotica. Este fork deja una unica ruta estandar para fine-tuning:

- Trainer canonico: `vla-scripts/train_omnivla.py`
- Config canonica: `config_nav/train_omnivla.yaml`
- CLI local: `scripts/omnivla_training.sh`
- Curacion de episodios: `omnivla_training/training_scripts/curate_episodes.py`
- Inspeccion de dataset: `vla-scripts/inspect_training_dataset.py`
- Submit opcional a Vertex AI: `omnivla_training/vertex/submit_training_job.py`
- Guia completa: `TRAINING.md`

Los scripts historicos de entrenamiento multi-dataset/MBRA fueron removidos para evitar rutas divergentes. El codigo de inferencia se mantiene en `inference/run_omnivla.py`.

## Instalacion

Para replicar este repo en otro PC:

```bash
bash scripts/omnivla_training.sh setup
conda activate omnivla
bash scripts/omnivla_training.sh doctor
```

El setup usa PyTorch CUDA 12.8 por defecto para soportar GPUs Blackwell/RTX 50
como la RTX 5090. Para GPUs antiguas o drivers viejos puedes fijar otra rueda
con `TORCH_VERSION`, `TORCHVISION_VERSION`, `TORCHAUDIO_VERSION` y
`TORCH_INDEX_URL`.

Consulta `TRAINING.md` para el paso a paso completo.

## Entrenamiento rapido

`config_nav/train_omnivla.yaml` viene listo para `robotcom/single_waypoints` como smoke dataset publico y esta comentado campo por campo. Para un dataset real, cambia solo el bloque `dataset`. Luego inspecciona un ejemplo:

```bash
bash scripts/omnivla_training.sh inspect
```

Smoke test de un paso:

```bash
bash scripts/omnivla_training.sh smoke
```

Antes del smoke test, puedes revisar si la máquina realmente está lista:

```bash
bash scripts/omnivla_training.sh doctor
```

Los tests estáticos que no requieren GPU se corren con:

```bash
bash scripts/omnivla_training.sh check
```

Entrenamiento completo:

```bash
bash scripts/omnivla_training.sh train
```

Todo lo demas, incluyendo que se entrena, como se preparan acciones, checkpoints, Vertex AI, W&B y troubleshooting, esta documentado en `TRAINING.md`.

## Vertex AI rapido

Define tu proyecto, buckets e imagen por variables de entorno; el repo no trae
IDs internos ni secretos:

```bash
export VERTEX_PROJECT_ID=<GCP_PROJECT_ID>
export VERTEX_OMNIVLA_IMAGE_URI=gcr.io/$VERTEX_PROJECT_ID/omnivla-training:latest
export VERTEX_OMNIVLA_STAGING_BUCKET=gs://<STAGING_BUCKET>/omnivla
export VERTEX_OMNIVLA_TRAIN_BUCKET=gs://<TRAINING_BUCKET>/omnivla

gcloud builds submit --config omnivla_training/cloudbuild.yaml .
python omnivla_training/vertex/submit_training_job.py --smoke-test --sync
```

Para datasets privados, exporta `HF_TOKEN` localmente o usa Secret Manager; no
lo escribas en archivos del repo.

## Inferencia con checkpoints fine-tuned

Los checkpoints producidos por `vla-scripts/train_omnivla.py` guardan:

- `lora_adapter/`
- `action_head--{step}_checkpoint.pt`
- `pose_projector--{step}_checkpoint.pt`
- `resolved_config.yaml`
- tokenizer/processor files

`inference/run_omnivla.py` puede leer el modelo base desde `resolved_config.yaml` o desde `lora_adapter/adapter_config.json`, asi que normalmente basta con configurar:

```python
class InferenceConfig:
    base_model_path: Optional[str] = None
    checkpoint_dir: str = "./checkpoints/<checkpoint_dir>"
    resume_step: int = 23500
```

## Checkpoints originales

Para usar los modelos publicados del paper:

```bash
git clone https://huggingface.co/NHirose/omnivla-original
git clone https://huggingface.co/NHirose/omnivla-original-balance
git clone https://huggingface.co/NHirose/omnivla-finetuned-cast
python inference/run_omnivla.py
```

## Cita

```bibtex
@misc{hirose2025omnivla,
      title={OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation},
      author={Noriaki Hirose and Catherine Glossop and Dhruv Shah and Sergey Levine},
      year={2025},
      eprint={2509.19480},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.19480},
}
```
