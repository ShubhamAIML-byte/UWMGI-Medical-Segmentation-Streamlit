---
title: Medical Image Segmentation Gradio App
emoji: 🏥🩺
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 3.37.0
app_file: app.py
pinned: false
---

# Medical Image Segmentation Gradio App

For the Gradio app we've removed the dependency on pytorch-lightning otherwise used in the project.
The `load_lightning_SD_to_Usual_SD.ipynb` notebook contains the steps used to convert pytorch-lightning checkpoint to a regular model checkpoint. This was mainly done to reduce the file size (977 MB --> 244 MB).

You can download the original saved checkpoint from over here: [wandb artifact](https://wandb.ai/veb-101/UM_medical_segmentation/artifacts/model/model-jsr2fn8v/v0/files)

Or via Python:

```python
import wandb
run = wandb.init()
artifact = run.use_artifact('veb-101/UM_medical_segmentation/model-jsr2fn8v:v0', type='model')
artifact_dir = artifact.download()
```
