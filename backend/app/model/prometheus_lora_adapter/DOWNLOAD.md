# LoRA Adapter Model Files

This directory should contain the fine-tuned LoRA adapter files for the Prometheus model.

## Required Files

The following files are needed for the full model to work:
- ✅ `adapter_config.json` (included in repo)
- ✅ `tokenizer_config.json` (included in repo)
- ✅ `tokenizer.json` (included in repo)
- ✅ `special_tokens_map.json` (included in repo)
- ❌ `adapter_model.safetensors` (~161 MB - **download required**)
- ❌ `tokenizer.model` (~500 KB - **download required**)

## Download Options

### Option 1: Download from Google Drive (Recommended)

Download the pre-trained LoRA adapter files:

**Google Drive Link**: [Coming Soon - Contact Repository Owner]

Files to download:
1. `adapter_model.safetensors` (161 MB)
2. `tokenizer.model` (500 KB)

Place them in this directory (`backend/app/model/prometheus_lora_adapter/`).

### Option 2: Train Your Own Model

You can train your own LoRA adapter using the included Colab notebook:

1. Open `Fine_Tune_Prometheus.ipynb` in Google Colab
2. Follow the instructions in the notebook
3. Training takes ~3 hours on a T4 GPU (free in Colab)
4. Download the generated files to this directory

**Training Dataset**: Already included in `services/ingest/data/training_dataset.jsonl` (1,000 examples)

### Option 3: Use Lightweight Mode (No Download Required)

The application works without these files using **Prometheus Light v1.0**:
- Pattern-based enhancement
- RAG with 811 expert guidelines
- ~80% quality of full model
- No GPU required
- Instant startup

The lightweight model is the default and requires no additional downloads!

## Verification

After placing the files, verify they're present:

```bash
ls -lh backend/app/model/prometheus_lora_adapter/

# You should see:
# adapter_config.json
# adapter_model.safetensors  ← Should be ~161 MB
# tokenizer.json
# tokenizer.model            ← Should be ~500 KB
# tokenizer_config.json
# special_tokens_map.json
```

## Model Details

- **Base Model**: mistralai/Mistral-7B-Instruct-v0.1
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: 1,000 prompt enhancement examples
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Training Time**: ~3 hours on T4 GPU
- **Quantization**: 8-bit (bitsandbytes)

## Hardware Requirements

To use the full fine-tuned model:
- **RAM**: 16GB+ recommended
- **GPU**: 8GB+ VRAM (or CPU with 32GB+ RAM)
- **Disk**: 15GB+ free space

If you don't meet these requirements, use Prometheus Light (no download needed).

## Troubleshooting

**Q: Can I use the app without downloading these files?**  
A: Yes! The lightweight model works perfectly without them.

**Q: How do I know if the model loaded correctly?**  
A: Check the `/health` endpoint. It will show which model is active.

**Q: The files are too large to download.**  
A: Use Prometheus Light or train your own model in Google Colab (free).

**Q: Where can I get help?**  
A: Open an issue in the GitHub repository.

---

**Last Updated**: November 16, 2025  
**Model Version**: 1.0.0
