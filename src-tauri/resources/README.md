# Translation Model Resources

Place your Opus-MT ONNX model files here. Required files:

- `encoder_model.onnx` — The encoder model
- `decoder_model.onnx` (or `decoder_model_merged.onnx`) — The decoder model
- `tokenizer.json` — HuggingFace tokenizer configuration

## Downloading Models

For Japanese → English:
```bash
# Install optimum CLI
pip install optimum[exporters]

# Export Opus-MT model to ONNX
optimum-cli export onnx --model Helsinki-NLP/opus-mt-ja-en ./ja-en/
```

Then copy the `.onnx` and `tokenizer.json` files into this directory.

## Supported Language Pairs

| Source | Target | HuggingFace Model |
|--------|--------|-------------------|
| Japanese | English | `Helsinki-NLP/opus-mt-ja-en` |
| Chinese | English | `Helsinki-NLP/opus-mt-zh-en` |
| Korean | English | `Helsinki-NLP/opus-mt-ko-en` |
