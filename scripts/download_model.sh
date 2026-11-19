#!/usr/bin/env sh
# Downloads OpenAI's GPT-2 weights and tokenizer files from Hugging Face.
#   sh scripts/download_model.sh            # gpt2 (124M parameters, ~550 MB)
#   sh scripts/download_model.sh gpt2-medium # 355M; also gpt2-large, gpt2-xl
set -eu
NAME="${1:-gpt2}"
DEST="models/${NAME}"
BASE="https://huggingface.co/openai-community/${NAME}/resolve/main"
mkdir -p "$DEST"
for f in model.safetensors vocab.json merges.txt; do
  if [ -s "$DEST/$f" ]; then
    echo "already have $DEST/$f"
  else
    echo "downloading $f ..."
    curl -fL --retry 3 -o "$DEST/$f.part" "$BASE/$f"
    mv "$DEST/$f.part" "$DEST/$f"
  fi
done
echo "done: $DEST"
