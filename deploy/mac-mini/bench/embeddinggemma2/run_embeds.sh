#!/bin/zsh
# Embeds the exported corpus once per configuration, each in a fresh process,
# with the production inference service idle and Gemma resident as in
# production, then runs the PyTorch reference.
#
# Layout under this directory: data/corpus.jsonl, data/queries.json and
# data/images/ (corpus-images.tar unpacked); models/ holds
#   embeddinggemma-2-bf16   mlx-community, revision 1a4ffdd
#   embeddinggemma-2-mxfp8  mlx-community, revision 0536fa4
#   embeddinggemma-2-hf     google/embeddinggemma-2, revision 914f7f8 (reference)
# fetched with `uv run hf download <repo> --revision <rev> --local-dir models/<name>`.
cd "$(dirname "$0")"
# Stop at the first failure, so RUN_DONE means every output was regenerated.
setopt err_exit pipe_fail
export PATH=$HOME/.proto/shims:$HOME/.proto/bin:$PATH
log=out/run.log
mkdir -p out
state() {
  echo "$(date '+%F %T') $1 health=$(curl -s 127.0.0.1:9500/health | sed -E 's/.*"loaded":(\[[^]]*\]).*/\1/')" \
    "$(sysctl -n vm.swapusage | awk '{print "swap_used="$6}')" \
    "free=$(memory_pressure | awk '/free percentage/{print $5}')" >> $log
}
eg2() { state "$1"; uv run python embed_corpus.py eg2 "${@:2}" --out "out/$1.json" >> $log 2>&1; }

state start
# The service's own nemotron_vl_mlx and virtualenv, at the service's settings.
state nemotron; ~/knowva-inference/app/inference/.venv/bin/python embed_corpus.py nemotron --out out/nemotron.json >> $log 2>&1
eg2 eg2-bf16-280 --model models/embeddinggemma-2-bf16 --budget 280
eg2 eg2-bf16-1120 --model models/embeddinggemma-2-bf16 --budget 1120
eg2 eg2-mxfp8-280 --model models/embeddinggemma-2-mxfp8 --budget 280
eg2 eg2-bf16-280-noaudio --model models/embeddinggemma-2-bf16 --budget 280 --no-audio
eg2 eg2-mxfp8-280-noaudio --model models/embeddinggemma-2-mxfp8 --budget 280 --no-audio
state reference; uv run --group reference python reference.py models/embeddinggemma-2-hf out/reference.json >> $log 2>&1
state end
echo RUN_DONE >> $log
