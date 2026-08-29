# Qwen3-VL on 8xH200

**Status: User-executed external verification.** This protocol is intended for
the user's separate single-node machine with eight H200 GPUs. It was not run on
the local development machine and must not be counted as passed until its JSON
response and logs are reviewed.

The launch shape follows the official Qwen3-VL SGLang example and changes
tensor parallelism to eight GPUs. The current official SGLang installation
documentation lists Python 3.10+ and release `v0.5.15`. Re-check compatibility
when changing the Qwen or SGLang revision:

- https://github.com/QwenLM/Qwen3-VL#deployment
- https://docs.sglang.io/docs/get-started/install

## Prerequisites

- One Linux node with eight visible NVIDIA H200 GPUs and a working CUDA 12/13
  driver stack.
- Sufficient local disk for the selected Qwen3-VL checkpoint and Hugging Face
  cache.
- The HAT4D repository and its base annotation environment.
- Loopback ports `22002`, `8020`, and optionally `8897` available.

Create a dedicated server environment. Pinning the release prevents a later
SGLang CLI change from silently altering this protocol:

```bash
conda create -n qwen3vl-sglang python=3.11 -y
conda activate qwen3vl-sglang
python -m pip install --upgrade pip uv
uv pip install 'sglang==0.5.15'
```

## Download Qwen

The `--model-path` value must be either a valid Hugging Face repository ID or
an existing local checkpoint directory containing `config.json` and all weight
shards. Paths such as `/absolute/path/...` and `/path/to/...` are documentation
placeholders, not usable model paths.

Keep the large checkpoint outside the HAT4D repository. This example chooses a
real default under `$HOME`; override it with a high-capacity local disk when
needed:

```bash
export QWEN_MODEL_DIR="$HOME/models/Qwen3-VL-235B-A22B-Instruct"
export HF_ENDPOINT=https://hf-mirror.com

wget -O /tmp/hfd.sh https://hf-mirror.com/hfd/hfd.sh
chmod a+x /tmp/hfd.sh
/tmp/hfd.sh Qwen/Qwen3-VL-235B-A22B-Instruct \
  --local-dir "$QWEN_MODEL_DIR"
```

`hfd.sh` uses `aria2c` by default. Install `aria2`, or append `--tool wget` to
the last command when `aria2c` is unavailable. Verify the download before
starting SGLang:

```bash
test -s "$QWEN_MODEL_DIR/config.json"
test -s "$QWEN_MODEL_DIR/model.safetensors.index.json"
SHARD_COUNT=$(find "$QWEN_MODEL_DIR" -maxdepth 1 \
  -name 'model-*-of-00096.safetensors' | wc -l)
test "$SHARD_COUNT" -eq 96
du -sh "$QWEN_MODEL_DIR"
```

## Launch Qwen

```bash
export QWEN_MODEL_PATH="$QWEN_MODEL_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
  --model-path "$QWEN_MODEL_PATH" \
  --host 127.0.0.1 \
  --port 22002 \
  --tp 8 \
  >/tmp/hat4d_qwen3vl_sglang.log 2>&1 &
QWEN_PID=$!
echo "$QWEN_PID" >/tmp/hat4d_qwen3vl_sglang.pid
echo "$QWEN_PID"
```

Do not continue until the model endpoint responds. Large checkpoints may take
several minutes to load:

```bash
until curl --fail --silent http://127.0.0.1:22002/v1/models \
  >/tmp/hat4d_qwen3vl_models.json; do
  if ! kill -0 "$QWEN_PID" 2>/dev/null; then
    tail -n 200 /tmp/hat4d_qwen3vl_sglang.log
    exit 1
  fi
  sleep 5
done
cat /tmp/hat4d_qwen3vl_models.json
```

Set `QWEN_SERVED_MODEL` to the exact `id` returned by `/v1/models`. For a model
server started from a local directory, this ID may be the local path. The next
section reads it from the endpoint again because shell variables do not carry
over to a second terminal.

If the model server exits, inspect the log before continuing:

```bash
tail -n 200 /tmp/hat4d_qwen3vl_sglang.log
```

## Launch HAT-4D VLM Backend

In a second shell, activate the HAT-4D annotation environment and create the
small test frames:

```bash
conda activate hat4d-annotation
cd /path/to/HAT4D
python annotation_tool/scripts/create_demo_data.py \
  /tmp/hat4d_qwen3vl_demo --force

curl --fail --silent http://127.0.0.1:22002/v1/models \
  >/tmp/hat4d_qwen3vl_models.json
export QWEN_SERVED_MODEL=$(
  python -c 'import json; print(json.load(open("/tmp/hat4d_qwen3vl_models.json"))["data"][0]["id"])'
)
test -n "$QWEN_SERVED_MODEL"

export OPENAI_API_KEY=EMPTY
python annotation_tool/viewer_plugins/vlm_assistor/backend_server.py \
  --host 127.0.0.1 \
  --port 8020 \
  --openai-base-url http://127.0.0.1:22002/v1 \
  --openai-model "$QWEN_SERVED_MODEL" \
  --default-max-frames 3 \
  --image-max-side 512 \
  --timeout 900 \
  >/tmp/hat4d_qwen3vl_backend.log 2>&1 &
VLM_PID=$!
echo "$VLM_PID" >/tmp/hat4d_qwen3vl_backend.pid

curl --fail http://127.0.0.1:8020/health
```

The health response must be:

```json
{"status": "ok", "service": "vlm_backend"}
```

## Integration Request

Send the generated three-frame sequence through the same endpoint used by the
frontend:

```bash
curl --fail --max-time 1200 \
  -X POST http://127.0.0.1:8020/agent/video_understanding \
  -H 'Content-Type: application/json' \
  -d '{
    "video_dir": "/tmp/hat4d_qwen3vl_demo/input",
    "object_text": "a red moving object",
    "relation_hint": "track motion and relative depth",
    "max_frames": 3
  }' \
  | tee /tmp/hat4d_qwen3vl_integration.json
```

Optionally connect the browser frontend to this backend:

```bash
MPLBACKEND=Agg python annotation_tool/viser_frontend.py \
  --gs-dir /tmp/hat4d_qwen3vl_demo/gs \
  --input-image-dir /tmp/hat4d_qwen3vl_demo/input \
  --eval-gt-dir /tmp/hat4d_qwen3vl_demo/eval_gt \
  --workspace-root /tmp/hat4d_qwen3vl_workspaces \
  --vlm-agent-url http://127.0.0.1:8020 \
  --no-raster-preview \
  --port 8897
```

## Expected output

`/tmp/hat4d_qwen3vl_integration.json` must be a JSON object containing:

- `ok: true`
- `model` equal to `QWEN_SERVED_MODEL`
- `agent: "video_understanding"`
- `frames_used: 3` and `frames_total_after_sampling: 3`
- `scene_graph` with at least `objects`, `relations`, `events`, `constraints`,
  and `x_depth_order_timeline`

Treat invalid JSON, `ok: false`, a missing `choices` error in the backend log,
or a response from a different model as a failed integration test.

## Resource Inspection

Record these outputs with the integration result:

```bash
QWEN_PID=$(cat /tmp/hat4d_qwen3vl_sglang.pid)
VLM_PID=$(cat /tmp/hat4d_qwen3vl_backend.pid)
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv
ps -fp "$QWEN_PID" "$VLM_PID"
tail -n 100 /tmp/hat4d_qwen3vl_sglang.log
tail -n 100 /tmp/hat4d_qwen3vl_backend.log
```

All eight H200 devices should show model memory allocation. A process can be
healthy while one tensor-parallel worker has failed, so inspect both the GPU
list and server log.

## Cleanup

Stop the frontend with `Ctrl-C`, then stop both services:

```bash
QWEN_PID=$(cat /tmp/hat4d_qwen3vl_sglang.pid)
VLM_PID=$(cat /tmp/hat4d_qwen3vl_backend.pid)
kill "$VLM_PID" "$QWEN_PID"
wait "$VLM_PID" "$QWEN_PID" 2>/dev/null || true
ss -ltnp | grep -E ':22002|:8020|:8897' || true
```

Keep only the small JSON/log evidence needed for review. Do not add model
weights, generated frames, logs, or responses to Git.
