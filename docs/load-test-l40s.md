# Load test on an L40S pod (and running the app on a laptop against it)

Goal: find out, with measurements, how many pages a minute one L40S node reads with the pipeline as it is, and whether the app can run on a
CPU machine (a laptop) with only the model on the GPU pod.

Two layouts, two different questions. **Never size the GPU from layout B.**

| Layout | Where things run | Question it answers |
|---|---|---|
| **A. Capacity** | everything on the pod (app, database, OCR, model); the load tool runs on the pod too | How many pages a minute does the GPU node finish, and what does it use (GPU, memory, CPU)? |
| **B. Split** | app + database + printed-text OCR on the laptop; only the model on the pod, through two SSH tunnels | Does the split work, do the same pages give the same tests, and what does the distance cost per page? |

Layout B is not a capacity test: the laptop has 4 cores and does its own OCR, and every model call crosses the tunnel.

**Layout C, the production replica** (a CPU pod for the app and an L40S pod for the model, joined by a tunnel) is in `docs/two-node-runpod.md`: what to rent, the scripts, the bring-up order and the load test on that layout.

Numbers in this guide are labelled MEASURED (seen in this repo's own runs), ESTIMATE (reasoned, not measured) or ASSUMPTION.

## 0. Rules for the pods (the owner's, do not skip)

* Connect only with the direct address and port: `ssh -p <port> -i ~/.ssh/id_ed25519 root@<ip>`. Never `ssh.runpod.io`. Never `-t` / `-tt`.
* Copy files with `scp -P <port> -i ~/.ssh/id_ed25519 ...` (capital P).
* Never `pkill -f` over ssh (it kills your own ssh shell). Stop things by process id.
* Prescription photos are patients' data. Keep them out of git, off shared folders, and delete the test records and photos from the pod volume when the test is over (section 8).

## 1. Prepare on the laptop (once)

1. Install Git (with Git Bash), Python 3.11 or newer (3.12 matches the pods), and the OpenSSH client. Put your key at `~/.ssh/id_ed25519` and add its public key to your RunPod account.
2. `git clone https://github.com/sayantanIFAI/emr2.git` and `cd emr2`. Work only in this repository.
3. Make a folder **outside the repository** with at least 10 handwritten pages (JPEG / PNG / PDF), for example `~/loadtest_pages`. Real handwritten pages matter: a handwritten page makes one model call per handwritten line plus the page-level reads, a printed synthetic page far fewer, so synthetic pages under-measure the GPU load. The owner decides which real pages may be used.
4. Have the two C-DAC packages in `~/Downloads`: `common-lab-codes-for-india-*.zip` and `CommonDrugCodesForIndia_FlatFilePackage*.zip` (not in git: their licence).

## 2. Bring up the L40S pod (layout A)

1. **Create the pod** in RunPod: GPU **L40S (48 GB)**; an Ubuntu / PyTorch template; container disk 50 GB or more; a **network volume of 100 GB or more mounted at `/workspace`**
   (the model files are about 16 GB, plus the vLLM environment pack, lists and backups). If the old volume is in the same data centre as an available L40S, attach it:
   the model, lists and medicine-name file are already on it and the first start is much shorter. Expose TCP port 22 (direct SSH) and HTTP port 8888.
2. **Test the connection and the GPU:**
   ```
   ssh -p <port> -i ~/.ssh/id_ed25519 root@<ip> "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
   ```
   Expect `NVIDIA L40S` and a little over 46,000 MiB (the card's 48 GB as the driver reports it).
3. **Deploy from git** (clones the repo, writes `.env` from `.env.runpod`, generates the admin password, downloads the model, starts everything):
   ```
   ssh -p <port> -i ~/.ssh/id_ed25519 root@<ip> "curl -fsSL https://raw.githubusercontent.com/sayantanIFAI/emr2/main/infra/runpod/deploy_fresh_pod.sh | bash"
   ```
   It returns quickly and keeps working in the background. Follow it: `ssh ... "tail -n 30 /workspace/logs/start_all.log"` until it prints `LIVE URL`.
   With an existing volume (the model and the vLLM pack already on it) the start took several minutes, under 15, on the RTX 4090 pod. A brand-new volume also downloads about 16 GB and installs vLLM: ESTIMATE 20 to 40 minutes, not measured.
4. **National lists and medicine names** (not in git). From the repository root on the laptop, in Git Bash:
   ```
   bash infra/runpod/push_indian_codes.sh <ip> <port>
   ```
   If the volume has no `/workspace/data/medicine_names.txt`, copy it from an older volume, or build it with `scripts/build_medicine_lexicon.py` from the A-Z medicines dataset, then
   `scp -P <port> -i ~/.ssh/id_ed25519 medicine_names.txt root@<ip>:/workspace/data/`. Without it, medicine lines can show up as lab tests (the result is not representative).
5. **Settings for the test.** On the pod, append to `/workspace/cdi/.env`:
   ```
   CDI_UPLOAD_RATE_PER_MINUTE=0
   ```
   (the service refuses more than 30 sends a minute by default, which would be measured instead of the GPU). The service also holds at most 10 pages in flight (`CDI_JOB_QUEUE_MAX`, default 10;
   more are refused with "please wait"). Steps up to 8 in flight need no change; for a step of 12 or more add `CDI_JOB_QUEUE_MAX=32`. Then reload the app: `bash /workspace/cdi/infra/runpod/update_pod.sh`.
   (Running `deploy_fresh_pod.sh` again rewrites `.env` from `.env.runpod` and drops these lines; `update_pod.sh` does not.)
6. **Check everything is up** (on the pod):
   ```
   curl -s http://127.0.0.1:8078/v1/models | head -c 120      # vLLM answers with the model name
   curl -s http://127.0.0.1:8077/healthz                      # gateway: "backend":"vllm"
   curl -s http://127.0.0.1:8888/healthz                      # app: db, s3, mlserve true
   cat /workspace/secrets/admin_password                      # the login: user admin
   ```
   The live URL is `https://<pod-id>-8888.proxy.runpod.net` (sign in with `admin` and that password).
7. **Copy the pages** to the pod: `ssh ... "mkdir -p /workspace/loadtest/images"` then
   `scp -P <port> -i ~/.ssh/id_ed25519 ~/loadtest_pages/* root@<ip>:/workspace/loadtest/images/`.
8. **L40S notes.** vLLM takes 80% of the GPU memory (`CDI_VLLM_GPU_UTIL=0.80`, about 38 GB of 48) and serves up to 16 sequences at once (`CDI_VLLM_MAX_SEQS=16`). The L40S is compute capability 8.9 (like the L4), so `start_vllm.sh` lets vLLM choose its own attention backend and the model runs in fp8.
   MEASURED so far: one page took 17 to 25 s on an RTX 4090 and 30 to 59 s on an L4. No L40S number exists yet: this test produces it.

## 3. Check one page first (layout A)

```
ssh -p <port> -i ~/.ssh/id_ed25519 root@<ip>
cd /workspace/cdi && . .venv/bin/activate
python scripts/load_test.py --base-url http://127.0.0.1:8888 --password-file /workspace/secrets/admin_password \
    --images /workspace/loadtest/images --steps 1 --per-step 1 --warmup 0
```

It must finish with `done 1/1` and no `FAILED` line. Also upload one page in the browser on the live URL and look at the result: this is the reference the later runs are compared with.
The load tool was tested against a pretend service only; this is the first run against the real one, so fix anything it shows here before going on.

## 4. The capacity test (layout A)

Run it detached, so a dropped ssh connection does not stop it:

```
cd /workspace/cdi && . .venv/bin/activate
bash infra/runpod/load_monitor.sh start run1
setsid nohup python scripts/load_test.py --base-url http://127.0.0.1:8888 --password-file /workspace/secrets/admin_password \
    --images /workspace/loadtest/images --steps 1,3,5,8 --per-step 20 --warmup 3 \
    --out /workspace/loadtest/run1.json > /workspace/loadtest/run1.txt 2>&1 < /dev/null &
tail -f /workspace/loadtest/run1.txt          # Ctrl+C leaves the test running
```

When the last line says `written ...`: `bash infra/runpod/load_monitor.sh stop run1` (prints the GPU, CPU, memory and vLLM summary).
Expected length: about 10 to 20 minutes (ESTIMATE: 20 pages per step, 17 to 40 s each, 1 to 8 in flight, plus the warm-up).

Then, one at a time, with a new run name each:

| Test | Command (after the common arguments) | What it shows |
|---|---|---|
| **Soak** | `--steps 5 --per-step 150` | 150 pages at 5 in flight, about 15 minutes: does speed, GPU memory or system memory drift? |
| **Burst** | `--steps 12 --per-step 36` with `CDI_JOB_QUEUE_MAX=32` | what happens above your "5 at most": how much slower, and does anything fail? |
| **Refusal limit** | `--steps 12 --per-step 36` with the default queue limit of 10 | how many sends are refused, and for how long: what a front desk would see |

## 5. Read the results

Copy the files back: `scp -P <port> -i ~/.ssh/id_ed25519 -r "root@<ip>:/workspace/loadtest/run1*" .` (the `run1` folder holds the monitor's files).

The tool prints a table and writes `run1.json` (every upload's time, state and the tests found). What each number means:

| You see | It means |
|---|---|
| **median / p95 / max seconds** | a page's time from accepted to finished. The target you give the front desk is p95. |
| **pages/min** at 5 in flight | the node's throughput at your stated concurrency. |
| **check (avg in flight)** | throughput times mean time: it should be close to the in-flight number. If not, the run was not steady (too few uploads, or a long tail). |
| **failed / refused sends** | anything above 0 needs an explanation before a number is quoted. |
| **Pages whose lab tests changed** | the same page gave a different list of lab tests on different reads. Some variation can be the model itself; a change that grows with load points at timeouts or fallbacks: look at `outcomes` in the JSON. |
| monitor: **GPU average / peak use** | below about 60% at 5 in flight with long times means the GPU is waiting for something else (CPU, OCR, the app). |
| monitor: **vLLM requests waiting** | above 0 for long stretches means the GPU is the limit. |
| monitor: **GPU memory peak** | expect up to about 38 GB (80% of 48). |
| monitor: **CPU used by this container** | compare with its limit (printed beside it): near the limit means the CPU is the limit, whatever the GPU says. |
| monitor: **memory peak** | the real need for RAM. |

Sizing from the result: pages a day = pages/min (at 5 in flight) x 60 x hours the documents arrive. For 10,000 a day the rate needed is about **6.9 / min over 24 h, 13.9 / min over 12 h, 16.7 / min over 10 h**.
Divide the needed rate (with 30% headroom) by the measured rate to get the number of GPU nodes. Document the arrival pattern you assumed; a rush hour needs more than an average.
Only pages that one reader handles one at a time are counted here: a 3-page document is 3 pages.

## 6. Layout B: the app on the laptop, the model on the pod

Use this to prove the split and to see what the distance costs. It does not size the GPU.

1. **Python environment** on the laptop (from the repository root):
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Git Bash: source .venv/Scripts/activate
   pip install -e ".[dev,ocr]"
   ```
   The `ml` extra (torch, transformers) is **not** needed: the model runs on the pod.
2. **PostgreSQL 16.** Easiest: Docker Desktop, then `docker compose -f infra/compose/docker-compose.yml up -d postgres` (user, password and database are all `cdi`, on `localhost:5432`).
   Without Docker: install PostgreSQL 16 for Windows, then create the role and database: `CREATE ROLE cdi LOGIN PASSWORD 'cdi'; CREATE DATABASE cdi OWNER cdi;` (the migrations need no extensions).
   **Redis is not needed**: the upload screen does not use it (only an optional cache, and a Redis outage is never an error). **No S3 server is needed**: the filesystem store is used.
3. **Lists from the pod** (so the laptop reads exactly as the pod does):
   ```
   mkdir -p data/codes
   scp -P <port> -i ~/.ssh/id_ed25519 "root@<ip>:/workspace/data/clci_labs.json" "root@<ip>:/workspace/data/cdci_drugs.json" "root@<ip>:/workspace/data/medicine_names.txt" data/codes/
   ```
4. **`.env`** in the repository root (it is git-ignored). Start from this; every model line matches `.env.runpod`:
   ```
   CDI_ENV=dev
   CDI_DATABASE_URL=postgresql+psycopg://cdi:cdi@localhost:5432/cdi
   CDI_OBJECT_STORE=filesystem
   CDI_OBJECT_STORE_DIR=./data/objects
   CDI_WEBAPP_PORT=8080
   CDI_ADMIN_USER=admin
   CDI_ADMIN_PASSWORD=choose-at-least-12-characters

   CDI_MLSERVE_BACKEND=vllm
   CDI_MLSERVE_URL=http://127.0.0.1:8077
   CDI_VLLM_URL=http://127.0.0.1:8078/v1
   CDI_VLLM_GUIDED=false
   CDI_QWEN_LINE_CONCURRENCY=12
   CDI_VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct

   CDI_RECOGNITION_V2=true
   CDI_OCR_ENGINE=rapidocr
   CDI_OCRHOST_URL=
   CDI_HANDWRITTEN_USES_VLM=true
   CDI_CHECKLIST_MARKS_ENABLED=false
   CDI_REVIEW_UI_ENABLED=false
   CDI_FHIR_ENABLED=false
   CDI_ABHA_ENABLED=false
   CDI_INDIAN_CODES_DIR=./data/codes
   CDI_MEDICINE_LEXICON_PATH=./data/codes/medicine_names.txt
   CDI_LICENSED_CODE_SYSTEMS=["LOINC","UCUM","CDCI"]
   CDI_UPLOAD_RATE_PER_MINUTE=0
   ```
   **Two settings the model needs beyond the URL.** `CDI_MLSERVE_BACKEND=vllm` is required: with the default `stub` the app never calls the pod and returns canned text. `CDI_VLLM_URL` is required too:
   the "is this test really written?" check (`extract/verify.py`) talks to vLLM directly on port 8078, not through the gateway on 8077.
5. **Create the tables and start the app** (from the repository root, with the venv active):
   ```
   alembic upgrade head
   python -m cdi_adapter.webapp
   ```
   It listens on `http://127.0.0.1:8080`. It must start without an error (it checks credentials, libraries and the model registry first).
6. **Open the two tunnels** in a second terminal and leave it open (the ports are the pod's gateway and vLLM; both are bound to the pod's own loopback):
   ```
   ssh -N -L 8077:127.0.0.1:8077 -L 8078:127.0.0.1:8078 -p <port> -i ~/.ssh/id_ed25519 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes root@<ip>
   ```
   Check them: `curl http://127.0.0.1:8077/healthz` (`"backend":"vllm"`) and `curl http://127.0.0.1:8078/v1/models`.
   The distance cost, in milliseconds per call: `curl -s -o /dev/null -w "%{time_total}\n" http://127.0.0.1:8077/healthz` (run it ten times). Multiply by the calls in a row on one page (ESTIMATE: 8 to 12) to expect the added seconds.
7. **Check one page** in the browser on `http://127.0.0.1:8080` (user `admin`). It must give the same lab tests as the same page on the pod's live URL. If not, compare `.env` with the pod's `.env` line by line before anything else.
8. **Time it** (laptop terminal, venv active; the load tool needs nothing but `httpx`, which the project installs):
   ```
   python scripts/load_test.py --base-url http://127.0.0.1:8080 --password-file adminpw.txt --images ~/loadtest_pages --steps 1,3 --per-step 10 --out split1.json
   ```
   (put the admin password alone in `adminpw.txt`, outside the repository.) Compare its median with layout A at the same number in flight: the difference is the cost of the split plus the laptop's slower OCR. Do not go above 3 or 4 in flight here: the laptop's four cores become the limit.

If a tunnel drops, a page in flight fails or waits for the long timeout: restart the tunnel and send it again.

## 7. What a result says, and what it cannot say

* It measures this pipeline as it is today (the main answer, the second look, the test-box reads, the choice votes and the name reads all run). Changing any of those changes the numbers.
* One L40S pod is one machine: its speed can vary with the host. Repeat a run before quoting a number.
* The time is the page's read time, not clinical accuracy. Accuracy needs a fixed truth (a typed list of the lab tests on each page) and is a separate measurement.
* **Verified when this guide was written:** every setting name in section 6 was read by the app's own settings (and the web app's start-up checks pass with those lines, with no Redis, no model files and no database running);
  the load tool passes its unit tests against a pretend service; the monitor script's summary was tested on sample files in the formats `nvidia-smi` and `vmstat` produce.
* **Not verified:** the laptop setup end to end on Windows (the machine used had no Docker or PostgreSQL), any run on an L40S, and the load tool against the real service. Each section above ends in a check so a failure shows early.

## 8. Clean up

1. Stop the monitor (`load_monitor.sh stop <run>`) and copy the results off the pod.
2. Delete the test records and photos from the pod: the uploads are in the pod's database and object store, and the pages are in `/workspace/loadtest/images`. Remove `/workspace/loadtest/images`, and ask the owner how the test records are to be removed from the database and `/workspace/seaweedfs-data`.
3. Stop (or delete) the pod: a running L40S pod is billed by the hour. If the network volume is kept, it still holds patient data.
4. On the laptop: stop the tunnel and the app, remove `data/objects` and the Postgres container volume (`docker compose -f infra/compose/docker-compose.yml down -v`), and delete the pages folder.
