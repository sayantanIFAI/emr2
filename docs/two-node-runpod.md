# Two machines on RunPod: an app node (CPU) and a GPU node (the model)

This replicates the production layout: the client's **app server** hosts everything except the model (web app, database, document store, printed-text OCR,
validation, billing hand-off) on a CPU machine, and the **GPU node** runs only the model (vLLM with Qwen2.5-VL-7B and the model gateway).
On RunPod that is two pods: a **CPU pod** and an **L40S pod**.

```
 you / the front desk
        |  https  (RunPod proxy, port 8888)
        v
 APP NODE  (CPU pod)                                       GPU NODE  (L40S pod)
 web app :8888, Postgres, document store, RapidOCR   ssh   vLLM  127.0.0.1:8078
 127.0.0.1:8077 and :8078  == the tunnel ===========>      gateway 127.0.0.1:8077
 (patients' data lives here)                                (the model only; nothing is stored here)
```

The app calls the model as `http://127.0.0.1:8077` and `http://127.0.0.1:8078` (both: the gateway, and vLLM directly for the "is this test really written?" check),
exactly as when the model is on the same machine; the SSH tunnel carries both ports. Neither port is ever published: the gateway and vLLM have no sign-in.

Numbers are labelled MEASURED (seen in this repo's own runs), ESTIMATE or ASSUMPTION.

## 1. What RunPod's documentation says that shaped this design (read on 16 Oct 2026)

* RunPod **CPU pods** exist: you choose a CPU type and a vCPU count, the RAM follows from the chosen flavour; they support a container disk and **network volumes**.
* RunPod's private networking between pods (Global Networking, `<pod-id>.runpod.internal`) is **only for NVIDIA GPU pods**, so a CPU pod cannot use it.
  The two nodes therefore talk over the public internet, through an SSH tunnel (encrypted, key-authenticated), not over a private network.
* Direct SSH (`ssh root@<ip> -p <port>`, with `scp`) needs a **public IP with TCP port 22 exposed**. The docs do **not** say whether a CPU pod can have one.
  **This is the first thing to check when you create the CPU pod** (section 2). If it cannot, tell me: the work then needs a different route.
* An exposed TCP port's external number is generally different from the internal one, and **the mapping changes whenever the pod resets**. After any reset, read the new
  port from the pod's Connect menu (Direct TCP Ports) and restart the tunnel (section 7). Secure Cloud IPs "should remain stable"; Community Cloud IPs can change.
* A pod with a network volume **cannot be stopped, only terminated** (it bills until you terminate it). The volume keeps `/workspace` and can be attached to a new pod.

Production note: there the two machines would normally be joined by a private network or VPN and the model gateway would sit behind it. The SSH tunnel here is the
simplest equivalent that needs no change to the code; it adds encryption and the real distance, which is what this test is for.

## 2. What to rent

Rent **two pods**, both on **Secure Cloud** if offered (stable public IPs), in the **same data centre if a CPU pod is available there**, otherwise the nearest.
A network volume belongs to one data centre; each pod gets its own.

| | GPU node | App node |
|---|---|---|
| Type | GPU pod: **NVIDIA L40S 48 GB** | **CPU pod** |
| vCPU / RAM | at least 8 vCPU and 32 GB (the model load needs about 16 GB of RAM for a few minutes) | **16 vCPU and 64 GB** (ESTIMATE; at least 8 and 32). General-purpose or compute-optimised, whichever is offered |
| Template | the RunPod **PyTorch** template on **Ubuntu 24.04** (Python 3.12; the project needs 3.11 or newer; SSH is preconfigured) | an **Ubuntu 24.04** template with SSH |
| Container disk | 50 GB | 50 GB |
| Network volume | **60 GB**, mounted at `/workspace` (model files about 16 GB, plus vLLM's environment pack) | **100 GB**, mounted at `/workspace` (document store, database dump, lists, load-test pages; ESTIMATE 5 to 20 GB a day at 10,000 documents) |
| Expose TCP | **22** (direct SSH). Do not expose any other port | **22** (direct SSH) |
| Expose HTTP | none | **8888** (the web app's address: `https://<pod-id>-8888.proxy.runpod.net`) |
| Name | `cdi-gpu` | `cdi-app` |

Both pods use your account's SSH key (RunPod, Credentials). Never use `ssh.runpod.io`; always `ssh -p <port> -i ~/.ssh/id_ed25519 root@<ip>`, never `-t`.

**When the pods are running, send me, for each pod:** its public IP, its direct SSH port (Connect menu, Direct TCP Ports, the one that maps to 22), its pod id, and for the app node
also confirm that the Connect menu shows a Direct TCP port for 22. I will check the GPU with `nvidia-smi` myself.

Check at creation: does the CPU pod offer **Expose TCP Ports** and show a public IP? If not, say so before doing anything else.

## 3. Prepare on your machine (before renting)

From the repository root, in Git Bash:

```
bash infra/runpod/preflight_laptop.sh stage      # builds the national lists and the medicine-name file; lists what is still missing
bash infra/runpod/preflight_laptop.sh            # must end with READY
```

The staging folder (default `~/pod_stage`, or `POD_STAGE=...`) lives **outside git**: the lists (C-DAC licence), the medicine names and the real pages for the load test. Pages are patients'
data: the owner decides which ones go there, and they are sent to a pod only with an explicit `--with-pages`.

## 4. Bring-up, in this order

Replace `<gpu-ip> <gpu-port> <app-ip> <app-port>` with the values from the Connect menus.

**4.1 GPU node.** It clones the repo, installs the project's base dependencies only (the gateway needs no torch, MEASURED: importing it loads none of torch, transformers or vLLM), downloads the model
(about 16 GB), installs and starts vLLM, and starts the gateway:

```
ssh -p <gpu-port> -i ~/.ssh/id_ed25519 root@<gpu-ip> "setsid nohup bash -c 'curl -fsSL https://raw.githubusercontent.com/sayantanIFAI/emr2/main/infra/runpod/start_gpu_node.sh | bash' > /workspace/gpu_node_start.log 2>&1 < /dev/null &"
ssh -p <gpu-port> -i ~/.ssh/id_ed25519 root@<gpu-ip> "tail -n 25 /workspace/gpu_node_start.log"        # repeat until it ends with GPU NODE READY
```

A new volume downloads the model and installs vLLM: ESTIMATE 20 to 40 minutes (a restart on an existing volume took several minutes on an earlier pod). The last lines print the pod's IP, SSH port and SSH host-key fingerprint.

**4.2 App node.** `CDI_REMOTE_MODEL=1` makes the deploy script skip the model download and not start any model server (this machine has no GPU):

```
ssh -p <app-port> -i ~/.ssh/id_ed25519 root@<app-ip> "CDI_REMOTE_MODEL=1 bash -c 'curl -fsSL https://raw.githubusercontent.com/sayantanIFAI/emr2/main/infra/runpod/deploy_fresh_pod.sh | bash'"
ssh -p <app-port> -i ~/.ssh/id_ed25519 root@<app-ip> "tail -n 30 /workspace/logs/start_all.log"          # until it prints LIVE URL
```

It starts Postgres, the object store, the OCR host and the web app. It will say the tunnel is not up yet: that is expected at this point.

**4.3 The tunnel's key.** The app node makes its own key; the GPU node accepts it **only** for forwarding ports 8077 and 8078 (no shell, nothing else):

```
ssh -p <app-port> -i ~/.ssh/id_ed25519 root@<app-ip> "bash /workspace/cdi/infra/runpod/model_tunnel.sh keygen"          # prints the PUBLIC key
ssh -p <gpu-port> -i ~/.ssh/id_ed25519 root@<gpu-ip> "printf '%s\n' '<the public key line>' > /workspace/secrets/app_node_key.pub && bash /workspace/cdi/infra/runpod/start_gpu_node.sh keyonly"
```

**4.4 Start the tunnel** (on the app node; it reconnects by itself within 5 seconds if it drops):

```
ssh -p <app-port> -i ~/.ssh/id_ed25519 root@<app-ip> "bash /workspace/cdi/infra/runpod/model_tunnel.sh start <gpu-ip> <gpu-port>"
```

It prints `vLLM answers through the tunnel`, then the status: the gateway line must show `"backend":"vllm"`, and it prints the **round trip to the GPU node in milliseconds**. Write that number down: it is the
price of the distance, paid on every model call.

**4.5 Data that is not in git** (from your machine; the pages only when you say so):

```
bash infra/runpod/push_to_app_node.sh <app-ip> <app-port> --with-pages
ssh -p <app-port> -i ~/.ssh/id_ed25519 root@<app-ip> "bash /workspace/cdi/infra/runpod/update_pod.sh CDI_UPLOAD_RATE_PER_MINUTE=0"
```

The second command loads the lists into the running app and lifts the 30-sends-a-minute limit for the test (otherwise that limit is measured). It does not touch the model.

## 5. Check it works, in this order

1. `bash /workspace/cdi/infra/runpod/model_tunnel.sh status` on the app node: tunnel running, gateway `"backend":"vllm"`, vLLM answers, a round-trip time.
2. `curl -s http://127.0.0.1:8888/healthz` on the app node: `db`, `s3` and `mlserve` all `true`.
3. Open `https://<app-pod-id>-8888.proxy.runpod.net` (user `admin`, password in `/workspace/secrets/admin_password` on the app node) and upload one real page. It must give the lab tests you expect.
4. One page through the load tool (it must end `done 1/1`):
   ```
   cd /workspace/cdi && . .venv/bin/activate
   python scripts/load_test.py --base-url http://127.0.0.1:8888 --password-file /workspace/secrets/admin_password --images /workspace/loadtest/images --steps 1 --per-step 1 --warmup 0
   ```

## 6. Load test on this layout

Run it **on the app node** (the load tool talks to the app's own address, so every page crosses the tunnel as it will in production). Record both machines:

```
bash /workspace/cdi/infra/runpod/load_monitor.sh start run1                                   # on the GPU node: GPU, vLLM queue (the GPU is here)
bash /workspace/cdi/infra/runpod/load_monitor.sh start run1                                   # on the app node: CPU and memory (the app is here)
# on the app node:
cd /workspace/cdi && . .venv/bin/activate
setsid nohup python scripts/load_test.py --base-url http://127.0.0.1:8888 --password-file /workspace/secrets/admin_password \
    --images /workspace/loadtest/images --steps 1,3,5,8 --per-step 20 --warmup 3 --out /workspace/loadtest/run1.json > /workspace/loadtest/run1.txt 2>&1 < /dev/null &
tail -f /workspace/loadtest/run1.txt
# when it says "written": stop both monitors (bash /workspace/cdi/infra/runpod/load_monitor.sh stop run1), then copy run1* from both pods with scp -P
```

The soak, burst and refusal tests, the reading of the result table and the sizing arithmetic are in `docs/load-test-l40s.md` (sections 4 and 5); use the same commands. What is different here:

* **Where each number comes from.** GPU use, GPU memory and the vLLM queue come from the GPU node's monitor; CPU used by the container and RAM from the app node's. The app node's monitor also shows the vLLM queue through the tunnel.
* **The distance.** Compare this median with a run of the same pages on a single L40S pod (layout A of `docs/load-test-l40s.md`): the difference is the tunnel plus the app node's slower CPU. If the model calls are many and the round trip is long, it shows.
* **Settings to try if the GPU is waiting on the app node:** more line reads at once, `CDI_QWEN_LINE_CONCURRENCY` (default 12 in `.env.runpod`).

## 7. When something goes wrong

| What you see | Cause | Do |
|---|---|---|
| `healthz` shows `mlserve: false` | the tunnel is down or the GPU node's gateway is not up | `model_tunnel.sh status` on the app node; if the tunnel is down, `model_tunnel.sh start <gpu-ip> <gpu-port>`; if the gateway is down, re-run `start_gpu_node.sh` on the GPU node |
| tunnel log: `Permission denied (publickey)` | the key is not installed on the GPU node (pod resets rewrite `authorized_keys`) | on the GPU node: `bash /workspace/cdi/infra/runpod/start_gpu_node.sh keyonly` |
| tunnel log: `Connection refused` or timeout | the GPU pod's SSH port changed after a reset | read the new Direct TCP port for 22 from its Connect menu, then `model_tunnel.sh start <gpu-ip> <new-port>` |
| pages fail or wait very long | the tunnel dropped during a page | restart the tunnel and send the page again |
| the SSH host key changed warning | the GPU pod was recreated | compare the fingerprint printed at the end of `start_gpu_node.sh`, delete `/workspace/secrets/gpu_node_known_hosts` on the app node and start the tunnel again |
| the app node restarted and nothing works | the database lives on the container disk and the tunnel is a process | `bash /workspace/cdi/infra/runpod/start_all.sh` (with `CDI_REMOTE_MODEL=1` already in its `.env`), then start the tunnel |

## 8. Security, in short

* The gateway and vLLM have no sign-in and are bound to the GPU node's loopback. **Never expose port 8077 or 8078 in RunPod's port settings.**
* The tunnel's key can do one thing: forward those two ports on the GPU node. It lives only on the app node, in `/workspace/secrets` (mode 700).
* The first connection trusts the GPU node's host key (`accept-new`); verify it against the fingerprint printed by `start_gpu_node.sh`.
* Patients' data: the app node holds the database, the stored pages and the load-test pages. The GPU node receives page images and crops inside model requests; as far as I can see in the code nothing there stores them,
  but I have not verified what vLLM itself keeps. Both volumes outlive the pods unless you delete them.

## 9. Clean up

1. Stop both monitors; copy the results off the pods.
2. On the app node: remove `/workspace/loadtest/images`; ask the owner how the test records are to be removed from the database and `/workspace/seaweedfs-data`.
3. **Terminate both pods** (a pod with a network volume cannot be stopped, only terminated; it bills until then). Delete the two network volumes unless you want them, because they hold the model and the patient test data.

## 10. Verified, and not

* **Verified here:** the scripts parse (`bash -n`); the tunnel script's key generation, status and stop; the preflight and the dry-run of the push script against a real staging folder; the gateway starts with the vLLM backend selected on the project's base
  dependencies alone; the app's start-up checks pass without a local model, Redis or model files; the deploy and start scripts behave exactly as before when `CDI_REMOTE_MODEL` is not set.
* **Not verified:** everything that needs the pods: `start_gpu_node.sh`, the tunnel's start loop (this machine has no `setsid` or SSH server), the restricted `authorized_keys` line, a CPU pod's public IP and SSH, the time to build a fresh volume, and the load numbers.
  Each step in section 4 ends in a visible check so a failure shows at once.
