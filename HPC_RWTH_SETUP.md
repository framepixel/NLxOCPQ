# Running OCPQ LoRA on RWTH HPC Cluster

This guide covers setting up and running the fine-tuned LoRA model server on the RWTH HPC cluster (c23g partition) with tunneling back to your local machine for Streamlit UI access.

## Architecture

```
┌─────────────────────┐
│  Local Machine      │
│  (Streamlit UI)     │
│  localhost:8501     │
└──────────┬──────────┘
           │ SSH tunnel (reverse port forward)
           │ localhost:8000 → HPC:8000
           │
┌──────────▼──────────┐
│  RWTH HPC (c23g)    │
│  compute node       │
│  colab_serve.py     │
│  localhost:8000     │
└─────────────────────┘
```

## Prerequisites

1. SSH access to RWTH HPC: `ssh rwth-hpc` with 2FA
2. Your `NLxOCPQ/` folder and adapter uploaded to HPC
3. `cloudflared` or `ngrok` installed locally (optional, for public tunneling)

## Step 1: Upload Files to HPC

From your local machine:

```bash
scp -r ~/Desktop/RWTH/Thesis/Bread/NLxOCPQ rwth-hpc:~/
scp -r ~/Desktop/RWTH/Thesis/Bread/Logs rwth-hpc:~/  # Optional, if running queries
```

Alternatively, use SFTP or sync tools like `rsync`.

## Step 2: SSH into RWTH HPC

```bash
ssh rwth-hpc  # Enter password + 2FA code
```

Once logged in, you should be on a login node. To check partition status:

```bash
sinfo  # View all partitions
sinfo -p c23g  # View c23g partition specifically
```

## Step 3: Submit SLURM Job

From the login node, submit the job:

```bash
cd ~
sbatch NLxOCPQ/hpc_serve.slurm
```

Check the job status:

```bash
squeue -u $USER
```

Example output:
```
JOBID  PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
 12345       c23g ocpq-serve user1 R       2:15      1 node123
```

Note the node name (e.g., `node123`) and job ID (e.g., `12345`).

## Step 4: Check Server Status

Once the job is running, check the log:

```bash
tail -f logs/12345_serve.log  # Replace 12345 with your job ID
```

Expected log output:
```
Starting OCPQ LoRA inference server on node123 at <date>
Server will be available at http://localhost:8000 on compute node node123
Remember to tunnel port 8000 back to your local machine!
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Once you see the `Uvicorn running` line, the server is ready.

## Step 5: Tunnel Port 8000 to Your Local Machine

You have two options:

### Option A: SSH Port Forwarding (Simple, Recommended)

In a **new terminal on your local machine**:

```bash
ssh -N -L 8000:node123:8000 rwth-hpc
```

Replace `node123` with the actual node name from Step 3.

This creates a reverse tunnel:
- Local: `localhost:8000` → HPC: `node123:8000`
- The connection stays open as long as the SSH session is active.

Test the tunnel:

```bash
curl -v http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","model":"unsloth/llama-3-8b-Instruct-bnb-4bit"}
```

### Option B: cloudflared Tunnel (Public, if needed)

If you need public HTTPS access (like in Colab), install `cloudflared` on the HPC compute node and run:

```bash
# From within the HPC compute node (after SSH-ing in)
# First, you'd need to edit hpc_serve.slurm to add a second job step,
# or SSH directly to the compute node and run:
cloudflared tunnel --url http://localhost:8000
```

This is less practical for HPC since you'd need manual SSH to the compute node. **Option A is recommended.**

## Step 6: Configure Streamlit UI

On your **local machine**, set up the Streamlit environment:

```bash
cd ~/Desktop/RWTH/Thesis/Bread
export HF_BASE_URL=http://localhost:8000  # Use the tunnel endpoint
streamlit run NLxOCPQ/ui.py
```

In the Streamlit sidebar:
- Select **"Custom endpoint"**
- **Endpoint base URL**: `http://localhost:8000`
- **Endpoint auth token (optional)**: Leave blank

Click "Generate & Execute Query" — your local UI will now use the HPC model server!

## Step 7: Cleanup

When done, cancel the job:

```bash
scancel 12345  # Replace with your job ID
```

Close the SSH tunnel:

```bash
# Ctrl+C in the tunnel terminal
```

## Troubleshooting

### Q: I get "connection refused" when testing the tunnel.

**A:** Check that the server started successfully:
- View the job log: `tail logs/JOBID_serve.log`
- Confirm the node name: `squeue -u $USER`
- Retry the tunnel with the correct node: `ssh -N -L 8000:node123:8000 rwth-hpc`

### Q: The server runs but Streamlit gets 404 on `/v1/chat/completions`.

**A:** Ensure the tunnel is active and the base URL in Streamlit is exactly `http://localhost:8000` (without trailing slash).

### Q: I see "OutOfMemoryError" in server logs.

**A:** The c23g GPU (typically A100 80GB) should suffice for llama-3-8b. If errors persist:
- Reduce `max_tokens` sent by Streamlit
- Enable `--no-load-in-4bit` (requires more VRAM, so not recommended)
- Use a smaller base model

### Q: I want to change the base model or adapter path.

**A:** Edit `hpc_serve.slurm` lines with `--base-model` and `--adapter-path`, then resubmit:
```bash
sbatch NLxOCPQ/hpc_serve.slurm
```

## Environment Variables

- `PYTORCH_ALLOC_CONF=expandable_segments:True` — Already set in `hpc_serve.slurm` to reduce CUDA fragmentation.
- `HF_TOKEN` — Set if the base model is gated. Pass via `sbatch` environment or edit the script.
- `CUDA_VISIBLE_DEVICES=0` — Use only GPU 0. Adjust if your job has multiple GPUs.

## Performance Notes

- The c23g partition typically offers A100 GPUs with 80GB VRAM and 12 CPU cores per GPU.
- Llama-3-8B with 4-bit loading uses ~10-15GB VRAM, leaving room for batch processing or higher `max_tokens`.
- Generation speed: ~50-100 tokens/sec depending on model parameters and generation length.

## References

- RWTH HPC Wiki: https://www.hpc.rwth-aachen.de/ (internal access)
- SLURM documentation: https://slurm.schedmd.com/sbatch.html
- `colab_serve.py` environment variables: See `COLAB_SERVER.md` in this repo.
