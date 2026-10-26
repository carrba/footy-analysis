# Running a match on an AWS GPU

One-shot job: upload a video to S3, launch a spot GPU instance that runs the pipeline and uploads the results, then terminates itself. Nothing is created until you run `terraform apply`.

Two independent Terraform stacks, each with its own state:
- `infra/aws/storage/`: the S3 bucket. Deploy once, keep it. Costs only the storage.
- `infra/aws/compute/`: the job instance, its IAM role and security group. `apply` launches a job, `destroy` removes everything in it. The bucket is never touched.

```
laptop (WSL)                         AWS (eu-west-2)
  Veo download ──aws s3 cp──►  S3 bucket  raw/<video>.mp4
  scripts/aws_push_code.sh ──►           code/footy-code.tgz
  terraform apply (compute) ──►  EC2 g4dn.xlarge (spot, no open ports)
                                   ├ pulls code + video from S3
                                   ├ ingest → audio → detect (GPU) → motion → inplay
                                   ├ pushes outputs ──►  results/<job>/
                                   └ shuts down → instance terminates
  aws s3 sync results/<job> ◄──  S3 bucket
```

Files: `infra/aws/storage/` (bucket), `infra/aws/compute/` (instance, IAM role, security group, and `user_data.sh.tftpl`, which is what the instance runs), `scripts/aws_push_code.sh` (ships the source).

The instance copies the video from S3 onto its own disk (`aws s3 cp`) before processing. It does not read the mp4 over the network: S3 is not a filesystem, and OpenCV seeks a lot. A same-region copy is fast and free of transfer charges.

## One-time setup

1. **Check credentials** (in WSL): `aws sts get-caller-identity`. If it fails, run `aws configure` (or `aws configure sso`) first. Use an account/user that can create S3, IAM and EC2 resources.
   Terraform will prompt for `aws_profile` (the profile name from `~/.aws/config`) on every plan/apply/destroy. To skip the prompt: `export TF_VAR_aws_profile=<name>`. For the AWS CLI commands below, `export AWS_PROFILE=<name>` (`scripts/aws_push_code.sh` prompts if it is unset).
2. **Request GPU quota.** New accounts usually have 0 vCPUs for G instances, and the launch fails until this is raised. In the console, Service Quotas, EC2 (region eu-west-2), request at least **4 vCPUs** for:
   - "All G and VT Spot Instance Requests" (for spot)
   - "Running On-Demand G and VT instances" (only if you set `use_spot=false`)
   Approval can take from minutes to a couple of days, so do this first. Use 8 vCPUs if you might want `g4dn.2xlarge`.
3. **Check the AMI lookup** (the name pattern in `main.tf` can drift):
   ```
   aws ec2 describe-images --owners amazon --region eu-west-2 \
     --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
     --query 'sort_by(Images,&CreationDate)[-1].[Name,ImageId]'
   ```
4. **Create the bucket** (storage stack; costs only the stored data):
   ```
   cd infra/aws/storage
   terraform init
   terraform plan      # read it: one bucket plus its public-access block, encryption and lifecycle rule
   terraform apply
   cd ../../..
   ```
   Terraform state is a local file next to each stack (`terraform.tfstate`, gitignored). Don't delete it while AWS resources exist.
   Now you can upload footage and push code whenever you like, with no instance running (see the next sections).
5. **Optional, for a live shell on the instance:** install the AWS Session Manager plugin locally. The job also runs without it.

## Benchmark first (about 3 minutes of video, cents)

Run the pipeline on a clip you already know the answers for, to confirm the GPU run reproduces your laptop results and to measure real speed and cost.

```
BUCKET=$(terraform -chdir=infra/aws/storage output -raw bucket)
aws s3 cp data/raw/001002-highlight.mp4 s3://$BUCKET/raw/001002-highlight.mp4
scripts/aws_push_code.sh
cd infra/aws/compute && terraform init      # first time only
terraform apply -var input_key=raw/001002-highlight.mp4 -var job_name=bench-001002 -var duration_seconds=180
cd ../../..
```
Wait for the job to finish (about 10-15 minutes including installs; `results/bench-001002/job.log` appears in S3 and the instance disappears). Then fetch the results and remove the compute stack:
```
aws s3 sync s3://$BUCKET/results/bench-001002 outputs/aws-bench-001002
terraform -chdir=infra/aws/compute destroy -var input_key=raw/001002-highlight.mp4 -var job_name=bench-001002
```
Compare `outputs/aws-bench-001002/match_stats.json` and `match_signals.parquet` with the local `001002` results (small differences from GPU numerics are expected), and read `job.log` for the per-step timings. Use those timings to estimate a full match.

## Full match

1. **Download** the full follow-cam match from Veo in the browser. Note whether the file includes both halves and how long half-time is.
2. **Set the periods** in `config.yaml` under `inplay.periods` to the match's real time ranges in the video (kick-off to end of first half, second-half kick-off to full time), in seconds from the start of the file. Example:
   ```
   periods:
     - {name: first_half,  start_s: 95,   end_s: 2850}
     - {name: second_half, start_s: 3790, end_s: 6550}
   ```
   Half-time footage is excluded entirely. The classifier is only tuned on short clips of live play, so expect the edges near half-time to need a look.
3. **Upload the video:** `aws s3 cp <file>.mp4 s3://<bucket>/raw/match1.mp4` (several GB; `aws s3 cp` does multipart uploads and retries).
4. **Ship the code:** `scripts/aws_push_code.sh` (picks up your edited `config.yaml`).
5. **Launch:**
   ```
   terraform -chdir=infra/aws/compute apply \
     -var input_key=raw/match1.mp4 -var job_name=match1 -var duration_seconds=6600
   ```
   `duration_seconds` must cover the whole file (the ingest default is only 300 s). Add `-var instance_type=g4dn.2xlarge` for more CPU, and `-var use_spot=false` if a spot interruption would hurt.
6. **Watch it** (optional):
   - Live: `aws ssm start-session --target <instance_id>` then `tail -f /var/log/footy-job.log` (the instance takes a few minutes to appear).
   - Console output: `aws ec2 get-console-output --instance-id <id> --latest --output text`.
   - Finished: `s3://<bucket>/results/match1/job.log` appears, and the instance is gone.
7. **Fetch results:** `aws s3 sync s3://<bucket>/results/match1 outputs/match1`. The stats are in `match_stats.json`, segments in `match_segments.csv`, and per-frame data in the parquet files.
8. **Tear down:** `terraform -chdir=infra/aws/compute destroy -var input_key=raw/match1.mp4 -var job_name=match1`. The instance has already terminated itself (that is what stops the billing), so this just removes the leftover role, profile and security group and clears the state. Destroy does the same if the job is still running or hung, so it is also the emergency stop. (The `-var` values only need to be present, not correct.)
9. **Compare with your stopwatch** times for a stretch of the match.

## Cost and time (estimates, to be replaced by the benchmark)

- g4dn.xlarge: roughly $0.2-0.3/hr spot, about $0.6/hr on demand in London. Check current prices.
- Detection at ~48,000 frames should take well under an hour on a T4 (the laptop CPU takes ~11 h).
- **The motion step is CPU-bound** (optical flow) and may take about as long as detection on 4 vCPUs. That is why the benchmark matters; `g4dn.2xlarge` doubles the CPU.
- Order of magnitude: about $1-3 per match, plus pennies for S3 storage of a few GB. The `max_runtime_minutes` watchdog (default 6 h) caps a runaway job at roughly $3.60 on demand.
- Between jobs only the bucket exists (a few GB of storage, pennies per month). To remove it completely, empty it by hand and run `terraform destroy` in `infra/aws/storage`. It refuses to delete itself while it holds data, by design.

## Risks and notes

- **Spot interruption** terminates the instance and the job is lost; rerun it. Detection is not resumable yet.
- **Disk:** the 10 fps working copy of an 80-minute match is a few GB at crf 23. The 200 GB root volume has plenty of room.
- **Security:** no inbound ports, IMDSv2 required, bucket private and encrypted, IAM limited to this bucket. Match footage of children's football stays in your own account.
- **Preview video** is skipped on the instance (`--no-preview`); at 48,000 frames it would be huge.
- Reproducibility: the code is shipped as a tarball of `src/`, `config.yaml` and `pyproject.toml`; Python packages are installed unpinned on the instance. Pinning versions (or a container) is a phase 8 item.

## Container instead? (phase 8)

Not for this first run: it adds an image build, a registry (ECR) and a roughly 6-8 GB CUDA image upload before we know the pipeline's real timings. Once the benchmark and a full-match check are done, containerising is the right step for phase 8: the Dockerfile is the same list of steps as `user_data.sh.tftpl` (apt ffmpeg, pip install, run the stages), and the same image can then run under ECS, AWS Batch or a worker that pulls jobs from a queue.
