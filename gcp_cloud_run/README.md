# GCP Cloud Run Deployment

Deploys the API to Cloud Run with a GPU instance using the Docker image built from `api/`.

## Prerequisites

- `gcloud` CLI installed and authenticated
- Cloud Run, Cloud Build, and Artifact Registry APIs enabled on the project
- GPU quota approved in the target region ([request here](https://cloud.google.com/run/docs/configuring/services/gpu))

## Steps

```bash
# 1. Edit config at the top of the script (PROJECT_ID, REGION, GPU_TYPE, etc.)
vim deploy.sh

# 2. Make executable and run
chmod +x deploy.sh
./deploy.sh
```

The script will:
1. Build the Docker image via Cloud Build (runs in GCP, not locally)
2. Push it to Google Container Registry (`gcr.io/...`)
3. Deploy to Cloud Run with GPU

## Calling the API (with IAM auth)

```bash
TOKEN=$(gcloud auth print-identity-token)
SERVICE_URL=$(gcloud run services describe whisper-diarization --region us-central1 --format 'value(status.url)')

# Submit job
curl -X POST "$SERVICE_URL/transcribe" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_url": "https://your-url.com/audio.mp4", "num_speakers": 2}'

# Poll result
curl "$SERVICE_URL/jobs/<job_id>" \
  -H "Authorization: Bearer $TOKEN"
```
