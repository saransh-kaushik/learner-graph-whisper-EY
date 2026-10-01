#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# gcp_cloud_run/deploy.sh
#
# Builds the Docker image via Cloud Build and deploys to Cloud Run (GPU).
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (`gcloud auth login`)
#   - Project has Cloud Run, Cloud Build, and Artifact Registry APIs enabled
#   - GPU quota approved for the chosen region (request via GCP console if needed)
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config — edit these ───────────────────────────────────────────────────────
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"             # must support Cloud Run GPU (us-central1, us-east4, europe-west4)
SERVICE_NAME="whisper-diarization"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

# GPU options: "nvidia-l4" (recommended) or "nvidia-tesla-t4"
GPU_TYPE="nvidia-l4"

# Scaling
MIN_INSTANCES=0      # 0 = scale to zero (cold start ~30-60s); set to 1 for always-warm
MAX_INSTANCES=3      # max parallel containers (each needs its own GPU)
# ─────────────────────────────────────────────────────────────────────────────

echo "▶ Building image with Cloud Build (context = project root)..."
gcloud builds submit \
  --tag "${IMAGE}" \
  --project "${PROJECT_ID}" \
  ../                            # build context is the project root

echo "▶ Deploying to Cloud Run (${REGION})..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  \
  `# GPU` \
  --gpu 1 \
  --gpu-type "${GPU_TYPE}" \
  \
  `# Resources` \
  --cpu 4 \
  --memory 16Gi \
  \
  `# Request handling` \
  --timeout 3600 \
  --concurrency 1 \
  \
  `# Scaling` \
  --min-instances "${MIN_INSTANCES}" \
  --max-instances "${MAX_INSTANCES}" \
  \
  `# Auth — remove --no-allow-unauthenticated to make it public` \
  --no-allow-unauthenticated \
  \
  `# Env vars` \
  --set-env-vars DEVICE=cuda,COMPUTE_TYPE=int8 \
  \
  --project "${PROJECT_ID}"

echo ""
echo "✅ Deployed! Service URL:"
gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format "value(status.url)"

echo ""
echo "Test it (requires gcloud auth token for --no-allow-unauthenticated):"
echo "  TOKEN=\$(gcloud auth print-identity-token)"
echo "  curl -X POST \$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')/transcribe \\"
echo "    -H \"Authorization: Bearer \$TOKEN\" \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"file_url\": \"https://your-url.com/audio.mp4\"}'"
