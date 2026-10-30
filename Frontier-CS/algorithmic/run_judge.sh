#!/bin/bash
# Run judge container without docker compose.
# Usage: ./run_judge.sh      (env JUDGE_WORKERS / GJ_PARALLELISM, default 64: a Frontier-CS
#        validation sends ~860 submissions at once and some jobs run for many minutes)
# Stop: docker stop Competitive-Programming

set -e
cd "$(dirname "$0")"

# Build image
# echo "Building image..."
# docker build -t frontiercs-judge .

# Stop existing container if any
docker rm -f Competitive-Programming 2>/dev/null || true

# Run
echo "Starting container..."
docker run -d \
  --name Competitive-Programming \
  --privileged \
  --shm-size=4g \
  -p 8082:8082 \
  -e PORT=8082 \
  -e GJ_ADDR=http://127.0.0.1:5050 \
  -e JUDGE_WORKERS=${JUDGE_WORKERS:-64} \
  -e GJ_PARALLELISM=${GJ_PARALLELISM:-64} \
  -e SAVE_OUTPUTS=false \
  -v "$(pwd)/problems:/app/problems" \
  -v "$(pwd)/submissions:/app/submissions" \
  -v "$(pwd)/data:/app/data" \
  frontiercs-judge

echo "Judge running at http://localhost:8082"
echo "Check: curl http://localhost:8082/health"
