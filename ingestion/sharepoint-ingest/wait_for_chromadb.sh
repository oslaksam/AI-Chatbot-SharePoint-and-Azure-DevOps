#!/bin/sh
# wait-for-chromadb.sh

set -e

host="$1"
port="$2"
shift 2
cmd="$@"

healthcheck_url="http://$host:$port/api/v1/heartbeat"

echo "Checking ChromaDB health at $healthcheck_url..."

until curl -sf "$healthcheck_url"; do
  >&2 echo "ChromaDB is unavailable - sleeping"
  sleep 1
done

>&2 echo "ChromaDB is up - executing the main script"
exec $cmd
