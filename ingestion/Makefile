redis-run-docker:
	docker run -d -p 6379:6379 redis/redis-stack-server:latest

redis-stop-docker:
	docker ps -aq --filter ancestor=redis/redis-stack-server:latest --format="{{.ID}}" | xargs docker stop | xargs docker rm

ingestion-build-image:
	docker build --platform linux/amd64 -t ai-chatbot-ingestion .

ingestion-run-docker: # Requires .env file
	docker run --env-file .env -p 8989:8989 ai-chatbot-ingestion

ingestion-run-docker-debug-mode: # Requires .env file
	docker run -it --env-file .env --entrypoint /bin/bash ai-chatbot-ingestion

ingestion-push-image: # Needs to be logged in to AWS
	aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 674877602397.dkr.ecr.eu-central-1.amazonaws.com
	docker tag ai-chatbot-ingestion:latest 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-chatbot-ingestion:latest
	docker push 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-chatbot-ingestion:latest

# Run locally
ingestion-scripts-install-requirements:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

ingestion-scripts-run:
	python3 src/__init__.py

ingestion-scripts-export-vars:
	export $(grep -v '^#' .env | xargs -d '\n')

kubectl-apply:
	kubectl -n kubernetes-namespace-template apply -f k8s/ingestion.yaml

kubectl-apply-test:
	kubectl -n kubernetes-namespace-template apply -f k8s/ingestion-job.yaml

update: ingestion-build-image ingestion-push-image kubectl-apply

update-test: ingestion-build-image ingestion-push-image kubectl-apply-test