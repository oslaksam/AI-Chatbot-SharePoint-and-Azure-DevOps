ingestion-build-image:
	docker build --platform linux/amd64 -t ai-chatbot-sharepoint-ingestion .

ingestion-run-docker: # Requires .env file
	docker run --env-file .env -p 8989:8989 ai-chatbot-sharepoint-ingestion

ingestion-run-docker-debug-mode: # Requires .env file
	docker run -it --env-file .env --entrypoint /bin/bash ai-chatbot-sharepoint-ingestion

ingestion-push-image: # Needs to be logged in to AWS
	aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 674877602397.dkr.ecr.eu-central-1.amazonaws.com
	docker tag ai-chatbot-sharepoint-ingestion:latest 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-hackathon-sharepoint-ingestion:latest
	docker push 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-hackathon-sharepoint-ingestion:latest


kubectl-apply:
	kubectl -n kubernetes-namespace-template apply -f ../k8s/sharepoint-ingestion.yaml

kubectl-apply-test:
	kubectl -n kubernetes-namespace-template apply -f ../k8s/sp-ingest.yaml

update: ingestion-build-image ingestion-push-image kubectl-apply

update-test: ingestion-build-image ingestion-push-image kubectl-apply-test