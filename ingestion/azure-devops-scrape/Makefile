ado-scrape-build-image:
	docker build --platform linux/amd64 -t ai-chatbot-ado-scrape .

ado-scrape-run-docker: # Requires .env file
	docker run --env-file .env -p 8989:8989 ai-chatbot-ado-scrape

ado-scrape-run-docker-debug-mode: # Requires .env file
	docker run -it --env-file .env --entrypoint /bin/bash ai-hackathon-ado-scrape

ado-scrape-push-image: # Needs to be logged in to AWS
	aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 674877602397.dkr.ecr.eu-central-1.amazonaws.com
	docker tag ai-hackathon-ado-scrape:latest 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-hackathon-ado-scrape:latest
	docker push 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-hackathon-ado-scrape:latest

kubectl-apply:
	kubectl -n kubernetes-namespace-template apply -f ../k8s/ado-scrape.yaml

kubectl-apply-test:
	kubectl -n kubernetes-namespace-template apply -f ../k8s/ado-scrape-job.yaml

update: ado-scrape-build-image ado-scrape-push-image kubectl-apply

update-test: ado-scrape-build-image ado-scrape-push-image kubectl-apply-test