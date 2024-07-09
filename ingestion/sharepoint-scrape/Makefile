scrape-build-image:
	docker build --platform linux/amd64 -t ai-chatbot-sharepoint-scrape .

scrape-run-docker: # Requires .env file
	docker run --env-file .env -p 8989:8989 ai-chatbot-sharepoint-scrape

scrape-run-docker-debug-mode: # Requires .env file
	docker run -it --env-file .env --entrypoint /bin/bash ai-chatbot-sharepoint-scrape

scrape-push-image: # Needs to be logged in to AWS
	aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 674877602397.dkr.ecr.eu-central-1.amazonaws.com
	docker tag ai-chatbot-sharepoint-scrape:latest 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-hackathon-sharepoint-scrape:latest
	docker push 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-hackathon-sharepoint-scrape:latest


kubectl-apply:
	kubectl -n kubernetes-namespace-template apply -f ../k8s/sharepoint-scrape.yaml

kubectl-apply-test:
	kubectl -n kubernetes-namespace-template apply -f ../k8s/sp-scrape.yaml

update: scrape-build-image scrape-push-image kubectl-apply

update-test: scrape-build-image scrape-push-image kubectl-apply-test