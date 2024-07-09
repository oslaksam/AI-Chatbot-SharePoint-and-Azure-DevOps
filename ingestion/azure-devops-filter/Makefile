ado-cleanup-build-image:
	docker build --platform linux/amd64 -t ai-chatbot-ado-cleanup .

ado-cleanup-run-docker: # Requires .env file
	docker run --env-file .env -p 8989:8989 ai-chatbot-ado-cleanup

ado-cleanup-run-docker-debug-mode: # Requires .env file
	docker run -it --env-file .env --entrypoint /bin/bash ai-chatbot-ado-cleanup

ado-cleanup-push-image: # Needs to be logged in to AWS
	aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 674877602397.dkr.ecr.eu-central-1.amazonaws.com
	docker tag ai-chatbot-ado-cleanup:latest 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-hackathon-ado-cleanup:latest
	docker push 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-hackathon-ado-cleanup:latest

kubectl-apply:
	kubectl -n kubernetes-namespace-template apply -f ../k8s/ado-cleanup.yaml

kubectl-apply-test:
	kubectl -n kubernetes-namespace-template apply -f ../k8s/ado-cleanup-job.yaml

update: ado-cleanup-build-image ado-cleanup-push-image kubectl-apply

update-test: ado-cleanup-build-image ado-cleanup-push-image kubectl-apply-test
