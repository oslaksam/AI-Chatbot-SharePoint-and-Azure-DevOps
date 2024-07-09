build-docker:
	docker build --platform linux/amd64 -t ai-chatbot-chainlit .

run-docker:
	docker run --env-file .env -p 15369:8080 ai-chatbot-chainlit

run-docker-debug:
	docker run -it --env-file .env --entrypoint "/bin/sh" ai-chatbot-chainlit

push-image:
	aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 674877602397.dkr.ecr.eu-central-1.amazonaws.com
	docker tag ai-chatbot-chainlit:latest 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-chatbot-chainlit:latest
	docker push 674877602397.dkr.ecr.eu-central-1.amazonaws.com/ai-hackathon-chainlit:latest

kubectl-apply:
	kubectl -n kubernetes-namespace-template apply -f k8s/deployment.yaml
	kubectl -n kubernetes-namespace-template apply -f k8s/service.yaml
	kubectl -n kubernetes-namespace-template apply -f k8s/ingressroute.yaml

update: build-docker push-image kubectl-apply