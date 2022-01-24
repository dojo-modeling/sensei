


VERSION := 0.1.3

DEV ?= $(strip $(if $(findstring y,$(prod)),,dev))

VERSION := ${VERSION}$(DEV:dev=-dev)

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "By default make targets assume DEV to run production pass in prod=y as a command line argument"
	@echo ""
	@echo "Targets:"
	@echo ""
	@grep -E '^([a-zA-Z_-])+%*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-40s\033[0m %s\n", $$1, $$2}'


check-%:
	@: $(if $(value $*),,$(error $* is undefined))



## Docker Section

.PHONY: docker_login-dockerhub
docker_login-dockerhub:| check-DOCKERHUB_USER check-DOCKERHUB_PASS  ## Login to docker registery. Requires DOCKERHUB_USER and DOCKERHUB_PASS to be set in the environment
	@printf "${DOCKERHUB_PASS}\n" | docker login -u "${DOCKERHUB_USER}" --password-stdin


.PHONY: docker_build
docker_build:  ## Builds docker image
	docker build -f Dockerfile -t jataware/sensei:${VERSION} .


.PHONY: docker_push
docker_push:| docker_login-dockerhub  ## Pushes docker image to docker hub
	@echo "push ${VERSION}"
	docker push "jataware/sensei:${VERSION}"


