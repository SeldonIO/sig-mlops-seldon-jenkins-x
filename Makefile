VERSION := $(shell cat VERSION)
IMAGE_NAME = seldonio/nmt-model-server
SELDON_BASE_WRAPPER = seldonio/seldon-core-s2i-python3:0.13

readme:
	jupyter nbconvert README.ipynb --to markdown

build:
	s2i build src/. ${SELDON_BASE_WRAPPER} sklearn-server:0.1 \
		--environment-file src/seldon_model.conf

push_to_dockerhub:
	docker push $(IMAGE_NAME)_rest:$(VERSION)

make test:
	(cd src && pytest -s --verbose -W ignore 2>&1)

make install_dev:
	pip install -r src/requirements-dev.txt

install_integration_dev:
	pip install -r integration/requirements-dev.txt

deploy_model:
	helm install charts/sklearn-model-server

delete_model:
	helm delete charts/sklearn-model-server


# INTEGRATION TESTS
install_helm:
	kubectl -n kube-system create sa tiller
	kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller
	helm init --service-account tiller
	kubectl rollout status deploy/tiller-deploy -n kube-system

install_ambassador:
	helm install stable/ambassador -f ambassador_values.yaml --name ambassador --set crds.keep=false --namespace seldon --set replicaCount=1
	kubectl rollout status deployment.apps/ambassador --namespace seldon

install_seldon:
	helm install seldon-core --name seldon-core --repo https://storage.googleapis.com/seldon-charts --namespace seldon

create_namespaces:
	kubectl create namespace seldon || echo "Namespace seldon already exists"
	kubectl config set-context $$(kubectl config current-context) --namespace=seldon

kind_setup: install_helm install_ambassador install_seldon create_namespaces deploy_model

kind_create_cluster:
	kind create cluster --config assets/kind_config.yaml

kind_delete_cluster:
	kind delete cluster


