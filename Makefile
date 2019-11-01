VERSION := $(shell cat VERSION)
IMAGE_NAME = seldonio/nmt-model-server

readme:
	jupyter nbconvert README.ipynb --to markdown

build_rest:
	s2i build src/. $SELDON_BASE_WRAPPER sklearn-server:0.1 \
		--environment-file src/seldon_model.conf

build_grpc:
	s2i build -E assets/s2i_envs/environment_grpc src/. \
			seldonio/seldon-core-s2i-python3:0.12 $(IMAGE_NAME)_grpc:$(VERSION)

push_to_dockerhub_rest:
	docker push $(IMAGE_NAME)_rest:$(VERSION)

push_to_dockerhub_grpc:
	docker push $(IMAGE_NAME)_grpc:$(VERSION)

make test:
	pytest -s --verbose -W ignore 2>&1

make install_dev:
	pip install -r src/requirements-dev.txt


