.PHONY: $(VERSIONS) build-% test-% test-all

VERSIONS = 3.11 3.12 3.13 3.14


build-%:
	docker build --build-arg PYTHON_VERSION=$* \
		-t pico_agent-test:$* -f Dockerfile.test .

test-%: build-%
	docker run --rm pico_agent-test:$*

test-all: $(addprefix test-, $(VERSIONS))
	@echo "✅ All versions done"

