build:
	docker build -t runway-demo:emotion -f Dockerfile web/

run:
	docker run -p 8000:80 runway-demo:emotion

build-run:
	make build
	make run
