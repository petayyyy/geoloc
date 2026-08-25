# geoloc build and test entry points.
#
# Cross-build for aarch64 is a BLOCKING PR step on purpose (plan/testing/06-ci.md):
# an ARM build failure found a week later costs far more than 10 minutes per PR.

.PHONY: help build-x86 build-arm test test-common lint clean deploy bench gen-skeletons

BOARD ?= orangepi.local
BOARD_USER ?= geoloc

help:
	@echo "build-x86       colcon build for development"
	@echo "build-arm       cross-build for RK3588 (blocking CI step)"
	@echo "test            full test suite"
	@echo "test-common     geoloc_common property tests, standalone (no ROS needed)"
	@echo "lint            pre-commit over all files"
	@echo "deploy          build-arm, push to BOARD, restart, healthcheck, rollback on failure"
	@echo "bench           runtime benchmark on the board (T31/T32 metrics)"
	@echo "gen-skeletons   regenerate node skeleton packages"

build-x86:
	colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

build-arm:
	docker buildx build --platform linux/arm64 -f docker/Dockerfile.cross \
		--load -t geoloc/cross:aarch64 .
	docker run --rm -v $(PWD):/ws -w /ws geoloc/cross:aarch64 \
		colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

test: build-x86
	colcon test --event-handlers console_direct+
	colcon test-result --verbose

# Standalone build of the property tests. Deliberately dependency-free so the
# same binary runs in a bare cross-build container with no ROS present.
test-common:
	@mkdir -p build/standalone
	g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
		-I src/geoloc_common/include -I /usr/include/eigen3 \
		src/geoloc_common/test/test_geoloc_common.cpp \
		-o build/standalone/test_geoloc_common
	./build/standalone/test_geoloc_common

lint:
	pre-commit run --all-files

gen-skeletons:
	python3 tools/gen_node_skeletons.py

deploy: build-arm
	./tools/deploy.sh $(BOARD_USER)@$(BOARD)

bench:
	./tools/bench.sh $(BOARD_USER)@$(BOARD)

clean:
	rm -rf build install log
