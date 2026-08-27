# geoloc build and test entry points.
#
# Cross-build for aarch64 is a BLOCKING PR step on purpose (plan/testing/06-ci.md):
# an ARM build failure found a week later costs far more than 10 minutes per PR.

.PHONY: help build-x86 build-arm test test-common test-phasecorr test-se2refine test-metrics lint clean deploy bench gen-skeletons

BOARD ?= orangepi.local
BOARD_USER ?= geoloc

help:
	@echo "build-x86       colcon build for development"
	@echo "build-arm       cross-build for RK3588 (blocking CI step)"
	@echo "test            full test suite"
	@echo "test-common     geoloc_common property tests, standalone (no ROS needed)"
	@echo "test-phasecorr  T19 phase-correlation channel tests, standalone"
	@echo "test-se2refine  T17 SE(2) refinement tests, standalone"
	@echo "test-metrics    T12 metrics harness tests (pytest, no ROS needed)"
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

# Standalone build of the phase-correlation channel tests (T19-U-01..05).
# Dependency-free like test-common: header-only FFT + matcher, Eigen only.
test-phasecorr:
	@mkdir -p build/standalone
	g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
		-I src/geoloc_matcher/include -I src/geoloc_common/include -I /usr/include/eigen3 \
	src/geoloc_matcher/test/test_phase_corr.cpp \
	-o build/standalone/test_phase_corr
	./build/standalone/test_phase_corr

# Standalone build of the SE(2) refinement tests (T17). Dependency-free like
# the phase-correlation tests: header-only refiner, Eigen only.
test-se2refine:
	@mkdir -p build/standalone
	g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
		-I src/geoloc_matcher/include -I src/geoloc_common/include -I /usr/include/eigen3 \
		src/geoloc_matcher/test/test_se2_refine.cpp \
		-o build/standalone/test_se2_refine
	./build/standalone/test_se2_refine

# T12 metrics harness: pure-Python, no ROS / rasterio. Runs in any bare env.
test-metrics:
	python3 -m pytest tools/metrics/tests -q

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
