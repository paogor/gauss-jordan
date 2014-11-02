ARCH = sm_20

test-performances: test-performances.cu *.hpp
	nvcc test-performances.cu -O3 -arch=$(ARCH) -DBOOST_NOINLINE='__attribute__ ((noinline))' -o=test-performances -lboost_timer -lboost_system

test-correctness: test-correctness.cu *.hpp
	nvcc test-correctness.cu -arch=$(ARCH) -DBOOST_NOINLINE='__attribute__ ((noinline))' -o=test-correctness

test-profile: test-profile.cu *.hpp
	nvcc test-profile.cu -arch=$(ARCH) -o=test-profile -lineinfo
	nvprof --output-profile timeline.nvprof ./test-profile
	nvprof --metrics achieved_occupancy,executed_ipc -o metrics.nvprof ./test-profile

clean:
	rm -f test-correctness test-performances test-profile


