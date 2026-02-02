# SM Profiler Library Makefile

NVCC = nvcc
ARCH ?= sm_90a

GENCODE_FLAGS := $(foreach arch,$(ARCH),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch))
NVCC_FLAGS = -O3 -std=c++14 $(GENCODE_FLAGS)

all: sm_profiler_demo

sm_profiler.o: sm_profiler.cu sm_profiler.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

sm_profiler_demo.o: sm_profiler_demo.cu sm_profiler.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

sm_profiler_demo: sm_profiler.o sm_profiler_demo.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

libsm_profiler.a: sm_profiler.o
	ar rcs $@ $^

clean:
	rm -f *.o sm_profiler_demo libsm_profiler.a sm_profile.json

run: sm_profiler_demo
	./sm_profiler_demo

.PHONY: all clean run
