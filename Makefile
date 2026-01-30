# SM Profiler Library Makefile

NVCC = nvcc
NVCC_FLAGS = -O3 -gencode=arch=compute_90a,code=sm_90a -std=c++14

# Library objects
LIB_OBJS = sm_profiler.o

# Default target: build the demo
all: sm_profiler_demo

# Compile sm_profiler.cu to object
sm_profiler.o: sm_profiler.cu sm_profiler.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Build demo
sm_profiler_demo: sm_profiler_demo.cu sm_profiler.o sm_profiler.h
	$(NVCC) $(NVCC_FLAGS) sm_profiler_demo.cu sm_profiler.o -o $@

# Clean
clean:
	rm -f *.o sm_profiler_demo sm_profile.json

# Run demo
run: sm_profiler_demo
	./sm_profiler_demo

.PHONY: all clean run
