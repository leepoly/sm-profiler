# SM Profiler Library Makefile

NVCC = nvcc
NVCC_FLAGS = -O3 -gencode=arch=compute_90a,code=sm_90a -std=c++14
AR = ar
ARFLAGS = rcs

# Library name (static library)
LIB_NAME = libsm_profiler.a

# Default target: build the library and demo
all: $(LIB_NAME) sm_profiler_demo

# Compile sm_profiler.cu to object file
sm_profiler.o: sm_profiler.cu sm_profiler.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Create static library
$(LIB_NAME): sm_profiler.o
	$(AR) $(ARFLAGS) $@ $<

# Build demo (statically link against the library)
sm_profiler_demo: sm_profiler_demo.cu $(LIB_NAME) sm_profiler.h
	$(NVCC) $(NVCC_FLAGS) sm_profiler_demo.cu -L. -lsm_profiler -o $@

# Clean
clean:
	rm -f *.o $(LIB_NAME) sm_profiler_demo sm_profile.json

# Run demo
run: sm_profiler_demo
	./sm_profiler_demo

.PHONY: all clean run
