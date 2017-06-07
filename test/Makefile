
# Location of the CUDA Toolkit
CUDA_BASE ?= /usr/local/cuda-8.0

# Common binaries
GCC  ?= g++
NVCC := $(CUDA_BASE)/bin/nvcc

# flags
NVCCFLAGS   := -Xptxas  --maxrregcount=32 --default-stream per-thread -O2 -std=c++11
CCFLAGS     := -O2 -fPIC -std=c++11

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif


# Common includes and paths for CUDA
#INCLUDES  := -I $(CUDA_BASE)/include
#LIBRARIES := -l $(CUDA_BASE)/lib64

# CUDA code generation flags
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM37    := -gencode arch=compute_37,code=sm_37
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM52    := -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60

GENCODE_FLAGS   := $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM37) $(GENCODE_SM50) $(GENCODE_SM52) $(GENCODE_SM60)

# Target rules
all: build

build: RawToDigiGPU

RawToDigiGPU.o: ../src/RawToDigiGPU.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

test.o: test.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

RawToDigiGPU: RawToDigiGPU.o test.o
	$(NVCC) -o $@  $^  #$+ $(LIBRARIES)

run: build
	./RawToDigiGPU

clean:
	rm -f RawToDigiGPU *.o 

clobber: clean
