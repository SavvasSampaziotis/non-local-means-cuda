################################################################################
#
# This is a Makefile coppied from the NVIDIA_CUDA-6.5_Samples that 
# were installed with the toolkit.
# 
# The complete Makefile has remained unchanged, since this is the only 
# stable way for the project to be build across platforms 
#
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH       ?= /usr/local/cuda-6.5

OSUPPER = $(shell uname -s 2>/dev/null | tr "[:lower:]" "[:upper:]")
OSLOWER = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

OS_SIZE    = $(shell uname -m | sed -e "s/x86_64/64/" -e "s/armv7l/32/" -e "s/aarch64/64/")
OS_ARCH    = $(shell uname -m)
ARCH_FLAGS =

DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
	XCODE_GE_5 = $(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5)
endif

# Take command line flags that override any of these settings
ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif
ifeq ($(ARMv7),1)
	OS_SIZE    = 32
	OS_ARCH    = armv7l
	ARCH_FLAGS = -target-cpu-arch ARM
endif
ifeq ($(aarch64),1)
	OS_SIZE    = 64
	OS_ARCH    = aarch64
	ARCH_FLAGS = -target-cpu-arch ARM
endif

# Common binaries
ifneq ($(DARWIN),)
ifeq ($(XCODE_GE_5),1)
  GCC ?= clang
else
  GCC ?= g++
endif
else
ifeq ($(ARMv7),1)
  GCC ?= arm-linux-gnueabihf-g++
else
  GCC ?= g++
endif
endif
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)

# internal flags
NVCCFLAGS   := -m${OS_SIZE} ${ARCH_FLAGS}
CCFLAGS     :=
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

# OS-specific build flags
ifneq ($(DARWIN),)
  LDFLAGS += -rpath $(CUDA_PATH)/lib
  CCFLAGS += -arch $(OS_ARCH)
else
  ifeq ($(OS_ARCH),armv7l)
    ifeq ($(abi),androideabi)
      NVCCFLAGS += -target-os-variant Android
    else
      ifeq ($(abi),gnueabi)
        CCFLAGS += -mfloat-abi=softfp
      else
        # default to gnueabihf
        override abi := gnueabihf
        LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
        CCFLAGS += -mfloat-abi=hard
      endif
    endif
  endif
endif

ifeq ($(ARMv7),1)
ifneq ($(TARGET_FS),)
GCCVERSIONLTEQ46 := $(shell expr `$(GCC) -dumpversion` \<= 4.6)
ifeq ($(GCCVERSIONLTEQ46),1)
CCFLAGS += --sysroot=$(TARGET_FS)
endif
LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
endif
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../../common/inc 
# Added by savvas 
#INCLUDES += 
LIBRARIES := -Iinclude 

################################################################################

SAMPLE_ENABLED := 1

# Gencode arguments
ifeq ($(OS_ARCH),armv7l)
SMS ?= 20 30 32 35 37 50 52
else
SMS ?= 11 20 30 35 37 50 52
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

TARGET:=demo_nlm_cuda

OBJ := array_utilities.o
OBJ += reduction.o
OBJ += nlm_cuda.o
OBJ += nlm_cuda_kernels.o

# Target rules
all: build nlm_benchmark.out test_bed.out

build: clear_screen $(TARGET).out 

clear_screen:
	@clear

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif


nlm_benchmark.out: array_utilities.o reduction.o nlm_cuda_kernels.o nlm_cuda.o  src/$(TARGET).cu
	@echo Compiling $@ ...
	$(EXEC) $(NVCC)  src/nlm_benchmark.cu $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $(OBJ) $(LIBRARIES)
	@echo ' '

demo_nlm_cuda.out: array_utilities.o reduction.o nlm_cuda_kernels.o nlm_cuda.o  src/$(TARGET).cu
	@echo Compiling $@ ...
	$(EXEC) $(NVCC)  src/demo_nlm_cuda.cu $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $(OBJ) $(LIBRARIES)
	@echo ' '

array_utilities.o: src/array_utilities.cu
	@echo Compiling $@ ...
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -o $@ -c $<
	@echo ' '

reduction.o: src/reduction.cu
	@echo Compiling $@ ...
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -o $@ -c $<
	@echo ' '

nlm_cuda_kernels.o: src/nlm_cuda_kernels.cu
	@echo Compiling $@ ...
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -o $@ -c $<
	@echo ' '

nlm_cuda.o: src/nlm_cuda.cu
	@echo Compiling $@ ...
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -o $@ -c $<
	@echo ' '

test_bed.out: clear_screen array_utilities.o reduction.o nlm_cuda_kernels.o nlm_cuda.o src/test_bed.cu
	@echo Compiling $@ ...
	$(EXEC) $(NVCC)  src/test_bed.cu $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $(OBJ) $(LIBRARIES)
	@echo ' '

run: build
	$(EXEC) ./$(TARGET).out ../data/house8.dat

clean:
	rm -f *.o 
	rm -f *.out 
	
clobber: clean
