CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -O2
CPPFLAGS = -Iinclude -MMD -MP

NVCC = nvcc
# Keep host compiler flags in sync via -Xcompiler; add your CUDA arch flags here if you want.
NVCCFLAGS = -O2 -std=c++11 --gpu-architecture=compute_80
NVCCDEPFLAGS = -MMD -MP

TARGET = program

SRC_DIR = src
BUILD_DIR = build

# Gather sources
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS  := $(wildcard $(SRC_DIR)/*.cu)
SRCS     := $(CPP_SRCS) $(CU_SRCS)

# Map sources -> objects in build/
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
CU_OBJS  := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
OBJS     := $(CPP_OBJS) $(CU_OBJS)

# Dependency files
DEPS := $(OBJS:.o=.d)

all: $(TARGET)

# Link with nvcc to ensure CUDA libs are included
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $^

# Ensure build dir exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile C++ into build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Compile CUDA into build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCCDEPFLAGS) -Iinclude -Xcompiler "$(CXXFLAGS)" -c $< -o $@

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean