SDKROOT ?= $(shell xcrun --sdk macosx --show-sdk-path)
MPS_FRAMEWORK_DIR = $(SDKROOT)/System/Library/Frameworks/MetalPerformanceShaders.framework/Frameworks
CXX = xcrun clang++
CXXFLAGS ?= -O2 -std=c++17 -fobjc-arc -DACCELERATE_NEW_LAPACK \
            -isysroot $(SDKROOT) -F$(MPS_FRAMEWORK_DIR)
LDFLAGS ?= -framework Foundation -framework IOSurface -framework Metal \
           -framework MetalPerformanceShaders -framework Accelerate

TARGETS = mps_gemm_probe metal_silu_probe metal_layernorm_probe \
	 metal_layernorm_bwd_probe metal_iosurface_probe

all: $(TARGETS)

mps_gemm_probe: mps_gemm_probe.mm
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

metal_silu_probe: metal_silu_probe.mm
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

metal_layernorm_probe: metal_layernorm_probe.mm
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

metal_layernorm_bwd_probe: metal_layernorm_bwd_probe.mm
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

metal_iosurface_probe: metal_iosurface_probe.mm
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
