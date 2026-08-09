CC = xcrun clang
SDKROOT ?= $(shell xcrun --show-sdk-path)
CFLAGS = -O2 -DACCELERATE_NEW_LAPACK -isysroot $(SDKROOT) -fobjc-arc
LDFLAGS = -framework Foundation -framework IOSurface -framework Metal -framework Accelerate

# Model selection: make MODEL=jishui_200m_2048
#   Production: jishui_200m, jishui_200m_2048
#   Reference-only MIL/GQA headers: stories110m, qwen3_06b
MODEL ?= qwen3_06b
MODEL_HDR = models/$(MODEL).h
BUILD_DIR ?= .build
MODEL_OBJ = $(BUILD_DIR)/train-$(MODEL).o

train: train.m config.h io.h cpu_ops.h ane_data.h metal_ops.h mil_dynamic.h $(MODEL_HDR)
	@echo "Building for model: $(MODEL)"
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -include $(MODEL_HDR) -c train.m -o $(MODEL_OBJ)
	$(CC) $(CFLAGS) $(MODEL_OBJ) $(LDFLAGS) -o train

clean:
	rm -f train $(BUILD_DIR)/train-*.o

.PHONY: clean train
