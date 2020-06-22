# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

RM=rm -f
CNN_AT_PATH = $(TILER_GENERATOR_PATH)/CNN

NB_FRAME ?= -1

APP = simple
APP_SRCS += main.c modelKernels.c \
            $(CNN_AT_PATH)/CNN_Conv_DP_BasicKernels.c\
            $(CNN_AT_PATH)/CNN_BiasReLULinear_BasicKernels.c \
            $(CNN_AT_PATH)/CNN_Conv_BasicKernels.c \
            $(CNN_AT_PATH)/CNN_Pooling_BasicKernels.c
APP_INC += $(TILER_INC) $(CNN_AT_PATH)

APP_CFLAGS += -O2 -mno-memcpy -DNB_FRAME=$(NB_FRAME) -fno-tree-loop-distribute-patterns  -fdata-sections -ffunction-sections
# The generated code outputs a maybe-uninitialized error which is rather difficult to suppress
# in a clean way.
APP_CFLAGS += -Wall -Wno-maybe-uninitialized -Wno-unused-but-set-variable
#APP_CFLAGS += -I$(TILER_INC) -I$(CNN_AT_PATH)

LDFLAGS +=  -flto -Wl,--gc-sections

PLPBRIDGE_FLAGS += -jtag

# Files to copy in flash

L3_FLASH_CONST=model_L3_Flash_Const.dat

FLASH_FILES = $(L3_FLASH_CONST)

PLPBRIDGE_FLAGS += -f $(FLASH_FILES)


#PMSIS_OS ?= freertos

ifeq ('$(BOARD_NAME)', 'gapuino')
APP_CFLAGS += -DGAPUINO
else ifeq ('$(BOARD_NAME)', 'gapoc_a')
PLPBRIDGE_FLAGS += -ftdi
APP_CFLAGS += -DGAPOC_A
endif

# The double colon allows us to force this to occur before the imported all target
# Link model generation to all step
all:: model

# Build the code generator
GenTile:
	gcc -o GenModel -I$(TILER_INC) -I$(CNN_AT_PATH) models/model.c $(CNN_AT_PATH)/CNN_Generators.c $(TILER_LIB)

# Run the code generator
modelKernels.c: GenTile
	./GenModel

model: modelKernels.c

clean::
	$(RM) GenCifar10 Cifar10Kernels.*

.PHONY: model clean

include $(GAP_SDK_HOME)/tools/rules/pmsis_rules.mk

