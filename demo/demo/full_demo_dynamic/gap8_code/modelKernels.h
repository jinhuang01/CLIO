#ifndef __MODELKERNEL_H__
#define __MODELKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "CNN_BasicKernels.h"
#include "model.h"
#define _L1_Memory_SIZE 51808
#define _model_L2_Memory_SIZE 21088
extern char *L1_Memory; /* Size given for generation: 52000 bytes, used: 51808 bytes */
extern char *model_L2_Memory; /* Size used for generation: 21088 bytes */
extern void S1_Conv2d_24x1x3x3_MaxPool_2x2_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out,
		unsigned int Norm,
		unsigned int NormBias);
extern void S2_Conv2d_16x24x3x3_MaxPool_2x2_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out,
		unsigned int Norm,
		unsigned int NormBias);
extern void S3_Conv2d_8x16x3x3_MaxPool_2x2_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out,
		unsigned int Norm,
		unsigned int NormBias);
extern int modelCNN_Construct();
extern int modelCNN_Destruct();
extern int modelCNN(
		short int *__restrict__ Input_1,
		short int *__restrict__ Output_1);
extern unsigned int AT_GraphOperInfosNames[3];
#endif
