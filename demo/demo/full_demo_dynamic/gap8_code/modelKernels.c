#include "modelKernels.h"
L1_CL_MEM AT_L1_POINTER L1_Memory;
L2_MEM AT_L2_POINTER model_L2_Memory;
AT_HYPERRAM_POINTER model_L3_Memory;
static AT_HYPERRAM_T HyperRam;
static AT_HYPERFLASH_FS_T HyperFlash;
void S1_Conv2d_24x1x3x3_MaxPool_2x2_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out,
		unsigned int Norm,
		unsigned int NormBias)

{
	/* Shared L1: 43568 bytes, L2 buffer: 12336 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	AT_HYPERRAM_CL_EVENT Uchan1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerReLUPool_fp_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Total=0, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _P_Out, _C_Out;
	unsigned int _SPP_Out, _SP_Out, _SC_Out;
	unsigned int _LPP_Out, _LP_Out, _LC_Out;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	unsigned int _N_In;
	unsigned int _SN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: 2][Tile0 Dim: 122][D0 Dim: 1]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -2, Max Pipe Depth: 0
		KerArgItSpace: 244 logical tiles, 244 physical tiles
			Total Size: 714432 [D1, 2 x 476288][Tile0, 122:[122x1, 120:122x1, 122x1], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 2 x 476288][Tile0, 122:[122x1, 120:122x1, 122x1], 2]
		Tile0: [0, 3904, 244], Tile1: [244, 3904, 244], Tile2; [488, 3904, 244]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 1], T2: [D1: 0][Tile0: 2]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 48 [D1, 2 x 32]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 2 x 32]
		Tile0: [0, 48, 48], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: D1
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 2 logical tiles, 2 physical tiles
			Total Size: 432 [D1, 2 x 288][D0, 1 x 288]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 2 x 288][D0, 1 x 288]
		Tile0: [0, 288, 288], Tile1: [288, 144, 144], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 1][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 122 logical tiles, 122 physical tiles
			Total Size: 119072 [D0, 1 x 119072][Tile0, 122:[244x3, 120:244x4, 244x3], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 122:[244x3, 120:244x4, 244x3], 2][D0, 1 x 119072]
		Tile0: [0, 1464, 1464], Tile1: [488, 1952, 1952], Tile2; [1464, 1952, 1952]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 1], T2: [D0: 0][Tile0: 2]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 244 logical tiles, 1 physical tiles
			Total Size: 5715456 [D1, 2 x 3810304][Tile0, 122:[244x2, 120:244x2, 244x2], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 2 x 3810304][Tile0, 122:[244x2, 120:244x2, 244x2], 4]
		Tile0: [0, 5715456, 5715456], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (L1_Memory+12336);
	KerArg0->W = (unsigned short int) (244);
	KerArg0->H = (unsigned short int) (2);
	KerArg0->Norm = (unsigned char) (Norm);
	KerArg0->NormBias = (unsigned char) (NormBias);
	KerArg1->W = (unsigned short int) (244);
	KerArg1->UsedW = (unsigned short int) (244);
	KerArg1->InFeatures = (unsigned short int) (1);
	KerArg1->Out = (int * __restrict__) (L1_Memory+12336);
	KerArg1->Norm = (unsigned char) (Norm);
	KerArg1->TotalInFeatures = (short int) (1);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (L1_Memory+12336);
	KerArg2->W = (unsigned short int) (244);
	KerArg2->H = (unsigned short int) (2);
	KerArg2->Out = (short int * __restrict__) (L1_Memory+12336);
	KerArg2->Norm = (unsigned char) (Norm);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	KerArg3->In = (short int * __restrict__) (L1_Memory+12336);
	KerArg3->W = (unsigned short int) (244);
	KerArg3->UsedW = (unsigned short int) (244);
	KerArg3->H = (unsigned short int) (2);
	KerArg3->UsedH = (unsigned short int) (2);
	KerArg3->Pad = (v4s) 0;
	KerArg3->Orientation = (unsigned char) (1);
	KerArg3->Oper = (unsigned char) (1);
	KerArg3->LB = (int) (0);
	KerArg3->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=3904; _LC_Out=244;
	_SPP_Out=0; _SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) L1_Memory+3904), 48, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	_N_Filter=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) L1_Memory+3952+0), 288, 0, &DmaR_Evt2);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) L1_Memory+0+0), 1464, 0, &DmaR_Evt3);
	/*============================= End Read Tiles Prolog ===============================*/
	for (D1Ind=0; D1Ind<2; D1Ind++, D1Ind_Total++) { /* Iteration on D1 */
		int D1Ind_Last = (D1Ind==1), D1Ind_NextLast = ((D1Ind+1)==1);
		/*================================= Prepare Tiles ===================================*/
		_SN_Filter = 0;
		if (!(D1Ind_Last)) {
			_N_Filter = _N_Filter + (288); _SN_Filter = ((D1Ind_NextLast)?144:288); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
		if (_SN_Filter) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) L1_Memory+3952+288*((D1Ind_Total+1)%2)),
					_SN_Filter, 0, &DmaR_Evt2);
		}
		/*============================= End Read Tiles ======================================*/
		for (T0Ind=0; T0Ind<122; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==121), T0Ind_NextLast = ((T0Ind+1)==121);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->OutFeatures = (unsigned short int) (D1Ind_Last?8:16);
			KerArg0->Bias = (short int * __restrict__) (L1_Memory+3904+((D1Ind)*32));
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1, D0Ind_NextLast = 1;
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(T0Ind_Last)) {
					_N_In = _N_In + (976-(488*(T0Ind==0))); _SN_In = ((T0Ind_NextLast)?1464:1952); 
				} else if (!(D1Ind_Last)) {
					_N_In = _N_In + (-117608); _SN_In = (1464); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) L1_Memory+0+1952*((D0Ind_Total+1)%2)),
							_SN_In, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (L1_Memory+0+1952*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (4-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (4-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->OutFeatures = (unsigned short int) (D1Ind_Last?8:16);
				KerArg1->Filter = (short int * __restrict__) (L1_Memory+3952+288*((D1Ind_Total)%2));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				
				/*============================= End Update Arg Pipeline =============================*/
				D0Ind_Total++;
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->InFeatures = (unsigned short int) (D1Ind_Last?8:16);
			AT_FORK(gap_ncore(), (void *) KerDP_IO_fp, (void *) KerArg2);
			__CALL(KerDP_IO_fp, KerArg2);
			KerArg3->OutFeatures = (unsigned short int) (D1Ind_Last?8:16);
			KerArg3->Out = (short int * __restrict__) (L1_Memory+4528+3904*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_fp, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_fp, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			if (_SPP_Out) AT_HYPERRAM_CL_WAIT(&HyperRam, &Uchan1); /* Wait previous uDMA write Out */
			if (_SP_Out) AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Out+_P_Out), ((AT_HYPERRAM_INT_ADDR_TYPE) model_L2_Memory+0+3904*((T0Ind_Total+-1)%2)),
						_SP_Out, 29768, _LP_Out, 1, &Uchan1);
			AT_L2_COPY(0, ((AT_HYPERRAM_EXT_ADDR_TYPE) model_L2_Memory+0+3904*((T0Ind_Total)%2)), ((AT_L2_INT_ADDR_TYPE) L1_Memory+4528+3904*((T0Ind_Total)%2)),
					_SC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SPP_Out = _SP_Out;_LPP_Out = _LP_Out;
			_P_Out = _C_Out;_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (244); _LC_Out = (244); _SC_Out = (((D1Ind_Last)?8:16)*_LC_Out); 
			} else if (!(D1Ind_Last)) {
				_C_Out = _C_Out + (476288)+(-29524); _LC_Out = (244); _SC_Out = (((D1Ind_NextLast)?8:16)*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
		/*================================= Update Arg Pipeline =============================*/
		
		/*============================= End Update Arg Pipeline =============================*/
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	if (_SPP_Out) AT_HYPERRAM_CL_WAIT(&HyperRam, &Uchan1); /* Wait previous uDMA write Out */
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Out+_P_Out), ((AT_HYPERRAM_INT_ADDR_TYPE) model_L2_Memory+0+3904*((T0Ind_Total+-1)%2)), _SP_Out, 29768, _LP_Out, 1, &Uchan1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, &Uchan1); /* Wait current uDMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S2_Conv2d_16x24x3x3_MaxPool_2x2_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out,
		unsigned int Norm,
		unsigned int NormBias)

{
	/* Shared L1: 46112 bytes, L2 buffer: 14880 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	AT_HYPERRAM_CL_EVENT Uchan1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerReLUPool_fp_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Total=0, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast, T0Ind_NextNextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast, D0Ind_NextNextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_Bias;
	unsigned int _SN_Bias;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	unsigned int _LN_Filter;
	unsigned int _NN_In;
	unsigned int _SN_In, _SNN_In;
	unsigned int _LN_In, _LNN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: 1][Tile0 Dim: 31][D0 Dim: 12]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 31 logical tiles, 31 physical tiles
			Total Size: 119072 [D1, 1 x 119072][Tile0, 31:[61x2, 29:61x2, 61x1], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 1 x 119072][Tile0, 31:[61x2, 29:61x2, 61x1], 2]
		Tile0: [0, 3904, 244], Tile1: [244, 3904, 244], Tile2; [488, 3904, 244]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 1], T2: [D1: 0][Tile0: 2]
	Ker Arg: Bias, Tiled Space: D1
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, 1 x 32]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 1 x 32]
		Tile0: [0, 32, 32], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: D0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 12 logical tiles, 12 physical tiles
			Total Size: 6912 [D1, 1 x 6912][D0, 12 x 576]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 1 x 6912][D0, 12 x 576]
		Tile0: [0, 576, 36], Tile1: [576, 576, 36], Tile2; [1152, 576, 36]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 1], T2: [D1: 0][D0: 2]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 2
		KerArgItSpace: 372 logical tiles, 372 physical tiles
			Total Size: 714432 [D0, 12 x 59536][Tile0, 31:[122x5, 29:122x6, 122x3], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 31:[122x5, 29:122x6, 122x3], 2][D0, 12 x 59536]
		Tile0: [0, 2440, 1220], Tile1: [59536, 2440, 1220], Tile2; [119072, 2440, 1220]
		T0: [D0: 0][Tile0: 0], T1: [D0: 1][Tile0: 0], T2: [D0: 2][Tile0: 0]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 31 logical tiles, 1 physical tiles
			Total Size: 952576 [D1, 1 x 952576][Tile0, 31:[122x4, 29:122x4, 122x2], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 1 x 952576][Tile0, 31:[122x4, 29:122x4, 122x2], 4]
		Tile0: [0, 952576, 952576], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (L1_Memory+14880);
	KerArg0->W = (unsigned short int) (122);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg0->Norm = (unsigned char) (Norm);
	KerArg0->NormBias = (unsigned char) (NormBias);
	KerArg1->W = (unsigned short int) (122);
	KerArg1->UsedW = (unsigned short int) (122);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->Out = (int * __restrict__) (L1_Memory+14880);
	KerArg1->Norm = (unsigned char) (Norm);
	KerArg1->TotalInFeatures = (short int) (2);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (L1_Memory+14880);
	KerArg2->W = (unsigned short int) (122);
	KerArg2->Out = (short int * __restrict__) (L1_Memory+14880);
	KerArg2->Norm = (unsigned char) (Norm);
	KerArg2->InFeatures = (unsigned short int) (16);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	KerArg3->In = (short int * __restrict__) (L1_Memory+14880);
	KerArg3->W = (unsigned short int) (122);
	KerArg3->UsedW = (unsigned short int) (122);
	KerArg3->OutFeatures = (unsigned short int) (16);
	KerArg3->Pad = (v4s) 0;
	KerArg3->Orientation = (unsigned char) (1);
	KerArg3->Oper = (unsigned char) (1);
	KerArg3->LB = (int) (0);
	KerArg3->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=3904; _LC_Out=244;
	_SP_Out=0;
	_N_Bias=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) L1_Memory+5856+0), 32, 0, &DmaR_Evt1);
	_N_Filter=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) L1_Memory+5920+0), 576, 0, &DmaR_Evt2);
	_NN_In=59536; _SN_In=2440;
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In+0), ((AT_HYPERRAM_INT_ADDR_TYPE) model_L2_Memory+0+0), 2440, 29768, 1220, 0, &Uchan1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, &Uchan1); /* Wait previous uDMA read In */
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In+59536), ((AT_HYPERRAM_INT_ADDR_TYPE) model_L2_Memory+0+2928), 2440, 29768, 1220, 0, &Uchan1);
	AT_L2_COPY(0, ((AT_HYPERRAM_EXT_ADDR_TYPE) model_L2_Memory+0+0), ((AT_L2_INT_ADDR_TYPE) L1_Memory+0+0), 2440, 0, &DmaR_Evt3);
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1, D1Ind_NextLast = 1;
		/*================================= Prepare Tiles ===================================*/
		_SN_Bias = 0;
		
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
		if (_SN_Bias) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+_N_Bias), ((AT_L2_INT_ADDR_TYPE) L1_Memory+5856+32*((D1Ind_Total+1)%2)),
					_SN_Bias, 0, &DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		for (T0Ind=0; T0Ind<31; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==30), T0Ind_NextLast = ((T0Ind+1)==30), T0Ind_NextNextLast = ((T0Ind+2)==30);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->H = (unsigned short int) (T0Ind_Last?2:4);
			KerArg0->Bias = (short int * __restrict__) (L1_Memory+5856+32*((D1Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<12; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==11), D0Ind_NextLast = ((D0Ind+1)==11), D0Ind_NextNextLast = ((D0Ind+2)==11);
				/*================================= Prepare Tiles ===================================*/
				_SN_Filter = 0;
				if (!(D0Ind_Last)) {
					_N_Filter = _N_Filter + (576); _LN_Filter = (36); _SN_Filter = (16*_LN_Filter); 
				} else if (!((T0Ind_Last))) {
					_N_Filter = _N_Filter + (-6336); _LN_Filter = (36); _SN_Filter = (16*_LN_Filter); 
				}
				_SNN_In = 0;
				if (!(D0Ind_Last)) {
					if (!(D0Ind_NextLast)) {
						_NN_In = _NN_In + (59536); _LNN_In = ((T0Ind_Last)?732:(1464-244*(T0Ind==0))); _SNN_In = (2*_LNN_In); 
					} else if (!(T0Ind_Last)) {
						_NN_In = _NN_In + (976-(244*(T0Ind==0)))+(-654896); _LNN_In = ((T0Ind_NextLast)?732:1464); _SNN_In = (2*_LNN_In); 
					} else if (!(1)) {
						_NN_In = _NN_In + (-29036)+(-654896); _LNN_In = (1220); _SNN_In = (2*_LNN_In); 
					}
				} else if (!(T0Ind_Last)) {
					_NN_In = _NN_In + (59536); _LNN_In = ((T0Ind_NextLast)?732:1464); _SNN_In = (2*_LNN_In); 
				} else if (!((1))) {
					_NN_In = _NN_In + (59536); _LNN_In = (1220); _SNN_In = (2*_LNN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
				if (_SN_Filter) {
					AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) L1_Memory+5920+576*((D0Ind_Total+1)%2)),
							_SN_Filter, 0, &DmaR_Evt2);
				}
				AT_HYPERRAM_CL_WAIT(&HyperRam, &Uchan1); /* Wait previous uDMA read In */
				if (_SNN_In) {
					AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In+_NN_In), ((AT_HYPERRAM_INT_ADDR_TYPE) model_L2_Memory+0+2928*((D0Ind_Total)%2)),
							_SNN_In, 29768, _LNN_In, 0, &Uchan1);
				}
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY(0, ((AT_HYPERRAM_EXT_ADDR_TYPE) model_L2_Memory+0+2928*((D0Ind_Total+1)%2)), ((AT_L2_INT_ADDR_TYPE) L1_Memory+0+2928*((D0Ind_Total+1)%2)),
							_SN_In, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (L1_Memory+0+2928*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (((T0Ind_Last)?3:6)-1*(T0Ind==0));
				KerArg1->UsedH = (unsigned short int) (((T0Ind_Last)?3:6)-1*(T0Ind==0));
				KerArg1->Filter = (short int * __restrict__) (L1_Memory+5920+576*((D0Ind_Total)%2));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				
				
				_SN_In = _SNN_In;_LN_In = _LNN_In;
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->H = (unsigned short int) (T0Ind_Last?2:4);
			AT_FORK(gap_ncore(), (void *) KerDP_IO_fp, (void *) KerArg2);
			__CALL(KerDP_IO_fp, KerArg2);
			KerArg3->H = (unsigned short int) (T0Ind_Last?2:4);
			KerArg3->UsedH = (unsigned short int) (T0Ind_Last?2:4);
			KerArg3->Out = (short int * __restrict__) (L1_Memory+7072+3904*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_fp, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_fp, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) L1_Memory+7072+3904*((T0Ind_Total)%2)),
					_SC_Out, 7442, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (244); _LC_Out = ((T0Ind_NextLast)?122:244); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
		/*================================= Update Arg Pipeline =============================*/
		
		/*============================= End Update Arg Pipeline =============================*/
		D1Ind_Total++;
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S3_Conv2d_8x16x3x3_MaxPool_2x2_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out,
		unsigned int Norm,
		unsigned int NormBias)

{
	/* Shared L1: 51808 bytes, L2 buffer: 21088 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerReLUPool_fp_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Total=0, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: 1][Tile0 Dim: 4][D0 Dim: 8]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 14400 [D1, 1 x 14400][Tile0, 4:[30x8, 2:30x8, 30x6], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 1 x 14400][Tile0, 4:[30x8, 2:30x8, 30x6], 2]
		Tile0: [0, 3840, 480], Tile1: [480, 3840, 480], Tile2; [960, 3840, 480]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 1], T2: [D1: 0][Tile0: 2]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, 1 x 16]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 1 x 16]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: D1
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 8 logical tiles, 1 physical tiles
			Total Size: 2304 [D1, 1 x 2304][D0, 8 x 288]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 1 x 2304][D0, 8 x 288]
		Tile0: [0, 2304, 2304], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 119072 [D0, 8 x 14884][Tile0, 4:[61x17, 2:61x18, 61x14], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[61x17, 2:61x18, 61x14], 2][D0, 8 x 14884]
		Tile0: [0, 4148, 2074], Tile1: [14884, 4148, 2074], Tile2; [29768, 4148, 2074]
		T0: [D0: 0][Tile0: 0], T1: [D0: 1][Tile0: 0], T2: [D0: 2][Tile0: 0]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 115200 [D1, 1 x 115200][Tile0, 4:[60x16, 2:60x16, 60x12], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, 1 x 115200][Tile0, 4:[60x16, 2:60x16, 60x12], 4]
		Tile0: [0, 115200, 115200], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (L1_Memory+21088);
	KerArg0->W = (unsigned short int) (60);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg0->Bias = (short int * __restrict__) (L1_Memory+8784);
	KerArg0->Norm = (unsigned char) (Norm);
	KerArg0->NormBias = (unsigned char) (NormBias);
	KerArg1->W = (unsigned short int) (61);
	KerArg1->UsedW = (unsigned short int) (61);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Out = (int * __restrict__) (L1_Memory+21088);
	KerArg1->Norm = (unsigned char) (Norm);
	KerArg1->TotalInFeatures = (short int) (16);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (L1_Memory+21088);
	KerArg2->W = (unsigned short int) (60);
	KerArg2->Out = (short int * __restrict__) (L1_Memory+21088);
	KerArg2->Norm = (unsigned char) (Norm);
	KerArg2->InFeatures = (unsigned short int) (8);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	KerArg3->In = (short int * __restrict__) (L1_Memory+21088);
	KerArg3->W = (unsigned short int) (60);
	KerArg3->UsedW = (unsigned short int) (60);
	KerArg3->OutFeatures = (unsigned short int) (8);
	KerArg3->Pad = (v4s) 0;
	KerArg3->Orientation = (unsigned char) (1);
	KerArg3->Oper = (unsigned char) (1);
	KerArg3->LB = (int) (0);
	KerArg3->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=3840; _LC_Out=480;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) L1_Memory+8784), 16, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	_N_Filter=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) L1_Memory+8800+0), 2304, 0, &DmaR_Evt2);
	_N_In=0;
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) L1_Memory+0+0), 4148, 7442, 2074, 0, &DmaR_Evt3);
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1, D1Ind_NextLast = 1;
		/*================================= Prepare Tiles ===================================*/
		_SN_Filter = 0;
		
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
		if (_SN_Filter) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) L1_Memory+8800+2304*((D1Ind_Total+1)%2)),
					_SN_Filter, 0, &DmaR_Evt2);
		}
		/*============================= End Read Tiles ======================================*/
		for (T0Ind=0; T0Ind<4; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==3), T0Ind_NextLast = ((T0Ind+1)==3);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->H = (unsigned short int) (T0Ind_Last?12:16);
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<8; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==7), D0Ind_NextLast = ((D0Ind+1)==7);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (14884); _LN_In = ((T0Ind_Last)?1708:(2196-122*(T0Ind==0))); _SN_In = (2*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (1952-(122*(T0Ind==0)))+(-104188); _LN_In = ((T0Ind_NextLast)?1708:2196); _SN_In = (2*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-5734)+(-104188); _LN_In = (2074); _SN_In = (2*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) L1_Memory+0+4392*((D0Ind_Total+1)%2)),
							_SN_In, 7442, _LN_In, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (L1_Memory+0+4392*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (((T0Ind_Last)?14:18)-1*(T0Ind==0));
				KerArg1->UsedH = (unsigned short int) (((T0Ind_Last)?14:18)-1*(T0Ind==0));
				KerArg1->Filter = (short int * __restrict__) (L1_Memory+8800+((D0Ind)*36)+2304*((D1Ind_Total)%2));
				KerArg1->Pad = (v4s) ((v4s){1,0,1*(T0Ind==0),0*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->H = (unsigned short int) (T0Ind_Last?12:16);
			AT_FORK(gap_ncore(), (void *) KerDP_IO_fp, (void *) KerArg2);
			__CALL(KerDP_IO_fp, KerArg2);
			KerArg3->H = (unsigned short int) (T0Ind_Last?12:16);
			KerArg3->UsedH = (unsigned short int) (T0Ind_Last?12:16);
			KerArg3->Out = (short int * __restrict__) (L1_Memory+13408+3840*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_fp, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_fp, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) L1_Memory+13408+3840*((T0Ind_Total)%2)),
					_SC_Out, 1800, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (480); _LC_Out = ((T0Ind_NextLast)?360:480); _SC_Out = (8*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
		/*================================= Update Arg Pipeline =============================*/
		
		/*============================= End Update Arg Pipeline =============================*/
		D1Ind_Total++;
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
int modelCNN_Construct()

{
	AT_HYPERFLASH_FS_FC_EVENT Uchan1;
	AT_HYPERRAM_CONF_T HyperRamConf;
	AT_HYPERFLASH_FS_CONF_T HyperFlashConf;
	int Error;
	AT_HYPERRAM_CONF_INIT(&HyperRamConf, AT_MEM_L3_HRAM, 0);
	AT_HYPERFLASH_FS_CONF_INIT(&HyperFlashConf, AT_MEM_L3_HFLASH, 0);
	AT_HYPERRAM_OPEN(&HyperRam, &HyperRamConf, &Error);
	if (Error) return 1;
	AT_HYPERFLASH_FS_OPEN(&HyperFlash, &HyperFlashConf, "model_L3_Flash_Const.dat", &Error);
	if (Error) return 1;
	model_L3_Memory = (AT_HYPERRAM_POINTER) AT_HYPERRAM_ALLOC(&HyperRam, 714432);
	model_L2_Memory = (AT_L2_POINTER) AT_L2_ALLOC(0, 134672);
	L1_Memory = (AT_L1_POINTER) AT_L1_ALLOC(0, 51808);
	/* Moving Step1Weights, size 432 from HyperFlash at 9216 to (size 432) L2 at 134144 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) model_L3_Flash + 9216), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) model_L2_Memory + 134144), 432, 0, &Uchan1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &Uchan1);
	/* Moving Step1Biases, size 48 from HyperFlash at 9648 to (size 48) L2 at 134576 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) model_L3_Flash + 9648), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) model_L2_Memory + 134576), 48, 0, &Uchan1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &Uchan1);
	/* Moving Step2Weights, size 6912 from HyperFlash at 0 to (size 6912) L2 at 124928 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) model_L3_Flash + 0), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) model_L2_Memory + 124928), 6912, 0, &Uchan1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &Uchan1);
	/* Moving Step2Biases, size 32 from HyperFlash at 9696 to (size 32) L2 at 134624 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) model_L3_Flash + 9696), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) model_L2_Memory + 134624), 32, 0, &Uchan1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &Uchan1);
	/* Moving Step3Weights, size 2304 from HyperFlash at 6912 to (size 2304) L2 at 131840 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) model_L3_Flash + 6912), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) model_L2_Memory + 131840), 2304, 0, &Uchan1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &Uchan1);
	/* Moving Step3Biases, size 16 from HyperFlash at 9728 to (size 16) L2 at 134656 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) model_L3_Flash + 9728), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) model_L2_Memory + 134656), 16, 0, &Uchan1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &Uchan1);
	return 0;
}
int modelCNN_Destruct()

{
	AT_HYPERRAM_FREE(&HyperRam, model_L3_Memory, 714432);
	AT_L2_FREE(0, model_L2_Memory, 134672);
	AT_L1_FREE(0, L1_Memory, 51808);
	AT_HYPERRAM_CLOSE(&HyperRam);
	AT_HYPERFLASH_FS_CLOSE(&HyperFlash);
	return 0;
}
int modelCNN(
		short int *__restrict__ Input_1,
		short int *__restrict__ Output_1)

{
	S1_Conv2d_24x1x3x3_MaxPool_2x2_Relu(
		((short int *__restrict__) Input_1), /* In */
		((short int *__restrict__) (model_L2_Memory+134144)), /* Filter */
		((short int *__restrict__) (model_L2_Memory+134576)), /* Bias */
		((short int *__restrict__) (model_L3_Memory+0)), /* Out */
		16, /* Norm */
		19 /* NormBias */
	);
	S2_Conv2d_16x24x3x3_MaxPool_2x2_Relu(
		((short int *__restrict__) (model_L3_Memory+0)), /* In */
		((short int *__restrict__) (model_L2_Memory+124928)), /* Filter */
		((short int *__restrict__) (model_L2_Memory+134624)), /* Bias */
		((short int *__restrict__) (model_L2_Memory+5856)), /* Out */
		16, /* Norm */
		20 /* NormBias */
	);
	S3_Conv2d_8x16x3x3_MaxPool_2x2_Relu(
		((short int *__restrict__) (model_L2_Memory+5856)), /* In */
		((short int *__restrict__) (model_L2_Memory+131840)), /* Filter */
		((short int *__restrict__) (model_L2_Memory+134656)), /* Bias */
		((short int *__restrict__) Output_1), /* Out */
		16, /* Norm */
		21 /* NormBias */
	);
	return 0;
}
