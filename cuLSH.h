/*
	Copyright (C) 2014 Aris Kagias <ariskagias@gmail.com>. All Rights Reserved.
	
	This file is part of cuLSH.
	cuLSH is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with cuLSH.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __cuLSH__
#define __cuLSH__

// C++
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cfloat>
#include <iterator>
#include <stdint.h>

// CUDA, CUBLAS, CURAND
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_profiler_api.h>

// THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/adjacent_difference.h>
#include <thrust/extrema.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>

// IF USED WITH MATLAB
#ifdef CULSH_MATLAB
	#include "mex.h"
	#include "matrix.h"
	
	#ifdef printf
		#undef printf
	#endif
	
	#define fprintf(X, Y, ...) printf(Y, ##__VA_ARGS__)
	#define printf(...) do { mexPrintf(__VA_ARGS__); mexEvalString("drawnow"); } while(0)
	
//	#define malloc(X) mxMalloc(X)
//	#define free(X) mxFree(X)
#endif

// DEFINES
#define BLOCK_SIZE 16
#define BLOCK_SIZE_REDUCE 8
#define BLOCK_SIZE_Y 32

#define DEF_MB(x) ( (x) / (float)(1<<20) )
#define DEF_SECONDS(x) ( (x) / (float)CLOCKS_PER_SEC )
#define DEF_MIN(x, y) ( ((x) < (y)) ? (x) : (y) )
#define DEF_STRING_EXPRESSION(x) #x
#define DEF_PRINT_ERROR(...) do { printf("[Error @ %s, l.%d] ", __FILE__, __LINE__); printf(__VA_ARGS__); } while(0)

// TYPEDEFS
typedef thrust::device_vector<float> ThrustFloatD;
typedef thrust::device_vector<unsigned> ThrustUnsignedD;
typedef thrust::device_vector<int> ThrustIntD;
typedef thrust::device_vector<uint64_t> ThrustUint64D;

typedef thrust::host_vector<float> ThrustFloatH;
typedef thrust::host_vector<unsigned> ThrustUnsignedH;
typedef thrust::host_vector<int> ThrustIntH;
typedef thrust::host_vector<uint64_t> ThrustUint64H;

// Now include cuLsh files!
#include "cuLSH_Kernels.cu"
#include "cuLSH_Indexing.cu"
#include "cuLSH_Querying.cu"
#include "cuLSH_HashTables.cu"
#include "cuLSH_SearchTables.cu"

#endif // #ifdef__cuLSH__

// #define for debugging
// mex interfaces
// makefile
