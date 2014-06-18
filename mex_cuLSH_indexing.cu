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

#define CULSH_MATLAB
#include "cuLSH.h"

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	/*
	1: (matrix, filename_load, [filename_save])	:2|3
	2: (matrix, L, M, W, [filename_save])		:4|5
	3: (matrix, L, M, W, A, b, [filename_save])	:6|7
	*/
	if(!mxIsNumeric(prhs[0])) {
		printf("[MEX]: 1st argument must be input matrix!\n");
		return;
		}
	
	const float *matrix = (float*) mxGetData(prhs[0]);
	const int D = mxGetM(prhs[0]);
	const int N = mxGetN(prhs[0]);
	cuLSH::HashTables tables;
	
	char *filenameFrom = 0, *filenameTo = 0;
	int L, M;
	float W;
	float **A = 0, **b = 0;
	bool givenLMW = false;
	bool givenAb = false;
	
	if(nrhs==2 || nrhs==3) {
		//(matrix, filename_load, [filename_save])	:2|3
		if(!mxIsChar(prhs[1])) {
			printf("[MEX]: 2nd argument must be a char array!\n");
			return;
			}
		int filename_length = mxGetM(prhs[1]) * mxGetN(prhs[1]) + 1;
		if(!(filenameFrom = (char*) malloc(filename_length * sizeof(char)))) {
			printf("[MEX]: Could not allocate memory for source filename...\n");
			return;
			}
		mxGetString(prhs[1], filenameFrom, filename_length);
		if(nrhs==3) {
			if(!mxIsChar(prhs[2])) {
				printf("[MEX]: 3rd [optional] argument must be a char array!\n");
				return;
				}
			filename_length = mxGetM(prhs[2]) * mxGetN(prhs[2]) + 1;
			if(!(filenameTo = (char*) malloc(filename_length * sizeof(char)))) {
				printf("[MEX]: Could not allocate memory for destination filename...\n");
				return;
				}
			mxGetString(prhs[2], filenameTo, filename_length);
			}
		}
	else if(nrhs==4 || nrhs==5) {
		//(matrix, L, M, W, [filename_save])	:4|5
		if(!mxIsNumeric(prhs[1]) || !mxIsNumeric(prhs[2]) || !mxIsNumeric(prhs[3])) {
			printf("[MEX]: Usage: [X] [L] [M] [W] [filename_save](optional)\n");
			return;
			}
		L = (int) mxGetScalar(prhs[1]);
		M = (int) mxGetScalar(prhs[2]);
		W = (float) mxGetScalar(prhs[3]);
		if(nrhs==5) {
			if(!mxIsChar(prhs[4])) {
				printf("[MEX]: 5th [optional] argument must be a char array!\n");
				return;
				}
			int filename_length = mxGetM(prhs[4]) * mxGetN(prhs[4]) + 1;
			if(!(filenameTo = (char*) malloc(filename_length * sizeof(char)))) {
				printf("[MEX]: Could not allocate memory for destination filename...\n");
				return;
				}
			mxGetString(prhs[4], filenameTo, filename_length);
			}
		givenLMW = true;
		}
	else if(nrhs==6 || nrhs==7) {
		//(matrix, L, M, W, A, b, [filename_save])	:6|7
		if(!mxIsNumeric(prhs[1]) || !mxIsNumeric(prhs[2]) || !mxIsNumeric(prhs[3]) || !mxIsCell(prhs[4]) || !mxIsCell(prhs[5])) {
			printf("[MEX]: Usage: [X] [L] [M] [W] [A]([1 x L] cell array) [b]([1 x L] cell array) [filename_save](optional)\n");
			return;
			}
		L = (int) mxGetScalar(prhs[1]);
		M = (int) mxGetScalar(prhs[2]);
		W = (float) mxGetScalar(prhs[3]);
		
		if(!(A = (float**) malloc(L * sizeof(float*)))) {
			printf("[MEX]: Could not allocate memory for matrices [A]...\n");
			return;
			}
		if(!(b = (float**) malloc(L * sizeof(float*)))) {
			printf("[MEX]: Could not allocate memory for vectors [b]...\n");
			return;
			}
		for(int table = 0; table < L; table++) {
//			int index;
			mwIndex mexTable = table;
//			index = mxCalcSingleSubscript(prhs[4], 1, &mexTable);
			A[table] = (float*) mxGetData(mxGetCell(prhs[4], mxCalcSingleSubscript(prhs[4], 1, &mexTable)));
//			index = mxCalcSingleSubscript(prhs[5], 1, &mexTable);
			b[table] = (float*) mxGetData(mxGetCell(prhs[5], mxCalcSingleSubscript(prhs[5], 1, &mexTable)));
			}
		if(nrhs==7) {
			if(!mxIsChar(prhs[6])) {
				printf("[MEX]: 7th [optional] argument must be a char array!\n");
				return;
				}
			int filename_length = mxGetM(prhs[6]) * mxGetN(prhs[6]) + 1;
			if(!(filenameTo = (char*) malloc(filename_length * sizeof(char)))) {
				printf("[MEX]: Could not allocate memory for destination filename...\n");
				return;
				}
			mxGetString(prhs[6], filenameTo, filename_length);
			}
		givenLMW = givenAb = true;
		}
	else {
		
		}
	// Reset
	if(givenLMW && givenAb) {
		if(!tables.reset(N, D, L, M, W, A, b, stdout)) {
			printf("[MEX]: Failed to reset table...\n");
			return;
			}
		}
	else if(givenLMW) {
		if(!tables.reset(N, D, L, M, W, stdout)) {
			printf("[MEX]: Failed to reset table...\n");
			return;
			}
		}
	else {
		if(!tables.reset(filenameFrom, stdout)) {
			printf("[MEX]: Failed to load from file \"%s\"...\n", filenameFrom);
			return;
			}
		L = tables.getL();
		M = tables.getM();
		W = tables.getW();
		}
	
	// Index
	clock_t t1 = clock();
	if(!tables.index(matrix, stdout)) {
		printf("[MEX]: Failed to index matrix. I don't know what the hell could have gone wrong... Good luck figuring it out!\n");
		return;
		}
	clock_t t2 = clock();
	printf("[MEX]: (L, M, W) = (%d, %d, %f) -> %f seconds to index...\n", L, M, W, DEF_SECONDS(t2 - t1));
	
	// Save
	if(filenameTo && !tables.save(filenameTo, stdout)) {
		printf("[MEX]: Failed to save to file \"%s\"...\n", filenameTo);
		return;
		}
	
	////////////////////////////////////////////
	if(!givenLMW) {
		L = tables.getL();
		M = tables.getM();
		W = tables.getW();
		}
	
	mwSize mexL = L;
	// COPY A, b to output arguments 0, 1
if(nlhs>=2) {
	if(!(plhs[0] = mxCreateCellArray(1, &mexL))) {
		printf("[MEX]: Could not create cell array for matrices [A]...\n");
		return;
		}
	if(!(plhs[1] = mxCreateCellArray(1, &mexL))) {
		printf("[MEX]: Could not create cell array for vectors [b]...\n");
		return;
		}
	
	for(int table = 0; table < L; table++) {
		mwIndex mexTable = table;
		mxArray *mexA, *mexb;
		
		if(!(mexA = mxCreateNumericMatrix(D, M, mxSINGLE_CLASS, mxREAL))) {
			printf("[MEX]: Could not allocate memory for output matrix [A] of table %d/%d...\n", table + 1, L);
			return;
			}
		if(!(mexb = mxCreateNumericMatrix(1, M, mxSINGLE_CLASS, mxREAL))) {
			printf("[MEX]: Could not allocate memory for output vector [b] of table %d/%d...\n", table + 1, L);
			return;
			}
		
		thrust::copy_n(tables.getA(table), D * M, (float*) mxGetData(mexA));
		thrust::copy_n(tables.getb(table), M, (float*) mxGetData(mexb));
		
		mxSetCell(plhs[0], mxCalcSingleSubscript(plhs[0], 1, &mexTable), mexA);
		mxSetCell(plhs[1], mxCalcSingleSubscript(plhs[1], 1, &mexTable), mexb);
		}
}
	// COPY bucket sizes to output argument 2
if(nlhs>=3) {
	if(!(plhs[2] = mxCreateNumericMatrix(1, L, mxUINT32_CLASS, mxREAL))) { printf("Memory allocation error...\n"); return; }
	unsigned *unsignedData = (unsigned*) mxGetData(plhs[2]);
	thrust::copy_n(tables.getBuckets(), L, unsignedData);
}
	// COPY bucket contents to output argument 3
if(nlhs>=4) {
	if(!(plhs[3] = mxCreateCellArray(1, &mexL))) {
		printf("[MEX]: Could not create cell array for bucket contents...\n");
		return;
		}
	
	for(int table = 0; table < L; table++) {
		mwIndex mexTable = table;
		mxArray *mexBucketContents;
		mxArray *mexAllBuckets;
		
		unsigned buckets = tables.getBuckets(table);
		const unsigned *buckets_totalIndices = tables.getBuckets_totalIndices(table);
		const unsigned *buckets_startingIndices = tables.getBuckets_startingIndices(table);
		const unsigned *buckets_sizes = tables.getBuckets_sizes(table);
		
		mwSize mexTableBuckets = buckets;
		if(!(mexAllBuckets = mxCreateCellArray(1, &mexTableBuckets))) {
				printf("[MEX]: Could not allocate memory for bucket contents of table %d/%d\n", table + 1, L);
				return;
				}
		
		for(int bucket = 0; bucket < buckets; bucket++) {
			mwIndex mexBucket = bucket;
			if(!(mexBucketContents = mxCreateNumericMatrix(1, buckets_sizes[bucket], mxUINT32_CLASS, mxREAL))) {
				printf("[MEX]: Could not allocate memory for bucket %d/%d of table %d/%d\n", bucket + 1, buckets, table + 1, L);
				return;
				}
			thrust::copy_n(buckets_totalIndices + buckets_startingIndices[bucket], buckets_sizes[bucket], (unsigned*) mxGetData(mexBucketContents));
			mxSetCell(mexAllBuckets, mxCalcSingleSubscript(mexAllBuckets, 1, &mexBucket), mexBucketContents);
			}
		mxSetCell(plhs[3], mxCalcSingleSubscript(plhs[3], 1, &mexTable), mexAllBuckets);
		}
}
	
	// COPY bucket codes to output argument 4
if(nlhs>=5) {
	if(!(plhs[4] = mxCreateCellArray(1, &mexL))) {
		printf("[MEX]: Could not create cell array for bucket codes...\n");
		return;
		}
	for(int table = 0; table < L; table++) {
		mwIndex mexTable = table;
		mxArray *mexCodes;
		unsigned buckets = tables.getBuckets(table);
		const float *buckets_codes = tables.getBuckets_codes(table);
		
		if(!(mexCodes = mxCreateNumericMatrix(buckets, M, mxSINGLE_CLASS, mxREAL))) {
			printf("[MEX]: Could not allocate memory for bucket codes of table %d/%d\n", table + 1, L);
			return;
			}
		thrust::copy_n(buckets_codes, buckets * M, (float*) mxGetData(mexCodes));
		mxSetCell(plhs[4], mxCalcSingleSubscript(plhs[4], 1, &mexTable), mexCodes);
		}
}
	
}

