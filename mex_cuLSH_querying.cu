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
	(queries, matrix, K, T, filename)
	*/
	if(nrhs < 5) {
		printf("[MEX]: Usage: [queries] [data] [K] [T] [filename_load]\n");
		return;
		}
	int choice = (nrhs==6) ? ((int) mxGetScalar(prhs[5])) : 0; // unused, only for testing purposes
	
	if(nrhs==5) printf("nrhs = 5\n");
	else printf("nrhs ~= 5\n");
	
	if(!mxIsNumeric(prhs[0]) || !mxIsNumeric(prhs[1]) || mxGetM(prhs[0]) != mxGetM(prhs[1])) {
		printf("[MEX]: Arguments 1 & 2 must be query and data matrices, with same number of rows...\n");
		return;
		}
	if(!mxIsNumeric(prhs[2]) || !mxIsNumeric(prhs[3]) || !(mxGetM(prhs[2])==1 && mxGetN(prhs[2])==1 && mxGetM(prhs[3])==1 && mxGetN(prhs[3])==1)) {
		printf("[MEX]: Arguments 3 & 4 must be scalar values K (# of nearest neighbors) and T (# of probing bins)...\n");
		return;
		}
	if(!mxIsChar(prhs[4])) {
		printf("[MEX]: Argument 5 must be filename for loading the indexing structure...\n");
		return;
		}
	
	const float *queries = (float*) mxGetData(prhs[0]);
	const float *dataset = (float*) mxGetData(prhs[1]);
	const int K = (int) mxGetScalar(prhs[2]);
	const int T = (int) mxGetScalar(prhs[3]);
	
	int Q = mxGetN(prhs[0]);
	
	int filename_length = mxGetM(prhs[4]) * mxGetN(prhs[4]) + 1;
	char filenameFrom[filename_length];
	mxGetString(prhs[4], filenameFrom, filename_length);
	
	cuLSH::HashTables hashtables;
	cuLSH::SearchTables searchtables;
	
	// Reset hashtables
	if(!hashtables.reset(filenameFrom, stdout)) {
		printf("[MEX]: Failed to load from file \"%s\"...\n", filenameFrom);
		return;
		}
	// Reset searchtables
	searchtables.reset(&hashtables, K, T);
	// Search
	clock_t t1 = clock();
	if(!searchtables.search(queries, Q, dataset, stdout)) {
		printf("[MEX]: Failed to perform searching...\n");
		return;
		}
	clock_t t2 = clock();
	
	int L = hashtables.getL();
	int M = hashtables.getM();
	float W = hashtables.getW();
	
	printf("[MEX]: (L, M, W) = (%d, %d, %f), (Q, K, T) = (%d, %d, %d) -> selectivity = %.2f%%, time = %f seconds to perform searching...\n", L, M, W, Q, K, T, searchtables.getSelectivity() * 100.0, DEF_SECONDS(t2 - t1));
	
	cuLSH::SearchTables::timesStruct times;
	searchtables.getTimes(&times);
	printf("[MEX]: times: buckets( %f ), collecting( %f seconds ), sorting( %f seconds ), unique( %f seconds ), distances( %f seconds ), knn( %f seconds )\n", times.buckets, times.collect, times.sort, times.unique, times.distances, times.knn);
	
//	printf("[MEX]:\n\tselectivity: %.2f%%\n\tNon-unique candidates percentage: %.2f%%\n", searchtables.getSelectivity() * 100.0, searchtables.getNonUniquePercentage() * 100.0);
	
	const int *ids = searchtables.getKnnIds();
	const float *distances = searchtables.getKnnDistances();
	
	if(!(plhs[0] = mxCreateNumericMatrix(K, Q, mxINT32_CLASS, mxREAL))) {
		printf("[MEX]: Error creating output matrix for knn ids...\n");
		return;
		}
	thrust::copy_n(ids, K * Q, (int*) mxGetData(plhs[0]));
	
if(nlhs>=2) {
/*	if(!(plhs[1] = mxCreateNumericMatrix(K, Q, mxSINGLE_CLASS, mxREAL))) {
		printf("[MEX]: Error creating output matrix for knn distances...\n");
		return;
		}
	thrust::copy_n(distances, K * Q, (float*) mxGetData(plhs[1]));
*/
plhs[1] = mxCreateDoubleScalar(searchtables.getSelectivity());
}
	
	mwSize mexQ = Q;
	
	/*
	mwSize mexL = L;
	plhs[2] = mxCreateCellArray(1, &mexL);
	for(int table = 0; table < L; table++) {
		mwIndex mexTable = table;
		mxArray *mexCodes = mxCreateNumericMatrix(T, Q, mxINT32_CLASS, mxREAL);
		thrust::copy_n(searchtables.getQueryBuckets(table), Q * T, (int*) mxGetData(mexCodes));
		mxSetCell(plhs[2], mxCalcSingleSubscript(plhs[2], 1, &mexTable), mexCodes);
		}
	*/
	
	// copy buckets
if(nlhs>=3) {
	plhs[2] = mxCreateCellArray(1, &mexQ);
	for(int query = 0; query < Q; query++) {
		mwIndex mexQuery = query;
		mxArray *mexBuckets = mxCreateNumericMatrix(T, L, mxINT32_CLASS, mxREAL);
		ThrustIntH queryBuckets(T * L);
		for(int table = 0; table < L; table++) thrust::copy_n(searchtables.getQueryBuckets(table) + query * T, T, queryBuckets.begin() + table * T);
		thrust::copy(queryBuckets.begin(), queryBuckets.end(), (int*) mxGetData(mexBuckets));
		mxSetCell(plhs[2], mxCalcSingleSubscript(plhs[2], 1, &mexQuery), mexBuckets);
		}
}
/*	
	// copy candidates
if(nlhs>=4) {
	plhs[3] = mxCreateCellArray(1, &mexQ);
	const unsigned *candidates_totalIndices = searchtables.getCandidates_totalIndices();
	const unsigned *candidates_startingIndices = searchtables.getCandidates_startingIndices();
	const unsigned *candidates_sizes = searchtables.getCandidates_sizes();
	for(int query = 0; query < Q; query++) {
		mwIndex mexQuery = query;
		mxArray *mexCandidates = mxCreateNumericMatrix(1, candidates_sizes[query], mxUINT32_CLASS, mxREAL);
		thrust::copy_n(candidates_totalIndices + candidates_startingIndices[query], candidates_sizes[query], (unsigned*) mxGetData(mexCandidates));
		mxSetCell(plhs[3], mxCalcSingleSubscript(plhs[3], 1, &mexQuery), mexCandidates);
		}
}
*/
}

