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

#include "cuLSH.h"

int main(int argc, char** argv)
{
	if(argc != 10) {
		printf("%d given arguments, while %d are required...\n", argc, 9);
		printf("Arguments: [filename] [D] [N] [L] [M] [W] [Q] [K] [T]\n");
		return 0;
		}
	FILE *fp;
	if(!(fp = fopen(argv[1], "rb"))) {
		printf("Could not open file with data matrix...\n");
		exit(1);
		}
	const int D = atoi(argv[2]);
	const int N = atoi(argv[3]);
	const int L = atoi(argv[4]);
	const int M = atoi(argv[5]);
	const float W = atof(argv[6]);
	const int Q = atoi(argv[7]);
	const int K = atoi(argv[8]);
	const int T = atoi(argv[9]);
	
	printf("data: [%d x %d], queries: [%d x %d], (L, M, W) = (%d, %d, %.2f), (Q, K, T) = (%d, %d, %d)\n", D, N, D, Q, L, M, W, Q, K, T);
	
	float *dataset, *queries;
	if(!(dataset = (float*) malloc(D * N * sizeof(float))) || !(queries = (float*) malloc(Q * D * sizeof(float)))) {
		printf("Memory allocation error...\n");
		exit(1);
		}
	//float dataset[D * N];
	//float queries[D * Q];
	time_t t;
	
	printf("asdadsfga\n");
	if(fread(dataset, sizeof(float), D * N, fp) != D * N) {
		printf("Could not load data matrix from file...\n");
		exit(1);
		}
	printf("asdadsfga\n");
	memcpy(queries, dataset, D * Q * sizeof(float));
	printf("asdadsfga\n");
	
	printf("data: [%d x %d], queries: [%d x %d], (L, M, W) = (%d, %d, %.2f), (Q, K, T) = (%d, %d, %d)\n", D, N, D, Q, L, M, W, Q, K, T);
	
	cuLSH::HashTables hashtables;
	cuLSH::SearchTables searchtables;
	
	if(!hashtables.reset(N, D, L, M, W, stdout)) {
		printf("Could not reset tables...\n");
		exit(1);
		}
	searchtables.reset(&hashtables, K, T);
	
	t = time(0);
	if(!hashtables.index(dataset, stdout)) {
		printf("Failed to perform indexing...\n");
		exit(1);
		}
	time_t timeIndex2 = time(0);
	int time_indexing = (int) difftime(time(0), t);
	
	printf("\n\n\n%d seconds to index!\n\n\n", time_indexing);
	
	t = time(0);
	if(!searchtables.search(queries, Q, dataset, stdout)) {
		printf("Failed to perform searching...\n");
		exit(1);
		}
	int time_searching = (int) difftime(time(0), t);
	
	printf("\n\n\n%d seconds to search!\n\n\n", time_searching);
	
	cuLSH::SearchTables::timesStruct times;
	searchtables.getTimes(&times);
	
	printf("\n\n\n");
	printf("%d seconds to index\n", time_indexing);
	printf("%d seconds to search\n", time_searching);
	printf("\tCollecting: %f\n", times.collect);
	printf("\tSorting:    %f\n", times.sort);
	printf("\tUnique:     %f\n", times.unique);
	printf("\tDistances:  %f\n", times.distances);
	printf("\tnn indices: %f\n", times.knn);
	
	printf("Selectivity = %.3f%%\n", searchtables.getSelectivity());
	printf("Duplicate candidates = %.3f%%\n", searchtables.getNonUniquePercentage());
	
	return 0;
}
