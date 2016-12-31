cuLSH
=====

***Important note***: The code is outdated and not really fancy. It was created during my thesis (the report is available in greek [here](https://dl.dropboxusercontent.com/u/50157048/cuLSH_thesis_greek.pdf), only abstract is in english, unfortunately no global english translation is available). I plan to re-write the code with major modifications. Indicatively I'd like to make the code object oriented, introduce a new message debugging functionality, create more and easier to use mex interfaces, add UML diagrams, use doxygen documentation. Also I'd like to make some modifications which I expect to speed up indexing & querying. I expect to do this until *~April of 2017*. I hope that version 2.0 will be easier to understand & use, and faster (I hope, hope, hope :D).

General information
-------------------

**cuLSH** is a library used to perform LSH indexing and searching for a given dataset. The matrices (dataset [D x N], queries [D x Q], N = # of data vectors, Q = # of query vectors) must be columnwise in one-dimensional arrays of floats (float dataset [D x N], queries [D x Q]).

The library belongs to the *cuLSH* namespace, and contains 2 important classes:

* `HashTables`: It is used to create the LSH hash tables in order to perform nearest neighbor searching.
* `SearchTables`: It is used to perform LSH nearest neighbor searching after the tables have been created.

If one wants to create his own C++ program using the cuLSH library, all he has to do is add the following preprocessor include command:

	#include "cuLSH.h"

Initialization
--------------

Both classes need to be intialized by the user by calling the function `reset()` of each class. The `reset()` prototypes are:

### Initializing the hash tables

* `bool HashTables::reset(int N, int D, int L, int M, float W, FILE* debugStream = 0);`
* `bool HashTables::reset(int N, int D, int L, int M, float W, float** A, float** b, FILE* debugStream = 0);`
* `bool HashTables::reset(const char* filename, FILE* debugStream = 0);`

The parameters used are:

* `N`: # of vectors in the dataset
* `D`: # of dimensions
* `L`: # of hash tables
* `M`: # of dimensions at projection space
* `W`: Bucket width
* `A`: A[i] is the [D x M] projection matrix of table *i*
* `b`: b[i] is the [1 x M] projection vector of table *i*
* `filename`: File path to load a previously stored indexing structure
* `debugStram`: Stream used to output debugging info

The functions return *`true`* if structure was reset successfully, *`false`* otherwise.

### Initializing the search tables

* `void SearchTables::reset(HashTables* hashtables, int K, int T = 1);`

The parameters used are:

* `hashtables`: Pointer to the indexing structure (`HashTables` instance) containing projection parameters and buckets
* `K`: # of nearest neighbors to be returned at searching
* `T`: # of total probing bins to be examined for each query (T = 1 for classic LSH)

The function returns *`true`* if structure was reset successfully, *`false`* otherwise.

Indexing
--------

After the `HashTables` instance has been reset the user can create the hash tables by calling `HashTables::index()`.

* `bool HashTables::index(const float* matrix, FILE* debugStream = 0);`

The parameters used are:

* `matrix`: [D x N] dataset
* `debugStream`: Stream used to output debugging info

The function returns *`true`* if indexing was performed successfully, *`false`* otherwise.

Searching
---------

After indexing has been performed and the `SearchTables` instance has been reset the user can search for nearest neighbors of multiple query vectors by calling `SearchTables::search()`.

* `bool SearchTables::search(const float* queries, const int Q, const float* dataset, FILE* debugStream = 0);`

The parameters used are:
* `queries`: [D x Q] matrix of queries
* `Q`: # of query vectors
* `dataset`: [D x N] dataset, must be the same that was used with indexing structure hashtables
* `debugStream`: Stream used to output debugging info

The function returns *`true`* if searching was performed successfully, *`false`* otherwise.

This function does not return any results. In order to get the neighbors found, user has to call `SearchTables::getKnnIds()` to get the neighbor ids (from the dataset). Check `SearchTables` for the rest of the functions (e.g. get the KNN distances, calculate selectivity metric, get duration of the search).

Matlab interfaces
-----------------

Two interfaces for using the matlab from Matlab were created. These mex files are:

* `mex_cuLSH_indexing.cu`: Creates (or loads from the hard disk) an indexing structure, creates the hash tables for a given dataset, and (optionally) saves them to the hard disk.
* `mex_cuLSH_querying.cu`: Loads the indexing structure from the hard disk and performs searching for a given query set.

### Indexing from inside Matlab

* `mex_cuLSH_indexing(matrix, L, M, W, [filename_save ])`
* `mex_cuLSH_indexing(matrix, L, M, W, A, b, [filename_save ])`
* `mex_cuLSH_indexing(matrix, filename_load , [filename_save ])`

The parameters used are:

* `matrix`: [D x N] SINGLE CLASS (**not** double) matrix containing the dataset
* `L`: # of hash tables
* `M`: # of dimensions at projection space
* `W`: Bucket width
* `A`: [1 x L] cell array, each cell containing [D x M] projection matrix [A]
* `b`: [1 x L] cell array, each cell containing [1 x M] projection vector [b]
* `filename_save`: The path to store the hash tables to the hard disk
* `filename_load`: The path to load the hash tables from the hard disk

The mex function returns up to 5 arguments in the following order:

* `A`: the projection matrices `[A]` used to create the hash tables
* `b`: The projection vectors `[b]` used to create the hash tables
* `B`: [1 x L] vector, # of buckets for each table
* `bucketContents`: [1 x L] cell array, `bucketContents(i)` containes `B(i)` cells, each one containing the points belonging to each bucket of table *i*
* `bucketCodes`: [1 x L] cell array, `bucketCodes(i)` is `[B(i) x M]`, containing the bucket codes of table *i*

### Querying from inside Matlab

* `mex_cuLSH_querying(queries, data, K, T, filename_load)`

The parameters used are:

* `queries`: [D x Q] query data
* `data`: [D x N] dataset
* `K`: # of nearest neighbors to retrieve
* `T`: # of total probing buckets for each query for each table (T = 1 for classic LSH)
* `filename_load`: The path to load the hash tables from the hard disk

The mex function returns up to 3 arguments in the following order:

* `knnIds`: [K x Q] matrix containing the ids of the queries' neighbors
* `selectivity`: The selectivity of the search, i.e. the percentage value of the dataset searched for the neighbors
* `bucketsSearched`: [1 x Q] cell array, `bucketsSearched(i)` is a [T x L] matrix containing the `T` probing bins matched to the query for each one of the `L` indexing tables

### Creating new mex files using cuLSH

If one wants to create his own mex files using the cuLSH library, he should type the following preprocessor commands:

	#define CULSH_MATLAB
	#include "cuLSH.h"

It is important that the `#define` command is placed before the `#include` command.

This causes the output buffer to be flushed after each `printf()` command (otherwise all outputs of the program functions might be printed to the Matlab console all at once after the program terminates). Also, this causes all the `fprintf()` commands to output everything to the console. This is done because even if the user defines the debugging stream to be stdout, this shouldn't redirect all the debugging outputs to the Matlab console. The user can freely remove the `#define fprintf(...) printf(...)` command from the *cuLSH.h* file and use a file stream for debugging purposes, when working with Matlab.

The user is encouraged to take a look at *cuLSH.h* and see for himself what exactly happens when defining *CULSH_MATLAB* for using cuLSH with Matlab.

Compiling
=========

In the Makefile, the following variables must be set before executing make:

* `P`: Name of the source code file to compile (without the extension .cu)
* `MATLAB`: The path to the Matlab root directory
* `CUDA`: The path to the CUDA library directory (usually /usr/local/cuda/lib)

In order to compile the given mex files, execute:

	make P=mex_cuLSH_indexing matlab
	make P=mex_cuLSH_querying matlab

In order to compile your own Matlab file named e.g. *mat.cu* using the cuLSH library:

	make P=mat matlab

In order to compile your own C++ file (containing main function) named e.g. *main.cu* using the cuLSH library:

	make P=main

The user can create his own Makefile according to his own program, but he should take into account that:

* *cuLSH.h* shall be included in the source code
* The CUDA libraries shall be included & linked when compiling
* The MEX libraries shall be included & linked when compiling a mex file


C++ example
===========

	/*
	Suppose we have already initialized the following variables:
	int D;          // # of dimensions
	int N, Q;       // # of data and query points
	float* matrix;  // dataset ([D x N] stored columnwise)
	float* queries; // queries ([D x Q] stored columnwise)
	int L;          // # of tables
	int M;          // # of projection dimensions
	float W;        // bucket width
	int K;          // # of neighbors
	int T;          // # of probing bins
	char* filename; // name of file to store or load hash tables
	*/
	
	//...
	
	// CREATE LSH HASH TABLES
	cuLSH::HashTables hashtables;
	
	// Reset tables with randomly chosen projection matrices A, b
	if(!hashtables.reset(N, D, L, M, W, stdout)) {
		printf("Failed to reset...\\n");
		// act accordingly
		}
	
	// Index dataset
	if(!hashtables.index(matrix, N, D, L, M, stdout) {
		printf("Failed to index...\\n");
		// act accordingly
		}
	
	// Save tables
	if(!hashtables.save(filename, stdout)) {
		printf("Failed to save...\\n");
		// act accordingly
		}
	
	// SEARCH FOR NEIGHBORS
	cuLSH::SearchTables searchtables;
	
	// Reset search tables
	searchtables.reset(&hashtables, K, T);
	
	// Perform searching
	if(!searchtables.search(queries, Q, matrix , stdout)) {
		printf("Failed to search...\\n");
		// act accordingly
		}
	
	// Retrieve the KNN ids
	const int* ids = searchtables.getKnnIds();
	
	// Retrieve the KNN distances
	const float* distances = searchtables.getKnnDistances();
	
	//...
	
	// Get complexity
	float complexity = searchtables.getSelectivity();
	
	// Get percentage of non-unique candidates for the queries out of the unique candidates
	float non_unique_percentage = searchtables.getNonUniquePercentage();
