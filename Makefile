P=a
########################################################
MATLAB=/usr/local/MATLAB/R2012a
CUDA=/usr/local/cuda/lib64
MEX_INCLUDE=$(MATLAB)/extern/include
########################################################
NVCC=nvcc -O4
MEX=mex -O
########################################################
CUDA_LIBRARIES=-lcuda -lcudart -lcublas -lcurand
MEX_LIBRARIES=-largeArrayDims
########################################################

SOURCE=$(P).cu
MEXCOMPILE=$(MEX) $(P).o -L$(CUDA) $(CUDA_LIBRARIES) $(MEX_LIBRARIES) -o $(P)

all: $(P).o
	$(MEXCOMPILE)
	rm -v $(P).o

compile:
	$(NVCC) -L$(CUDA) $(CUDA_LIBRARIES) $(SOURCE) -o $(P)

$(P).o: $(SOURCE)
	$(NVCC) -I$(MEX_INCLUDE) -Xcompiler -fPIC -c $(SOURCE)

clean: 
	rm -v *~ *.o *.mexa64

