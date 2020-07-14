INC=-I/usr/local/cuda/include
NVCC=/usr/local/cuda/bin/nvcc
NVCC_OPT=-std=c++11

all:
	$(NVCC) $(NVCC_OPT) gpu-example.cu -o gpu-example
	$(NVCC) $(NVCC_OPT) gpu-matrix.cu -o gpu-matrix
	$(NVCC) $(NVCC_OPT) gpu-deviceinfo.cu -o gpu-deviceinfo
	$(NVCC) $(NVCC_OPT) gpu-thread1.cu -o gpu-thread1

clean:
	-rm -f gpu-example
	-rm -f gpu-matrix
	-rm -f gpu-deviceinfo
	-rm -f gpu-thread1

