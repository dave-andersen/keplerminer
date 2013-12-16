CFLAGS=-O3 -march=native 
NVFLAGS= --gpu-architecture sm_30 -O3 -g 
#sm_35 -O3  # for Tesla
all: key_per_thread hasher_bench hasher_c.so hasher_test

key_per_thread: key_per_thread.cu
	nvcc $(NVFLAGS) key_per_thread.cu -o key_per_thread

hasher.o: hasher.h hasher.cu
	nvcc $(NVFLAGS) -c hasher.cu -o hasher.o

percival_scrypt.o: percival_scrypt.cu percival_scrypt.h
	nvcc $(NVFLAGS) -c percival_scrypt.cu -o percival_scrypt.o

hasher_bench: hasher_bench.cu hasher.o scrypt_cores.cu
	nvcc $(NVFLAGS) hasher.o hasher_bench.cu -o hasher_bench

hasher_test: hasher_test.cu hasher.o percival_scrypt.cu percival_scrypt.o scrypt_cores.cu
	nvcc $(NVFLAGS) hasher.o hasher_test.cu percival_scrypt.o -o hasher_test

# Not working right now.  Must figure out linking issues (OS X)
hasher_c.o: hasher_c.cu scrypt_cores.cu
	nvcc $(NVFLAGS) -dlink -o hasher_c.o hasher_c.cu hasher.cu

hasher_c.so: hasher_c.cu hasher.o scrypt_cores.cu
	nvcc -Xcompiler -fPIC $(NVFLAGS) --shared -o hasher_c.so hasher_c.cu hasher.o

clean:
	@rm -f key_per_thread hasher_bench hasher_test *.o
