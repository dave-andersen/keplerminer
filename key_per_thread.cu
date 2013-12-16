#include <stdio.h>
#include <inttypes.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 192

__global__
void hasher_kernel(uint32_t *B, uint32_t *Bx) {
  /* From scrypt.c */
  uint32_t j;

  /* Two iterations of salsa20_8 */
  uint32_t x00,x01,x02,x03,x04,x05,x06,x07,x08,x09,x10,x11,x12,x13,x14,x15;
  size_t i;
  __shared__ uint32_t Bcache[THREADS_PER_BLOCK*16];
  __shared__ uint32_t Bxcache[THREADS_PER_BLOCK*16];
  
	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;

	// parallel test init
#if 0
	B[tid*16+0] = tid; B[tid*16+1] = tid; B[tid*16+2] = tid;
	B[tid*16+3] = tid; B[tid*16+4] = tid; B[tid*16+5] = tid;
	B[tid*16+6] = tid; B[tid*16+7] = tid; B[tid*16+8] = tid;
	B[tid*16+9] = tid; B[tid*16+10] = tid; B[tid*16+11] = tid;
	B[tid*16+12] = tid; B[tid*16+13] = tid; B[tid*16+14] = tid;
	B[tid*16+15] = tid;
#endif

	Bx[tid*16+0] = tid; Bx[tid*16+1] = tid; Bx[tid*16+2] = tid;
	Bx[tid*16+3] = tid; Bx[tid*16+4] = tid; Bx[tid*16+5] = tid;
	Bx[tid*16+6] = tid; Bx[tid*16+7] = tid; Bx[tid*16+8] = tid;
	Bx[tid*16+9] = tid; Bx[tid*16+10] = tid; Bx[tid*16+11] = tid;
	Bx[tid*16+12] = tid; Bx[tid*16+13] = tid; Bx[tid*16+14] = tid;
	Bx[tid*16+15] = tid;


	#pragma unroll
	for (int i = 0; i < 16; i++) {
	  uint32_t b, bx;
	  b = B[tid*16+0];
	  bx = Bx[tid*16+0];
	  Bxcache[threadIdx.x + (16*i)] = bx;
	  Bcache[threadIdx.x + (16*i)] = b;
	}

	/* This is adapted at the end of the iteration now */
	x00 = (Bcache[threadIdx.x + (16*0)] ^= Bxcache[threadIdx.x + (16*0)]);
	x01 = (Bcache[threadIdx.x + (16*1)] ^= Bxcache[threadIdx.x + (16*1)]);
	x02 = (Bcache[threadIdx.x + (16*2)] ^= Bxcache[threadIdx.x + (16*2)]);
	x03 = (Bcache[threadIdx.x + (16*3)] ^= Bxcache[threadIdx.x + (16*3)]);
	x04 = (Bcache[threadIdx.x + (16*4)] ^= Bxcache[threadIdx.x + (16*4)]);
	x05 = (Bcache[threadIdx.x + (16*5)] ^= Bxcache[threadIdx.x + (16*5)]);
	x06 = (Bcache[threadIdx.x + (16*6)] ^= Bxcache[threadIdx.x + (16*6)]);
	x07 = (Bcache[threadIdx.x + (16*7)] ^= Bxcache[threadIdx.x + (16*7)]);
	x08 = (Bcache[threadIdx.x + (16*8)] ^= Bxcache[threadIdx.x + (16*8)]);
	x09 = (Bcache[threadIdx.x + (16*9)] ^= Bxcache[threadIdx.x + (16*9)]);
	x10 = (Bcache[threadIdx.x + (16*10)] ^= Bxcache[threadIdx.x + (16*10)]);
	x11 = (Bcache[threadIdx.x + (16*11)] ^= Bxcache[threadIdx.x + (16*11)]);
	x12 = (Bcache[threadIdx.x + (16*12)] ^= Bxcache[threadIdx.x + (16*12)]);
	x13 = (Bcache[threadIdx.x + (16*13)] ^= Bxcache[threadIdx.x + (16*13)]);
	x14 = (Bcache[threadIdx.x + (16*14)] ^= Bxcache[threadIdx.x + (16*14)]);
	x15 = (Bcache[threadIdx.x + (16*15)] ^= Bxcache[threadIdx.x + (16*15)]);

#define R(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
#define NITERS 2048 // should be 2048 for full version
	for (j = 0; j < NITERS; j++) {

#pragma unroll
	  for (i = 0; i < 4; i++) {
                /* Operate on columns. */
                x04 ^= R(x00+x12, 7);        x09 ^= R(x05+x01, 7);        x14 ^= R(x10+x06, 7);        x03 ^= R(x15+x11, 7);
                x08 ^= R(x04+x00, 9);        x13 ^= R(x09+x05, 9);        x02 ^= R(x14+x10, 9);        x07 ^= R(x03+x15, 9);
                x12 ^= R(x08+x04,13);        x01 ^= R(x13+x09,13);        x06 ^= R(x02+x14,13);        x11 ^= R(x07+x03,13);
                x00 ^= R(x12+x08,18);        x05 ^= R(x01+x13,18);        x10 ^= R(x06+x02,18);        x15 ^= R(x11+x07,18);

                /* Operate on rows. */
                x01 ^= R(x00+x03, 7);        x06 ^= R(x05+x04, 7);        x11 ^= R(x10+x09, 7);        x12 ^= R(x15+x14, 7);
                x02 ^= R(x01+x00, 9);        x07 ^= R(x06+x05, 9);        x08 ^= R(x11+x10, 9);        x13 ^= R(x12+x15, 9);
                x03 ^= R(x02+x01,13);        x04 ^= R(x07+x06,13);        x09 ^= R(x08+x11,13);        x14 ^= R(x13+x12,13);
                x00 ^= R(x03+x02,18);        x05 ^= R(x04+x07,18);        x10 ^= R(x09+x08,18);        x15 ^= R(x14+x13,18);
        }

#define GETBX(REG, OFFSET)			\
	REG += Bcache[threadIdx.x + (16*OFFSET)]; \
	Bcache[threadIdx.x + (16*OFFSET)] = REG; \
	REG ^= Bxcache[threadIdx.x + (16*OFFSET)]; \
	Bxcache[threadIdx.x + (16*0)] = REG

	GETBX(x00, 0);
	GETBX(x01, 1);
	GETBX(x02, 2);
	GETBX(x03, 3);
	GETBX(x04, 4);
	GETBX(x05, 5);
	GETBX(x06, 6);
	GETBX(x07, 7);
	GETBX(x08, 8);
	GETBX(x09, 9);
	GETBX(x10, 10);
	GETBX(x11, 11);
	GETBX(x12, 12);
	GETBX(x13, 13);
	GETBX(x14, 14);
	GETBX(x15, 15);

#pragma unroll
        for (i = 0; i < 4; i++) {
                /* Operate on columns. */
                x04 ^= R(x00+x12, 7);        x09 ^= R(x05+x01, 7);        x14 ^= R(x10+x06, 7);        x03 ^= R(x15+x11, 7);
                x08 ^= R(x04+x00, 9);        x13 ^= R(x09+x05, 9);        x02 ^= R(x14+x10, 9);        x07 ^= R(x03+x15, 9);
                x12 ^= R(x08+x04,13);        x01 ^= R(x13+x09,13);        x06 ^= R(x02+x14,13);        x11 ^= R(x07+x03,13);
                x00 ^= R(x12+x08,18);        x05 ^= R(x01+x13,18);        x10 ^= R(x06+x02,18);        x15 ^= R(x11+x07,18);

                /* Operate on rows. */
                x01 ^= R(x00+x03, 7);        x06 ^= R(x05+x04, 7);        x11 ^= R(x10+x09, 7);        x12 ^= R(x15+x14, 7);
                x02 ^= R(x01+x00, 9);        x07 ^= R(x06+x05, 9);        x08 ^= R(x11+x10, 9);        x13 ^= R(x12+x15, 9);
                x03 ^= R(x02+x01,13);        x04 ^= R(x07+x06,13);        x09 ^= R(x08+x11,13);        x14 ^= R(x13+x12,13);
                x00 ^= R(x03+x02,18);        x05 ^= R(x04+x07,18);        x10 ^= R(x09+x08,18);        x15 ^= R(x14+x13,18);
        }

#undef R
#define SAVEBX(REG, OFFSET)			\
	REG += Bxcache[threadIdx.x + (16*OFFSET)];	\
	Bxcache[threadIdx.x + (16*OFFSET)] = REG;	\
	/* Leave the register set for the next loop iteration and the exit */ \
	REG ^= Bcache[threadIdx.x + (16*OFFSET)] \

        SAVEBX(x00, 0);
        SAVEBX(x01, 1);
        SAVEBX(x02, 2);
        SAVEBX(x03, 3);
        SAVEBX(x04, 4);
        SAVEBX(x05, 5);
        SAVEBX(x06, 6);
        SAVEBX(x07, 7);
        SAVEBX(x08, 8);
        SAVEBX(x09, 9);
        SAVEBX(x10, 10);
        SAVEBX(x11, 11);
        SAVEBX(x12, 12);
        SAVEBX(x13, 13);
        SAVEBX(x14, 14);
        SAVEBX(x15, 15);
	}

        B[tid*16+ 0] = x00;
        B[tid*16+ 1] = x01;
        B[tid*16+ 2] = x02;
        B[tid*16+ 3] = x03;
        B[tid*16+ 4] = x04;
        B[tid*16+ 5] = x05;
        B[tid*16+ 6] = x06;
        B[tid*16+ 7] = x07;
        B[tid*16+ 8] = x08;
        B[tid*16+ 9] = x09;
        B[tid*16+10] = x10;
        B[tid*16+11] = x11;
        B[tid*16+12] = x12;
        B[tid*16+13] = x13;
        B[tid*16+14] = x14;
        B[tid*16+15] = x15;
	return;
}

double timeval_diff(const struct timeval * const start, const struct timeval * const end)
{
    /* Calculate the second difference*/
    double r = end->tv_sec - start->tv_sec;

    /* Calculate the microsecond difference */
    if (end->tv_usec > start->tv_usec)
        r += (end->tv_usec - start->tv_usec)/1000000.0;
    else if (end->tv_usec < start->tv_usec)
        r -= (start->tv_usec - end->tv_usec)/1000000.0;

    return r;
}

int main() {
#define SCRYPT_WIDTH 16
#define N 1024 * 128


  printf("hi\n");

  struct timeval tv_start, tv_end;
  uint32_t *dev_a, *dev_b;

  uint32_t *mydat = (uint32_t *)malloc(N*SCRYPT_WIDTH*sizeof(uint32_t));
  printf("First malloc\n");fflush(stdout);
  printf("Foo: %lu\n", N*SCRYPT_WIDTH*sizeof(uint32_t));fflush(stdout);
  for (int i = 0; i < N*SCRYPT_WIDTH; i++) {
    mydat[i] = i+tv_start.tv_sec; // Confuse the optimizer. 
  }
  if (cudaMalloc((void **) &dev_a, N*SCRYPT_WIDTH*sizeof(uint32_t)) != cudaSuccess) {
    fprintf(stderr, "Could not allocate array\n");
    exit(0);
  }
  gettimeofday(&tv_start, NULL);
  cudaMemcpy(dev_a, mydat, N*SCRYPT_WIDTH*sizeof(uint32_t), cudaMemcpyHostToDevice);
  printf("Second malloc\n");fflush(stdout);
  if (cudaMalloc((void **) &dev_b, N*SCRYPT_WIDTH*sizeof(uint32_t)) != cudaSuccess) {
    fprintf(stderr, "Could not allocate array\n");
    exit(0);
  }
  printf("Starting kernel\n");fflush(stdout);
  hasher_kernel<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_a, dev_b);
  printf("Memcpy result\n");

  cudaMemcpy(mydat, dev_a, N*SCRYPT_WIDTH*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  gettimeofday(&tv_end, NULL);
#if 1
  for (int i = 0; i < 10; i++) {
    printf("%x\n", mydat[i*SCRYPT_WIDTH]);
  }
#endif

  cudaFree(dev_a);
  cudaFree(dev_b);
  free(mydat);
  cudaDeviceReset();


  printf("%2.2f\n", timeval_diff(&tv_start, &tv_end));
  printf("Done\n");
}
