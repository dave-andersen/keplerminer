/* Copyright (C) 2013 David G. Andersen.  All rights reserved.
 *
 * Use of this code is covered under the Apache 2.0 license, which
 * can be found in the file "LICENSE"
 */

#include <sys/time.h>
#include "hasher.h"
#include "scrypt_cores.cu"

/* write_keys writes the 8 keys being processed by a warp to the global
 * scratchpad.  To effectively use memory bandwidth, it performs the writes
 * (and reads, for read_keys) 128 bytes at a time per memory location
 * by __shfl'ing the 4 entries in bx to the threads in the next-up
 * thread group.  It then has eight threads together perform uint4
 * (128 bit) writes to the destination region.  This seems to make
 * quite effective use of memory bandwidth.  An approach that spread
 * uint32s across more threads was slower because of the increased
 * computation it required.
 *
 * "start" is the loop iteration producing the write - the offset within
 * the block's memory.
 *
 * Internally, this algorithm first __shfl's the 4 bx entries to
 * the next up thread group, and then uses a conditional move to
 * ensure that odd-numbered thread groups exchange the b/bx ordering
 * so that the right parts are written together.
 *
 * Thanks to Babu for helping design the 128-bit-per-write version.
 * 
 * _direct lets the caller specify the absolute start location instead of
 * the relative start location, as an attempt to reduce some recomputation.
 */

__device__
inline void write_keys_direct(const uint32_t b[4], const uint32_t bx[4], uint32_t *scratch, uint32_t start) {

  uint4 t, t2;
  t.x = b[0]; t.y = b[1]; t.z = b[2]; t.w = b[3];

  int target_thread = (threadIdx.x + 4)%32;
  t2.x = __shfl((int)bx[0], target_thread);
  t2.y = __shfl((int)bx[1], target_thread);
  t2.z = __shfl((int)bx[2], target_thread);
  t2.w = __shfl((int)bx[3], target_thread);

  int t2_start = __shfl((int)start, target_thread) + 4;

  bool c = (threadIdx.x & 0x4);

  int loc = c ? t2_start : start;
  *((uint4 *)(&scratch[loc])) = (c ? t2 : t);
  loc = c ? start : t2_start;
  *((uint4 *)(&scratch[loc])) = (c ? t : t2);
}

__device__
inline void write_keys(const uint32_t b[4], const uint32_t bx[4], uint32_t *scratch, uint32_t start) {
  int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_SCRYPT_BLOCK;
  start = scrypt_block*SCRYPT_SCRATCH_PER_BLOCK + (32*start) + 8*(threadIdx.x%4);
  write_keys_direct(b, bx, scratch, start);
}


inline __device__ void read_xor_keys_direct(uint32_t b[4], uint32_t bx[4], const __restrict__ uint32_t *scratch, uint32_t start) {

  uint4 t, t2;

  // Tricky bit:  We do the work on behalf of thread+4, but then when
  // we steal, we have to steal from (thread+28)%32 to get the right
  // stuff back.
  start = __shfl((int)start, (threadIdx.x & 0x7c)) + 8*(threadIdx.x%4);

  int target_thread = (threadIdx.x + 4)%32;
  int t2_start = __shfl((int)start, target_thread) + 4;

  bool c = (threadIdx.x & 0x4);

  int loc = c ? t2_start : start;
  t = *((uint4 *)(&scratch[loc]));
  loc = c ? start : t2_start;
  t2 = *((uint4 *)(&scratch[loc]));

  uint4 tmp = t; t = (c ? t2 : t); t2 = (c ? tmp : t2); 
  
  b[0] ^= t.x; b[1] ^= t.y; b[2] ^= t.z; b[3] ^= t.w;

  int steal_target = (threadIdx.x + 28)%32;

  bx[0] ^= __shfl((int)t2.x, steal_target);
  bx[1] ^= __shfl((int)t2.y, steal_target);
  bx[2] ^= __shfl((int)t2.z, steal_target);
  bx[3] ^= __shfl((int)t2.w, steal_target);
}


inline __device__ void read_xor_keys(uint32_t b[4], uint32_t bx[4], const __restrict__ uint32_t *scratch, uint32_t start) {
  int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_SCRYPT_BLOCK;
  start = scrypt_block*SCRYPT_SCRATCH_PER_BLOCK + (32*start);
  read_xor_keys_direct(b, bx, scratch, start);
}


inline __device__ void primary_order_shuffle(uint32_t b[4], uint32_t bx[4]) {
  /* Inner loop shuffle targets */
  int x1_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+1)&0x3);
  int x2_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+2)&0x3);
  int x3_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+3)&0x3);
  
  b[3] = __shfl((int)b[3], x1_target_lane);
  b[2] = __shfl((int)b[2], x2_target_lane);
  b[1] = __shfl((int)b[1], x3_target_lane);
  uint32_t tmp = b[1]; b[1] = b[3]; b[3] = tmp;
  
  bx[3] = __shfl((int)bx[3], x1_target_lane);
  bx[2] = __shfl((int)bx[2], x2_target_lane);
  bx[1] = __shfl((int)bx[1], x3_target_lane);
  tmp = bx[1]; bx[1] = bx[3]; bx[3] = tmp;
}

/*
 * load_key loads a 32*32bit key from a contiguous region of memory in B.
 * The input keys are in external order (i.e., 0, 1, 2, 3, ...).
 * After loading, each thread has its four b and four bx keys stored
 * in internal processing order.
 */

inline __device__ void load_key(const uint32_t *B, uint32_t b[4], uint32_t bx[4]) {
  int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_SCRYPT_BLOCK;
  int key_offset = scrypt_block * 32;
  uint32_t thread_in_block = threadIdx.x % 4;

  // Read in permuted order.  Key loads are not our bottleneck right now.
  for (int i = 0; i < 4; i++) {
    b[i] = B[key_offset + 4*thread_in_block + (thread_in_block+i)%4];
    bx[i] = B[key_offset + 4*thread_in_block +  (thread_in_block+i)%4 + 16];
  }

  primary_order_shuffle(b, bx);
  
}

/*
 * store_key performs the opposite transform as load_key, taking
 * internally-ordered b and bx and storing them into a contiguous
 * region of B in external order.
 */

inline __device__ void store_key(uint32_t *B, uint32_t b[4], uint32_t bx[4]) {
  int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_SCRYPT_BLOCK;
  int key_offset = scrypt_block * 32;
  uint32_t thread_in_block = threadIdx.x % 4;

  primary_order_shuffle(b, bx);

  for (int i = 0; i < 4; i++) {
    B[key_offset + 4*thread_in_block + (thread_in_block+i)%4] = b[i];
    B[key_offset + 4*thread_in_block + (thread_in_block+i)%4 + 16] = bx[i];
  }
}


/*
 * salsa_xor_core does the equivalent of the xor_salsa8 loop from
 * tarsnap's implementation of scrypt.  The original scrypt called:
 *   xor_salsa8(&X[0], &X[16]);   <-- the "b" loop
 *   xor_salsa8(&X[16], &X[0]);   <-- the "bx" loop
 * This version is unrolled to handle both of these loops in a single
 * call to avoid unnecessary data movement.
 */

inline __device__ void salsa_xor_core(uint32_t b[4], uint32_t bx[4], uint32_t x[4],
				      const int x1_target_lane,
				      const int x2_target_lane,
				      const int x3_target_lane) {
  uint32_t tmp;

#pragma unroll
    for (int i = 0; i < 4; i++) {
      b[i] ^= bx[i];
      x[i] = b[i];
    }

#define XOR_ROTATE_ADD(dst, s1, s2, amt) do { tmp = x[s1]+x[s2]; x[dst] ^= ((tmp<<amt)|(tmp>>(32-amt))); } while(0)

    // Enter in "column" mode (t0 has 0, 4, 8, 12)

    for (int j = 0; j < 4; j++) {
    
      // Mixing phase of salsa
      XOR_ROTATE_ADD(1, 0, 3, 7);
      XOR_ROTATE_ADD(2, 1, 0, 9);
      XOR_ROTATE_ADD(3, 2, 1, 13);
      XOR_ROTATE_ADD(0, 3, 2, 18);
      
      /* Transpose rows and columns. */
      /* Unclear if this optimization is needed:  These are ordered based
       * upon the dependencies needed in the later xors.  Compiler should be
       * able to figure this out, but might as well give it a hand. */
      x[1] = __shfl((int)x[1], x3_target_lane);
      x[3] = __shfl((int)x[3], x1_target_lane);
      x[2] = __shfl((int)x[2], x2_target_lane);
      
      /* The next XOR_ROTATE_ADDS could be written to be a copy-paste of the first,
       * but the register targets are rewritten here to swap x[1] and x[3] so that
       * they can be directly shuffled to and from our peer threads without
       * reassignment.  The reverse shuffle then puts them back in the right place.
       */
      
      XOR_ROTATE_ADD(3, 0, 1, 7);
      XOR_ROTATE_ADD(2, 3, 0, 9);
      XOR_ROTATE_ADD(1, 2, 3, 13);
      XOR_ROTATE_ADD(0, 1, 2, 18);
      
      x[3] = __shfl((int)x[3], x3_target_lane);
      x[1] = __shfl((int)x[1], x1_target_lane);
      x[2] = __shfl((int)x[2], x2_target_lane);
    }

    for (int i = 0; i < 4; i++) {
      b[i] += x[i];
      // The next two lines are the beginning of the BX-centric loop iteration
      bx[i] ^= b[i];
      x[i] = bx[i];
    }

    // This is a copy of the same loop above, identical but stripped of comments.
    // Duplicated so that we can complete a bx-based loop with fewer register moves.
    for (int j = 0; j < 4; j++) {
      XOR_ROTATE_ADD(1, 0, 3, 7);
      XOR_ROTATE_ADD(2, 1, 0, 9);
      XOR_ROTATE_ADD(3, 2, 1, 13);
      XOR_ROTATE_ADD(0, 3, 2, 18);
      
      x[1] = __shfl((int)x[1], x3_target_lane);
      x[3] = __shfl((int)x[3], x1_target_lane);
      x[2] = __shfl((int)x[2], x2_target_lane);
      
      XOR_ROTATE_ADD(3, 0, 1, 7);
      XOR_ROTATE_ADD(2, 3, 0, 9);
      XOR_ROTATE_ADD(1, 2, 3, 13);
      XOR_ROTATE_ADD(0, 1, 2, 18);
      
      x[3] = __shfl((int)x[3], x3_target_lane);
      x[1] = __shfl((int)x[1], x1_target_lane);
      x[2] = __shfl((int)x[2], x2_target_lane);
    }

    // At the end of these iterations, the data is in primary order again.
#undef XOR_ROTATE_ADD

    for (int i = 0; i < 4; i++) {
      bx[i] += x[i];
    }
}


/* 
 * The hasher_gen_kernel operates on a group of 1024-bit input keys
 * in B, stored as:
 * B = {  k1B  k1Bx  k2B  k2Bx ... }
 * and fills up the scratchpad with the iterative hashes derived from
 * those keys:
 * scratch { k1h1B k1h1Bx K1h2B K1h2Bx ... K2h1B K2h1Bx K2h2B K2h2Bx ... }
 * scratch is 1024 times larger than the input keys B.
 * It is extremely important to stream writes effectively into scratch;
 * less important to coalesce the reads from B.
 *
 * Key ordering note:  Keys are input from B in "original" order:
 * K = {k1, k2, k3, k4, k5, ..., kx15, kx16, kx17, ..., kx31 }
 * After inputting into kernel_gen, each component k and kx of the
 * key is transmuted into a permuted internal order to make processing faster:
 * K = k, kx   with:
 * k = 0, 4, 8, 12,   5, 9, 13, 1,    10, 14, 2, 6,   15, 3, 7, 11
 * and similarly for kx.
 */

__global__
void hasher_gen_kernel(__restrict__ uint32_t *B, __restrict__ uint32_t *scratch) {

  /* Each thread operates on four of the sixteen B and Bx variables.  Thus,
   * each key is processed by four threads in parallel.  salsa_scrypt_core
   * internally shuffles the variables between threads (and back) as
   * needed.
   */
  uint32_t b[4], bx[4], x[4];

  load_key(B, b, bx);
  
  /* Inner loop shuffle targets */
  int x1_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+1)&0x3);
  int x2_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+2)&0x3);
  int x3_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+3)&0x3);

  int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_SCRYPT_BLOCK;
  int start = scrypt_block*SCRYPT_SCRATCH_PER_BLOCK + 8*(threadIdx.x%4);

  for (int i = 0; i < 1024; i++) {
    write_keys_direct(b, bx, scratch, start+32*i);
    salsa_xor_core(b, bx, x, x1_target_lane, x2_target_lane, x3_target_lane);
  }

  store_key(B, b, bx);
}


/* 
 * hasher_hash_kernel runs the second phase of scrypt after the scratch 
 * buffer is filled with the iterative hashes:  It bounces through
 * the scratch buffer in pseudorandom order, mixing the key as it goes.
 */

__global__
void hasher_hash_kernel(__restrict__ uint32_t *B, const __restrict__ uint32_t *scratch) {

  /* Each thread operates on a group of four variables that must be processed
   * together.  Shuffle between threaads in a warp between iterations.
   */
  uint32_t b[4], bx[4], x[4];

  load_key(B, b, bx);

  /* Inner loop shuffle targets */
  int x1_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+1)&0x3);
  int x2_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+2)&0x3);
  int x3_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+3)&0x3);

  for (int i = 0; i < 1024; i++) {

    // Bounce through the key space and XOR the new keys in.
    // Critical thing:  (X[16] & 1023) tells us the next slot to read.
    // X[16] in the original is bx[0]
    int slot = bx[0] & 1023;
    read_xor_keys(b, bx, scratch, slot);
    salsa_xor_core(b, bx, x, x1_target_lane, x2_target_lane, x3_target_lane);
  }

  store_key(B, b, bx);
}

/*
 * hasher_combo_kernel runs the functions of both hasher_gen_kernel
 * and hasher_hash_kernel in a single invocation.  It is
 * designed to reduce kernel launch downtime a bit and omit one
 * intermediate store_key operation to global memory.  This is faster on
 * my GT 550m, but seems a bit slower on a Tesla, probably because one of
 * the two individual kernels can use fewer registers alone.
 */

__global__
void hasher_combo_kernel(__restrict__ uint32_t *B, __restrict__ uint32_t *scratch) {

  uint32_t b[4], bx[4], x[4];

  load_key(B, b, bx);
  
  int x1_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+1)&0x3);
  int x2_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+2)&0x3);
  int x3_target_lane = (threadIdx.x & 0xfc) + (((threadIdx.x & 0x03)+3)&0x3);

  int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_SCRYPT_BLOCK;
  int start = scrypt_block*SCRYPT_SCRATCH_PER_BLOCK + 8*(threadIdx.x%4);

  for (int i = 0; i < 1024; i++) {
    write_keys_direct(b, bx, scratch, start);
    start += 32;
    salsa_xor_core(b, bx, x, x1_target_lane, x2_target_lane, x3_target_lane);
  }

  start = scrypt_block*SCRYPT_SCRATCH_PER_BLOCK;

  for (int i = 0; i < 1024; i++) {
    int slot = bx[0] & 1023;
    read_xor_keys_direct(b, bx, scratch, start+32*slot);
    salsa_xor_core(b, bx, x, x1_target_lane, x2_target_lane, x3_target_lane);
  }

  store_key(B, b, bx);
}


/*
 * scrypt_hash_start_kernel takes a job description (job) and a starting nonce number (n_base)
 * and generates (batchsize) starting hashes using HMAC_SHA256 and PBKDF2_SHA256.
 * This is the first step in scrypt computation before the salsa core executions
 * that introduce the memory difficulty.
 * 
 * This function stores three outputs:
 *   output - the 1024-bit intermediate state fed to scrypt_core
 *   tstate, ostate - intermediate PBKDF2 state that needs to be used again
 *                    after the execution of scrypt_core.
 */

__global__
void scrypt_hash_start_kernel(const __restrict__ scan_job *job, __restrict__ uint32_t *output, __restrict__ uint32_t *ostate_out, __restrict__ uint32_t *tstate_out, uint32_t n_base) {
  uint32_t tstate[8];
  uint32_t ostate[8];
  uint32_t data[20];

  int blockid = (blockIdx.x*blockDim.x + threadIdx.x);

  /* data:  the input.
   * tstate, ostate, output must have sufficient space
   * -> batchsize * 8 * sizeof(uint32_t)
   */

  /* Trivial implementation:  Each thread processes one key.  This is lame, but
   * PBKDF related processing is only about 3.5% of runtime right now. */
  const uint32_t *in_data = job->data;
  uint64_t blockstart = blockid*32;
  {
    uint4 d;
    for (int i = 0; i < 20; i+= 4) {
      d = *(uint4 *)&in_data[i];
      data[i] = d.x; data[i+1] = d.y; data[i+2] = d.z; data[i+3] = d.w;
    }
  }
  data[19] = n_base + blockid;

  read_8_as_uint4(job->initial_midstate, tstate);
  dev_HMAC_SHA256_80_init(data, tstate, ostate);

  /* This writes directly to output and does no shuffling or cleverness
   * to coalesce writes.  Unnecessary memory transactions, but not worth
   * fixing yet - PBKDF is less than 4% of runtime overall still. */

  dev_PBKDF2_SHA256_80_128(tstate, ostate, data, &output[blockstart]);

  /* Write out (and read back) tstate and ostate interleaved across
     threads to improve mem b/w.  Easy change, though small benefit.  */
  int bigblockstart = (blockIdx.x*blockDim.x)*8;
  int threadsInBlock = blockDim.x;

  for (int i = 0; i < 8; i++) {
    tstate_out[bigblockstart + threadsInBlock*i + threadIdx.x ] = tstate[i];
    ostate_out[bigblockstart + threadsInBlock*i + threadIdx.x ] = ostate[i];
  }
  
}

/*
 * scrypt_hash_finish_kernel takes the output state from scrypt_core and
 * recombines it with the saved PBKDF2 tstate and ostate from
 * scrypt_hash_start_kernel to produce an output key.
 *
 * It then compares the output key ("hash") to the target number
 * specified in the job to determine whether hash < job->target.
 * If it is, it puts its block/thread ID + 1 into *dev_output.
 * If it is not, it does nothing.  The caller should ensure that
 * dev_output is zero before the call to scrypt_hash_finish_kernel, and
 * should subtract one from non-zero output to determine the
 * actual thread ID (and thus, the nonce used for hashing).
 * 
 * This method does not guarantee that the lowest-numbered or
 * smallest acceptable output is returned, merely that one satisfying
 * output is returned if any exist.
 */

__global__ void scrypt_hash_finish_kernel(const __restrict__ uint32_t *dev_keys, __restrict__ uint32_t *dev_tstate, __restrict__ uint32_t *dev_ostate, __restrict__ uint32_t *dev_output, __restrict__ const scan_job *job) {
  uint32_t tstate[8];
  uint32_t ostate[8];
  uint32_t hash[32];

  int blockid = (blockIdx.x*blockDim.x + threadIdx.x);

  int bigblockstart = blockIdx.x*blockDim.x*8;
  int threadsInBlock = blockDim.x;

  /* As in start_kernel, reads tstate/ostate interleaved between threads */
  for (int i = 0; i < 8; i++) {
    tstate[i] = dev_tstate[bigblockstart + threadsInBlock*i + threadIdx.x ];
    ostate[i] = dev_ostate[bigblockstart + threadsInBlock*i + threadIdx.x ];
  }

  uint64_t blockstart = blockid*32;
  uint4 t;
  for (int i = 0; i < 32; i+= 4) {
    t = *(uint4 *)&dev_keys[blockstart+i];
    hash[i] = t.x; hash[i+1] = t.y; hash[i+2] = t.z; hash[i+3] = t.w;
  }
  
  dev_PBKDF2_SHA256_128_32(tstate, ostate, hash);

  uint32_t foundit = 0x00000000;
  uint32_t maybe = 0xffffffff;

  uint32_t target[8];
  read_8_as_uint4(job->target, target);

  for (int j = 7; j >= 0; j--) {
    uint32_t tmp = swab32(ostate[j]);
    maybe = (maybe & (tmp <= target[j]));
    foundit = (foundit | (maybe & (tmp < target[j])));
  }
  foundit = foundit ? (blockid+1) : foundit;

  if (foundit) {
    // Finding the lowest doesn't matter.  Just let the first writer win.
    uint32_t oldval = atomicCAS(&dev_output[0], 0, foundit);
  }
}

/* Unit test kernel to expose loading, writing, reading, and storing keys */

__global__
void test_load_store_kernel(__restrict__ uint32_t *B, __restrict__ uint32_t *scratch) {
  uint32_t b[4], bx[4];
  load_key(B, b, bx);
  for (int i = 0; i < 4; i++) { b[i]++; bx[i]++; }
  for (int slot = 0; slot < 1024; slot++) {
    write_keys(b, bx, scratch, slot);
    for (int i = 0; i < 4; i++) { b[i] = bx[i] = 0; }
    read_xor_keys(b, bx, scratch, slot);
  }
  store_key(B, b, bx);
}

/* 
 * The CudaHasher constructor does nothing.  
 * You must call Initialize() and check the error code.
 */

CudaHasher::CudaHasher() :
  dev_keys(NULL), dev_scratch(NULL), dev_output(NULL), 
  dev_tstate(NULL), dev_ostate(NULL), scan_output(NULL),
  dev_job(NULL) {
  //  dev_keys = NULL;
  //  dev_scratch = NULL;
}

/*
 * Initialize() sets up the GPU state.
 * Between creation and destruction, a lot of memory
 * may be eaten on the GPU.  For performance, it is
 * _not_ freed when ComputeHashes is not running.
 * ergo:  If you're going to be idle for a long time
 * and want to be nice to other uses of the GPU, destroy
 * the CudaHasher and create a new one.
 *
 * If initialize fails, destroy the object to reset the GPU.
 */

int CudaHasher::Initialize() {
  cudaError_t error;
  /* Stop eating CPU while waiting for results! */
  error = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  if (error != cudaSuccess) { 
    fprintf(stderr, "Could not set blocking sync (error %d)\n", error);
  }

  /* Determine device memory to size batch */
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  //printf("Initializing.  Device has %ld free of %ld total bytes of memory\n", free, total);
  int mem_per_job = 160000; /* A little conservative */
  int max_batchsize = free / mem_per_job;

  /* We need 4 threads per work unit and each block should have 192 threads */
  /* The number of blocks should also probably be a multiple of the
   * number of multiprocessors.  I'm using 8 here as a simple way
   * to get a pretty-OK answer that's probably not optimal for big GPUs */
  batchsize = (max_batchsize/(2*THREADS_PER_CUDA_BLOCK))*2*THREADS_PER_CUDA_BLOCK;
  n_blocks = (batchsize*THREADS_PER_SCRYPT_BLOCK/THREADS_PER_CUDA_BLOCK);

  error = cudaMalloc((void **) &dev_job, sizeof(scan_job));
  if (error != cudaSuccess) {
    fprintf(stderr, "Could not allocate CUDA array, error code %d, line(%d)\n", error, __LINE__);
    dev_job = NULL;
    return -1;
  }

  error = cudaMalloc((void **) &dev_keys, sizeof(scrypt_hash) * batchsize);
  if (error != cudaSuccess) {
    fprintf(stderr, "Could not allocate CUDA array, error code %d, line(%d)\n", error, __LINE__);
    dev_keys = NULL;
    return -1;
  }

  size_t scratchBufSize = sizeof(scrypt_hash)*1024*batchsize;

  error = cudaMalloc((void **) &dev_scratch, scratchBufSize);
  if (error != cudaSuccess) {
    fprintf(stderr, "Could not allocate CUDA array, error code %d, line(%d)\n", error, __LINE__);
    dev_scratch = NULL;
    return -1;
  }

  /* dev_output holds one int per warp indicating if a thread in that warp solved the block */
  error = cudaMalloc((void **) &dev_output, sizeof(uint32_t));
  if (error != cudaSuccess) {
    fprintf(stderr, "Could not allocate CUDA array, error code %d, line(%d)\n", error, __LINE__);
    dev_output = NULL;
    return -1;
  }

  error = cudaMalloc((void **) &dev_tstate, 8*sizeof(uint32_t) * batchsize);
  if (error != cudaSuccess) {
    fprintf(stderr, "Could not allocate CUDA array, error code %d, line(%d)\n", error, __LINE__);
    dev_tstate = NULL;
    return -1;
  }

  error = cudaMalloc((void **) &dev_ostate, 8 * sizeof(uint32_t) * batchsize);
  if (error != cudaSuccess) {
    fprintf(stderr, "Could not allocate CUDA array, error code %d, line(%d)\n", error, __LINE__);
    dev_ostate = NULL;
    return -1;
  }

  scan_output = (uint32_t *)malloc(sizeof(uint32_t));
  if (!scan_output) {
    perror("Could not allocate scan output buffer (host)");
    return -1;
  }

  cudaFuncSetCacheConfig(hasher_hash_kernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(hasher_gen_kernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(hasher_combo_kernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(scrypt_hash_start_kernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(scrypt_hash_finish_kernel, cudaFuncCachePreferL1);

  return 0;
}
  
CudaHasher::~CudaHasher() {
  /* Free host memory */
  if (scan_output) free(scan_output);

  /* Free device memory */
  if (dev_scratch != NULL) cudaFree(dev_scratch);
  if (dev_keys != NULL) cudaFree(dev_keys);
  if (dev_output != NULL) cudaFree(dev_output);
  if (dev_tstate != NULL) cudaFree(dev_tstate);
  if (dev_ostate != NULL) cudaFree(dev_ostate);
  if (dev_job != NULL) cudaFree(dev_job);

  cudaDeviceReset();
}

static void init_test_keys(scrypt_hash *keys_in, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < SCRYPT_WIDTH; j++) {
      keys_in[i].b[j] = SCRYPT_WIDTH*2*i + j;
    }
    for (int j = SCRYPT_WIDTH; j < 2*SCRYPT_WIDTH; j++) {
      keys_in[i].bx[j-SCRYPT_WIDTH] = SCRYPT_WIDTH*2*i + j;
    }
  }
}

int verify_test_keys(scrypt_hash *keys_in, int n, std::string testname, int extra) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < SCRYPT_WIDTH; j++) {
      if (keys_in[i].b[j] != (SCRYPT_WIDTH*2*i + j + extra)) { 
	fprintf(stderr, "Failed %s validation of keys_in[%d].b[%d]\n", testname.c_str(), i, j);
	fprintf(stderr, "Got %d expected %d\n", keys_in[i].b[j], SCRYPT_WIDTH*2*i+j+extra);
	return -1;
      }
    }
    for (int j = SCRYPT_WIDTH; j < 2*SCRYPT_WIDTH; j++) {
      if (keys_in[i].bx[j-SCRYPT_WIDTH] != (SCRYPT_WIDTH*2*i + j + extra)) {
	fprintf(stderr, "Failed %s validation of keys_in[%d].bx[%d]\n", testname.c_str(), i, j-SCRYPT_WIDTH);
	return -1;
      }
    }
  }
  return 0;
}

int CudaHasher::TestLoadStore() {
  cudaError error;
  int success = -1;
  scrypt_hash *keys_in = (scrypt_hash *)malloc(batchsize * sizeof(struct scrypt_hash));
  if (!keys_in) {
    perror("TestLoadStore:  Could not allocate host keys");
    goto failed;
  }

  init_test_keys(keys_in, batchsize);

  if (verify_test_keys(keys_in, batchsize, "pre-kernel", 0) == -1) { 
    goto failed;
  }

  error = cudaMemcpy(dev_keys, keys_in, sizeof(scrypt_hash) * batchsize, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    fprintf(stderr, "TestLoadStore:  Could not memcpy to device, error %d\n", error);
    goto failed;
  }
  
  test_load_store_kernel<<<n_blocks, THREADS_PER_CUDA_BLOCK>>>(dev_keys, dev_scratch);

  error = cudaMemcpy(keys_in, dev_keys, batchsize * sizeof(scrypt_hash), cudaMemcpyDeviceToHost);

  if (error != cudaSuccess) {
    fprintf(stderr, "TestLoadStore:  Could not memcpy from device, error %d\n", error);
    goto failed;
  }

  if (verify_test_keys(keys_in, batchsize, "post-kernel", 1) == -1) { goto failed; }

  fprintf(stderr, "TestLoadStore passed\n");
  success = 0;

 failed:
  free(keys_in);
  return success;
}


/*
 * The simplest workhorse of our computation:  takes a set of n_hashes which 
 * and computes the output.  Note that if n_hashes is not equal to the batchsize,
 * only n_hashes will be computed.  If it is less, then random hashes will be computed
 * alongside to keep the batch full.  Version 1, remember?
 *
 * Returns when complete.
 *
 * Only one thread should call ComputeHashes at a time on any given device.
 */

int CudaHasher::ComputeHashes(const scrypt_hash *in, scrypt_hash *out, int n_hashes)
{
  cudaError error = cudaMemcpy(dev_keys, in, sizeof(scrypt_hash) * n_hashes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    fprintf(stderr, "Could not memcpy to device, error code %d, line(%d)\n", error, __LINE__);
    return -1;
  }
  
//  hasher_gen_kernel<<<n_blocks, THREADS_PER_CUDA_BLOCK>>>(dev_keys, dev_scratch);
//  hasher_hash_kernel<<<n_blocks, THREADS_PER_CUDA_BLOCK>>>(dev_keys, dev_scratch);
  hasher_combo_kernel<<<n_blocks, THREADS_PER_CUDA_BLOCK>>>(dev_keys, dev_scratch);
  cudaDeviceSynchronize();
  error = cudaMemcpy(out, dev_keys, n_hashes * sizeof(scrypt_hash), cudaMemcpyDeviceToHost);

  if (error != cudaSuccess) {
    fprintf(stderr, "Could not memcpy from device, error code %d, line(%d)\n", error, __LINE__);
    return -1;
  }

  return 0;

}

int CudaHasher::ScanNCoins(uint32_t *pdata, const uint32_t *ptarget, int n_hashes, volatile int *stop, unsigned long *hashes_done)
{
  int n_done = 0;

  uint32_t data[20];
  uint32_t n = pdata[19]; /* sigh, cpuminer */

  memcpy(data, pdata, sizeof(data));
  scan_job j;
  memcpy(j.data, data, sizeof(data));
  memcpy(j.target, ptarget, sizeof(uint32_t)*8);

  /* Set up the job midstate once on the CPU */
  sha256_init(j.initial_midstate);
  sha256_transform(j.initial_midstate, data, 0);
  *scan_output = 0;

  cudaError error = cudaMemcpy(dev_job, &j, sizeof(scan_job), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    fprintf(stderr, "Could not memcpy job to device, error code %d, line(%d)\n", error, __LINE__);
    exit(-1);
  }

  error = cudaMemcpy(dev_output, scan_output, sizeof(uint32_t), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    fprintf(stderr, "Could not memcpy job to device, error code %d, line(%d)\n", error, __LINE__);
    exit(-1);
  }


  while (n_done < n_hashes && (!*stop)) {
    data[19] = n;

    // Note:  The /4 is very important below:  TPCB is set for the scrypt kernel;
    // the hashing kernel uses 1 thread per hash.
    static const int threads = THREADS_PER_CUDA_BLOCK;
    
    scrypt_hash_start_kernel<<<n_blocks/4, threads>>>(dev_job, dev_keys, dev_ostate, dev_tstate, n);
    hasher_combo_kernel<<<n_blocks, threads>>>(dev_keys, dev_scratch);
    scrypt_hash_finish_kernel<<<n_blocks/4, threads>>>(dev_keys, dev_tstate, dev_ostate, dev_output, dev_job);

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      fprintf(stderr, "Kernel execution failed, error code %d, line(%d)\n", error, __LINE__);
      exit(-1);
    }

    error = cudaMemcpy(scan_output, dev_output, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) {
      fprintf(stderr, "Could not memcpy from device, error code %d, line(%d)\n", error, __LINE__);
      exit(-1);
    }

    if (*scan_output != 0) {
      if (hashes_done != NULL) *hashes_done += n_done;
      return n_done + *scan_output - 1; // -1 because of 0 blockIdx
    }

    n_done += batchsize;
    n += batchsize;
  }

  if (hashes_done != NULL) *hashes_done += n_done;
  return -1;
}
