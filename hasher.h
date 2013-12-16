#ifndef _CUDAHASHER_H_
#define _CUDAHASHER_H_

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string>


struct scrypt_hash {
  uint32_t b[16];
  uint32_t bx[16];
} __attribute__((packed));

typedef struct {
  uint32_t data[20];
  uint32_t target[8];
  uint32_t initial_midstate[8];
} scan_job;

class CudaHasher {
public:
  CudaHasher();
  int Initialize();
  int ComputeHashes(const scrypt_hash *in, scrypt_hash *out, int n_hashes);
  ~CudaHasher();

  int ScanNCoins(uint32_t *pdata, const uint32_t *ptarget, int n, volatile int *stop, unsigned long *hashes_done);

  int TestLoadStore();

  int GetBatchSize() const { return batchsize; }

private:
  uint32_t *dev_keys; // internal code is still viewing these as uint32_t blobs.
  uint32_t *dev_scratch;
  uint32_t *dev_output;
  uint32_t *dev_tstate;
  uint32_t *dev_ostate;
  scan_job *dev_job;

  uint32_t *scan_output;
  int batchsize;
  int n_blocks;
};

static const int THREADS_PER_SCRYPT_BLOCK = 4;
static const int THREADS_PER_CUDA_BLOCK = 192; // Must be a multiple of TPScB
static const int SCRYPT_SCRATCH_PER_BLOCK = (32*1024);
static const int SCRYPT_WIDTH = 16;


#endif /* _CUDAHASHER_H_ */
