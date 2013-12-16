#include <stdio.h>
#include <sys/time.h>
#include "hasher.h"

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

  printf("cudacoin hasher starting\n");

  struct timeval tv_start, tv_end;

  CudaHasher *h = new CudaHasher();
  if (h->Initialize() != 0) {
    fprintf(stderr, "Could not initialize hasher.  Exiting\n");
    delete h;
    return(0);
  }

  uint32_t job[32];
  uint32_t target[8];
  for (int i = 0; i < 32; i++) {
    job[i] = 0x55555555;
  }
  job[19] = 0;
  job[20] = 0x80000000;
  job[31] = 0x00000280;
  bzero(target, sizeof(target));
  /* The current litecoin mining difficulty as of 11/2013 - just for fun */
  uint32_t ttmp[]= { 0xCCCCCCCC, 0xCCCCCCCC, 0xCCCCCCCC, 0xCCCCCCCC, 0xCCCCCCCC, 0xCCCCCCCC, 0x369CCC, 0x0 };
  memcpy(target, ttmp, sizeof(uint32_t)*8);
  gettimeofday(&tv_start, NULL);
  static const int n_attempts = 200000;
  int stop = 0;
  int rc = h->ScanNCoins(job, target, n_attempts, &stop, NULL);
  gettimeofday(&tv_end, NULL);

#if 0
    /* Note:  This code has bit-rotted.  N is no longer defined.
    * Change to use GetBatchSize() if you want to run it more. */
  scrypt_hash *keys_in = (scrypt_hash *)malloc(N*sizeof(scrypt_hash));
  if (!keys_in) {
    exit(-1);
  }
  scrypt_hash *keys_out = (scrypt_hash *)malloc(N*sizeof(scrypt_hash));
  if (!keys_out) {
    free(keys_out);
    exit(-1);
  }

  printf("Host malloc complete: %ld bytes\n", N*sizeof(scrypt_hash));
  fflush(stdout);

  init_test_keys(keys_in, h.GetBatchsize()*100);
  gettimeofday(&tv_end, NULL);

  for (int i = 0; i < BENCHMARK_LOOPS; i++) { // benchmark loop
    h->ComputeHashes(keys_in, keys_out, N);
  }
  for (int key = 0; key < 2; key++) {
#if 1
  for (int i = 0; i < 16; i++) {
    printf("%8.8x ", keys_out[key].b[i]);
    if (i%8 == 7) { printf("\n"); }
  }
  for (int i = 0; i < 16; i++) {
    printf("%8.8x ", keys_out[key].bx[i]);
    if (i%8 == 7) { printf("\n"); }
  }

  printf("\n");
  }
#endif
#endif

  double n_sec = timeval_diff(&tv_start, &tv_end);

  printf("%d hashes in %2.2f seconds (%2.2f kh/s)\n", 
	 n_attempts,
	 n_sec,
	 (n_attempts/n_sec)/1000);
  printf("Done\n");
#if 0
    free(keys_in);
  free(keys_out);
#endif
  delete h;
}
