#include <stdio.h>
#include <sys/time.h>
#include "hasher.h"
#include "percival_scrypt.h"

/* 
 * TestScan and TestScrypt are "external" black-box tests
 * that validate the correctness of the external behavior.
 *
 * At this time, there are also several tests invoked inside the
 * class itself.
 */

/* TestScrypt validates random hashes against the original CPU version */
int TestScrypt(CudaHasher *h) {
  int success = -1;
  scrypt_hash *tests = NULL, *tests_out = NULL;
  uint64_t fnv = 1099511628211;
  uint64_t r = 1;

  int N = h->GetBatchSize();
  tests = (scrypt_hash *)malloc(sizeof(scrypt_hash)*N);
  if (!tests) {
    printf("TestScrypt failed allocating test in data\n");
    goto cleanup;
  }

  tests_out = (scrypt_hash *)malloc(sizeof(scrypt_hash)*N);
  if (!tests) {
    printf("TestScrypt failed allocating test out data\n");
    goto cleanup;
  }

  /* Can't include <random> in cuda/c++, so use successive
   * multiplies by the FNV prime to change lots of bits. */

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < SCRYPT_WIDTH; j++) {
      tests[i].b[j] = uint32_t(r);
      r = r * fnv + fnv;
      tests[i].bx[j] = uint32_t(r);
      r = r * fnv + fnv;
    }
  }

  h->ComputeHashes(tests, tests_out, N);

  for (int i = 0; i < N; i++) {
    scrypt_core_original((uint32_t *)&tests[i].b[0]);
    for (int j = 0; j < SCRYPT_WIDTH; j++) {
      if (tests[i].b[j] != tests_out[i].b[j] || tests[i].bx[0] != tests_out[i].bx[0]) {
	printf("TestScrypt validation failed on test %d\n", i);
	goto cleanup;
      }
    }
  }
  success = 0;
  
  printf("TestScrypt passed\n");

  cleanup:
  if (tests) free(tests);
  if (tests_out) free(tests_out);
  return success;
}



/* An overly simple test to make sure scanning kinda sorta works.
 * needs more.
 */

int TestScan(CudaHasher *h) {

  uint32_t job[32];
  uint32_t target[8];
  for (int i = 0; i < 32; i++) {
    job[i] = 0x55555555;
  }
  job[19] = 0;
  job[20] = 0x80000000;
  job[31] = 0x00000280;
  bzero(target, sizeof(target));
  uint32_t ttmp[] = { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x0000a78e, 0x00010000 };
  memcpy(target, ttmp, sizeof(uint32_t)*8);
  static const int n_attempts = 400000;
  int stop = 0;
  int rc = h->ScanNCoins(job, target, n_attempts, &stop, NULL);
  if (rc != 7204) {
    fprintf(stderr, "TestScan failed, expected 7204 got %d\n", rc);
    return -1;
  }
  job[19] = 10000;
  rc = h->ScanNCoins(job, target, n_attempts, &stop, NULL);
  if (rc != 34949) {
    fprintf(stderr, "TestScan failed, expected 34949 got %d\n", rc);
    return -1;
  }
  printf("TestScan passed\n");
  return 0;
}

int main() {
  fprintf(stderr, "hasher testing starting\n");

  CudaHasher *h = new CudaHasher();
  if (h->Initialize() != 0) {
    printf("Could not initialize hasher.  Exiting\n");
    delete h;
    return(0);
  }

  printf("Initialization complete.  Beginning tests.\n");

  if (h->TestLoadStore() != 0) {
    printf("Failed loadstore test.  Exiting.\n"); 
    delete h;
    return(-1);
  }

  if (TestScrypt(h) != 0) {
    printf("Failed Scrypt validation test.  Exiting.\n"); 
    delete h;
    return(-1);
  }

  if (TestScan(h) != 0) {
    printf("Failed litecoin scan test.  Exiting\n");
    delete h;
    return(-1);
  }

  printf("All tests passed.\n");
}
