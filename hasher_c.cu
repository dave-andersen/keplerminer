/* Copyright (C) 2013 David G. Andersen.  All rights reserved.
 *
 * Use of this code is covered under the Apache 2.0 license, which
 * can be found in the file "LICENSE"
 */

#include "hasher.h"

CudaHasher *h_global = NULL;

extern "C" {
int scanhash_c(uint32_t *pdata, unsigned char *unused, const uint32_t *ptarget,
	   uint32_t max_nonce, unsigned long *hashes_done, volatile int *work_restart) {
  if (!h_global) {
    printf("Initializing h_global GPU\n");
    h_global = new CudaHasher();
    h_global->Initialize();
  }
  uint32_t start_nonce = pdata[19];
  fprintf(stderr, "scanhash_c from %u to %u  (%d)\n", start_nonce, max_nonce, *work_restart);

  int n_to_scan = max_nonce - start_nonce;
  pdata[19] = start_nonce;
  int rc = h_global->ScanNCoins(pdata, ptarget, n_to_scan, work_restart, hashes_done);
  if (rc != -1) {
    pdata[19] = start_nonce+rc;
    fprintf(stderr, "Returning success with %u\n", pdata[19]);
    return 1;
  }
  return 0;
}
}
