/* Copyright (C) 2013 David G. Andersen.  All rights reserved.
 *
 * Use of this code is covered under the Apache 2.0 license, which
 * can be found in the file "LICENSE"
 */


int scanhash_c(uint32_t *pdata, unsigned char *unused, const uint32_t *ptarget,
	   uint32_t max_nonce, unsigned long *hashes_done, volatile int *work_restart);

