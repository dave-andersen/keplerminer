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
  static const int n_attempts = 200000; // todo:  command line this

  printf("cudacoin hasher starting\n");

  struct timeval tv_start, tv_end;

  CudaHasher *h = new CudaHasher();
  if (h->Initialize() != 0) {
    fprintf(stderr, "Could not initialize hasher.  Exiting\n");
    delete h;
    return(0);
  }

  uint32_t job[32];
  /* Initializing the job in the same way cpuminer did for comparability */
  for (int i = 0; i < 32; i++) {
    job[i] = 0x55555555;
  }
  job[19] = 0;
  job[20] = 0x80000000;
  job[31] = 0x00000280;
  
  uint32_t target[8];
  /* The current litecoin mining difficulty as of 11/2013 - just for fun */
  uint32_t ttmp[]= { 0xCCCCCCCC, 0xCCCCCCCC, 0xCCCCCCCC, 0xCCCCCCCC, 0xCCCCCCCC, 0xCCCCCCCC, 0x369CCC, 0x0 };
  memcpy(target, ttmp, sizeof(uint32_t)*8);

  gettimeofday(&tv_start, NULL);
  int stop = 0;
  int rc = h->ScanNCoins(job, target, n_attempts, &stop, NULL);

  gettimeofday(&tv_end, NULL);
  double n_sec = timeval_diff(&tv_start, &tv_end);

  printf("%d hashes in %2.2f seconds (%2.2f kh/s)\n", 
	 n_attempts,
	 n_sec,
	 (n_attempts/n_sec)/1000);
  printf("Done\n");
  delete h;
}
