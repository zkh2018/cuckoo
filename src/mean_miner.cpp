// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp

#include "mean_miner.hpp"
#include <unistd.h>
#include <sys/time.h>

int main(int argc, char **argv) {
  int nthreads = 1;
  int ntrims   = 0;
  int nonce = 0;
  int range = 1;
  struct timeval time0, time1;
  u32 timems;
  char header[HEADERLEN];
  unsigned len;
  int c;

  memset(header, 0, sizeof(header));
  while ((c = getopt (argc, argv, "h:m:n:r:t:x:")) != -1) {
    switch (c) {
      case 'h':
        len = strlen(optarg);
        assert(len <= sizeof(header));
        memcpy(header, optarg, len);
        break;
      case 'x':
        len = strlen(optarg)/2;
        assert(len == sizeof(header));
        for (int i=0; i<len; i++)
          sscanf(optarg+2*i, "%2hhx", header+i);
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'm':
        ntrims = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }
  printf("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, EDGEBITS+1, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges, %d trims, %d threads\n", ntrims, nthreads);

  solver_ctx ctx(nthreads, ntrims);

  u64 sbytes = ctx.sharedbytes();
  u64 tbytes = ctx.threadbytes();
  int sunit,tunit;
  for (sunit=0; sbytes >= 1024; sbytes>>=10,sunit++) ;
  for (tunit=0; tbytes >= 1024; tbytes>>=10,tunit++) ;
  printf("Using %d%cB bucket memory, %d%cB thread memory, %d-way siphash, and %d-byte edgehash\n", (int)sbytes, " KMGT"[sunit], (int)tbytes, " KMGT"[tunit], NSIPHASH, EDGEHASH_BYTES);

  thread_ctx *threads = (thread_ctx *)calloc(nthreads, sizeof(thread_ctx));
  assert(threads);

  u32 sumnsols = 0;
  for (int r = 0; r < range; r++) {
    gettimeofday(&time0, 0);
    ctx.setheadernonce(header, sizeof(header), nonce + r);
    printf("k0 k1 %lx %lx\n", ctx.sip_keys.k0, ctx.sip_keys.k1);
    u32 nsols = ctx.solve();
    gettimeofday(&time1, 0);
    timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    printf("Time: %d ms\n", timems);

    for (unsigned s = 0; s < ctx.nsols; s++) {
      printf("Solution");
      for (int i = 0; i < PROOFSIZE; i++)
        printf(" %jx", (uintmax_t)ctx.sols[s][i]);
      printf("\n");
    }
    sumnsols += nsols;
  }
  printf("%d total solutions\n", sumnsols);
  return 0;
}
