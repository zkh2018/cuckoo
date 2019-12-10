// Cuckarood Cycle, a memory-hard proof-of-work by John Tromp and team Grin
// Copyright (c) 2018 Jiri Photon Vadura and John Tromp
// This GGM miner file is covered by the FAIR MINING license

//Includes for IntelliSense
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

#include <stdio.h>
#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;


const u32 NX = 64;//1 << XBITS;
const u32 NY = 96;//1 << YBITS;
const u32 NX2 = 6144;
const u32 NZ = 174763;//(NEDGES/NX+NZ-1)/NZ;//1 << ZBITS;

__device__ __forceinline__ uint2 ld_cs_u32_v2(const uint2 *p_src)
{
  uint2 n_result;
  asm("ld.global.cs.v2.u32 {%0,%1}, [%2];"  : "=r"(n_result.x), "=r"(n_result.y) : "l"(p_src));
  return n_result;
}

__device__ __forceinline__ void st_cg_u32_v2(uint2 *p_dest, const uint2 n_value)
{
  asm("st.global.cg.v2.u32 [%0], {%1, %2};" :: "l"(p_dest), "r"(n_value.x), "r"(n_value.y));
}

__device__ __forceinline__ void st_cg_u32_v4(uint4 *p_dest, const uint4 n_value)
{
  asm("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(p_dest), "r"(n_value.x), "r"(n_value.y), "r"(n_value.z), "r"(n_value.w));
}

__device__ __forceinline__  void setbit(u32 * ecounters, const int bucket)
{
  const int word = bucket >> 5;
  const unsigned char bit = ((bucket & 31));
  const u32 mask = 1 << bit;
  u32 old = atomicOr(ecounters + word, mask) & mask;
  if(old){
    atomicOr(ecounters + word + NZ/32, mask);
  }
}

__device__ __forceinline__  bool testbit(u32 * ecounters, const int bucket)
{
  const int word = bucket >> 5;
  const unsigned char bit = ((bucket & 31));
  return (ecounters[word+NZ/32] >> (bit)) & 1;
}

__constant__ siphash_keys dipkeys;
__constant__ u64 recovery[42];

#define ROTL(x,b) ( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND {\
  v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
  v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
  v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
  v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
  v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
}
#define SIPBLOCK(b) {\
  v3 ^= (b);\
  SIPROUND;SIPROUND;SIPROUND;SIPROUND;\
  v0 ^= (b);\
  v2 ^= 0xff;\
  SIPROUND;SIPROUND;SIPROUND;SIPROUND;\
  SIPROUND;SIPROUND;SIPROUND;SIPROUND;\
}

__device__ u64 dipblock(const siphash_keys &keys, const word_t edge, u64 *buf) {
  diphash_state<> shs(keys);
  word_t edge0 = edge & ~EDGE_BLOCK_MASK;
  u32 i;
  for (i=0; i < EDGE_BLOCK_MASK; i++) {
    shs.hash24(edge0 + i);
    buf[i] = shs.xor_lanes();
  }
  shs.hash24(edge0 + i);
  buf[i] = 0;
  return shs.xor_lanes();
}

__device__ u32 endpoint(uint2 nodes, int uorv) {
  return uorv ? nodes.y : nodes.x;
}

#define DUMP(E, nonce) {\
  u64 edge = E;\
  u32 node0 = edge & EDGEMASK;\
  u32 node1 = (edge >> 32) & EDGEMASK;\
  int bucket = node0 / NZ;\
  for (u64 ret = atomicCAS(&magazine[bucket], 0, edge); ret; ) {\
    u64 ret2 = atomicCAS(&magazine[bucket], ret, 0);\
        if (ret2 == ret) {\
      int block = bucket / (NX2 / NA);\
      int shift = (maxOut* NX2) * block;\
      int position = (min(maxOut- 2, (atomicAdd(indexes + bucket, 2))));\
      int idx = (shift+((bucket%(NX2 / NA)) * (maxOut) + position)) / 2;\
      u32 node2 = ret & EDGEMASK;\
      u32 node3 = (ret>>32)&EDGEMASK;\
      buffer[idx] = make_uint4(node2, node3, node0, node1);\
      break;\
    }\
    ret = ret2 ? ret2 : atomicCAS(&magazine[bucket], 0, edge);\
  }\
}
template<int tpb, int maxOut>
__global__  void FluffySeed(uint4 * __restrict__ buffer, u32 * __restrict__ indexes, const u32 offset)
{
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int gid = group * dim + lid;
  const int nthreads = gridDim.x * dim;
  __shared__ unsigned long long magazine[NX2];
  const int nloops = (NEDGES / NA / EDGE_BLOCK_SIZE - gid + nthreads-1) / nthreads;
  ulonglong4 sipblockL[EDGE_BLOCK_SIZE/4 - 1];
  uint64_t v0, v1, v2, v3;

#if tpb && NX2 % tpb == 0
  for (int i = 0; i < NX2/tpb; i++)
#else
  for (int i = 0; i < (NX2 - lid + tpb-1) / tpb; i++)
#endif
    magazine[lid + tpb * i] = 0;
  __syncthreads();

  for (int i = 0; i < nloops; i++) {
    u64 blockNonce = offset + (gid * nloops * EDGE_BLOCK_SIZE + i * EDGE_BLOCK_SIZE);

    v0 = dipkeys.k0;
    v1 = dipkeys.k1;
    v2 = dipkeys.k2;
    v3 = dipkeys.k3;

    // do one block of 64 edges
    for (int b = 0; b < EDGE_BLOCK_SIZE-4; b += 4) {
      SIPBLOCK(blockNonce + b);
      u64 e1 = (v0 ^ v1) ^ (v2  ^ v3);
      SIPBLOCK(blockNonce + b + 1);
      u64 e2 = (v0 ^ v1) ^ (v2  ^ v3);
      SIPBLOCK(blockNonce + b + 2);
      u64 e3 = (v0 ^ v1) ^ (v2  ^ v3);
      SIPBLOCK(blockNonce + b + 3);
      u64 e4 = (v0 ^ v1) ^ (v2  ^ v3);
      sipblockL[b / 4] = make_ulonglong4(e1, e2, e3, e4);
    }

    SIPBLOCK(blockNonce + EDGE_BLOCK_SIZE-4);
    u64 e1 = (v0 ^ v1) ^ (v2  ^ v3);
    SIPBLOCK(blockNonce + EDGE_BLOCK_SIZE-3);
    u64 e2 = (v0 ^ v1) ^ (v2  ^ v3);
    SIPBLOCK(blockNonce + EDGE_BLOCK_SIZE-2);
    u64 e3 = (v0 ^ v1) ^ (v2  ^ v3);
    SIPBLOCK(blockNonce + EDGE_BLOCK_SIZE-1);
    u64 last = (v0 ^ v1) ^ (v2  ^ v3);

    DUMP(last,      blockNonce+EDGE_BLOCK_SIZE-1);
    DUMP(e1 ^ last, blockNonce+EDGE_BLOCK_SIZE-4);
    DUMP(e2 ^ last, blockNonce+EDGE_BLOCK_SIZE-3);
    DUMP(e3 ^ last, blockNonce+EDGE_BLOCK_SIZE-2);

    for (int s = 14; s >= 0; s--) {
      ulonglong4 edges = sipblockL[s];
      DUMP(edges.x ^ last, blockNonce+EDGE_BLOCK_SIZE-s*4);
      DUMP(edges.y ^ last, blockNonce+EDGE_BLOCK_SIZE-s*4+1);
      DUMP(edges.z ^ last, blockNonce+EDGE_BLOCK_SIZE-s*4+2);
      DUMP(edges.w ^ last, blockNonce+EDGE_BLOCK_SIZE-s*4+3);
    }
  }

  __syncthreads();

  for (int i = 0; i < NX2/tpb; i++) {
    int bucket = lid + (tpb * i);
    u64 edge = magazine[bucket];
    if (edge != 0) {
      int block = bucket / (NX2 / NA);
      int shift = (maxOut * NX2) * block;
      int position = (min(maxOut - 2, (atomicAdd(indexes + bucket, 2))));
      int idx = (shift + ((bucket % (NX2 / NA)) * maxOut+ position)) / 2;
      buffer[idx] = make_uint4((u32)(edge&EDGEMASK), (u32)((edge >> 32)&EDGEMASK), 0, 0);
    }
  }
}

template<int tpb, int bktInSize, int bktOutSize>
__global__  void FluffyRound_A1(const uint2 * source, uint2 * destination, 
        const u32 * sourceIndexes, u32 * destinationIndexes, const int offset)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;

  __shared__ u32 ecounters[NZ/16];
  int nloops[NA];

  for (int i = lid; i < NZ/16; i+=blockDim.x)
    ecounters[i] = 0;//two_bit_set[group * NZ/32 + i];

  for (int a = 0; a < NA; a++)
    nloops[a] = (min(sourceIndexes[a * NX2 + offset + group], bktInSize) - lid + tpb-1) / tpb;

  const int rowOffset = offset * NA;
  source += bktInSize * (rowOffset + group) + lid;

  __syncthreads();
 for (int a = 0; a < NA; a++) {
   const int delta = a * (NX2/NA) * bktInSize;
   for (int i = 0; i < nloops[a]; i++) {
     uint2 edge = source[delta + i * tpb];
     if (edge.x == 0 && edge.y == 0) continue;
     setbit(ecounters, (edge.x % NZ));
   }
 }
 __syncthreads();

 for (int a = 0; a < NA; a++) {
   const int delta = a * (NX2/NA) * bktInSize;
   for (int i = 0; i < nloops[a]; i++) {
     uint2 edge = source[delta + i * tpb];
     if (edge.x == 0 && edge.y == 0) continue;
     if (testbit(ecounters, (edge.x % NZ))) {
       int bucket = edge.y / NZ;
       int bktIdx = min(atomicAdd(destinationIndexes + bucket + rowOffset, 1), bktOutSize - 1);
       destination[((bucket * bktOutSize) + bktIdx)] = make_uint2(edge.y, edge.x);
     }
   }
 }
}

template<int tpb, int bktInSize, int bktOutSize>
__global__  void FluffyRound_A2(const uint2 * source, uint2 * destination, const u32 * sourceIndexes, u32 * destinationIndexes)
 {
  const int lid = threadIdx.x;
  const int group = blockIdx.x;

  __shared__ u32 ecounters[NZ/16];
  const int nloops = (min(sourceIndexes[group], bktInSize) - lid + tpb-1) / tpb;
  source += bktInSize * group + lid;

  for (int i = 0; i < NZ/16/tpb; i++)
    ecounters[lid + (tpb * i)] = 0;

  __syncthreads();

  for (int i = 0; i < nloops; i++){
    uint2 edge = source[i*tpb];
    setbit(ecounters, edge.x % NZ);
  }

  __syncthreads();

  for (int i = nloops; --i >= 0;) {
    uint2 edge = source[i * tpb];
    if (testbit(ecounters, (edge.x % NZ))) {
      const int bucket = edge.y / NZ;
      const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
      st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
    }
  }
}

template<int tpb, int bktInSize, int bktOutSize>
__global__  void FluffyRound_A3(uint2 * source, uint2 * destination, const u32 * sourceIndexes, u32 * destinationIndexes)
{
  const int lid = threadIdx.x;
  const int group = blockIdx.x;

  __shared__ u32 ecounters[NZ/16];
  int nloops[NA];

  for (int i = lid; i < NZ/16; i+=blockDim.x)
    ecounters[i] = 0;// two_bit_set[group*NZ/16 + i];

  for (int a = 0; a < NA; a++)
    nloops[a] = (min(sourceIndexes[group + a*NX2], bktInSize) - lid + tpb-1) / tpb;

  source += bktInSize * group + lid;
  __syncthreads();
  for (int a = 0; a < NA; a++) {
    const int delta = a * bktInSize * NX2;
    for (int i = 0; i < nloops[a]; i++) {
      uint2 edge = source[delta + i * tpb];
      if (edge.x == 0 && edge.y == 0) continue;
      setbit(ecounters, edge.x % NZ);
    }
  }
  __syncthreads();

  for (int a = 0; a < NA; a++) {
    const int delta = a * bktInSize * NX2;
    for (int i = 0; i < nloops[a]; i++) {
      uint2 edge = ld_cs_u32_v2(&source[delta + i * tpb]);
      if (edge.x == 0 && edge.y == 0) continue;
      if (testbit(ecounters, (edge.x % NZ))) {
        const int bucket = edge.y / NZ;
        const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
        st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
      }
    }
  }
}


template<int maxIn>
__global__ void Tail(const uint2 *source, uint2 *destination, const u32 *srcIdx, u32 *dstIdx) {
  const int lid = threadIdx.x;
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  int myEdges = min(srcIdx[group], maxIn);
  __shared__ int destIdx;

  if (lid == 0)
    destIdx = atomicAdd(dstIdx, myEdges);
  __syncthreads();
  for (int i = lid; i < myEdges; i += dim){
    uint2 edge = source[group * maxIn + i];
    destination[destIdx + i] = source[group * maxIn + i];
  }
}

__global__  void FluffyRecovery(u32 * indexes)
{
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;

  __shared__ u32 nonces[PROOFSIZE];
  u64 buf[EDGE_BLOCK_SIZE];

  const int nloops = (NEDGES / EDGE_BLOCK_SIZE - gid + nthreads-1) / nthreads;
  if (lid < PROOFSIZE) nonces[lid] = 0;

  __syncthreads();

  for (int i = 0; i < nloops; i++) {
    u64 blockNonce = gid * nloops * EDGE_BLOCK_SIZE + i * EDGE_BLOCK_SIZE;

    const u64 last = dipblock(dipkeys, blockNonce, buf);
    buf[EDGE_BLOCK_SIZE - 1] = 0;

    for (int s = EDGE_BLOCK_SIZE; --s >= 0; ) {
      u64 lookup = buf[s] ^ last;

      u64 u = (lookup & EDGEMASK);
      u64 v = (lookup >> 32) & EDGEMASK;

      u64 a = u | (v << 32);
      u64 b = v | (u << 32);

      for (int i = 0; i < PROOFSIZE; i++) {
        if (recovery[i] == a || recovery[i] == b){
          nonces[i] = blockNonce + s;
        }
      }
    }
  }

  __syncthreads();

  if (lid < PROOFSIZE) {
    if (nonces[lid] > 0)
      indexes[lid] = nonces[lid];
  }
}
