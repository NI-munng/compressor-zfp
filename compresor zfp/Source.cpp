/*******************************************************************************
* Copyright 2020 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>

#include <ippdc.h>
#include <ipps.h>

#define MIN(a, b) (a < b? a : b)
#define MAX(a, b) (a > b? a : b)
/* Max dimension size is limited for this example, since dealing with huge arrays requires "long" IPP functions usage */
#define MAX_DIM 640

void InitSrcArray(Ipp32f*, int, int, int);
void Compress(int threads, const Ipp32f* pSrc, int maxX, int maxY, int maxZ, Ipp64f accur, double ratio, Ipp8u* pChunkBuffer, Ipp8u* pDst, int* pComprLen);
void Decompress(int threads, const Ipp8u* pSrc, int srcLen, int maxX, int maxY, int maxZ, Ipp64f accur, double ratio, Ipp32f* pDst);
static double GetWallTime();
static const int MAX_BYTES_PER_BLOCK = (IppZFPMAXBITS + 7) >> 3;

static unsigned int GetBitWidth(double ratio)
{
    const int BITS_PER_32FP_VALUE = (int)floor(32 / ratio);
    unsigned int n = 1u << (2 * 3);
    unsigned int bits = (unsigned int)floor(n * BITS_PER_32FP_VALUE + 0.5);
    bits = MAX(bits, 1 + 8u);
    return bits;
}

int main(int argc, const char* argv[])
{
    Ipp32f* pSrcArray, * pDstArray;
    Ipp8u* pBuffer, * pMergeBuffer;
    double accuracy = 0, ratio = 10.0;
    int nx, ny, nz, numFloats = 0, maxThreads = omp_get_max_threads(), threads;
    double waitTime = 1.;
    long iter;
    int i;

    nx = ny = nz = 100; /* Default dimension */
    for (i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-d", 2) == 0) {
            nx = atoi(argv[++i]);
            ny = atoi(argv[++i]);
            nz = atoi(argv[++i]);
            /* Check if dimensions are multiple of four */
            assert(nx % 4 == 0 && ny % 4 == 0 && nz % 4 == 0);
            /* Limit size of source data array. Larger sizes require different coding */
            assert(nx <= MAX_DIM && ny <= MAX_DIM && nz <= MAX_DIM);
            continue;
        }
        if (strncmp(argv[i], "-w", 2) == 0) {
            waitTime = atof(argv[++i]);
            assert(waitTime >= 0);
            continue;
        }
        if (strncmp(argv[i], "-mt", 3) == 0) {
            maxThreads = atoi(argv[++i]);
            assert(maxThreads > 0 && maxThreads <= omp_get_max_threads());
            continue;
        }
        if (strncmp(argv[i], "-r", 2) == 0) {
            ratio = atof(argv[++i]);
            /* In fixed rate mode max ratio is 32, i.e. 32 FP -> 1 bit compressed */
            assert(ratio > 0 && ratio <= 32);
            continue;
        }
        if (strncmp(argv[i], "-a", 2) == 0) {
            accuracy = atof(argv[++i]);
            assert(accuracy > 0);
            continue;
        }
    }
    numFloats = nx * ny * nz;
    pSrcArray = ippsMalloc_32f(numFloats);
    pDstArray = ippsMalloc_32f(numFloats);
    InitSrcArray(pSrcArray, nx, ny, nz);
    int bufLen = MAX_BYTES_PER_BLOCK * ((numFloats + 63) / 64);
    pBuffer = ippsMalloc_8u(bufLen);
    pMergeBuffer = ippsMalloc_8u(bufLen);
    if (accuracy > 0)
        printf("Threads Accuracy Ratio   Compr.time.(msec)   Dec.time(ST, msec)    Max.err\n");
    else
        printf("Threads Bits/block Ratio   Compr.time.(msec)   Decompr.time(msec)    Max.err\n");
    printf("----------------------------------------------------------------------------\n");
    for (threads = 1; threads <= maxThreads; threads++) {
        printf("%4d", threads);
        int comprLen, i;
        double maxErr;
        double timeStop;
        double timeStart = GetWallTime();
        iter = 0;
        do {
            Compress(threads, pSrcArray, nx, ny, nz, accuracy, ratio, pBuffer, pMergeBuffer, &comprLen);
            iter++;
            timeStop = GetWallTime();
        } while (timeStop - timeStart < waitTime);
        double resRatio = (double)(sizeof(Ipp32f) * numFloats) / comprLen;
        double execTime = (timeStop - timeStart) / iter * 1000;
        if (accuracy > 0)
            printf("%12.2g%6.1f%11.1f", accuracy, resRatio, execTime);
        else
            printf("%11d%9.1f%11.1f", GetBitWidth(ratio), resRatio, execTime);
        iter = 0;
        timeStart = GetWallTime();
        do {
            Decompress(threads, pMergeBuffer, comprLen, nx, ny, nz, accuracy, ratio, pDstArray);
            iter++;
            timeStop = GetWallTime();
        } while (timeStop - timeStart < waitTime);
        execTime = (timeStop - timeStart) / iter * 1000;
        /* Absolute error calculation */
        maxErr = 0.;
        for (i = 0; i < numFloats; i++)
        {
            double locErr = fabs(pSrcArray[i] - pDstArray[i]);
            if (locErr > maxErr)
                maxErr = locErr;
        }
        printf("%20.1f%21.4g\n", execTime, maxErr);
    }
    ippsFree(pBuffer);
    ippsFree(pSrcArray);
    ippsFree(pDstArray);
    ippsFree(pMergeBuffer);
}
/* Data initialization from ZFP's "simple" example */
void InitSrcArray(Ipp32f* pSrc, int dimX, int dimY, int dimZ)
{
    int i, j, k;

    for (k = 0; k < dimZ; k++)
        for (j = 0; j < dimY; j++)
            for (i = 0; i < dimX; i++) {
                double x = 2.0 * i / dimX;
                double y = 2.0 * j / dimY;
                double z = 2.0 * k / dimZ;
                pSrc[i + dimX * (j + dimY * k)] = (Ipp32f)exp(-(x * x + y * y + z * z));
            }
}

void Compress(int threads, const Ipp32f* pSrc, int maxX, int maxY, int maxZ, Ipp64f accuracy, double ratio, Ipp8u* pChunkBuffer, Ipp8u* pDst, int* pComprLen)
{
    int encStateSize;
    Ipp8u* pEncState;
    int i;
    int yStep = maxY, zStep = maxX * maxY;
    int chunk;
    int bx = (maxX + 3) / 4;
    int by = (maxY + 3) / 4;
    int bz = (maxZ + 3) / 4;
    int blocks = bx * by * bz;

    assert(ippsEncodeZfpGetStateSize_32f(&encStateSize) == ippStsNoErr);
    pEncState = (Ipp8u*)ippsMalloc_8u(encStateSize * threads);
    int blocksPerThread = blocks / threads;
    const int maxBytesPerThread = MAX_BYTES_PER_BLOCK * blocksPerThread;
    for (i = 0; i < threads; i++) {
        int offset = i * maxBytesPerThread;
        IppEncodeZfpState_32f* pState = (IppEncodeZfpState_32f*)(pEncState + i * encStateSize);
        assert(ippStsNoErr == ippsEncodeZfpInit_32f(pChunkBuffer + offset, maxBytesPerThread, pState));
        if (accuracy > 0)
            assert(ippStsNoErr == ippsEncodeZfpSetAccuracy_32f(accuracy, pState));
        else {
            unsigned int bits = GetBitWidth(ratio);
            assert(ippStsNoErr == ippsEncodeZfpSet_32f(bits, bits, IppZFPMAXPREC, IppZFPMINEXP, pState));
        }
    }

#pragma omp parallel for num_threads(threads)
    for (chunk = 0; chunk < threads; chunk++)
    {
        int blockMin = blocksPerThread * chunk;
        int blockMax = blocksPerThread * (chunk + 1);
        if (chunk == (threads - 1))
            blockMax += blocks % blocksPerThread;
        int block;
        int count = 0;
        const Ipp32f* pPrev = pSrc;
        for (block = blockMin; block < blockMax; block++) {
            int x, y, z;
            int b = block;
            x = 4 * (b % bx); b /= bx;
            y = 4 * (b % by); b /= by;
            z = 4 * b;
            const Ipp32f* p = pSrc + x + yStep * y + zStep * z;
            assert(ippStsNoErr == ippsEncodeZfp444_32f((const Ipp32f*)p, yStep * sizeof(Ipp32f), zStep * sizeof(Ipp32f), (IppEncodeZfpState_32f*)(pEncState + chunk * encStateSize)));
            pPrev = p;
            count++;
        }
    }
    /* Gather chunk bits into merged buffer */
    int totalBits = 0;
    int totalBytes = 0;
    int dstBitOffset = 0;
    int byteSize;
    for (chunk = 0; chunk < threads; chunk++) {
        Ipp64u bitSize;
        IppEncodeZfpState_32f* pState = (IppEncodeZfpState_32f*)(pEncState + chunk * encStateSize);

        assert(ippStsNoErr == ippsEncodeZfpGetCompressedBitSize_32f(pState, &bitSize));
        assert(ippStsNoErr == ippsEncodeZfpFlush_32f(pState));
        assert(ippStsNoErr == ippsEncodeZfpGetCompressedSize_32f(pState, &byteSize));
        assert(ippStsNoErr == ippsCopyBE_1u(pChunkBuffer + chunk * maxBytesPerThread, 0, pDst + (totalBits >> 3), dstBitOffset, (int)bitSize));
        totalBits += (int)bitSize;
        dstBitOffset = totalBits & 7;
    }
    /* Reset rest of bits in last byte */
    totalBytes = (totalBits + 7) >> 3;
    pDst[totalBytes - 1] &= (1 << (totalBits & 7)) - 1;
    *pComprLen = totalBytes;
    ippsFree((IppEncodeZfpState_32f*)pEncState);
}
void Decompress(int threads, const Ipp8u* pSrc, int srcLen, int maxX, int maxY, int maxZ, Ipp64f accuracy, double ratio, Ipp32f* pDst)
{
    /* Allocate ZFP decoding structures and buffer for splitted compressed data */
    int decStateSize;
    assert(ippStsNoErr == ippsDecodeZfpGetStateSize_32f(&decStateSize));
    if (accuracy > 0)
        threads = 1;    /* We cannot do parallel decompression in fixed accuracy mode */
    Ipp8u* pDecState = ippsMalloc_8u(threads * decStateSize);
    int bx = (maxX + 3) / 4;
    int by = (maxY + 3) / 4;
    int bz = (maxZ + 3) / 4;
    int blocks = bx * by * bz;
    int blocksPerThread = blocks / threads;
    unsigned int bitsPerBlock = GetBitWidth(ratio);
    unsigned int bitsPerThread = bitsPerBlock * blocksPerThread;
    int comprChunkSize = (bitsPerThread + 7) / 8;
    int comprBufferLen = comprChunkSize * (threads - 1) + (IppZFPMAXBITS + 7) / 8 * blocksPerThread;
    Ipp8u* comprChunkBuffer = ippsMalloc_8u(comprBufferLen);
    /* Split compressed data and initialize each ZFP decoding state */
    int chunk;
    for (chunk = 0; chunk < threads; chunk++) {
        IppDecodeZfpState_32f* pState = (IppDecodeZfpState_32f*)(pDecState + decStateSize * chunk);
        const Ipp8u* pCurSrc = pSrc + (bitsPerThread * chunk) / 8;
        int srcOffset = (bitsPerThread * chunk) % 8;
        if (chunk < threads - 1) {
            assert(ippStsNoErr == ippsCopyBE_1u(pCurSrc, srcOffset, comprChunkBuffer + comprChunkSize * chunk, 0, bitsPerThread));
            assert(ippStsNoErr == ippsDecodeZfpInit_32f(comprChunkBuffer + comprChunkSize * chunk, comprChunkSize, pState));
        }
        else {
            assert(ippStsNoErr == ippsCopyBE_1u(pCurSrc, srcOffset, comprChunkBuffer + comprChunkSize * chunk, 0, srcLen * 8 - (bitsPerThread * (threads - 1))));
            assert(ippStsNoErr == ippsDecodeZfpInit_32f(comprChunkBuffer + comprChunkSize * chunk, srcLen - comprChunkSize * (threads - 1), pState));
        }
        if (accuracy > 0)
            assert(ippStsNoErr == ippsDecodeZfpSetAccuracy_32f(accuracy, pState));
        else
            assert(ippStsNoErr == ippsDecodeZfpSet_32f(bitsPerBlock, bitsPerBlock, IppZFPMAXPREC, IppZFPMINEXP, pState));
    }
    /* Decode in parallel or */
    int yStep = maxY, zStep = maxX * maxY;
#pragma omp parallel for num_threads(threads)
    for (chunk = 0; chunk < threads; chunk++)
    {
        int blockMin = blocksPerThread * chunk;
        int blockMax = blocksPerThread * (chunk + 1);
        if (chunk == (threads - 1))
            blockMax += blocks % blocksPerThread;
        int block;
        int count = 0;

        for (block = blockMin; block < blockMax; block++) {
            int x, y, z;
            int b = block;
            x = 4 * (b % bx); b /= bx;
            y = 4 * (b % by); b /= by;
            z = 4 * b;
            Ipp32f* p = pDst + x + yStep * y + zStep * z;
            assert(ippStsNoErr == ippsDecodeZfp444_32f((IppDecodeZfpState_32f*)(pDecState + chunk * decStateSize), p, yStep * sizeof(Ipp32f), zStep * sizeof(Ipp32f)));
            count++;
        }
    }
    ippsFree(pDecState);
    ippsFree(comprChunkBuffer);
}
//  Windows
#ifdef _WIN32
#include <Windows.h>
static double GetWallTime()
{
    LARGE_INTEGER time, freq;
    if (!QueryPerformanceFrequency(&freq)) {
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)) {
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}
//  Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
static double GetWallTime() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
#endif