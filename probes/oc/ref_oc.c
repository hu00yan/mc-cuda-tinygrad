/* ref_oc.c - CPU reference harness for blaze/core/obs_camera.h.
 *
 * Same region tensor, same eye, same 6 poses as blaze/cuda/obs_camera.cu, so
 * this is directly comparable to its printed rays/s line and its hex dump.
 * Emits a flat binary instead of hex text: a header then, per (seed, pose),
 * OC_NPIX u16 ids + OC_NPIX u8 depth + OC_NPIX u8 edge.
 *
 * build: clang -O2 -ffp-contract=off -I<blaze/core> ref_oc.c -o ref_oc -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "region_tensor.h"
#include "obs_camera.h"

/* identical to blaze/cuda/obs_camera.cu */
#define RG_X0 (-48)
#define RG_Y0 28
#define RG_Z0 (-48)
#define RG_NX 113
#define RG_NY 104
#define RG_NZ 113

static const float k_poses[][2] = {
    {180.0f, 0.0f}, {0.0f, 0.0f},   {90.0f, 15.0f},
    {270.0f, -30.0f}, {45.0f, 60.0f}, {200.0f, -75.0f},
};
#define NPOSES ((int)(sizeof k_poses / sizeof k_poses[0]))

#define EYE_X 8.5
#define EYE_Y 80.0
#define EYE_Z 8.5

/* SplitMix64: same generator we will use on the TinyGrad side so extra
 * benchmark poses are reproducible without shipping a pose file. */
static unsigned long long sm64(unsigned long long *s) {
    unsigned long long z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    const char *out_path = NULL;
    const char *in_path = NULL;
    const char *dump_cells = NULL;
    int nseeds = 1, reps = 0, nposes = NPOSES, quiet = 0;
    unsigned long long pseed = 0x5EED1234ULL;
    long long seeds[64];
    int nseed_max = 64, i, s, p;

    seeds[0] = 0;
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-cells") && i + 1 < argc) dump_cells = argv[++i];
        else if (!strcmp(argv[i], "--seeds") && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            nseeds = 0;
            while (tok && nseeds < nseed_max) {
                seeds[nseeds++] = strtoll(tok, 0, 10);
                tok = strtok(0, ",");
            }
        } else if (!strcmp(argv[i], "--poses") && i + 1 < argc) nposes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pose-seed") && i + 1 < argc) pseed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
    }

    /* Extra poses beyond the 6 canonical ones: a uniform sweep of yaw in
     * [0,360) x pitch in [-89,89], so throughput is not measured on 6
     * nearly-identical rays that all hit the same wall. */
    float *yaw = (float *)malloc(sizeof(float) * (size_t)nposes);
    float *pitch = (float *)malloc(sizeof(float) * (size_t)nposes);
    for (p = 0; p < NPOSES && p < nposes; ++p) {
        yaw[p] = k_poses[p][0];
        pitch[p] = k_poses[p][1];
    }
    for (p = NPOSES; p < nposes; ++p) {
        unsigned long long a = sm64(&pseed), b = sm64(&pseed);
        yaw[p] = (float)((double)(a >> 11) / 9007199254740992.0 * 360.0);
        pitch[p] = (float)((double)(b >> 11) / 9007199254740992.0 * 178.0 - 89.0);
    }

    McSinTable *st = (McSinTable *)malloc(sizeof(McSinTable));
    ChunkPrimer *primer = (ChunkPrimer *)malloc(sizeof(ChunkPrimer));
    CpScratch *sc = (CpScratch *)malloc(sizeof(CpScratch));
    long ncells = rt_count(RG_NX, RG_NY, RG_NZ);
    u16 *cells = (u16 *)malloc((size_t)ncells * sizeof(u16));
    u16 *ids = (u16 *)malloc(sizeof(u16) * (size_t)OC_NPIX);
    u8 *depth = (u8 *)malloc(OC_NPIX);
    u8 *edge = (u8 *)malloc(OC_NPIX);
    if (!st || !primer || !sc || !cells || !ids || !depth || !edge) {
        fprintf(stderr, "oom\n");
        return 1;
    }
    mc_sin_table_init(st);

    if (dump_cells) {
        /* Dump the region tensor so the TinyGrad side never has to run
         * worldgen; both sides then read byte-identical cells. */
        long ci;
        FILE *f = fopen(dump_cells, "wb");
        if (!f) { fprintf(stderr, "open %s\n", dump_cells); return 1; }
        if (!in_path)
            rt_fill(cells, (u64)seeds[0], RG_X0, RG_Y0, RG_Z0, RG_NX, RG_NY,
                    RG_NZ, primer, sc, st);
        for (ci = 0; ci < ncells; ++ci) cells[ci] = (u16)(cells[ci] << 4);
        fwrite(cells, sizeof(u16), (size_t)ncells, f);
        fclose(f);
        fprintf(stderr, "[ref] wrote %s: %dx%dx%d u16 packed states\n",
                dump_cells, RG_NX, RG_NY, RG_NZ);
        free(yaw); free(pitch); free(st); free(primer); free(sc);
        free(cells); free(ids); free(depth); free(edge);
        return 0;
    }

    if (in_path) {
        /* Re-render from a region tensor dumped by --dump-cells, so the
         * reference is decoupled from worldgen. */
        FILE *f = fopen(in_path, "rb");
        long got;
        if (!f) { fprintf(stderr, "open %s\n", in_path); return 1; }
        got = (long)fread(cells, sizeof(u16), (size_t)ncells, f);
        fclose(f);
        if (got != ncells) { fprintf(stderr, "short read %ld/%ld\n", got, ncells); return 1; }
        goto render;
    }

    for (s = 0; s < nseeds; ++s) {
        long ci;
        double t0 = now_s();
        rt_fill(cells, (u64)seeds[s], RG_X0, RG_Y0, RG_Z0, RG_NX, RG_NY, RG_NZ,
                primer, sc, st);
        for (ci = 0; ci < ncells; ++ci) cells[ci] = (u16)(cells[ci] << 4);
        if (!quiet)
            fprintf(stderr, "[ref] seed=%lld region filled in %.2fs\n",
                    (long long)seeds[s], now_s() - t0);
    }

render: {
    OcRegion rg;
    rg.cells = cells;
    rg.x0 = RG_X0; rg.y0 = RG_Y0; rg.z0 = RG_Z0;
    rg.nx = RG_NX; rg.ny = RG_NY; rg.nz = RG_NZ;

    if (reps > 0) {
        double t0, t1;
        long rays;
        /* warm */
        for (p = 0; p < nposes; ++p)
            oc_pixel(&rg, st, EYE_X, EYE_Y, EYE_Z, yaw[p], pitch[p],
                     OC_NPIX - 1, 0, &ids[0], &depth[0], &edge[0]);
        t0 = now_s();
        for (i = 0; i < reps; ++i)
            for (p = 0; p < nposes; ++p)
                oc_render(&rg, st, EYE_X, EYE_Y, EYE_Z, yaw[p], pitch[p],
                          ids, depth, edge);
        t1 = now_s();
        rays = (long)reps * nposes * OC_NPIX;
        fprintf(stderr, "[ref] %.3fM rays/s  (%ld rays, %d reps, 1 thread, scalar)\n",
                (double)rays / (t1 - t0) / 1e6, rays, reps);
        free(yaw); free(pitch); free(st); free(primer); free(sc);
        free(cells); free(ids); free(depth); free(edge);
        return 0;
    }

    if (out_path) {
        /* header: magic, nposes, nseeds, then per (seed,pose) triples */
        unsigned char hdr[16];
        FILE *f;
        memcpy(hdr, "OCREF1\0\0", 7);
        memcpy(hdr + 7, &nposes, 4);
        memcpy(hdr + 11, &nseeds, 4);
        memcpy(hdr + 15, &(unsigned char){0}, 1);
        f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "open %s\n", out_path); return 1; }
        fwrite(hdr, 1, 16, f);
        for (s = 0; s < nseeds; ++s) {
            if (!in_path) {
                long ci;
                rt_fill(cells, (u64)seeds[s], RG_X0, RG_Y0, RG_Z0,
                        RG_NX, RG_NY, RG_NZ, primer, sc, st);
                for (ci = 0; ci < ncells; ++ci) cells[ci] = (u16)(cells[ci] << 4);
            }
            for (p = 0; p < nposes; ++p) {
                oc_render(&rg, st, EYE_X, EYE_Y, EYE_Z, yaw[p], pitch[p],
                          ids, depth, edge);
                fwrite(ids, sizeof(u16), OC_NPIX, f);
                fwrite(depth, 1, OC_NPIX, f);
                fwrite(edge, 1, OC_NPIX, f);
            }
        }
        fclose(f);
        fprintf(stderr, "[ref] wrote %s: %d seeds x %d poses\n", out_path, nseeds, nposes);
    }
}

    free(yaw); free(pitch); free(st); free(primer); free(sc);
    free(cells); free(ids); free(depth); free(edge);
    return 0;
}
