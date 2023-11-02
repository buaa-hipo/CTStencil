#include "common.h"
#include <simd.h>
#include <slave.h>

/*
double a[M][N][P];
double b[M][N][P];
double c0, c1, c2, c3, c4, c5, c6;

for (long k = 1; k < M - 1; ++k)
{
    for (long j = 1; j < N - 1; ++j)
    {
        for (long i = 1; i < P - 1; ++i)
        {
            b[k][j][i] = c0 * a[k][j][i] + c1 * a[k][j][i - 1] + c2 * a[k][j][i + 1]
                       + c3 * a[k - 1][j][i] + c4 * a[k + 1][j][i] + c5 * a[k][j - 1][i]
                       + c6 * a[k][j + 1][i];
        }
    }
}
*/


#define vload_3p_0(c, v_l, v_c, v_r, arr)   \
    {                                       \
        simd_load(v_c, &arr[c]);            \
        simd_load(v_tc1, &arr[c - 4]);      \
        simd_load(v_tc2, &arr[c + 4]);      \
        v_t1 = simd_vshff(v_c, v_tc1, rc1); \
        v_l = simd_vshff(v_c, v_t1, rc2);   \
        v_t2 = simd_vshff(v_tc2, v_c, rc1); \
        v_r = simd_vshff(v_t2, v_c, rc2);   \
    }

#define vload_3p_1(c, v_l, v_c, v_r, arr)  \
    {                                      \
        simd_load(v_l, &arr[c - 1]);       \
        simd_load(v_tc1, &arr[c + 3]);     \
        v_r = simd_vshff(v_tc1, v_l, rc1); \
        v_c = simd_vshff(v_r, v_l, rc2);   \
    }

#define vload_3p_2(c, v_l, v_c, v_r, arr)    \
    {                                        \
        simd_load(v_tc1, &arr[c - 2]);       \
        simd_load(v_tc2, &arr[c + 2]);       \
        v_c = simd_vshff(v_tc2, v_tc1, rc1); \
        v_l = simd_vshff(v_c, v_tc1, rc2);   \
        v_r = simd_vshff(v_tc2, v_c, rc2);   \
    }

#define vload_3p_3(c, v_l, v_c, v_r, arr)  \
    {                                      \
        simd_load(v_tc1, &arr[c - 3]);     \
        simd_load(v_r, &arr[c + 1]);       \
        v_l = simd_vshff(v_r, v_tc1, rc1); \
        v_c = simd_vshff(v_r, v_l, rc2);   \
    }

#define vload_1p_1(c, v_l, v_c, v_r, arr) vload_3p_1(c, v_l, v_c, v_r, arr)

#define vload_1p_2(c, v_l, v_c, v_r, arr)    \
    {                                        \
        simd_load(v_tc1, &arr[c - 2]);       \
        simd_load(v_tc2, &arr[c + 2]);       \
        v_c = simd_vshff(v_tc2, v_tc1, rc1); \
    }

#define vload_1p_3(c, v_l, v_c, v_r, arr) vload_3p_3(c, v_l, v_c, v_r, arr)


extern DEFINED_DATATYPE c[NUMPOINTS];
// __thread_local DEFINED_DATATYPE local_tIn_t[DIMT][(1 + 2 * R) * (DIMY + 2 * R * DIMT) * (DIMX + 2 * R * DIMT)];
// __thread_local DEFINED_DATATYPE local_tOut[DIMX + 2 * R * DIMT];

void spe_func(struct spe_parameter *spe_param)
{
    int DIMT, DIMX, DIMY, DIMZ;
    DEFINED_DATATYPE c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4], c5 = c[5], c6 = c[6];
    DEFINED_DATATYPE **temp;
    DEFINED_DATATYPE *local_tIn, *local_tOut, *local_tOut_1, *local_tOut_2, *local_tIn_0, **local_tIn_t;
    int blockNum_z, blockNum_y, blockNum_x;
    int blockID_z, blockID_y, blockID_x;
    int DIMZ_padding, DIMY_padding, DIMX_padding;
    int odd_block_z, odd_block_y, odd_block_x;
    int DIMZ_final, DIMY_final, DIMX_final;
    int left_z_block, right_z_block, left_y_block, right_y_block, left_x_block, right_x_block;
    int left_z_load, right_z_load, left_y_load, right_y_load, left_x_load, right_x_load, load_size;
    int blockSize_x, blockSize_y, blockSize_z, blockSize_z_0;
    int left_z_compute_t[DIMT + 1];
    int t, z, global_y, y, global_x, x;
    int nt, nz, ny, nx;
    int in = 1, out = 0;
    DEFINED_DATATYPE *global_tIn, *global_tOut;
    int offset_z_block_t, N_z_block_t, C_z_block_t, S_z_block_t, offset_z_block_t_next;
    int C_next, C, B, T, W, E, S, N;
    int C_global, C_local;
    volatile int DMA_reply, DMA_push;
    volatile int DMA_preload, wait_num;
    DEFINED_DATATYPE *tOut_ptr, *tOut_push_ptr;
    int i;

    int aligned;
    DEFINED_V_DATATYPE v_C, v_B, v_T, v_W, v_E, v_S, v_N, v_res;
    DEFINED_V_DATATYPE v_tc1, v_tc2, v_t1, v_t2, v_l, v_r;
    int rc1 = 0x4E;
    int rc2 = 0x99;
    DEFINED_V_DATATYPE v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6;
#ifdef FLOAT
    v_c0 = simd_set_floatv4(c0, c0, c0, c0);
    v_c1 = simd_set_floatv4(c1, c1, c1, c1);
    v_c2 = simd_set_floatv4(c2, c2, c2, c2);
    v_c3 = simd_set_floatv4(c3, c3, c3, c3);
    v_c4 = simd_set_floatv4(c4, c4, c4, c4);
    v_c5 = simd_set_floatv4(c5, c5, c5, c5);
    v_c6 = simd_set_floatv4(c6, c6, c6, c6);
#else
    v_c0 = simd_set_doublev4(c0, c0, c0, c0);
    v_c1 = simd_set_doublev4(c1, c1, c1, c1);
    v_c2 = simd_set_doublev4(c2, c2, c2, c2);
    v_c3 = simd_set_doublev4(c3, c3, c3, c3);
    v_c4 = simd_set_doublev4(c4, c4, c4, c4);
    v_c5 = simd_set_doublev4(c5, c5, c5, c5);
    v_c6 = simd_set_doublev4(c6, c6, c6, c6);
#endif

    temp = spe_param->temp;
    nt = spe_param->nt;
    nx = spe_param->nx;
    ny = spe_param->ny;
    nz = spe_param->nz;
    DIMT = spe_param->DIMT;
    DIMX = spe_param->DIMX;
    DIMY = spe_param->DIMY;
    DIMZ = spe_param->DIMZ;

    blockNum_z = LAYERS / DIMZ;
    blockNum_y = NUMROWS / DIMY;
    blockNum_x = NUMCOLS / DIMX;

    blockID_z = _MYID / (blockNum_y * blockNum_x);
    blockID_y = _MYID % (blockNum_y * blockNum_x) / blockNum_x;
    blockID_x = _MYID % blockNum_x;

    DIMZ_padding = DIMZ + (LAYERS % DIMZ) / blockNum_z;
    DIMY_padding = DIMY + (NUMROWS % DIMY) / blockNum_y;
    DIMX_padding = DIMX + (NUMCOLS % DIMX) / blockNum_x;

    odd_block_z = LAYERS % DIMZ % blockNum_z;
    odd_block_y = NUMROWS % DIMY % blockNum_y;
    odd_block_x = NUMCOLS % DIMX % blockNum_x;

    DIMZ_final = DIMZ_padding, DIMY_final = DIMY_padding, DIMX_final = DIMX_padding;

    if (blockID_z < odd_block_z)
    {
        DIMZ_final += 1;
        left_z_block = blockID_z * (DIMZ_padding + 1) - R * DIMT;
        right_z_block = (blockID_z + 1) * (DIMZ_padding + 1) + R * DIMT;
    }
    else
    {
        left_z_block = odd_block_z * (DIMZ_padding + 1) + (blockID_z - odd_block_z) * DIMZ_padding - R * DIMT;
        right_z_block = odd_block_z * (DIMZ_padding + 1) + (blockID_z - odd_block_z + 1) * DIMZ_padding + R * DIMT;
    }
    if (blockID_y < odd_block_y)
    {
        DIMY_final += 1;
        left_y_block = blockID_y * (DIMY_padding + 1) - R * DIMT;
        right_y_block = (blockID_y + 1) * (DIMY_padding + 1) + R * DIMT;
    }
    else
    {
        left_y_block = odd_block_y * (DIMY_padding + 1) + (blockID_y - odd_block_y) * DIMY_padding - R * DIMT;
        right_y_block = odd_block_y * (DIMY_padding + 1) + (blockID_y - odd_block_y + 1) * DIMY_padding + R * DIMT;
    }
    if (blockID_x < odd_block_x)
    {
        DIMX_final += 1;
        left_x_block = blockID_x * (DIMX_padding + 1) - R * DIMT;
        right_x_block = (blockID_x + 1) * (DIMX_padding + 1) + R * DIMT;
    }
    else
    {
        left_x_block = odd_block_x * (DIMX_padding + 1) + (blockID_x - odd_block_x) * DIMX_padding - R * DIMT;
        right_x_block = odd_block_x * (DIMX_padding + 1) + (blockID_x - odd_block_x + 1) * DIMX_padding + R * DIMT;
    }

    left_z_load = left_z_block >= 0 ? left_z_block : 0;
    right_z_load = right_z_block <= nz ? right_z_block : nz;
    left_y_load = left_y_block >= 0 ? left_y_block : 0;
    right_y_load = right_y_block <= ny ? right_y_block : ny;
    left_x_load = left_x_block >= 0 ? left_x_block : 0;
    right_x_load = right_x_block <= nx ? right_x_block : nx;
    load_size = right_x_load - left_x_load;

    blockSize_x = DIMX_final + 2 * R * DIMT;
    if (DIMT == 1)
        blockSize_x = blockSize_x + (4 - blockSize_x % 4);
    blockSize_y = DIMY_final + 2 * R * DIMT;
    blockSize_z = 1 + 2 * R;
    blockSize_z_0 = blockSize_z + 1; // preload

    for (i = 0; i < NUMTIMESTEPS / DIMT; i++)
    {
        int tmp = in;
        in = out;
        out = tmp;

        global_tIn = temp[in];
        global_tOut = temp[out];

        local_tIn_0 = (DEFINED_DATATYPE *)ldm_malloc(blockSize_z_0 * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE)); // preload
        if (DIMT >= 2)
        {
            local_tIn = (DEFINED_DATATYPE *)ldm_malloc((DIMT - 1) * blockSize_z * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
            local_tIn_t = (DEFINED_DATATYPE **)ldm_malloc(DIMT * sizeof(DEFINED_DATATYPE *));
            for (t = 0; t < DIMT - 1; t++)
                local_tIn_t[t + 1] = &local_tIn[t * blockSize_z * blockSize_y * blockSize_x];
        }
        local_tOut_1 = (DEFINED_DATATYPE *)ldm_malloc(blockSize_x * sizeof(DEFINED_DATATYPE));
        local_tOut_2 = (DEFINED_DATATYPE *)ldm_malloc(blockSize_x * sizeof(DEFINED_DATATYPE));

        tOut_ptr = local_tOut_1;
        tOut_push_ptr = local_tOut_2;
        DMA_push = 1;

        for (t = 0; t <= DIMT; t++)
            left_z_compute_t[t] = left_z_load + 1 + 2 * R * t - 1;

        for (z = left_z_load; z < left_z_compute_t[DIMT]; z++)
        {
            t = 0;
            offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z_0;
            for (y = left_y_load; y < right_y_load; y++)
            {
                C_global = z * ny * nx + y * nx + left_x_load;
                C_local = offset_z_block_t * blockSize_y * blockSize_x + (y - left_y_load) * blockSize_x;
                DMA_reply = 0;
                athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_reply, 0, 0, 0);
                while (DMA_reply != 1)
                    ;
            }

            if (z < left_z_compute_t[t + 1])
                continue;

            N_z_block_t = (offset_z_block_t - R - 1 + blockSize_z_0) % blockSize_z_0;
            C_z_block_t = (offset_z_block_t - R + blockSize_z_0) % blockSize_z_0;
            S_z_block_t = (offset_z_block_t - R + 1 + blockSize_z_0) % blockSize_z_0;
            offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
            for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
            {
                for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                {
                    C_next = offset_z_block_t_next * blockSize_y * blockSize_x + y * blockSize_x + x;
                    C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    W = C - 1;
                    E = C + 1;
                    N = C - blockSize_x;
                    S = C + blockSize_x;
                    B = N_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    T = S_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    local_tIn_t[t + 1][C_next] = c0 * local_tIn_0[C] + c1 * local_tIn_0[B] + c2 * local_tIn_0[T] + c3 * local_tIn_0[W] + c4 * local_tIn_0[E] + c5 * local_tIn_0[S] + c6 * local_tIn_0[N];
                }
            }

            for (t = 1; t < DIMT; t++)
            {
                offset_z_block_t = (z - left_z_compute_t[t] + blockSize_z) % blockSize_z;
                if (z < left_z_compute_t[t + 1])
                    continue;
                N_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                S_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                    {
                        C_next = offset_z_block_t_next * blockSize_y * blockSize_x + y * blockSize_x + x;
                        C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        W = C - 1;
                        E = C + 1;
                        N = C - blockSize_x;
                        S = C + blockSize_x;
                        B = N_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        T = S_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        local_tIn_t[t + 1][C_next] = c0 * local_tIn_t[t][C] + c1 * local_tIn_t[t][B] + c2 * local_tIn_t[t][T] + c3 * local_tIn_t[t][W] + c4 * local_tIn_t[t][E] + c5 * local_tIn_t[t][S] + c6 * local_tIn_t[t][N];
                    }
                }
            }
        }

        t = 0;
        z = left_z_compute_t[DIMT];
        offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z_0;
        for (y = left_y_load; y < right_y_load; y++)
        {
            C_global = z * ny * nx + y * nx + left_x_load;
            C_local = offset_z_block_t * blockSize_y * blockSize_x + (y - left_y_load) * blockSize_x;
            DMA_reply = 0;
            athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_reply, 0, 0, 0);
            while (DMA_reply != 1)
                ;
        }

        for (z = left_z_compute_t[DIMT]; z < right_z_load; z++)
        {
            t = 0;
            DMA_preload = 0;
            wait_num = 0;
            if (z + 1 < right_z_load)
            {
                int offset_z_block_0_preload = (z + 1 - left_z_compute_t[t]) % blockSize_z_0;
                for (y = left_y_load; y < right_y_load; y++)
                {
                    C_global = (z + 1) * ny * nx + y * nx + left_x_load;
                    C_local = offset_z_block_0_preload * blockSize_y * blockSize_x + (y - left_y_load) * blockSize_x;
                    athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_preload, 0, 0, 0);
                    wait_num++;
                }
            }

            offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z_0;
            N_z_block_t = (offset_z_block_t - R - 1 + blockSize_z_0) % blockSize_z_0;
            C_z_block_t = (offset_z_block_t - R + blockSize_z_0) % blockSize_z_0;
            S_z_block_t = (offset_z_block_t - R + 1 + blockSize_z_0) % blockSize_z_0;
            offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
            aligned = (R * (t + 1) + 2) % 4;
            if (DIMT >= 2)
            {
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    for (global_x = left_x_load + R * (t + 1) - aligned, x = R * (t + 1) - aligned; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                    {
                        C_next = offset_z_block_t_next * blockSize_y * blockSize_x + y * blockSize_x + x;
                        C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        W = C - 1;
                        E = C + 1;
                        N = C - blockSize_x;
                        S = C + blockSize_x;
                        B = N_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        T = S_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        vload_3p_0(C, v_W, v_C, v_E, local_tIn_0);
                        simd_load(v_B, &local_tIn_0[B]);
                        simd_load(v_T, &local_tIn_0[T]);
                        simd_load(v_S, &local_tIn_0[S]);
                        simd_load(v_N, &local_tIn_0[N]);
                        v_res = v_c0 * v_C + v_c1 * v_B + v_c2 * v_T + v_c3 * v_W + v_c4 * v_E + v_c5 * v_S + v_c6 * v_N;
                        simd_store(v_res, &local_tIn_t[t + 1][C_next]);
                        // local_tIn_t[t + 1][C_next] = c0 * local_tIn_0[C] + c1 * local_tIn_0[B] + c2 * local_tIn_0[T] + c3 * local_tIn_0[W] + c4 * local_tIn_0[E] + c5 * local_tIn_0[S] + c6 * local_tIn_0[N];
                    }
                }
            }
            else
            {
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    for (global_x = left_x_load + R * (t + 1) - aligned, x = R * (t + 1) - aligned; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                    {
                        C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        W = C - 1;
                        E = C + 1;
                        N = C - blockSize_x;
                        S = C + blockSize_x;
                        B = N_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        T = S_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        vload_3p_0(C, v_W, v_C, v_E, local_tIn_0);
                        simd_load(v_B, &local_tIn_0[B]);
                        simd_load(v_T, &local_tIn_0[T]);
                        simd_load(v_S, &local_tIn_0[S]);
                        simd_load(v_N, &local_tIn_0[N]);
                        v_res = v_c0 * v_C + v_c1 * v_B + v_c2 * v_T + v_c3 * v_W + v_c4 * v_E + v_c5 * v_S + v_c6 * v_N;
                        simd_store(v_res, &tOut_ptr[x]);
                        // local_tIn_t[t + 1][C_next] = c0 * local_tIn_0[C] + c1 * local_tIn_0[B] + c2 * local_tIn_0[T] + c3 * local_tIn_0[W] + c4 * local_tIn_0[E] + c5 * local_tIn_0[S] + c6 * local_tIn_0[N];
                    }

                    while (DMA_push != 1)
                        ;

                    DEFINED_DATATYPE *t;
                    t = tOut_ptr;
                    tOut_ptr = tOut_push_ptr;
                    tOut_push_ptr = t;

                    DMA_push = 0;
                    C_global = (z - R * DIMT) * ny * nx + global_y * nx + left_x_load + R * DIMT;
                    athread_put(PE_MODE, tOut_push_ptr + R * DIMT, &global_tOut[C_global], (load_size - 2 * R * DIMT) * sizeof(DEFINED_DATATYPE), &DMA_push, 0, 0);
                }

                while (DMA_preload != wait_num)
                    ;
            }

            if (DIMT >= 2)
            {
                for (t = 1; t < DIMT - 1; t++)
                {
                    offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z;
                    N_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                    C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                    S_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                    offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
                    aligned = (R * (t + 1)) % 4;
                    if (DIMT == 4)
                        aligned = (R * (t + 1) + 2) % 4;
                    for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                    {
                        for (global_x = left_x_load + R * (t + 1) - aligned, x = R * (t + 1) - aligned; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                        {
                            C_next = offset_z_block_t_next * blockSize_y * blockSize_x + y * blockSize_x + x;
                            C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                            W = C - 1;
                            E = C + 1;
                            N = C - blockSize_x;
                            S = C + blockSize_x;
                            B = N_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                            T = S_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                            vload_3p_0(C, v_W, v_C, v_E, local_tIn_t[t]);
                            simd_load(v_B, &local_tIn_t[t][B]);
                            simd_load(v_T, &local_tIn_t[t][T]);
                            simd_load(v_S, &local_tIn_t[t][S]);
                            simd_load(v_N, &local_tIn_t[t][N]);
                            v_res = v_c0 * v_C + v_c1 * v_B + v_c2 * v_T + v_c3 * v_W + v_c4 * v_E + v_c5 * v_S + v_c6 * v_N;
                            simd_store(v_res, &local_tIn_t[t + 1][C_next]);
                            // local_tIn_t[t + 1][C_next] = c0 * local_tIn_t[t][C] + c1 * local_tIn_t[t][B] + c2 * local_tIn_t[t][T] + c3 * local_tIn_t[t][W] + c4 * local_tIn_t[t][E] + c5 * local_tIn_t[t][S] + c6 * local_tIn_t[t][N];
                        }
                    }
                }

                t = DIMT - 1;
                offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z;
                N_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                S_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                aligned = (R * (t + 1) + 2) % 4;
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    for (global_x = left_x_load + R * (t + 1) - aligned, x = R * (t + 1) - aligned; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                    {
                        C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        W = C - 1;
                        E = C + 1;
                        N = C - blockSize_x;
                        S = C + blockSize_x;
                        B = N_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        T = S_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        vload_3p_0(C, v_W, v_C, v_E, local_tIn_t[t]);
                        simd_load(v_B, &local_tIn_t[t][B]);
                        simd_load(v_T, &local_tIn_t[t][T]);
                        simd_load(v_S, &local_tIn_t[t][S]);
                        simd_load(v_N, &local_tIn_t[t][N]);
                        v_res = v_c0 * v_C + v_c1 * v_B + v_c2 * v_T + v_c3 * v_W + v_c4 * v_E + v_c5 * v_S + v_c6 * v_N;
                        simd_storeu(v_res, &tOut_ptr[x]);
                        // tOut_ptr[x] = c0 * local_tIn_t[t][C] + c1 * local_tIn_t[t][B] + c2 * local_tIn_t[t][T] + c3 * local_tIn_t[t][W] + c4 * local_tIn_t[t][E] + c5 * local_tIn_t[t][S] + c6 * local_tIn_t[t][N];
                    }

                    while (DMA_push != 1)
                        ;

                    DEFINED_DATATYPE *t;
                    t = tOut_ptr;
                    tOut_ptr = tOut_push_ptr;
                    tOut_push_ptr = t;

                    DMA_push = 0;
                    C_global = (z - R * DIMT) * ny * nx + global_y * nx + left_x_load + R * DIMT;
                    athread_put(PE_MODE, tOut_push_ptr + R * DIMT, &global_tOut[C_global], (load_size - 2 * R * DIMT) * sizeof(DEFINED_DATATYPE), &DMA_push, 0, 0);
                }

                while (DMA_preload != wait_num)
                    ;
            }
        }

        while (DMA_push != 1)
            ;

        athread_syn(ARRAY_SCOPE, 0xffff);

        ldm_free(local_tIn_0, blockSize_z_0 * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
        if (DIMT >= 2)
        {
            ldm_free(local_tIn, (DIMT - 1) * blockSize_z * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
            ldm_free(local_tIn_t, DIMT * sizeof(DEFINED_DATATYPE *));
        }
        ldm_free(local_tOut_1, blockSize_x * sizeof(DEFINED_DATATYPE));
        ldm_free(local_tOut_2, blockSize_x * sizeof(DEFINED_DATATYPE));
    }
    spe_param->out = out;
}
