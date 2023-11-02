#include "common.h"
#include <simd.h>
#include <slave.h>


#define vload_1p_1(c, v_l, v_c, v_r, arr)  \
    {                                      \
        simd_load(v_l, &arr[c - 1]);       \
        simd_load(v_tc1, &arr[c + 3]);     \
        v_r = simd_vshff(v_tc1, v_l, rc1); \
        v_c = simd_vshff(v_r, v_l, rc2);   \
    }

#define vload_1p_2(c, v_l, v_c, v_r, arr)    \
    {                                        \
        simd_load(v_tc1, &arr[c - 2]);       \
        simd_load(v_tc2, &arr[c + 2]);       \
        v_c = simd_vshff(v_tc2, v_tc1, rc1); \
    }

#define vload_1p_3(c, v_l, v_c, v_r, arr)  \
    {                                      \
        simd_load(v_tc1, &arr[c - 3]);     \
        simd_load(v_r, &arr[c + 1]);       \
        v_l = simd_vshff(v_r, v_tc1, rc1); \
        v_c = simd_vshff(v_r, v_l, rc2);   \
    }

#define vload_5p_0(c, v_l2, v_l1, v_c, v_r1, v_r2, arr) \
    {                                                   \
        simd_load(v_c, &arr[c]);                        \
        simd_load(v_tc1, &arr[c - 4]);                  \
        simd_load(v_tc2, &arr[c + 4]);                  \
        v_l2 = simd_vshff(v_c, v_tc1, rc1);             \
        v_l1 = simd_vshff(v_c, v_l2, rc2);              \
        v_r2 = simd_vshff(v_tc2, v_c, rc1);             \
        v_r1 = simd_vshff(v_r2, v_c, rc2);              \
    }

#define vload_5p_1(c, v_l2, v_l1, v_c, v_r1, v_r2, arr) \
    {                                                   \
        simd_load(v_l1, &arr[c - 1]);                   \
        simd_load(v_tc1, &arr[c + 3]);                  \
        simd_load(v_tc2, &arr[c - 5]);                  \
        v_r1 = simd_vshff(v_tc1, v_l1, rc1);            \
        v_c = simd_vshff(v_r1, v_l1, rc2);              \
        v_r2 = simd_vshff(v_tc1, v_r1, rc2);            \
        v_t1 = simd_vshff(v_l1, v_tc2, rc1);            \
        v_l2 = simd_vshff(v_l1, v_t1, rc2);             \
    }

#define vload_5p_2(c, v_l2, v_l1, v_c, v_r1, v_r2, arr) \
    {                                                   \
        simd_load(v_l2, &arr[c - 2]);                   \
        simd_load(v_r2, &arr[c + 2]);                   \
        v_c = simd_vshff(v_r2, v_l2, rc1);              \
        v_l1 = simd_vshff(v_c, v_l2, rc2);              \
        v_r1 = simd_vshff(v_r2, v_c, rc2);              \
    }

#define vload_5p_3(c, v_l2, v_l1, v_c, v_r1, v_r2, arr) \
    {                                                   \
        simd_load(v_tc1, &arr[c - 3]);                  \
        simd_load(v_r1, &arr[c + 1]);                   \
        simd_load(v_tc2, &arr[c + 5]);                  \
        v_l1 = simd_vshff(v_r1, v_tc1, rc1);            \
        v_c = simd_vshff(v_r1, v_l1, rc2);              \
        v_l2 = simd_vshff(v_l1, v_tc1, rc2);            \
        v_t1 = simd_vshff(v_tc2, v_r1, rc1);            \
        v_r2 = simd_vshff(v_t1, v_r1, rc2);             \
    }

// __thread_local DEFINED_DATATYPE local_tIn_t[DIMT][(1 + 2 * R) * (DIMY + 2 * R * DIMT) * (DIMX + 2 * R * DIMT)];
// __thread_local DEFINED_DATATYPE local_tOut[DIMX + 2 * R * DIMT];

void spe_func(struct spe_parameter *spe_param)
{
    int DIMT, DIMX, DIMY, DIMZ;
    DEFINED_DATATYPE a, b, h2inv, c, d, e, f;
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
    int offset_z_block_t, N2_z_block_t, N1_z_block_t, C_z_block_t, S1_z_block_t, S2_z_block_t, offset_z_block_t_next;
    int C_next, C, B1, B2, T1, T2, W1, W2, E1, E2, S1, S2, N1, N2;
    int C_global, C_local;
    volatile int DMA_reply, DMA_push;
    volatile int DMA_preload, wait_num;
    DEFINED_DATATYPE *tOut_ptr, *tOut_push_ptr;
    int i;

    temp = spe_param->temp;
    a = spe_param->a;
    b = spe_param->b;
    h2inv = spe_param->h2inv;
    nt = spe_param->nt;
    nx = spe_param->nx;
    ny = spe_param->ny;
    nz = spe_param->nz;
    DIMT = spe_param->DIMT;
    DIMX = spe_param->DIMX;
    DIMY = spe_param->DIMY;
    DIMZ = spe_param->DIMZ;

    c = b * h2inv * 0.0833;
    d = c * 1.0;
    e = c * 16.0;
    f = c * 90.0;

    int aligned;
    DEFINED_V_DATATYPE v_C, v_B1, v_B2, v_T1, v_T2, v_W1, v_W2, v_E1, v_E2, v_S1, v_S2, v_N1, v_N2, v_res;
    DEFINED_V_DATATYPE v_tc1, v_tc2, v_t1, v_t2, v_l, v_r;
    int rc1 = 0x4E;
    int rc2 = 0x99;
    DEFINED_V_DATATYPE v_a, v_f, v_e, v_d, v_asf;
#ifdef FLOAT
    v_a = simd_set_floatv4(a, a, a, a);
    v_f = simd_set_floatv4(f, f, f, f);
    v_e = simd_set_floatv4(e, e, e, e);
    v_d = simd_set_floatv4(d, d, d, d);
#else
    v_a = simd_set_doublev4(a, a, a, a);
    v_f = simd_set_doublev4(f, f, f, f);
    v_e = simd_set_doublev4(e, e, e, e);
    v_d = simd_set_doublev4(d, d, d, d);
#endif
    v_asf = v_a - v_f;

    blockNum_z = LAYERS / DIMZ;
    blockNum_y = NUMROWS / DIMY;
    blockNum_x = NUMCOLS / DIMX;

    DIMZ_padding = DIMZ + (LAYERS % DIMZ) / blockNum_z;
    DIMY_padding = DIMY + (NUMROWS % DIMY) / blockNum_y;
    DIMX_padding = DIMX + (NUMCOLS % DIMX) / blockNum_x;

    odd_block_z = LAYERS % DIMZ % blockNum_z;
    odd_block_y = NUMROWS % DIMY % blockNum_y;
    odd_block_x = NUMCOLS % DIMX % blockNum_x;

    blockID_z = _MYID / (blockNum_y * blockNum_x);
    blockID_y = _MYID % (blockNum_y * blockNum_x) / blockNum_x;
    blockID_x = _MYID % blockNum_x;

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
        local_tIn = (DEFINED_DATATYPE *)ldm_malloc((DIMT - 1) * blockSize_z * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
        local_tIn_t = (DEFINED_DATATYPE **)ldm_malloc(DIMT * sizeof(DEFINED_DATATYPE *));
        for (t = 0; t < DIMT - 1; t++)
            local_tIn_t[t + 1] = &local_tIn[t * blockSize_z * blockSize_y * blockSize_x];
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

            // athread_syn(ARRAY_SCOPE, 0xffff);

            N2_z_block_t = (offset_z_block_t - R - 2 + blockSize_z_0) % blockSize_z_0;
            N1_z_block_t = (offset_z_block_t - R - 1 + blockSize_z_0) % blockSize_z_0;
            C_z_block_t = (offset_z_block_t - R + blockSize_z_0) % blockSize_z_0;
            S1_z_block_t = (offset_z_block_t - R + 1 + blockSize_z_0) % blockSize_z_0;
            S2_z_block_t = (offset_z_block_t - R + 2 + blockSize_z_0) % blockSize_z_0;
            offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
            aligned = (R * (t + 1) + 2) % 4;
            for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
            {
                for (global_x = left_x_load + R * (t + 1) - aligned, x = R * (t + 1) - aligned; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                {
                    C_next = offset_z_block_t_next * blockSize_y * blockSize_x + y * blockSize_x + x;
                    C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    W1 = C - 1;
                    W2 = C - 2;
                    E1 = C + 1;
                    E2 = C + 2;
                    N1 = C - blockSize_x;
                    N2 = C - 2 * blockSize_x;
                    S1 = C + blockSize_x;
                    S2 = C + 2 * blockSize_x;
                    B1 = N1_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    B2 = N2_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    T1 = S1_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    T2 = S2_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    vload_5p_0(C, v_W2, v_W1, v_C, v_E1, v_E2, local_tIn_0);
                    simd_load(v_B1, &local_tIn_0[B1]);
                    simd_load(v_B2, &local_tIn_0[B2]);
                    simd_load(v_T1, &local_tIn_0[T1]);
                    simd_load(v_T2, &local_tIn_0[T2]);
                    simd_load(v_S1, &local_tIn_0[S1]);
                    simd_load(v_S2, &local_tIn_0[S2]);
                    simd_load(v_N1, &local_tIn_0[N1]);
                    simd_load(v_N2, &local_tIn_0[N2]);
                    v_res = v_asf * v_C + v_e * v_B1 + v_d * v_B2 + v_e * v_T1 + v_d * v_T2 + v_e * v_W1 + v_d * v_W2 + v_e * v_E1 + v_d * v_E2 + v_e * v_S1 + v_d * v_S2 + v_e * v_N1 + v_d * v_N2;
                    simd_store(v_res, &local_tIn_t[t + 1][C_next]);
                    // local_tIn_t[t + 1][C_next] = (a - f) * local_tIn_0[C] + e * local_tIn_0[B1] + d * local_tIn_0[B2] + e * local_tIn_0[T1] + d * local_tIn_0[T2] + e * local_tIn_0[W1] + d * local_tIn_0[W2] + e * local_tIn_0[E1] + d * local_tIn_0[E2] + e * local_tIn_0[S1] + d * local_tIn_0[S2] + e * local_tIn_0[N1] + d * local_tIn_0[N2];
                }
            }
            for (t = 1; t < DIMT; t++)
            {
                offset_z_block_t = (z - left_z_compute_t[t] + blockSize_z) % blockSize_z;
                if (z < left_z_compute_t[t + 1])
                    continue;
                N2_z_block_t = (offset_z_block_t - R - 2 + blockSize_z) % blockSize_z;
                N1_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                S1_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                S2_z_block_t = (offset_z_block_t - R + 2 + blockSize_z) % blockSize_z;
                offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
                aligned = (R * (t + 1) + 2) % 4;
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    for (global_x = left_x_load + R * (t + 1) - aligned, x = R * (t + 1) - aligned; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                    {
                        C_next = offset_z_block_t_next * blockSize_y * blockSize_x + y * blockSize_x + x;
                        C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        W1 = C - 1;
                        W2 = C - 2;
                        E1 = C + 1;
                        E2 = C + 2;
                        N1 = C - blockSize_x;
                        N2 = C - 2 * blockSize_x;
                        S1 = C + blockSize_x;
                        S2 = C + 2 * blockSize_x;
                        B1 = N1_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        B2 = N2_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        T1 = S1_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        T2 = S2_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        vload_5p_0(C, v_W2, v_W1, v_C, v_E1, v_E2, local_tIn_t[t]);
                        simd_load(v_B1, &local_tIn_t[t][B1]);
                        simd_load(v_B2, &local_tIn_t[t][B2]);
                        simd_load(v_T1, &local_tIn_t[t][T1]);
                        simd_load(v_T2, &local_tIn_t[t][T2]);
                        simd_load(v_S1, &local_tIn_t[t][S1]);
                        simd_load(v_S2, &local_tIn_t[t][S2]);
                        simd_load(v_N1, &local_tIn_t[t][N1]);
                        simd_load(v_N2, &local_tIn_t[t][N2]);
                        v_res = v_asf * v_C + v_e * v_B1 + v_d * v_B2 + v_e * v_T1 + v_d * v_T2 + v_e * v_W1 + v_d * v_W2 + v_e * v_E1 + v_d * v_E2 + v_e * v_S1 + v_d * v_S2 + v_e * v_N1 + v_d * v_N2;
                        simd_store(v_res, &local_tIn_t[t + 1][C_next]);
                        // local_tIn_t[t + 1][C_next] = (a - f) * local_tIn_t[t][C] + e * local_tIn_t[t][B1] + d * local_tIn_t[t][B2] + e * local_tIn_t[t][T1] + d * local_tIn_t[t][T2] + e * local_tIn_t[t][W1] + d * local_tIn_t[t][W2] + e * local_tIn_t[t][E1] + d * local_tIn_t[t][E2] + e * local_tIn_t[t][S1] + d * local_tIn_t[t][S2] + e * local_tIn_t[t][N1] + d * local_tIn_t[t][N2];
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

            // athread_syn(ARRAY_SCOPE, 0xffff);

            offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z_0;
            N2_z_block_t = (offset_z_block_t - R - 2 + blockSize_z_0) % blockSize_z_0;
            N1_z_block_t = (offset_z_block_t - R - 1 + blockSize_z_0) % blockSize_z_0;
            C_z_block_t = (offset_z_block_t - R + blockSize_z_0) % blockSize_z_0;
            S1_z_block_t = (offset_z_block_t - R + 1 + blockSize_z_0) % blockSize_z_0;
            S2_z_block_t = (offset_z_block_t - R + 2 + blockSize_z_0) % blockSize_z_0;
            offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
            aligned = (R * (t + 1) + 2) % 4;
            for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
            {
                for (global_x = left_x_load + R * (t + 1) - aligned, x = R * (t + 1) - aligned; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                {
                    C_next = offset_z_block_t_next * blockSize_y * blockSize_x + y * blockSize_x + x;
                    C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    W1 = C - 1;
                    W2 = C - 2;
                    E1 = C + 1;
                    E2 = C + 2;
                    N1 = C - blockSize_x;
                    N2 = C - 2 * blockSize_x;
                    S1 = C + blockSize_x;
                    S2 = C + 2 * blockSize_x;
                    B1 = N1_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    B2 = N2_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    T1 = S1_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    T2 = S2_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    vload_5p_0(C, v_W2, v_W1, v_C, v_E1, v_E2, local_tIn_0);
                    simd_load(v_B1, &local_tIn_0[B1]);
                    simd_load(v_B2, &local_tIn_0[B2]);
                    simd_load(v_T1, &local_tIn_0[T1]);
                    simd_load(v_T2, &local_tIn_0[T2]);
                    simd_load(v_S1, &local_tIn_0[S1]);
                    simd_load(v_S2, &local_tIn_0[S2]);
                    simd_load(v_N1, &local_tIn_0[N1]);
                    simd_load(v_N2, &local_tIn_0[N2]);
                    v_res = v_asf * v_C + v_e * v_B1 + v_d * v_B2 + v_e * v_T1 + v_d * v_T2 + v_e * v_W1 + v_d * v_W2 + v_e * v_E1 + v_d * v_E2 + v_e * v_S1 + v_d * v_S2 + v_e * v_N1 + v_d * v_N2;
                    simd_store(v_res, &local_tIn_t[t + 1][C_next]);
                    // local_tIn_t[t + 1][C_next] = (a - f) * local_tIn_0[C] + e * local_tIn_0[B1] + d * local_tIn_0[B2] + e * local_tIn_0[T1] + d * local_tIn_0[T2] + e * local_tIn_0[W1] + d * local_tIn_0[W2] + e * local_tIn_0[E1] + d * local_tIn_0[E2] + e * local_tIn_0[S1] + d * local_tIn_0[S2] + e * local_tIn_0[N1] + d * local_tIn_0[N2];
                }
            }

            for (t = 1; t < DIMT - 1; t++)
            {
                offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z;
                N2_z_block_t = (offset_z_block_t - R - 2 + blockSize_z) % blockSize_z;
                N1_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                S1_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                S2_z_block_t = (offset_z_block_t - R + 2 + blockSize_z) % blockSize_z;
                offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
                aligned = (R * (t + 1) + 2) % 4;
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    for (global_x = left_x_load + R * (t + 1) - aligned, x = R * (t + 1) - aligned; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                    {
                        C_next = offset_z_block_t_next * blockSize_y * blockSize_x + y * blockSize_x + x;
                        C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        W1 = C - 1;
                        W2 = C - 2;
                        E1 = C + 1;
                        E2 = C + 2;
                        N1 = C - blockSize_x;
                        N2 = C - 2 * blockSize_x;
                        S1 = C + blockSize_x;
                        S2 = C + 2 * blockSize_x;
                        B1 = N1_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        B2 = N2_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        T1 = S1_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        T2 = S2_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        vload_5p_0(C, v_W2, v_W1, v_C, v_E1, v_E2, local_tIn_t[t]);
                        simd_load(v_B1, &local_tIn_t[t][B1]);
                        simd_load(v_B2, &local_tIn_t[t][B2]);
                        simd_load(v_T1, &local_tIn_t[t][T1]);
                        simd_load(v_T2, &local_tIn_t[t][T2]);
                        simd_load(v_S1, &local_tIn_t[t][S1]);
                        simd_load(v_S2, &local_tIn_t[t][S2]);
                        simd_load(v_N1, &local_tIn_t[t][N1]);
                        simd_load(v_N2, &local_tIn_t[t][N2]);
                        v_res = v_asf * v_C + v_e * v_B1 + v_d * v_B2 + v_e * v_T1 + v_d * v_T2 + v_e * v_W1 + v_d * v_W2 + v_e * v_E1 + v_d * v_E2 + v_e * v_S1 + v_d * v_S2 + v_e * v_N1 + v_d * v_N2;
                        simd_store(v_res, &local_tIn_t[t + 1][C_next]);
                        // local_tIn_t[t + 1][C_next] = (a - f) * local_tIn_t[t][C] + e * local_tIn_t[t][B1] + d * local_tIn_t[t][B2] + e * local_tIn_t[t][T1] + d * local_tIn_t[t][T2] + e * local_tIn_t[t][W1] + d * local_tIn_t[t][W2] + e * local_tIn_t[t][E1] + d * local_tIn_t[t][E2] + e * local_tIn_t[t][S1] + d * local_tIn_t[t][S2] + e * local_tIn_t[t][N1] + d * local_tIn_t[t][N2];
                    }
                }
            }

            t = DIMT - 1;
            offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z;
            N2_z_block_t = (offset_z_block_t - R - 2 + blockSize_z) % blockSize_z;
            N1_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
            C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
            S1_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
            S2_z_block_t = (offset_z_block_t - R + 2 + blockSize_z) % blockSize_z;
            offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
            aligned = (R * (t + 1) + 2) % 4;
            for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
            {
                for (global_x = left_x_load + R * (t + 1) - aligned, x = R * (t + 1) - aligned; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                {
                    C_next = offset_z_block_t_next * blockSize_y * blockSize_x + y * blockSize_x + x;
                    C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    W1 = C - 1;
                    W2 = C - 2;
                    E1 = C + 1;
                    E2 = C + 2;
                    N1 = C - blockSize_x;
                    N2 = C - 2 * blockSize_x;
                    S1 = C + blockSize_x;
                    S2 = C + 2 * blockSize_x;
                    B1 = N1_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    B2 = N2_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    T1 = S1_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    T2 = S2_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                    vload_5p_0(C, v_W2, v_W1, v_C, v_E1, v_E2, local_tIn_t[t]);
                    simd_load(v_B1, &local_tIn_t[t][B1]);
                    simd_load(v_B2, &local_tIn_t[t][B2]);
                    simd_load(v_T1, &local_tIn_t[t][T1]);
                    simd_load(v_T2, &local_tIn_t[t][T2]);
                    simd_load(v_S1, &local_tIn_t[t][S1]);
                    simd_load(v_S2, &local_tIn_t[t][S2]);
                    simd_load(v_N1, &local_tIn_t[t][N1]);
                    simd_load(v_N2, &local_tIn_t[t][N2]);
                    v_res = v_asf * v_C + v_e * v_B1 + v_d * v_B2 + v_e * v_T1 + v_d * v_T2 + v_e * v_W1 + v_d * v_W2 + v_e * v_E1 + v_d * v_E2 + v_e * v_S1 + v_d * v_S2 + v_e * v_N1 + v_d * v_N2;
                    simd_storeu(v_res, &tOut_ptr[x]);
                    // tOut_ptr[x] = (a - f) * local_tIn_t[t][C] + e * local_tIn_t[t][B1] + d * local_tIn_t[t][B2] + e * local_tIn_t[t][T1] + d * local_tIn_t[t][T2] + e * local_tIn_t[t][W1] + d * local_tIn_t[t][W2] + e * local_tIn_t[t][E1] + d * local_tIn_t[t][E2] + e * local_tIn_t[t][S1] + d * local_tIn_t[t][S2] + e * local_tIn_t[t][N1] + d * local_tIn_t[t][N2];
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
        while (DMA_push != 1)
            ;

        athread_syn(ARRAY_SCOPE, 0xffff);

        ldm_free(local_tIn_0, blockSize_z_0 * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
        ldm_free(local_tIn, (DIMT - 1) * blockSize_z * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
        ldm_free(local_tIn_t, DIMT * sizeof(DEFINED_DATATYPE *));
        ldm_free(local_tOut_1, blockSize_x * sizeof(DEFINED_DATATYPE));
        ldm_free(local_tOut_2, blockSize_x * sizeof(DEFINED_DATATYPE));
    }
    spe_param->out = out;
}
