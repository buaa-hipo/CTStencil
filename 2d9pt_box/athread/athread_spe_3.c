#include "common.h"
#include <simd.h>
#include <slave.h>

/*
    double a[M][N];
    double b[M][N];
    double c0, c1, c2, c3, c4, c5, c6, c7, c8;
    for (long k = 1; k < M - 1; ++k)
    {
        for (long j = 1; j < N - 1; ++j)
        {
            b[k][j] = c0 * a[k][j] + c1 * a[k][j - 1] + c2 * a[k - 1][j]
                    + c3 * a[k][j + 1] + c4 * a[k + 1][j] + c5 * a[k - 1][j + 1]
                    + c6 * a[k + 1][j - 1] + c7 * a[k - 1][j - 1] + c8 * a[k + 1][j + 1];
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

extern DEFINED_DATATYPE c[NUMPOINTS];
// tIn, 0, 1, 2 ... DIMT - 1, DIMT(local_out), tOut
// __thread_local DEFINED_DATATYPE local_tIn_t[DIMT][(1 + 2 * R) * (DIMX + 2 * R * DIMT)];
// __thread_local DEFINED_DATATYPE local_tOut[DIMX + 2 * R * DIMT];

void spe_func(struct spe_parameter *spe_param)
{
    int DIMT, DIMX, DIMY;
    DEFINED_DATATYPE **temp;
    DEFINED_DATATYPE *local_tIn, *local_tOut_1, *local_tOut_2, *local_tIn_0, **local_tIn_t;
    int blockNum_x, blockNum_y, blockNum_y_group;
    int blockID_y, blockID_x;
    int odd_block_x, odd_block_y;
    int DIMX_padding, DIMY_padding, DIMX_final, DIMY_final;
    int blockSize_x, blockSize_y, blockSize_y_0;
    int t, y, global_x, x;
    int nt, nx, ny;
    int in = 1, out = 0;
    DEFINED_DATATYPE *global_tIn, *global_tOut;
    int left_y_block, right_y_block, left_x_block, right_x_block;
    int left_y_load, right_y_load, left_x_load, right_x_load, load_size, offset_x_block;
    int left_y_compute_t[DIMT + 1];
    DEFINED_DATATYPE c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7], c8 = c[8];
    int offset_y_block_t, N1_y_block_t, C_y_block_t, S1_y_block_t, offset_y_block_t_next;
    int C0_next, C0, C1, C2, C3, C4, C5, C6, C7, C8;
    int C_global, C_local;
    volatile int DMA_reply, DMA_push;
    volatile int DMA_preload, wait_num;
    DEFINED_DATATYPE *tOut_ptr, *tOut_push_ptr;
    int i;

    int aligned, offset;
    DEFINED_V_DATATYPE v_C0, v_C1, v_C2, v_C3, v_C4, v_C5, v_C6, v_C7, v_C8, v_res;
    DEFINED_V_DATATYPE v_tc1, v_tc2, v_t1, v_t2;
    int rc1 = 0x4E;
    int rc2 = 0x99;
    DEFINED_V_DATATYPE v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6, v_c7, v_c8;
#ifdef FLOAT
    v_c0 = simd_set_floatv4(c0, c0, c0, c0);
    v_c1 = simd_set_floatv4(c1, c1, c1, c1);
    v_c2 = simd_set_floatv4(c2, c2, c2, c2);
    v_c3 = simd_set_floatv4(c3, c3, c3, c3);
    v_c4 = simd_set_floatv4(c4, c4, c4, c4);
    v_c5 = simd_set_floatv4(c5, c5, c5, c5);
    v_c6 = simd_set_floatv4(c6, c6, c6, c6);
    v_c7 = simd_set_floatv4(c7, c7, c7, c7);
    v_c8 = simd_set_floatv4(c8, c8, c8, c8);
#else
    v_c0 = simd_set_doublev4(c0, c0, c0, c0);
    v_c1 = simd_set_doublev4(c1, c1, c1, c1);
    v_c2 = simd_set_doublev4(c2, c2, c2, c2);
    v_c3 = simd_set_doublev4(c3, c3, c3, c3);
    v_c4 = simd_set_doublev4(c4, c4, c4, c4);
    v_c5 = simd_set_doublev4(c5, c5, c5, c5);
    v_c6 = simd_set_doublev4(c6, c6, c6, c6);
    v_c7 = simd_set_doublev4(c7, c7, c7, c7);
    v_c8 = simd_set_doublev4(c8, c8, c8, c8);
#endif

    temp = spe_param->temp;
    nt = spe_param->nt;
    nx = spe_param->nx;
    ny = spe_param->ny;
    DIMT = spe_param->DIMT;
    DIMX = spe_param->DIMX;
    DIMY = spe_param->DIMY;

    blockNum_x = NUMCOLS / DIMX;
    blockNum_y = NUMROWS / DIMY;

    blockID_x = _MYID % blockNum_x;
    blockID_y = _MYID / blockNum_x;

    odd_block_x = NUMCOLS % DIMX % blockNum_x;
    odd_block_y = NUMROWS % DIMY % blockNum_y;

    DIMX_padding = DIMX + (NUMCOLS % DIMX) / blockNum_x;
    DIMY_padding = DIMY + (NUMROWS % DIMY) / blockNum_y;

    DIMX_final = DIMX_padding, DIMY_final = DIMY_padding;

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

    left_y_load = left_y_block >= 0 ? left_y_block : 0;
    right_y_load = right_y_block <= ny ? right_y_block : ny;
    left_x_load = left_x_block >= 0 ? left_x_block : 0;
    right_x_load = right_x_block <= nx ? right_x_block : nx;
    load_size = right_x_load - left_x_load;

    blockSize_x = DIMX_final + 2 * R * DIMT;
    if (DIMT == 1)
        blockSize_x = blockSize_x + (4 - blockSize_x % 4);
    blockSize_y = 1 + 2 * R;
    blockSize_y_0 = blockSize_y + 1; // preload

    for (i = 0; i < NUMTIMESTEPS / DIMT; i++)
    {
        int tmp = in;
        in = out;
        out = tmp;

        global_tIn = temp[in];
        global_tOut = temp[out];

        local_tIn_0 = (DEFINED_DATATYPE *)ldm_malloc(blockSize_y_0 * blockSize_x * sizeof(DEFINED_DATATYPE)); // preload
        if (DIMT >= 2)
        {
            local_tIn_t = (DEFINED_DATATYPE **)ldm_malloc(DIMT * sizeof(DEFINED_DATATYPE *));
            local_tIn = (DEFINED_DATATYPE *)ldm_malloc((DIMT - 1) * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
            for (t = 0; t < DIMT - 1; t++)
                local_tIn_t[t + 1] = &local_tIn[t * blockSize_y * blockSize_x];
        }

        local_tOut_1 = (DEFINED_DATATYPE *)ldm_malloc(blockSize_x * sizeof(DEFINED_DATATYPE));
        local_tOut_2 = (DEFINED_DATATYPE *)ldm_malloc(blockSize_x * sizeof(DEFINED_DATATYPE));

        tOut_ptr = local_tOut_1;
        tOut_push_ptr = local_tOut_2;
        DMA_push = 1;

        for (t = 0; t <= DIMT; t++)
            left_y_compute_t[t] = left_y_load + 1 + 2 * R * t - 1;

        for (y = left_y_load; y < left_y_compute_t[DIMT]; y++)
        {
            t = 0;
            offset_y_block_t = (y - left_y_compute_t[t]) % blockSize_y_0;
            DMA_reply = 0;
            C_global = y * nx + left_x_load;
            C_local = offset_y_block_t * blockSize_x;
            athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_reply, 0, 0, 0);
            while (DMA_reply != 1)
                ;

            if (y < left_y_compute_t[t + 1])
                continue;

            N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y_0) % blockSize_y_0;
            C_y_block_t = (offset_y_block_t - R + blockSize_y_0) % blockSize_y_0;
            S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y_0) % blockSize_y_0;
            offset_y_block_t_next = (y - left_y_compute_t[t + 1]) % blockSize_y;
            for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
            {
                C0_next = offset_y_block_t_next * blockSize_x + x;
                C2 = N1_y_block_t * blockSize_x + x;
                C7 = C2 - 1;
                C5 = C2 + 1;
                C0 = C_y_block_t * blockSize_x + x;
                C1 = C0 - 1;
                C3 = C0 + 1;
                C4 = S1_y_block_t * blockSize_x + x;
                C6 = C4 - 1;
                C8 = C4 + 1;
                local_tIn_t[t + 1][C0_next] = c0 * local_tIn_0[C0] + c1 * local_tIn_0[C1] + c2 * local_tIn_0[C2] + c3 * local_tIn_0[C3] + c4 * local_tIn_0[C4] + c5 * local_tIn_0[C5] + c6 * local_tIn_0[C6] + c7 * local_tIn_0[C7] + c8 * local_tIn_0[C8];
            }

            for (t = 1; t < DIMT; t++)
            {
                offset_y_block_t = (y - left_y_compute_t[t] + blockSize_y) % blockSize_y;
                if (y < left_y_compute_t[t + 1])
                    continue;
                N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y) % blockSize_y;
                C_y_block_t = (offset_y_block_t - R + blockSize_y) % blockSize_y;
                S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y) % blockSize_y;
                offset_y_block_t_next = (y - left_y_compute_t[t + 1]) % blockSize_y;
                for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                {
                    C0_next = offset_y_block_t_next * blockSize_x + x;
                    C2 = N1_y_block_t * blockSize_x + x;
                    C7 = C2 - 1;
                    C5 = C2 + 1;
                    C0 = C_y_block_t * blockSize_x + x;
                    C1 = C0 - 1;
                    C3 = C0 + 1;
                    C4 = S1_y_block_t * blockSize_x + x;
                    C6 = C4 - 1;
                    C8 = C4 + 1;
                    local_tIn_t[t + 1][C0_next] = c0 * local_tIn_t[t][C0] + c1 * local_tIn_t[t][C1] + c2 * local_tIn_t[t][C2] + c3 * local_tIn_t[t][C3] + c4 * local_tIn_t[t][C4] + c5 * local_tIn_t[t][C5] + c6 * local_tIn_t[t][C6] + c7 * local_tIn_t[t][C7] + c8 * local_tIn_t[t][C8];
                }
            }
        }

        t = 0;
        y = left_y_compute_t[DIMT];
        offset_y_block_t = (y - left_y_compute_t[t]) % blockSize_y_0;
        DMA_reply = 0;
        C_global = y * nx + left_x_load;
        C_local = offset_y_block_t * blockSize_x;
        athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_reply, 0, 0, 0);
        while (DMA_reply != 1)
            ;

        for (y = left_y_compute_t[DIMT]; y < right_y_load; y++)
        {
            t = 0;
            DMA_preload = 0;
            wait_num = 0;
            if (y + 1 < right_y_load)
            {
                int offset_y_block_0_preload = (y + 1 - left_y_compute_t[t]) % blockSize_y_0;
                C_global = (y + 1) * nx + left_x_load;
                C_local = offset_y_block_0_preload * blockSize_x;
                athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_preload, 0, 0, 0);
                wait_num++;
            }

            offset_y_block_t = (y - left_y_compute_t[t]) % blockSize_y_0;
            N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y_0) % blockSize_y_0;
            C_y_block_t = (offset_y_block_t - R + blockSize_y_0) % blockSize_y_0;
            S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y_0) % blockSize_y_0;
            offset_y_block_t_next = (y - left_y_compute_t[t + 1]) % blockSize_y;
            aligned = (R * (t + 1) + 3) % 4;
            offset = (aligned == 0 ? 3 : aligned - 1);
            if (DIMT >= 2)
            {
                for (global_x = left_x_load + R * (t + 1) - offset, x = R * (t + 1) - offset; global_x < right_x_load - R * (t + 1);)
                {
                    C0_next = offset_y_block_t_next * blockSize_x + x;
                    C2 = N1_y_block_t * blockSize_x + x;
                    C7 = C2 - 1;
                    C5 = C2 + 1;
                    C0 = C_y_block_t * blockSize_x + x;
                    C1 = C0 - 1;
                    C3 = C0 + 1;
                    C4 = S1_y_block_t * blockSize_x + x;
                    C6 = C4 - 1;
                    C8 = C4 + 1;
                    if (global_x >= left_x_load + 4 && global_x <= right_x_load - 4)
                    {
                        vload_3p_1(C2, v_C7, v_C2, v_C5, local_tIn_0);
                        vload_3p_1(C0, v_C1, v_C0, v_C3, local_tIn_0);
                        vload_3p_1(C4, v_C6, v_C4, v_C8, local_tIn_0);
                        v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8;
                        simd_storeu(v_res, &local_tIn_t[t + 1][C0_next]);
                        global_x += 4;
                        x += 4;
                    }
                    else
                    {
                        local_tIn_t[t + 1][C0_next] = c0 * local_tIn_0[C0] + c1 * local_tIn_0[C1] + c2 * local_tIn_0[C2] + c3 * local_tIn_0[C3] + c4 * local_tIn_0[C4] + c5 * local_tIn_0[C5] + c6 * local_tIn_0[C6] + c7 * local_tIn_0[C7] + c8 * local_tIn_0[C8];
                        global_x++;
                        x++;
                    }
                }
            }
            else
            {
                for (global_x = left_x_load + R * (t + 1) - offset, x = R * (t + 1) - offset; global_x < right_x_load - R * (t + 1);)
                {
                    C2 = N1_y_block_t * blockSize_x + x;
                    C7 = C2 - 1;
                    C5 = C2 + 1;
                    C0 = C_y_block_t * blockSize_x + x;
                    C1 = C0 - 1;
                    C3 = C0 + 1;
                    C4 = S1_y_block_t * blockSize_x + x;
                    C6 = C4 - 1;
                    C8 = C4 + 1;
                    if (global_x >= left_x_load + 4 && global_x <= right_x_load - 4)
                    {
                        vload_3p_1(C2, v_C7, v_C2, v_C5, local_tIn_0);
                        vload_3p_1(C0, v_C1, v_C0, v_C3, local_tIn_0);
                        vload_3p_1(C4, v_C6, v_C4, v_C8, local_tIn_0);
                        v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8;
                        simd_storeu(v_res, &tOut_ptr[x]);
                        global_x += 4;
                        x += 4;
                    }
                    else
                    {
                        tOut_ptr[x] = c0 * local_tIn_0[C0] + c1 * local_tIn_0[C1] + c2 * local_tIn_0[C2] + c3 * local_tIn_0[C3] + c4 * local_tIn_0[C4] + c5 * local_tIn_0[C5] + c6 * local_tIn_0[C6] + c7 * local_tIn_0[C7] + c8 * local_tIn_0[C8];
                        global_x++;
                        x++;
                    }
                }
            }

            if (DIMT >= 2)
            {
                for (t = 1; t < DIMT - 1; t++)
                {
                    offset_y_block_t = (y - left_y_compute_t[t]) % blockSize_y;
                    N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y) % blockSize_y;
                    C_y_block_t = (offset_y_block_t - R + blockSize_y) % blockSize_y;
                    S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y) % blockSize_y;
                    offset_y_block_t_next = (y - left_y_compute_t[t + 1]) % blockSize_y;
                    aligned = (R * (t + 1) + 2) % 4;
                    offset = (aligned == 0 ? 3 : aligned - 1);
                    for (global_x = left_x_load + R * (t + 1) - offset, x = R * (t + 1) - offset; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                    {
                        C0_next = offset_y_block_t_next * blockSize_x + x;
                        C2 = N1_y_block_t * blockSize_x + x;
                        C7 = C2 - 1;
                        C5 = C2 + 1;
                        C0 = C_y_block_t * blockSize_x + x;
                        C1 = C0 - 1;
                        C3 = C0 + 1;
                        C4 = S1_y_block_t * blockSize_x + x;
                        C6 = C4 - 1;
                        C8 = C4 + 1;
                        vload_3p_1(C2, v_C7, v_C2, v_C5, local_tIn_t[t]);
                        vload_3p_1(C0, v_C1, v_C0, v_C3, local_tIn_t[t]);
                        vload_3p_1(C4, v_C6, v_C4, v_C8, local_tIn_t[t]);
                        v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8;
                        simd_storeu(v_res, &local_tIn_t[t + 1][C0_next]);
                        // local_tIn_t[t + 1][C0_next] = c0 * local_tIn_t[t][C0] + c1 * local_tIn_t[t][C1] + c2 * local_tIn_t[t][C2] + c3 * local_tIn_t[t][C3] + c4 * local_tIn_t[t][C4] + c5 * local_tIn_t[t][C5] + c6 * local_tIn_t[t][C6] + c7 * local_tIn_t[t][C7] + c8 * local_tIn_t[t][C8];
                    }
                }

                t = DIMT - 1;
                offset_y_block_t = (y - left_y_compute_t[t]) % blockSize_y;
                N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y) % blockSize_y;
                C_y_block_t = (offset_y_block_t - R + blockSize_y) % blockSize_y;
                S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y) % blockSize_y;
                aligned = (R * (t + 1) + 2) % 4;
                offset = (aligned == 0 ? 3 : aligned - 1);
                for (global_x = left_x_load + R * (t + 1) - offset, x = R * (t + 1) - offset; global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
                {
                    C2 = N1_y_block_t * blockSize_x + x;
                    C7 = C2 - 1;
                    C5 = C2 + 1;
                    C0 = C_y_block_t * blockSize_x + x;
                    C1 = C0 - 1;
                    C3 = C0 + 1;
                    C4 = S1_y_block_t * blockSize_x + x;
                    C6 = C4 - 1;
                    C8 = C4 + 1;
                    vload_3p_1(C2, v_C7, v_C2, v_C5, local_tIn_t[t]);
                    vload_3p_1(C0, v_C1, v_C0, v_C3, local_tIn_t[t]);
                    vload_3p_1(C4, v_C6, v_C4, v_C8, local_tIn_t[t]);
                    v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8;
                    simd_storeu(v_res, &tOut_ptr[x]);
                    // tOut_ptr[x] = c0 * local_tIn_t[t][C0] + c1 * local_tIn_t[t][C1] + c2 * local_tIn_t[t][C2] + c3 * local_tIn_t[t][C3] + c4 * local_tIn_t[t][C4] + c5 * local_tIn_t[t][C5] + c6 * local_tIn_t[t][C6] + c7 * local_tIn_t[t][C7] + c8 * local_tIn_t[t][C8];
                }
            }

            while (DMA_push != 1)
                ;

            DEFINED_DATATYPE *t;
            t = tOut_ptr;
            tOut_ptr = tOut_push_ptr;
            tOut_push_ptr = t;

            DMA_push = 0;
            C_global = (y - R * DIMT) * nx + left_x_load + R * DIMT;
            athread_put(PE_MODE, tOut_push_ptr + R * DIMT, &global_tOut[C_global], (load_size - 2 * R * DIMT) * sizeof(DEFINED_DATATYPE), &DMA_push, 0, 0);

            while (DMA_preload != wait_num)
                ;
        }
        while (DMA_push != 1)
            ;
        athread_syn(ARRAY_SCOPE, 0xffff);

        ldm_free(local_tIn_0, blockSize_y_0 * blockSize_x * sizeof(DEFINED_DATATYPE));
        if (DIMT >= 2)
        {
            ldm_free(local_tIn, (DIMT - 1) * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
            ldm_free(local_tIn_t, DIMT * sizeof(DEFINED_DATATYPE *));
        }
        ldm_free(local_tOut_1, blockSize_x * sizeof(DEFINED_DATATYPE));
        ldm_free(local_tOut_2, blockSize_x * sizeof(DEFINED_DATATYPE));
    }
    spe_param->out = out;
}
