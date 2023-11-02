#include "common.h"
#include <slave.h>

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
    int offset_y_block_t, N1_y_block_t, N2_y_block_t, C_y_block_t, S1_y_block_t, S2_y_block_t, offset_y_block_t_next;
    int C12_next, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24;
    int C_global, C_local;
    volatile int DMA_reply, DMA_push;
    volatile int DMA_preload, wait_num;
    DEFINED_DATATYPE *tOut_ptr, *tOut_push_ptr;
    int i;

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
            local_tIn = (DEFINED_DATATYPE *)ldm_malloc((DIMT - 1) * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
            local_tIn_t = (DEFINED_DATATYPE **)ldm_malloc(DIMT * sizeof(DEFINED_DATATYPE *));
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
            N2_y_block_t = (offset_y_block_t - R - 2 + blockSize_y_0) % blockSize_y_0;
            N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y_0) % blockSize_y_0;
            C_y_block_t = (offset_y_block_t - R + blockSize_y_0) % blockSize_y_0;
            S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y_0) % blockSize_y_0;
            S2_y_block_t = (offset_y_block_t - R + 2 + blockSize_y_0) % blockSize_y_0;
            offset_y_block_t_next = (y - left_y_compute_t[t + 1]) % blockSize_y;
            for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
            {
                C12_next = offset_y_block_t_next * blockSize_x + x;
                C2 = N2_y_block_t * blockSize_x + x;
                C0 = C2 - 2;
                C1 = C2 - 1;
                C3 = C2 + 1;
                C4 = C2 + 2;
                C7 = N1_y_block_t * blockSize_x + x;
                C5 = C7 - 2;
                C6 = C7 - 1;
                C8 = C7 + 1;
                C9 = C7 + 2;
                C12 = C_y_block_t * blockSize_x + x;
                C10 = C12 - 2;
                C11 = C12 - 1;
                C13 = C12 + 1;
                C14 = C12 + 2;
                C17 = S1_y_block_t * blockSize_x + x;
                C15 = C17 - 2;
                C16 = C17 - 1;
                C18 = C17 + 1;
                C19 = C17 + 2;
                C22 = S2_y_block_t * blockSize_x + x;
                C20 = C22 - 2;
                C21 = C22 - 1;
                C23 = C22 + 1;
                C24 = C22 + 2;
                local_tIn_t[t + 1][C12_next] = 2 * local_tIn_0[C0] + 4 * local_tIn_0[C1] + 5 * local_tIn_0[C2] + 4 * local_tIn_0[C3] + 2 * local_tIn_0[C4] + 4 * local_tIn_0[C5] + 9 * local_tIn_0[C6] + 12 * local_tIn_0[C7] + 9 * local_tIn_0[C8] + 4 * local_tIn_0[C9] + 5 * local_tIn_0[C10] + 12 * local_tIn_0[C11] + 15 * local_tIn_0[C12] + 12 * local_tIn_0[C13] + 5 * local_tIn_0[C14] + 4 * local_tIn_0[C15] + 9 * local_tIn_0[C16] + 12 * local_tIn_0[C17] + 9 * local_tIn_0[C18] + 4 * local_tIn_0[C19] + 2 * local_tIn_0[C20] + 4 * local_tIn_0[C21] + 5 * local_tIn_0[C22] + 4 * local_tIn_0[C23] + 2 * local_tIn_0[C24];
            }
            for (t = 1; t < DIMT; t++)
            {
                offset_y_block_t = (y - left_y_compute_t[t] + blockSize_y) % blockSize_y;
                if (y < left_y_compute_t[t + 1])
                    continue;
                N2_y_block_t = (offset_y_block_t - R - 2 + blockSize_y) % blockSize_y;
                N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y) % blockSize_y;
                C_y_block_t = (offset_y_block_t - R + blockSize_y) % blockSize_y;
                S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y) % blockSize_y;
                S2_y_block_t = (offset_y_block_t - R + 2 + blockSize_y) % blockSize_y;
                offset_y_block_t_next = (y - left_y_compute_t[t + 1]) % blockSize_y;
                for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                {
                    C12_next = offset_y_block_t_next * blockSize_x + x;
                    C2 = N2_y_block_t * blockSize_x + x;
                    C0 = C2 - 2;
                    C1 = C2 - 1;
                    C3 = C2 + 1;
                    C4 = C2 + 2;
                    C7 = N1_y_block_t * blockSize_x + x;
                    C5 = C7 - 2;
                    C6 = C7 - 1;
                    C8 = C7 + 1;
                    C9 = C7 + 2;
                    C12 = C_y_block_t * blockSize_x + x;
                    C10 = C12 - 2;
                    C11 = C12 - 1;
                    C13 = C12 + 1;
                    C14 = C12 + 2;
                    C17 = S1_y_block_t * blockSize_x + x;
                    C15 = C17 - 2;
                    C16 = C17 - 1;
                    C18 = C17 + 1;
                    C19 = C17 + 2;
                    C22 = S2_y_block_t * blockSize_x + x;
                    C20 = C22 - 2;
                    C21 = C22 - 1;
                    C23 = C22 + 1;
                    C24 = C22 + 2;
                    local_tIn_t[t + 1][C12_next] = 2 * local_tIn_t[t][C0] + 4 * local_tIn_t[t][C1] + 5 * local_tIn_t[t][C2] + 4 * local_tIn_t[t][C3] + 2 * local_tIn_t[t][C4] + 4 * local_tIn_t[t][C5] + 9 * local_tIn_t[t][C6] + 12 * local_tIn_t[t][C7] + 9 * local_tIn_t[t][C8] + 4 * local_tIn_t[t][C9] + 5 * local_tIn_t[t][C10] + 12 * local_tIn_t[t][C11] + 15 * local_tIn_t[t][C12] + 12 * local_tIn_t[t][C13] + 5 * local_tIn_t[t][C14] + 4 * local_tIn_t[t][C15] + 9 * local_tIn_t[t][C16] + 12 * local_tIn_t[t][C17] + 9 * local_tIn_t[t][C18] + 4 * local_tIn_t[t][C19] + 2 * local_tIn_t[t][C20] + 4 * local_tIn_t[t][C21] + 5 * local_tIn_t[t][C22] + 4 * local_tIn_t[t][C23] + 2 * local_tIn_t[t][C24];
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
            N2_y_block_t = (offset_y_block_t - R - 2 + blockSize_y_0) % blockSize_y_0;
            N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y_0) % blockSize_y_0;
            C_y_block_t = (offset_y_block_t - R + blockSize_y_0) % blockSize_y_0;
            S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y_0) % blockSize_y_0;
            S2_y_block_t = (offset_y_block_t - R + 2 + blockSize_y_0) % blockSize_y_0;
            offset_y_block_t_next = (y - left_y_compute_t[t + 1]) % blockSize_y;
            if (DIMT >= 2)
            {
                for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                {
                    C12_next = offset_y_block_t_next * blockSize_x + x;
                    C2 = N2_y_block_t * blockSize_x + x;
                    C0 = C2 - 2;
                    C1 = C2 - 1;
                    C3 = C2 + 1;
                    C4 = C2 + 2;
                    C7 = N1_y_block_t * blockSize_x + x;
                    C5 = C7 - 2;
                    C6 = C7 - 1;
                    C8 = C7 + 1;
                    C9 = C7 + 2;
                    C12 = C_y_block_t * blockSize_x + x;
                    C10 = C12 - 2;
                    C11 = C12 - 1;
                    C13 = C12 + 1;
                    C14 = C12 + 2;
                    C17 = S1_y_block_t * blockSize_x + x;
                    C15 = C17 - 2;
                    C16 = C17 - 1;
                    C18 = C17 + 1;
                    C19 = C17 + 2;
                    C22 = S2_y_block_t * blockSize_x + x;
                    C20 = C22 - 2;
                    C21 = C22 - 1;
                    C23 = C22 + 1;
                    C24 = C22 + 2;
                    local_tIn_t[t + 1][C12_next] = 2 * local_tIn_0[C0] + 4 * local_tIn_0[C1] + 5 * local_tIn_0[C2] + 4 * local_tIn_0[C3] + 2 * local_tIn_0[C4] + 4 * local_tIn_0[C5] + 9 * local_tIn_0[C6] + 12 * local_tIn_0[C7] + 9 * local_tIn_0[C8] + 4 * local_tIn_0[C9] + 5 * local_tIn_0[C10] + 12 * local_tIn_0[C11] + 15 * local_tIn_0[C12] + 12 * local_tIn_0[C13] + 5 * local_tIn_0[C14] + 4 * local_tIn_0[C15] + 9 * local_tIn_0[C16] + 12 * local_tIn_0[C17] + 9 * local_tIn_0[C18] + 4 * local_tIn_0[C19] + 2 * local_tIn_0[C20] + 4 * local_tIn_0[C21] + 5 * local_tIn_0[C22] + 4 * local_tIn_0[C23] + 2 * local_tIn_0[C24];
                }
            }
            else
            {
                for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                {
                    C2 = N2_y_block_t * blockSize_x + x;
                    C0 = C2 - 2;
                    C1 = C2 - 1;
                    C3 = C2 + 1;
                    C4 = C2 + 2;
                    C7 = N1_y_block_t * blockSize_x + x;
                    C5 = C7 - 2;
                    C6 = C7 - 1;
                    C8 = C7 + 1;
                    C9 = C7 + 2;
                    C12 = C_y_block_t * blockSize_x + x;
                    C10 = C12 - 2;
                    C11 = C12 - 1;
                    C13 = C12 + 1;
                    C14 = C12 + 2;
                    C17 = S1_y_block_t * blockSize_x + x;
                    C15 = C17 - 2;
                    C16 = C17 - 1;
                    C18 = C17 + 1;
                    C19 = C17 + 2;
                    C22 = S2_y_block_t * blockSize_x + x;
                    C20 = C22 - 2;
                    C21 = C22 - 1;
                    C23 = C22 + 1;
                    C24 = C22 + 2;
                    tOut_ptr[x] = 2 * local_tIn_0[C0] + 4 * local_tIn_0[C1] + 5 * local_tIn_0[C2] + 4 * local_tIn_0[C3] + 2 * local_tIn_0[C4] + 4 * local_tIn_0[C5] + 9 * local_tIn_0[C6] + 12 * local_tIn_0[C7] + 9 * local_tIn_0[C8] + 4 * local_tIn_0[C9] + 5 * local_tIn_0[C10] + 12 * local_tIn_0[C11] + 15 * local_tIn_0[C12] + 12 * local_tIn_0[C13] + 5 * local_tIn_0[C14] + 4 * local_tIn_0[C15] + 9 * local_tIn_0[C16] + 12 * local_tIn_0[C17] + 9 * local_tIn_0[C18] + 4 * local_tIn_0[C19] + 2 * local_tIn_0[C20] + 4 * local_tIn_0[C21] + 5 * local_tIn_0[C22] + 4 * local_tIn_0[C23] + 2 * local_tIn_0[C24];
                }
            }

            if (DIMT >= 2)
            {
                for (t = 1; t < DIMT - 1; t++)
                {
                    offset_y_block_t = (y - left_y_compute_t[t]) % blockSize_y;
                    N2_y_block_t = (offset_y_block_t - R - 2 + blockSize_y) % blockSize_y;
                    N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y) % blockSize_y;
                    C_y_block_t = (offset_y_block_t - R + blockSize_y) % blockSize_y;
                    S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y) % blockSize_y;
                    S2_y_block_t = (offset_y_block_t - R + 2 + blockSize_y) % blockSize_y;
                    offset_y_block_t_next = (y - left_y_compute_t[t + 1]) % blockSize_y;
                    for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                    {
                        C12_next = offset_y_block_t_next * blockSize_x + x;
                        C2 = N2_y_block_t * blockSize_x + x;
                        C0 = C2 - 2;
                        C1 = C2 - 1;
                        C3 = C2 + 1;
                        C4 = C2 + 2;
                        C7 = N1_y_block_t * blockSize_x + x;
                        C5 = C7 - 2;
                        C6 = C7 - 1;
                        C8 = C7 + 1;
                        C9 = C7 + 2;
                        C12 = C_y_block_t * blockSize_x + x;
                        C10 = C12 - 2;
                        C11 = C12 - 1;
                        C13 = C12 + 1;
                        C14 = C12 + 2;
                        C17 = S1_y_block_t * blockSize_x + x;
                        C15 = C17 - 2;
                        C16 = C17 - 1;
                        C18 = C17 + 1;
                        C19 = C17 + 2;
                        C22 = S2_y_block_t * blockSize_x + x;
                        C20 = C22 - 2;
                        C21 = C22 - 1;
                        C23 = C22 + 1;
                        C24 = C22 + 2;
                        local_tIn_t[t + 1][C12_next] = 2 * local_tIn_t[t][C0] + 4 * local_tIn_t[t][C1] + 5 * local_tIn_t[t][C2] + 4 * local_tIn_t[t][C3] + 2 * local_tIn_t[t][C4] + 4 * local_tIn_t[t][C5] + 9 * local_tIn_t[t][C6] + 12 * local_tIn_t[t][C7] + 9 * local_tIn_t[t][C8] + 4 * local_tIn_t[t][C9] + 5 * local_tIn_t[t][C10] + 12 * local_tIn_t[t][C11] + 15 * local_tIn_t[t][C12] + 12 * local_tIn_t[t][C13] + 5 * local_tIn_t[t][C14] + 4 * local_tIn_t[t][C15] + 9 * local_tIn_t[t][C16] + 12 * local_tIn_t[t][C17] + 9 * local_tIn_t[t][C18] + 4 * local_tIn_t[t][C19] + 2 * local_tIn_t[t][C20] + 4 * local_tIn_t[t][C21] + 5 * local_tIn_t[t][C22] + 4 * local_tIn_t[t][C23] + 2 * local_tIn_t[t][C24];
                    }
                }

                t = DIMT - 1;
                offset_y_block_t = (y - left_y_compute_t[t]) % blockSize_y;
                N2_y_block_t = (offset_y_block_t - R - 2 + blockSize_y) % blockSize_y;
                N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y) % blockSize_y;
                C_y_block_t = (offset_y_block_t - R + blockSize_y) % blockSize_y;
                S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y) % blockSize_y;
                S2_y_block_t = (offset_y_block_t - R + 2 + blockSize_y) % blockSize_y;
                for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                {
                    C2 = N2_y_block_t * blockSize_x + x;
                    C0 = C2 - 2;
                    C1 = C2 - 1;
                    C3 = C2 + 1;
                    C4 = C2 + 2;
                    C7 = N1_y_block_t * blockSize_x + x;
                    C5 = C7 - 2;
                    C6 = C7 - 1;
                    C8 = C7 + 1;
                    C9 = C7 + 2;
                    C12 = C_y_block_t * blockSize_x + x;
                    C10 = C12 - 2;
                    C11 = C12 - 1;
                    C13 = C12 + 1;
                    C14 = C12 + 2;
                    C17 = S1_y_block_t * blockSize_x + x;
                    C15 = C17 - 2;
                    C16 = C17 - 1;
                    C18 = C17 + 1;
                    C19 = C17 + 2;
                    C22 = S2_y_block_t * blockSize_x + x;
                    C20 = C22 - 2;
                    C21 = C22 - 1;
                    C23 = C22 + 1;
                    C24 = C22 + 2;
                    tOut_ptr[x] = 2 * local_tIn_t[t][C0] + 4 * local_tIn_t[t][C1] + 5 * local_tIn_t[t][C2] + 4 * local_tIn_t[t][C3] + 2 * local_tIn_t[t][C4] + 4 * local_tIn_t[t][C5] + 9 * local_tIn_t[t][C6] + 12 * local_tIn_t[t][C7] + 9 * local_tIn_t[t][C8] + 4 * local_tIn_t[t][C9] + 5 * local_tIn_t[t][C10] + 12 * local_tIn_t[t][C11] + 15 * local_tIn_t[t][C12] + 12 * local_tIn_t[t][C13] + 5 * local_tIn_t[t][C14] + 4 * local_tIn_t[t][C15] + 9 * local_tIn_t[t][C16] + 12 * local_tIn_t[t][C17] + 9 * local_tIn_t[t][C18] + 4 * local_tIn_t[t][C19] + 2 * local_tIn_t[t][C20] + 4 * local_tIn_t[t][C21] + 5 * local_tIn_t[t][C22] + 4 * local_tIn_t[t][C23] + 2 * local_tIn_t[t][C24];
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
