#include "common.h"
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

extern DEFINED_DATATYPE c[NUMPOINTS];
// tIn, 0, 1, 2 ... DIMT - 1, DIMT(local_out), tOut
// __thread_local DEFINED_DATATYPE local_tIn_t[DIMT][(1 + 2 * R) * (DIMX + 2 * R * DIMT)];
// __thread_local DEFINED_DATATYPE local_tOut[DIMX + 2 * R * DIMT];

void spe_func(struct spe_parameter *spe_param)
{
    int DIMT, DIMX, DIMY;
    DEFINED_DATATYPE **temp;
    DEFINED_DATATYPE *local_tIn, *local_tOut, **local_tIn_t;
    int blockNum_x, blockNum_y;
    int odd_block_x, odd_block_y;
    int DIMX_padding, DIMY_padding, blockSize_x, blockSize_y, DIMX_final, DIMY_final;
    int t, y, global_x, x;
    int nt, nx, ny;
    int in = 1, out = 0;
    int blockID_y, blockID_x;
    int left_y_block, right_y_block, left_x_block, right_x_block;
    int left_y_load, right_y_load, left_x_load, right_x_load, load_size, offset_x_block;
    int left_y_compute_t[DIMT + 1];
    DEFINED_DATATYPE c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7], c8 = c[8];
    int offset_y_block_t, N1_y_block_t, C_y_block_t, S1_y_block_t, offset_y_block_t_next;
    int C0_next, C0, C1, C2, C3, C4, C5, C6, C7, C8;
    int C_global, C_local;
    volatile int DMA_reply, DMA_push;
    int i;
    int ng, blockNum_y_group, g;

    temp = spe_param->temp;
    nt = spe_param->nt;
    nx = spe_param->nx;
    ny = spe_param->ny;
    DIMT = spe_param->DIMT;
    DIMX = spe_param->DIMX;
    DIMY = spe_param->DIMY;

    blockNum_x = NUMCOLS / DIMX;
    blockNum_y = NUMROWS / DIMY;
    ng = blockNum_x * blockNum_y / MAX_THREADS;
    blockNum_y_group = blockNum_y / ng;

    DIMX_padding = DIMX + (NUMCOLS % DIMX) / blockNum_x;
    DIMY_padding = DIMY + (NUMROWS % DIMY) / blockNum_y;

    odd_block_x = NUMCOLS % DIMX % blockNum_x;
    odd_block_y = NUMROWS % DIMY % blockNum_y;

    blockSize_x = DIMX_padding + 2 * R * DIMT;
    blockSize_y = 1 + 2 * R;

    for (i = 0; i < NUMTIMESTEPS / DIMT; i++)
    {
        int tmp = in;
        in = out;
        out = tmp;

        local_tIn = (DEFINED_DATATYPE *)ldm_malloc(DIMT * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
        local_tIn_t = (DEFINED_DATATYPE **)ldm_malloc(DIMT * sizeof(DEFINED_DATATYPE *));
        for (t = 0; t < DIMT; t++)
            local_tIn_t[t] = &local_tIn[t * blockSize_y * blockSize_x];
        local_tOut = (DEFINED_DATATYPE *)ldm_malloc(blockSize_x * sizeof(DEFINED_DATATYPE));

        for (g = 0; g < ng; g++)
        {
            blockID_x = _MYID % blockNum_x;
            blockID_y = _MYID / blockNum_x + blockNum_y_group * g;

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

            for (t = 0; t <= DIMT; t++)
                left_y_compute_t[t] = left_y_load + 1 + 2 * R * t - 1;

            for (y = left_y_load; y < left_y_compute_t[DIMT]; y++)
            {
                for (t = 0; t < DIMT; t++)
                {
                    offset_y_block_t = (y - left_y_compute_t[t] + blockSize_y) % blockSize_y;
                    if (t == 0)
                    {
                        DMA_reply = 0;
                        C_global = y * nx + left_x_load;
                        C_local = offset_y_block_t * blockSize_x;
                        athread_get(PE_MODE, &temp[in][C_global], &local_tIn_t[0][C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_reply, 0, 0, 0);
                        while (DMA_reply != 1)
                            ;
                    }
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

            for (y = left_y_compute_t[DIMT]; y < right_y_load; y++)
            {
                int offset_y_block_0 = (y - left_y_compute_t[0]) % blockSize_y;
                DMA_reply = 0;
                C_global = y * nx + left_x_load;
                C_local = offset_y_block_0 * blockSize_x;
                athread_get(PE_MODE, &temp[in][C_global], &local_tIn_t[0][C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_reply, 0, 0, 0);
                while (DMA_reply != 1)
                    ;
                for (t = 0; t < DIMT - 1; t++)
                {
                    offset_y_block_t = (y - left_y_compute_t[t]) % blockSize_y;
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

                t = DIMT - 1;
                offset_y_block_t = (y - left_y_compute_t[t]) % blockSize_y;
                N1_y_block_t = (offset_y_block_t - R - 1 + blockSize_y) % blockSize_y;
                C_y_block_t = (offset_y_block_t - R + blockSize_y) % blockSize_y;
                S1_y_block_t = (offset_y_block_t - R + 1 + blockSize_y) % blockSize_y;
                for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
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
                    local_tOut[x] = c0 * local_tIn_t[t][C0] + c1 * local_tIn_t[t][C1] + c2 * local_tIn_t[t][C2] + c3 * local_tIn_t[t][C3] + c4 * local_tIn_t[t][C4] + c5 * local_tIn_t[t][C5] + c6 * local_tIn_t[t][C6] + c7 * local_tIn_t[t][C7] + c8 * local_tIn_t[t][C8];
                }
                DMA_push = 0;
                C_global = (y - R * DIMT) * nx + left_x_load + R * DIMT;
                athread_put(PE_MODE, &local_tOut[R * DIMT], &temp[out][C_global], (load_size - 2 * R * DIMT) * sizeof(DEFINED_DATATYPE), &DMA_push, 0, 0);
                while (DMA_push != 1)
                    ;
            }
            athread_syn(ARRAY_SCOPE, 0xffff);
        }

        ldm_free(local_tIn, DIMT * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
        ldm_free(local_tIn_t, DIMT * sizeof(DEFINED_DATATYPE *));
        ldm_free(local_tOut, blockSize_x * sizeof(DEFINED_DATATYPE));
    }
    spe_param->out = out;
}
