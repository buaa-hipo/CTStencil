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
// __thread_local DEFINED_DATATYPE local_tIn[DIMY + 2 * R][DIMX + 2 * R];
// __thread_local DEFINED_DATATYPE local_tOut[DIMX + 2 * R];

void spe_func(struct spe_parameter *spe_param)
{
    int DIMT, DIMX, DIMY;
    DEFINED_DATATYPE **temp;
    DEFINED_DATATYPE *local_tIn, *local_tOut;
    int blockNum_x, blockNum_y, blockNum_y_group;
    int odd_block_x, odd_block_y;
    int DIMX_padding, DIMY_padding;
    int blockSize_x, blockSize_y;
    int t, g, x, y;
    int nt, ng, nx, ny;
    int in = 1, out = 0, tmp;
    int blockID_y, blockID_x;
    int DIMX_final, DIMY_final;
    int left_y_block, right_y_block, left_x_block, right_x_block;
    int left_y_load, right_y_load, left_x_load, right_x_load, load_size, offset_x_block;
    int left_y_compute, left_x_compute;
    int ty, tx;
    DEFINED_DATATYPE c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7], c8 = c[8];
    int C0, C1, C2, C3, C4, C5, C6, C7, C8;
    int C_global, C_local;
    volatile int DMA_reply, DMA_push;

    temp = spe_param->temp;
    nt = spe_param->nt;
    nx = spe_param->nx;
    ny = spe_param->ny;
    DIMX = spe_param->DIMX;
    DIMY = spe_param->DIMY;

    blockNum_x = NUMCOLS / DIMX;
    blockNum_y = NUMROWS / DIMY;
    ng = blockNum_x * blockNum_y / MAX_THREADS;
    blockNum_y_group = blockNum_y / ng;

    odd_block_x = NUMCOLS % DIMX % blockNum_x;
    odd_block_y = NUMROWS % DIMY % blockNum_y;

    DIMX_padding = DIMX + (NUMCOLS % DIMX) / blockNum_x;
    DIMY_padding = DIMY + (NUMROWS % DIMY) / blockNum_y;

    blockSize_x = DIMX_padding + 1 + 2 * R;
    blockSize_y = DIMY_padding + 1 + 2 * R;

    local_tIn = (DEFINED_DATATYPE *)ldm_malloc(blockSize_x * blockSize_y * sizeof(DEFINED_DATATYPE));
    local_tOut = (DEFINED_DATATYPE *)ldm_malloc((DIMY_padding + 1) * sizeof(DEFINED_DATATYPE));

    for (t = 1; t <= nt; t++)
    {
        tmp = in;
        in = out;
        out = tmp;
        for (g = 0; g < ng; g++)
        {
            blockID_y = _MYID / blockNum_x + blockNum_y_group * g;
            blockID_x = _MYID % blockNum_x;

            DIMX_final = DIMX_padding, DIMY_final = DIMY_padding;
            if (blockID_y < odd_block_y)
            {
                DIMY_final += 1;
                left_y_block = blockID_y * (DIMY_padding + 1) - R;
                right_y_block = (blockID_y + 1) * (DIMY_padding + 1) + R;
            }
            else
            {
                left_y_block = odd_block_y * (DIMY_padding + 1) + (blockID_y - odd_block_y) * DIMY_padding - R;
                right_y_block = odd_block_y * (DIMY_padding + 1) + (blockID_y - odd_block_y + 1) * DIMY_padding + R;
            }
            if (blockID_x < odd_block_x)
            {
                DIMX_final += 1;
                left_x_block = blockID_x * (DIMX_padding + 1) - R;
                right_x_block = (blockID_x + 1) * (DIMX_padding + 1) + R;
            }
            else
            {
                left_x_block = odd_block_x * (DIMX_padding + 1) + (blockID_x - odd_block_x) * DIMX_padding - R;
                right_x_block = odd_block_x * (DIMX_padding + 1) + (blockID_x - odd_block_x + 1) * DIMX_padding + R;
            }

            left_y_load = left_y_block >= 0 ? left_y_block : 0;
            right_y_load = right_y_block <= ny ? right_y_block : ny;
            left_x_load = left_x_block >= 0 ? left_x_block : 0;
            right_x_load = right_x_block <= nx ? right_x_block : nx;
            load_size = right_x_load - left_x_load;
            offset_x_block = left_x_block >= 0 ? 0 : R;

            for (y = left_y_block; y < right_y_block; y++)
            {
                if (y < left_y_load || y >= right_y_load)
                    continue;
                DMA_reply = 0;
                C_global = y * nx + left_x_load;
                C_local = (y - left_y_block) * blockSize_x + offset_x_block;
                athread_get(PE_MODE, &temp[in][C_global], &local_tIn[C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_reply, 0, 0, 0);
                while (DMA_reply != 1)
                    ;
            }

            athread_syn(ARRAY_SCOPE, 0xffff);

            left_y_compute = left_y_block + R;
            left_x_compute = left_x_block + R;

            for (y = 0; y < DIMY_final; y++)
            {
                if (left_y_compute + y < 0 + R * t || left_y_compute + y >= ny - R * t)
                    continue;
                for (x = 0; x < DIMX_final; x++)
                {
                    if (left_x_compute + x < 0 + R * t || left_x_compute + x >= nx - R * t)
                        continue;
                    ty = y + R;
                    tx = x + R;
                    C0 = ty * blockSize_x + tx;
                    C7 = C0 - blockSize_x - 1;
                    C2 = C0 - blockSize_x;
                    C5 = C0 - blockSize_x + 1;
                    C1 = C0 - 1;
                    C3 = C0 + 1;
                    C6 = C0 + blockSize_x - 1;
                    C4 = C0 + blockSize_x;
                    C8 = C0 + blockSize_x + 1;
                    local_tOut[x] = c0 * local_tIn[C0] + c1 * local_tIn[C1] + c2 * local_tIn[C2] + c3 * local_tIn[C3] + c4 * local_tIn[C4] + c5 * local_tIn[C5] + c6 * local_tIn[C6] + c7 * local_tIn[C7] + c8 * local_tIn[C8];
                }
                DMA_push = 0;
                C_global = (left_y_compute + y) * nx + left_x_compute;
                athread_put(PE_MODE, local_tOut, &temp[out][C_global], DIMX_final * sizeof(DEFINED_DATATYPE), &DMA_push, 0, 0);
                while (DMA_push != 1)
                    ;
            }

            athread_syn(ARRAY_SCOPE, 0xffff);
        }
    }
    spe_param->out = out;
}
