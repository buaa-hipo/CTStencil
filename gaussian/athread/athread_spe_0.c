#include "common.h"
#include <slave.h>

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
    int C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24;
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

            blockSize_x = DIMX_final + 2 * R;
            blockSize_y = DIMY_final + 2 * R;

            local_tIn = (DEFINED_DATATYPE *)ldm_malloc(blockSize_x * blockSize_y * sizeof(DEFINED_DATATYPE));
            local_tOut = (DEFINED_DATATYPE *)ldm_malloc(DIMX_final * sizeof(DEFINED_DATATYPE));

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
                    C2 = (ty - 2) * blockSize_x + tx;
                    C0 = C2 - 2;
                    C1 = C2 - 1;
                    C3 = C2 + 1;
                    C4 = C2 + 2;
                    C7 = (ty - 1) * blockSize_x + tx;
                    C5 = C7 - 2;
                    C6 = C7 - 1;
                    C8 = C7 + 1;
                    C9 = C7 + 2;
                    C12 = ty * blockSize_x + tx;
                    C10 = C12 - 2;
                    C11 = C12 - 1;
                    C13 = C12 + 1;
                    C14 = C12 + 2;
                    C17 = (ty + 1) * blockSize_x + tx;
                    C15 = C17 - 2;
                    C16 = C17 - 1;
                    C18 = C17 + 1;
                    C19 = C17 + 2;
                    C22 = (ty + 2) * blockSize_x + tx;
                    C20 = C22 - 2;
                    C21 = C22 - 1;
                    C23 = C22 + 1;
                    C24 = C22 + 2;
                    local_tOut[x] = 2 * local_tIn[C0] + 4 * local_tIn[C1] + 5 * local_tIn[C2] + 4 * local_tIn[C3] + 2 * local_tIn[C4] + 4 * local_tIn[C5] + 9 * local_tIn[C6] + 12 * local_tIn[C7] + 9 * local_tIn[C8] + 4 * local_tIn[C9] + 5 * local_tIn[C10] + 12 * local_tIn[C11] + 15 * local_tIn[C12] + 12 * local_tIn[C13] + 5 * local_tIn[C14] + 4 * local_tIn[C15] + 9 * local_tIn[C16] + 12 * local_tIn[C17] + 9 * local_tIn[C18] + 4 * local_tIn[C19] + 2 * local_tIn[C20] + 4 * local_tIn[C21] + 5 * local_tIn[C22] + 4 * local_tIn[C23] + 2 * local_tIn[C24];
                }
                DMA_push = 0;
                C_global = (left_y_compute + y) * nx + left_x_compute;
                athread_put(PE_MODE, local_tOut, &temp[out][C_global], DIMX_final * sizeof(DEFINED_DATATYPE), &DMA_push, 0, 0);
                while (DMA_push != 1)
                    ;
            }
            athread_syn(ARRAY_SCOPE, 0xffff);

            ldm_free(local_tIn, blockSize_x * blockSize_y * sizeof(DEFINED_DATATYPE));
            ldm_free(local_tOut, DIMX_final * sizeof(DEFINED_DATATYPE));
        }
    }
    spe_param->out = out;
}
