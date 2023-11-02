#include "common.h"
#include <slave.h>

/*
double a[M][N][P];
double b[M][N][P];
double c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26;

for (long k = 1; k < M - 1; ++k)
{
    for (long j = 1; j < N - 1; ++j)
    {
        for (long i = 1; i < P - 1; ++i)
        {
            b[k][j][i] = c0 * a[k][j][i] + c1 * a[k - 1][j - 1][i - 1] + c2 * a[k][j - 1][i - 1]
                       + c3 * a[k + 1][j - 1][i - 1] + c4 * a[k - 1][j][i - 1] + c5 * a[k][j][i - 1]
                       + c6 * a[k + 1][j][i - 1] + c7 * a[k - 1][j + 1][i - 1] + c8 * a[k][j + 1][i - 1]
                       + c9 * a[k + 1][j + 1][i - 1] + c10 * a[k - 1][j - 1][i] + c11 * a[k][j - 1][i]
                       + c12 * a[k + 1][j - 1][i] + c13 * a[k - 1][j][i] + c14 * a[k + 1][j][i]
                       + c15 * a[k - 1][j + 1][i] + c16 * a[k][j + 1][i] + c17 * a[k + 1][j + 1][i]
                       + c18 * a[k - 1][j - 1][i + 1] + c19 * a[k][j - 1][i + 1] + c20 * a[k + 1][j - 1][i + 1]
                       + c21 * a[k - 1][j][i + 1] + c22 * a[k][j][i + 1] + c23 * a[k + 1][j][i + 1]
                       + c24 * a[k - 1][j + 1][i + 1] + c25 * a[k][j + 1][i + 1] + c26 * a[k + 1][j + 1][i + 1];
        }
    }
}
*/

extern DEFINED_DATATYPE c[NUMPOINTS];
// __thread_local DEFINED_DATATYPE local_tIn[(DIMZ + 2 * R) * (DIMY + 2 * R) * (DIMX + 2 * R)];
// __thread_local DEFINED_DATATYPE local_tOut[DIMX];

void spe_func(struct spe_parameter *spe_param)
{
    int DIMT, DIMX, DIMY, DIMZ;
    DEFINED_DATATYPE **temp;
    DEFINED_DATATYPE *local_tIn, *local_tOut;
    int blockNum_z, blockNum_y, blockNum_x, blockNum_z_group;
    int odd_block_x, odd_block_y, odd_block_z;
    int DIMX_padding, DIMY_padding, DIMZ_padding;
    int blockSize_x, blockSize_y, blockSize_z, blockSize_yx;
    int t, g, z, y, x;
    int nt, ng, nz, ny, nx;
    int in = 1, out = 0, tmp;
    DEFINED_DATATYPE *global_tIn, *global_tOut;
    int blockID_z, blockID_y, blockID_x;
    int DIMX_final, DIMY_final, DIMZ_final;
    int left_z_block, right_z_block, left_y_block, right_y_block, left_x_block, right_x_block;
    int left_z_load, right_z_load, left_y_load, right_y_load, left_x_load, right_x_load, load_size, offset_x_block;
    int left_z_compute, left_y_compute, left_x_compute;
    int tz, ty, tx;
    int C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25, C26;
    DEFINED_DATATYPE c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7], c8 = c[8], c9 = c[9],
                     c10 = c[10], c11 = c[11], c12 = c[12], c13 = c[13], c14 = c[14], c15 = c[15], c16 = c[16], c17 = c[17], c18 = c[18],
                     c19 = c[19], c20 = c[20], c21 = c[21], c22 = c[22], c23 = c[23], c24 = c[24], c25 = c[25], c26 = c[26];
    int C_global, C_local;
    volatile int DMA_reply, DMA_push;

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
    ng = blockNum_z * blockNum_y * blockNum_x / MAX_THREADS;
    blockNum_z_group = blockNum_z / ng;

    odd_block_x = NUMCOLS % DIMX % blockNum_x;
    odd_block_y = NUMROWS % DIMY % blockNum_y;
    odd_block_z = LAYERS % DIMZ % blockNum_z;

    DIMX_padding = DIMX + (NUMCOLS % DIMX) / blockNum_x;
    DIMY_padding = DIMY + (NUMROWS % DIMY) / blockNum_y;
    DIMZ_padding = DIMZ + (LAYERS % DIMZ) / blockNum_z;

    blockSize_x = DIMX_padding + 1 + 2 * R;
    blockSize_y = DIMY_padding + 1 + 2 * R;
    blockSize_z = DIMZ_padding + 1 + 2 * R;
    blockSize_yx = blockSize_y * blockSize_x;

    local_tIn = (DEFINED_DATATYPE *)ldm_malloc(blockSize_x * blockSize_y * blockSize_z * sizeof(DEFINED_DATATYPE));
    local_tOut = (DEFINED_DATATYPE *)ldm_malloc((DIMX_padding + 1) * sizeof(DEFINED_DATATYPE));

    for (t = 1; t <= nt; t++)
    {
        tmp = in;
        in = out;
        out = tmp;
        global_tIn = temp[in];
        global_tOut = temp[out];
        for (g = 0; g < ng; g++)
        {
            blockID_z = _MYID / (blockNum_y * blockNum_x) + g * blockNum_z_group;
            blockID_y = _MYID % (blockNum_y * blockNum_x) / blockNum_x;
            blockID_x = _MYID % blockNum_x;

            DIMX_final = DIMX_padding, DIMY_final = DIMY_padding, DIMZ_final = DIMZ_padding;
            if (blockID_z < odd_block_z)
            {
                DIMZ_final += 1;
                left_z_block = blockID_z * (DIMZ_padding + 1) - R;
                right_z_block = (blockID_z + 1) * (DIMZ_padding + 1) + R;
            }
            else
            {
                left_z_block = odd_block_z * (DIMZ_padding + 1) + (blockID_z - odd_block_z) * DIMZ_padding - R;
                right_z_block = odd_block_z * (DIMZ_padding + 1) + (blockID_z - odd_block_z + 1) * DIMZ_padding + R;
            }
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

            left_z_load = left_z_block >= 0 ? left_z_block : 0;
            right_z_load = right_z_block <= nz ? right_z_block : nz;
            left_y_load = left_y_block >= 0 ? left_y_block : 0;
            right_y_load = right_y_block <= ny ? right_y_block : ny;
            left_x_load = left_x_block >= 0 ? left_x_block : 0;
            right_x_load = right_x_block <= nx ? right_x_block : nx;
            load_size = right_x_load - left_x_load;

            offset_x_block = left_x_block >= 0 ? 0 : R;

            for (z = left_z_block; z < right_z_block; z++)
            {
                if (z < left_z_load || z >= right_z_load)
                    continue;
                for (y = left_y_block; y < right_y_block; y++)
                {
                    if (y < left_y_load || y >= right_y_load)
                        continue;
                    DMA_reply = 0;
                    C_global = z * ny * nx + y * nx + left_x_load;
                    C_local = (z - left_z_block) * blockSize_y * blockSize_x + (y - left_y_block) * blockSize_x + offset_x_block;
                    athread_get(PE_MODE, &global_tIn[C_global], &local_tIn[C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_reply, 0, 0, 0);
                    while (DMA_reply != 1)
                        ;
                }
            }

            athread_syn(ARRAY_SCOPE, 0xffff);

            left_z_compute = left_z_block + R;
            left_y_compute = left_y_block + R;
            left_x_compute = left_x_block + R;

            for (z = 0; z < DIMZ_final; z++)
            {
                if (left_z_compute + z < 0 + R * t || left_z_compute + z >= nz - R * t)
                    continue;
                for (y = 0; y < DIMY_final; y++)
                {
                    if (left_y_compute + y < 0 + R * t || left_y_compute + y >= ny - R * t)
                        continue;
                    for (x = 0; x < DIMX_final; x++)
                    {
                        if (left_x_compute + x < 0 + R * t || left_x_compute + x >= nx - R * t)
                            continue;
                        tz = z + R;
                        ty = y + R;
                        tx = x + R;
                        C0 = tz * blockSize_yx + ty * blockSize_x + tx;
                        C1 = C0 - blockSize_yx - blockSize_x - 1;
                        C2 = C1 + blockSize_yx;
                        C3 = C2 + blockSize_yx;
                        C4 = C0 - blockSize_yx - 1;
                        C5 = C4 + blockSize_yx;
                        C6 = C5 + blockSize_yx;
                        C7 = C0 - blockSize_yx + blockSize_x - 1;
                        C8 = C7 + blockSize_yx;
                        C9 = C8 + blockSize_yx;
                        C10 = C1 + 1;
                        C11 = C2 + 1;
                        C12 = C3 + 1;
                        C13 = C4 + 1;
                        C14 = C6 + 1;
                        C15 = C7 + 1;
                        C16 = C8 + 1;
                        C17 = C9 + 1;
                        C18 = C10 + 1;
                        C19 = C11 + 1;
                        C20 = C12 + 1;
                        C21 = C13 + 1;
                        C22 = C0 + 1;
                        C23 = C14 + 1;
                        C24 = C15 + 1;
                        C25 = C16 + 1;
                        C26 = C17 + 1;
                        local_tOut[x] = c0 * local_tIn[C0] + c1 * local_tIn[C1] + c2 * local_tIn[C2] + c3 * local_tIn[C3] + c4 * local_tIn[C4] + c5 * local_tIn[C5] + c6 * local_tIn[C6] + c7 * local_tIn[C7] + c8 * local_tIn[C8] + c9 * local_tIn[C9] + c10 * local_tIn[C10] + c11 * local_tIn[C11] + c12 * local_tIn[C12] + c13 * local_tIn[C13] + c14 * local_tIn[C14] + c15 * local_tIn[C15] + c16 * local_tIn[C16] + c17 * local_tIn[C17] + c18 * local_tIn[C18] + c19 * local_tIn[C19] + c20 * local_tIn[C20] + c21 * local_tIn[C21] + c22 * local_tIn[C22] + c23 * local_tIn[C23] + c24 * local_tIn[C24] + c25 * local_tIn[C25] + c26 * local_tIn[C26];
                    }
                    DMA_push = 0;
                    C_global = (left_z_compute + z) * ny * nx + (left_y_compute + y) * nx + left_x_compute;
                    athread_put(PE_MODE, local_tOut, &global_tOut[C_global], DIMX_final * sizeof(DEFINED_DATATYPE), &DMA_push, 0, 0);
                    while (DMA_push != 1)
                        ;
                }
            }
            athread_syn(ARRAY_SCOPE, 0xffff);
        }
    }
    spe_param->out = out;
}
