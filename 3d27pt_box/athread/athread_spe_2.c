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
// __thread_local DEFINED_DATATYPE local_tIn_t[DIMT][DIMT * (1 + 2 * R) * (DIMY + 2 * R * NUMTIMESTEPS) * (DIMX + 2 * R * NUMTIMESTEPS)];
// __thread_local DEFINED_DATATYPE local_tOut[DIMX + 2 * R * NUMTIMESTEPS];

void spe_func(struct spe_parameter *spe_param)
{
    int DIMT, DIMX, DIMY, DIMZ;
    DEFINED_DATATYPE c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7], c8 = c[8], c9 = c[9],
                     c10 = c[10], c11 = c[11], c12 = c[12], c13 = c[13], c14 = c[14], c15 = c[15], c16 = c[16], c17 = c[17], c18 = c[18],
                     c19 = c[19], c20 = c[20], c21 = c[21], c22 = c[22], c23 = c[23], c24 = c[24], c25 = c[25], c26 = c[26];
    DEFINED_DATATYPE **temp;
    DEFINED_DATATYPE *local_tIn, *local_tOut, *local_tOut_1, *local_tOut_2, *local_tIn_0, **local_tIn_t;
    int blockNum_z, blockNum_y, blockNum_x;
    int blockID_z, blockID_y, blockID_x;
    int DIMZ_padding, DIMY_padding, DIMX_padding;
    int odd_block_z, odd_block_y, odd_block_x;
    int DIMZ_final, DIMY_final, DIMX_final;
    int left_z_block, right_z_block, left_y_block, right_y_block, left_x_block, right_x_block;
    int left_z_load, right_z_load, left_y_load, right_y_load, left_x_load, right_x_load, load_size;
    int blockSize_x, blockSize_y, blockSize_z, blockSize_z_0, blockSize_yx;
    int left_z_compute_t[DIMT + 1];
    int t, z, global_y, y, global_x, x;
    int nt, nz, ny, nx;
    int in = 1, out = 0;
    DEFINED_DATATYPE *global_tIn, *global_tOut;
    int offset_z_block_t, N_z_block_t, C_z_block_t, S_z_block_t, offset_z_block_t_next;
    int C0_next, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25, C26;
    int C_global, C_local;
    volatile int DMA_reply, DMA_push;
    volatile int DMA_preload, wait_num;
    DEFINED_DATATYPE *tOut_ptr, *tOut_push_ptr;
    int i;

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
    blockSize_y = DIMY_final + 2 * R * DIMT;
    blockSize_z = 1 + 2 * R;
    blockSize_z_0 = blockSize_z + 1; // preload
    blockSize_yx = blockSize_y * blockSize_x;

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
                    C0_next = offset_z_block_t_next * blockSize_yx + y * blockSize_x + x;
                    C13 = N_z_block_t * blockSize_yx + y * blockSize_x + x;
                    C0 = C_z_block_t * blockSize_yx + y * blockSize_x + x;
                    C14 = S_z_block_t * blockSize_yx + y * blockSize_x + x;
                    C1 = C13 - blockSize_x - 1;
                    C10 = C1 + 1;
                    C18 = C10 + 1;
                    C4 = C13 - 1;
                    C21 = C13 + 1;
                    C7 = C13 + blockSize_x - 1;
                    C15 = C7 + 1;
                    C24 = C15 + 1;
                    C2 = C0 - blockSize_x - 1;
                    C11 = C2 + 1;
                    C19 = C11 + 1;
                    C5 = C0 - 1;
                    C22 = C0 + 1;
                    C8 = C0 + blockSize_x - 1;
                    C16 = C8 + 1;
                    C25 = C16 + 1;
                    C3 = C14 - blockSize_x - 1;
                    C12 = C3 + 1;
                    C20 = C12 + 1;
                    C6 = C14 - 1;
                    C23 = C14 + 1;
                    C9 = C14 + blockSize_x - 1;
                    C17 = C9 + 1;
                    C26 = C17 + 1;
                    local_tIn_t[t + 1][C0_next] = c0 * local_tIn_0[C0] + c1 * local_tIn_0[C1] + c2 * local_tIn_0[C2] + c3 * local_tIn_0[C3] + c4 * local_tIn_0[C4] + c5 * local_tIn_0[C5] + c6 * local_tIn_0[C6] + c7 * local_tIn_0[C7] + c8 * local_tIn_0[C8] + c9 * local_tIn_0[C9] + c10 * local_tIn_0[C10] + c11 * local_tIn_0[C11] + c12 * local_tIn_0[C12] + c13 * local_tIn_0[C13] + c14 * local_tIn_0[C14] + c15 * local_tIn_0[C15] + c16 * local_tIn_0[C16] + c17 * local_tIn_0[C17] + c18 * local_tIn_0[C18] + c19 * local_tIn_0[C19] + c20 * local_tIn_0[C20] + c21 * local_tIn_0[C21] + c22 * local_tIn_0[C22] + c23 * local_tIn_0[C23] + c24 * local_tIn_0[C24] + c25 * local_tIn_0[C25] + c26 * local_tIn_0[C26];
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
                        C0_next = offset_z_block_t_next * blockSize_yx + y * blockSize_x + x;
                        C13 = N_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C0 = C_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C14 = S_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C1 = C13 - blockSize_x - 1;
                        C10 = C1 + 1;
                        C18 = C10 + 1;
                        C4 = C13 - 1;
                        C21 = C13 + 1;
                        C7 = C13 + blockSize_x - 1;
                        C15 = C7 + 1;
                        C24 = C15 + 1;
                        C2 = C0 - blockSize_x - 1;
                        C11 = C2 + 1;
                        C19 = C11 + 1;
                        C5 = C0 - 1;
                        C22 = C0 + 1;
                        C8 = C0 + blockSize_x - 1;
                        C16 = C8 + 1;
                        C25 = C16 + 1;
                        C3 = C14 - blockSize_x - 1;
                        C12 = C3 + 1;
                        C20 = C12 + 1;
                        C6 = C14 - 1;
                        C23 = C14 + 1;
                        C9 = C14 + blockSize_x - 1;
                        C17 = C9 + 1;
                        C26 = C17 + 1;
                        local_tIn_t[t + 1][C0_next] = c0 * local_tIn_t[t][C0] + c1 * local_tIn_t[t][C1] + c2 * local_tIn_t[t][C2] + c3 * local_tIn_t[t][C3] + c4 * local_tIn_t[t][C4] + c5 * local_tIn_t[t][C5] + c6 * local_tIn_t[t][C6] + c7 * local_tIn_t[t][C7] + c8 * local_tIn_t[t][C8] + c9 * local_tIn_t[t][C9] + c10 * local_tIn_t[t][C10] + c11 * local_tIn_t[t][C11] + c12 * local_tIn_t[t][C12] + c13 * local_tIn_t[t][C13] + c14 * local_tIn_t[t][C14] + c15 * local_tIn_t[t][C15] + c16 * local_tIn_t[t][C16] + c17 * local_tIn_t[t][C17] + c18 * local_tIn_t[t][C18] + c19 * local_tIn_t[t][C19] + c20 * local_tIn_t[t][C20] + c21 * local_tIn_t[t][C21] + c22 * local_tIn_t[t][C22] + c23 * local_tIn_t[t][C23] + c24 * local_tIn_t[t][C24] + c25 * local_tIn_t[t][C25] + c26 * local_tIn_t[t][C26];
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
            if (DIMT >= 2)
            {
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                    {
                        C0_next = offset_z_block_t_next * blockSize_yx + y * blockSize_x + x;
                        C13 = N_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C0 = C_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C14 = S_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C1 = C13 - blockSize_x - 1;
                        C10 = C1 + 1;
                        C18 = C10 + 1;
                        C4 = C13 - 1;
                        C21 = C13 + 1;
                        C7 = C13 + blockSize_x - 1;
                        C15 = C7 + 1;
                        C24 = C15 + 1;
                        C2 = C0 - blockSize_x - 1;
                        C11 = C2 + 1;
                        C19 = C11 + 1;
                        C5 = C0 - 1;
                        C22 = C0 + 1;
                        C8 = C0 + blockSize_x - 1;
                        C16 = C8 + 1;
                        C25 = C16 + 1;
                        C3 = C14 - blockSize_x - 1;
                        C12 = C3 + 1;
                        C20 = C12 + 1;
                        C6 = C14 - 1;
                        C23 = C14 + 1;
                        C9 = C14 + blockSize_x - 1;
                        C17 = C9 + 1;
                        C26 = C17 + 1;
                        local_tIn_t[t + 1][C0_next] = c0 * local_tIn_0[C0] + c1 * local_tIn_0[C1] + c2 * local_tIn_0[C2] + c3 * local_tIn_0[C3] + c4 * local_tIn_0[C4] + c5 * local_tIn_0[C5] + c6 * local_tIn_0[C6] + c7 * local_tIn_0[C7] + c8 * local_tIn_0[C8] + c9 * local_tIn_0[C9] + c10 * local_tIn_0[C10] + c11 * local_tIn_0[C11] + c12 * local_tIn_0[C12] + c13 * local_tIn_0[C13] + c14 * local_tIn_0[C14] + c15 * local_tIn_0[C15] + c16 * local_tIn_0[C16] + c17 * local_tIn_0[C17] + c18 * local_tIn_0[C18] + c19 * local_tIn_0[C19] + c20 * local_tIn_0[C20] + c21 * local_tIn_0[C21] + c22 * local_tIn_0[C22] + c23 * local_tIn_0[C23] + c24 * local_tIn_0[C24] + c25 * local_tIn_0[C25] + c26 * local_tIn_0[C26];
                    }
                }
            }
            else
            {
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                    {
                        C13 = N_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C0 = C_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C14 = S_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C1 = C13 - blockSize_x - 1;
                        C10 = C1 + 1;
                        C18 = C10 + 1;
                        C4 = C13 - 1;
                        C21 = C13 + 1;
                        C7 = C13 + blockSize_x - 1;
                        C15 = C7 + 1;
                        C24 = C15 + 1;
                        C2 = C0 - blockSize_x - 1;
                        C11 = C2 + 1;
                        C19 = C11 + 1;
                        C5 = C0 - 1;
                        C22 = C0 + 1;
                        C8 = C0 + blockSize_x - 1;
                        C16 = C8 + 1;
                        C25 = C16 + 1;
                        C3 = C14 - blockSize_x - 1;
                        C12 = C3 + 1;
                        C20 = C12 + 1;
                        C6 = C14 - 1;
                        C23 = C14 + 1;
                        C9 = C14 + blockSize_x - 1;
                        C17 = C9 + 1;
                        C26 = C17 + 1;
                        tOut_ptr[x] = c0 * local_tIn_0[C0] + c1 * local_tIn_0[C1] + c2 * local_tIn_0[C2] + c3 * local_tIn_0[C3] + c4 * local_tIn_0[C4] + c5 * local_tIn_0[C5] + c6 * local_tIn_0[C6] + c7 * local_tIn_0[C7] + c8 * local_tIn_0[C8] + c9 * local_tIn_0[C9] + c10 * local_tIn_0[C10] + c11 * local_tIn_0[C11] + c12 * local_tIn_0[C12] + c13 * local_tIn_0[C13] + c14 * local_tIn_0[C14] + c15 * local_tIn_0[C15] + c16 * local_tIn_0[C16] + c17 * local_tIn_0[C17] + c18 * local_tIn_0[C18] + c19 * local_tIn_0[C19] + c20 * local_tIn_0[C20] + c21 * local_tIn_0[C21] + c22 * local_tIn_0[C22] + c23 * local_tIn_0[C23] + c24 * local_tIn_0[C24] + c25 * local_tIn_0[C25] + c26 * local_tIn_0[C26];
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
                    for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                    {
                        for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                        {
                            C0_next = offset_z_block_t_next * blockSize_yx + y * blockSize_x + x;
                            C13 = N_z_block_t * blockSize_yx + y * blockSize_x + x;
                            C0 = C_z_block_t * blockSize_yx + y * blockSize_x + x;
                            C14 = S_z_block_t * blockSize_yx + y * blockSize_x + x;
                            C1 = C13 - blockSize_x - 1;
                            C10 = C1 + 1;
                            C18 = C10 + 1;
                            C4 = C13 - 1;
                            C21 = C13 + 1;
                            C7 = C13 + blockSize_x - 1;
                            C15 = C7 + 1;
                            C24 = C15 + 1;
                            C2 = C0 - blockSize_x - 1;
                            C11 = C2 + 1;
                            C19 = C11 + 1;
                            C5 = C0 - 1;
                            C22 = C0 + 1;
                            C8 = C0 + blockSize_x - 1;
                            C16 = C8 + 1;
                            C25 = C16 + 1;
                            C3 = C14 - blockSize_x - 1;
                            C12 = C3 + 1;
                            C20 = C12 + 1;
                            C6 = C14 - 1;
                            C23 = C14 + 1;
                            C9 = C14 + blockSize_x - 1;
                            C17 = C9 + 1;
                            C26 = C17 + 1;
                            local_tIn_t[t + 1][C0_next] = c0 * local_tIn_t[t][C0] + c1 * local_tIn_t[t][C1] + c2 * local_tIn_t[t][C2] + c3 * local_tIn_t[t][C3] + c4 * local_tIn_t[t][C4] + c5 * local_tIn_t[t][C5] + c6 * local_tIn_t[t][C6] + c7 * local_tIn_t[t][C7] + c8 * local_tIn_t[t][C8] + c9 * local_tIn_t[t][C9] + c10 * local_tIn_t[t][C10] + c11 * local_tIn_t[t][C11] + c12 * local_tIn_t[t][C12] + c13 * local_tIn_t[t][C13] + c14 * local_tIn_t[t][C14] + c15 * local_tIn_t[t][C15] + c16 * local_tIn_t[t][C16] + c17 * local_tIn_t[t][C17] + c18 * local_tIn_t[t][C18] + c19 * local_tIn_t[t][C19] + c20 * local_tIn_t[t][C20] + c21 * local_tIn_t[t][C21] + c22 * local_tIn_t[t][C22] + c23 * local_tIn_t[t][C23] + c24 * local_tIn_t[t][C24] + c25 * local_tIn_t[t][C25] + c26 * local_tIn_t[t][C26];
                        }
                    }
                }

                t = DIMT - 1;
                offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z;
                N_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                S_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                    {
                        C13 = N_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C0 = C_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C14 = S_z_block_t * blockSize_yx + y * blockSize_x + x;
                        C1 = C13 - blockSize_x - 1;
                        C10 = C1 + 1;
                        C18 = C10 + 1;
                        C4 = C13 - 1;
                        C21 = C13 + 1;
                        C7 = C13 + blockSize_x - 1;
                        C15 = C7 + 1;
                        C24 = C15 + 1;
                        C2 = C0 - blockSize_x - 1;
                        C11 = C2 + 1;
                        C19 = C11 + 1;
                        C5 = C0 - 1;
                        C22 = C0 + 1;
                        C8 = C0 + blockSize_x - 1;
                        C16 = C8 + 1;
                        C25 = C16 + 1;
                        C3 = C14 - blockSize_x - 1;
                        C12 = C3 + 1;
                        C20 = C12 + 1;
                        C6 = C14 - 1;
                        C23 = C14 + 1;
                        C9 = C14 + blockSize_x - 1;
                        C17 = C9 + 1;
                        C26 = C17 + 1;
                        tOut_ptr[x] = c0 * local_tIn_t[t][C0] + c1 * local_tIn_t[t][C1] + c2 * local_tIn_t[t][C2] + c3 * local_tIn_t[t][C3] + c4 * local_tIn_t[t][C4] + c5 * local_tIn_t[t][C5] + c6 * local_tIn_t[t][C6] + c7 * local_tIn_t[t][C7] + c8 * local_tIn_t[t][C8] + c9 * local_tIn_t[t][C9] + c10 * local_tIn_t[t][C10] + c11 * local_tIn_t[t][C11] + c12 * local_tIn_t[t][C12] + c13 * local_tIn_t[t][C13] + c14 * local_tIn_t[t][C14] + c15 * local_tIn_t[t][C15] + c16 * local_tIn_t[t][C16] + c17 * local_tIn_t[t][C17] + c18 * local_tIn_t[t][C18] + c19 * local_tIn_t[t][C19] + c20 * local_tIn_t[t][C20] + c21 * local_tIn_t[t][C21] + c22 * local_tIn_t[t][C22] + c23 * local_tIn_t[t][C23] + c24 * local_tIn_t[t][C24] + c25 * local_tIn_t[t][C25] + c26 * local_tIn_t[t][C26];
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
