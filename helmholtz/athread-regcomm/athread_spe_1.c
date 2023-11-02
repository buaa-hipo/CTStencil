#include "common.h"
#include <slave.h>

// __thread_local DEFINED_DATATYPE local_tIn_t[DIMT][(1 + 2 * R) * (DIMY + 2 * R * DIMT) * (DIMX + 2 * R * DIMT)];
// __thread_local DEFINED_DATATYPE local_tOut[DIMX + 2 * R * DIMT];

void spe_func(struct spe_parameter *spe_param)
{
    int DIMT, DIMX, DIMY, DIMZ;
    DEFINED_DATATYPE a, b, h2inv, c, d, e, f;
    DEFINED_DATATYPE **temp;
    DEFINED_DATATYPE *local_tIn, *local_tOut, **local_tIn_t;
    int blockNum_z, blockNum_y, blockNum_x;
    int blockID_z, blockID_y, blockID_x;
    int DIMZ_padding, DIMY_padding, DIMX_padding;
    int odd_block_z, odd_block_y, odd_block_x;
    int DIMZ_final, DIMY_final, DIMX_final;
    int left_z_block, right_z_block, left_y_block, right_y_block, left_x_block, right_x_block;
    int left_z_load, right_z_load, left_y_load, right_y_load, left_x_load, right_x_load, load_size;
    int blockSize_x, blockSize_y, blockSize_z;
    int left_z_compute_t[DIMT + 1];
    int t, z, global_y, y, global_x, x;
    int nt, nz, ny, nx;
    int in = 1, out = 0;
    DEFINED_DATATYPE *global_tIn, *global_tOut;
    int offset_z_block_t, N2_z_block_t, N1_z_block_t, C_z_block_t, S1_z_block_t, S2_z_block_t, offset_z_block_t_next;
    int C_next, C, B1, B2, T1, T2, W1, W2, E1, E2, S1, S2, N1, N2;
    int C_global, C_local;
    volatile int DMA_reply, DMA_push;
    int i;
    int ng, blockNum_z_group, g;
    int N2_blocksize_z, N1_blocksize_z, C_blocksize_z, S1_blocksize_z, S2_blocksize_z, C_next_blocksize_z, m2_blocksize_x;
    int N2_blocksize_zy, N1_blocksize_zy, C_blocksize_zy, S1_blocksize_zy, S2_blocksize_zy, C_next_blocksize_zy;

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

    blockNum_z = LAYERS / DIMZ;
    blockNum_y = NUMROWS / DIMY;
    blockNum_x = NUMCOLS / DIMX;
    ng = blockNum_z * blockNum_y * blockNum_x / MAX_THREADS;
    blockNum_z_group = blockNum_z / ng;

    DIMZ_padding = DIMZ + (LAYERS % DIMZ) / blockNum_z;
    DIMY_padding = DIMY + (NUMROWS % DIMY) / blockNum_y;
    DIMX_padding = DIMX + (NUMCOLS % DIMX) / blockNum_x;

    odd_block_z = LAYERS % DIMZ % blockNum_z;
    odd_block_y = NUMROWS % DIMY % blockNum_y;
    odd_block_x = NUMCOLS % DIMX % blockNum_x;

    blockSize_x = DIMX_padding + 2 * R * DIMT;
    blockSize_y = DIMY_padding + 2 * R * DIMT;
    blockSize_z = 1 + 2 * R;

    m2_blocksize_x = 2 * blockSize_x;

    int total_float_ops = 0;
    int total_get_size = 0;
    int total_put_size = 0;

    for (i = 0; i < NUMTIMESTEPS / DIMT; i++)
    {
        int tmp = in;
        in = out;
        out = tmp;

        global_tIn = temp[in];
        global_tOut = temp[out];

        local_tIn = (DEFINED_DATATYPE *)ldm_malloc(DIMT * blockSize_z * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
        local_tIn_t = (DEFINED_DATATYPE **)ldm_malloc(DIMT * sizeof(DEFINED_DATATYPE *));
        for (t = 0; t < DIMT; t++)
            local_tIn_t[t] = &local_tIn[t * blockSize_z * blockSize_y * blockSize_x];
        local_tOut = (DEFINED_DATATYPE *)ldm_malloc(blockSize_x * sizeof(DEFINED_DATATYPE));

        for (g = 0; g < ng; g++)
        {
            blockID_z = _MYID / (blockNum_y * blockNum_x) + g * blockNum_z_group;
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

            for (t = 0; t <= DIMT; t++)
                left_z_compute_t[t] = left_z_load + 1 + 2 * R * t - 1;

            for (z = left_z_load; z < left_z_compute_t[DIMT]; z++)
            {
                for (t = 0; t < DIMT; t++)
                {
                    offset_z_block_t = (z - left_z_compute_t[t] + blockSize_z) % blockSize_z;
                    if (t == 0)
                    {
                        for (y = left_y_load; y < right_y_load; y++)
                        {
                            C_global = z * ny * nx + y * nx + left_x_load;
                            C_local = offset_z_block_t * blockSize_y * blockSize_x + (y - left_y_load) * blockSize_x;
                            DMA_reply = 0;
                            athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_t[0][C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_reply, 0, 0, 0);
                            while (DMA_reply != 1)
                                ;
                            total_get_size += load_size;
                        }
                    }
                    if (z < left_z_compute_t[t + 1])
                        continue;

                    // athread_syn(ARRAY_SCOPE, 0xffff);

                    N2_z_block_t = (offset_z_block_t - R - 2 + blockSize_z) % blockSize_z;
                    N1_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                    C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                    S1_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                    S2_z_block_t = (offset_z_block_t - R + 2 + blockSize_z) % blockSize_z;
                    offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
                    C_next_blocksize_z = offset_z_block_t_next * blockSize_y * blockSize_x;
                    N2_blocksize_z = N2_z_block_t * blockSize_y * blockSize_x;
                    N1_blocksize_z = N1_z_block_t * blockSize_y * blockSize_x;
                    C_blocksize_z = C_z_block_t * blockSize_y * blockSize_x;
                    S1_blocksize_z = S1_z_block_t * blockSize_y * blockSize_x;
                    S2_blocksize_z = S2_z_block_t * blockSize_y * blockSize_x;
                    for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                    {
                        C_next_blocksize_zy = C_next_blocksize_z + y * blockSize_x;
                        N2_blocksize_zy = N2_blocksize_z + y * blockSize_x;
                        N1_blocksize_zy = N1_blocksize_z + y * blockSize_x;
                        C_blocksize_zy = C_blocksize_z + y * blockSize_x;
                        S1_blocksize_zy = S1_blocksize_z + y * blockSize_x;
                        S2_blocksize_zy = S2_blocksize_z + y * blockSize_x;
                        for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                        {
                            C_next = C_next_blocksize_zy + x;
                            C = C_blocksize_zy + x;
                            W1 = C - 1;
                            W2 = C - 2;
                            E1 = C + 1;
                            E2 = C + 2;
                            N1 = C - blockSize_x;
                            N2 = C - m2_blocksize_x;
                            S1 = C + blockSize_x;
                            S2 = C + m2_blocksize_x;
                            B1 = N1_blocksize_zy + x;
                            B2 = N2_blocksize_zy + x;
                            T1 = S1_blocksize_zy + x;
                            T2 = S2_blocksize_zy + x;
                            local_tIn_t[t + 1][C_next] = (a - f) * local_tIn_t[t][C] + e * local_tIn_t[t][B1] + d * local_tIn_t[t][B2] + e * local_tIn_t[t][T1] + d * local_tIn_t[t][T2] + e * local_tIn_t[t][W1] + d * local_tIn_t[t][W2] + e * local_tIn_t[t][E1] + d * local_tIn_t[t][E2] + e * local_tIn_t[t][S1] + d * local_tIn_t[t][S2] + e * local_tIn_t[t][N1] + d * local_tIn_t[t][N2];
                            total_float_ops += 26;
                        }
                    }
                }
            }

            for (z = left_z_compute_t[DIMT]; z < right_z_load; z++)
            {
                int offset_z_block_0 = (z - left_z_compute_t[0]) % blockSize_z;
                for (y = left_y_load; y < right_y_load; y++)
                {
                    C_global = z * ny * nx + y * nx + left_x_load;
                    C_local = offset_z_block_0 * blockSize_y * blockSize_x + (y - left_y_load) * blockSize_x;
                    DMA_reply = 0;
                    athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_t[0][C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_reply, 0, 0, 0);
                    while (DMA_reply != 1)
                        ;
                    total_get_size += load_size;
                }

                for (t = 0; t < DIMT - 1; t++)
                {
                    offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z;
                    N2_z_block_t = (offset_z_block_t - R - 2 + blockSize_z) % blockSize_z;
                    N1_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                    C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                    S1_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                    S2_z_block_t = (offset_z_block_t - R + 2 + blockSize_z) % blockSize_z;
                    offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
                    C_next_blocksize_z = offset_z_block_t_next * blockSize_y * blockSize_x;
                    N2_blocksize_z = N2_z_block_t * blockSize_y * blockSize_x;
                    N1_blocksize_z = N1_z_block_t * blockSize_y * blockSize_x;
                    C_blocksize_z = C_z_block_t * blockSize_y * blockSize_x;
                    S1_blocksize_z = S1_z_block_t * blockSize_y * blockSize_x;
                    S2_blocksize_z = S2_z_block_t * blockSize_y * blockSize_x;
                    for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                    {
                        C_next_blocksize_zy = C_next_blocksize_z + y * blockSize_x;
                        N2_blocksize_zy = N2_blocksize_z + y * blockSize_x;
                        N1_blocksize_zy = N1_blocksize_z + y * blockSize_x;
                        C_blocksize_zy = C_blocksize_z + y * blockSize_x;
                        S1_blocksize_zy = S1_blocksize_z + y * blockSize_x;
                        S2_blocksize_zy = S2_blocksize_z + y * blockSize_x;
                        for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                        {
                            C_next = C_next_blocksize_zy + x;
                            C = C_blocksize_zy + x;
                            W1 = C - 1;
                            W2 = C - 2;
                            E1 = C + 1;
                            E2 = C + 2;
                            N1 = C - blockSize_x;
                            N2 = C - m2_blocksize_x;
                            S1 = C + blockSize_x;
                            S2 = C + m2_blocksize_x;
                            B1 = N1_blocksize_zy + x;
                            B2 = N2_blocksize_zy + x;
                            T1 = S1_blocksize_zy + x;
                            T2 = S2_blocksize_zy + x;
                            local_tIn_t[t + 1][C_next] = (a - f) * local_tIn_t[t][C] + e * local_tIn_t[t][B1] + d * local_tIn_t[t][B2] + e * local_tIn_t[t][T1] + d * local_tIn_t[t][T2] + e * local_tIn_t[t][W1] + d * local_tIn_t[t][W2] + e * local_tIn_t[t][E1] + d * local_tIn_t[t][E2] + e * local_tIn_t[t][S1] + d * local_tIn_t[t][S2] + e * local_tIn_t[t][N1] + d * local_tIn_t[t][N2];
                            total_float_ops += 26;
                        }
                    }
                }

                // athread_syn(ARRAY_SCOPE, 0xffff);

                t = DIMT - 1;
                offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z;
                N2_z_block_t = (offset_z_block_t - R - 2 + blockSize_z) % blockSize_z;
                N1_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                S1_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                S2_z_block_t = (offset_z_block_t - R + 2 + blockSize_z) % blockSize_z;
                offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
                C_next_blocksize_z = offset_z_block_t_next * blockSize_y * blockSize_x;
                N2_blocksize_z = N2_z_block_t * blockSize_y * blockSize_x;
                N1_blocksize_z = N1_z_block_t * blockSize_y * blockSize_x;
                C_blocksize_z = C_z_block_t * blockSize_y * blockSize_x;
                S1_blocksize_z = S1_z_block_t * blockSize_y * blockSize_x;
                S2_blocksize_z = S2_z_block_t * blockSize_y * blockSize_x;
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    C_next_blocksize_zy = C_next_blocksize_z + y * blockSize_x;
                    N2_blocksize_zy = N2_blocksize_z + y * blockSize_x;
                    N1_blocksize_zy = N1_blocksize_z + y * blockSize_x;
                    C_blocksize_zy = C_blocksize_z + y * blockSize_x;
                    S1_blocksize_zy = S1_blocksize_z + y * blockSize_x;
                    S2_blocksize_zy = S2_blocksize_z + y * blockSize_x;
                    for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                    {
                        C_next = C_next_blocksize_zy + x;
                        C = C_blocksize_zy + x;
                        W1 = C - 1;
                        W2 = C - 2;
                        E1 = C + 1;
                        E2 = C + 2;
                        N1 = C - blockSize_x;
                        N2 = C - m2_blocksize_x;
                        S1 = C + blockSize_x;
                        S2 = C + m2_blocksize_x;
                        B1 = N1_blocksize_zy + x;
                        B2 = N2_blocksize_zy + x;
                        T1 = S1_blocksize_zy + x;
                        T2 = S2_blocksize_zy + x;
                        local_tOut[x] = (a - f) * local_tIn_t[t][C] + e * local_tIn_t[t][B1] + d * local_tIn_t[t][B2] + e * local_tIn_t[t][T1] + d * local_tIn_t[t][T2] + e * local_tIn_t[t][W1] + d * local_tIn_t[t][W2] + e * local_tIn_t[t][E1] + d * local_tIn_t[t][E2] + e * local_tIn_t[t][S1] + d * local_tIn_t[t][S2] + e * local_tIn_t[t][N1] + d * local_tIn_t[t][N2];
                        total_float_ops += 26;
                    }
                    DMA_push = 0;
                    C_global = (z - R * DIMT) * ny * nx + global_y * nx + left_x_load + R * DIMT;
                    athread_put(PE_MODE, &local_tOut[R * DIMT], &global_tOut[C_global], (load_size - 2 * R * DIMT) * sizeof(DEFINED_DATATYPE), &DMA_push, 0, 0);
                    while (DMA_push != 1)
                        ;
                    total_put_size += (load_size - 2 * R * DIMT);
                }
            }
            athread_syn(ARRAY_SCOPE, 0xffff);
        }

        ldm_free(local_tIn, DIMT * blockSize_z * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
        ldm_free(local_tIn_t, DIMT * sizeof(DEFINED_DATATYPE *));
        ldm_free(local_tOut, blockSize_x * sizeof(DEFINED_DATATYPE));
    }
    spe_param->out = out;

    spe_param->slave_float_ops[_MYID] = total_float_ops;
    spe_param->slave_get_size[_MYID] = total_get_size;
    spe_param->slave_put_size[_MYID] = total_put_size;
}
