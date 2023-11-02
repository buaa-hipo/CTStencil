#include "common.h"
#include <simd.h>
#include <slave.h>

/*
double a[M][N][P];
double b[M][N][P];
double c0, c1, c2, c3, c4, c5, c6;

for (long k = 1; k < M - 1; ++k)
{
    for (long j = 1; j < N - 1; ++j)
    {
        for (long i = 1; i < P - 1; ++i)
        {
            b[k][j][i] = c0 * a[k][j][i] + c1 * a[k][j][i - 1] + c2 * a[k][j][i + 1] 
                       + c3 * a[k - 1][j][i] + c4 * a[k + 1][j][i] + c5 * a[k][j - 1][i] 
                       + c6 * a[k][j + 1][i];
        }
    }
}
*/

#define COL(x) (x & 0x07)
#define ROW(x) ((x & 0x38) >> 3)
#define REG_PUTR(var, dst) asm volatile("putr %0,%1\n" ::"r"(var), "r"(dst))
#define REG_PUTC(var, dst) asm volatile("putc %0,%1\n" ::"r"(var), "r"(dst))
#define REG_GETR(var) asm volatile("getr %0\n" \
                                   : "=r"(var))
#define REG_GETC(var) asm volatile("getc %0\n" \
                                   : "=r"(var))
#define REG_LOAD(va, addr) asm volatile("vldd %0,%1\n" \
                                        : "=r"(va)     \
                                        : "m"(addr))
#define REG_STORE(va, addr) asm volatile("vstd %0,%1\n" ::"r"(va), "m"(addr))
#define REG_LOADER(va, addr) asm volatile("ldder %0,%1\n" \
                                          : "=r"(va)      \
                                          : "m"(addr))
#define REG_LOADR(va, addr) asm volatile("vldr %0,%1\n" \
                                         : "=r"(va)     \
                                         : "m"(addr))
#define REG_LOADC(va, addr) asm volatile("vldc %0,%1\n" \
                                         : "=r"(va)     \
                                         : "m"(addr))

extern DEFINED_DATATYPE c[NUMPOINTS];
// __thread_local DEFINED_DATATYPE local_tIn_t[DIMT][(1 + 2 * R) * (DIMY + 2 * R * DIMT) * (DIMX + 2 * R * DIMT)];
// __thread_local DEFINED_DATATYPE local_tOut[DIMX + 2 * R * DIMT];

void spe_func(struct spe_parameter *spe_param)
{
    int DIMT, DIMX, DIMY, DIMZ, NTH;
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
    int i, j, k;
    int aligned;

    temp = spe_param->temp;
    nt = spe_param->nt;
    nx = spe_param->nx;
    ny = spe_param->ny;
    nz = spe_param->nz;
    DIMT = spe_param->DIMT;
    DIMX = spe_param->DIMX;
    DIMY = spe_param->DIMY;
    DIMZ = spe_param->DIMZ;
    NTH = spe_param->NTH;

    blockNum_z = LAYERS / DIMZ;
    blockNum_y = NUMROWS / DIMY;
    blockNum_x = NUMCOLS / DIMX;

    DIMZ_padding = DIMZ + (LAYERS % DIMZ) / blockNum_z;
    DIMY_padding = DIMY + (NUMROWS % DIMY) / blockNum_y;
    DIMX_padding = DIMX + (NUMCOLS % DIMX) / blockNum_x;

    odd_block_z = LAYERS % DIMZ % blockNum_z;
    odd_block_y = NUMROWS % DIMY % blockNum_y;
    odd_block_x = NUMCOLS % DIMX % blockNum_x;

    DIMZ_final = DIMZ_padding, DIMY_final = DIMY_padding, DIMX_final = DIMX_padding;

    blockID_z = _MYID / (blockNum_y * blockNum_x);
    blockID_y = _MYID % (blockNum_y * blockNum_x) / blockNum_x;
    blockID_x = _MYID % blockNum_x;

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
    blockSize_x = blockSize_x % 4 ? blockSize_x + (4 - blockSize_x % 4) : blockSize_x;
    blockSize_y = DIMY_final + 2 * R * DIMT;
    blockSize_z = 1 + 2 * R;
    blockSize_z_0 = blockSize_z + 1; // preload

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

    int comm_step;
    int comm_mp[3][4] = {{1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
    switch (NTH)
    {
    case 2:
        comm_step = 1;
        break;
    case 4:
        comm_step = 3;
        break;
    default:
        break;
    }

    int CPE0_ID = _MYID - _MYID % NTH;
    int col_CPE0 = COL(CPE0_ID);
    int my_CPE_NUM = _MYID % NTH;
    int256 load_info[4];
    ((int *)(&load_info[my_CPE_NUM]))[0] = my_CPE_NUM;
    ((int *)(&load_info[my_CPE_NUM]))[1] = left_x_load;
    ((int *)(&load_info[my_CPE_NUM]))[2] = right_x_load;
    ((int *)(&load_info[my_CPE_NUM]))[3] = load_size;

    int s;
    int comm_CPE_NUM;
    for (s = 0; s < comm_step; s++)
    {
        comm_CPE_NUM = comm_mp[s][my_CPE_NUM];
        REG_PUTR(load_info[my_CPE_NUM], col_CPE0 + comm_mp[s][my_CPE_NUM]);
        REG_GETR(load_info[comm_CPE_NUM]);
        athread_syn(ROW_SCOPE, 0xff);
    }
    int co_load_size = ((int *)(&load_info[NTH - 1]))[2] - ((int *)(&load_info[0]))[1];

    int offset_j[4], offset_sum_j[4], right_x_cal_j[4], cal_size_j[4];
    int offset_sum = 0;
    for (j = 0; j <= NTH - 1; j++)
    {
        offset_j[j] = 2 * R * DIMT + (blockSize_x - ((int *)(&load_info[j]))[3]);
        offset_sum_j[j] = offset_sum;
        offset_sum += offset_j[j];
        cal_size_j[j] = ((int *)(&load_info[j]))[3] - 2 * R * DIMT;
    }

    int offset_z_block_0_preload;
    // offset_local > co_load_size
    int offset_local = NTH * blockSize_x;
    int repeat_y = (right_y_load - left_y_load) / NTH - 1;
    int left_y = (right_y_load - left_y_load) % NTH + NTH;

    for (i = 0; i < NUMTIMESTEPS / DIMT; i++)
    {
        int tmp = in;
        in = out;
        out = tmp;

        global_tIn = temp[in];
        global_tOut = temp[out];

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
                // int offset_z_block_0_preload = (z + 1 - left_z_compute_t[t]) % blockSize_z_0;
                // for (y = left_y_load; y < right_y_load; y++)
                // {
                //     C_global = (z + 1) * ny * nx + y * nx + left_x_load;
                //     C_local = offset_z_block_0_preload * blockSize_y * blockSize_x + (y - left_y_load) * blockSize_x;
                //     athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_preload, 0, 0, 0);
                //     wait_num++;
                // }

                offset_z_block_0_preload = (z + 1 - left_z_compute_t[t]) % blockSize_z_0;
                C_local = offset_z_block_0_preload * blockSize_y * blockSize_x;
                for (y = left_y_load + my_CPE_NUM; y < right_y_load - left_y; y += NTH)
                {
                    C_global = (z + 1) * ny * nx + y * nx + ((int *)(&load_info[0]))[1];
                    athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], co_load_size * sizeof(DEFINED_DATATYPE), &DMA_preload, 0, 0, 0);
                    wait_num++;
                    C_local += offset_local;
                }

                for (y = right_y_load - left_y; y < right_y_load; y++)
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
            }
            else
            {
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
                        tOut_ptr[x] = c0 * local_tIn_0[C] + c1 * local_tIn_0[B] + c2 * local_tIn_0[T] + c3 * local_tIn_0[W] + c4 * local_tIn_0[E] + c5 * local_tIn_0[S] + c6 * local_tIn_0[N];
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

                t = DIMT - 1;
                offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z;
                N_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                S_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                {
                    for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x++, x++)
                    {
                        C = C_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        W = C - 1;
                        E = C + 1;
                        N = C - blockSize_x;
                        S = C + blockSize_x;
                        B = N_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        T = S_z_block_t * blockSize_y * blockSize_x + y * blockSize_x + x;
                        tOut_ptr[x] = c0 * local_tIn_t[t][C] + c1 * local_tIn_t[t][B] + c2 * local_tIn_t[t][T] + c3 * local_tIn_t[t][W] + c4 * local_tIn_t[t][E] + c5 * local_tIn_t[t][S] + c6 * local_tIn_t[t][N];
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

                // duplicating
                int i_local = offset_z_block_0_preload * blockSize_y * blockSize_x;
                for (i = 0; i < repeat_y; i++)
                {
                    for (j = 0; j <= NTH - 1; j++)
                    {
                        right_x_cal_j[j] = i_local + ((int *)(&load_info[j]))[2] - ((int *)(&load_info[0]))[1] - R * DIMT - 1 + offset_sum_j[j];
                    }

                    int halo;
                    for (j = NTH - 1; j >= 1; j--)
                    {
                        halo = (j == NTH - 1 ? R * DIMT : 0);
                        for (k = right_x_cal_j[j] + halo; k > right_x_cal_j[j] - cal_size_j[j]; k--)
                            local_tIn_0[k] = local_tIn_0[k - offset_sum_j[j]];
                    }

                    for (j = 0; j < NTH - 1; j++)
                    {
                        // copy，blue halo
                        for (k = right_x_cal_j[j] + 1; k <= right_x_cal_j[j] + R * DIMT; k++)
                        {
                            local_tIn_0[k] = local_tIn_0[k + offset_j[j]];
                        }
                        // copy，yellow halo
                        for (k = right_x_cal_j[j]; k > right_x_cal_j[j] - R * DIMT; k--)
                        {
                            local_tIn_0[k + offset_j[j]] = local_tIn_0[k];
                        }
                    }

                    i_local += offset_local;
                }

                // exchanging
                DEFINED_V_DATATYPE v_p, v_g;
                aligned = 2;
                for (s = 0; s < comm_step; s++)
                {
                    int comm_CPE_NUM = comm_mp[s][my_CPE_NUM];
                    int col_comm_CPE = col_CPE0 + comm_CPE_NUM;
                    int i_local = offset_z_block_0_preload * blockSize_y * blockSize_x;
                    for (i = 0; i < repeat_y; i++)
                    {
                        int begin = i_local + blockSize_x * comm_CPE_NUM;
                        int right_x_block_comm = begin + blockSize_x;
                        j = begin - aligned;
                        simd_load(v_p, &local_tIn_0[j]);
                        REG_PUTR(v_p, col_comm_CPE);
                        REG_GETR(v_g);
                        for (k = 3; j + k >= begin; k--)
                            local_tIn_0[j + k] = ((DEFINED_DATATYPE *)(&v_g))[k];
                        for (j = begin - aligned + 4; j + 4 <= right_x_block_comm; j += 4)
                        {
                            simd_load(v_p, &local_tIn_0[j]);
                            REG_PUTR(v_p, col_comm_CPE);
                            REG_GETR(v_g);
                            ((DEFINED_V_DATATYPE *)(&local_tIn_0[j]))[0] = v_g;
                        }
                        simd_load(v_p, &local_tIn_0[j]);
                        REG_PUTR(v_p, col_comm_CPE);
                        REG_GETR(v_g);
                        for (k = 0; j + k < right_x_block_comm; k++)
                            local_tIn_0[j + k] = ((DEFINED_DATATYPE *)(&v_g))[k];
                        i_local += offset_local;
                    }
                    athread_syn(ROW_SCOPE, 0xff);
                }
            }
        }

        while (DMA_push != 1)
            ;
        athread_syn(ARRAY_SCOPE, 0xffff);
    }

    ldm_free(local_tIn_0, blockSize_z_0 * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
    if (DIMT >= 2)
    {
        ldm_free(local_tIn, (DIMT - 1) * blockSize_z * blockSize_y * blockSize_x * sizeof(DEFINED_DATATYPE));
        ldm_free(local_tIn_t, DIMT * sizeof(DEFINED_DATATYPE *));
    }
    ldm_free(local_tOut_1, blockSize_x * sizeof(DEFINED_DATATYPE));
    ldm_free(local_tOut_2, blockSize_x * sizeof(DEFINED_DATATYPE));

    spe_param->out = out;
}
