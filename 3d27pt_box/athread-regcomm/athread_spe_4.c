#include "common.h"
#include <simd.h>
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
    int it;
    int g, ng, blockNum_y_group;
    int i, j, k;

    int aligned;
    DEFINED_V_DATATYPE v_C0, v_C1, v_C2, v_C3, v_C4, v_C5, v_C6, v_C7, v_C8, v_C9, v_C10, v_C11, v_C12, v_C13, v_C14, v_C15, v_C16, v_C17, v_C18, v_C19, v_C20, v_C21, v_C22, v_C23, v_C24, v_C25, v_C26, v_res;
    DEFINED_V_DATATYPE v_tc1, v_tc2, v_t1, v_t2;
    int rc1 = 0x4E;
    int rc2 = 0x99;
    DEFINED_V_DATATYPE v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6, v_c7, v_c8, v_c9, v_c10, v_c11, v_c12, v_c13, v_c14, v_c15, v_c16, v_c17, v_c18, v_c19, v_c20, v_c21, v_c22, v_c23, v_c24, v_c25, v_c26;
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
    v_c9 = simd_set_floatv4(c9, c9, c9, c9);
    v_c10 = simd_set_floatv4(c10, c10, c10, c10);
    v_c11 = simd_set_floatv4(c11, c11, c11, c11);
    v_c12 = simd_set_floatv4(c12, c12, c12, c12);
    v_c13 = simd_set_floatv4(c13, c13, c13, c13);
    v_c14 = simd_set_floatv4(c14, c14, c14, c14);
    v_c15 = simd_set_floatv4(c15, c15, c15, c15);
    v_c16 = simd_set_floatv4(c16, c16, c16, c16);
    v_c17 = simd_set_floatv4(c17, c17, c17, c17);
    v_c18 = simd_set_floatv4(c18, c18, c18, c18);
    v_c19 = simd_set_floatv4(c19, c19, c19, c19);
    v_c20 = simd_set_floatv4(c20, c20, c20, c20);
    v_c21 = simd_set_floatv4(c21, c21, c21, c21);
    v_c22 = simd_set_floatv4(c22, c22, c22, c22);
    v_c23 = simd_set_floatv4(c23, c23, c23, c23);
    v_c24 = simd_set_floatv4(c24, c24, c24, c24);
    v_c25 = simd_set_floatv4(c25, c25, c25, c25);
    v_c26 = simd_set_floatv4(c26, c26, c26, c26);
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
    v_c9 = simd_set_doublev4(c9, c9, c9, c9);
    v_c10 = simd_set_doublev4(c10, c10, c10, c10);
    v_c11 = simd_set_doublev4(c11, c11, c11, c11);
    v_c12 = simd_set_doublev4(c12, c12, c12, c12);
    v_c13 = simd_set_doublev4(c13, c13, c13, c13);
    v_c14 = simd_set_doublev4(c14, c14, c14, c14);
    v_c15 = simd_set_doublev4(c15, c15, c15, c15);
    v_c16 = simd_set_doublev4(c16, c16, c16, c16);
    v_c17 = simd_set_doublev4(c17, c17, c17, c17);
    v_c18 = simd_set_doublev4(c18, c18, c18, c18);
    v_c19 = simd_set_doublev4(c19, c19, c19, c19);
    v_c20 = simd_set_doublev4(c20, c20, c20, c20);
    v_c21 = simd_set_doublev4(c21, c21, c21, c21);
    v_c22 = simd_set_doublev4(c22, c22, c22, c22);
    v_c23 = simd_set_doublev4(c23, c23, c23, c23);
    v_c24 = simd_set_doublev4(c24, c24, c24, c24);
    v_c25 = simd_set_doublev4(c25, c25, c25, c25);
    v_c26 = simd_set_doublev4(c26, c26, c26, c26);
#endif

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

    if (DIMZ == nz)
    {
        ng = blockNum_x * blockNum_y / MAX_THREADS;
        blockNum_y_group = blockNum_y / ng;
    }
    else
    {
        ng = blockNum_x * blockNum_y * blockNum_z / MAX_THREADS;
    }

    DIMZ_padding = DIMZ + (LAYERS % DIMZ) / blockNum_z;
    DIMY_padding = DIMY + (NUMROWS % DIMY) / blockNum_y;
    DIMX_padding = DIMX + (NUMCOLS % DIMX) / blockNum_x;

    odd_block_z = LAYERS % DIMZ % blockNum_z;
    odd_block_y = NUMROWS % DIMY % blockNum_y;
    odd_block_x = NUMCOLS % DIMX % blockNum_x;

    DIMZ_final = DIMZ_padding, DIMY_final = DIMY_padding, DIMX_final = DIMX_padding;

    blockSize_x = DIMX_final + 2 * R * DIMT;
    blockSize_x = blockSize_x % 4 ? blockSize_x + (4 - blockSize_x % 4) : blockSize_x;
    blockSize_y = DIMY_final + 2 * R * DIMT;
    blockSize_z = 1 + 2 * R;
    blockSize_z_0 = blockSize_z + 1; // preload
    blockSize_yx = blockSize_y * blockSize_x;

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

    int total_float_ops = 0;
    int total_get_size = 0;
    int total_put_size = 0;

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

    for (it = 0; it < NUMTIMESTEPS / DIMT; it++)
    {
        int tmp = in;
        in = out;
        out = tmp;

        global_tIn = temp[in];
        global_tOut = temp[out];

        for (g = 0; g < ng; g++)
        {
            if (DIMZ == nz)
            {
                blockID_z = _MYID / (blockNum_y * blockNum_x);
                blockID_y = _MYID / blockNum_x + blockNum_y_group * g;
                blockID_x = _MYID % blockNum_x;
            }
            else
            {
                blockID_z = _MYID / (blockNum_y * blockNum_x);
                blockID_y = _MYID % (blockNum_y * blockNum_x) / blockNum_x;
                blockID_x = _MYID % blockNum_x;
            }

            if (blockID_z < odd_block_z)
            {
                DIMZ_final = DIMZ_padding + 1;
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
                DIMY_final = DIMY_padding + 1;
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
                DIMX_final = DIMX_padding + 1;
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

            int CPE0_ID = _MYID - _MYID % NTH;
            int col_CPE0 = COL(CPE0_ID);
            int my_CPE_NUM = _MYID % NTH;
            int256 load_info[4];
            ((int *)(&load_info[my_CPE_NUM]))[0] = my_CPE_NUM;
            ((int *)(&load_info[my_CPE_NUM]))[1] = left_x_load;
            ((int *)(&load_info[my_CPE_NUM]))[2] = right_x_load;
            ((int *)(&load_info[my_CPE_NUM]))[3] = load_size;
            // ((int *)(&load_info[my_CPE_NUM]))[4] = blockSize_x;

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

            // if (_MYID == 0)
            // {
            //     for (i = 0; i < NTH; i++)
            //         printf("%d %d %d\n", ((int *)(&load_info[i]))[0], ((int *)(&load_info[i]))[1], ((int *)(&load_info[i]))[2]);
                
            //     printf("load_size: %d, co_load_size: %d\n", load_size, co_load_size);
            // }

            int offset_j[4], offset_sum_j[4], right_x_cal_j[4], cal_size_j[4];
            int offset_sum = 0;
            for (j = 0; j <= NTH - 1; j++)
            {
                offset_j[j] = 2 * R * DIMT + (blockSize_x - ((int *)(&load_info[j]))[3]);
                offset_sum_j[j] = offset_sum;
                offset_sum += offset_j[j];
                cal_size_j[j] = ((int *)(&load_info[j]))[3] - 2 * R * DIMT;

                // if (_MYID == 1)
                //     printf("offset_j: %d, offset_sum_j: %d, cal_size_j: %d\n", offset_j[j], offset_sum_j[j], cal_size_j[j]);
            }

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
                    total_get_size += load_size;
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
                        total_float_ops += 53;
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
                            total_float_ops += 53;
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
                total_get_size += load_size;
            }

            int offset_z_block_0_preload;
            // offset_local > co_load_size
            int offset_local = NTH * blockSize_x;
            int repeat_y = (right_y_load - left_y_load) / NTH - 1;
            int left_y = (right_y_load - left_y_load) % NTH + NTH;

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
                    //     total_get_size += load_size;
                    // }

                    offset_z_block_0_preload = (z + 1 - left_z_compute_t[t]) % blockSize_z_0;

                    C_local = offset_z_block_0_preload * blockSize_y * blockSize_x;
                    for (y = left_y_load + my_CPE_NUM; y < right_y_load - left_y; y += NTH)
                    {
                        C_global = (z + 1) * ny * nx + y * nx + ((int *)(&load_info[0]))[1];
                        // C_global = 0;
                        // C_local = 0;
                        athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], co_load_size * sizeof(DEFINED_DATATYPE), &DMA_preload, 0, 0, 0);
                        // athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], (co_load_size - co_load_size % 4) * sizeof(DEFINED_DATATYPE), &DMA_preload, 0, 0, 0);
                        wait_num++;
                        C_local += offset_local;
                        total_get_size += co_load_size;
                    }

                    for (y = right_y_load - left_y; y < right_y_load; y++)
                    {
                        C_global = (z + 1) * ny * nx + y * nx + left_x_load;
                        C_local = offset_z_block_0_preload * blockSize_y * blockSize_x + (y - left_y_load) * blockSize_x;
                        athread_get(PE_MODE, &global_tIn[C_global], &local_tIn_0[C_local], load_size * sizeof(DEFINED_DATATYPE), &DMA_preload, 0, 0, 0);
                        wait_num++;
                        total_get_size += load_size;
                    }
                }

                offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z_0;
                N_z_block_t = (offset_z_block_t - R - 1 + blockSize_z_0) % blockSize_z_0;
                C_z_block_t = (offset_z_block_t - R + blockSize_z_0) % blockSize_z_0;
                S_z_block_t = (offset_z_block_t - R + 1 + blockSize_z_0) % blockSize_z_0;
                offset_z_block_t_next = (z - left_z_compute_t[t + 1]) % blockSize_z;
                aligned = (R * (t + 1) + 2) % 4;
                if (DIMT >= 2)
                {
                    for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                    {
                        for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
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
                            switch (aligned)
                            {
                            case 0:
                                vload_3p_0(C10, v_C1, v_C10, v_C18, local_tIn_0);
                                vload_3p_0(C13, v_C4, v_C13, v_C21, local_tIn_0);
                                vload_3p_0(C15, v_C7, v_C15, v_C24, local_tIn_0);
                                vload_3p_0(C11, v_C2, v_C11, v_C19, local_tIn_0);
                                vload_3p_0(C0, v_C5, v_C0, v_C22, local_tIn_0);
                                vload_3p_0(C16, v_C8, v_C16, v_C25, local_tIn_0);
                                vload_3p_0(C12, v_C3, v_C12, v_C20, local_tIn_0);
                                vload_3p_0(C14, v_C6, v_C14, v_C23, local_tIn_0);
                                vload_3p_0(C17, v_C9, v_C17, v_C26, local_tIn_0);
                                v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                simd_store(v_res, &local_tIn_t[t + 1][C0_next]);
                                break;
                            case 1:
                                vload_3p_1(C10, v_C1, v_C10, v_C18, local_tIn_0);
                                vload_3p_1(C13, v_C4, v_C13, v_C21, local_tIn_0);
                                vload_3p_1(C15, v_C7, v_C15, v_C24, local_tIn_0);
                                vload_3p_1(C11, v_C2, v_C11, v_C19, local_tIn_0);
                                vload_3p_1(C0, v_C5, v_C0, v_C22, local_tIn_0);
                                vload_3p_1(C16, v_C8, v_C16, v_C25, local_tIn_0);
                                vload_3p_1(C12, v_C3, v_C12, v_C20, local_tIn_0);
                                vload_3p_1(C14, v_C6, v_C14, v_C23, local_tIn_0);
                                vload_3p_1(C17, v_C9, v_C17, v_C26, local_tIn_0);
                                v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                simd_storeu(v_res, &local_tIn_t[t + 1][C0_next]);
                                break;
                            case 2:
                                vload_3p_2(C10, v_C1, v_C10, v_C18, local_tIn_0);
                                vload_3p_2(C13, v_C4, v_C13, v_C21, local_tIn_0);
                                vload_3p_2(C15, v_C7, v_C15, v_C24, local_tIn_0);
                                vload_3p_2(C11, v_C2, v_C11, v_C19, local_tIn_0);
                                vload_3p_2(C0, v_C5, v_C0, v_C22, local_tIn_0);
                                vload_3p_2(C16, v_C8, v_C16, v_C25, local_tIn_0);
                                vload_3p_2(C12, v_C3, v_C12, v_C20, local_tIn_0);
                                vload_3p_2(C14, v_C6, v_C14, v_C23, local_tIn_0);
                                vload_3p_2(C17, v_C9, v_C17, v_C26, local_tIn_0);
                                v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                simd_storeu(v_res, &local_tIn_t[t + 1][C0_next]);
                                break;
                            case 3:
                                vload_3p_3(C10, v_C1, v_C10, v_C18, local_tIn_0);
                                vload_3p_3(C13, v_C4, v_C13, v_C21, local_tIn_0);
                                vload_3p_3(C15, v_C7, v_C15, v_C24, local_tIn_0);
                                vload_3p_3(C11, v_C2, v_C11, v_C19, local_tIn_0);
                                vload_3p_3(C0, v_C5, v_C0, v_C22, local_tIn_0);
                                vload_3p_3(C16, v_C8, v_C16, v_C25, local_tIn_0);
                                vload_3p_3(C12, v_C3, v_C12, v_C20, local_tIn_0);
                                vload_3p_3(C14, v_C6, v_C14, v_C23, local_tIn_0);
                                vload_3p_3(C17, v_C9, v_C17, v_C26, local_tIn_0);
                                v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                simd_storeu(v_res, &local_tIn_t[t + 1][C0_next]);
                                break;
                            default:
                                break;
                            }
                            total_float_ops += 53 * 4;
                            // local_tIn_t[t + 1][C0_next] = c0 * local_tIn_0[C0] + c1 * local_tIn_0[C1] + c2 * local_tIn_0[C2] + c3 * local_tIn_0[C3] + c4 * local_tIn_0[C4] + c5 * local_tIn_0[C5] + c6 * local_tIn_0[C6] + c7 * local_tIn_0[C7] + c8 * local_tIn_0[C8] + c9 * local_tIn_0[C9] + c10 * local_tIn_0[C10] + c11 * local_tIn_0[C11] + c12 * local_tIn_0[C12] + c13 * local_tIn_0[C13] + c14 * local_tIn_0[C14] + c15 * local_tIn_0[C15] + c16 * local_tIn_0[C16] + c17 * local_tIn_0[C17] + c18 * local_tIn_0[C18] + c19 * local_tIn_0[C19] + c20 * local_tIn_0[C20] + c21 * local_tIn_0[C21] + c22 * local_tIn_0[C22] + c23 * local_tIn_0[C23] + c24 * local_tIn_0[C24] + c25 * local_tIn_0[C25] + c26 * local_tIn_0[C26];
                        }
                    }
                }
                else
                {
                    for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                    {
                        for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
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
                            switch (aligned)
                            {
                            case 0:
                                vload_3p_0(C10, v_C1, v_C10, v_C18, local_tIn_0);
                                vload_3p_0(C13, v_C4, v_C13, v_C21, local_tIn_0);
                                vload_3p_0(C15, v_C7, v_C15, v_C24, local_tIn_0);
                                vload_3p_0(C11, v_C2, v_C11, v_C19, local_tIn_0);
                                vload_3p_0(C0, v_C5, v_C0, v_C22, local_tIn_0);
                                vload_3p_0(C16, v_C8, v_C16, v_C25, local_tIn_0);
                                vload_3p_0(C12, v_C3, v_C12, v_C20, local_tIn_0);
                                vload_3p_0(C14, v_C6, v_C14, v_C23, local_tIn_0);
                                vload_3p_0(C17, v_C9, v_C17, v_C26, local_tIn_0);
                                v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                simd_store(v_res, &tOut_ptr[x]);
                                break;
                            case 1:
                                vload_3p_1(C10, v_C1, v_C10, v_C18, local_tIn_0);
                                vload_3p_1(C13, v_C4, v_C13, v_C21, local_tIn_0);
                                vload_3p_1(C15, v_C7, v_C15, v_C24, local_tIn_0);
                                vload_3p_1(C11, v_C2, v_C11, v_C19, local_tIn_0);
                                vload_3p_1(C0, v_C5, v_C0, v_C22, local_tIn_0);
                                vload_3p_1(C16, v_C8, v_C16, v_C25, local_tIn_0);
                                vload_3p_1(C12, v_C3, v_C12, v_C20, local_tIn_0);
                                vload_3p_1(C14, v_C6, v_C14, v_C23, local_tIn_0);
                                vload_3p_1(C17, v_C9, v_C17, v_C26, local_tIn_0);
                                v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                simd_storeu(v_res, &tOut_ptr[x]);
                                break;
                            case 2:
                                vload_3p_2(C10, v_C1, v_C10, v_C18, local_tIn_0);
                                vload_3p_2(C13, v_C4, v_C13, v_C21, local_tIn_0);
                                vload_3p_2(C15, v_C7, v_C15, v_C24, local_tIn_0);
                                vload_3p_2(C11, v_C2, v_C11, v_C19, local_tIn_0);
                                vload_3p_2(C0, v_C5, v_C0, v_C22, local_tIn_0);
                                vload_3p_2(C16, v_C8, v_C16, v_C25, local_tIn_0);
                                vload_3p_2(C12, v_C3, v_C12, v_C20, local_tIn_0);
                                vload_3p_2(C14, v_C6, v_C14, v_C23, local_tIn_0);
                                vload_3p_2(C17, v_C9, v_C17, v_C26, local_tIn_0);
                                v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                simd_storeu(v_res, &tOut_ptr[x]);
                                break;
                            case 3:
                                vload_3p_3(C10, v_C1, v_C10, v_C18, local_tIn_0);
                                vload_3p_3(C13, v_C4, v_C13, v_C21, local_tIn_0);
                                vload_3p_3(C15, v_C7, v_C15, v_C24, local_tIn_0);
                                vload_3p_3(C11, v_C2, v_C11, v_C19, local_tIn_0);
                                vload_3p_3(C0, v_C5, v_C0, v_C22, local_tIn_0);
                                vload_3p_3(C16, v_C8, v_C16, v_C25, local_tIn_0);
                                vload_3p_3(C12, v_C3, v_C12, v_C20, local_tIn_0);
                                vload_3p_3(C14, v_C6, v_C14, v_C23, local_tIn_0);
                                vload_3p_3(C17, v_C9, v_C17, v_C26, local_tIn_0);
                                v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                simd_storeu(v_res, &tOut_ptr[x]);
                                break;
                            default:
                                break;
                            }
                            total_float_ops += 53 * 4;
                            // local_tIn_t[t + 1][C0_next] = c0 * local_tIn_0[C0] + c1 * local_tIn_0[C1] + c2 * local_tIn_0[C2] + c3 * local_tIn_0[C3] + c4 * local_tIn_0[C4] + c5 * local_tIn_0[C5] + c6 * local_tIn_0[C6] + c7 * local_tIn_0[C7] + c8 * local_tIn_0[C8] + c9 * local_tIn_0[C9] + c10 * local_tIn_0[C10] + c11 * local_tIn_0[C11] + c12 * local_tIn_0[C12] + c13 * local_tIn_0[C13] + c14 * local_tIn_0[C14] + c15 * local_tIn_0[C15] + c16 * local_tIn_0[C16] + c17 * local_tIn_0[C17] + c18 * local_tIn_0[C18] + c19 * local_tIn_0[C19] + c20 * local_tIn_0[C20] + c21 * local_tIn_0[C21] + c22 * local_tIn_0[C22] + c23 * local_tIn_0[C23] + c24 * local_tIn_0[C24] + c25 * local_tIn_0[C25] + c26 * local_tIn_0[C26];
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
                        total_put_size += (load_size - 2 * R * DIMT);
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
                        aligned = (R * (t + 1) + 2) % 4;
                        for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                        {
                            for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
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
                                switch (aligned)
                                {
                                case 0:
                                    vload_3p_0(C10, v_C1, v_C10, v_C18, local_tIn_t[t]);
                                    vload_3p_0(C13, v_C4, v_C13, v_C21, local_tIn_t[t]);
                                    vload_3p_0(C15, v_C7, v_C15, v_C24, local_tIn_t[t]);
                                    vload_3p_0(C11, v_C2, v_C11, v_C19, local_tIn_t[t]);
                                    vload_3p_0(C0, v_C5, v_C0, v_C22, local_tIn_t[t]);
                                    vload_3p_0(C16, v_C8, v_C16, v_C25, local_tIn_t[t]);
                                    vload_3p_0(C12, v_C3, v_C12, v_C20, local_tIn_t[t]);
                                    vload_3p_0(C14, v_C6, v_C14, v_C23, local_tIn_t[t]);
                                    vload_3p_0(C17, v_C9, v_C17, v_C26, local_tIn_t[t]);
                                    v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                    simd_store(v_res, &local_tIn_t[t + 1][C0_next]);
                                    break;
                                case 1:
                                    vload_3p_1(C10, v_C1, v_C10, v_C18, local_tIn_t[t]);
                                    vload_3p_1(C13, v_C4, v_C13, v_C21, local_tIn_t[t]);
                                    vload_3p_1(C15, v_C7, v_C15, v_C24, local_tIn_t[t]);
                                    vload_3p_1(C11, v_C2, v_C11, v_C19, local_tIn_t[t]);
                                    vload_3p_1(C0, v_C5, v_C0, v_C22, local_tIn_t[t]);
                                    vload_3p_1(C16, v_C8, v_C16, v_C25, local_tIn_t[t]);
                                    vload_3p_1(C12, v_C3, v_C12, v_C20, local_tIn_t[t]);
                                    vload_3p_1(C14, v_C6, v_C14, v_C23, local_tIn_t[t]);
                                    vload_3p_1(C17, v_C9, v_C17, v_C26, local_tIn_t[t]);
                                    v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                    simd_storeu(v_res, &local_tIn_t[t + 1][C0_next]);
                                    break;
                                case 2:
                                    vload_3p_2(C10, v_C1, v_C10, v_C18, local_tIn_t[t]);
                                    vload_3p_2(C13, v_C4, v_C13, v_C21, local_tIn_t[t]);
                                    vload_3p_2(C15, v_C7, v_C15, v_C24, local_tIn_t[t]);
                                    vload_3p_2(C11, v_C2, v_C11, v_C19, local_tIn_t[t]);
                                    vload_3p_2(C0, v_C5, v_C0, v_C22, local_tIn_t[t]);
                                    vload_3p_2(C16, v_C8, v_C16, v_C25, local_tIn_t[t]);
                                    vload_3p_2(C12, v_C3, v_C12, v_C20, local_tIn_t[t]);
                                    vload_3p_2(C14, v_C6, v_C14, v_C23, local_tIn_t[t]);
                                    vload_3p_2(C17, v_C9, v_C17, v_C26, local_tIn_t[t]);
                                    v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                    simd_storeu(v_res, &local_tIn_t[t + 1][C0_next]);
                                    break;
                                case 3:
                                    vload_3p_3(C10, v_C1, v_C10, v_C18, local_tIn_t[t]);
                                    vload_3p_3(C13, v_C4, v_C13, v_C21, local_tIn_t[t]);
                                    vload_3p_3(C15, v_C7, v_C15, v_C24, local_tIn_t[t]);
                                    vload_3p_3(C11, v_C2, v_C11, v_C19, local_tIn_t[t]);
                                    vload_3p_3(C0, v_C5, v_C0, v_C22, local_tIn_t[t]);
                                    vload_3p_3(C16, v_C8, v_C16, v_C25, local_tIn_t[t]);
                                    vload_3p_3(C12, v_C3, v_C12, v_C20, local_tIn_t[t]);
                                    vload_3p_3(C14, v_C6, v_C14, v_C23, local_tIn_t[t]);
                                    vload_3p_3(C17, v_C9, v_C17, v_C26, local_tIn_t[t]);
                                    v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                                    simd_storeu(v_res, &local_tIn_t[t + 1][C0_next]);
                                    break;
                                default:
                                    break;
                                }
                                total_float_ops += 53 * 4;
                            }
                            // local_tIn_t[t + 1][C0_next] = c0 * local_tIn_t[t][C0] + c1 * local_tIn_t[t][C1] + c2 * local_tIn_t[t][C2] + c3 * local_tIn_t[t][C3] + c4 * local_tIn_t[t][C4] + c5 * local_tIn_t[t][C5] + c6 * local_tIn_t[t][C6] + c7 * local_tIn_t[t][C7] + c8 * local_tIn_t[t][C8] + c9 * local_tIn_t[t][C9] + c10 * local_tIn_t[t][C10] + c11 * local_tIn_t[t][C11] + c12 * local_tIn_t[t][C12] + c13 * local_tIn_t[t][C13] + c14 * local_tIn_t[t][C14] + c15 * local_tIn_t[t][C15] + c16 * local_tIn_t[t][C16] + c17 * local_tIn_t[t][C17] + c18 * local_tIn_t[t][C18] + c19 * local_tIn_t[t][C19] + c20 * local_tIn_t[t][C20] + c21 * local_tIn_t[t][C21] + c22 * local_tIn_t[t][C22] + c23 * local_tIn_t[t][C23] + c24 * local_tIn_t[t][C24] + c25 * local_tIn_t[t][C25] + c26 * local_tIn_t[t][C26];
                        }
                    }

                    t = DIMT - 1;
                    offset_z_block_t = (z - left_z_compute_t[t]) % blockSize_z;
                    N_z_block_t = (offset_z_block_t - R - 1 + blockSize_z) % blockSize_z;
                    C_z_block_t = (offset_z_block_t - R + blockSize_z) % blockSize_z;
                    S_z_block_t = (offset_z_block_t - R + 1 + blockSize_z) % blockSize_z;
                    aligned = (R * (t + 1) + 2) % 4;
                    for (global_y = left_y_load + R * (t + 1), y = R * (t + 1); global_y < right_y_load - R * (t + 1); global_y++, y++)
                    {
                        for (global_x = left_x_load + R * (t + 1), x = R * (t + 1); global_x < right_x_load - R * (t + 1); global_x += 4, x += 4)
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
                            switch (aligned)
                            {
                            case 0:
                                vload_3p_0(C10, v_C1, v_C10, v_C18, local_tIn_t[t]);
                                vload_3p_0(C13, v_C4, v_C13, v_C21, local_tIn_t[t]);
                                vload_3p_0(C15, v_C7, v_C15, v_C24, local_tIn_t[t]);
                                vload_3p_0(C11, v_C2, v_C11, v_C19, local_tIn_t[t]);
                                vload_3p_0(C0, v_C5, v_C0, v_C22, local_tIn_t[t]);
                                vload_3p_0(C16, v_C8, v_C16, v_C25, local_tIn_t[t]);
                                vload_3p_0(C12, v_C3, v_C12, v_C20, local_tIn_t[t]);
                                vload_3p_0(C14, v_C6, v_C14, v_C23, local_tIn_t[t]);
                                vload_3p_0(C17, v_C9, v_C17, v_C26, local_tIn_t[t]);
                                break;
                            case 1:
                                vload_3p_1(C10, v_C1, v_C10, v_C18, local_tIn_t[t]);
                                vload_3p_1(C13, v_C4, v_C13, v_C21, local_tIn_t[t]);
                                vload_3p_1(C15, v_C7, v_C15, v_C24, local_tIn_t[t]);
                                vload_3p_1(C11, v_C2, v_C11, v_C19, local_tIn_t[t]);
                                vload_3p_1(C0, v_C5, v_C0, v_C22, local_tIn_t[t]);
                                vload_3p_1(C16, v_C8, v_C16, v_C25, local_tIn_t[t]);
                                vload_3p_1(C12, v_C3, v_C12, v_C20, local_tIn_t[t]);
                                vload_3p_1(C14, v_C6, v_C14, v_C23, local_tIn_t[t]);
                                vload_3p_1(C17, v_C9, v_C17, v_C26, local_tIn_t[t]);
                                break;
                            case 2:
                                vload_3p_2(C10, v_C1, v_C10, v_C18, local_tIn_t[t]);
                                vload_3p_2(C13, v_C4, v_C13, v_C21, local_tIn_t[t]);
                                vload_3p_2(C15, v_C7, v_C15, v_C24, local_tIn_t[t]);
                                vload_3p_2(C11, v_C2, v_C11, v_C19, local_tIn_t[t]);
                                vload_3p_2(C0, v_C5, v_C0, v_C22, local_tIn_t[t]);
                                vload_3p_2(C16, v_C8, v_C16, v_C25, local_tIn_t[t]);
                                vload_3p_2(C12, v_C3, v_C12, v_C20, local_tIn_t[t]);
                                vload_3p_2(C14, v_C6, v_C14, v_C23, local_tIn_t[t]);
                                vload_3p_2(C17, v_C9, v_C17, v_C26, local_tIn_t[t]);
                                break;
                            case 3:
                                vload_3p_3(C10, v_C1, v_C10, v_C18, local_tIn_t[t]);
                                vload_3p_3(C13, v_C4, v_C13, v_C21, local_tIn_t[t]);
                                vload_3p_3(C15, v_C7, v_C15, v_C24, local_tIn_t[t]);
                                vload_3p_3(C11, v_C2, v_C11, v_C19, local_tIn_t[t]);
                                vload_3p_3(C0, v_C5, v_C0, v_C22, local_tIn_t[t]);
                                vload_3p_3(C16, v_C8, v_C16, v_C25, local_tIn_t[t]);
                                vload_3p_3(C12, v_C3, v_C12, v_C20, local_tIn_t[t]);
                                vload_3p_3(C14, v_C6, v_C14, v_C23, local_tIn_t[t]);
                                vload_3p_3(C17, v_C9, v_C17, v_C26, local_tIn_t[t]);
                                break;
                            default:
                                break;
                            }
                            v_res = v_c0 * v_C0 + v_c1 * v_C1 + v_c2 * v_C2 + v_c3 * v_C3 + v_c4 * v_C4 + v_c5 * v_C5 + v_c6 * v_C6 + v_c7 * v_C7 + v_c8 * v_C8 + v_c9 * v_C9 + v_c10 * v_C10 + v_c11 * v_C11 + v_c12 * v_C12 + v_c13 * v_C13 + v_c14 * v_C14 + v_c15 * v_C15 + v_c16 * v_C16 + v_c17 * v_C17 + v_c18 * v_C18 + v_c19 * v_C19 + v_c20 * v_C20 + v_c21 * v_C21 + v_c22 * v_C22 + v_c23 * v_C23 + v_c24 * v_C24 + v_c25 * v_C25 + v_c26 * v_C26;
                            simd_storeu(v_res, &tOut_ptr[x]);
                            // tOut_ptr[x] = c0 * local_tIn_t[t][C0] + c1 * local_tIn_t[t][C1] + c2 * local_tIn_t[t][C2] + c3 * local_tIn_t[t][C3] + c4 * local_tIn_t[t][C4] + c5 * local_tIn_t[t][C5] + c6 * local_tIn_t[t][C6] + c7 * local_tIn_t[t][C7] + c8 * local_tIn_t[t][C8] + c9 * local_tIn_t[t][C9] + c10 * local_tIn_t[t][C10] + c11 * local_tIn_t[t][C11] + c12 * local_tIn_t[t][C12] + c13 * local_tIn_t[t][C13] + c14 * local_tIn_t[t][C14] + c15 * local_tIn_t[t][C15] + c16 * local_tIn_t[t][C16] + c17 * local_tIn_t[t][C17] + c18 * local_tIn_t[t][C18] + c19 * local_tIn_t[t][C19] + c20 * local_tIn_t[t][C20] + c21 * local_tIn_t[t][C21] + c22 * local_tIn_t[t][C22] + c23 * local_tIn_t[t][C23] + c24 * local_tIn_t[t][C24] + c25 * local_tIn_t[t][C25] + c26 * local_tIn_t[t][C26];
                            total_float_ops += 53 * 4;
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
                        total_put_size += (load_size - 2 * R * DIMT);
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
                            // if (_MYID == 1)
                            //     printf("right_x_cal_j: %d\n", right_x_cal_j[j] - i_local);
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
                            // if (_MYID == 0)
                            //     printf("%d %d %d\n", i_local, blockSize_x, begin);
                            j = begin - aligned;
                            simd_load(v_p, &local_tIn_0[j]);
                            REG_PUTR(v_p, col_comm_CPE);
                            REG_GETR(v_g);
                            for (k = 3; j + k >= begin; k--)
                                local_tIn_0[j + k] = ((DEFINED_DATATYPE *)(&v_g))[k];
                            for (j = begin - aligned + 4; j + 4 <= right_x_block_comm; j += 4)
                            {
                                // begin % 4 == 0 ? simd_load(v_p, &local_tIn_0[j]) : simd_loadu(v_p, &local_tIn_0[j]);

                                simd_load(v_p, &local_tIn_0[j]);
                                REG_PUTR(v_p, col_comm_CPE);
                                REG_GETR(v_g);
                                // local_tIn_0[j] = ((DEFINED_DATATYPE *)(&v_g))[0];
                                // local_tIn_0[j + 1] = ((DEFINED_DATATYPE *)(&v_g))[1];
                                // local_tIn_0[j + 2] = ((DEFINED_DATATYPE *)(&v_g))[2];
                                // local_tIn_0[j + 3] = ((DEFINED_DATATYPE *)(&v_g))[3];
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
                        // athread_syn(ARRAY_SCOPE, 0xffff);
                    }
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

    spe_param->slave_float_ops[_MYID] = total_float_ops;
    spe_param->slave_get_size[_MYID] = total_get_size;
    spe_param->slave_put_size[_MYID] = total_put_size;
}
