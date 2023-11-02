#include "common.h"
#include <assert.h>
#include <athread.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

extern void slave_spe_func(struct spe_parameter *spe_param);

void usage(int argc, char **argv);
void printConfig(char method, int nt, int nx, int ny, int nz, int DIMT, int DIMX, int DIMY, int DIMZ, int NTH);
void readinput(DEFINED_DATATYPE *vect, int layers, int grid_rows, int grid_cols, char *file);
void fatal(char *s);
int computeTempCPE(DEFINED_DATATYPE *temp[2], DEFINED_DATATYPE a, DEFINED_DATATYPE b, DEFINED_DATATYPE h2inv, int nt, int nz, int ny, int nx, int DIMT, int DIMX, int DIMY, int DIMZ, int NTH, int *slave_float_ops, int *slave_get_size, int *slave_put_size);
int computeTempMPE(DEFINED_DATATYPE *temp[2], DEFINED_DATATYPE a, DEFINED_DATATYPE b, DEFINED_DATATYPE h2inv, int nt, int nz, int ny, int nx);
DEFINED_DATATYPE accuracy(DEFINED_DATATYPE *arr1, DEFINED_DATATYPE *arr2, int nt, int nz, int ny, int nx);
DEFINED_DATATYPE accuracy_debug(DEFINED_DATATYPE *arr1, DEFINED_DATATYPE *arr2, int nt, int nz, int ny, int nx, char *file);
void writeoutput(DEFINED_DATATYPE *vect, int layers, int grid_rows, int grid_cols, char *file);
void writeoutput_debug(DEFINED_DATATYPE *vect1, DEFINED_DATATYPE *vect2, int layers, int grid_rows, int grid_cols, char *file);
void roofline_analysis(int *slave_float_ops, int *slave_get_size, int *slave_put_size, DEFINED_DATATYPE time);

int main(int argc, char **argv)
{
    if (argc != 6)
        usage(argc, argv);

    char method = argv[0][strlen(argv[0]) - 1];
    int DIMT = atoi(argv[1]);
    int DIMX = atoi(argv[2]);
    int DIMY = atoi(argv[3]);
    int DIMZ = atoi(argv[4]);
    int NTH = atoi(argv[5]);

    int slave_float_ops[MAX_THREADS] = {0};
    int slave_get_size[MAX_THREADS] = {0};
    int slave_put_size[MAX_THREADS] = {0};

    athread_init();

    char *tfile, *ofile, *afile;
    tfile = FILEPATH;
    ofile = "output.out";
    afile = "accuracy.out";

    printConfig(method, NUMTIMESTEPS, NUMCOLS, NUMROWS, LAYERS, DIMT, DIMX, DIMY, DIMZ, NTH);

    DEFINED_DATATYPE a = 0.12, b = 0.1, h2inv = 0.4;

    DEFINED_DATATYPE *tempMPE[2], *tempCPE[2];
    tempMPE[0] = (DEFINED_DATATYPE *)calloc(SIZE, sizeof(DEFINED_DATATYPE));
    tempMPE[1] = (DEFINED_DATATYPE *)calloc(SIZE, sizeof(DEFINED_DATATYPE));
    tempCPE[0] = malloc(SIZE * sizeof(DEFINED_DATATYPE));
    tempCPE[1] = (DEFINED_DATATYPE *)calloc(SIZE, sizeof(DEFINED_DATATYPE));

    readinput(tempMPE[0], LAYERS, NUMROWS, NUMCOLS, tfile);
    memcpy(tempCPE[0], tempMPE[0], sizeof(DEFINED_DATATYPE) * SIZE);

    int MPE_out = computeTempMPE(tempMPE, a, b, h2inv, NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS);
    struct timeval start, stop;
    DEFINED_DATATYPE time;
    DEFINED_DATATYPE total_time = 0.0;
    DEFINED_DATATYPE avg_time = 0.0;
    int i;
    int CPE_out;
    for (i = 0; i < NUMREPEATS; i++)
    {
        gettimeofday(&start, NULL);
        CPE_out = computeTempCPE(tempCPE, a, b, h2inv, NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS, DIMT, DIMX, DIMY, DIMZ, NTH, slave_float_ops, slave_get_size, slave_put_size);
        gettimeofday(&stop, NULL);
        time = ((stop.tv_usec - start.tv_usec) + (stop.tv_sec - start.tv_sec) * 1e6);
        total_time += time;
        if (i == 0)
        {
#ifndef DEBUG
            DEFINED_DATATYPE acc = accuracy(tempCPE[CPE_out], tempMPE[MPE_out], NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS);
            // writeoutput(tempCPE[CPE_out], LAYERS, NUMROWS, NUMCOLS, ofile);
#else
            DEFINED_DATATYPE acc = accuracy_debug(tempCPE[CPE_out], tempMPE[MPE_out], NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS, afile);
            writeoutput_debug(tempCPE[CPE_out], tempMPE[MPE_out], LAYERS, NUMROWS, NUMCOLS, ofile);
#endif
            printf("Accuracy: %e\n", acc);
        }
    }
    avg_time = total_time / NUMREPEATS;
    printf("Time: %.3f (us)\n", avg_time);

    roofline_analysis(slave_float_ops, slave_get_size, slave_put_size, avg_time);

    free(tempCPE[0]);
    free(tempCPE[1]);
    free(tempMPE[0]);
    free(tempMPE[1]);

    athread_halt();
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <the degree of temporal tiling> <block size along the x dimension> <block size along the y dimension> <block size along the z dimension>\n", argv[0]);
}

void printConfig(char method, int nt, int nx, int ny, int nz, int DIMT, int DIMX, int DIMY, int DIMZ, int NTH)
{
    switch (method)
    {
    case '0':
        printf("3D spatial tiling\n");
        break;
    case '1':
        printf("3.5D tiling(2.5D spatial tiling + 1D temporal tiling)\n");
        break;
    case '2':
        printf("3.5D tiling + double buffering\n");
        break;
    case '3':
        printf("3.5D tiling + double buffering + SIMD\n");
        break;
    default:
        break;
    }

    printf("NUMTIMESTEPS: %d\n", nt);
    printf("NUMCOLS: %d\n", nx);
    printf("NUMROWS: %d\n", ny);
    printf("LAYERS: %d\n", nz);
    printf("DIMT: %d\n", DIMT);
    printf("DIMX: %d\n", DIMX);
    printf("DIMY: %d\n", DIMY);
    printf("DIMZ: %d\n", DIMZ);
    printf("NTH: %d\n", NTH);
}

void readinput(DEFINED_DATATYPE *vect, int layers, int grid_rows, int grid_cols, char *file)
{
    int i, j, k;
    FILE *fp;
    char str[STR_SIZE];
    DEFINED_DATATYPE val;

    if ((fp = fopen(file, "r")) == 0)
        fatal("The file was not opened");

    for (i = 0; i <= grid_rows - 1; i++)
        for (j = 0; j <= grid_cols - 1; j++)
            for (k = 0; k <= layers - 1; k++)
            {
                if (fgets(str, STR_SIZE, fp) == NULL)
                    fatal("Error reading file\n");
                if (feof(fp))
                    fatal("not enough lines in file");
                if ((sscanf(str, "%lf", &val) != 1))
                    fatal("invalid file format");
                // i*grid_cols+j+k*grid_rows*grid_cols：？
                // j:col, i:row, k:layer
                vect[i * grid_cols + j + k * grid_rows * grid_cols] = val;
            }

    fclose(fp);
}

void fatal(char *s)
{
    fprintf(stderr, "Error: %s\n", s);
}

int computeTempCPE(DEFINED_DATATYPE *temp[2], DEFINED_DATATYPE a, DEFINED_DATATYPE b, DEFINED_DATATYPE h2inv, int nt, int nz, int ny, int nx, int DIMT, int DIMX, int DIMY, int DIMZ, int NTH, int *slave_float_ops, int *slave_get_size, int *slave_put_size)
{
    struct spe_parameter spe_param = {
        .temp = temp,
        .a = a,
        .b = b,
        .h2inv = h2inv,
        .nt = nt,
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .DIMT = DIMT,
        .DIMZ = DIMZ,
        .DIMY = DIMY,
        .DIMX = DIMX,
        .NTH = NTH,
        .slave_float_ops = slave_float_ops,
        .slave_get_size = slave_get_size,
        .slave_put_size = slave_put_size};

    athread_spawn(spe_func, &spe_param);
    athread_join();

    return spe_param.out;
}

int computeTempMPE(DEFINED_DATATYPE *temp[2], DEFINED_DATATYPE a, DEFINED_DATATYPE b, DEFINED_DATATYPE h2inv, int nt, int nz, int ny, int nx)
{
    DEFINED_DATATYPE c = b * h2inv * 0.0833;
    DEFINED_DATATYPE d = c * 1.0;
    DEFINED_DATATYPE e = c * 16.0;
    DEFINED_DATATYPE f = c * 90.0;

    int t, z, y, x;
    int in = 1, out = 0, tmp;
    int C, B1, B2, T1, T2, W1, W2, E1, E2, S1, S2, N1, N2;
    for (t = 1; t <= nt; t++)
    {
        tmp = in;
        in = out;
        out = tmp;
        for (z = R * t; z < nz - R * t; z++)
        {
            for (y = R * t; y < ny - R * t; y++)
            {
                for (x = R * t; x < nx - R * t; x++)
                {
                    C = x + y * nx + z * nx * ny;
                    W1 = C - 1;
                    W2 = C - 2;
                    E1 = C + 1;
                    E2 = C + 2;
                    N1 = C - nx;
                    N2 = C - nx * 2;
                    S1 = C + nx;
                    S2 = C + nx * 2;
                    B1 = C - nx * ny;
                    B2 = C - 2 * nx * ny;
                    T1 = C + nx * ny;
                    T2 = C + 2 * nx * ny;
                    temp[out][C] = (a - f) * temp[in][C] + e * temp[in][B1] + d * temp[in][B2] + e * temp[in][T1] + d * temp[in][T2] + e * temp[in][W1] + d * temp[in][W2] + e * temp[in][E1] + d * temp[in][E2] + e * temp[in][S1] + d * temp[in][S2] + e * temp[in][N1] + d * temp[in][N2];
                }
            }
        }
    }

    return out;
}

DEFINED_DATATYPE accuracy(DEFINED_DATATYPE *arr1, DEFINED_DATATYPE *arr2, int nt, int nz, int ny, int nx)
{
    DEFINED_DATATYPE err = 0.0;
    int z, y, x, c;
    for (z = R * nt; z < nz - R * nt; z++)
    {
        for (y = R * nt; y < ny - R * nt; y++)
        {
            for (x = R * nt; x < nx - R * nt; x++)
            {
                c = x + y * nx + z * nx * ny;
                err += (arr1[c] - arr2[c]) * (arr1[c] - arr2[c]);
            }
        }
    }

    return (DEFINED_DATATYPE)sqrt(err / ((nz - 2 * R * nt) * (ny - 2 * R * nt) * (nx - 2 * R * nt)));
}

DEFINED_DATATYPE accuracy_debug(DEFINED_DATATYPE *arr1, DEFINED_DATATYPE *arr2, int nt, int nz, int ny, int nx, char *file)
{
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");

    DEFINED_DATATYPE err = 0.0;
    int z, y, x, c;
    for (z = R * nt; z < nz - R * nt; z++)
    {
        for (y = R * nt; y < ny - R * nt; y++)
        {
            for (x = R * nt; x < nx - R * nt; x++)
            {
                c = x + y * nx + z * nx * ny;
                err += (arr1[c] - arr2[c]) * (arr1[c] - arr2[c]);

                if ((arr1[c] - arr2[c]) * (arr1[c] - arr2[c]) > 1e-12)
                {
                    sprintf(str, "%d\t%d\t%d\t%d\t%f\n", c, z, y, x, arr1[c] - arr2[c]);
                    fputs(str, fp);
                }
            }
        }
    }

    fclose(fp);

    return (DEFINED_DATATYPE)sqrt(err / ((nz - 2 * R * nt) * (ny - 2 * R * nt) * (nx - 2 * R * nt)));
}

void writeoutput(DEFINED_DATATYPE *vect, int layers, int grid_rows, int grid_cols, char *file)
{
    int i, j, k, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");

    for (i = 0; i < layers; i++)
    {
        for (j = 0; j < grid_rows; j++)
        {
            for (k = 0; k < grid_cols; k++)
            {
                sprintf(str, "%d\t%g\n", index, vect[i * grid_rows * grid_cols + j * grid_cols + k]);
                fputs(str, fp);
                index++;
            }
        }
    }

    fclose(fp);
}

void writeoutput_debug(DEFINED_DATATYPE *vect1, DEFINED_DATATYPE *vect2, int layers, int grid_rows, int grid_cols, char *file)
{
    int i, j, k, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");

    for (i = 0; i < layers; i++)
    {
        for (j = 0; j < grid_rows; j++)
        {
            for (k = 0; k < grid_cols; k++)
            {
                sprintf(str, "%d\t%f\t%f\t%f\n", index, vect1[i * grid_rows * grid_cols + j * grid_cols + k], vect2[i * grid_rows * grid_cols + j * grid_cols + k], vect1[i * grid_rows * grid_cols + j * grid_cols + k] - vect2[i * grid_rows * grid_cols + j * grid_cols + k]);
                fputs(str, fp);
                index++;
            }
        }
    }

    fclose(fp);
}

void roofline_analysis(int *slave_float_ops, int *slave_get_size, int *slave_put_size, DEFINED_DATATYPE time)
{
    int i, total_float_ops = 0, total_get_size = 0, total_put_size = 0;

    for (i = 0; i < MAX_THREADS; i++)
        total_float_ops += slave_float_ops[i];
    printf("Total float operations: %d\n", total_float_ops);
    for (i = 0; i < MAX_THREADS; i++)
        total_get_size += slave_get_size[i];
    printf("Total get size: %d\n", total_get_size);
    for (i = 0; i < MAX_THREADS; i++)
        total_put_size += slave_put_size[i];
    printf("Total put size: %d\n", total_put_size);
    printf("Arithmetic Intensity: %f\n", total_float_ops * 1.0 / ((total_get_size + total_put_size) * 8));

    printf("Core group performance: %fGFLOPs\n", total_float_ops * 1.0 / time * 1e-3);
}
