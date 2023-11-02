#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define FILEPATH "../../data/helmholtz/128_128_128_double"
#define NUMTIMESTEPS 4
#define NUMCOLS 128
#define NUMROWS 128
#define LAYERS 128

#define R 2
#define NUMPOINTS 13

#define DEFINED_DATATYPE double
// #define DEBUG

#define SIZE (LAYERS * NUMROWS * NUMCOLS)
#define STR_SIZE (256)

void printConfig(int nt, int nx, int ny, int nz);
void readinput(DEFINED_DATATYPE *vect, int layers, int grid_rows, int grid_cols, char *file);
void fatal(char *s);
int computeTempCPE(DEFINED_DATATYPE *temp[2], DEFINED_DATATYPE a, DEFINED_DATATYPE b, DEFINED_DATATYPE h2inv, int nt, int nz, int ny, int nx);
int computeTempMPE(DEFINED_DATATYPE *temp[2], DEFINED_DATATYPE a, DEFINED_DATATYPE b, DEFINED_DATATYPE h2inv, int nt, int nz, int ny, int nx);
DEFINED_DATATYPE accuracy(DEFINED_DATATYPE *arr1, DEFINED_DATATYPE *arr2, int nt, int nz, int ny, int nx);
DEFINED_DATATYPE accuracy_debug(DEFINED_DATATYPE *arr1, DEFINED_DATATYPE *arr2, int nt, int nz, int ny, int nx, char *file);
void writeoutput(DEFINED_DATATYPE *vect, int layers, int grid_rows, int grid_cols, char *file);
void writeoutput_debug(DEFINED_DATATYPE *vect1, DEFINED_DATATYPE *vect2, int layers, int grid_rows, int grid_cols, char *file);

int main()
{
    char *tfile, *ofile, *afile;
    tfile = FILEPATH;
    ofile = "output.out";
    afile = "accuracy.out";

    printConfig(NUMTIMESTEPS, NUMCOLS, NUMROWS, LAYERS);

    DEFINED_DATATYPE a = 0.12, b = 0.1, h2inv = 0.4;

    DEFINED_DATATYPE *tempMPE[2], *tempCPE[2];
    tempMPE[0] = (DEFINED_DATATYPE *)calloc(SIZE, sizeof(DEFINED_DATATYPE));
    tempMPE[1] = (DEFINED_DATATYPE *)calloc(SIZE, sizeof(DEFINED_DATATYPE));
    tempCPE[0] = malloc(SIZE * sizeof(DEFINED_DATATYPE));
    tempCPE[1] = (DEFINED_DATATYPE *)calloc(SIZE, sizeof(DEFINED_DATATYPE));

    readinput(tempMPE[0], LAYERS, NUMROWS, NUMCOLS, tfile);
    memcpy(tempCPE[0], tempMPE[0], sizeof(DEFINED_DATATYPE) * SIZE);

    struct timeval start, stop;
    DEFINED_DATATYPE time;
    gettimeofday(&start, NULL);
    int CPE_out = computeTempCPE(tempCPE, a, b, h2inv, NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS);
    gettimeofday(&stop, NULL);
    time = (stop.tv_usec - start.tv_usec) + (stop.tv_sec - start.tv_sec) * 1e6;
    printf("Time: %.3f (us)\n", time);
    int MPE_out = computeTempMPE(tempMPE, a, b, h2inv, NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS);
#ifndef DEBUG
    DEFINED_DATATYPE acc = accuracy(tempCPE[CPE_out], tempMPE[MPE_out], NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS);
    // writeoutput(tempCPE[CPE_out], tempMPE[MPE_out], LAYERS, NUMROWS, NUMCOLS, ofile);
#else
    DEFINED_DATATYPE acc = accuracy_debug(tempCPE[CPE_out], tempMPE[MPE_out], NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS, afile);
    writeoutput_debug(tempCPE[CPE_out], tempMPE[MPE_out], LAYERS, NUMROWS, NUMCOLS, ofile);
#endif
    printf("Accuracy: %e\n", acc);

    free(tempCPE[0]);
    free(tempCPE[1]);
    free(tempMPE[0]);
    free(tempMPE[1]);
}

void printConfig(int nt, int nx, int ny, int nz)
{
    printf("MPE\n");
    printf("NUMTIMESTEPS: %d\n", nt);
    printf("NUMCOLS: %d\n", nx);
    printf("NUMROWS: %d\n", ny);
    printf("LAYERS: %d\n", nz);
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

int computeTempCPE(DEFINED_DATATYPE *temp[2], DEFINED_DATATYPE a, DEFINED_DATATYPE b, DEFINED_DATATYPE h2inv, int nt, int nz, int ny, int nx)
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

                if ((arr1[c] - arr2[c]) * (arr1[c] - arr2[c]) > 0.0)
                {
                    sprintf(str, "%d\t%d\t%d\t%f\n", z, y, x, arr1[c] - arr2[c]);
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
