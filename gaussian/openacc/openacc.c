#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

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

#define FILEPATH "../../data/gaussian/2048_2048_double"
#define NUMTIMESTEPS 4
#define NUMCOLS 2048
#define NUMROWS 2048

#define R 2
#define NUMPOINTS 25

#define DEFINED_DATATYPE double
// #define DEBUG

#define SIZE (NUMROWS * NUMCOLS)
#define STR_SIZE (256)

void printConfig(int nt, int nx, int ny);
void genRandomCoeffients_debug(DEFINED_DATATYPE *c, int num_points);
void readinput(DEFINED_DATATYPE *vect, int grid_rows, int grid_cols, char *file);
void fatal(char *s);
int computeTempCPE(DEFINED_DATATYPE **temp, int nt, int ny, int nx);
int computeTempMPE(DEFINED_DATATYPE *temp[2], int nt, int ny, int nx);
DEFINED_DATATYPE accuracy(DEFINED_DATATYPE *arr1, DEFINED_DATATYPE *arr2, int nt, int ny, int nx);
DEFINED_DATATYPE accuracy_debug(DEFINED_DATATYPE *arr1, DEFINED_DATATYPE *arr2, int nt, int ny, int nx, char *file);
void writeoutput(DEFINED_DATATYPE *vect, int grid_rows, int grid_cols, char *file);
void writeoutput_debug(DEFINED_DATATYPE *vect1, DEFINED_DATATYPE *vect2, int grid_rows, int grid_cols, char *file);

int main()
{
    char *tfile, *ofile, *afile;
    tfile = FILEPATH;
    ofile = "output.out";
    afile = "accuracy.out";

    printConfig(NUMTIMESTEPS, NUMCOLS, NUMROWS);

#ifdef DEBUG
    genRandomCoeffients_debug(c, NUMPOINTS);
#endif

    DEFINED_DATATYPE *tempMPE[2], *tempCPE[2];
    tempMPE[0] = (DEFINED_DATATYPE *)calloc(SIZE, sizeof(DEFINED_DATATYPE));
    tempMPE[1] = (DEFINED_DATATYPE *)calloc(SIZE, sizeof(DEFINED_DATATYPE));
    tempCPE[0] = malloc(SIZE * sizeof(DEFINED_DATATYPE));
    tempCPE[1] = (DEFINED_DATATYPE *)calloc(SIZE, sizeof(DEFINED_DATATYPE));

    readinput(tempMPE[0], NUMROWS, NUMCOLS, tfile);
    memcpy(tempCPE[0], tempMPE[0], sizeof(DEFINED_DATATYPE) * SIZE);

    struct timeval start, stop;
    DEFINED_DATATYPE time;
    gettimeofday(&start, NULL);
    int CPE_out = computeTempCPE(tempCPE, NUMTIMESTEPS, NUMROWS, NUMCOLS);
    gettimeofday(&stop, NULL);
    time = (stop.tv_usec - start.tv_usec) + (stop.tv_sec - start.tv_sec) * 1e6;
    printf("Time: %.3f (us)\n", time);
    int MPE_out = computeTempMPE(tempMPE, NUMTIMESTEPS, NUMROWS, NUMCOLS);
#ifndef DEBUG
    DEFINED_DATATYPE acc = accuracy(tempCPE[CPE_out], tempMPE[MPE_out], NUMTIMESTEPS, NUMROWS, NUMCOLS);
    // writeoutput(tempCPE[CPE_out], NUMROWS, NUMCOLS, ofile);
#else
    DEFINED_DATATYPE acc = accuracy_debug(tempCPE[CPE_out], tempMPE[MPE_out], NUMTIMESTEPS, NUMROWS, NUMCOLS, afile);
    writeoutput_debug(tempCPE[CPE_out], tempMPE[MPE_out], NUMROWS, NUMCOLS, ofile);
#endif
    printf("Accuracy: %e\n", acc);

    free(tempCPE[0]);
    free(tempCPE[1]);
    free(tempMPE[0]);
    free(tempMPE[1]);
}

void printConfig(int nt, int nx, int ny)
{
    printf("openacc\n");
    printf("NUMTIMESTEPS: %d\n", nt);
    printf("NUMCOLS: %d\n", nx);
    printf("NUMROWS: %d\n", ny);
}

void genRandomCoeffients_debug(DEFINED_DATATYPE *c, int num_points)
{
    srand(time(NULL));
    printf("Random coeffients: ");
    int i;
    for (i = 0; i < num_points; i++)
    {
        c[i] = (DEFINED_DATATYPE)rand() / (DEFINED_DATATYPE)RAND_MAX;
        printf("%f ", c[i]);
    }
    printf("\n");
}

void readinput(DEFINED_DATATYPE *vect, int grid_rows, int grid_cols, char *file)
{
    int i;
    FILE *fp;
    char str[STR_SIZE];
    DEFINED_DATATYPE val;

    if ((fp = fopen(file, "r")) == 0)
        fatal("The file was not opened");

    for (i = 0; i < grid_rows * grid_cols; i++)
    {
        if (fgets(str, STR_SIZE, fp) == NULL)
            fatal("Error reading file\n");
        if (feof(fp))
            fatal("not enough lines in file");
        if ((sscanf(str, "%lf", &val) != 1))
            fatal("invalid file format");
        vect[i] = val;
    }

    fclose(fp);
}

void fatal(char *s)
{
    fprintf(stderr, "Error: %s\n", s);
}

int computeTempCPE(DEFINED_DATATYPE **temp, int nt, int ny, int nx)
{
    int t, y, x;
    int in = 1, out = 0;
    for (t = 1; t <= nt; t++)
    {
        int tmp = in;
        in = out;
        out = tmp;
        DEFINED_DATATYPE *global_In = temp[in];
        DEFINED_DATATYPE *global_Out = temp[out];
#pragma acc parallel loop copyin(ny, nx, t, y, x) annotate(readonly(ny, nx, t, global_In, global_Out))
        for (y = R * t; y < ny - R * t; y++)
        {
            int C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24;
            for (x = R * t; x < nx - R * t; x++)
            {
                C2 = (y - 2) * nx + x;
                C0 = C2 - 2;
                C1 = C2 - 1;
                C3 = C2 + 1;
                C4 = C2 + 2;
                C7 = (y - 1) * nx + x;
                C5 = C7 - 2;
                C6 = C7 - 1;
                C8 = C7 + 1;
                C9 = C7 + 2;
                C12 = y * nx + x;
                C10 = C12 - 2;
                C11 = C12 - 1;
                C13 = C12 + 1;
                C14 = C12 + 2;
                C17 = (y + 1) * nx + x;
                C15 = C17 - 2;
                C16 = C17 - 1;
                C18 = C17 + 1;
                C19 = C17 + 2;
                C22 = (y + 2) * nx + x;
                C20 = C22 - 2;
                C21 = C22 - 1;
                C23 = C22 + 1;
                C24 = C22 + 2;
                global_Out[C12] = 2 * global_In[C0] + 4 * global_In[C1] + 5 * global_In[C2] + 4 * global_In[C3] + 2 * global_In[C4] + 4 * global_In[C5] + 9 * global_In[C6] + 12 * global_In[C7] + 9 * global_In[C8] + 4 * global_In[C9] + 5 * global_In[C10] + 12 * global_In[C11] + 15 * global_In[C12] + 12 * global_In[C13] + 5 * global_In[C14] + 4 * global_In[C15] + 9 * global_In[C16] + 12 * global_In[C17] + 9 * global_In[C18] + 4 * global_In[C19] + 2 * global_In[C20] + 4 * global_In[C21] + 5 * global_In[C22] + 4 * global_In[C23] + 2 * global_In[C24];
            }
        }
    }
    return out;
}

int computeTempMPE(DEFINED_DATATYPE *temp[2], int nt, int ny, int nx)
{
    int t, y, x;
    int in = 1, out = 0, tmp;
    int C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24;
    for (t = 1; t <= nt; t++)
    {
        tmp = in;
        in = out;
        out = tmp;
        for (y = R * t; y < ny - R * t; y++)
        {
            for (x = R * t; x < nx - R * t; x++)
            {
                C2 = (y - 2) * nx + x;
                C0 = C2 - 2;
                C1 = C2 - 1;
                C3 = C2 + 1;
                C4 = C2 + 2;
                C7 = (y - 1) * nx + x;
                C5 = C7 - 2;
                C6 = C7 - 1;
                C8 = C7 + 1;
                C9 = C7 + 2;
                C12 = y * nx + x;
                C10 = C12 - 2;
                C11 = C12 - 1;
                C13 = C12 + 1;
                C14 = C12 + 2;
                C17 = (y + 1) * nx + x;
                C15 = C17 - 2;
                C16 = C17 - 1;
                C18 = C17 + 1;
                C19 = C17 + 2;
                C22 = (y + 2) * nx + x;
                C20 = C22 - 2;
                C21 = C22 - 1;
                C23 = C22 + 1;
                C24 = C22 + 2;
                temp[out][C12] = 2 * temp[in][C0] + 4 * temp[in][C1] + 5 * temp[in][C2] + 4 * temp[in][C3] + 2 * temp[in][C4] + 4 * temp[in][C5] + 9 * temp[in][C6] + 12 * temp[in][C7] + 9 * temp[in][C8] + 4 * temp[in][C9] + 5 * temp[in][C10] + 12 * temp[in][C11] + 15 * temp[in][C12] + 12 * temp[in][C13] + 5 * temp[in][C14] + 4 * temp[in][C15] + 9 * temp[in][C16] + 12 * temp[in][C17] + 9 * temp[in][C18] + 4 * temp[in][C19] + 2 * temp[in][C20] + 4 * temp[in][C21] + 5 * temp[in][C22] + 4 * temp[in][C23] + 2 * temp[in][C24];
            }
        }
    }
    return out;
}

DEFINED_DATATYPE accuracy(DEFINED_DATATYPE *arr1, DEFINED_DATATYPE *arr2, int nt, int ny, int nx)
{
    DEFINED_DATATYPE err = 0.0;
    int y, x, C;
    for (y = R * nt; y < ny - R * nt; y++)
    {
        for (x = R * nt; x < nx - R * nt; x++)
        {
            C = y * nx + x;
            err += (arr1[C] - arr2[C]) * (arr1[C] - arr2[C]);
        }
    }
    return (DEFINED_DATATYPE)sqrt(err / ((ny - 2 * R * nt) * (nx - 2 * R * nt)));
}

DEFINED_DATATYPE accuracy_debug(DEFINED_DATATYPE *arr1, DEFINED_DATATYPE *arr2, int nt, int ny, int nx, char *file)
{
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");

    DEFINED_DATATYPE err = 0.0;
    int y, x, C;
    for (y = R * nt; y < ny - R * nt; y++)
    {
        for (x = R * nt; x < nx - R * nt; x++)
        {
            C = y * nx + x;
            err += (arr1[C] - arr2[C]) * (arr1[C] - arr2[C]);

            if ((arr1[C] - arr2[C]) * (arr1[C] - arr2[C]) > 0.0)
            {
                sprintf(str, "%d\t%d\t%f\n", y, x, arr1[C] - arr2[C]);
                fputs(str, fp);
            }
        }
    }

    fclose(fp);

    return (DEFINED_DATATYPE)sqrt(err / ((ny - 2 * R * nt) * (nx - 2 * R * nt)));
}

void writeoutput(DEFINED_DATATYPE *vect, int grid_rows, int grid_cols, char *file)
{
    int i, j, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");

    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++)
        {
            sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
            fputs(str, fp);
            index++;
        }

    fclose(fp);
}

void writeoutput_debug(DEFINED_DATATYPE *vect1, DEFINED_DATATYPE *vect2, int grid_rows, int grid_cols, char *file)
{
    int i, j, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");

    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++)
        {
            sprintf(str, "%d\t%f\t%f\t%f\n", index, vect1[i * grid_cols + j], vect2[i * grid_cols + j], vect1[i * grid_cols + j] - vect2[i * grid_cols + j]);
            fputs(str, fp);
            index++;
        }

    fclose(fp);
}
