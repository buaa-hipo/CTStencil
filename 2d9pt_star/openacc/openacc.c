#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

/*
	double a[M][N];
	double b[M][N];
	double c0, c1, c2, c3, c4, c5, c6, c7, c8;

	for (long k = 1; k < M - 1; ++k)
	{
		for (long j = 1; j < N - 1; ++j)
		{
			b[k][j] = c0 * a[k][j] + c1 * a[k][j - 1] + c2 * a[k][j - 2]
					+ c3 * a[k][j + 1]  + c4 * a[k][j + 2] + c5 * a[k - 1][j] 
					+ c6 * a[k - 2][j] + c7 * a[k + 1][j] + c8 * a[k + 2][j];
		}
	}
*/

#define FILEPATH "../../data/2d/2048_2048_double"
#define NUMTIMESTEPS 4
#define NUMCOLS 2048
#define NUMROWS 2048

#define R 2
#define NUMPOINTS 9

#define DEFINED_DATATYPE double
// #define DEBUG

#define SIZE (NUMROWS * NUMCOLS)
#define STR_SIZE (256)

// 	const DEFINED_DATATYPE c0 = -30. / 12., c1 = 16. / 12., c2 = - 1. / 12.; /*差分系数*/
DEFINED_DATATYPE c[NUMPOINTS] = {-30. / 12., 16. / 12., -1. / 12., 16. / 12., -1. / 12., 16. / 12., -1. / 12., 16. / 12., -1. / 12.};

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
    DEFINED_DATATYPE c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7], c8 = c[8];
    int t, y, x;
    int in = 1, out = 0;
    for (t = 1; t <= nt; t++)
    {
        int tmp = in;
        in = out;
        out = tmp;
        DEFINED_DATATYPE *global_In = temp[in];
        DEFINED_DATATYPE *global_Out = temp[out];
#pragma acc parallel loop copyin(ny, nx, t, y, x, c0, c1, c2, c3, c4, c5, c6, c7, c8) annotate(readonly(ny, nx, t, c0, c1, c2, c3, c4, c5, c6, c7, c8, global_In, global_Out))
        for (y = R * t; y < ny - R * t; y++)
        {
            int C, N1, N2, W1, W2, S1, S2, E1, E2;
            for (x = R * t; x < nx - R * t; x++)
            {
                C = y * nx + x;
                N1 = C - nx;
                N2 = C - nx * 2;
                W1 = C - 1;
                W2 = C - 2;
                S1 = C + nx;
                S2 = C + nx * 2;
                E1 = C + 1;
                E2 = C + 2;
                global_Out[C] = c0 * global_In[C] + c1 * global_In[W1] + c2 * global_In[W2] + c3 * global_In[E1] + c4 * global_In[E2] + c5 * global_In[N1] + c6 * global_In[N2] + c7 * global_In[S1] + c8 * global_In[S2];
            }
        }
    }
    return out;
}

int computeTempMPE(DEFINED_DATATYPE *temp[2], int nt, int ny, int nx)
{
    DEFINED_DATATYPE c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7], c8 = c[8];

    int t, y, x;
    int in = 1, out = 0, tmp;
    int C, N1, N2, W1, W2, S1, S2, E1, E2;
    for (t = 1; t <= nt; t++)
    {
        tmp = in;
        in = out;
        out = tmp;
        for (y = R * t; y < ny - R * t; y++)
        {
            for (x = R * t; x < nx - R * t; x++)
            {
                C = y * nx + x;
                N1 = C - nx;
                N2 = C - nx * 2;
                W1 = C - 1;
                W2 = C - 2;
                S1 = C + nx;
                S2 = C + nx * 2;
                E1 = C + 1;
                E2 = C + 2;
                temp[out][C] = c0 * temp[in][C] + c1 * temp[in][W1] + c2 * temp[in][W2] + c3 * temp[in][E1] + c4 * temp[in][E2] + c5 * temp[in][N1] + c6 * temp[in][N2] + c7 * temp[in][S1] + c8 * temp[in][S2];
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

            if (abs(arr1[C] - arr2[C]) > 0.0)
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