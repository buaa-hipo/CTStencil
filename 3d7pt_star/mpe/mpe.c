#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

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

#define FILEPATH "../../data/3d/128_128_128_double"
#define NUMTIMESTEPS 4
#define NUMCOLS 128
#define NUMROWS 128
#define LAYERS 128

#define R 1
#define NUMPOINTS 27

#define DEFINED_DATATYPE double
// #define DEBUG

#define SIZE (LAYERS * NUMROWS * NUMCOLS)
#define STR_SIZE (256)

DEFINED_DATATYPE c[NUMPOINTS] = {1. / 16807., -1. / 7., 1. / 7., -2. / 7., 2. / 7., -4. / 7., 4. / 7., -8. / 49., 8. / 49., -16. / 49., 16. / 49., -32. / 49., 32. / 49.,
                                 -64. / 343., 64. / 343., -128. / 343., 128. / 343., -256. / 343., 256. / 343., -512. / 2401., 512. / 2401., -1024. / 2401., 1024. / 2401.,
                                 -2048. / 2401., 2048. / 2401., -4096. / 16807., 4096. / 16807.};

void printConfig(int nt, int nx, int ny, int nz);
void genRandomCoeffients_debug(DEFINED_DATATYPE *c, int num_points);
void readinput(DEFINED_DATATYPE *vect, int layers, int grid_rows, int grid_cols, char *file);
void fatal(char *s);
int computeTempCPE(DEFINED_DATATYPE *temp[2], int nt, int nz, int ny, int nx);
int computeTempMPE(DEFINED_DATATYPE *temp[2], int nt, int nz, int ny, int nx);
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

#ifdef DEBUG
    genRandomCoeffients_debug(c, NUMPOINTS);
#endif

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
    int CPE_out = computeTempCPE(tempCPE, NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS);
    gettimeofday(&stop, NULL);
    time = (stop.tv_usec - start.tv_usec) + (stop.tv_sec - start.tv_sec) * 1e6;
    printf("Time: %.3f (us)\n", time);
    int MPE_out = computeTempMPE(tempMPE, NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS);
#ifndef DEBUG
    DEFINED_DATATYPE acc = accuracy(tempCPE[CPE_out], tempMPE[MPE_out], NUMTIMESTEPS, LAYERS, NUMROWS, NUMCOLS);
    // writeoutput(tempCPE[CPE_out], tempMPE[MPE_out], LAYERS, NUMROWS, NUMCOLS);
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

int computeTempCPE(DEFINED_DATATYPE *temp[2], int nt, int nz, int ny, int nx)
{
    DEFINED_DATATYPE c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7], c8 = c[8], c9 = c[9],
                     c10 = c[10], c11 = c[11], c12 = c[12], c13 = c[13], c14 = c[14], c15 = c[15], c16 = c[16], c17 = c[17], c18 = c[18],
                     c19 = c[19], c20 = c[20], c21 = c[21], c22 = c[22], c23 = c[23], c24 = c[24], c25 = c[25], c26 = c[26];

    int t, z, y, x;
    int in = 1, out = 0, tmp;
    int C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25, C26;
    int nyx = ny * nx;
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
                    C0 = z * nyx + y * nx + x;
                    C1 = C0 - nyx - nx - 1;
                    C2 = C1 + nyx;
                    C3 = C2 + nyx;
                    C4 = C0 - nyx - 1;
                    C5 = C4 + nyx;
                    C6 = C5 + nyx;
                    C7 = C0 - nyx + nx - 1;
                    C8 = C7 + nyx;
                    C9 = C8 + nyx;
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
                    temp[out][C0] = c0 * temp[in][C0] + c1 * temp[in][C1] + c2 * temp[in][C2] + c3 * temp[in][C3] + c4 * temp[in][C4] + c5 * temp[in][C5] + c6 * temp[in][C6] + c7 * temp[in][C7] + c8 * temp[in][C8] + c9 * temp[in][C9] + c10 * temp[in][C10] + c11 * temp[in][C11] + c12 * temp[in][C12] + c13 * temp[in][C13] + c14 * temp[in][C14] + c15 * temp[in][C15] + c16 * temp[in][C16] + c17 * temp[in][C17] + c18 * temp[in][C18] + c19 * temp[in][C19] + c20 * temp[in][C20] + c21 * temp[in][C21] + c22 * temp[in][C22] + c23 * temp[in][C23] + c24 * temp[in][C24] + c25 * temp[in][C25] + c26 * temp[in][C26];
                }
            }
        }
    }
    return out;
}

int computeTempMPE(DEFINED_DATATYPE *temp[2], int nt, int nz, int ny, int nx)
{
    DEFINED_DATATYPE c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7], c8 = c[8], c9 = c[9],
                     c10 = c[10], c11 = c[11], c12 = c[12], c13 = c[13], c14 = c[14], c15 = c[15], c16 = c[16], c17 = c[17], c18 = c[18],
                     c19 = c[19], c20 = c[20], c21 = c[21], c22 = c[22], c23 = c[23], c24 = c[24], c25 = c[25], c26 = c[26];

    int t, z, y, x;
    int in = 1, out = 0, tmp;
    int C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25, C26;
    int nyx = ny * nx;
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
                    C0 = z * nyx + y * nx + x;
                    C1 = C0 - nyx - nx - 1;
                    C2 = C1 + nyx;
                    C3 = C2 + nyx;
                    C4 = C0 - nyx - 1;
                    C5 = C4 + nyx;
                    C6 = C5 + nyx;
                    C7 = C0 - nyx + nx - 1;
                    C8 = C7 + nyx;
                    C9 = C8 + nyx;
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
                    temp[out][C0] = c0 * temp[in][C0] + c1 * temp[in][C1] + c2 * temp[in][C2] + c3 * temp[in][C3] + c4 * temp[in][C4] + c5 * temp[in][C5] + c6 * temp[in][C6] + c7 * temp[in][C7] + c8 * temp[in][C8] + c9 * temp[in][C9] + c10 * temp[in][C10] + c11 * temp[in][C11] + c12 * temp[in][C12] + c13 * temp[in][C13] + c14 * temp[in][C14] + c15 * temp[in][C15] + c16 * temp[in][C16] + c17 * temp[in][C17] + c18 * temp[in][C18] + c19 * temp[in][C19] + c20 * temp[in][C20] + c21 * temp[in][C21] + c22 * temp[in][C22] + c23 * temp[in][C23] + c24 * temp[in][C24] + c25 * temp[in][C25] + c26 * temp[in][C26];
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
