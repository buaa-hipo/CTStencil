#define FILEPATH "../../data/helmholtz/128_128_128_double"
#define NUMTIMESTEPS 4
#define NUMCOLS 128
#define NUMROWS 128
#define LAYERS 128

#define R 2
#define NUMPOINTS 13
#define MAX_THREADS 64

#define DOUBLE
#define DEFINED_DATATYPE double
#define DEFINED_V_DATATYPE doublev4

// #define DEBUG

#define SIZE (LAYERS * NUMROWS * NUMCOLS)
#define STR_SIZE (256)

struct spe_parameter
{
    DEFINED_DATATYPE **temp;
    DEFINED_DATATYPE a, b, h2inv;
    int out;
    int nt, nz, ny, nx;
    int DIMT, DIMX, DIMY, DIMZ;
    int NTH;
};
