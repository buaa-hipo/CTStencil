#define FILEPATH "../../data/3d/128_128_128_double"
#define NUMTIMESTEPS 4
#define NUMCOLS 128
#define NUMROWS 128
#define LAYERS 128

#define R 1
#define NUMPOINTS 27
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
    int out;
    int nt, nz, ny, nx;
    int DIMT, DIMX, DIMY, DIMZ;
    int NTH;
};
