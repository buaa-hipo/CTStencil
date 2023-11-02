#define FILEPATH "../../data/gaussian/2048_2048_double"
#define NUMTIMESTEPS 4
#define NUMCOLS 2048
#define NUMROWS 2048

#define R 2
#define NUMPOINTS 25
#define MAX_THREADS 64

#define DOUBLE
#define DEFINED_DATATYPE double
#define DEFINED_V_DATATYPE doublev4

// #define DEBUG

#define SIZE (NUMROWS * NUMCOLS)
#define STR_SIZE (256)

struct spe_parameter
{
    DEFINED_DATATYPE **temp;
    int out;
    int nt, nx, ny;
    int DIMT, DIMX, DIMY;
    int NTH;
};
