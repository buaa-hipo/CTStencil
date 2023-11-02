#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Error: Generate 2D/3D data.\n";
        exit(1);
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int nz = 1;
    if (argc == 1 + 3)
        nz = atoi(argv[3]);

    stringstream ss;
    argc == 1 + 2 ? ss << nx << "_" << ny << "_double" : ss << nx << "_" << ny << "_" << nz << "_double";
    ofstream outf(ss.str().c_str(), ios::out | ios::trunc);

    srand(time(NULL));
    // outf.precision(6);
    // outf.setf(ios::fixed);
    for (int i = 0; i < nx * ny * nz; i++)
    {
        outf << ((double)rand() / (double)RAND_MAX) << endl;
    }
    outf.close();
    cout << "Generated data: " << ss.str() << "\n";
}
