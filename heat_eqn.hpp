#ifndef HEAT_EQN_HPP
#define HEAT_EQN_HPP

#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <mpi.h>


class HeatSimulation {
public:
    HeatSimulation(int argc, char* argv[]);
    void run();

private:

    int id;
	int nprocs, nworkers;
	int source, dest;
	int up, down, right, left;
	int nxLocal, nyLocal;
	int message, rc;
    double tstart, tend;

	int nx = 200;
	int ny = 200;
	int time_steps = 10000;
	int MASTER = 0;
	int BEGIN = 1;
	int END = 2;
	int COMM = 3;
    MPI_Request request[8];
    MPI_Datatype colA, colB, rowA, rowB, matrixA, matrixB;

	double t_diff = 0.0001;
	double width = 2 * 3.1416;
	double height = 2 * 3.1416;
	double x_diff = width / nx;
	double y_diff = height / ny;

    std::vector<int> dims;
    std::vector<double> grid;
    std::vector<double> A;
    std::vector<double> B;


    void initialise_grid(int x, int y);

    void update(int x, int y, int i, int j, int t, std::vector<double>& A, std::vector<double>& B, int id);

    void assign_boundaries(int x, int y, int t, std::vector<double>& A, std::vector<double>& B, int id);

    void print_to_file(int x, int y, int n);

    double f(int x, int y, int t);

    double initial_heat(int x, int y, int t);
};

#endif // HEAT_EQN_HPP
