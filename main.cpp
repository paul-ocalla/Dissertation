#include <mpi.h>
#include <iostream>
#include "heat_eqn.hpp"

int main(int argc, char* argv[]) {

	/*
		Commands:

				Compile: mpic++ -o A1 heat_eqn.cpp main.cpp -lm -std=c++11
				
				Run:     mpirun -np <Num_processors> ./A1
	*/
    HeatSimulation heat(argc, argv);
    heat.run();
    return 0;
}
