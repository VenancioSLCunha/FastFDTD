#include <chrono>
#include <pybind11/pybind11.h>
#include "solver_matriz.h"
#include "solver_func.h"
#include "fdtd_timestep.cu"

using namespace std::chrono;
using namespace std;

struct FastFDTD {

    Tensor E;
    Tensor H;
    Tensor J;

    Params params;

    //element_matriz arr2;

    FastFDTD() {}

    // Setters for parameters
    void setGridSize(int s_0, int s_1, 
      double*Ex,double*Ey,double*Ez,
      double*Hx,double*Hy,double*Hz,
      double*Jx,double*Jy,double*Jz) {

        make_elemento(E,Ex,Ey,Ez,s_0,s_1);
        make_elemento(H,Hx,Hy,Hz,s_0,s_1);
        make_elemento(J,Jx,Jy,Jz,s_0,s_1);
        //arr2.make_elemento(s_0, s_1);
    }

    void setSimulationParameters(double dx, double dt) {

        //constantes 
        double mu0 = 4 * M_PI * 1e-7;
        double eps0 = 8.85e-12;

        params.Hc = (1 / mu0) * (dt / dx);
        params.Ec = (1 / eps0) * (dt / dx);
        params.Jc = (1 / eps0) * dt;
        params.t = 0; 
    }

    void setSimulationParameters(double dx, double dt, double di, double dj, double omega, double tau) {

        double mu0 = 4 * M_PI * 1e-7;
        double eps0 = 8.85e-12;

        params.Hc = (1 / mu0) * (dt / dx);
        params.Ec = (1 / eps0) * (dt / dx);
        params.Jc = (1 / eps0) * dt;
        params.omega = omega;
        params.tau = tau;
        params.t = 0;
        params.dt = dt;
        params.source_i = di;
        params.source_j = dj;
    }

    void setSource(double di, double dj, double omega, double tau){

        params.omega = omega;
        params.tau = tau;
        params.dt = dt;
        params.source_i = di;
        params.source_j = dj;
    }

    /*void setDetector(){

    }*/

    /*void setObject(){

    }*/

    /*void setBoundary(){

    }*/
    
    py::array_t<double> run(int timesteps) {

        int blockSize = 512; //128-1024
        int numBlocks = (E.Hx.length + blockSize - 1) / blockSize;

        E.CopyToDevice();
        H.CopyToDevice();
        J.CopyToDevice();

        for (int i = 0; i < timesteps; i++) {

            auto start = high_resolution_clock::now();

            timestepE<<<numBlocks, blockSize>>>(E, H, J, params);
            timestepH<<<numBlocks, blockSize>>>(E, H, J, params);

            params.t += params.dt;

            auto stop = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(stop - start);
        }

        E.CopyToHost();
        H.CopyToHost();
        J.CopyToHost();

        result_ptr = py::array_t<double>(E.Hx.length);
        auto result_data = result_ptr.mutable_data();
        
        for (int i = 0; i < E.Hx.length; i++) {
            result_data[i] = E.Hx.h_data[i];
        }
    }

    py::array_t<double> getResult() const {
        return result_ptr;
    }
    ~ FastFDTD() {
        destruct(E);
        destruct(H);
        destruct(J);
    }
};
