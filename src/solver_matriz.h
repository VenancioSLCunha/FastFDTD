import "solver_func.h"

struct element_matriz {

    int size_0;
    int size_1;

    int length;
    
    double* h_data;
    double* d_data;

    void CopyToHost() {
        cudaMemcpy(h_data, d_data, length * sizeof(double), cudaMemcpyDeviceToHost);
    }

    void CopyToDevice() {
        cudaMemcpy(d_data, h_data, length * sizeof(double), cudaMemcpyHostToDevice);
    }
};

struct Tensor {

  element_matriz Hx;
  element_matriz Ey;
  element_matriz Hz;

  void CopyToHost(){

    Hx.CopyToHost();
    Ey.CopyToHost();
    Hz.CopyToHost();
  }
  void CopyToDevice(){

    Hx.CopyToDevice();
    Ey.CopyToDevice();
    Hz.CopyToDevice();
  }
};

__device__
void converte_index(const element_matriz &arr, const int raveled_index, int &i,int &j) {
  
    i= raveled_index / arr.size_1;
    j= raveled_index % arr.size_1;
}

void make_elemento(element_matriz &arr, double* data, int size_0, int size_1) {
  arr.size_0=size_0;
  arr.size_1=size_1;
  arr.length=size_0*size_1;
  
  cudaMalloc(&arr.d_data,arr.length*sizeof(double));
  
  arr.h_data = data;
}

void make_elemento(Tensor& tensor, double*xdata,double*ydata,double*zdata,int size_0,int size_1) {

  make_elemento(tensor.Hx, xdata, size_0, size_1);
  make_elemento(tensor.Ey, ydata, size_0, size_1);
  make_elemento(tensor.Hz, zdata, size_0, size_1);
}
