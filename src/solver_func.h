struct Param {

  double Hc;
  double Jc;
  double Ec;

  double t;

  int source_i;
  int source_j;
  double tau; // duration of excitation
  double omega; // frequency of excitation
  double dt; // frequency of excitation
};


void copyToDevice(element_matriz &array) {
  cudaMemcpy(array.d_data, array.h_data, array.length*sizeof(double), cudaMemcpyHostToDevice);
}

void copyToHost(element_matriz &array) {
  cudaMemcpy(array.h_data, array.d_data, array.length*sizeof(double), cudaMemcpyDeviceToHost);
}

void destruct(element_matriz &array){
  cudaFree(array.d_data);
}

void destruct(Tensor& tensor)
{
  destruct(tensor.Hx);
  destruct(tensor.Ey);
  destruct(tensor.Hz);
}

__device__
double Get(const element_matriz array, int i_0, int i_1) {
    
    return array.d_data[i_0 * array.tam_1 + i_1];
}

__device__
void Set(element_matriz array, int i_0, int i_1, double value) {

    array.d_data[i_0 * array.tam_1 + i_1] = value;
}

__device__
double Get(const element_matriz array, int i_0, int i_1) {

    return array.d_data[i_0 * array.tam_1 + i_1];
}
