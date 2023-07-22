#include <stdio.h>
#include <unistd.h>

__global__ void printNumber(int number)
{
  printf("%d\n", number);
}

int main()
{
  // https://stackoverflow.com/questions/26009152/is-there-a-way-to-dynamically-determine-the-number-of-cuda-streams
  cudaStream_t streams[5];

  for (int i = 0; i < 5; ++i)
  {
    cudaStreamCreate(&(streams[i]));
  }

  for (int i = 0; i < 5; ++i)
  {
    printNumber<<<1, 1, 0, streams[i]>>>(i);
  }

  for (int i = 0; i < 5; ++i)
  {
    cudaStreamDestroy(streams[i]);
  }

  cudaDeviceSynchronize();
}

