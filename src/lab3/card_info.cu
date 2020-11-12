#include <container_ops.h>


int main(int argc, char **argv) {
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                cadLog("No CUDA GPU has been detected");
                return -1;
            } else if (deviceCount == 1) {

                cadLog("There is 1 device supporting CUDA");
            } else {
                cadLog("There are " << deviceCount << " devices supporting CUDA");
            }
        }

        cadLog("Device " << dev << " name: " << deviceProp.name);
        cadLog(" Computational Capabilities: " << deviceProp.major << "." << deviceProp.minor);
        cadLog(" Maximum global memory size: " << deviceProp.totalGlobalMem);
        cadLog(" Maximum constant memory size: " << deviceProp.totalConstMem);
        cadLog(" Maximum shared memory size per block: " << deviceProp.sharedMemPerBlock);
        cadLog(" Maximum block dimensions: " << deviceProp.maxThreadsDim[0]
                 << " x " << deviceProp.maxThreadsDim[1]
                 << " x " << deviceProp.maxThreadsDim[2]);
        cadLog(" Maximum grid dimensions: " << deviceProp.maxGridSize[0]
                 << " x " << deviceProp.maxGridSize[1]
                 << " x " << deviceProp.maxGridSize[2]);
        cadLog(" Warp size: " << deviceProp.warpSize);

        return 0;
    }

    return 0;
}

