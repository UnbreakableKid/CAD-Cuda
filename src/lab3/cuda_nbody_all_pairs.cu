/**
 * Hervé Paulino
 * tD7Ymjz5m$$RrxFN
 */

#include <nbody/cuda_nbody_all_pairs.h>

static constexpr int thread_block_size = 512;

static constexpr int nStreams = 3;

namespace cadlabs {

    cuda_nbody_all_pairs::cuda_nbody_all_pairs(
            const int number_particles,
            const float t_final,
            const unsigned number_of_threads,
            const universe_t universe,
            const unsigned universe_seed) :
            nbody(number_particles, t_final, universe, universe_seed),
            number_blocks((number_particles + thread_block_size - 1) / thread_block_size) {


        cudaMalloc((void **) &gpu_particles, number_particles * sizeof(particle_t));
    }

    cuda_nbody_all_pairs::~cuda_nbody_all_pairs() {
        cudaFree(gpu_particles);
    }

#pragma region
#if __CUDA_ARCH__ < 600

    __device__ double atomicAdd(double *address, double val) {
        unsigned long long int *address_as_ull =
                (unsigned long long int *) address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                                 __longlong_as_double(assumed)));

            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
    }

#endif
    __device__ static double atomicMax(double *address, double val) {
        unsigned long long int *address_as_i = (unsigned long long int *) address;
        unsigned long long int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                              __double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
        } while (assumed != old);
        return __longlong_as_double(old);
    }

    __device__ static float atomicMax(float *address, float val) {
        int *address_as_i = (int *) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                              __float_as_int(::fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }
#pragma endregion

    //! Function With paralelized adding, using global memory and atomic adds, takes too long
    //! \param particles
    //! \param number_particles
    //! \param pi
    __global__ void addingWithGlobalMemory(particle_t *particles, const unsigned number_particles, particle_t *pi) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < number_particles) {

            particle_t *pj = &particles[index];

            double x_sep, y_sep, dist_sq, grav_base;
            x_sep = pj->x_pos - pi->x_pos;
            y_sep = pj->y_pos - pi->y_pos;
            dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

            /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
            grav_base = GRAV_CONSTANT * (pi->mass) * (pj->mass) / dist_sq;

            double x = grav_base * x_sep;
            double y = grav_base * y_sep;

            atomicAdd(&(pi->x_force), x);
            atomicAdd(&(pi->y_force), y);
        }
    }

    __global__ void test(particle_t *particles, const unsigned number_particles, particle_t *pi) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int lindex = threadIdx.x;

        extern __shared__ particle_t temp[]; //make sure every thread has loaded

        temp[lindex] = particles[index];
        __syncthreads();

        if (index < number_particles) {

            particle_t *pj = &temp[lindex];

            double x_sep, y_sep, dist_sq, grav_base;
            x_sep = pj->x_pos - pi->x_pos;
            y_sep = pj->y_pos - pi->y_pos;
            dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

            /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
            grav_base = GRAV_CONSTANT * (pi->mass) * (pj->mass) / dist_sq;

            double x = grav_base * x_sep;
            double y = grav_base * y_sep;

            atomicAdd(&(pi->x_force), x);
            atomicAdd(&(pi->y_force), y);

        }
    }

    __global__ void nbody_kernel(particle_t *particles, const unsigned number_particles, int offset) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < number_particles) {

            particle_t *pi = &particles[index]; //fazer em mem partilhada

            pi->x_force = 0;
            pi->y_force = 0;

            for (int j = 0; j < number_particles; j++) {
                particle_t *p = &particles[j];
                /* compute the force of particle j on particle i */
                double x_sep, y_sep, dist_sq, grav_base;

                x_sep = p->x_pos - pi->x_pos;
                y_sep = p->y_pos - pi->y_pos;
                dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

                /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
                grav_base = GRAV_CONSTANT * (pi->mass) * (p->mass) / dist_sq;

                pi->x_force += grav_base * x_sep;
                pi->y_force += grav_base * y_sep;            }

        }

    }

    void cuda_nbody_all_pairs::calculate_forces() {
        /* First calculate force for particles. */

        particle_t *out;
        cudaMalloc((void **) &out, number_particles * sizeof(particle_t));


        cudaStream_t stream1, stream2, stream3;
        cudaStreamCreate ( &stream1) ;
        cudaStreamCreate ( &stream2) ;
        cudaStreamCreate ( &stream3) ;

        cudaStream_t stream[nStreams] = {stream1, stream2, stream3};

        int streamSize =number_particles;

        for (int i = 0; i < nStreams; ++i) {
            int offset = i * streamSize;
            cudaMemcpyAsync(&gpu_particles[offset], &particles[offset], number_particles * sizeof(particle_t), cudaMemcpyHostToDevice, stream[i]);
            nbody_kernel<<<streamSize/number_blocks, thread_block_size, 0, stream[i]>>>( gpu_particles, number_particles,offset);
            cudaMemcpyAsync(&particles[offset], &gpu_particles[offset], number_particles * sizeof(particle_t), cudaMemcpyDeviceToHost, stream[i]);
        }

//        cudaMemcpy(gpu_particles, particles, number_particles * sizeof(particle_t), cudaMemcpyHostToDevice);
//        nbody_kernel<<<number_blocks, thread_block_size>>>(gpu_particles, number_particles, out);
//
//        for (int i = 0; i <number_particles; i++){
//
//            test<<<number_blocks, thread_block_size>>>(gpu_particles, number_particles, &gpu_particles[i]);
//        }
//
//        cudaMemcpy(particles, gpu_particles, number_particles * sizeof(particle_t), cudaMemcpyDeviceToHost);

    }


    //@ statistics variables
    __device__ double sum_speed_sq = 0;
    __device__ double max_acc = 0;
    __device__ double max_speed = 0;

//    __global__ void all_pairs_kernel(particle_t *particles, const int number_particles, double step) {
//
//        int index = blockIdx.x * blockDim.x + threadIdx.x;
//
////        printf("TEMP = %f\n", step);
//
//        if (index < number_particles) {
//            particle_t *pi = &particles[index];
//
////            printf("Here 0 y_vel  %f\n", step);
//
//            pi->x_pos += (pi->x_vel) * step;
//            pi->y_pos += (pi->y_vel) * step;
//            double x_acc = pi->x_force / pi->mass;
//            double y_acc = pi->y_force / pi->mass;
//
//            pi->x_vel += x_acc * step;
//            pi->y_vel += y_acc * step;
//
//
//            /* compute statistics */
//            double cur_acc = (x_acc * x_acc + y_acc * y_acc);
//            cur_acc = sqrt(cur_acc);
//            double speed_sq = (pi->x_vel) * (pi->x_vel) + (pi->y_vel) * (pi->y_vel);
//            double cur_speed = sqrt(speed_sq);
//        }
//    }

    void cuda_nbody_all_pairs::move_all_particles(double step) {

       nbody::move_all_particles(step);
    }

    void cuda_nbody_all_pairs::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }

}

