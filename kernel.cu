
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <ctime>

#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <cmath>

#include "gpuErrchk.h"
#include "sphere.h"
#include "prism.h"
#include "pyramid.h"
#include "icosahedron.h"
#include "hitableList.h"
#include "camera.h"
#include "material.h"

/// TODO
/// - test rectangular prism and maybe implement triangular and rectangular pyramid (best method to test that refraction is working properly)
/// - optimize literally everything, from loops to memory access (probably memory bound, but also look at instruction cost and count, especially loops)
/// - implement the diffuse material
/// - figure out how to use openGL and directx and windows to display image in real time (and manipulate it)
/// - look at how to implement on ray-tracing cores and optimize for this purpose
/// - rework vector so that color is part of the object (???)
/// - rework vector so that brightness of vector changes each time an interaction occurs (??? already done?)
/// - rework coloring so that result is given by value of vector rather than static grey (???)

#define DIV_ROUNDUP(a, b) (((a) + (b) - 1)/(b))

using namespace std;

__device__ vec3 color(const ray& r, hitable **world, curandState *rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for(int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0, 0, 0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			//float t = 0.5 * (unit_direction.y() + 1.0);
			//vec3 c = (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			float t = unit_direction.y();
			//vec3 c = (t > 0.95) ? vec3(1.0, 1.0, 1.0) : (1.0 - t) * vec3(0.25, 0.25, 0.25) + t * vec3(0.125, 0.175, 0.25);
			vec3 c = (t > 0.95) ? vec3(1.0, 1.0, 1.0) : vec3(0.0, 0.0, 0.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0);
}

__global__ void render_init(const int nx, const int ny, const int offset, curandState* d_rand_state) {
	int IND = threadIdx.x + blockIdx.x * blockDim.x;
	int INDoffset = IND + offset;
	if (INDoffset >= nx * ny) return;
	curandState * local_rand_state = &d_rand_state[INDoffset];
	curand_init(1984, INDoffset, 0, local_rand_state);
}

__global__ void render(vec3* fb, const int nx, const int ny, const int ns, const int nsi, camera** cam, float gamma, hitable ** world, curandState * rand_state) {
	int COL = blockIdx.x * blockDim.x + threadIdx.x;
	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	if (ROW >= ny || COL >= nx) return;
	int IND = ROW * nx + COL;
	ray r;
	curandState * local_rand_state = &rand_state[IND];
	vec3 col = fb[ROW * nx + COL];
	for (int s = nsi; s < nsi+ns; s++) {
		float u = float(COL + curand_uniform(local_rand_state)) / float(nx);
		float v = float(ROW + curand_uniform(local_rand_state)) / float(ny);
		r = (*cam)->get_ray(u, v, local_rand_state);
		vec3 col_int = color(r, world, local_rand_state); // TODO could place this at the end for a single computation, might have use here later on though
		col += (pow(col_int, gamma)-col)/float(s+1);
	}
	fb[ROW * nx + COL] = col;
}

#define NUM_OBJS 4

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, const int nx, const int ny) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_list[0] = new sphere(vec3(0, -100.5, 0), 100, new lambertian(vec3(0.8, 0.8, 0.8)));
		d_list[1] = new sphere(vec3(0.5, 0, -2), 0.5, new polish(vec3(0.8, 0.3, 0.3), 1.7));
		d_list[2] = new sphere(vec3(-0.5, 0, -2), 0.5, new lambertian(vec3(0.6, 0.2, 0.8)));
		d_list[3] = new icosahedron(vec3(0, 0, -1), vec3(0, 1, 0), vec3(0, 0, 1), 0.5, new dielectric(1.5));
		*d_world = new hitable_list(d_list, NUM_OBJS);
		vec3 lookfrom = vec3(0, 0, 2);
		vec3 lookat = vec3(0, 0, -1);
		float dist_to_focus = (lookfrom - lookat).length();
		float aperture = 0.0;
		*d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 45, float(nx) / float(ny), aperture, dist_to_focus);
	}
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
	delete ((sphere*)d_list[0])->mat_ptr;
	delete ((sphere*)d_list[1])->mat_ptr;
	delete ((sphere*)d_list[2])->mat_ptr;
	delete ((tetrahedral*)d_list[3])->mat_ptr;
	delete d_list[0];
	delete d_list[1];
	delete d_list[2];
	delete d_list[3];
	delete *d_world;
	delete* d_camera;
}

int main() {
	gpuErrchk(cudaDeviceReset());
	gpuErrchk(cudaSetDeviceFlags(cudaDeviceScheduleYield));

	int maxBlocksize2d = 32;
	size_t* val = new(size_t);
	cudaDeviceGetLimit(val, cudaLimitStackSize);
	fprintf(stderr, "stack limit: %d\n", *val);
	*val = 32768;
	gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, *val));
	fprintf(stderr, "set stack limit to %d\n", *val);

	int nx = 1920;
	int ny = 1080;
	int ns = 1000;
	float gamma = 0.5;

	int num_pixels = nx * ny;
	int num_iterations = DIV_ROUNDUP(DIV_ROUNDUP(ns, 100) * num_pixels, 512*256);
	int ns_per_iteration = ns / num_iterations;

	size_t fb_size = num_pixels * sizeof(vec3);
	vec3* fb_h;
	gpuErrchk(cudaMallocHost(&fb_h, fb_size));
	vec3* fb_d;
	gpuErrchk(cudaMalloc(&fb_d, fb_size));

	curandState* d_rand_state;
	gpuErrchk(cudaMalloc(&d_rand_state, num_pixels * sizeof(curandState)));

	int list_length = NUM_OBJS;
	size_t list_size = list_length * sizeof(hitable *);
	hitable** d_list;
	gpuErrchk(cudaMalloc(&d_list, list_size));
	hitable** d_world;
	gpuErrchk(cudaMalloc(&d_world, sizeof(hitable *)));
	camera** d_camera;
	gpuErrchk(cudaMalloc(&d_camera, sizeof(camera*)));
	create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny);
	gpuErrchk(cudaDeviceSynchronize());

	clock_t begin1 = clock();

	dim3 threadsPerBlockI(num_pixels, 1);
	dim3 blocksPerGridI(1, 1);
	if (threadsPerBlockI.x > maxBlocksize2d * maxBlocksize2d) {
		threadsPerBlockI.x = maxBlocksize2d * maxBlocksize2d;
		blocksPerGridI.x = (num_pixels + threadsPerBlockI.x - 1) / threadsPerBlockI.x;
	}
	int blocksItemp = blocksPerGridI.x;
	if (blocksPerGridI.x > 16) {
		blocksPerGridI.x = 16;
	}
	fprintf(stderr, "launching init kernel <<<16, %d>>> %d times\n", threadsPerBlockI.x, DIV_ROUNDUP(blocksItemp, 16));
	for (int i = 0; i < blocksItemp; i += 16) {
		if (blocksItemp - i < 16)
			blocksPerGridI.x = blocksItemp - i;
		render_init <<<blocksPerGridI, threadsPerBlockI>>> (nx, ny, i * maxBlocksize2d * maxBlocksize2d, d_rand_state);
		gpuErrchk(cudaDeviceSynchronize());
		fprintf(stderr, "%d/%d\n", i, blocksItemp);
	}
	gpuErrchk(cudaDeviceSynchronize());
	clock_t end1 = clock();
	clock_t begin2 = clock();
	dim3 threadsPerBlock(nx, ny);
	dim3 blocksPerGrid(1, 1);
	if (threadsPerBlock.x > maxBlocksize2d) {
		threadsPerBlock.x = maxBlocksize2d;
		blocksPerGrid.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
	}
	if (threadsPerBlock.y > maxBlocksize2d) {
		threadsPerBlock.y = maxBlocksize2d;
		blocksPerGrid.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
	}
	fprintf(stderr, "launching full kernel <<<(%d, %d), (%d, %d)>>> %d times\n", blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y, num_iterations);
	for (int iteration = 0; iteration < num_iterations; iteration++) {
		render<<<blocksPerGrid, threadsPerBlock>>>(fb_d, nx, ny, ns_per_iteration, ns_per_iteration*iteration, d_camera, gamma, d_world, d_rand_state);
		// TODO got here, memory issue, likely in accessing pointers within the prism hit function
		gpuErrchk(cudaDeviceSynchronize());
		fprintf(stderr, "%d/%d\n", iteration, num_iterations);
	}
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(fb_h, fb_d, fb_size, cudaMemcpyDeviceToHost));

	clock_t end2 = clock();

	ofstream outfile;
	outfile.open("result.ppm");
	outfile << "P3\n";
	outfile << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			int pixel_index = j * nx + i;
			vec3 col = fb_h[pixel_index];
			vec3 col_gamma = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
			int ir = int(255.99 * col_gamma[0]);
			int ig = int(255.99 * col_gamma[1]);
			int ib = int(255.99 * col_gamma[2]);
			outfile << ir << " " << ig << " " << ib << "\n";
		}
	}
	outfile.close();

	clock_t end = clock();
	double elapsed_secs1 = double(end1 - begin1) / CLOCKS_PER_SEC * 1000.0;
	double elapsed_secs2 = double(end2 - begin2) / CLOCKS_PER_SEC * 1000.0;
	printf("init time: %gms\n", elapsed_secs1);
	printf("kernel time: %gms\n", elapsed_secs2);

	free_world <<<1, 1 >>> (d_list, d_world, d_camera);
	gpuErrchk(cudaFree(d_list));
	gpuErrchk(cudaFree(d_world));
	gpuErrchk(cudaFree(d_camera));
	gpuErrchk(cudaFree(d_rand_state));
	gpuErrchk(cudaFreeHost(fb_h));
	gpuErrchk(cudaFree(fb_d));

	gpuErrchk(cudaDeviceReset());
	return 0;
}
