/*
 ============================================================================
 Name        : rle.cu
 Author      : Witaut Bajaryn
 Version     :
 Copyright   : Copyright (c) 2017 Witaut Bajaryn
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <string>

#include "hemi/hemi.h"
#include "hemi/kernel.h"
#include "hemi/parallel_for.h"

#define CUB_STDERR // For CubDebugExit
#include "cub/util_allocator.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/device/device_run_length_encode.cuh"

using in_elt_t = int;

#define BUILD_NUMBER 14

#define CUB_RLE_MAX_WORKING_INPUT_PIECE_SIZE (274ull * 1024 * 1024)
#define GPU_RLE_MAX_WORKING_INPUT_PIECE_SIZE (237ull * 1024 * 1024)

template<typename elt_t>
struct array
{
	elt_t *data;
	size_t size; // the number of elt_t elements in data

	static array<elt_t> new_on_device(size_t size)
	{
		array<elt_t> d_result{nullptr, size};
		d_result.cudaMalloc();
		return d_result;
	}

	static array<elt_t> vector_view_on_host(std::vector<elt_t> &v)
	{
		return array<elt_t>{v.data(), v.size()};
	}

	array<elt_t> subview(size_t offset, size_t subview_size)
	{
		size_t result_size = std::min(subview_size, size - offset);
		return array<elt_t>{data + offset, result_size};
	}

	elt_t &operator[](const size_t i)
	{
		return data[i];
	}

	void cudaMalloc()
	{
		checkCuda(::cudaMalloc(&data, size * sizeof(*data)));
	}

	void cudaFree()
	{
		checkCuda(::cudaFree(data));
	}
};

// From https://erkaman.github.io/posts/cuda_rle.html
int cpuRLEImpl(const in_elt_t *in, int n, in_elt_t* symbolsOut, int* countsOut)
{
	if (n == 0)
		return 0; // nothing to compress!

	int outIndex = 0;
	in_elt_t symbol = in[0];
	int count = 1;

	for (int i = 1; i < n; ++i) {
		if (in[i] != symbol) {
			// run is over.
			// So output run.
			symbolsOut[outIndex] = symbol;
			countsOut[outIndex] = count;
			outIndex++;

			// and start new run:
			symbol = in[i];
			count = 1;
		} else {
			++count; // run is not over yet.
		}
	}

	// output last run.
	symbolsOut[outIndex] = symbol;
	countsOut[outIndex] = count;
	outIndex++;

	return outIndex;
}

void cpuRLE(
		array<in_elt_t> in,
		std::vector<in_elt_t> &out_symbols,
		std::vector<int> &out_counts,
		int &out_end)
{
	out_end = cpuRLEImpl(in.data, in.size,
			out_symbols.data(),
			out_counts.data());
}

void inclusive_prefix_sum(array<uint8_t> d_in, array<int> d_out)
{
    cub::CachingDeviceAllocator allocator(true);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // Estimate temp_storage_bytes
    CubDebugExit(cub::DeviceScan::InclusiveSum(
    		d_temp_storage, temp_storage_bytes,
    		d_in.data, d_out.data, d_in.size,
    		0, true));
    CubDebugExit(allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    CubDebugExit(cudaPeekAtLastError());
    hemi::deviceSynchronize();
    // Run
    std::cout << "Running prefix sum kernel" << std::endl;
    auto err = (cub::DeviceScan::InclusiveSum(
    		d_temp_storage, temp_storage_bytes,
    		d_in.data, d_out.data, d_in.size,
    		0, true));
    std::cerr << cudaGetErrorString(err) << std::endl;
    checkCuda(err);
    CubDebugExit(err);
    std::cout << "Done" << std::endl;
}

void deviceRLE(
		array<in_elt_t> d_in,
		array<in_elt_t> d_out_symbols,
		array<int> d_out_counts,
		array<int> d_end)
{
	// Idea: https://erkaman.github.io/posts/cuda_rle.html

	auto d_backward_mask = array<uint8_t>::new_on_device(d_in.size);
	hemi::parallel_for(0, d_backward_mask.size, [=] HEMI_LAMBDA(size_t i) {
		if (i == 0) {
			d_backward_mask.data[i] = 1;
			return;
		}
		d_backward_mask.data[i] = d_in.data[i] != d_in.data[i - 1];
	});

	auto d_scanned_backward_mask = array<int>::new_on_device(d_in.size);
	inclusive_prefix_sum(d_backward_mask, d_scanned_backward_mask);

	auto d_compacted_backward_mask = array<int>::new_on_device(d_in.size + 1);
	hemi::parallel_for(0, d_in.size, [=] HEMI_LAMBDA(size_t i) {
		if (i == 0) {
			d_compacted_backward_mask.data[i] = 0;
			return;
		}

		size_t out_pos = d_scanned_backward_mask.data[i] - 1;

		if (i == d_in.size - 1) {
			*d_end.data = out_pos + 1;
			d_compacted_backward_mask.data[out_pos + 1] = i + 1;
		}

		// or if (d_scanned_backward_mask.data[i] !=
		//        d_scanned_backward_mask.data[i - 1])
		if (d_backward_mask.data[i])
			d_compacted_backward_mask.data[out_pos] = i;
	});

	// Not hemi::parallel_for because d_end is only on the device now.
	hemi::launch([=] HEMI_LAMBDA() {
		for (size_t i: hemi::grid_stride_range(0, *d_end.data)) {
			int current = d_compacted_backward_mask.data[i];
			int right = d_compacted_backward_mask.data[i + 1];
			d_out_counts.data[i] = right - current;
			d_out_symbols.data[i] = d_in.data[current];
		}
	});

	// Call it explicitly
	// to make the profiler not show other calls as taking a long time.
	hemi::deviceSynchronize();

	d_compacted_backward_mask.cudaFree();
	d_scanned_backward_mask.cudaFree();
	d_backward_mask.cudaFree();
}

void cubDeviceRLE(
		array<in_elt_t> d_in,
		array<in_elt_t> d_out_symbols,
		array<int> d_out_counts,
		array<int> d_end)
{
	array<uint8_t> d_temp_storage{nullptr, 0};
	// Estimate d_temp_storage.size
	CubDebugExit(cub::DeviceRunLengthEncode::Encode(
			d_temp_storage.data, d_temp_storage.size,
			d_in.data,
			d_out_symbols.data, d_out_counts.data, d_end.data, d_in.size));
	d_temp_storage.cudaMalloc();
	CubDebugExit(cub::DeviceRunLengthEncode::Encode(
			d_temp_storage.data, d_temp_storage.size,
			d_in.data,
			d_out_symbols.data, d_out_counts.data, d_end.data, d_in.size));

	hemi::deviceSynchronize();
}

void gpuRLE(
		array<in_elt_t> in,
		std::vector<in_elt_t> &out_symbols,
		std::vector<int> &out_counts,
		int &out_end,
		bool use_cub_impl = false)
{
	auto d_in = array<in_elt_t>::new_on_device(in.size);
	auto d_out_symbols = array<in_elt_t>::new_on_device(in.size);
	auto d_out_counts = array<int>::new_on_device(in.size);
	auto d_end = array<int>::new_on_device(1);

	checkCuda(cudaMemcpy(d_in.data, in.data,
			d_in.size * sizeof(*d_in.data),
			cudaMemcpyHostToDevice));

	if (use_cub_impl)
		cubDeviceRLE(d_in, d_out_symbols, d_out_counts, d_end);
	else
		deviceRLE(d_in, d_out_symbols, d_out_counts, d_end);

	checkCuda(cudaMemcpy(out_symbols.data(), d_out_symbols.data,
			out_symbols.size() * sizeof(*out_symbols.data()),
			cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(out_counts.data(), d_out_counts.data,
			out_counts.size() * sizeof(*out_counts.data()),
			cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(&out_end, d_end.data,
			sizeof(out_end),
			cudaMemcpyDeviceToHost));

	d_in.cudaFree();
	d_out_symbols.cudaFree();
	d_out_counts.cudaFree();
	d_end.cudaFree();
}

void run_rle_impl(
		array<in_elt_t> in,
		std::vector<in_elt_t> &out_symbols,
		std::vector<int> &out_counts,
		int &out_end,
		bool use_cpu_impl,
		bool use_cub_impl)
{
	if (use_cpu_impl)
		cpuRLE(in, out_symbols, out_counts, out_end);
	else
		gpuRLE(in, out_symbols, out_counts, out_end, use_cub_impl);
}

void append_partial_result(
		std::vector<in_elt_t> &out_symbols,
		std::vector<int> &out_counts,
		std::vector<in_elt_t> &full_out_symbols,
		std::vector<int> &full_out_counts)
{
	size_t offset = 0;

	if (full_out_symbols.size() > 0 && out_symbols.size() > 0) {
		std::cout << "append_partial_result: Piecing together needed"
				  << std::endl;

		size_t prev_full_end{out_symbols.size() - 1};
		if (full_out_symbols[prev_full_end] == out_symbols[0]) {
			full_out_counts[prev_full_end] += out_counts[0];
			offset = 1;
		}
	}

	std::cout << "append_partial_result: Copying partial result" << std::endl;
	std::copy(out_symbols.begin() + offset, out_symbols.end(),
			std::back_inserter(full_out_symbols));
	std::copy(out_counts.begin() + offset, out_counts.end(),
			std::back_inserter(full_out_counts));
}

void rle(
		std::vector<in_elt_t> &in_owner,
		std::vector<in_elt_t> &full_out_symbols,
		std::vector<int> &full_out_counts,
		size_t piece_size,
		bool use_cpu_impl,
		bool use_cub_impl)
{
	array<in_elt_t> full_in = array<in_elt_t>::vector_view_on_host(in_owner);

	for (size_t start = 0; start < in_owner.size(); start += piece_size) {
		array<in_elt_t> in = full_in.subview(start, piece_size);
		std::cout << "Partial in start: " << start
				  << ", size: " << in.size << std::endl;

		// TODO Could actually be allocated once
		std::vector<in_elt_t> out_symbols(in.size);
		std::vector<int> out_counts(in.size);
		int end{0};

		run_rle_impl(in, out_symbols, out_counts, end,
				use_cpu_impl, use_cub_impl);

		out_symbols.resize(end);
		out_counts.resize(end);

		append_partial_result(out_symbols, out_counts,
				full_out_symbols, full_out_counts);
	}
}

bool verify_rle(
		std::vector<in_elt_t> &in,
		std::vector<in_elt_t> &out_symbols,
		std::vector<int> &out_counts)
{
	std::vector<in_elt_t> decompressed{};
	for (size_t i = 0; i < out_symbols.size(); i++)
		for (int j = 0; j < out_counts[i]; j++)
			decompressed.push_back(out_symbols[i]);

	if (decompressed.size() != in.size()) {
		std::cout << "Uncompressed output size (" << decompressed.size()
				  << ") != input size (" << in.size() << ")." << std::endl;
		return false;
	}

	for (size_t i = 0; i < decompressed.size(); i++)
		if (decompressed[i] != in[i]) {
			std::cout << "Uncompressed output element[" << i
					  << "] (" << decompressed[i]
					  << ") != input element (" << in[i] << ")." << std::endl;
			return false;
		}
	return true;
}

std::vector<in_elt_t> generate_input(size_t size)
{
	std::vector<in_elt_t> result{};

	int run_count = 1;
	int run_value = 0;
	int run_i = 0;
	for (size_t i = 0; i < size; i++) {
		result.push_back(run_value);

		run_i++;

		if (run_i >= run_count) {
			run_count++;
			run_value++;
			run_i = 0;
		}
	}

	return result;
}

void parse_args(
		int argc,
		char *argv[],
		size_t *input_size,
		size_t *input_piece_size,
		bool *use_cpu_impl,
		bool *use_cub_impl)
{
	int opt;

	while ((opt = getopt(argc, argv, "cus:p:")) != -1) {
		switch (opt) {
		case 'c':
			*use_cpu_impl = true;
			break;
		case 'u':
			*use_cub_impl = true;
			*use_cpu_impl = false;
			break;
		case 's':
			*input_size = atoll(optarg);
			break;
		case 'p':
			*input_piece_size = atoll(optarg);
			break;
		default:
			fprintf(stderr, "Usage: %s [-c|-u] [-s input_size] [-p input_piece_size]\n", argv[0]);
			exit(EXIT_FAILURE);
		}
	}
}

void assert_safe_piece_size(
		size_t piece_size,
		size_t max_safe_piece_size,
		std::string impl_name)
{
	if (piece_size > max_safe_piece_size) {
		std::cout << "Compressing the input by pieces of more than "
				  << max_safe_piece_size << " elements requested. "
				  << impl_name << " produces incorrect results"
				  << " or runs out of memory for such inputs."
				  << " Terminating without running anything." << std::endl;
		exit(EFBIG);
	}
}

int main(int argc, char *argv[])
{
	size_t input_size = 200llu * 1024 * 1024;
	size_t input_piece_size = 200llu * 1024 * 1024;
	bool use_cpu_impl = false;
	bool use_cub_impl = false;

	parse_args(argc, argv,
			&input_size, &input_piece_size,
			&use_cpu_impl, &use_cub_impl);

	std::cout << "Build " << BUILD_NUMBER << std::endl;
	std::cout << "Generating an input with " << input_size
			  << " elements (" << input_size * sizeof(in_elt_t) << " bytes)"
			  << std::endl;
	std::cout << "Will compress " << input_piece_size << " elements ("
			  << input_piece_size * sizeof(in_elt_t) << " bytes) at a time"
			  << std::endl;

	if (use_cpu_impl)
		std::cout << "Using the CPU implementation" << std::endl;
	else if (use_cub_impl)
		std::cout << "Using the Cub GPU implementation" << std::endl;
	else
		std::cout << "Using the GPU implementation" << std::endl;

	if (use_cub_impl)
		assert_safe_piece_size(
				input_piece_size, CUB_RLE_MAX_WORKING_INPUT_PIECE_SIZE,
				"CUB RLE");
	else if (!use_cpu_impl)
		assert_safe_piece_size(
				input_piece_size, GPU_RLE_MAX_WORKING_INPUT_PIECE_SIZE,
				"This GPU RLE implementation");

	std::vector<in_elt_t> in_owner = generate_input(input_size);

	std::vector<in_elt_t> out_symbols{};
	//out_symbols.reserve(in_owner.size());
	std::vector<int> out_counts{};
	//out_counts.reserve(in_owner.size());

	rle(in_owner, out_symbols, out_counts,
			input_piece_size,
			use_cpu_impl, use_cub_impl);

	/*
	std::cout << "[";
	for (int i = 0; i < out_symbols.size(); i++)
		std::cout << "(" << out_counts[i]
			 << ", " << out_symbols[i]
			 << "), ";
	std::cout << "]" << std::endl;
	*/

	std::cout << "Verifying the output" << std::endl;
	if (verify_rle(in_owner, out_symbols, out_counts))
		std::cout << "The output is correct." << std::endl;
	else
		std::cout << "The output is INCORRECT." << std::endl;

	return 0;
}
