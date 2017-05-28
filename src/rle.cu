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
#include <vector>

#include "hemi/hemi.h"
#include "hemi/kernel.h"
#include "hemi/parallel_for.h"

using in_elt_t = int;

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

void serialRLE(
		std::vector<in_elt_t> &in,
		std::vector<in_elt_t> &out_symbols,
		std::vector<int> &out_counts,
		int &out_end)
{
	out_end = cpuRLEImpl(in.data(), in.size(),
			out_symbols.data(),
			out_counts.data());
}

void gpuRLE(
		std::vector<in_elt_t> &in,
		std::vector<in_elt_t> &out_symbols,
		std::vector<int> &out_counts,
		int &out_end)
{
	auto d_in = array<in_elt_t>::new_on_device(in.size());
	auto d_out_symbols = array<in_elt_t>::new_on_device(in.size());
	auto d_out_counts = array<int>::new_on_device(in.size());
	auto d_end = array<int>::new_on_device(1);

	checkCuda(cudaMemcpy(d_in.data, in.data(),
			d_in.size * sizeof(*d_in.data),
			cudaMemcpyHostToDevice));

	// Idea: https://erkaman.github.io/posts/cuda_rle.html

	auto d_backward_mask = array<uint8_t>::new_on_device(in.size());
	hemi::parallel_for(0, d_backward_mask.size, [=] HEMI_LAMBDA(size_t i) {
		if (i == 0) {
			d_backward_mask.data[i] = 1;
			return;
		}
		d_backward_mask.data[i] = d_in.data[i] != d_in.data[i - 1];
	});

	auto d_scanned_backward_mask = array<int>::new_on_device(in.size());
	hemi::parallel_for(0, d_in.size, [=] HEMI_LAMBDA(size_t i) {
		d_scanned_backward_mask.data[i] = 0;
		for (size_t j = 0; j <= i; j++)
			d_scanned_backward_mask.data[i] += d_backward_mask.data[j];
	});

	auto d_compacted_backward_mask = array<int>::new_on_device(in.size() + 1);
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

	checkCuda(cudaMemcpy(&out_end, d_end.data,
			sizeof(out_end),
			cudaMemcpyDeviceToHost));

	hemi::parallel_for(0, out_end, [=] HEMI_LAMBDA(size_t i) {
		int current = d_compacted_backward_mask.data[i];
		int right = d_compacted_backward_mask.data[i + 1];
		d_out_counts.data[i] = right - current;
		d_out_symbols.data[i] = d_in.data[current];
	});

	d_compacted_backward_mask.cudaFree();
	d_scanned_backward_mask.cudaFree();
	d_backward_mask.cudaFree();

	checkCuda(cudaMemcpy(out_symbols.data(), d_out_symbols.data,
			out_symbols.size() * sizeof(*out_symbols.data()),
			cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(out_counts.data(), d_out_counts.data,
			out_counts.size() * sizeof(*out_counts.data()),
			cudaMemcpyDeviceToHost));

	d_in.cudaFree();
	d_out_symbols.cudaFree();
	d_out_counts.cudaFree();
	d_end.cudaFree();
}

std::vector<in_elt_t> generate_input(int size)
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

int main(void)
{
	//std::vector<in_elt_t> in = generate_input(21);
	std::vector<in_elt_t> in = generate_input(2 * 1024 * 1024);

	std::vector<in_elt_t> out_symbols(in.size());
	std::vector<int> out_counts(in.size());
	int end{0};

	serialRLE(in, out_symbols, out_counts, end);

	//gpuRLE(in, out_symbols, out_counts, end);

	out_symbols.resize(end);
	out_counts.resize(end);

	std::cout << "[";
	for (int i = 0; i < out_symbols.size(); i++)
		std::cout << "(" << out_counts[i]
			 << ", " << out_symbols[i]
			 << "), ";
	std::cout << "]" << std::endl;

	return 0;
}
