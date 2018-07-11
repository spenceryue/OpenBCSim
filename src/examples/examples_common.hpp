#pragma once
#include <vector>
#include <string>
#include <fstream>

template <typename T>
inline void dump_rf_line(const std::string& filename,  std::vector<T>& samples) {
    std::ofstream out_file(filename);
    for (size_t i = 0; i < samples.size(); i++) {
        out_file << samples[i] << " ";
    }
    out_file << std::endl;
}

template <>
inline void dump_rf_line(const std::string& filename, std::vector<std::complex<float>>& samples) {
	std::ofstream out_file(filename);
	for (const auto& x : samples)
	{
		out_file << x.real() << "," << x.imag() << "\n";
	}
	out_file << std::endl;
}