#include "common.h"
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

// Basic tensor structure
struct Tensor {
    float* data;
    std::vector<int> shape;
    int size;
};

// Function to load weights from the binary file
std::map<std::string, Tensor> load_weights(const std::string& path) {
    std::map<std::string, Tensor> weights;
    std::ifstream infile(path, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening weight file: " << path << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string name;
    std::string shape_str;
    while (std::getline(infile, name) && std::getline(infile, shape_str)) {
        Tensor t;
        std::string num;
        for (char c : shape_str) {
            if (c == ',') {
                t.shape.push_back(std::stoi(num));
                num = "";
            } else {
                num += c;
            }
        }
        t.shape.push_back(std::stoi(num));

        t.size = 1;
        for (int dim : t.shape) {
            t.size *= dim;
        }

        // Allocate memory and read data
        t.data = new float[t.size];
        infile.read(reinterpret_cast<char*>(t.data), t.size * sizeof(float));
        
        // Skip the newline character after the binary data
        infile.ignore(1);

        weights[name] = t;
    }

    return weights;
}
