/*
MIT License

Copyright (c) 2024 Murilo Salviato Pileggi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#pragma region Includes
#include <algorithm>
#include <any>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <valarray>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>


#pragma endregion

namespace MLPP 
{
    
#pragma region New Data Structure declarations
    // Declaring 2d Matrix template, which is a vector that holds other vectors
    // Ex: Mat2d[0] = vector stored in at index 0, Mat2d[0][0] = single value stored at index (0,0)
    // Essentially Mat2d[x][y] will return value stored in row x, column y
    template <typename T> using Mat2d = std::vector<std::vector<T>>;
    // Declare 3d matrix template which is a vector that holds 2d matrices
    // Ex: Mat3d[0] = 2d matrix stored at index 0, Mat3d[0][0] = vector stored at index 0 in stored 2d matrix,
    // Mat3d[0][0][0] = single value stored at index (0,0,0)
    // Essentially Mat3d[x][y][z] will return value stored in row x, column y, depth z
    template <typename T> using Mat3d = std::vector<Mat2d<T>>;  
    // Declare 4d matrix template where each element is a 3d matrix
    // Ex: Mat4d[0] = 3d matrix stored at index 0, Mat4d[0][0] = 2d matrix stored at index (0,0),
    // Mat4d[0][0][0] = vector stored at index (0,0,0), Mat4d[0][0][0][0] = single value stored at index (0,0,0,0)
    // Essentially Mat4d[a][x][y][z] = will return values stored in 3d matrix a, in row x, column y, depth z
    template <typename T> using Mat4d = std::vector<Mat3d<T>>; 
#pragma endregion

#pragma region Enums
    // Activation functions enum for neural networks
    enum Activation { RELU, TANH, SIGMOID, SOFTMAX };
    // Formatter utility enum to help with method overload
    enum Formatter { ROW, COLUMN, ROWANDCOLUMN };
    // Matrix type enums, easier to keep track of matrices typing
    enum MatrixType { MAT_2D, MAT_3D, MAT_4D };
    // Padding enum for convolution layers
    enum Padding { SAME, VALID };
    // Layer type enum used to identify layers in serialization
    enum LayerType { AVERAGEPOOLING2D, CONV2D, DENSE, FLATTEN, MAXPOOLING2D, NONE };
    
#pragma endregion

#pragma region NumPP
    // Class that contains all methods that are needed for numeric computation
    class NumPP
    {
    public:

        // Method that adds two vectors of same shape
        template <typename T>
        static std::vector<T> add(const std::vector<T>& a, const std::vector<T>& b)
        {
            if (a.empty() || b.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            if (a.size() != b.size()) {
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(a.size());

            for (size_t i = 0; i < output_vector.size(); i++) {
                output_vector[i] = a[i] + b[i];
            }

            return output_vector;
        }
        
        // Overload of add method that adds two matrices of same shape 
        template <typename T>
        static Mat2d<T> add(const Mat2d<T>& a, const Mat2d<T>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            if (a.size() != b.size() || a[0].size() != b[0].size()) {
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(a.size(), std::vector<T>(a.size()));

            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[i].size(); j++) {
                    result[i][j] = a[i][j] + b[i][j];
                }
            }

            return result;
        }

        // Overload of add method that adds vector to each row of matrix 
        template <typename T>
        static Mat2d<T> add(const Mat2d<T>& mat, const std::vector<T>& vec)
        {
            if (mat.empty() || vec.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(mat.size(), std::vector<T>(vec.size()));

            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[i].size(); j++) {
                    result[i][j] = mat[i][j] + vec[j];
                }
            }

            return result;
        }

        // Method that receives two 2d matrices and returns their dot product
        template <typename T> 
        static Mat2d<T> dot(const Mat2d<T>& a, const Mat2d<T>& b)
        {       
            if (!a.empty() || !b.empty()) {
                if (a[0].size() == b.size()) { 
                    // Create new 2d matrix to store result
                    Mat2d<T> result(a.size(), std::vector<T>(b[0].size())); 
                    for (size_t a_row = 0; a_row < a.size(); a_row++) {
                        //std::vector<T> nRow(result[a_row].size());
                        for (size_t b_col = 0; b_col < b[0].size(); b_col++) {
                            T r = 0; // Create smart pointer to store temporary calculation results
                            for (size_t k = 0; k < b.size(); k++) {
                                r += a[a_row][k] * b[k][b_col];
                            }      
                            result[a_row][b_col] = r;
                        }
                    }
                    return result; // Return matrix with new values
                }
                // Display error that informs matrix a column size is different to matrix b row size
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Input empty" << std::endl;
            return Mat2d<T>(); // Return empty matrix
        }

        // Overload of dot method that multiplies vector with 2d matrix and return a vector
        // I.E, we treat vector as 2d matrix of size (0,R) and multiply with matrix of size (R,C)
        // Returning a matrix of size (0,C), which is treated as vector to return
        template <typename T> 
        static std::vector<T> dot(const std::vector<T>& a, const Mat2d<T>& b)
        {
            if (a.empty() || b.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            if (a.size() != b.size()) {
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(b[0].size()); // Create output vector with b matrix columns size
            T r; // Create variable to store temporary results

            for (size_t i = 0; i < output_vector.size(); i++) {
                r = 0;
                for (size_t j = 0; j < a.size(); j++) {
                    r += a[j] * b[j][i];
                }
                output_vector[i] = r; // Fill output vector with dot result from row j and column i
            }

            return output_vector;
        }

        // Method that finds max value in vector
        template <typename T>
        static T find_max_value(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return 0;
            }

            return static_cast<T>(*std::max_element(vec.begin(), vec.end()));
        }

        // Overload that finds max value in 2d matrix
        template <typename T>
        static T find_max_value(const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return 0;
            }

            T result = 0;

            for (size_t i = 0; i < mat.size(); i++) {
                T max = *std::max_element(mat[i].begin(), mat[i].begin());
                if (max >= result) {
                    result = max;
                }
            }

            return result;
        }

        // Method that finds min value in vector
        template <typename T>
        static T find_min_value(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return 0;
            }

            return static_cast<T>(*std::min_element(vec.begin(), vec.end()));
        }
        
        // Overload that finds min value in 2d matrix
        template <typename T>
        static T find_min_value(const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return 0;
            }

            T result = 0;

            for (size_t i = 0; i < mat.size(); i++) {
                T min = *std::min_element(mat[i].begin(), mat[i].begin());
                if (min <= result) {
                    result = min;
                }
            }

            return result;
        }

        // Method that rotates given matrix along given axis
        // Flip matrix vertically, Axis = 0
        // Flip matrix horizontally, Axis = 1
        template <typename T>
        static Mat2d<T> flip_matrix(const Mat2d<T>& mat, const bool& vertical, const bool& horizontal = false)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>();
            }

            // Create matrix of zeros of same dimensions as original matrix
            Mat2d<T> flipped_matrix = zeros<T>(mat.size(), mat[0].size());
            const size_t rows = flipped_matrix.size(); // Store number of rows
            const size_t cols = flipped_matrix[0].size(); // Store number of columns

            // Check for vertical axis flag
            if (vertical) {
                // If horizontal flag is also set to true, flip on both axis
                if (horizontal) {
                    for (size_t i = 0; i < rows; i++) {
                        for (size_t j = 0; j < cols; j++) {
                            flipped_matrix[i][j] = mat[rows-1-i][cols-1-j];
                        }
                    }
                    return flipped_matrix;
                }
                // Flip matrix vertically
                for (size_t i = 0; i < rows; i++) {
                    flipped_matrix[i] = mat[rows-1-i];
                }
                return flipped_matrix;
            }
            // Flix matrix horizontally
            for (size_t i = 0; i < cols; i++) {
                for (size_t j = 0; j < rows; j++) {
                    flipped_matrix[j][i] = mat[j][cols-1-i];
                }
            }
            return flipped_matrix;

        } 

        // Method that calculates and returns average of given vector
        template <typename T>
        static T get_average(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return T(0);
            }

            T sum = get_sum_of_vector(vec); // Get sum of all values in vector

            return sum / static_cast<T>(vec.size()); // Return average          
        }

        // Overload that gets average in 2d matrix
        template <typename T>
        static T get_average(const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return T(0);
            }

            size_t rows = mat.size(); // Variable to store number of rows in matrix
            size_t cols = mat[0].size(); // Variable to store number of columns in matrix 
            T num_of_elements = rows * cols; // Variable to store number of elements in matrix
            T sum = 0; // Variable to store sum of values
            T average = 0; // Variable to return calculated average

            for (size_t i = 0; i < rows; i++) {
                sum += get_sum_of_vector(mat[i]); // Get sum of current row
            }

            return sum / num_of_elements; // Return average          
        }

        // Method that generates square 2d matrix
        template <typename T>
        static Mat2d<T>* gen_square_matrix(const size_t& size, const T& value = 0)
        {
            // Create new 2d matrix object with specified parameters to later return
            Mat2d<T>* result = new Mat2d<T>(size, std::vector<T>(size));

            // Fill matrix with specified value
            for (size_t rows = 0; rows < result->size(); rows++) {
                for (size_t cols = 0; cols < (*result)[rows].size(); cols++) {
                    (*result)[rows][cols] = value;
                }
            }

            // Return random 2d matrix
            return result;
        }
       
        // Method that returns center coordinates of given 2d matrix
        template <typename T>
        static std::vector<int> get_center(const Mat2d<T>* matPtr)
        {
            if (matPtr == nullptr) {
                std::vector<int> pos {0,0};
                // Display error that informs data matrix is empty
                std::cerr << "Error: Null matrix pointer" << std::endl;
                return pos; // Return empty vector
            }

            std::vector<int> pos {matPtr->size()/2, (*matPtr)[matPtr->size()/2].size()/2}; // Create vector pointer to store position     
            return pos;
        }

        // Overload to get center of 1d matrix, i.e, a vector
        template <typename T>
        static int get_center(const std::vector<T>* vecPtr)
        {
            return vecPtr->size()/2;
        }
       
        // Method that gets position of max element in vector
        template <typename T>
        static size_t get_max_element_pos(const std::vector<T>& vec)
        {
            T max_element = 0;
            size_t pos;

            for (size_t i = 0; i < vec.size(); i++) {
                if (vec[i] > max_element) {
                    max_element = vec[i];
                    pos = i;
                }
            }

            return pos;
        }

        // Overload of get max element position that gets position of max element in 2d matrix
        template <typename T> 
        static std::pair<size_t, size_t> get_max_element_pos(const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Erro: Input is empty" << std::endl;
                return std::pair<int, int>();
            }

            size_t max_x = 0; // Create variable to store index of max element X position
            size_t max_y = 0; // Create variable to store index of max element Y position
            T max_element = 0; // Create variable to store index of max element value

            for (size_t x = 0; x < mat.size(); x++) {
                for (size_t y = 0; y < mat[x].size(); y++) {
                    if (max_element > mat[x][y]) {
                        break;
                    }
                    max_x = x;
                    max_y = y;
                    max_element = mat[x][y];
                }
            }

            std::pair<size_t, size_t> pos = std::make_pair(max_x, max_y);
            return pos;
        }

        // Method that returns dimensions (shape) of 2d matrix
        template <typename T>
        static std::pair<int, int> get_shape(const Mat2d<T>& mat)
        {
            int rows = mat.size();
            int cols = (rows > 0) ? mat[0].size() : 0;
            return std::make_pair(rows, cols);
        }

        // Overload of get shape method that returns dimensions(shape) of 3d matrix
        template <typename T>
        static std::array<int, 3> get_shape(const Mat3d<T>& mat)
        {
            int rows = mat.size();
            int cols = (rows > 0) ? mat[0].size() : 0;
            int depth = (cols > 0) ? mat[0][0].size() : 0;
            return {rows, cols, depth};
        }

        // Overload of get shape method that returns dimensions(shape) of 3d matrix
        template <typename T>
        static std::array<int, 4> get_shape(const Mat4d<T>& mat)
        {
            int samples = mat.size();
            int rows = (samples > 0) ? mat[0].size() : 0;
            int cols = (rows > 0) ? mat[0][0].size() : 0;
            int depth = (cols > 0) ? mat[0][0][0].size() : 0;
            return {samples, rows, cols, depth};
        }
       
        // Method that returns sum of all elements in vector
        template <typename T>
        static T get_sum_of_vector(const std::vector<T>& vec)
        {
            T r;
            for (size_t i = 0; i < vec.size(); i++) {
                r += vec[i];
            }
            return r;
        }

        // Get sum of vector overload for diffferent type input and output
        template <typename T, typename R>
        static R get_sum_of_vector(const std::vector<T>& vec)
        {
            R r;
            for (size_t i = 0; i < vec.size(); i++) {
                r += static_cast<R>(vec[i]);
            }
            return r;
        }
       
        // Method to multiply corresponding elements of 2 2d matrices
        // Matrices must be same size
        template <typename T> 
        static Mat2d<T> mat_mul_matching_elements(const Mat2d<T>& a, const Mat2d<T>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            if (a.size() != b.size() || a[0].size() != b[0].size()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(a.size(), std::vector<T>(a[0].size()));

            for (size_t row = 0; row < a.size(); row++) {
                for (size_t col = 0; col < a[row].size(); col++) {
                    result[row][col] = a[row][col] * b[row][col];
                }
            }

            return result;
        }

        // Overload method to multiply corresponding elements of 2 3d matrices
        // Matrices must be same size
        template <typename T> 
        static Mat3d<T> mat_mul_matching_elements(const Mat3d<T>& a, const Mat3d<T>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat3d<T>(); // Return empty matrix
            }

            if (a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return Mat3d<T>(); // Return empty matrix
            }

            Mat3d<T> result;

            for (size_t depth = 0; depth < a.size(); depth++) {
                Mat2d<T> current_mat = mat_mul_matching_elements(a[depth], b[depth]);
                result.push_back(current_mat);
            }

            return result;
        }

        // Overload method to multiply corresponding elements of 2 4d matrices
        // Matrices must be same size
        template <typename T> 
        static Mat4d<T> mat_mul_matching_elements(const Mat4d<T>& a, const Mat4d<T>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat4d<T>(); // Return empty matrix
            }

            if (a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size() || a[0][0][0].size() != b[0][0][0].size()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return Mat4d<T>(); // Return empty matrix
            }

            Mat4d<T> result;

            for (size_t i = 0; i < a.size(); i++) {
                Mat3d<T> current_mat = mat_mul_matching_elements(a[i], b[i]);
                result.push_back(current_mat);
            }

            return result;
        }
       
        // Method that generates vector of size N filled with 1
        template <typename T>
        static std::vector<T> ones(const int& size)
        {
            return std::vector<T>(size, 1);
        }

        // OVerload of zeros that generates 2d matrix of size R x C filled with 1
        template <typename T>
        static Mat2d<T> ones(const size_t& rows, const size_t& cols)
        {
            return Mat2d<T>(rows, std::vector<T>(cols, 1));
        }

        // Method that elevates each element of given matrix to the power of n
        template <typename T>
        static std::vector<double> power(const std::vector<T>* vecPtr, const size_t& power = 2)
        {
            if (vecPtr == nullptr) {
                std::cerr << "Error: Matrix pointer is null" << std::endl;
                return std::vector<double>(); // Return empty matrix
            }

            std::vector<double> results(vecPtr->size());

            for (size_t i = 0; i < results.size(); i++)
            {
                results[i] = std::pow((*vecPtr)[i], power);
            }
            
            return results;
        }

        // Overload for 2d matrix
        template <typename T>
        static Mat2d<double> power(const Mat2d<T>& mat, const size_t& power = 2)
        {
            if (mat.empty()) {
                std::cerr << "Error: Matrix pointer is null" << std::endl;
                return Mat2d<double>(); // Return empty matrix
            }
            
            Mat2d<double> results(mat.size(), std::vector<double>(mat[0].size()));

            for (size_t row = 0; row < results.size(); row++) {
                for (size_t col = 0; col < results[row].size(); col++) {
                    results[row][col] = std::pow(mat[row][col], power);
                }
            }

            return results;
        }
       
        // Method that generates random 2d matrix 
        template <typename T>
        static Mat2d<T> rand(const size_t& rows, const size_t& cols, const double& mean = 0.0, const double& stddev = 1.0)
        {
            // Create new 2d matrix object with specified parameters to later return
            Mat2d<T> result(rows, std::vector<T>(cols));

            // Initialize random generator - Normal Distribution, receives mean and standard deviation as paramaters
            std::random_device rd;
            std::mt19937 generator(rd());
            std::normal_distribution<double> distribution(mean, stddev);

            // Fill matrix with random values
            for (size_t rows = 0; rows < result.size(); rows++) {
                for (size_t cols = 0; cols < result[rows].size(); cols++) {
                    result[rows][cols] = static_cast<T>(distribution(generator));
                }
            }

            // Return random 2d matrix
            return result;
        }

        // Overload for method with uniform distribution
        template <typename T>
        static Mat2d<T> rand(const size_t& rows, const size_t& cols, const int& min, const int& max)
        {
            // Create new 2d matrix object with specified parameters to later return
            Mat2d<T> result(rows, std::vector<T>(cols));

            // Initialize random generator - Uniform distribution, generate integer values between set interval
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_int_distribution<int> distribution(min, max);

            // Fill matrix with random values
            for (size_t rows = 0; rows < result.size(); rows++) {
                for (size_t cols = 0; cols < result[rows].size(); cols++) {
                    result[rows][cols] = static_cast<T>(distribution(generator));
                }
            }

            return result;
        }

        // Method that applies Rectified Linear Unit function to input
        template <typename T>
        static T relu(const T& input)
        {
            if (input <= 0) {
                return 0;
            }
            return input;
        }

        // Overload method that applies Rectified Linear Unit to every value in vector
        template <typename T>
        static std::vector<T> relu(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(vec.size());

            for (size_t i = 0; i < output_vector.size(); i++) {
                output_vector[i] = relu(vec[i]);
            }

            return output_vector;
        }

        // Overload method that applies Rectified Linear Unit to every value in 3d matrix
        template <typename T>
        static Mat2d<T> relu(const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat2d<T>();
            }

            Mat2d<T> relu_mat(mat.size(),std::vector<T>(mat[0].size()));

            for (size_t i = 0; i < relu_mat.size(); i++) {
                for (size_t j = 0; j < relu_mat[i].size(); j++) {
                    relu_mat[i][j] = relu(mat[i][j]);
                }
            }

            return relu_mat;
        }

        // Overload method that applies Rectified Linear Unit to every value in 3d matrix
        template <typename T>
        static Mat3d<T> relu(const Mat3d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat3d<T>();
            }

            Mat3d<T> relu_mat(mat.size(), Mat2d<T>(mat[0].size(), std::vector<T>(mat[0][0].size())));

            for (size_t i = 0; i < relu_mat.size(); i++) {
                for (size_t j = 0; j < relu_mat[i].size(); j++) {
                    for (size_t k = 0; k < relu_mat[i][j].size(); k++) {
                        relu_mat[i][j][k] = relu(mat[i][j][k]);
                    }
                }
            }

            return relu_mat;
        }

        // Method that applies Rectified Linear Unit derivative function to input
        template <typename T>
        static T relu_derivative(const T& input)
        {
            if (input <= 0) {
                return 0;
            }
            return 1;
        }

        // Overload method that applies Rectified Linear Unit derivative to every value in vector
        template <typename T>
        static std::vector<T> relu_derivative(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(vec.size());

            for (size_t i = 0; i < output_vector.size(); i++) {
                output_vector[i] = relu_derivative(vec[i]);
            }

            return output_vector;
        }

        // Overload method that applies Rectified Linear Unit derivative to every value in 3d matrix
        template <typename T>
        static Mat2d<T> relu_derivative(const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat2d<T>();
            }

            Mat2d<T> relu_mat;

            for (size_t i = 0; i < mat.size(); i++) {
                std::vector<T> current_vec = relu_derivative(mat[i]);
                relu_mat.push_back(current_vec);
            }

            return relu_mat;
        }

        // Overload method that applies Rectified Linear Unit derivative to every value in 3d matrix
        template <typename T>
        static Mat3d<T> relu_derivative(const Mat3d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat3d<T>();
            }

            Mat3d<T> relu_mat;

            for (size_t i = 0; i < mat.size(); i++) {
                Mat2d<T> current_mat = relu_derivative(mat[i]);
                relu_mat.push_back(current_mat);
            }

            return relu_mat;
        }

        // Overload method that applies Rectified Linear Unit derivative to every value in 4d matrix
        template <typename T>
        static Mat4d<T> relu_derivative(const Mat4d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat4d<T>();
            }

            Mat4d<T> relu_mat;

            for (size_t i = 0; i < mat.size(); i++) {
                Mat3d<T> current_mat = relu_derivative(mat[i]);
                relu_mat.push_back(current_mat);
            }

            return relu_mat;
        }

        // Method that reshapes 2d matrix into 4d matrix
        template <typename T>
        static Mat4d<T> reshape(const Mat2d<T>& input, const std::array<int, 4>& output_shape)
        {
            if (input.empty() || output_shape.empty()) {
                return Mat4d<T>();
            }

            int batch_size = output_shape[0]; // Store batch size value
            int rows = output_shape[1]; // Store output rows size
            int cols = output_shape[2]; // Store output columns size
            int depth = output_shape[3]; // Store output depth size

            Mat4d<T> reshaped_mat(batch_size, Mat3d<T>(rows, Mat2d<T>(cols, std::vector<T>(depth))));           

            for (size_t i = 0; i < input.size(); i++) {
                // Iterate through 2d matrix rows
                size_t row_tracker = 0; // Keep track of current row in 4d matrix
                size_t column_tracker = 0; // Keep track of current column in 4d matrix
                size_t depth_tracker = 0; // Keep track of current depth in 4d matrix
                for (size_t j = 0; j < input[i].size(); j++) {
                    // Iterate through 2d matrix columns
                    reshaped_mat[i][row_tracker][column_tracker][depth_tracker] = input[i][j];
                    depth_tracker++;
                    if (depth_tracker >= depth) {
                        depth_tracker = 0;
                        column_tracker++;
                    }
                    if (column_tracker >= cols) {
                        column_tracker = 0;
                        row_tracker++;
                    }         
                }
            }
        
            return reshaped_mat;
        }

        // Method that multiplies given number with all elements in 2d matrix
        template <typename T>
        static Mat2d<T> scalar_mat_mul(const Mat2d<T>& mat, const T& scalar)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(mat.size(), std::vector<T>(mat[0].size()));

            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[i].size(); j++) {
                    result[i][j] = mat[i][j] * scalar;
                }
            }

            return result;
        }
        
        // Overload for different input and output type
        template <typename T, typename R>
        static Mat2d<R> scalar_mat_mul(const Mat2d<T>& mat, const R& scalar)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<R>(); // Return empty matrix
            }

            Mat2d<R> result(mat.size(), std::vector<R>(mat[0].size()));

            for (size_t row = 0; row < result.size(); row++) {
                for (size_t col = 0; col < result[row].size(); col++) {
                    result[row][col] = static_cast<R>(mat[row][col]) * scalar;
                }
            }

            return result;
        }

        // Overload that multiplies given number with all elements in 3d matrix
        template <typename T>
        static Mat3d<T> scalar_mat_mul(const Mat3d<T>& mat, const T& scalar)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat3d<T>(); // Return empty matrix
            }

            Mat3d<T> result(mat.size(), Mat2d<T>(mat[0].size(), std::vector<T>(mat[0][0].size())));

            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[i].size(); j++) {
                    for (size_t k = 0; k < result[i][j].size(); k++) {
                        result[i][j][k] = mat[i][j][k] * scalar;
                    }
                }
            }

            return result;
        }

        // Method that subtracts every element in mat by given number
        template <typename T>
        static Mat2d<T> scalar_mat_sub(const Mat2d<T>& mat, const T& scalar)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(mat.size(), std::vector<T>(mat[0].size()));

            for (auto &row : result) {
                for (auto &el : row) {
                    el = (el - scalar);
                }
            }

            return result;
        }
        
        // Overload that subtracts given number by every element in matrix
        template <typename T>
        static Mat2d<T> scalar_mat_sub(const T& scalar, const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(mat.size(), std::vector<T>(mat[0].size()));

            for (auto &row : result) {
                for (auto &el : row) {
                    el = (scalar - el);
                }
            }

            return result;
        }

        // Method that multiplies given number with all elements in vector
        template <typename T>
        static std::vector<T> scalar_vec_mul(const std::vector<T>& vec, const T& scalar)
        {
            if (vec.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return std::vector<T>(); // Return empty vector
            }

            std::vector<T> result(vec.size());
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = vec[i] * scalar;
            }
            
            return result;        
        }

        // Method that applies sigmoid function to vector
        template <typename T> 
        static std::vector<T> sigmoid(const std::vector<T>& input_vector)
        {
            if (input_vector.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(input_vector.size());
            const T scaling_factor = 10000.0; // Smooth values to fit c++ calculations limit
            T adjusted_value = 0.0;

            for (size_t i = 0; i < output_vector.size(); i++) {
                adjusted_value = input_vector[i];        
               
                //std::cout << adjusted_value << std::endl;
                output_vector[i] = static_cast<T>(1 / (1 + std::exp(static_cast<double>(-adjusted_value))));
                adjusted_value = 0;
            }

            return output_vector;
        }

        // Method that applies softmax function to vector
        // !!! Still need to work on method a little more, make sure everything is working properly !!!
        template <typename T> 
        static std::vector<T> softmax(const std::vector<T>& input_vector)
        {
            if (input_vector.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            //std::cout << input_vector[0] << ", " << input_vector[1] << ", " << input_vector[2] << std::endl;
            std::vector<T> exp_values(input_vector.size()); // Create vector to store computed exponential values

            // Find the maximum value in the input to improve numerical stability
            T max_input = static_cast<double>(NumPP::find_max_value(input_vector));
            // Determine a scaling factor to ensure values fall within a manageable range
            const T scaling_factor = 0.1;
            // Create vector to hold adjusted values
            std::vector<T> adjusted_values(input_vector.size());

            // Subtract max input to current input value and apply scaling
            for (size_t i = 0; i < adjusted_values.size(); i++)
            {   
                // Get adjusted value for current input value
                //adjusted_values[i] = (input_vector[i] - max_input) * scaling_factor;
                //adjusted_values[i] = (input_vector[i] - max_input);
                //std::cout << "Adjusted Value " << i << ": " << adjusted_values[i] << std::endl;
                // Apply exponential function to adjusted value
                //exp_values[i] = std::exp(adjusted_values[i]);
                exp_values[i] = std::exp(input_vector[i]); 
                //std::cout << "Exp Value " << i << ": " << exp_values[i] << std::endl;
            }         
            
            // Compute the sum of exponentials
            T sum_exp = std::accumulate(exp_values.begin(), exp_values.end(), static_cast<T>(0));
            //std::cout << "Sum " << sum_exp << std::endl;

            // Normalize exponentials to get probabilities
            for (size_t i = 0; i < exp_values.size(); i++)
            {   
                exp_values[i] /= sum_exp;
            }
            
            return exp_values;
        }
        
        // Method to subtract two vectors
        template <typename T>
        static std::vector<T> subtract(const std::vector<T>& a, const std::vector<T>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return std::vector<T>(); // Return null pointer        
            }

            if (a.size() != b.size()) {
                // Display error that informs vectors are of different sizes
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return std::vector<T>(); // Return null pointer
            }

            std::vector<T> result(a.size());

            for (size_t i = 0; i < result.size(); i++)
            {
                result[i] = a[i] - b[i];
            }

            return result;         
        }

        // Overload of subtract method to subtract two 2d matrices
        template <typename T>
        static Mat2d<T> subtract(const Mat2d<T>& a, const Mat2d<T>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return null pointer        
            }

            if ((a.size() != b.size()) || (a[0].size() != b[0].size())) {
                // Display error that informs matrix a column size is different to matrix b row size
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return Mat2d<T>(); // Return null pointer
            }

            Mat2d<T> result(a.size(), std::vector<T>(a[0].size()));

            for (size_t row = 0; row < result.size(); row++) {
                for (size_t col = 0; col < result[row].size(); col++) {
                    result[row][col] = a[row][col] - b[row][col];
                }
            }

            return result;
        }

        // Overload of subtract method to subtract two 3d matrices
        template <typename T>
        static Mat3d<T> subtract(const Mat3d<T>& a, const Mat3d<T>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat3d<T>(); // Return null pointer        
            }

            if ((a.size() != b.size()) || (a[0].size() != b[0].size()) || (a[0][0].size() != b[0][0].size())) {
                // Display error that informs matrix a column size is different to matrix b row size
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return Mat3d<T>(); // Return null pointer
            }

            Mat3d<T> result(a.size(), Mat2d<T>(a[0].size(), std::vector<T>(a[0][0].size())));

            for (size_t row = 0; row < result.size(); row++) {
                for (size_t col = 0; col < result[row].size(); col++) {
                    for (size_t depth = 0; depth < result[row][col].size(); depth++) {
                        result[row][col][depth] = a[row][col][depth] - b[row][col][depth];
                    }  
                }
            }

            return result;
        }

        // Method that collapses 2d matrix into vector with each element
        // being the sum of each row or each column from 2d matrix, 0 = rows, 1 = columns
        template <typename T> 
        static std::vector<T> sum(const Mat2d<T>& mat, const size_t& axis = 0)
        {
            if (mat.empty()) {
                std::cerr << "Error: Matrix is empty" << std::endl;
                return std::vector<T>();
            }

            if (axis == 1) {
                // Get sum of each column into vector
                std::vector<T> result(mat[0].size(), 0); // Create vector to store result
                for (size_t i = 0; i < result.size(); i++) {
                    for (size_t j = 0; j < mat.size(); j++) {
                        result[i] += mat[j][i];
                    }
                }

                return result;
            }

            std::vector<T> result(mat.size()); // Create vector to store result

            for (size_t i = 0; i < result.size(); i++) {
                result[i] = get_sum_of_vector<T>(mat[i]);
            }

            return result;
        }

        // Returns sum of all multiplications, matrices must have same size
        template <typename T> 
        static T sum_mat_mul_matching_elements(const Mat2d<T>* a, const Mat2d<T>* b)
        {
            if (a->empty() || b->empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return 0; // Return 0
            }

            T r = 0; // Create variable to store results
            // Iterate through every element in a, and multiply by matching elemnt in b
            for (size_t i = 0; i < a->size(); i++) {
                for (size_t j = 0; j < (*a)[i].size(); j++) {
                    r += (*a)[i][j] * (*b)[i][j];
                }
            }

            return r; 
        } 
        
        // Overload that receives matrices of differente types and returns specified type
        template <typename R, typename A, typename B> 
        static R sum_mat_mul_matching_elements(const Mat2d<A>& a, const Mat2d<B>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return 0; // Return 0
            }

            R r = 0; // Create variable to store results
            // Iterate through every element in a, and multiply by matching elemnt in b
            for (size_t i = 0; i < a.size(); i++) {
                for (size_t j = 0; j < a[i].size(); j++) {
                    r += static_cast<R>(a[i][j] * b[i][j]);
                }
            }

            return r; 
        } 

        // Method that apply tanh function to every element in vector
        template <typename T>
        static std::vector<T> tanh(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(vec.size());

            for (size_t i = 0; i < output_vector.size(); i++) {
                output_vector[i] = static_cast<T>(std::tanh(vec[i]));
            }

            return output_vector;
        }

        // Overload of tanh ethod that apply tanh function to every element in 2d matrix
        template <typename T>
        static Mat2d<T> tanh(const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return null pointer        
            }

            Mat2d<T> tanhMat(mat.size(), std::vector<T>(mat[0].size()));

            for (size_t i = 0; i < tanhMat.size(); i++) {
                for (size_t j = 0; j < tanhMat[i].size(); j++) {
                    tanhMat[i][j] = static_cast<T>(std::tanh(mat[i][j]));
                }
            }
            return tanhMat;
        }  

        // Overload of tanh method to deal with 3d matrices
        template <typename T>
        static Mat3d<T> tanh(const Mat3d<T>& mat)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat3d<T>(); // Return null pointer        
            }

            Mat3d<T> tanhMat(mat.size(), Mat2d<T>(mat[0].size(), std::vector<T>(mat[0][0].size())));

            for (size_t i = 0; i < tanhMat.size(); i++) {
                for (size_t j = 0; j < tanhMat[i].size(); j++) {
                    for (size_t k = 0; k < tanhMat[i][j].size(); k++) {
                        tanhMat[i][j][k] = static_cast<T>(std::tanh(mat[i][j][k]));
                    }
                }
            }
            return tanhMat;
        }
   
        // Method to transpose 2d matrix, [100, 50] -> [50, 100]
        template <typename T> 
        static Mat2d<T> transpose(const Mat2d<T>& to_transpose)
        {
            if (!to_transpose.empty()) {
                // Get original matrix dimensions
                size_t rows = to_transpose.size();
                size_t cols = rows > 0 ? to_transpose[0].size() : 0;

                // Create new 2d matrix object with reshaped dimensions
                Mat2d<T> t(cols, std::vector<T>(rows));
                // Transpose the original matrix to reshape it
                for (size_t i = 0; i < cols; i++) {              
                    for (size_t j = 0; j < rows; j++) {
                        t[i][j] = to_transpose[j][i];
                    }      
                }
                return t;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Matrix is empty" << std::endl;
            return Mat2d<T>(); // Return empty matrix
        }
        
        // Method that generates vector of size N filled with 0
        template <typename T>
        static std::vector<T> zeros(const int& size)
        {
            return std::vector<T>(size, 0);
        }

        // OVerload of zeros that generates 2d matrix of size R x C filled with 0
        template <typename T>
        static Mat2d<T> zeros(const size_t& rows, const size_t& cols)
        {
            return Mat2d<T>(rows, std::vector<T>(cols, 0));
        }

        // OVerload of zeros that generates 3d matrix of size R x C x D filled with 0
        template <typename T>
        static Mat3d<T> zeros(const size_t& rows, const size_t& cols, const size_t& depth)
        {
            return Mat3d<T>(rows, Mat2d<T>(cols, std::vector<T>(depth, 0)));
        }

        // OVerload of zeros that generates 4d matrix of size S x R x C x D filled with 0
        template <typename T>
        static Mat4d<T> zeros(const size_t& size, const size_t& rows, const size_t& cols, const size_t& depth)
        {
            return Mat4d<T>(size, Mat3d<T>(rows, Mat2d<T>(cols, std::vector<T>(depth, 0))));
        }

        // Further methods to be implemented
    };

#pragma endregion

#pragma region DataAnalysis
    // Class that contains all methods that are needed for Data Analysis
    class DataAnalysis 
    {
    public:
        // Method to display all elements in vector
        template <typename T> 
        static void display_all(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            } 

            for (size_t i = 0; i < vec.size(); i++)
            {
                std::cout << std::to_string(vec[i]) << " ";
            }
            std::cout << std::endl;
        }

        // Overload to display all elements of 2d matrix
        template <typename T> 
        static void display_all(const Mat2d<T>& dataMatrix)
        {
            if (!dataMatrix.empty()) {
                for (auto& row : dataMatrix) {
                    for (auto& cell : row) {
                        std::cout << " " << std::to_string(cell);
                    }
                    std::cout << std::endl;
                }
                return;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Mehtod to display last five colunms method, Display lastdisplayBfive rows
        template <typename T> 
        static void display_bottom(const Mat2d<T>& dataMatrix, int rowsToDisplay = 5)
        {
            if (!dataMatrix.empty()) {
                if (rowsToDisplay >= 0 && rowsToDisplay < dataMatrix.size() - 1) {
                    for (int row = dataMatrix.size() - rowsToDisplay;
                        row < dataMatrix.size(); row++) {
                        for (int col = 0; col < dataMatrix[row].size(); col++) {
                            std::cout << " " << dataMatrix[row][col];
                        }
                        std::cout << std::endl;
                    }
                    return;
                }
                // Display error that informs number of rows is invalid matrix is
                // empty
                std::cerr << "Error: Rows to display is invalid" << std::endl;
                return;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Method to display all elements in given columns
        template <typename T> 
        static void display_columns(const Mat2d<T>& dataMatrix, const std::vector<int>& colsToDisplay)
        {
            if (!dataMatrix.empty() && !colsToDisplay.empty()) {
                for (size_t row = 0; row < dataMatrix.size(); row++) {
                    if (colsToDisplay[row] < dataMatrix.size()) {
                        std::cout << row << "-";
                        for (auto& col : colsToDisplay) {
                            std::cout << " " << dataMatrix[row][col];
                        }
                        std::cout << std::endl;
                    }
                }
                // Display error that informs that one or more columns to display
                // are invalid std::cerr << "Error: One or more columns to display
                // is invalid" << std::endl;
                return;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Method to display first five rows method, Display first 5 rows + display_head row
        template <typename T> 
        static void display_head(const Mat2d<T>& dataMatrix, int rowsToDisplay = 5)
        {
            if (!dataMatrix.empty()) {
                if (rowsToDisplay >= 0 && rowsToDisplay < dataMatrix.size() - 1) {
                    for (int row = 0; row < rowsToDisplay + 1; row++) {
                        for (int col = 0; col < dataMatrix[row].size(); col++) {
                            std::cout << " " << dataMatrix[row][col];
                        }
                        std::cout << std::endl;
                    }
                    return;
                }
                // Display error that informs number of rows is invalid matrix is
                // empty
                std::cerr << "Error: Rows to display is invalid" << std::endl;
                return;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Method to display all elements in given rows
        template <typename T> 
        static void display_rows(const Mat2d<T>& dataMatrix, const std::vector<int>& rowsToDisplay)
        {
            if (!dataMatrix.empty() && !rowsToDisplay.empty()) {
                for (size_t row = 0; row < rowsToDisplay.size(); row++) {
                    std::cout << rowsToDisplay[row] << "-";
                    if (rowsToDisplay[row] < dataMatrix.size()) {
                        for (size_t col = 0; col < dataMatrix[row].size(); col++) {
                            std::cout << " " << dataMatrix[row][col];
                        }
                        std::cout << std::endl;
                    }
                }
                // Display error that informs one or more of rows to display is
                // invalid std::cerr << "Error: One or more rows to display is
                // invalid" << std::endl;
                return;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Method to display 2d matrix dimensions(shape)
        template <typename T> 
        static void display_shape(const Mat2d<T>& dataMatrix)
        {
            if (dataMatrix.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }

            auto shape = NumPP::get_shape(dataMatrix);
            std::cout << "Matrix shape = " << "(" << shape.first;
            std::cout << "," << shape.second << ")" << std::endl;
        }

        // Method to display 3d matrix dimensions(shape)
        template <typename T> 
        static void display_shape(const Mat3d<T>& dataMatrix)
        {
            if (dataMatrix.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }

            auto shape = NumPP::get_shape(dataMatrix);
            std::cout << "Matrix shape = " << "(" << shape[0] << ",";
            std::cout << shape[1] << "," << shape[2] << ")" << std::endl;
        }

        // Method to display 3d matrix dimensions(shape)
        template <typename T> 
        static void display_shape(const Mat4d<T>& dataMatrix)
        {
            if (dataMatrix.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }

            auto shape = NumPP::get_shape(dataMatrix);
            std::cout << "Matrix shape = " << "(" << shape[0] << "," << shape[1];
            std::cout << "," << shape[2] << "," << shape[3] << ")" << std::endl;
        }

        // Method to search data matrix for first appearance of desired element, return first position found
        template <typename T> 
        static std::vector<int> find(const Mat2d<T>& dataMatrix, const T& desiredElement)
        {
            // Creat position vector
            std::vector<int> pos{0, 0};

            if (!dataMatrix.empty()) {
                // Loop through matrix to check for desired element
                for (size_t row = 0; row < dataMatrix.size(); row++) {
                    // std::cout << row << std::endl;
                    for (size_t col = 0; col < dataMatrix[row].size(); col++) {
                        // std::cout << col << std::endl;
                        if (dataMatrix[row][col] == desiredElement) {
                            pos[0] = row;
                            pos[1] = col;
                            return pos;
                        }
                    }
                }
                // Return empty vector
                std::cerr << "Error: Desired element was not found" << std::endl;
                return pos;
            }
            // Return 0,0 vector
            std::cerr << "Error: Data Matrix is empty" << std::endl;
            return pos;
        }
        
        // Method to search data matrix for all appearances of desired element, return
        // vector of positions Outside vecotor holds all rows indexes element was found,
        // and inside vector holds all columns indexes element was found
        template <typename T> 
        static Mat2d<int> find_all(const Mat2d<T>& dataMatrix, const T& desiredElement)
        {
            // Creat position vector
            Mat2d<int> pos;

            if (!dataMatrix.empty()) {
                // Loop through matrix to check for desired element
                for (int row = 0; row < dataMatrix.size(); row++) {
                    for (int col = 0; col < dataMatrix[row].size(); col++) {
                        if (dataMatrix[row][col] == desiredElement) {
                            std::vector<int> currentPos{row, col};
                            pos.push_back(currentPos);
                        }
                    }
                }
                if (pos.empty()) {
                    // Return empty position matrix
                    std::cerr << "Error: Element was not found" << std::endl;
                    return pos;
                }
                // Return found positions
                return pos;
            }
            // Return empty position matrix
            std::cerr << "Error: Data Matrix is empty" << std::endl;
            return pos;
        }
        
        // Overload of find_all that returns number of times element was found, not their position (Mat2d)
        template <typename T>
        static int find_all(const T& desiredElement, const Mat2d<T>& dataMatrix)
        {
            if (dataMatrix.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return 0;
            }

            int counter = 0;

            for (size_t i = 0; i < dataMatrix.size(); i++) {
                for (size_t j = 0; j < dataMatrix[i].size(); j++) {
                    if (dataMatrix[i][j] == desiredElement) {
                        counter++;
                    }
                }
            }

            return counter;
        }

        // Overload of find_all that returns number of times element was found, not their position (Mat3d)
        template <typename T>
        static int find_all(const T& desiredElement, const Mat3d<T>& dataMatrix)
        {
            if (dataMatrix.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return 0;
            }

            int counter = 0;

            for (size_t d = 0; d < dataMatrix[0][0].size(); d++) {
                for (size_t i = 0; i < dataMatrix.size(); i++) {
                    for (size_t j = 0; j < dataMatrix[i].size(); j++) {
                        if (dataMatrix[i][j][d] == desiredElement) {
                            counter++;
                        }
                    }
                }
            }

            return counter;
        }

        // Method to search data matrix by position, return element at requqested position on 2d Matrix
        template <typename T> 
        static T find_by_pos(const Mat2d<T>& dataMatrix, std::vector<int>& pos)
        {
            T element;

            if (!dataMatrix.empty()) {
                element = dataMatrix[pos[0]][pos[1]];
                return element;
            }
            // Return empty element
            std::cerr << "Error: Data Matrix is empty" << std::endl;
            return element;
        }
        
        // Method that generates hot-encoded-label, used for classification tasks
        // Hot Enconded Label is nothing more than a vector filled with zeros
        // Except for one position, which has the value of one
        template <typename T>
        static std::vector<T> gen_hot_encoded_label(const size_t& size, const size_t& pos)
        {
            if (size <= 0 || pos < 0 || pos >= size) {
                std::cerr << "Error: Invalid Parameters" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> hot_encoded_label(size, 0);

            for (size_t i = 0; i < hot_encoded_label.size(); i++) {
                if (i == pos) {
                    hot_encoded_label[i] = static_cast<T>(1);
                    break;
                }
            }

            return hot_encoded_label;
        }

        // Method to convert string to differnt data type, returns new 2d matrix with passed type
        template<typename T> 
        static Mat2d<T> matrix_converter(const Mat2d<std::string>& stringMatrix)
        {
            // Create 2d matrix of new typing
            Mat2d<T> convertedMatrix;

            if (!stringMatrix.empty()) {
                // Loop through all elements changing their typing
                for (const auto& row : stringMatrix) {
                    std::vector<T> convertedRow;

                    for (const auto& element : row) {
                        // Convert string to type T
                        std::stringstream ss(element);
                        T convertedElement;
                        ss >> convertedElement;
                        convertedRow.push_back(convertedElement);
                    }
                    // Replace string row with newly converted row to new 2d matrix
                    convertedMatrix.push_back(convertedRow);
                }
                // Return new data matrix of type T
                return convertedMatrix;
            } 
            else {
                // Display error to inform data is empty
                std::cerr << "Error: Data matrix is empty!" << std::endl;
                return convertedMatrix;
            }
        }
        
        // Converter overload that converts matrix other than string to different data type
        template <typename T, typename R>
        static Mat2d<R> matrix_converter(const Mat2d<T>& dataMatrix)
        {
            if (dataMatrix.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat2d<R>();
            }

            Mat2d<R> result(dataMatrix.size(), std::vector<R>(dataMatrix[0].size()));

            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[i].size(); j++) {
                    result[i][j] = static_cast<R>(dataMatrix[i][j]);
                }
            }

            return result;
        }

        // Method to format data matrix by removing rows and or columns for better data
        // analysis Template for removing single row or column
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const int& toBeRemoved)
        {
            if (!dataMatrix.empty()) {
                if (f == COLUMN) {
                    if (toBeRemoved >= 0) {
                        // Iterate over each row and erase the element at colIndex
                        for (auto& row : dataMatrix) {
                            if (toBeRemoved < row.size()) {
                                row.erase(row.begin() + toBeRemoved);
                                return;
                            }
                        }
                        // Display error to inform invalid index to remove
                        std::cerr << "Error: Invalid index of column to remove"
                                << std::endl;
                    }
                    // Display error to inform invalid index to remove
                    std::cerr << "Error: Invalid index of column to remove"
                            << std::endl;
                } else if (f == ROW) {
                    if (toBeRemoved >= 0 && toBeRemoved <= dataMatrix.size()) {
                        // Delete selected row from matrix
                        dataMatrix.erase(dataMatrix.begin() + toBeRemoved);
                        return;
                    }
                    // Display error to inform invalid index to remove
                    std::cerr << "Error: Invalid index of row to remove" << std::endl;
                } else {
                    // Display error to inform wrong formatter enum was passed
                    std::cerr << "Error: Wrong formatter passed (2nd Paramater)"
                            << std::endl;
                }
            }
            // Display error to inform data is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Overload to remove multiple row or columns
        // rocToRemove = rows or columns to remove, vector will hold all rows or columns to remove
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const std::vector<int>& rocToRemove)
        {
            if (!dataMatrix.empty()) {
                if (f == COLUMN) {
                    if (!rocToRemove.empty()) {
                        // Iterate over each row and erase the element at colIndexexes
                        for (int row = 0; row < dataMatrix.size(); row++) {
                            for (auto col = rocToRemove.rbegin(); col != rocToRemove.rend(); ++col) {
                                size_t colIndex = *col;
                                if (colIndex < dataMatrix[row].size()) {
                                    dataMatrix[row].erase(dataMatrix[row].begin() +
                                                        colIndex);
                                }
                            }
                        }
                        return;
                    }
                    // Display error to inform position data is empty
                    std::cerr << "Error: No rows or columns to remove were found" << std::endl;
                } 
                else if (f == ROW) {
                    if (!rocToRemove.empty()) {
                        // Iterate through declared rows to be removed from matrix
                        for (auto item = rocToRemove.rbegin(); item != rocToRemove.rend(); ++item) {
                            size_t rowIndex = *item;
                            if (rowIndex < dataMatrix.size()) {
                                dataMatrix.erase(dataMatrix.begin() + rowIndex);
                            }
                        }
                        return;
                    }
                    // Display error to inform position data is empty
                    std::cerr << "Error: No rows or columns to remove were found" << std::endl;
                } 
                else {
                    // Display error to inform wrong formatter enum was passed
                    std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
                }
            }
            // Display error to inform data is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Overload to remove single or multiple row and columns
        // racToRemove = rows and columns to remove, outside vector will hold all rows and inside vector all columns to remove
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const Mat2d<int>& racToRemove)
        {
            if (!dataMatrix.empty()) {
                if (f == ROWANDCOLUMN) {
                    if (!racToRemove.empty()) {
                        matrix_formatter(dataMatrix, ROW, racToRemove[0]);
                        matrix_formatter(dataMatrix, COLUMN, racToRemove[1]);
                        return;
                    }
                    // Display error if no position data  was found
                    std::cerr << "Error: No rows and columns to remove" << std::endl;
                }
                // Display error to inform wrong formatter enum was passed
                std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
            }
            std::cerr << "Error: Data is empty" << std::endl;
        }
        
        // Overload to add single row or column
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const int& indexToAdd, const std::vector<T>& dataToAdd)
        {
            if (!dataMatrix.empty()) {
                if (f == COLUMN) {
                    // Check if indexToAdd is valid and within bounds
                    if (indexToAdd >= 0 && indexToAdd <= dataMatrix[0].size()) {
                        for (int i = 0; i < dataToAdd.size(); i++) {
                            // Insert dataToAdd[i] at selected index in each row
                            dataMatrix[i].insert(dataMatrix[i].begin() + indexToAdd,
                                                dataToAdd[i]);
                        }
                        return;
                    }
                    std::cerr << "Error: Invalid index to add for inserting column." << std::endl;
                } 
                else if (f == ROW) {
                    // Check if indexToAdd is valid and within bounds
                    if (indexToAdd >= 0 && indexToAdd <= dataMatrix.size()) {
                        dataMatrix.insert(dataMatrix.begin() + indexToAdd, dataToAdd);
                        return;
                    }
                    std::cerr << "Error: Invalid index to add for inserting row." << std::endl;
                } 
                else {
                    // Display error to inform formatter enum error no data was
                    // found
                    std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
                }
            }
            // Display error if no data was found
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Overload to add multiple rows or columns
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const std::vector<int>& indexesToAdd, const Mat2d<T>& dataToAdd)
        {
            if (!dataMatrix.empty()) {
                if (f == COLUMN) {
                    // Check if indexesToAdd is not empty
                    if (!indexesToAdd.empty()) {
                        for (size_t i = 0; i < dataToAdd.size(); i++) {
                            for (size_t j = 0; j < dataToAdd[i].size(); j++) {
                                for (size_t k = 0; k < indexesToAdd.size(); k++) {
                                    if (indexesToAdd[k] < dataMatrix[j].size()) {
                                        dataMatrix[j].insert(dataMatrix[j].begin() + indexesToAdd[k],
                                                            dataToAdd[i][j]);
                                    }
                                }
                            }
                        }
                        return;
                    }
                    std::cerr << "Error: Indexes to add column are missing." << std::endl;
                    return;
                } 
                else if (f == ROW) {
                    // Check if indexesToAdd is not empty
                    if (!indexesToAdd.empty()) {
                        for (size_t i = 0; i < dataToAdd.size(); i++)
                        {
                            if (indexesToAdd[i] < dataMatrix.size() && i < dataToAdd.size())
                            {
                                dataMatrix.insert(dataMatrix.begin() + indexesToAdd[i], dataToAdd[i]);
                            }                     
                        }                
                        return;
                    }
                    std::cerr << "Error: Invalid index to add for inserting row." << std::endl;
                    return;
                } 
                else {
                    // Display error to inform formatter enum error no data was
                    // found
                    std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
                    return;
                }
            }
            // Display error if no data was found
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Overload to add single or multiple rows and columns 
        // Indexes to add matrix, first row will hold all row indexes, second will hold all column indexes                                                           
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const Mat2d<int>& indexesToAdd, const Mat2d<T>& dataToAdd)
        {
            if (!dataMatrix.empty()) {
                if (f == ROWANDCOLUMN) {
                    if (!dataToAdd.empty()) {
                        // Still need to figure out how to separate between row and column data
                        // in dataToAdd matrix
                        matrix_formatter(dataMatrix, ROW, indexesToAdd[0], dataToAdd);
                        matrix_formatter(dataMatrix, COLUMN, indexesToAdd[1], dataToAdd);
                        return;
                    }
                    // Display error if no position data  was found
                    std::cerr << "Error: No rows and columns to remove" << std::endl;
                    return;
                }
                // Display error to inform wrong formatter enum was passed
                std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
                return;
            }
            std::cerr << "Error: Data is empty" << std::endl;
        }
        
        // Method that encondes given dataset into one hot label format
        // This method will go through all directories inside root directory, and create the 
        // appropiate data structure to fit directory structure, i.e, vector will be sized
        // based on number of directories and matrix size will be based on elements inside 
        // sub directories P.S = Data musted be sorted, elements from class 0 must be inside
        // first sub-directory and so on. 
        template <typename T>
        static Mat2d<T> one_hot_label_encoding(const std::string& root_directory_path)
        {
            if (!std::filesystem::exists(root_directory_path) && !std::filesystem::is_directory(root_directory_path)) {
                std::cerr << "Error: Directory not valid" << std::endl;
                return Mat2d<T>();
            }

            std::vector<std::filesystem::directory_entry> directories; // Create vector to store all directoies in root directory

            // Store all entries
            for (const auto &entry : std::filesystem::directory_iterator(root_directory_path)) {
                directories.push_back(entry);
            }

            const size_t num_of_directories = directories.size(); // Get number of directories present
            size_t index_tracker = 0; // Variable to correctly track which index to place "1" in one-hot-encoded label

            // Sort entries by path
            std::sort(directories.begin(), directories.end(), [](const std::filesystem::directory_entry &a,
                const std::filesystem::directory_entry &b) {
                    return a.path() < b.path();
            });

            Mat2d<T> one_hot_encoded_labels;

            for (const auto &directory : directories) { // Loop through all directories
                std::cout << directory.path() << std::endl;
                // Loop through all elements in current directory
                for (const auto &element : std::filesystem::directory_iterator(directory.path())) {
                    // Generate one-hot-encoded label and store it in return matrix
                    one_hot_encoded_labels.push_back(gen_hot_encoded_label<T>(num_of_directories, index_tracker));
                }

                index_tracker++; // Move on to next index
            }

            return one_hot_encoded_labels;
        }

        // Method to parse CSV line with quoted fields and blank spaces
        static std::vector<std::string> parse_csv_line(const std::string& line)
        {
            std::vector<std::string> row;

            if (!line.empty()) {
                std::stringstream ss(line);
                std::string cell;

                // Tokenize line by comma
                while (std::getline(ss, cell, ',')) {
                    // Check if cell starts with a quote
                    if (cell.front() == '"') {
                        std::string quotedCell;
                        quotedCell += cell;

                        // Keep reading until we find the closing quote
                        while (ss && cell.back() != '"') {
                            std::getline(ss, cell, ',');
                            quotedCell += "," + cell;
                        }
                        // Fill out row vector with parsed cells with quotes removed
                        quotedCell = quotedCell.substr(1, quotedCell.size() - 2);
                        row.push_back(quotedCell);
                    } 
                    else if (cell.find(' ') == std::string::npos) {
                        // Fill out row vector with parsed cells
                        // Using istringstream for cells with no space works better
                        std::istringstream iss(cell);
                        std::string val;
                        iss >> val;
                        row.push_back(val);
                    }
                    else {
                        row.push_back(cell);
                    }
                }
                // Return filled out row
                return row;
            }
            // Display error and return empty row
            std::cerr << "Error: Line provided is empty" << std::endl;
            return row;
        }

        // Method to read CSV files into 2d string matrix
        static Mat2d<std::string> read_csv_file(const std::string& filePath)
        {
            // Create 2d matrix to store data
            Mat2d<std::string> data;
            // Open CSV file from path provided
            std::ifstream file(filePath);
            // Check if file opened correctly
            if (!file.is_open()) {
                // Display error and return empty data if file couldn't open
                std::cerr << "Error: Coun't open file provided at '" << filePath << "'"
                        << std::endl;
                return data;
            }
            // Reach each line of the file
            std::string line;

            while (std::getline(file, line)) {
                // Parse CSV line and add it to the data matrix
                std::vector<std::string> row = parse_csv_line(line);
                data.push_back(row);
            }
            // Close file
            file.close();
            // Return data in matrix
            return data;
        }
    
        // Method that verifies integrity of two vectors, i.e, check if all values match
        template <typename T>
        static bool verify_integrity_between_matrices(const std::vector<T>& a, const std::vector<T>& b)
        {
            if (a.empty() || b.empty()) {
                throw std::runtime_error("Error: Input parameters is empty");
            }

            if (a.size() != b.size()) {
                throw std::runtime_error("Error: Incompatible sizes");
            }

            for (size_t i = 0; i < a.size(); i++) {
                if (a[i] != b[i]) {
                    return false;
                }
            }

            return true;
        }

        // Method that verifies integrity of two 2d matrices, i.e, check if all values match
        template <typename T>
        static bool verify_integrity_between_matrices(const Mat2d<T>& a, const Mat2d<T>& b)
        {
            if (a.empty() || b.empty()) {
                throw std::runtime_error("Error: Input parameters is empty");
            }

            if ((a.size() != b.size()) || a[0].size() != b[0].size()) {
                throw std::runtime_error("Error: Incompatible sizes");
            }

            for (size_t i = 0; i < a.size(); i++) {
                bool check = verify_integrity_between_matrices(a[i], b[i]);
                if (!check) {
                    return false;
                }
            }

            return true;
        }

        // Method that verifies integrity of two 3d matrices, i.e, check if all values match
        template <typename T>
        static bool verify_integrity_between_matrices(const Mat3d<T>& a, const Mat3d<T>& b)
        {
            if (a.empty() || b.empty()) {
                throw std::runtime_error("Error: Input parameters is empty");
            }

            if ((a.size() != b.size()) || (a[0].size() != b[0].size()) || (a[0][0].size() != b[0][0].size())) {
                throw std::runtime_error("Error: Incompatible sizes");
            }

            for (size_t i = 0; i < a.size(); i++) {
                for (size_t j = 0; j < a[i].size(); j++) {
                    bool check = verify_integrity_between_matrices(a[i][j], b[i][j]);
                    if (!check) {
                        return false;
                    }
                }
            }

            return true;
        }

        // Method that verifies integrity of two 3d matrices, i.e, check if all values match
        template <typename T>
        static bool verify_integrity_between_matrices(const Mat4d<T>& a, const Mat4d<T>& b)
        {
            if (a.empty() || b.empty()) {
                throw std::runtime_error("Error: Input parameters is empty");
            }

            if ((a.size() != b.size()) || (a[0].size() != b[0].size()) || 
                (a[0][0].size() != b[0][0].size()) || (a[0][0][0].size() != b[0][0][0].size())) {
                throw std::runtime_error("Error: Incompatible sizes");
            }

            for (size_t i = 0; i < a.size(); i++) {
                for (size_t j = 0; j < a[i].size(); j++) {
                    for (size_t k = 0; k < a[i][j].size(); k++) {
                        bool check = verify_integrity_between_matrices(a[i][j], b[i][j]);
                        if (!check) {
                            return false;
                        }
                    }
                }
            }

            return true;
        }
    
    }; 

#pragma endregion

#pragma region ComputerVision
    // Class that will hold support methods for now, and later will include all methods
    // Required for computer vision
    class ComputerVision
    {
    public:
        // Method to convert 3d matrix into 2d matrix, where each element
        // Will be a sum of R, G and B values
        static Mat2d<int16_t>* get_sum_all_pixels(const Mat3d<u_int8_t>* mat3dPtr)
        {
            if (mat3dPtr == nullptr) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: 3d matrix pointer is null" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat2d pointer and allocate memory for it
            Mat2d<int16_t> *mat2dPtr = new Mat2d<int16_t>(mat3dPtr->size(), std::vector<int16_t>((*mat3dPtr)[0].size()));

            for (size_t i = 0; i < mat2dPtr->size(); i++) {
                for (size_t j = 0; j < (*mat2dPtr)[i].size(); j++) {
                    auto v = (*mat3dPtr)[i][j];
                    int16_t e = NumPP::get_sum_of_vector<u_int8_t, int16_t>(v);
                    (*mat2dPtr)[i][j] = e;
                }
            }

            return mat2dPtr;
        }

        // Overload to receive double 3d matrix
        static Mat2d<double>* get_sum_all_pixels(const Mat3d<double>* mat3dPtr)
        {
            if (mat3dPtr == nullptr) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: 3d matrix pointer is null" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat2d pointer and allocate memory for it
            Mat2d<double>* mat2dPtr = new Mat2d<double>(mat3dPtr->size(), std::vector<double>((*mat3dPtr)[0].size()));

            for (size_t i = 0; i < mat2dPtr->size(); i++) {
                for (size_t j = 0; j < (*mat2dPtr)[i].size(); j++) {
                    auto v = (*mat3dPtr)[i][j];
                    double e = NumPP::get_sum_of_vector(v);
                    (*mat2dPtr)[i][j] = e;
                }
            }

            return mat2dPtr;
        }
       
        // Method that normalize pixel values to interval [0,1]
        template <typename T>
        static Mat3d<T> normalize_pixels(const Mat3d<T>* matPtr)
        {
            if (matPtr == nullptr) {
                std::cerr << "Error: Input is null" << std::endl;
                return Mat3d<T>();
            }

            Mat3d<T> nMat(matPtr->size(), Mat2d<T>((*matPtr)[0].size(), std::vector<T>((*matPtr)[0][0].size())));

            for (size_t i = 0; i < matPtr->size(); i++) {
                for (size_t j = 0; j < (*matPtr)[i].size(); j++) {
                    for (size_t k = 0; k < (*matPtr)[i][j].size(); k++) {
                        nMat[i][j][k] = static_cast<T>((*matPtr)[i][j][k]) / 255.0;
                    }
                }
            }

            return nMat;
        }

    };

#pragma endregion

#pragma region NeuralNetwork
    // Abstract base layer class, which will be the base for all layers
    class LayerBase 
    {
    protected:
        const bool m_use_weights_and_biases;
    public:
        LayerBase(const bool use_weights_and_biases) : m_use_weights_and_biases(use_weights_and_biases) {}
        virtual ~LayerBase() = default;
        // Declaration of virtual backward method, every layer has own method implementation
        virtual void backward(std::vector<std::any>& layer_output, std::vector<std::any>& layer_linear_outputs, std::vector<std::any>& layer_weights, 
                                std::vector<std::any>& layer_biases, std::vector<std::any>& gradients, const size_t& backward_pass_locator, const size_t& w_and_b_locator,
                                 const std::any& target_labels, const std::any& learning_rate) = 0;
        // Declaration of virtual de-serialization method, every layer has own method implementation
        virtual void deserialize_layer_data(std::ifstream& in) = 0;
        // Declaration of virtual forward method, every layer has own method implementation
        virtual void forward(std::any& input, std::any& output, std::any& linear_output, std::any& weights, std::any& bias) = 0;
        // Declaration of virtual get type method
        virtual LayerType get_layer_type() { return NONE; }
        bool get_w_and_b() {return m_use_weights_and_biases;} // testing only
        // Declaration fo virtual predict method, every layer has own method implementation
        virtual void predict(std::any& input, std::any& output, std::any& weights, std::any& bias) = 0;
        // Declaration of virtual serialization method, every layer has own method implementation
        // Serialize data in array of bytes so we can export each layers relevant info into binary file
        virtual void serialize_layer_data(std::ofstream& out) = 0;
    };

    // Basic layer class, which will be the base for specialized layers
    template <typename InputType, typename OutputType, typename DataType>
    class Layer : public LayerBase
    {
    public:
        Layer(const bool use_weights_and_biases) : LayerBase(use_weights_and_biases) {};
        virtual ~Layer() = default;
    protected:
        // Basic empty backward overrirde from LayerBase, so correct method gets called in specialized layer 
        void backward(std::vector<std::any>& layer_output, std::vector<std::any>& layer_linear_outputs, std::vector<std::any>& layer_weights, 
                        std::vector<std::any>& layer_biases, std::vector<std::any>& gradients, const size_t& backward_pass_locator, const size_t& w_and_b_locator, 
                        const std::any& target_labels, const std::any& learning_rate) override {}
        // Basic empty override from LayerBase, so correct method gets called in specialized layer
        void deserialize_layer_data(std::ifstream& in) override {}
        // Basic empty forward overrirde from LayerBase, so correct method gets called in specialized layer
        void forward(std::any& input, std::any& output, std::any& linear_output, std::any& weights, std::any& bias) override {}
        // Virtual declaration of activation function application, to be implemented in applicable specialized layers
        virtual Mat3d<DataType> activation_process(const Mat3d<DataType>& mat, const Activation& activation_function) {return Mat3d<DataType>();}
        // Virtual method override declaration of activation function application(std::vector)
        virtual std::vector<DataType> activation_process(const std::vector<DataType>& vec, const Activation& activation_function) {return std::vector<DataType>();}
        // Virtual declaration of average pooling process for average pooling layer, implemented in average pooling specialized layer
        virtual Mat3d<DataType> average_pool_process(const Mat3d<DataType>& mat, const int& size, const int& stride, const int& depth) {return Mat3d<DataType>();}
        // Virtual declaration of convolution process for convolutional layer, implemented in convolutional specialized layer
        virtual Mat3d<DataType> conv_2d_process(const Mat3d<DataType>& mat, const Mat3d<DataType>& kernel_mats, const std::vector<DataType>& bias_vec, 
                                                const Padding& padding, const int& kernel_size, const int& number_of_filters) {return Mat3d<DataType>();}
        // Virtual declaration of flattening process for flatten layer, implemented in flattening specialized layer
        virtual std::vector<DataType> flatten_process(const Mat3d<DataType>& mat) {return std::vector<DataType>();}
        // Declaration of virtual get type method
        virtual LayerType get_layer_type() { return NONE; }
        // Virtual declaration of method that cuts 2d matrix from given 3d matrix, implemented on applicable specialized layers
        virtual Mat2d<DataType> get_2d_block_from_mat(const Mat3d<DataType>& mat, const int& block_size, const size_t& offset, 
                                                        std::pair<size_t, size_t>& output_loc, const size_t& depth) {return Mat2d<DataType>();}
        // Virtual declaration of max pooling process for max pooling layer, implemented in max pooling specialized layer
        virtual Mat3d<DataType> max_pool_process(const Mat3d<DataType>& mat, const int& size, const int& stride, const int& depth) {return Mat3d<DataType>();}
        // Basic empty predict override from LayerBase, so correct method gets called in specialized layer
        void predict(std::any& input, std::any& output, std::any& weights, std::any& bias) override {}
        // Basic empty serialize override from LayerBase, so correct method gets called in specialized layer
        void serialize_layer_data(std::ofstream& out) override {}
    };

    // Specialized Layer Class that creates Batch Normalization layer
    template <typename InputType, typename OutputType, typename DataType>
    class BatchNormalization : public Layer<InputType,OutputType,DataType>
    {
    private:

    public:


    };

    // Specialized Layer Class that creates Convolutional layer
    template <typename InputType, typename OutputType, typename DataType>
    class Conv2D : public Layer<InputType,OutputType,DataType>
    {
    private:
        const bool m_is_input_layer; // Member variable that flags if constructed layer is the input layer
        LayerType m_layer_type = CONV2D; // Member variable to hold layer typing
        Activation m_activation_function; // Member Variable to hold passed activation function method
        Padding m_padding; // Member Variable to hold passed padding to layer
        int m_number_of_filters; // Member Variable to hold passed number of filters
        int m_kernel_size; // Member Variable to hold kernel matrix size, square matrix so only 1 value needed
        std::vector<DataType> m_bias_vector; // Member variable to holl bias vector
        Mat3d<DataType> m_feature_maps; // Member Variable to hold current created feature map
        Mat3d<DataType> m_activated_feature_maps; // Member Variable to hold current activated feature map
        Mat2d<DataType> m_filter{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}; // !!! Testing only !!!
        InputType m_input; // !!! Testing only !!!

        // Method that applies activation function 
        Mat3d<DataType> activation_process(const Mat3d<DataType>& mat, const Activation& activation_function) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat3d<DataType>();
            }

            switch (activation_function)
            {
            case RELU:
                return NumPP::relu(mat);
            case TANH:
                return NumPP::tanh(mat);
            default:
                return Mat3d<DataType>();
            }
        }

        // Method that computes and returns gradient of loss with respect to input of Layer 
        // Not necessary if current layer is the input layer for network
        InputType compute_dA(const OutputType& dZ_L_4, const Mat3d<DataType>& weights, const InputType& A_L_5)
        {
            if (dZ_L_4.empty() || weights.empty() || A_L_5.empty()) {
                std::cerr << "Error: Input is empyt" << std::endl;
                return InputType();
            }

            // Rotate weight by 180 degrees 
            Mat3d<DataType> flipped_weights;
            for (size_t i = 0; i < weights.size(); i++) {
                flipped_weights.push_back(NumPP::flip_matrix<DataType>(weights[i], true, true));
            }

            const size_t batch_size = A_L_5.size(); // Get number of examples in current batch
            const size_t height = A_L_5[0].size(); // Get number of height (rows) of examples in batch
            const size_t width = A_L_5[0][0].size(); // Get number of width (columns) of examples in batch
            const size_t input_channels = A_L_5[0][0][0].size(); // Get number of channels (depth) of examples in batch
            InputType dA = NumPP::zeros<DataType>(batch_size, height, width, input_channels); // Initialize dA with zeros matrix of same dimensions as layer input

            // Iterate over batch
            for (size_t i = 0; i < batch_size; i++) {
                // Iterate over dA rows (height)
                for (size_t h = 0; h < height; h++) {
                    // Iterate over dA columns (width)
                    for (size_t w = 0; w < width; w++) {
                        // Iterate over dA channels
                        for (size_t c = 0; c < input_channels; c++) {
                            // Iterate each filter for current layer
                            for (size_t f = 0; f < m_number_of_filters; f++) {
                                // Iterate through filters rows (height)
                                for (size_t h_prime = 0; h_prime < m_kernel_size; h_prime++) {
                                    // Iterate through filters columns (witdth)
                                    for (size_t w_prime = 0; w_prime < m_kernel_size; w_prime++) {
                                        // Calculate corresponding position in dZ_L_4
                                        size_t h_z = h + h_prime - 1;
                                        size_t w_z = w + w_prime - 1;
                                        // Ensure we are within bounds of dZ_L_4
                                        if ((h_z >= 0 && h_z < height) && (w_z >= 0 && w_z < width)) {
                                            // Get corresponding value in dZ_L_4 for current position (i,h_z,w_z,f)
                                            DataType dZ_value = dZ_L_4[i][h_z][w_z][f];
                                            // Get corresponding value in rotated filter W_L
                                            DataType w_value = flipped_weights[f][h_prime][w_prime];
                                            // Accumulate gradient value at position (i,h,w,c)
                                            dA[i][h][w][c] += (dZ_value * w_value);  
                                        }
                                    }       
                                }
                            }    
                        }
                    }
                }
            }
            
            // Return computed gradient
            return dA;
        }

        // Method that computes and returns gradient of loss with respect to bias
        std::vector<DataType> compute_db(const OutputType& dZ_L_4)
        {
            if (dZ_L_4.empty()) {
                std::cerr << "Error: Input is empyt" << std::endl;
                return std::vector<DataType>();
            }

            std::vector<DataType> db = NumPP::zeros<DataType>(m_number_of_filters); // Create vector with the size equals to number of filters

            for (size_t f = 0; f < db.size(); f++) { // Loop over filters = channels
                for (size_t i = 0; i < dZ_L_4.size(); i++) { // Loop over all examples in batch
                    for (size_t h = 0; h < dZ_L_4[i].size(); h++) { // Loop over current example height
                        for (size_t w = 0; w < dZ_L_4[i][h].size(); w++) { // Loop over current example width
                            // Get sum of all examples in current channel f and 
                            // store sum to corresponding filter in bias vector
                            db[f] += dZ_L_4[i][h][w][f];
                        }
                    }
                }
            }

            return db;
        }

        // Method that computes and returns gradient of loss with respect to weigths (Filter=Kernels)
        Mat3d<DataType> compute_dW(const OutputType& dZ_L_4, const InputType& A_L_5)
        {
            if (dZ_L_4.empty() || A_L_5.empty()) {
                std::cerr << "Error: Input is empyt" << std::endl;
                return Mat3d<DataType>();
            }

            Mat3d<DataType> dW = NumPP::zeros<DataType>(m_number_of_filters, m_kernel_size, m_kernel_size); // Initialize dW with zeros and same dimensions as layer weight matrix
            const size_t batch_size = dZ_L_4.size(); // Get number of examples in current batch
            const size_t height = dZ_L_4[0].size(); // Get number of height (rows) of examples in batch
            const size_t width = dZ_L_4[0][0].size(); // Get number of width (columns) of examples in batch

            for (size_t f = 0; f < dW.size(); f++) { // Loop over each filter f
                for (size_t i = 0; i < batch_size; i++) { // Loop over each example in batch 
                    for (size_t h_prime = 0; h_prime < height; h_prime++) { // Loop over height of gradient upstream (dZ_L_4)
                        for (size_t w_prime = 0; w_prime < width; w_prime++) { // Loop over width of gradient upstream (dZ_L_4)
                            DataType dZ_value = dZ_L_4[i][h_prime][w_prime][f]; // Extract the current value of gradient at (i,h',w',f) pos
                            for (size_t h = 0; h < dW[f].size(); h++) { // Loop over each position in current filter
                                for (size_t w = 0; w < dW[f][h].size(); w++) {
                                    // Calculate the corresponding position in the input A_L_5
                                    // h_input and w_input are is the height and width position
                                    // for where the filter's current element was applied during 
                                    // forward propagation, making this step necessary for adjusting
                                    // the weights for next pass of forward propagation
                                    size_t h_input = h_prime - h; 
                                    size_t w_input = w_prime - w;
                                    // Ensure we are within bounds of input matrix
                                    if ((h_input >= 0 && h_input < height) && (w_input >= 0 && w_input < width)) {
                                        DataType A_value = A_L_5[i][h_input][w_input][0]; // Corresponding value from input
                                        dW[f][h][w] += (dZ_value * A_value); // Update gradient for current filter at position (h,w)
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return dW;
        }

        // Method that applies filter(kernel) to 3d matrix, will be used in Convolutional layer for neural network applications     
        Mat3d<DataType> conv_2d_process(const Mat3d<DataType>& mat, const Mat3d<DataType>& kernel_mats, const std::vector<DataType>& bias_vec, 
                                        const Padding& padding, const int& kernel_size, const int& number_of_filters) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat3d<DataType>();
            }
            
            size_t offset; // Create variable to store offset, determines output size

            if (padding == SAME) {
                offset = 0;
            }
            else {
                offset = kernel_size - 1;
            }

            // Create filtered matrix variable to store and return result
            Mat3d<DataType> feature_maps(mat.size()-offset, Mat2d<DataType>(mat[0].size()-offset, std::vector<DataType>(number_of_filters))); 
            std::pair<size_t, size_t> output_location; // Create pair to store output(filtered matrix) current location
            int channels = mat[0][0].size(); // Get depth of original image
            std::vector<DataType> filtered_pixel(channels); // Create vector to store convolution iteration result    

            // Loop through input matrix N times and fill feature maps based on number of filters
            for (size_t f = 0; f < number_of_filters; f++) {
                for (size_t i = 0; i < feature_maps.size(); i++) {
                    for (size_t j = 0; j < feature_maps[i].size(); j++) {
                        output_location = {i, j};
                        for (size_t k = 0; k < channels; k++) { 
                            Mat2d<DataType> block_mat = get_2d_block_from_mat(mat, kernel_size, offset, output_location, k);
                            filtered_pixel[k] = NumPP::sum_mat_mul_matching_elements<DataType>(block_mat, kernel_mats[f]);
                           /*  if (k == 0) {
                                // Blue channel
                                filtered_pixel[k] = NumPP::sum_mat_mul_matching_elements<DataType>(block_mat, kernel_mat);
                            }
                            else if (k == 1) {
                                // Green channel
                                filtered_pixel[k] = NumPP::sum_mat_mul_matching_elements<DataType>(block_mat, kernel_mat);
                            }
                            else {
                                // Red channel
                                filtered_pixel[k] = NumPP::sum_mat_mul_matching_elements<DataType>(block_mat, kernel_mat);
                            } */
                        }
                        feature_maps[i][j][f] = NumPP::get_sum_of_vector<DataType>(filtered_pixel) + bias_vec[f]; // Get sum of all channel and add bias
                    }
                }
            }

            return feature_maps;
        }
        
        // Method that returns 2d matrix block taken from 3d matrix   
        Mat2d<DataType> get_2d_block_from_mat(const Mat3d<DataType>& mat, const int& block_size, const size_t& offset, std::pair<size_t, size_t>& output_loc,
                                                const size_t& depth) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<DataType>();
            }

            Mat2d<DataType> b_mat(block_size, std::vector<DataType>(block_size)); // Create block mat variable to store result

            for (size_t i = 0; i < b_mat.size(); i++) {
                for (size_t j = 0; j < b_mat[i].size(); j++) {   
                    std::pair<size_t, size_t> og_mat_loc = {output_loc.first+i, output_loc.second+j};
                    if (og_mat_loc.first < mat.size() - offset && og_mat_loc.second < mat[i].size() - offset) {
                        b_mat[i][j] = mat[og_mat_loc.first][og_mat_loc.second][depth];
                    }
                }
            }

            return b_mat;
        }

        // Overload of update weights and biases method, !!! to be declared (Layer) !!!
        void update_weights_and_biases(std::any& weights, std::any& biases, const Mat3d<DataType>& dW, 
                                        const std::vector<DataType>& dB, const std::any& learning_rate)
        {
            if (!weights.has_value() || !biases.has_value() || !learning_rate.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }  
            
            Mat3d<DataType>* weights_input = std::any_cast<Mat3d<DataType>>(&weights); // Get weights for this layer
            std::vector<DataType>* biases_input = std::any_cast<std::vector<DataType>>(&biases); // Get biases for this layer
            const DataType* learn_rate = std::any_cast<DataType>(&learning_rate); // Get learning rate set on fit method (Neural Network)

            // Update weights for this layer
            Mat3d<DataType> weights_output = NumPP::subtract(*weights_input, NumPP::scalar_mat_mul<DataType>(dW, *learn_rate));
            // Update biases for this layer
            std::vector<DataType> biases_output = NumPP::subtract(*biases_input, NumPP::scalar_vec_mul(dB, *learn_rate));

            weights = weights_output;
            biases = biases_output;
        }

    public:
        // Class constructor to receive correct hyperparameters
        Conv2D(const int& number_of_filters, const int& kernel_size,
                const Activation& activation_function, const Padding& padding,
                    const bool& input_layer_flag = false):
            Layer<InputType, OutputType, DataType>(true), 
            m_number_of_filters(number_of_filters),
            m_kernel_size(kernel_size),
            m_activation_function(activation_function),
            m_padding(padding),
            m_is_input_layer(input_layer_flag)
        {}

        // Specialized override of layer backward method declared in base class (LayerBase)
        void backward(std::vector<std::any>& layer_output, std::vector<std::any>& layer_linear_outputs, std::vector<std::any>& layer_weights, 
                        std::vector<std::any>& layer_biases, std::vector<std::any>& gradients, const size_t& backward_pass_locator, const size_t& w_and_b_locator,
                        const std::any& target_labels, const std::any& learning_rate) override 
        {
            // Gradient descent for input layer
            if (m_is_input_layer) {
                
                OutputType* dA_L_4 = std::any_cast<OutputType>(&gradients[gradients.size()-1]); // Get gradient propagated from previously executed layer (Avrg Pooling Layer)
                OutputType* Z_L_4 = std::any_cast<OutputType>(&layer_linear_outputs[backward_pass_locator]); // Get the linear output of current layer (ReLU Conv Layer L-4)
                InputType A_L_5 = m_input; // Get input of current layer, for now, training batch

                // Compute gradient of the loss with respect to Convolutional Layer Linear Output (L-4)
                OutputType dZ_L_4 = NumPP::mat_mul_matching_elements<DataType>(*dA_L_4, NumPP::relu_derivative(*Z_L_4));
                // Compute gradient of the loss with respect to to weights of convolutional layer (L-4)
                Mat3d<DataType> dW_L_4 = compute_dW(dZ_L_4, A_L_5);
                // Compute gradient of the loss with respect to to biases of convolutional layer (L-4)
                std::vector<DataType> db_L_4 = compute_db(dZ_L_4);
                // Update weights and biases for next forward pass
                update_weights_and_biases(layer_weights[w_and_b_locator], layer_biases[w_and_b_locator], dW_L_4, db_L_4, learning_rate);
                return;
            }
            // Gradient descent for non input layers
            Mat3d<DataType>* W_L_4 = std::any_cast<Mat3d<DataType>>(&layer_weights[w_and_b_locator]); // Get weights matrix for convolutional layer
            OutputType* dA_L_4 = std::any_cast<OutputType>(&gradients[gradients.size()-1]); // Get gradient propagated from previously executed layer (Avrg Pooling Layer)
            OutputType* Z_L_4 = std::any_cast<OutputType>(&layer_linear_outputs[backward_pass_locator]); // Get the linear output of current layer (ReLU Conv Layer L-4)
            InputType* A_L_5 = std::any_cast<InputType>(&layer_output[backward_pass_locator-1]); // Get activated output from forward pass previous layer

            // Compute gradient of the loss with respect to Convolutional Layer Linear Output (L-4)
            OutputType dZ_L_4 = NumPP::mat_mul_matching_elements<DataType>(*dA_L_4, NumPP::relu_derivative(*Z_L_4));
            // Compute gradient of the loss with respect to to weights of convolutional layer (L-4)
            Mat3d<DataType> dW_L_4 = compute_dW(dZ_L_4, *A_L_5);
            // Compute gradient of the loss with respect to to biases of convolutional layer (L-4)
            std::vector<DataType> db_L_4 = compute_db(dZ_L_4);
            // Compute gradient of the loss with respect to the input of previous layer 
            InputType dA_L_5 = compute_dA(dZ_L_4, *W_L_4, *A_L_5);
            // Update weights and biases for next forward pass
            update_weights_and_biases(layer_weights[w_and_b_locator], layer_biases[w_and_b_locator], dW_L_4, db_L_4, learning_rate);
            // Store gradients calculated in this layer to propagate backwards to previous layers
            // This is important, otherwise we will need to make the same calculations over and over again, 
            // greatly impacting performance, look into optimizing later
            gradients.push_back(dA_L_5);
        }
        
        // Specialized override of layer deserialize data method declared in base class (LayerBase)
        void deserialize_layer_data(std::ifstream& in) override 
        {
            std::streampos stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
            in.seekg(stream_pos);

            in.read(reinterpret_cast<char*>(&m_number_of_filters), sizeof(m_number_of_filters)); // Read and assign number of filters to member variable

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Number of filters Read: " << std::to_string(m_number_of_filters) << std::endl;

            stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
            in.seekg(stream_pos);

            in.read(reinterpret_cast<char*>(&m_kernel_size), sizeof(m_kernel_size)); // Read and assign kernel size to member variable

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Kernel Size Read: " << std::to_string(m_kernel_size) << std::endl;

            stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
            in.seekg(stream_pos);

            in.read(reinterpret_cast<char*>(&m_activation_function), sizeof(m_activation_function)); // Read and assign Layer's Activation function to member variable

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Activation function Read: " << std::to_string(m_activation_function) << std::endl;

            stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
            in.seekg(stream_pos);

            in.read(reinterpret_cast<char*>(&m_padding), sizeof(m_padding)); // Read and assign Padding type to member variable

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Padding Read: " << std::to_string(m_padding) << std::endl; 
        }

        // Specialized override of layer forward method declared in base class (LayerBase)
        void forward(std::any& input, std::any& output, std::any& linear_output, std::any& weights, std::any& bias) override
        {
            if (!input.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }   

            if (!weights.has_value()) {
                weights = Mat3d<DataType>();
            }

            if (!bias.has_value()) {
                bias = std::vector<DataType>();
            }

            output = OutputType();
            
            InputType* typed_input = std::any_cast<InputType>(&input);
            OutputType* typed_output = std::any_cast<OutputType>(&output);
            OutputType non_activated_output;
            m_input = *typed_input;
            size_t size = typed_input->size();
            Mat3d<DataType>* weights_input = std::any_cast<Mat3d<DataType>>(&weights);       
            std::vector<DataType>* biases_input = std::any_cast<std::vector<DataType>>(&bias);
       
            if (weights_input->size() == 0) {
                 for (size_t i = 0; i < m_number_of_filters; i++) {
                    weights_input->push_back(NumPP::rand<DataType>(m_kernel_size, m_kernel_size, 0.0, 1.0));
                    //weights_input->push_back(m_filter); // Testing
                }
            }     

            if (biases_input->size() == 0) {
                for (size_t i = 0; i < m_number_of_filters; i++) {
                    biases_input->push_back(0);
                }
            }
     
            // Loop through image and apply convolution process
            for (size_t i = 0; i < size; i++) {
                m_feature_maps = conv_2d_process((*typed_input)[i], *weights_input, *biases_input, m_padding, m_kernel_size, m_number_of_filters); 
                m_activated_feature_maps = activation_process(m_feature_maps, m_activation_function);
                typed_output->push_back(m_activated_feature_maps);
                non_activated_output.push_back(m_feature_maps);
            }
  
            linear_output = non_activated_output;  
                    
        }

        // Override for get layer type method declared in base class (LayerBase)
        LayerType get_layer_type() override
        {
            return m_layer_type;
        }

        // Override for predict method declared in base class (LayerBase)
        void predict(std::any& input, std::any& output, std::any& weights, std::any& bias) override 
        {
            if (!input.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            } 

            // For now hardcoded to image predctions 
            Mat3d<DataType>* typed_input = std::any_cast<Mat3d<DataType>>(&input);
            size_t size = typed_input->size();
            Mat3d<DataType>* weights_input = std::any_cast<Mat3d<DataType>>(&weights);       
            std::vector<DataType>* biases_input = std::any_cast<std::vector<DataType>>(&bias);

            m_feature_maps = conv_2d_process(*typed_input, *weights_input, *biases_input, m_padding, m_kernel_size, m_number_of_filters); 
            m_activated_feature_maps = activation_process(m_feature_maps, m_activation_function);

            output = m_activated_feature_maps;
        }

        // Specialize override of layer serializatin method declared base class (LayerBase)
        void serialize_layer_data(std::ofstream& out) override 
        {
            // Write data into passed binary file
            out.write(reinterpret_cast<const char*>(&m_number_of_filters), sizeof(m_number_of_filters)); // Write number of filters data

            // Check if data stream is in a good state (No errors occured)
            if (!out) {
                throw std::runtime_error("Output stream is in a bad state. Cannot serialize Conv2D layer");
            }

            std::cout << "Number of filters Write: " << std::to_string(m_number_of_filters) << std::endl;

            out.write(reinterpret_cast<const char*>(&m_kernel_size), sizeof(m_kernel_size)); // Write number of filters data

            // Check if data stream is in a good state (No errors occured)
            if (!out) {
                throw std::runtime_error("Output stream is in a bad state. Cannot serialize Conv2D layer");
            }

            std::cout << "Kernel Size Write: " << std::to_string(m_kernel_size) << std::endl;

            out.write(reinterpret_cast<const char*>(&m_activation_function), sizeof(m_activation_function)); // Write selected activation function data

            // Check if data stream is in a good state (No errors occured)
            if (!out) {
                throw std::runtime_error("Output stream is in a bad state. Cannot serialize Conv2D layer");
            }

            std::cout << "Activation function Write: " << std::to_string(m_activation_function) << std::endl;

            out.write(reinterpret_cast<const char*>(&m_padding), sizeof(m_padding)); // Write selected padding data
            
            // Check if data stream is in a good state (No errors occured)
            if (!out) {
                throw std::runtime_error("Output stream is in a bad state. Cannot serialize Conv2D layer");
            } 

            std::cout << "Padding Write: " << std::to_string(m_padding) << std::endl;
        }
    };

    // Specialized Layer Class that creates Max Pooling layer
    template <typename InputType, typename OutputType, typename DataType>
    class MaxPooling2D : public Layer<InputType,OutputType,DataType>
    {
    private:
        const LayerType m_layer_type = MAXPOOLING2D; // Member variable to hold layer typing
        const int m_size; // Member variable that stores how many pixels to pool
        const int m_stride; // Member variable that stores stride to traverse through matrix
        int m_depth; // Member variable that stores matrix depth
        Mat3d<DataType> m_pooled_mat; // Member variable that stores current pooled matrix

        // Method that propagates gradient to next layer
        InputType backward_process(const InputType& a_Previous, const OutputType& dA_current)
        {
            auto result_shape = NumPP::get_shape(a_Previous);
            // Need to change to dA_previous
            InputType result = NumPP::zeros<DataType>(result_shape[0], result_shape[1], result_shape[2], result_shape[3]);        
            auto da_current_shape = NumPP::get_shape(dA_current);

            for (size_t i = 0; i < da_current_shape[0]; i++) {
                std::pair<size_t, size_t> og_mat_loc = {0, 0};
                for (size_t d = 0; d < da_current_shape[3]; d++) {
                    og_mat_loc.first = 0;
                    for (size_t h = 0; h < da_current_shape[1]; h++) {
                        og_mat_loc.second = 0;
                        for (size_t w = 0; w < da_current_shape[2]; w++) {
                            Mat2d<DataType> block_mat = get_2d_block_from_mat(a_Previous[i], m_size, 0, og_mat_loc, d);
                            std::pair<size_t, size_t> max_index = NumPP::get_max_element_pos(block_mat);
                            result[i][og_mat_loc.first + max_index.first][og_mat_loc.second + max_index.second][d] += dA_current[i][h][w][d];
                            og_mat_loc.second += m_stride;
                        }
                        og_mat_loc.first += m_stride;
                    }
                }
            }

            return result;
        }

        // Method that returns 2d matrix block taken from 3d matrix   
        Mat2d<DataType> get_2d_block_from_mat(const Mat3d<DataType>& mat, const int& block_size, const size_t& offset, std::pair<size_t, size_t>& output_loc,
                                                const size_t& depth) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<DataType>();
            }

            Mat2d<DataType> b_mat(block_size, std::vector<DataType>(block_size)); // Create block mat variable to store result

            for (size_t i = 0; i < b_mat.size(); i++) {
                for (size_t j = 0; j < b_mat[i].size(); j++) {   
                    std::pair<size_t, size_t> og_mat_loc = {output_loc.first+i, output_loc.second+j};
                    if (og_mat_loc.first < mat.size() - offset && og_mat_loc.second < mat[i].size() - offset) {
                        b_mat[i][j] = mat[og_mat_loc.first][og_mat_loc.second][depth];
                    }
                }
            }

            return b_mat;
        }

        // Method that iterates through 3d matrix and reduces it by selecting max value for each iteration
        Mat3d<DataType> max_pool_process(const Mat3d<DataType>& mat, const int& size, const int& stride, const int& depth) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat3d<DataType>();
            }

            Mat3d<DataType> pool_mat(mat.size()/2, Mat2d<DataType>(mat[0].size()/2, std::vector<DataType>(depth)));
            std::pair<size_t, size_t> og_mat_loc = {0, 0};

            for (size_t f = 0; f < depth; f++) {
                og_mat_loc.first = 0;
                for (size_t i = 0; i < pool_mat.size(); i++) {
                    og_mat_loc.second = 0;
                    for (size_t j = 0; j < pool_mat[i].size(); j++) {
                        Mat2d<DataType> block_mat = get_2d_block_from_mat(mat, size, 0, og_mat_loc, f);
                        pool_mat[i][j][f] = NumPP::find_max_value(block_mat);
                        og_mat_loc.second += stride;
                    }     
                }
            }

            return pool_mat;
        } 

    public:
        // Class constructor to receive correct hyperparameters
        MaxPooling2D(const int& size, const int& stride):
            Layer<InputType, OutputType, DataType>(false), 
            m_size(size),
            m_stride(stride)
        {}

        // Specialized override of layer forward method declared in base class (LayerBase)
        void forward(std::any& input, std::any& output, std::any& linear_output, std::any& weights, std::any& bias) override
        {
            if (!input.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }  

            output = OutputType();

            InputType* typedInput = std::any_cast<InputType>(&input);
            OutputType* typedOutput = std::any_cast<OutputType>(&output);

            m_depth = (*typedInput)[0][0][0].size();
            size_t size = typedInput->size();

            // Loop through image and apply max pooling process
            for (size_t i = 0; i < size; i++) {
                m_pooled_mat = max_pool_process((*typedInput)[i], m_size, m_stride, m_depth);
                typedOutput->push_back(m_pooled_mat);
            }
        } 

        // Override for get layer type method declared in base class (LayerBase)
        LayerType get_layer_type() override
        {
            return m_layer_type;
        }
   
    };

    // Specialized Layer Class that creates Average Pooling layer
    template <typename InputType, typename OutputType, typename DataType>
    class AveragePooling2d : public Layer<InputType,OutputType,DataType>
    {
    private:
        LayerType m_layer_type = AVERAGEPOOLING2D; // Member variable to hold layer typing
        int m_size; // Member variable that stores how many pixels to pool
        int m_stride; // Member variable that stores stride to traverse through matrix 
        int m_depth; // Member variable that stores depth of current matrix 
        Mat3d<DataType> m_pooled_mat; // Member variable that stores current pooled matrix 

        // Method that iterates through 3d matrix and reduces it by averaging the values contained in each iteration(matrix block)
        Mat3d<DataType> average_pool_process(const Mat3d<DataType>& mat, const int& size, const int& stride, const int& depth) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat3d<DataType>();
            }

            Mat3d<DataType> pool_mat(mat.size()/2, Mat2d<DataType>(mat[0].size()/2, std::vector<DataType>(depth)));
            std::pair<size_t, size_t> og_mat_loc = {0, 0};

            for (size_t f = 0; f < depth; f++) {
                og_mat_loc.first = 0;
                for (size_t i = 0; i < pool_mat.size(); i++) {
                    og_mat_loc.second = 0;
                    for (size_t j = 0; j < pool_mat[i].size(); j++) {
                        Mat2d<DataType> block_mat = get_2d_block_from_mat(mat, size, 0, og_mat_loc, f);
                        pool_mat[i][j][f] = NumPP::get_average(block_mat);
                        og_mat_loc.second += stride;
                    }                
                }
            }

            return pool_mat;
        } 

        // Method that distributes the gradients from previous layers in backward propagation equally to pooling windows 
        void backward_propagation_process(InputType& dA_L_4, const OutputType* dA_L_3) 
        {
            auto pooled_mat_shape = NumPP::get_shape(*dA_L_3); // Get current layer output dimensions
            //std::pair<size_t, size_t> pooled_mat_loc = {0, 0}; // Pair to keep track of spacial dimensions to select correct pooling windows
            DataType pooling_window_size = m_size * m_size; // Get pooling window size for average distribution

            for (size_t i = 0; i < pooled_mat_shape[0]; i++) { // Iterate through examples in batch
                for (size_t c = 0; c < pooled_mat_shape[3]; c++) { // Iterate through channels in batch
                    for (size_t j = 0; j < pooled_mat_shape[1]; j++) { // Iterate through rows in batch
                        for (size_t k = 0; k < pooled_mat_shape[2]; k++) { // Iterate through columns in batch
                            DataType gradient_value = (*dA_L_3)[i][j][k][c]; // Get gradient from previous layer
                            DataType average_value = gradient_value / pooling_window_size; // Average gradient value to distribute gradient equally throughout window
                            std::vector<size_t> pos = {j,k,c}; // Get current position in pooled mat
                            distribute_gradient_to_pooling_window(dA_L_4[i], pos, average_value);
                        }
                    }
                }
            }
        }

        // Method that distributes average gradient value equally throughout current pooling window
        void distribute_gradient_to_pooling_window(Mat3d<DataType>& mat, const std::vector<size_t>& pos, const DataType& value_to_distribute)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }
            
            //const std::pair<size_t, size_t> pool_mat_loc(pos[0], pos[1]); // Get current spacial dimension (Rows and Columns respectively)
            const size_t start_row = pos[0] * 2; // Get starting row index to select pooling window
            const size_t start_col = pos[1] * 2; // Get starting column index to select pooling window
            const size_t channel = pos[2]; // Get current channel

            for (size_t i = start_row; i < start_row + m_stride; i++) {
                for (size_t j = start_col; j < start_col + m_stride; j++) {
                    mat[i][j][channel] = value_to_distribute;
                }
            }      
        }

        // Method that returns 2d matrix block taken from 3d matrix   
        Mat2d<DataType> get_2d_block_from_mat(const Mat3d<DataType>& mat, const int& block_size, const size_t& offset, std::pair<size_t, size_t>& output_loc,
                                                const size_t& depth) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<DataType>();
            }

            Mat2d<DataType> b_mat(block_size, std::vector<DataType>(block_size)); // Create block mat variable to store result

            for (size_t i = 0; i < b_mat.size(); i++) {
                for (size_t j = 0; j < b_mat[i].size(); j++) {   
                    std::pair<size_t, size_t> og_mat_loc = {output_loc.first+i, output_loc.second+j};
                    if (og_mat_loc.first < mat.size() - offset && og_mat_loc.second < mat[i].size() - offset) {
                        b_mat[i][j] = mat[og_mat_loc.first][og_mat_loc.second][depth];
                    }
                }
            }

            return b_mat;
        }
        
    public:
        // Class constructor to receive correct hyperparameters
        AveragePooling2d(const int& size, const int& stride):
            Layer<InputType, OutputType, DataType>(false), 
            m_size(size), 
            m_stride(stride)
        {}

        // Specialized override of layer backward method declared in base class (LayerBase)
        void backward(std::vector<std::any>& layer_output, std::vector<std::any>& layer_linear_outputs, std::vector<std::any>& layer_weights, 
                        std::vector<std::any>& layer_biases, std::vector<std::any>& gradients, const size_t& backward_pass_locator, const size_t& w_and_b_locator, 
                        const std::any& target_labels, const std::any& learning_rate) override 
        {
            if (layer_output.empty() || layer_weights.empty() || layer_biases.empty() || !target_labels.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }  

            InputType* A_L_4 = std::any_cast<InputType>(&layer_output[backward_pass_locator-1]); // Get output of previous layer output (Conv 2d Layer L-4)
            OutputType* dA_L_3 = std::any_cast<OutputType>(&gradients[gradients.size()-1]); // Get reshaped gradient calculated in previously executed layer (Flatten Layer L-3)
            auto input_shape = NumPP::get_shape<DataType>(*A_L_4); // Get shape of input to current layer, used to initialize gradient to be propagated

            // Initialize gradient of loss with respect to output of previous layer (Conv 2d Layer L-4)
            InputType dA_L_4 = NumPP::zeros<DataType>(input_shape[0],input_shape[1],input_shape[2],input_shape[3]);
            // Calculate gradient of loss with respect to output of previous layer (dA_L_4) from Output gradient (dA_L_3)
            backward_propagation_process(dA_L_4, dA_L_3);
            // Store gradients calculated in this layer to propagate backwards to previous layers
            // This is important, otherwise we will need to make the same calculations over and over again, 
            // greatly impacting performance, look into optimizing later
            gradients.push_back(dA_L_4);
        }
        
        // Specialized override of layer deserialize data method declared in base class (LayerBase)
        void deserialize_layer_data(std::ifstream& in) override 
        {
            std::streampos stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
            in.seekg(stream_pos);

            in.read(reinterpret_cast<char*>(&m_size), sizeof(m_size)); // Read and assign number of filters to member variable

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Pooling Window Size Read: " << std::to_string(m_size) << std::endl;

            stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
            in.seekg(stream_pos);

            in.read(reinterpret_cast<char*>(&m_stride), sizeof(m_stride)); // Read and assign kernel size to member variable

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Pooling Stride Read: " << std::to_string(m_stride) << std::endl;
        }

        // Specialized override of layer forward method declared in base class (LayerBase)
        void forward(std::any& input, std::any& output, std::any& linear_output, std::any& weights, std::any& bias) override
        {
            if (!input.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }  

            output = OutputType();

            InputType* typedInput = std::any_cast<InputType>(&input);
            OutputType* typedOutput = std::any_cast<OutputType>(&output);

            m_depth = (*typedInput)[0][0][0].size();
            size_t size = typedInput->size();

            for (size_t i = 0; i < size; i++) {
                m_pooled_mat = average_pool_process((*typedInput)[i], m_size, m_stride, m_depth);
                typedOutput->push_back(m_pooled_mat);
            }
            
        }   
    
        // Override for get layer type method declared in base class (LayerBase)
        LayerType get_layer_type() override
        {
            return m_layer_type;
        }

        // Override for predict method declared in base class (LayerBase)
        void predict(std::any& input, std::any& output, std::any& weights, std::any& bias) override 
        {
            if (!input.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            } 

            // For now hardcoded
            Mat3d<DataType>* typedInput = std::any_cast<Mat3d<DataType>>(&input);
            m_depth = (*typedInput)[0][0].size();
            size_t size = typedInput->size();

            m_pooled_mat = average_pool_process(*typedInput, m_size, m_stride, m_depth);

            output = m_pooled_mat;
        }

        // Specialize override of layer serializatin method declared base class (LayerBase)
        void serialize_layer_data(std::ofstream& out) override 
        {
            // Write data into passed binary file
            out.write(reinterpret_cast<const char*>(&m_size), sizeof(m_size)); // Write number of filters data

            // Check if data stream is in a good state (No errors occured)
            if (!out) {
                throw std::runtime_error("Output stream is in a bad state. Cannot serialize Conv2D layer");
            }

            std::cout << "Pooling Window Size Write: " << std::to_string(m_size) << std::endl;

            out.write(reinterpret_cast<const char*>(&m_stride), sizeof(m_stride)); // Write number of filters data

            // Check if data stream is in a good state (No errors occured)
            if (!out) {
                throw std::runtime_error("Output stream is in a bad state. Cannot serialize Conv2D layer");
            }

            std::cout << "Pooling Stride Write: " << std::to_string(m_stride) << std::endl;
        }

    };

    // Specialized Layer Class that creates Flattening layer
    template <typename InputType, typename OutputType, typename DataType>
    class Flatten : public Layer<InputType,OutputType,DataType>
    {
    private:
        const LayerType m_layer_type = FLATTEN; // Member variable to hold layer typing
        std::vector<DataType> m_image_vector; // Member variable to store current image vector

        //Method that flattens 3d matrix into vector
        std::vector<DataType> flatten_process(const Mat3d<DataType>& mat) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<DataType>();
            }

            int depth = mat[0][0].size();
            std::vector<DataType> output_vector;

            for (size_t d = 0; d < depth; d++) {
                for (size_t i = 0; i < mat.size(); i++) {
                    for (size_t j = 0; j < mat[i].size(); j++) {
                        output_vector.push_back(mat[i][j][d]);
                    }
                }
            } 
            
            return output_vector;
        }
    
    public:
        Flatten() : Layer<InputType, OutputType, DataType>(false) {}

        // Specialized override of layer backward method declared in base class (LayerBase)
        void backward(std::vector<std::any>& layer_output, std::vector<std::any>& layer_linear_outputs, std::vector<std::any>& layer_weights, 
                        std::vector<std::any>& layer_biases, std::vector<std::any>& gradients, const size_t& backward_pass_locator, const size_t& w_and_b_locator,
                        const std::any& target_labels, const std::any& learning_rate) override 
        {
            if (layer_output.empty() || layer_weights.empty() || layer_biases.empty() || !target_labels.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }  

            InputType* A_L_3 = std::any_cast<InputType>(&layer_output[backward_pass_locator-1]); // Get output from previous forward pass layer (Average Pooling Layer L-3)
            OutputType* dA_L_2 = std::any_cast<OutputType>(&gradients[gradients.size()-1]); // Get propagated gradient calculated in previously executed layer (ReLU dense layer = L-1)
            InputType dA_L_3 = NumPP::reshape<DataType>(*dA_L_2, NumPP::get_shape(*A_L_3)); // Reshape gradient into original shape
            // Store gradients calculated in this layer to propagate backwards to previous layers
            // This is important, otherwise we will need to make the same calculations over and over again, 
            // greatly impacting performance, look into optimizing later
            gradients.push_back(dA_L_3);
        }
        
        // Specialized override of layer forward method declared in base class (LayerBase)
        void forward(std::any& input, std::any& output, std::any& linear_output, std::any& weights, std::any& bias) override 
        {
            if (!input.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }  

            output = OutputType();

            InputType* typedInput = std::any_cast<InputType>(&input);
            OutputType* typedOutput = std::any_cast<OutputType>(&output);

            for (size_t i = 0; i < typedInput->size(); i++) {
                m_image_vector = flatten_process((*typedInput)[i]);
                typedOutput->push_back(m_image_vector);
            }
        }  

        // Override for predict method declared in base class (LayerBase)
        void predict(std::any& input, std::any& output, std::any& weights, std::any& bias) override 
        {
            if (!input.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            } 

            // For now hardcoded to image predctions 
            Mat3d<DataType>* typed_input = std::any_cast<Mat3d<DataType>>(&input);
            m_image_vector = flatten_process(*typed_input);

            output = m_image_vector;
        }

        // Override for get layer type method declared in base class (LayerBase)
        LayerType get_layer_type() override
        {
            return m_layer_type;
        }

    };

    // Specialized Layer Class that creates Max Pooling layer
    template <typename InputType, typename OutputType, typename DataType>
    class Dense : public Layer<InputType,OutputType,DataType>
    {
    private:
        LayerType m_layer_type = DENSE; // Member variable to hold layer typing
        int m_output_size; // Member Variable to hold layer output size
        Activation m_activation_function; // Member Variable to hold passed activation function method
        std::vector<DataType> m_transformed_vector; // Member variable to store current result of linear transformation
        std::vector<DataType> m_activated_vector; // Member variable to store current result of activation function

        // Overload of activation process method to work with vectors (Layer)
        // Applies selected activation function to vector
        std::vector<DataType> activation_process(const std::vector<DataType>& vec, const Activation& activation_function) override
        {
            if (vec.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<DataType>();
            }

            switch (activation_function)
            {
            case RELU:
                return NumPP::relu(vec);
            case TANH:
                return NumPP::tanh(vec);
            case SIGMOID:
                return NumPP::sigmoid(vec);
            case SOFTMAX:
                return NumPP::softmax(vec);
            default:
                return std::vector<DataType>();
            }
        }

        // Overload of update weights and biases method, !!! to be declared (Layer) !!!
        void update_weights_and_biases(std::any& weights, std::any& biases, const Mat2d<DataType>& dW, 
                                        const std::vector<DataType>& dB, const std::any& learning_rate)
        {
            if (!weights.has_value() || !biases.has_value() || !learning_rate.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }  
            
            Mat2d<DataType>* weights_input = std::any_cast<Mat2d<DataType>>(&weights); // Get weights for this layer
            std::vector<DataType>* biases_input = std::any_cast<std::vector<DataType>>(&biases); // Get biases for this layer
            const DataType* learn_rate = std::any_cast<DataType>(&learning_rate); // Get learning rate set on fit method (Neural Network)

            // Update weights for this layer
            Mat2d<DataType> weights_output = NumPP::subtract(*weights_input, NumPP::scalar_mat_mul<DataType>(dW, *learn_rate));
            // Update biases for this layer
            std::vector<DataType> biases_output = NumPP::subtract(*biases_input, NumPP::scalar_vec_mul(dB, *learn_rate));

            weights = weights_output;
            biases = biases_output;
        }

    public:
        // Class constructor to receive correct hyperparameters
        Dense(const int& output_size, const Activation& activation_function):
            Layer<InputType, OutputType, DataType>(true),
            m_output_size(output_size),
            m_activation_function(activation_function)
        {}

        // Specialized override of layer backward method declared in base class (LayerBase)
        void backward(std::vector<std::any>& layer_output, std::vector<std::any>& layer_linear_outputs, std::vector<std::any>& layer_weights, 
                        std::vector<std::any>& layer_biases, std::vector<std::any>& gradients, const size_t& backward_pass_locator, const size_t& w_and_b_locator, 
                        const std::any& target_labels, const std::any& learning_rate) override 
        {
            if (layer_output.empty() || layer_weights.empty() || layer_biases.empty() || !target_labels.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }  

            // Output layer backward propagation, will be optimized in the future
            if (m_activation_function == SOFTMAX) {
                const OutputType* Y = std::any_cast<OutputType>(&target_labels); // Get true labels matrix
                const DataType batch_size = Y->size(); // Get number of samples in batch, used for data normalization
                OutputType* W_L = std::any_cast<OutputType>(&layer_weights[w_and_b_locator]); // Get weights matrix for softmax layer (L)
                InputType* A_L_1 = std::any_cast<InputType>(&layer_output[backward_pass_locator-1]); // Get activated output of previous layer (L-1)
                OutputType* A_L = std::any_cast<OutputType>(&layer_output[backward_pass_locator]); // Get activated output of sofmax layer (L)         

                // Calculate gradient of loss with respect to output
                // In this case, Softmax Activation function + Cross-Entropy Loss, the gradients of loss
                // with respect to linear and activated outputs are the same
                Mat2d<DataType> dZ_L = NumPP::subtract<DataType>(*A_L, *Y);
                // Calculate gradient of loss with respect to weights
                Mat2d<DataType> dW_L = NumPP::scalar_mat_mul(NumPP::dot(NumPP::transpose(*A_L_1), dZ_L), 1/batch_size);
                // Calculate gradient of loss with respect to biases
                std::vector db_L = NumPP::scalar_vec_mul(NumPP::sum(dZ_L, 1), 1/batch_size);
                // Calculate gradient of loss with respect to previous forward pass layer activated input
                Mat2d<DataType> dA_L_1 = NumPP::dot(dZ_L, NumPP::transpose(*W_L));
                // Update weights and biases for next pass
                update_weights_and_biases(layer_weights[w_and_b_locator], layer_biases[w_and_b_locator], dW_L, db_L, learning_rate);
                
                // Store gradients calculated in this layer to propagate backwards to previous layers
                // This is important, otherwise we will need to make the same calculations over and over again, 
                // greatly impacting performance, look into optimizing later
                gradients.push_back(dA_L_1);
                return;
            }

            // Dense 1 layer backward propagation, will be optimized in the future
            const OutputType* Y = std::any_cast<OutputType>(&target_labels); // Get true labels matrix
            const DataType batch_size = Y->size(); // Get number of samples in batch, used for data normalization
            OutputType* W_L_1 = std::any_cast<OutputType>(&layer_weights[w_and_b_locator]); // Get weights matrix for ReLu dense layer (L-1)
            OutputType* dA_L_1 = std::any_cast<OutputType>(&gradients[gradients.size()-1]); // Get propagated gradient calculated in previously executed layer (Output layer = L)
            OutputType* Z_L_1 = std::any_cast<OutputType>(&layer_linear_outputs[backward_pass_locator]); // Get linear output for current layer (ReLU dense layer = L-1)
            InputType* A_L_2 = std::any_cast<InputType>(&layer_output[backward_pass_locator-1]); // Get activated output from forward pass previous layer (Flatten Layer = L-2)

            // Calculate gradient of loss with respect to linear output of current layer
            OutputType dZ_L_1 = NumPP::mat_mul_matching_elements(*dA_L_1, NumPP::relu_derivative(*Z_L_1));
            // Calculate gradient of loss with respect to weights 
            Mat2d<DataType> dW_L_1 = NumPP::scalar_mat_mul(NumPP::dot(NumPP::transpose(*A_L_2), dZ_L_1), 1/batch_size);
            // Calculate gradient of loss with respect to biases
            std::vector db_L_1 = NumPP::scalar_vec_mul(NumPP::sum(dZ_L_1, 1), 1/batch_size);
            // Calculate gradient of loss with respect to previous forward pass layer activated output
            Mat2d<DataType> dA_L_2 = NumPP::dot(dZ_L_1, NumPP::transpose(*W_L_1));
            // Update weights and biases for next pass
            update_weights_and_biases(layer_weights[w_and_b_locator], layer_biases[w_and_b_locator], dW_L_1, db_L_1, learning_rate);

            // Store gradients calculated in this layer to propagate backwards to previous layers
            // This is important, otherwise we will need to make the same calculations over and over again, 
            // greatly impacting performance, look into optimizing later
            gradients.push_back(dA_L_2);  
        }

        // Specialized override of layer deserialize data method declared in base class (LayerBase)
        void deserialize_layer_data(std::ifstream& in) override 
        {
            std::streampos stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
            in.seekg(stream_pos);

            in.read(reinterpret_cast<char*>(&m_output_size), sizeof(m_output_size)); // Read and assign number of filters to member variable

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Dense Output Size Read: " << std::to_string(m_output_size) << std::endl;

            stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
            in.seekg(stream_pos);

            in.read(reinterpret_cast<char*>(&m_activation_function), sizeof(m_activation_function)); // Read and assign kernel size to member variable

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Activation Function Read: " << std::to_string(m_activation_function) << std::endl;
        }

        // Specialized override of layer forward method declared in base class (LayerBase)
        void forward(std::any& input, std::any& output, std::any& linear_output, std::any& weights, std::any& bias) override 
        {
            if (!input.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }  

            output = OutputType();

            InputType* typedInput = std::any_cast<InputType>(&input);
            OutputType* typedOutput = std::any_cast<OutputType>(&output);
            OutputType non_activated_output;

            if (!weights.has_value()) {
                weights = Mat2d<DataType>();
            }

            if (!bias.has_value()) {
                bias = std::vector<DataType>();
            }
 
            Mat2d<DataType>* weights_input = std::any_cast<Mat2d<DataType>>(&weights);
            std::vector<DataType>* biases_input = std::any_cast<std::vector<DataType>>(&bias); 

            if (weights_input->size() == 0) {
                *weights_input = NumPP::rand<DataType>((*typedInput)[0].size(), m_output_size, 0.0, 0.5);
            } 

            if (biases_input->size()  == 0) {
                *biases_input = NumPP::zeros<DataType>(m_output_size);
            }     

            for (size_t i = 0; i < typedInput->size(); i++) {
                m_transformed_vector = NumPP::add(NumPP::dot((*typedInput)[i], *weights_input), *biases_input);
                m_activated_vector = activation_process(m_transformed_vector, m_activation_function);
                typedOutput->push_back(m_activated_vector);
                non_activated_output.push_back(m_transformed_vector); // testing
            }

            linear_output = non_activated_output; // testing
        } 
    
        // Override for get layer type method declared in base class (LayerBase)
        LayerType get_layer_type() override
        {
            return m_layer_type;
        }

        // Override for predict method declared in base class (LayerBase)
        void predict(std::any& input, std::any& output, std::any& weights, std::any& bias) override 
        {
            if (!input.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            } 

            // For now hardcoded to image predctions 
            std::vector<DataType>* typed_input = std::any_cast<std::vector<DataType>>(&input);
            Mat2d<DataType>* weights_input = std::any_cast<Mat2d<DataType>>(&weights);
            std::vector<DataType>* biases_input = std::any_cast<std::vector<DataType>>(&bias); 

            m_transformed_vector = NumPP::add(NumPP::dot(*typed_input, *weights_input), *biases_input);
            m_activated_vector = activation_process(m_transformed_vector, m_activation_function);

            output = m_activated_vector;
        }

        // Specialize override of layer serializatin method declared base class (LayerBase)
        void serialize_layer_data(std::ofstream& out) override 
        {
            // Write data into passed binary file
            out.write(reinterpret_cast<const char*>(&m_output_size), sizeof(m_output_size)); // Write number of filters data

            // Check if data stream is in a good state (No errors occured)
            if (!out) {
                throw std::runtime_error("Output stream is in a bad state. Cannot serialize Conv2D layer");
            }

            std::cout << "Dense Output Size Write: " << std::to_string(m_output_size) << std::endl;

            out.write(reinterpret_cast<const char*>(&m_activation_function), sizeof(m_activation_function)); // Write number of filters data

            // Check if data stream is in a good state (No errors occured)
            if (!out) {
                throw std::runtime_error("Output stream is in a bad state. Cannot serialize Conv2D layer");
            }

            std::cout << "Activation Function Write: " << std::to_string(m_activation_function) << std::endl;
        }

    };
 
    // Class to create, train and export neural network models
    class NeuralNetwork 
    {
    private:
        // Member variable that store base class to determine sequence of layers in forward and backward pass
        std::vector<LayerBase*> m_layers; // Member vector that stores network class sequence
        std::vector<std::any> m_outputs; // Member vector that stores each layer result to use in backward propagation
        std::vector<std::any> m_linear_outputs; // Member vector that stores layer linear output, if no activation function is the same as layer output
        std::vector<std::any> m_weights; // Member vector that, if applicable, stores weight values for layer
        std::vector<std::any> m_biases; // Member vector that, if applicable, stores biases values for layer
        std::vector<std::any> m_loss_gradients; // Member vector that stores gradient of losses for backward propagation
        std::any m_current_input; // Member blank variable to store current input in forward pass
        std::any m_current_output; // Member blank variable to store current output in forward pass
        
        // Method that applies backward propagation with given layer sequence
        void backward_pass(const std::any& target_labels, const std::any& learning_rate)
        {
            if (!target_labels.has_value())  {
                std::cerr << "Error: Params for backward pass are null" << std::endl;
                return;
            } 
            
            m_loss_gradients.clear(); // Clear loss gradients between passes

            size_t w_and_b_counter = m_weights.size() -1; // Determine which element of weights and biases vectors
            size_t backward_pass_locator = m_layers.size() - 1;

            // Iterate through layers in reverse order for backward propagation
            for (auto i = m_layers.rbegin(); i != m_layers.rend(); i++) {   
                auto& layer = *i;
                layer->backward(m_outputs, m_linear_outputs, m_weights, m_biases, m_loss_gradients, backward_pass_locator, w_and_b_counter, target_labels, learning_rate);
                if (layer->get_w_and_b() && w_and_b_counter > 0) {
                    w_and_b_counter--;
                }
                backward_pass_locator--;
            }
        }

        // Method that calculates batch loss for epoch iteration
        template <typename T>
        T calc_batch_loss(std::any& predictions, const std::any& true_labels)
        {
            if (!predictions.has_value() || !true_labels.has_value()) {
                std::cerr << "Error: Params for batch loss are null" << std::endl;
                return 0;
            } 

            Mat2d<T>* epoch_result = std::any_cast<Mat2d<T>>(&predictions); // Get typed predictions pointer
            const Mat2d<T>* labels = std::any_cast<Mat2d<T>>(&true_labels); // Get typed labels pointer
            T batch_size = epoch_result->size(); // Get number of elements in batch (i.e number of images)
            T loss = 0; // Variable to store single batch loss (i.e single image loss)
            T sum = 0; // Variable to store the sum of all losses in batch (i.e all images losses)
            T batch_loss = 0; // Variable to store whole batch loss (i.e all images losses / batch size)

            // Iterate through batch to get individual losses and get sum
            for (size_t i = 0; i < batch_size; i++) {
                loss = calc_loss((*epoch_result)[i], (*labels)[i]);
                sum += loss;
                loss = 0;
            }

            batch_loss = sum / batch_size; // Get batch loss

            return batch_loss;
        }

        // Method that calculates loss for single input
        template <typename T>
        T calc_loss(const std::vector<T>& predictions, const std::vector<T>& target)
        {
            if (predictions.empty() || target.empty()) {
                std::cerr << "Error: Params for batch loss are null" << std::endl;
                return 0;
            }

            T log_result = 0; // Variable to store current .log result
            T loss = 0; // Variable to store loss
            
            for (size_t i = 0; i < predictions.size(); i++) {
                //std::cout << "Predictions " << i << ": " << predictions[i] << std::endl;
                if (predictions[i] != 0) {
                    log_result = std::log(predictions[i]); // Apply natural logarithm to prediction values (base e)
                    //std::cout << "Log result " << i << ": " << std::to_string(log_result) << std::endl;
                    loss += log_result * target[i]; // Multiply log result with target value 
                    log_result = 0; // Clear log result
                }    
            }

            //std::cout << loss << std::endl;
            return -(loss); // Return negative value of loss
        }

        // Method that applies forward propagation with given layer sequence
        void forward_pass(std::any& input, std::any& output) 
        {
            if (!input.has_value())  {
                std::cerr << "Error: Params for forward pass are null" << std::endl;
                return;
            } 

            m_current_input = input; // Assign current input to initial input data
            m_current_output = output; // Assign output reference ot current output
            std::any m_current_linear_output; // Assign blank container to store current liner output

            int w_and_b_counter = 0; // Determine which element of weights and biases vectors
            int it_counter = 0; // Counts layer iterations to assign correct layer output to layer results vector

            // Iterate through network layers and call layer method for forward propagation
            for (auto &layer : m_layers) {
                m_current_output.reset(); // Clear current output for next layer
                m_current_linear_output.reset(); // Clear current linear output for next layer

                // Call current layer forward method
                layer->forward(m_current_input, m_current_output, m_current_linear_output, m_weights[w_and_b_counter], m_biases[w_and_b_counter]); 
                m_current_input.reset(); // Clear current input for next layer
                m_current_input = m_current_output; // Update input for next layer  
                
                // Check if layer has weights and biases flag, if so add to counter
                if (layer->get_w_and_b() && w_and_b_counter < m_weights.size()) {
                    w_and_b_counter++;
                }  

                m_outputs[it_counter] = m_current_output; // Assign current layer output to correct layer results element
                m_linear_outputs[it_counter] = m_current_linear_output;
                it_counter++; // Increase iteration counter
            }

            output = m_current_output; // Assign output of final layer to passed output reference
        }

        // Method that serialize biases of neural network so we can store it in binary format
        void serialize_biases(std::ofstream& out)
        {
            // Write the number of weight tensors (matrices) present in network
            size_t number_of_biases_tensors = m_weights.size();
            out.write(reinterpret_cast<const char*>(&number_of_biases_tensors), sizeof(number_of_biases_tensors));

            if (!out) {
                throw std::runtime_error("Error: Couldn't write data to file");
            }

            std::cout << "Number of Bias Tensors Write: " << number_of_biases_tensors << std::endl;
            
            // Iterate through each bias tensor (1D tensor = Vector)
            for (const auto& bias : m_biases) {
                const auto& vec = std::any_cast<const std::vector<double>&>(bias);
                size_t vector_size = vec.size();

                out.write(reinterpret_cast<const char*>(&vector_size), sizeof(vector_size));

                if (!out) {
                    throw std::runtime_error("Error: Couldn't write data to file");
                }

                std::cout << "Size of Bias Tensor Write: " << vector_size << std::endl;

                out.write(reinterpret_cast<const char*>(vec.data()), vector_size * sizeof(double));

                if (!out) {
                    throw std::runtime_error("Error: Couldn't write data to file");
                }
            }
        }

        // Method that serialize layers to be able to store in binary format
        void serialize_layers(std::ofstream& out)
        {
            // Write to file number of layers in network
            size_t number_of_layers = m_layers.size();
            out.write(reinterpret_cast<const char*>(&number_of_layers), sizeof(number_of_layers));

            if (!out) {
                throw std::runtime_error("Error: Couldn't write data to file");
            }

            std::cout << "Number of layers Write: " << number_of_layers << std::endl;

            // Loop through each layer in network to serialize and write to file any relevant data
            for (const auto &layer : m_layers) {
                LayerType type = layer->get_layer_type();
                out.write(reinterpret_cast<const char*>(&type), sizeof(type));
                std::cout << "Layer ID Write: " << std::to_string(type) << std::endl;
                if (!out) {
                    throw std::runtime_error("Error: Couldn't write data to file");
                }
                layer->serialize_layer_data(out);
            }
        }

        // Mathod that serialize weights of neural network so we can store it in binary format
        void serialize_weights(std::ofstream& out)
        {   
            // Write the number of weight tensors (matrices) present in network
            size_t number_of_weight_tensors = m_weights.size();
            out.write(reinterpret_cast<const char*>(&number_of_weight_tensors), sizeof(number_of_weight_tensors));

            if (!out) {
                throw std::runtime_error("Error: Couldn't write data to file");
            }

            std::cout << "Number of Weight Tensors Write: " << number_of_weight_tensors << std::endl;

            // Iterate through each weight tensor
            for (const auto &weight : m_weights) {
                // Determine shape of weight tensor
                if (weight.type() == typeid(Mat2d<double>)) {
                    // Serialize weight typing for faster loading
                    MatrixType type = MatrixType::MAT_2D;
                    out.write(reinterpret_cast<const char*>(&type), sizeof(type));

                    if (!out) {
                        throw std::runtime_error("Error: Couldn't write data to file");
                    }

                    std::cout << "Tensor Type Write: " << type << std::endl;

                    // Serialize the weight tensor data
                    const auto& tensor = std::any_cast<const Mat2d<double>&>(weight);
                    size_t rows = tensor.size();
                    size_t cols = tensor[0].size();
                    // Save tensor spacial dimensions for faster loading
                    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows)); 
                    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

                    if (!out) {
                        throw std::runtime_error("Error: Couldn't write data to file");
                    }

                    std::cout << "Tensor Rows Size Write: " << rows << std::endl;
                    std::cout << "Tensor Columns Size Write: " << cols << std::endl;

                    /// Loop through tensor to serialize data
                    for (size_t i = 0; i < rows; i++) {
                        // Write current row data into binary file
                        out.write(reinterpret_cast<const char*>(tensor[i].data()), cols * sizeof(double));

                        if (!out) {
                            throw std::runtime_error("Error: Couldn't write data to file");
                        }
                    } 
                }
                // Determine shape of weight tensor
                if (weight.type() == typeid(Mat3d<double>)) {
                    // Serialize weight typing for faster loading
                    MatrixType type = MatrixType::MAT_3D;
                    out.write(reinterpret_cast<const char*>(&type), sizeof(type));

                    if (!out) {
                        throw std::runtime_error("Error: Couldn't write data to file");
                    }

                    std::cout << "Tensor Type Write: " << type << std::endl;

                    // Serialize the weight tensor data
                    const auto& tensor = std::any_cast<const Mat3d<double>&>(weight);
                    size_t rows = tensor.size();
                    size_t cols = tensor[0].size();
                    size_t depth = tensor[0][0].size();
                    // Save tensor spacial dimensions for faster loading
                    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows)); 
                    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
                    out.write(reinterpret_cast<const char*>(&depth), sizeof(depth));

                    if (!out) {
                        throw std::runtime_error("Error: Couldn't write data to file");
                    }

                    std::cout << "Tensor Rows Size Write: " << rows << std::endl;
                    std::cout << "Tensor Columns Size Write: " << cols << std::endl;
                    std::cout << "Tensor Depth Size Write: " << depth << std::endl;

                    // Loop through current tensor to serialize data
                    for (size_t i = 0; i < rows; i++) {
                        for (size_t j = 0; j < cols; j++) {
                            // Write data of vector at row i and column j into binary file
                            out.write(reinterpret_cast<const char*>(tensor[i][j].data()), depth * sizeof(double));

                            if (!out) {
                                throw std::runtime_error("Error: Couldn't write data to file");
                            }
                        }
                    }
                }
            }
            
        }

        // Method that serialize biases of neural network so we can store it in binary format
        std::vector<std::any> deserialize_biases(std::ifstream& in)
        {
            // Get current position in data stream
            std::streampos stream_pos = in.tellg();

            // Get number of layers in network
            size_t number_of_bias_tensors;
            in.read(reinterpret_cast<char*>(&number_of_bias_tensors), sizeof(number_of_bias_tensors));

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Number of Bias Tensors Read: " << number_of_bias_tensors << std::endl;

            stream_pos = in.tellg();

            std::vector<std::any> biases;

            for (size_t i = 0; i < number_of_bias_tensors; i++) {
                std::any bias;

                // Set starting reading position of data stream for each weight
                in.seekg(stream_pos);

                size_t size_of_bias_tensor;
                in.read(reinterpret_cast<char*>(&size_of_bias_tensor), sizeof(size_of_bias_tensor));

                if (!in) {
                    throw std::runtime_error("Error reading from file.");
                }

                std::cout << "Size of Bias Tensor Read: " << size_of_bias_tensor << std::endl;

                stream_pos = in.tellg();
                in.seekg(stream_pos);

                std::vector<double> typed_bias(size_of_bias_tensor);

                in.read(reinterpret_cast<char *>(typed_bias.data()), size_of_bias_tensor * sizeof(double));

                if (!in) {
                    throw std::runtime_error("Error reading from file.");
                }

                stream_pos = in.tellg();
                in.seekg(stream_pos);

                bias = typed_bias;

                if (bias.has_value()) {
                    biases.push_back(bias);
                }

                stream_pos = in.tellg();
                in.seekg(stream_pos);
            }

            return biases;
        }

        // Method that de-serialize layers to be able to store in binary format
        std::vector<LayerBase*> deserialize_layers(std::ifstream& in)
        {
            // Get number of layers in network
            size_t number_of_layers;
            in.read(reinterpret_cast<char*>(&number_of_layers), sizeof(number_of_layers));

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Number of layers Read: " << number_of_layers << std::endl;

            // Record position of data stream
            std::streampos stream_pos = in.tellg();

            std::vector<LayerBase*> layers;

            // Loop through detected layers to detect and deserialize specific layer data
            for (size_t i = 0; i < number_of_layers; i++) {
                // Move pointer to current stream position
                in.seekg(stream_pos);

                LayerType type;
                in.read(reinterpret_cast<char*>(&type), sizeof(type));

                if (!in) {
                    throw std::runtime_error("Error reading from file.");
                }

                std::cout << "Layer " << i << " ID Read: " << std::to_string(type) << std::endl;

                stream_pos = in.tellg();

                LayerBase* layer = nullptr; // Create empty pointer to store layer type

                switch (type)
                {
                case LayerType::AVERAGEPOOLING2D:
                    layer = new AveragePooling2d<Mat4d<double>, Mat4d<double>, double>(0,0);
                    break;
                case LayerType::CONV2D:
                    layer = new Conv2D<Mat4d<double>, Mat4d<double>, double>(0, 0, Activation::SIGMOID, Padding::VALID);
                    break;
                case LayerType::DENSE:
                    layer = new Dense<Mat2d<double>, Mat2d<double>, double>(0, Activation::SIGMOID);
                    break;
                case LayerType::FLATTEN:
                    layer = new Flatten<Mat4d<double>, Mat2d<double>, double>();
                    break;
                case LayerType::MAXPOOLING2D:
                    layer = new MaxPooling2D<Mat4d<double>, Mat4d<double>, double>(0,0);
                    break;
                default:
                    break;
                }

                if (layer) {
                    layer->deserialize_layer_data(in);
                    layers.push_back(layer);
                }

                stream_pos = in.tellg();
            }

            return layers;
        }

        // Method that de-serialize layers to be able to store in binary format
        std::vector<std::any> deserialize_weights(std::ifstream& in) 
        {
            // Get current position in data stream
            std::streampos stream_pos = in.tellg();

            // Get number of layers in network
            size_t number_of_weight_tensors;
            in.read(reinterpret_cast<char*>(&number_of_weight_tensors), sizeof(number_of_weight_tensors));

            if (!in) {
                throw std::runtime_error("Error reading from file.");
            }

            std::cout << "Number of Weight Tensors Read: " << number_of_weight_tensors << std::endl;

            stream_pos = in.tellg();

            std::vector<std::any> weights;

            // Loop through detected weight tensors to detect and deserialize specific layer data
            for (size_t i = 0; i < number_of_weight_tensors; i++) {
                // Set starting reading position of data stream for each weight
                in.seekg(stream_pos);

                // Serialize weight typing for faster loading
                MatrixType type;
                in.read(reinterpret_cast<char*>(&type), sizeof(type));

                if (!in) {
                    throw std::runtime_error("Error reading from file.");
                }

                std::cout << "Tensor Type Read: " << type << std::endl;

                stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
                
                std::any tensor;
                Mat3d<double> tensor_1; // Testing only
                Mat2d<double> tensor_2; // Testing only 

                switch (type)
                {
                    case MatrixType::MAT_2D:
                        in.seekg(stream_pos);
                        size_t rows;
                        size_t cols;

                        in.read(reinterpret_cast<char*>(&rows), sizeof(rows));

                        if (!in) {
                            throw std::runtime_error("Error reading from file.");
                        }   

                        stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file

                        in.seekg(stream_pos);
                        in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

                        if (!in) {
                            throw std::runtime_error("Error reading from file.");
                        }
                        
                        stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
                        in.seekg(stream_pos);  

                        std::cout << "Tensor Rows Size Read: " << rows << std::endl;
                        std::cout << "Tensor Columns Size Read: " << cols << std::endl;

                        tensor_2 = Mat2d<double>(rows, std::vector<double>(cols));

                        for (size_t i = 0; i < rows; i++) {
                            in.read(reinterpret_cast<char *>(tensor_2[i].data()), cols * sizeof(double));

                            if (!in) {
                                throw std::runtime_error("Error reading from file.");
                            }

                            stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
                            in.seekg(stream_pos);  
                        }
                        tensor = tensor_2;
                        break;
                    case MatrixType::MAT_3D:
                        in.seekg(stream_pos);

                        size_t rows_1;
                        size_t cols_1;
                        size_t depth_1;

                        in.read(reinterpret_cast<char*>(&rows_1), sizeof(rows_1));

                        if (!in) {
                            throw std::runtime_error("Error reading from file.");
                        }   

                        stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
                        in.seekg(stream_pos);

                        in.read(reinterpret_cast<char*>(&cols_1), sizeof(cols_1));

                        if (!in) {
                            throw std::runtime_error("Error reading from file.");
                        }

                        stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
                        in.seekg(stream_pos);   

                        in.read(reinterpret_cast<char*>(&depth_1), sizeof(depth_1));

                        if (!in) {
                            throw std::runtime_error("Error reading from file.");
                        }

                        std::cout << "Tensor Rows Size Read: " << rows_1 << std::endl;
                        std::cout << "Tensor Columns Size Read: " << cols_1 << std::endl;
                        std::cout << "Tensor Depth Size Read: " << depth_1 << std::endl;

                        tensor_1 = Mat3d<double>(rows_1, Mat2d<double>(cols_1, std::vector<double>(depth_1)));

                        stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
                        in.seekg(stream_pos);  
                        
                        // Loop through current tensor to serialize data
                        for (size_t i = 0; i < rows_1; i++) {
                            for (size_t j = 0; j < cols_1; j++) {
                                in.read(reinterpret_cast<char *>(tensor_1[i][j].data()), depth_1 * sizeof(double));

                                if (!in) {
                                    throw std::runtime_error("Error reading from file.");
                                }

                                stream_pos = in.tellg(); // Get current data stream position, so we can read the correct data from file
                                in.seekg(stream_pos); 
                            }
                        }          
                        tensor = tensor_1; 
                        break;
                    default:
                        break;
                }

                if (tensor.has_value()) {
                    weights.push_back(tensor);
                }

                stream_pos = in.tellg(); 
            }

            return weights;
        }

    public:
        // Method that creates and add layer to sequence
        void add_layer(LayerBase* layerPtr)
        {
            std::any o; // Create empty any container to store layer output
            std::any lo; // Create empty any container to store layer linear output

            m_layers.push_back(layerPtr); // Store created layer into member layers vector
            m_outputs.push_back(o); // Store empty container in member layer outputs vector
            m_linear_outputs.push_back(lo);

            // Check if layer has weights and biases flag
            if (layerPtr->get_w_and_b()) {   
                std::any w; // Create empty any container for weights
                std::any b; // Create empty any container for biases 
                m_weights.push_back(w); // Store empty container in member weights vector
                m_biases.push_back(b); // Store empty container in member biases vector
            }
        }

        // Method that exports trained model and saves it into binary file 
        void export_model(const std::string& fileName)
        {
            if (fileName.empty()) {
                throw std::runtime_error("Error: Cannot create file with empty name");
            }

            std::ofstream out(fileName, std::ios::binary);

            if (!out.is_open()) {
                throw std::runtime_error("Error: Failed to open file to export model");
            }

            serialize_layers(out);
            serialize_weights(out);
            serialize_biases(out);

            // Debugging code
            //Mat3d<double>* tensor_1 = std::any_cast<Mat3d<double>>(&m_weights[0]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all((*tensor_1)[0]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all((*tensor_1)[1]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all((*tensor_1)[2]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all((*tensor_1)[3]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all((*tensor_1)[4]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //std::vector<double>* bias_0 = std::any_cast<std::vector<double>>(&m_biases[0]);
            //std::vector<double>* bias_1 = std::any_cast<std::vector<double>>(&m_biases[1]);
            //std::vector<double>* bias_2 = std::any_cast<std::vector<double>>(&m_biases[2]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all(*bias_0);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all(*bias_1);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all(*bias_2);

            out.close();            
        }

        // Method that applies training to created neural network
        template <typename T>
        Mat2d<T> fit(std::any input, const std::any target_labels, const size_t& batch_size, const size_t& epochs, const T& learning_rate)
        {
            std::any output; // Create null any class container
            T epoch_loss = 0; // Initialize batch loss variable with network data type
            T lr = learning_rate; // Testing
            Mat4d<T>* original_input = std::any_cast<Mat4d<T>>(&input); // Get and store original input matrix
            const Mat2d<T>* true_target_labels = std::any_cast<Mat2d<T>>(&target_labels); // Get and store original labels matrix
            const size_t num_of_batches = original_input->size() / batch_size; // Get total num of num_of_batches

            // Perform training steps for declared number of epochs
            for (size_t epoch = 0; epoch < epochs; epoch++) {
                output.reset(); // Reset output for each epoch
                output = Mat3d<T>(1, Mat2d<T>(true_target_labels->size(), std::vector<T>((*true_target_labels)[0].size())));
                // Create typed pointer to store all num_of_batches outputs for loss calculation
                Mat3d<T>* current_epoch_output = std::any_cast<Mat3d<T>>(&output);

                // Separate dataset into num_of_batches, for faster execution and better learning
                for (size_t batch = 0; batch < num_of_batches; batch++) {
                    std::any current_batch_input; // Empty container to store current batch input
                    std::any current_batch_labels_input; // Empty container to store curren batch labels for input
                    std::any current_batch_output; // Empty container to store current batch output
                    current_batch_input = Mat4d<T>(); // Create empty 4d matrix at empty container adress regarding input
                    current_batch_labels_input = Mat2d<T>(); // Create empty 2d matrix at empty container adress regarding labels input
                    const size_t start_row = batch * batch_size;
                    const size_t end_row = start_row + batch_size;

                    // Create typed pointer variable pointing to current batch input for operations
                    Mat4d<T>* typed_current_batch_input = std::any_cast<Mat4d<T>>(&current_batch_input);
                    // Create typed pointer varible pointing to current batch labels input
                    Mat2d<T>* typed_current_batch_input_labels = std::any_cast<Mat2d<T>>(&current_batch_labels_input); 
                    
                    // Fill current batch input with correct window of data from unbatched input
                    // and true target labels matrix
                    for (size_t i = start_row; i < end_row; i++) {
                        typed_current_batch_input->push_back((*original_input)[i]);
                        typed_current_batch_input_labels->push_back((*true_target_labels)[i]);
                    }

                    forward_pass(current_batch_input, current_batch_output); // Apply forward pass
                    backward_pass(current_batch_labels_input, lr); // Apply backward pass

                    // Create typed pointer varible pointing to current batch labels input
                    Mat2d<T>* typed_current_batch_output = std::any_cast<Mat2d<T>>(&current_batch_output);

                    for (size_t i = 0; i < typed_current_batch_output->size(); i++) {
                        for (size_t j = 0; j < (*typed_current_batch_output)[0].size(); j++) {
                            (*current_epoch_output)[0][i+start_row][j] = (*typed_current_batch_output)[i][j];
                        }
                    }

                    current_batch_input.reset(); // Reset current batch input memory space for each iteration
                    current_batch_labels_input.reset(); // Reset current batch labels input memory space for each iteration
                    current_batch_output.reset(); // Reset current batch output memory space for each iteration 
                } 

                output = (*current_epoch_output)[0];


                epoch_loss = calc_batch_loss<T>(output, target_labels); // Get batch loss for performance checking
                auto epoch_accuracy = get_accuracy<T>(output, target_labels);
                std::cout << "Epoch " << (epoch + 1) << " Loss: " << std::to_string(epoch_loss) << " ";
                std::cout << "Accuracy: " << std::to_string(epoch_accuracy) << " ";
            }  

            Mat2d<T>* result = std::any_cast<Mat2d<T>>(&output);

            return *result;
        }
        
        // Overload of fit method with multi-threading 
        /* 
        template <typename T>
        Mat2d<T> fit_multithreaded(std::any input, const std::any target_labels, const size_t& batch_size, const size_t& epochs, const T& learning_rate)
        {
            // Shared resourcers
            std::mutex mtx; // Used to protect updates to shared variables
            std::any output; // Empty memory space to store output
            Mat4d<T>* original_input = std::any_cast<Mat4d<T>>(&input); // Store received input matrix
            const Mat2d<T>* true_target_labels = std::any_cast<Mat2d<T>>(&target_labels); // Store received labels matrix
            const size_t num_of_batches = original_input->size() / batch_size; // Get number of batches (If batch size is 10 and input size 100 = 10 batches)
            const int num_of_threads = std::min(static_cast<unsigned int>(num_of_batches), std::thread::hardware_concurrency()); // Get current hardward available threads
            T epoch_loss = 0; // Initialize batch loss variable with network data type

            std::vector<std::thread> threads(num_of_threads); // Thread pool or vector of threads

            for (size_t epoch = 0; epoch < epochs; epoch++) {
                auto t1 = Benchmark::startBenchmark(); // Performance benchmarking
                // Reset output for each epoch
                output.reset();
                // Create 3d matrix at output memory address to store predicted labels for all examples in training data
                output = Mat3d<T>(1, Mat2d<T>(true_target_labels->size(), std::vector<T>((*true_target_labels)[0].size()))); 
                // Create epoch output pointer to user matrix methods
                Mat3d<T>* current_epoch_output = std::any_cast<Mat3d<T>>(&output);

                std::any testing;
                std::any testing2;

                auto process_batch = [&](size_t batch_start, size_t batch_end) {
                    for (size_t batch = batch_start; batch < batch_end; batch++) {
                        // Process each batch independtly
                        const std::any current_batch_input_address; // Store current batch input selection
                        const std::any current_batch_labels_input_address; // Store current batch assigned labels
                        const std::any current_batch_output_address; // Store current batch output
                        //current_batch_input = Mat4d<T>(); // Create empty 4d matrix at empty container adress regarding input
                        //current_batch_labels_input = Mat2d<T>(); // Create empty 2d matrix at empty container adress regarding labels input
                        const size_t start_row = batch * batch_size;
                        const size_t end_row = start_row + batch_size;
                        // Create typed pointer variable pointing to current batch input for operations
                        //current_batch_input_address = Mat4d<T>();
                        // Create typed pointer varible pointing to current batch labels input
                        //current_batch_labels_input_address = Mat2d<T>();

                        Mat4d<T> current_batch_input = std::any_cast<Mat4d<T>>(current_batch_input);
                        Mat2d<T> current_batch_labels_input = std::any_cast<Mat2d<T>>(current_batch_labels_input_address);
                        Mat2d<T> current_batch_output = std::any_cast<Mat2d<T>>(current_batch_output_address);

                        // Prepare batch data
                        for (size_t i = 0; i < end_row; i++) {
                            current_batch_input.push_back((*original_input)[i]);
                            current_batch_labels_input.push_back((*true_target_labels)[i]);
                        }   

                        DataAnalysis::display_shape(current_batch_input);
                        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
                        DataAnalysis::display_shape(current_batch_labels_input);
                        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

                        //forward_pass(current_batch_input, current_batch_output); // Apply forward pass
                        //DataAnalysis::display_shape(current_batch_labels_input);
                        //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
                        //backward_pass(current_batch_labels_input, learning_rate); // Apply backward pass

                        // Create typed pointer varible pointing to current batch labels input
                        //Mat2d<T>* typed_current_batch_output = std::any_cast<Mat2d<T>>(&current_batch_output);

                        // Aggregate results safely
                        std::lock_guard<std::mutex> lock(mtx);
                        for (size_t i = 0; i < current_epoch_output->size(); i++) {
                            for (size_t j = 0;j < (*current_epoch_output)[i].size(); j++) {
                                (*current_epoch_output)[0][i+start_row][j] = current_batch_output[i][j];
                            }
                        }
                        
                    }
                };
            
                // Determine workload for each thread
                size_t batches_per_thread = num_of_batches / num_of_threads;
                for (size_t t = 0; t < num_of_threads; t++) {
                    size_t batch_start = t * batches_per_thread;
                    size_t batch_end = (t == num_of_threads - 1) ? num_of_batches : batch_start + batches_per_thread;
                    threads[t] = std::thread(process_batch, batch_start, batch_end);
                }       

                // Join threads
                for (auto &t : threads) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
                
                output = (*current_epoch_output)[0];

                auto t2 = Benchmark::stopBenchmark(); // Performance benchmarkingx
                epoch_loss = calc_batch_loss<T>(output, target_labels); // Get batch loss for performance checking
                std::cout << "Epoch " << (epoch + 1) << " Loss: " << std::to_string(epoch_loss) << " ";
                std::cout << Benchmark::getDuration(t1, t2, Benchmark::Seconds) << std::endl;

            }

            Mat2d<T>* result = std::any_cast<Mat2d<T>>(&output);

            return *result;
        } */

        // Method that calculates accuracy of training process
        template <typename T>
        T get_accuracy(const std::any predicted_labels, const std::any target_labels)
        {
            if (!target_labels.has_value()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return 0;
            }

            const Mat2d<T>* predictions = std::any_cast<Mat2d<T>>(&predicted_labels); // Get training result labels predictions
            const Mat2d<T>* true_target_labels = std::any_cast<Mat2d<T>>(&target_labels); // Get correct label for all examples in dataset
            const T num_of_examples = true_target_labels->size(); // Get number of examples in dataset
            T incorrect_predictions = 0; // Variable to store number of incorrect predictions 

            // Loop over all predictions and check against correct class for current example
            for (size_t i = 0; i < num_of_examples; i++) {
                // Extract detected class from result and compare to original class
                auto og_class = NumPP::get_max_element_pos((*true_target_labels)[i]);
                auto pred_class = NumPP::get_max_element_pos((*predictions)[i]);
                if (og_class != pred_class) {
                    incorrect_predictions++;
                }
            }

            T correct_predictions = num_of_examples - incorrect_predictions; // Get number of correct predictions
            
            return correct_predictions / num_of_examples; // return accuracy of model in decimal format
        }

        // Method that loads exported model back into original data structures
        void load_model(const std::string& filePath)
        {
            if (filePath.empty()) {
                throw std::runtime_error("Error: No file path detected");
            }

            std::ifstream in(filePath, std::ios::binary); // Open binary file at given path

            if (!in.is_open()) {
                throw std::runtime_error("Error: Failed to open model binary file");
            }

            m_layers = deserialize_layers(in);
            m_weights = deserialize_weights(in);
            m_biases = deserialize_biases(in);
            
            // Debugging Code
            //Mat3d<double>* tensor_1 = std::any_cast<Mat3d<double>>(&m_weights[0]);
            //Mat2d<double>* tensor_2 = std::any_cast<Mat2d<double>>(&m_weights[1]);
            //Mat2d<double>* tensor_3 = std::any_cast<Mat2d<double>>(&m_weights[2]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_shape(*tensor_1);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_shape(*tensor_2);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_shape(*tensor_3);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all((*tensor_1)[0]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all((*tensor_1)[1]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all((*tensor_1)[2]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all((*tensor_1)[3]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all((*tensor_1)[4]);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //std::vector<double>* bias_0 = std::any_cast<std::vector<double>>(&m_biases[0]);
            //std::vector<double>* bias_1 = std::any_cast<std::vector<double>>(&m_biases[1]);
            //std::vector<double>* bias_2 = std::any_cast<std::vector<double>>(&m_biases[2]);
            //std::cout << bias_0->size() << std::endl;
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //std::cout << bias_1->size() << std::endl;
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //std::cout << bias_2->size() << std::endl;
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all(*bias_0);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all(*bias_1);
            //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            //DataAnalysis::display_all(*bias_2);

            in.close();

            std::cout << "Model Parameters Loaded!" << std::endl;
        }
        
        // Method that loads single input to be analysed by network
        template <typename T>
        size_t predict(std::any input) 
        {
            if (!input.has_value()) {
                throw std::runtime_error("Error: Input parameter is empty");
            }

            std::any output; // Create null any class container

            m_current_input = input; // Assign current input to initial input data
            m_current_output = output; // Assign output reference ot current output
            int w_and_b_counter = 0; // Determine which element of weights and biases vectors
            int it_counter = 0; // Counts layer iterations to assign correct layer output to layer results vector

            for (auto &layer : m_layers) {
                m_current_output.reset(); // Clear current output for next layer 

                // Call current layer forward method
                layer->predict(m_current_input, m_current_output, m_weights[w_and_b_counter], m_biases[w_and_b_counter]); 
                m_current_input.reset(); // Clear current input for next layer
                m_current_input = m_current_output; // Update input for next layer 

                // Check if layer has weights and biases flag, if so add to counter
                if (layer->get_w_and_b()) {
                    w_and_b_counter++;
                }  

                /*  Debugging Code        
                    if (layer->get_layer_type() == LayerType::CONV2D) {
                    Mat3d<T>* typedInput = std::any_cast<Mat3d<T>>(&m_current_input);
                    DataAnalysis::display_shape(*typedInput);
                }
                if (layer->get_layer_type() == LayerType::AVERAGEPOOLING2D) {
                    Mat3d<T>* typedInput = std::any_cast<Mat3d<T>>(&m_current_input);
                    DataAnalysis::display_shape(*typedInput);
                }
                if (layer->get_layer_type() == LayerType::FLATTEN) {
                    std::vector<T>* typedInput = std::any_cast<std::vector<T>>(&m_current_input);
                    std::cout << typedInput->size() << std::endl;
                }
                if (layer->get_layer_type() == LayerType::DENSE) {
                    std::vector<T>* typedInput = std::any_cast<std::vector<T>>(&m_current_input);
                    std::cout << typedInput->size() << std::endl;
                } */

            }
            
            output = m_current_output;
            std::vector<T>* result = std::any_cast<std::vector<T>>(&output);

            //DataAnalysis::display_all(*result);

            size_t prediction = NumPP::get_max_element_pos(*result);
            return prediction;
        }

        // Real time detection using open cv
        // Method used to read frame data from video feed 
        // And use said frame as input for network real time image classification
        template <typename T>
        size_t predict_frame(const std::any& input)
        {
            if (!input.has_value()) {
                throw std::runtime_error("Error: Frame data is empty");
            }
            
            std::any output; // Create null any class container

            m_current_input = input; // Assign current input to initial input data
            m_current_output = output; // Assign output reference ot current output
            int w_and_b_counter = 0; // Determine which element of weights and biases vectors
            int it_counter = 0; // Counts layer iterations to assign correct layer output to layer results vector

            for (auto &layer : m_layers) {
                m_current_output.reset(); // Clear current output for next layer 

                // Call current layer forward method
                layer->predict(m_current_input, m_current_output, m_weights[w_and_b_counter], m_biases[w_and_b_counter]); 
                m_current_input.reset(); // Clear current input for next layer
                m_current_input = m_current_output; // Update input for next layer 

                // Check if layer has weights and biases flag, if so add to counter
                if (layer->get_w_and_b()) {
                    w_and_b_counter++;
                }  

                /*  Debugging Code        
                    if (layer->get_layer_type() == LayerType::CONV2D) {
                    Mat3d<T>* typedInput = std::any_cast<Mat3d<T>>(&m_current_input);
                    DataAnalysis::display_shape(*typedInput);
                }
                if (layer->get_layer_type() == LayerType::AVERAGEPOOLING2D) {
                    Mat3d<T>* typedInput = std::any_cast<Mat3d<T>>(&m_current_input);
                    DataAnalysis::display_shape(*typedInput);
                }
                if (layer->get_layer_type() == LayerType::FLATTEN) {
                    std::vector<T>* typedInput = std::any_cast<std::vector<T>>(&m_current_input);
                    std::cout << typedInput->size() << std::endl;
                }
                if (layer->get_layer_type() == LayerType::DENSE) {
                    std::vector<T>* typedInput = std::any_cast<std::vector<T>>(&m_current_input);
                    std::cout << typedInput->size() << std::endl;
                } */

            }
            
            output = m_current_output;
            std::vector<T>* result = std::any_cast<std::vector<T>>(&output);

            //DataAnalysis::display_all(*result);
            
            size_t prediction = NumPP::get_max_element_pos(*result);            
            return prediction;
        }

        // Overload of predict frame method that returns the whole output vector
        // Not the predicted class, i.e, output vector max element position
        template <typename T>
        std::vector<T> predict_output_vector(const std::any& input)
        {
            if (!input.has_value()) {
                throw std::runtime_error("Error: Frame data is empty");
            }
            
            std::any output; // Create null any class container

            m_current_input = input; // Assign current input to initial input data
            m_current_output = output; // Assign output reference ot current output
            int w_and_b_counter = 0; // Determine which element of weights and biases vectors
            int it_counter = 0; // Counts layer iterations to assign correct layer output to layer results vector

            for (auto &layer : m_layers) {
                m_current_output.reset(); // Clear current output for next layer 

                // Call current layer forward method
                layer->predict(m_current_input, m_current_output, m_weights[w_and_b_counter], m_biases[w_and_b_counter]); 
                m_current_input.reset(); // Clear current input for next layer
                m_current_input = m_current_output; // Update input for next layer 

                // Check if layer has weights and biases flag, if so add to counter
                if (layer->get_w_and_b()) {
                    w_and_b_counter++;
                }  

                /*  Debugging Code        
                    if (layer->get_layer_type() == LayerType::CONV2D) {
                    Mat3d<T>* typedInput = std::any_cast<Mat3d<T>>(&m_current_input);
                    DataAnalysis::display_shape(*typedInput);
                }
                if (layer->get_layer_type() == LayerType::AVERAGEPOOLING2D) {
                    Mat3d<T>* typedInput = std::any_cast<Mat3d<T>>(&m_current_input);
                    DataAnalysis::display_shape(*typedInput);
                }
                if (layer->get_layer_type() == LayerType::FLATTEN) {
                    std::vector<T>* typedInput = std::any_cast<std::vector<T>>(&m_current_input);
                    std::cout << typedInput->size() << std::endl;
                }
                if (layer->get_layer_type() == LayerType::DENSE) {
                    std::vector<T>* typedInput = std::any_cast<std::vector<T>>(&m_current_input);
                    std::cout << typedInput->size() << std::endl;
                } */

            }
            
            output = m_current_output;
            std::vector<T>* result = std::any_cast<std::vector<T>>(&output);

            //DataAnalysis::display_all(*result);
         
            return *result;
        }

        ~NeuralNetwork()
        {
            for (auto &layer : m_layers) {
                delete layer;
            }
        }
    };

#pragma endregion

#pragma region OpenCvIntegration
    // Class that transforms open cv data structures into mine for later analysis
    // This is temporary, I do not plan to use open cv in public release, this is for learning and testing purposes
    class OpenCvIntegration
    {
    private:
        // Get sum of a open cv Vec3b data type, returns the sum in declared data type
        template <typename T>
        static T get_sum_of_vector(const cv::Vec3b& vec)
        {
            T result; // Store result in desired data type

            for (size_t i = 0; i < vec.channels; i++) {
                result += static_cast<T>(vec[i]);
            }

            return result;

        }
    
        // Method that pre-process image based on parameters
        static cv::Mat pre_process_image(const cv::Mat* src_image_ptr, const bool& crop, const bool& resize, const std::pair<size_t, size_t>& shape, const bool& gray_scale, 
                                            const bool& apply_blurring, const bool& apply_edge_detection)
        {
            if (src_image_ptr->empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return cv::Mat();
            }
            
            cv::Mat p_image = *src_image_ptr; // Create variable to store processed image data

            if (crop) {
                // Define the ROI coordinates
                int x_start = 0;  // x-coordinate of the top-left corner
                int y_start = 280;   // y-coordinate of the top-left corner
                int x_end = 1920;    // x-coordinate of the bottom-right corner
                int y_end = 1080;    // y-coordinate of the bottom-right corner

                // Define the rectangle for the region of interest
                cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);

                // Crop the image
                cv::Mat croppedImage = p_image(roi);
            }

            if (gray_scale) {
                cv::cvtColor(p_image, p_image, cv::COLOR_BGR2GRAY); // Convert image to gray scale
            }

            if (apply_blurring) {
                cv::GaussianBlur(p_image, p_image, cv::Size(5, 5), 0, 0);
            }

            if (apply_edge_detection) {
                cv::Canny(p_image, p_image, 50, 250);
            }

            if (resize) {
                cv::resize(p_image, p_image, cv::Size(shape.first, shape.second)); // Re-scale image based on shape parameter
            }

            return p_image;
        }
    public:
        // Struct to hold parameters for prepare_training_data method
        struct TrainingDataParameters
        {
            const std::string dir_path;
            const bool crop = false;
            const bool resize = false;
            const std::pair<size_t, size_t> shape = {240, 240};
            const bool gray_scale = false;
            const bool apply_blurring = false;
            const bool apply_edge_detection = false;

            TrainingDataParameters(const std::string& dir_path, const bool& crop, const bool& resize, const std::pair<size_t, size_t>& shape, 
            const bool& gray_scale, const bool& apply_blurring, const bool& apply_edge_detection):
                dir_path(dir_path),
                crop(crop),
                resize(resize),
                shape(shape),
                gray_scale(gray_scale),
                apply_blurring(apply_blurring), 
                apply_edge_detection(apply_edge_detection)  
            {}
        };

        // Method that applies Warp perspective to image and return image in 
        // Bird's Eye View (bev) perspective
        static cv::Mat change_perspective_to_birds_eye_view(const cv::Mat& original_image)
        {
            if (original_image.empty()) {
                std::cerr << "Error: Input is null" << std::endl;
                return cv::Mat();
            }

            // Get image dimensions
            int img_width = original_image.cols;
            int img_height = original_image.rows;

            // Calculate source points in original image
            std::vector<cv::Point2f> src_points;
            src_points.push_back(cv::Point2f(250, 250));  // top-left corner
            src_points.push_back(cv::Point2f(750, 750));  // top-right corner
            src_points.push_back(cv::Point2f(750, 250));  // bottom-right corner
            src_points.push_back(cv::Point2f(750, 250));  // bottom-left corner

            // Define destination points for bird's eye view perspective
            std::vector<cv::Point2f> dst_points;
            dst_points.push_back(cv::Point2f(0, 0));  // top-left corner
            dst_points.push_back(cv::Point2f(500, 0));  // top-right corner
            dst_points.push_back(cv::Point2f(500, 500));  // bottom-right corner
            dst_points.push_back(cv::Point2f(0, 500));  // bottom-left corner

            // Compute perpective transform matrix
            cv::Mat perspective_mat = cv::getPerspectiveTransform(src_points, dst_points);

            // Apply perspective transform to return matrix
            cv::Mat birds_eye_view_image;
            cv::warpPerspective(original_image, birds_eye_view_image, perspective_mat, cv::Size(500, 500));

            return birds_eye_view_image;
        }

        // Get open cv color image matrix and convert it to our 3d matrix data structure
        // Default format is Blue, Green, Red
        template <typename T>
        static Mat3d<T>* convert_color_image(const cv::Mat* image) 
        {
            if (!image->data) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Image data empty" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat3d pointer and allocate memory for it
            Mat3d<T>* nMatPtr = new Mat3d<T>(image->rows, Mat2d<T>(image->cols, std::vector<T>(3)));
            // Create pointer to hold open cv pixel values
            cv::Vec3b* cvPixelPtr = new cv::Vec3b(3);
            // Iterate over each pixel in the given image
            for (int i = 0; i < image->rows; i++) {
                for (int j = 0; j < image->cols; j++) {
                    // Assing current pixel data to pointer
                    (*cvPixelPtr) = image->at<cv::Vec3b>(i, j);
                    // Access the corresponding pixel in the new matrix and copy pixel values
                    (*nMatPtr)[i][j] = {static_cast<T>((*cvPixelPtr)[0]),
                                        static_cast<T>((*cvPixelPtr)[1]), 
                                        static_cast<T>((*cvPixelPtr)[2])};
                }
            }

            delete cvPixelPtr;
            // Return the pointer to the final matrix
            return nMatPtr;
        }

        // Overload of convert_image method for gray images
        template <typename T>
        static Mat3d<T>* convert_gray_image(const cv::Mat* image) 
        {
            if (!image->data) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Image data empty" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat3d pointer and allocate memory for it
            Mat3d<T>* nMatPtr = new Mat3d<T>(image->rows, Mat2d<T>(image->cols, std::vector<T>(1)));
            // Create pointer to hold open cv pixel values
            uchar* cvPixelPtr = new uchar(1);
            // Iterate over each pixel in the given image
            for (int i = 0; i < image->rows; i++) {
                for (int j = 0; j < image->cols; j++) {
                    // Assing current pixel data to pointer
                    (*cvPixelPtr) = image->at<uchar>(i, j);
                    // Access the corresponding pixel in the new matrix and copy pixel values
                    (*nMatPtr)[i][j][0] = static_cast<T>((*cvPixelPtr));
                }
            }

            delete cvPixelPtr;
            // Return the pointer to the final matrix
            return nMatPtr;
        }

        // Method that transform individual frames in video file and save them as images
        static void convert_video_to_images(std::string& video_path, std::string& save_path)
        {
            if (video_path.empty()) {
                std::cerr << "Error: Input parameter is empty" << std::endl;
                return;
            }

            cv::VideoCapture video(video_path); // Get video file and store it in video file

            if(!video.isOpened()) {
                std::cerr << "Error: Video file failed to open" << std::endl;
                return;
            }


            cv::Mat* framePtr = new cv::Mat(); // Create pointer to store current frame 
            size_t frame_counter = 0; // Create variable to count number of frames
            save_path = save_path.append("/Frame-"); // Add Img- to differentiate frames 

            // Loop while video has frames remaining 
            while (video.isOpened())
            {
                std::string image_path = save_path; // Create local variable to complete image path every frame

                video >> *framePtr; // Store current frame in created pointer 

                if (framePtr->empty()) {
                    std::cerr << "Error: Frame data is empty" << std::endl;
                    break;
                }
                
                image_path.append(std::to_string(frame_counter)); // Add current frame number to image name
                image_path.append(".jpg"); // Add which compression type to use to store image

                cv::imwrite(image_path, *framePtr); // Save image to correct path using data in frame pointer

                std::cout << image_path << std::endl;
                frame_counter++; // Increase frame counter
            }

            delete framePtr; // Delete pointer created

            video.release(); // Close video file or camera opened
        }

        // Convert open cv matrix, to 2d matrix, where each element will be the sum
        // Of Blue, Green and Red values
        static Mat2d<int16_t>* get_sum_pixels(const cv::Mat* image)
        {
            if (!image->data) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Image data empty" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat2d pointer and allocate memory for it
            Mat2d<int16_t> *mat2dPtr = new Mat2d<int16_t>(image->rows, std::vector<int16_t>(image->cols));

            for (size_t i = 0; i < mat2dPtr->size(); i++) {
                for (size_t j = 0; j < (*mat2dPtr)[i].size(); j++) {
                    auto v = image->at<cv::Vec3b>(i, j);
                    int16_t e = get_sum_of_vector<int16_t>(v);
                    (*mat2dPtr)[i][j] = e;
                }
            }

            return mat2dPtr;
        }
        
        // Method to convert 3d matrix into open cv color matrix
        template <typename T>
        static cv::Mat get_open_cv_color_mat(const Mat3d<T>* matPtr)
        {
            if (matPtr == nullptr) {
                std::cerr << "Error: Input is empty" << std::endl;
                return cv::Mat();
            }

            // Get dimensions of the input matrix
            int rows = static_cast<int>(matPtr->size());
            int cols = static_cast<int>((*matPtr)[0].size());
            int depth = static_cast<int>((*matPtr)[0][0].size());

            // Determine OpenCV data type based on the template parameter T
            int cvDataType;
            if (std::is_same<T, uchar>::value) {
                cvDataType = CV_8UC(depth);
                cv::Mat opencv_mat(rows, cols, cvDataType);

                for (int d = 0; d < depth; d++) {
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            if (depth == 1) {
                                opencv_mat.at<cv::Vec<T, 1>>(i, j)[d] = static_cast<T>((*matPtr)[i][j][d]);
                            } else if (depth == 2) {
                                opencv_mat.at<cv::Vec<T, 2>>(i, j)[d] = static_cast<T>((*matPtr)[i][j][d]);
                            } else {
                                opencv_mat.at<cv::Vec<T, 3>>(i, j)[d] = static_cast<T>((*matPtr)[i][j][d]);
                            }
                        }
                    }
                }

                return opencv_mat;
            } 
            else {
                cvDataType = CV_32FC(depth);

                cv::Mat opencv_mat(rows, cols, cvDataType);

                for (int d = 0; d < depth; d++) {
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            if (depth == 1) {
                                opencv_mat.at<cv::Vec<float, 1>>(i, j)[d] = static_cast<float>((*matPtr)[i][j][d]);
                            } else if (depth == 2) {
                                opencv_mat.at<cv::Vec<float, 2>>(i, j)[d] = static_cast<float>((*matPtr)[i][j][d]);
                            } else {
                                opencv_mat.at<cv::Vec<float, 3>>(i, j)[d] = static_cast<float>((*matPtr)[i][j][d]);
                            }
                        }
                    }
                }

                return opencv_mat;
            } 
        }

        // Method to convert 3d matrix into open cv matrix
        // Added conv_channel parameter to test if all convolutional feature maps are being
        // generated correctly
        template <typename T>
        static cv::Mat get_open_cv_gray_mat(const Mat3d<T>& matPtr, int conv_channel = 0)
        {
            if (matPtr.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return cv::Mat();
            }

            // Get dimensions of the input matrix
            int rows = matPtr.size();
            int cols = matPtr[0].size();

            cv::Mat opencv_mat(rows, cols, CV_8UC1);

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    opencv_mat.at<uchar>(i, j) = static_cast<uchar>(matPtr[i][j][conv_channel]);
                }
            }

            return opencv_mat;
        }
        
        // Method that iterates through all images in given directory and return Mat4d with all images converted to Mat3d
        template <typename T>
        static Mat4d<T> prepare_training_data(const TrainingDataParameters& params)
        {
            if (!std::filesystem::exists(params.dir_path) && !std::filesystem::is_directory(params.dir_path)) {
                std::cerr << "Error: Directory not valid" << std::endl;
                return Mat4d<T>();
            }

            cv::Mat* imagePtr = new cv::Mat(); // Create pointer to store open cv matrices read from each image    
            Mat4d<T> training_data_mat; // Create 4d matrix that will hold all images matrices
            cv::Mat p_image; // Create pointer to store image after pre-processing  
            bool multiple_directories = false; // Create bool variable to control which block of code to execute

            // Check root directory for more directories
            for (const auto &entry : std::filesystem::directory_iterator(params.dir_path)) {
                if (std::filesystem::is_directory(entry.path())) {
                    multiple_directories = true;
                    break;
                }
            }

            // If root directory contains more directories inside
            if (multiple_directories) {
                std::vector<std::filesystem::directory_entry> entries; // Create vector to store all directoies in root directory

                // Store all entries
                for (const auto &entry : std::filesystem::directory_iterator(params.dir_path)) {
                    entries.push_back(entry);
                }

                // Sort entries by path
                std::sort(entries.begin(), entries.end(), [](const std::filesystem::directory_entry& a,
                    const std::filesystem::directory_entry& b) {
                        return a.path() < b.path();
                });

                // Iterate over the contents of directories
                for (const auto &entry : entries) {
                    std::cout << entry.path() << std::endl;
                    for (const auto &sub_entry : std::filesystem::directory_iterator(entry.path())) {
                        //std::cout << sub_entry.path() << std::endl;

                        if (sub_entry.path().extension() == ".jpg" || sub_entry.path().extension() == ".png") {
                            *imagePtr = cv::imread(sub_entry.path()); // Read current entry image in directory into pointer
                            p_image = pre_process_image(imagePtr, params.crop, params.resize, params.shape, params.gray_scale, params.apply_blurring, params.apply_edge_detection);
                            Mat3d<T>* converted_image_mat = convert_gray_image<T>(&p_image); // Convert opencv matrix to Mat3d
                            training_data_mat.push_back(*converted_image_mat); // Add Mat3d into training data

                            delete converted_image_mat;
                        }  
                    }
                }

                delete imagePtr;

                return training_data_mat;
            }

            // Block of code to execute if there are no more directories inside root directory
            for (const auto &entry : std::filesystem::directory_iterator(params.dir_path)) {
                std::cout << entry.path() << std::endl;

                if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                    *imagePtr = cv::imread(entry.path()); // Read current entry image in directory into pointer
                    p_image = pre_process_image(imagePtr, params.crop, params.resize, params.shape, params.gray_scale, params.apply_blurring, params.apply_edge_detection);
                    Mat3d<T>* converted_image_mat = convert_gray_image<T>(&p_image); // Convert opencv matrix to Mat3d
                    training_data_mat.push_back(*converted_image_mat); // Add Mat3d into training data
                    delete converted_image_mat;
                }  
            }

            delete imagePtr;

            return training_data_mat;       
        }       

    };

#pragma endregion

// Further classes to be implemented

}