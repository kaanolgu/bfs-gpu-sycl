#include <map>
#define MAX_NUM_CU 4
// using MyUint1 = ac_int<1, false>;
#include "unrolled_loop.hpp"
int numRows = 0,numCols =0,numNonz =0;

constexpr int log2(int num) {
    int result = 0;
    int running = num;

    while (running > 1) {
        result++;
        running /= 2;
    }

    int comp = 1;

    for (int i = 0; i < result; i++) {
        comp *= 2;
    }

    if (num != comp) {
        result++;
    }
    
    return result;
}

constexpr int BUFFER_SIZE = 16;
using MyUint1 = char; 
using d_type3 = char;
using Uint32 = unsigned int;


class Timer {
public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};



class CommandLineParser {
public:
    // Method to add an argument with a default value
    void addArgument(const std::string& name, const std::string& defaultValue) {
        arguments[name] = defaultValue;
    }

    // Method to parse the command line arguments
    void parseArguments(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.find("--") == 0) {
                // Handle "--name=value" format
                auto pos = arg.find('=');
                if (pos != std::string::npos) {
                    std::string name = arg.substr(2, pos - 2); // Extract key
                    std::string value = arg.substr(pos + 1);    // Extract value
                    std::cout << "key L " << name << ", value = " << value << std::endl;
                    if (arguments.find(name) != arguments.end()) {
                        arguments[name] = value;
                    }
                } 
                // Handle "--name value" format
                else if (i + 1 < argc && argv[i + 1][0] != '-') {
                    std::string name = arg.substr(2); // Extract key
                    std::string value = argv[++i]; // Get the next argument as value
                    std::cout << "key S " << name << ", value = " << value << std::endl;
                    if (arguments.find(name) != arguments.end()) {
                        arguments[name] = value;
                    }
                } else {
                    std::cerr << "Warning: No value provided for argument " << arg << std::endl;
                }
            }
        }
    }

    // Method to get the value of an argument
    std::string getArgument(const std::string& name) const {
        auto it = arguments.find(name);
        if (it != arguments.end()) {
            return it->second;
        }
        return ""; // Return an empty string if the argument is not found
    }

    // Method to print all arguments in a formatted table
    void printArguments() const {
        std::cout << std::setw(20) << std::left << "Argument"
                  << std::setw(20) << "Value" << std::endl;
        std::cout << std::string(40, '-') << std::endl;

        for (const auto& pair : arguments) {
            std::cout << std::setw(20) << std::left << "--" + pair.first
                      << std::setw(20) << pair.second << std::endl;
        }
    }

private:
    std::map<std::string, std::string> arguments;
};


// Function to calculate the mean of the data
double calculateMean(const std::vector<double>& data) {
    double sum = 0.0;
    for(double num : data) {
        sum += num;
    }
    return sum / data.size();
}

// Function to calculate the standard deviation of the data
double calculateStandardDeviation(const std::vector<double>& data, double mean) {
    double sum = 0.0;
    for(double num : data) {
        sum += (num - mean) * (num - mean);
    }
    return std::sqrt(sum / data.size());
}

// Function to remove anomalies from the data
std::vector<double> removeAnomalies(const std::vector<double>& data, double threshold) {
    std::vector<double> filteredData;
    double mean = calculateMean(data);
    double stdDev = calculateStandardDeviation(data, mean);

    for(double num : data) {
        if(std::abs(num - mean) <= threshold * stdDev) {
            filteredData.push_back(num);
        }
    }
    return filteredData;
}