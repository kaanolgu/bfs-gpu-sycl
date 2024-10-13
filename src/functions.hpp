#include <map>
#include "unrolled_loop.hpp"
#include "json.hpp"



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
                    if (arguments.find(name) != arguments.end()) {
                        arguments[name] = value;
                    }
                } 
                // Handle "--name value" format
                else if (i + 1 < argc && argv[i + 1][0] != '-') {
                    std::string name = arg.substr(2); // Extract key
                    std::string value = argv[++i]; // Get the next argument as value
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
        for (const auto& pair : arguments) {
            std::cout << std::setw(20) << std::left << "- " + pair.first
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

std::vector<double> removeAnomaliesnofilter(const std::vector<double>& data) {
    std::vector<double> filteredData;

    // Calculate the number of elements to skip (10% of the data)
    size_t numToSkip = static_cast<size_t>(data.size() * 0.1);
    
    // Start from the 10% index to the end of the data
    for (size_t i = numToSkip; i < data.size(); ++i) {
        filteredData.push_back(data[i]);
    }

    return filteredData;
}
// Print out data 

// Constants for formatting
const char separator = ' ';
const int nameWidth = 24;
const int numWidth = 24;



// Function to print a formatted row with a label and value
void printRow(const std::string& label, const std::string& value) {
    std::cout << "- " << std::left << std::setw(nameWidth) << std::setfill(separator) << label
              << " " << std::setw(numWidth) << std::setfill(separator) << value << "\n";
}

// Function to format double values with fixed precision
std::string formatDouble(double value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

bool areAllRowsEqualToLast(const std::vector<std::vector<int>>& h_distancesGPU) {
    // Check if the vector is empty or has only one row
    if (h_distancesGPU.size() < 2) {
        return true; // Consider it as matching
    }

    const auto& lastRow = h_distancesGPU.back(); // Get the last row

    // Check if all rows match the last row using std::equal
    for (size_t i = 0; i < h_distancesGPU.size() - 1; ++i) {
        // Use std::equal to compare the current row with the last row
        if (!std::equal(h_distancesGPU[i].begin(), h_distancesGPU[i].end(), lastRow.begin())) {
            std::cout << "distanceGPU[" << i << "] does not match the last row ";
            for (int i = 0; i < h_distancesGPU[i].size(); i++) {
                int countB = std::count(h_distancesGPU[i].begin(), h_distancesGPU[i].end(), i);
                std::cout << std::to_string(countB) << ", "; 
            }
            return false; // Return false if any row doesn't match
        }
    }

    return true; // All rows matched the last row
}

struct DeviceInfo {
    int gpuID;  // Changed from deviceID to gpuID
    long long edgesProcessed;  // Changed from totalEdgesProcessed to edgesProcessed
    double percentage;
};

void printDeviceInfo(const std::vector<DeviceInfo>& devices) {
    std::cout << std::left << std::setw(10) << "GPU ID"
              << std::setw(30) << "Edges Processed (*)"
              << std::setw(15) << "Percentage" << std::endl;

    std::cout << std::string(55, '-') << std::endl; // Separator line

    for (const auto& device : devices) {
        std::cout << std::left << std::setw(10) << device.gpuID
                  << std::setw(30) << device.edgesProcessed
                  << std::setw(15) << std::fixed << std::setprecision(2) << device.percentage << " %" << std::endl;
    }

    // Add the note at the bottom
    std::cout << std::string(55, '-') << std::endl; // Another separator line
    std::cout << "* : including already visited because we keep asking \"are you visited?\"]" << std::endl;
}