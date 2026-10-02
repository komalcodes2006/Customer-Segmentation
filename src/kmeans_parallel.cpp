#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cctype>
#include <omp.h>

using namespace std;

// -----------------------------
// K-Means Configuration
// -----------------------------
const int FEATURES = 3;
const int K = 4;
const int MAX_ITERATIONS = 100;
const double TOLERANCE = 1e-4;

using Point = array<double, FEATURES>;

// -----------------------------
// Load CSV Data
// -----------------------------
string trim(const string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == string::npos) {
        return "";
    }

    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

vector<string> splitCsvLine(const string& line) {
    vector<string> fields;
    string value;
    stringstream ss(line);

    while (getline(ss, value, ',')) {
        fields.push_back(trim(value));
    }

    return fields;
}

int findColumn(const vector<string>& headers, const string& name) {
    for (size_t i = 0; i < headers.size(); i++) {
        if (trim(headers[i]) == name) {
            return static_cast<int>(i);
        }
    }

    return -1;
}

vector<Point> loadData(const string& filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    vector<Point> data;
    string line;

    if (!getline(file, line)) {
        throw runtime_error("CSV is empty: " + filename);
    }

    const vector<string> headers = splitCsvLine(line);
    const int ordersColumn = findColumn(
        headers, "lifetime_orders"
    );
    const int spendColumn = findColumn(
        headers, "lifetime_spend"
    );
    const int aovColumn = findColumn(
        headers, "average_order_value"
    );

    if (ordersColumn < 0 || spendColumn < 0 || aovColumn < 0) {
        throw runtime_error(
            "CSV must contain lifetime_orders, lifetime_spend, and "
            "average_order_value columns"
        );
    }

    while (getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        const vector<string> fields = splitCsvLine(line);
        Point point{};

        if (ordersColumn >= static_cast<int>(fields.size()) ||
            spendColumn >= static_cast<int>(fields.size()) ||
            aovColumn >= static_cast<int>(fields.size())) {
            throw runtime_error("Invalid CSV row: " + line);
        }

        point[0] = stod(fields[ordersColumn]);
        point[1] = stod(fields[spendColumn]);
        point[2] = stod(fields[aovColumn]);

        for (int j = 0; j < FEATURES; j++) {
            if (!isfinite(point[j])) {
                throw runtime_error("Non-finite value found in dataset");
            }
        }

        data.push_back(point);
    }

    if (data.size() < K) {
        throw runtime_error("Dataset must contain at least K points");
    }

    return data;
}

// -----------------------------
// Squared Euclidean Distance
// -----------------------------
double squaredDistance(
    const Point& a,
    const Point& b
) {
    double distance = 0.0;

    for (int j = 0; j < FEATURES; j++) {
        double diff = a[j] - b[j];
        distance += diff * diff;
    }

    return distance;
}

// -----------------------------
// Initialize Centroids
// Same initialization as sequential
// -----------------------------
vector<Point> initializeCentroids(
    const vector<Point>& data
) {
    mt19937 generator(42);

    vector<int> indices(data.size());

    for (int i = 0; i < static_cast<int>(data.size()); i++) {
        indices[i] = i;
    }

    // Partial Fisher-Yates shuffle
    for (int i = 0; i < K; i++) {
        uniform_int_distribution<int> distribution(
            i,
            static_cast<int>(data.size()) - 1
        );

        int j = distribution(generator);
        swap(indices[i], indices[j]);
    }

    vector<Point> centroids;

    for (int i = 0; i < K; i++) {
        centroids.push_back(data[indices[i]]);
    }

    return centroids;
}

// -----------------------------
// Parallel Cluster Assignment
// -----------------------------
void assignClustersParallel(
    const vector<Point>& data,
    const vector<Point>& centroids,
    vector<int>& labels
) {
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); i++) {

        double bestDistance = numeric_limits<double>::max();
        int bestCluster = 0;

        for (int c = 0; c < K; c++) {

            double distance =
                squaredDistance(data[i], centroids[c]);

            if (distance < bestDistance) {
                bestDistance = distance;
                bestCluster = c;
            }
        }

        labels[i] = bestCluster;
    }
}

// -----------------------------
// Parallel Centroid Update
// Using thread-local accumulation
// -----------------------------
vector<Point> updateCentroidsParallel(
    const vector<Point>& data,
    const vector<int>& labels,
    const vector<Point>& oldCentroids
) {
    int numThreads = omp_get_max_threads();

    // threadSums[thread][cluster][feature]
    vector<vector<Point>> threadSums(
        numThreads,
        vector<Point>(K)
    );

    // threadCounts[thread][cluster]
    vector<array<int, K>> threadCounts(numThreads);

    // Initialize thread-local arrays
    for (int t = 0; t < numThreads; t++) {

        for (int c = 0; c < K; c++) {
            threadCounts[t][c] = 0;

            for (int j = 0; j < FEATURES; j++) {
                threadSums[t][c][j] = 0.0;
            }
        }
    }

    // Each thread works on its own sums and counts
    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < static_cast<int>(data.size()); i++) {

            int cluster = labels[i];

            threadCounts[threadId][cluster]++;

            for (int j = 0; j < FEATURES; j++) {
                threadSums[threadId][cluster][j] += data[i][j];
            }
        }
    }

    // Reduce thread-local results
    vector<Point> newCentroids(K);
    vector<int> totalCounts(K, 0);

    for (int c = 0; c < K; c++) {

        for (int t = 0; t < numThreads; t++) {

            totalCounts[c] += threadCounts[t][c];

            for (int j = 0; j < FEATURES; j++) {
                newCentroids[c][j] +=
                    threadSums[t][c][j];
            }
        }
    }

    // Calculate new centroids
    for (int c = 0; c < K; c++) {

        if (totalCounts[c] == 0) {
            // Preserve old centroid if cluster is empty
            newCentroids[c] = oldCentroids[c];
        }
        else {
            for (int j = 0; j < FEATURES; j++) {
                newCentroids[c][j] /=
                    totalCounts[c];
            }
        }
    }

    return newCentroids;
}

// -----------------------------
// Parallel K-Means
// -----------------------------
vector<Point> runKMeansParallel(
    const vector<Point>& data,
    vector<int>& labels
) {
    vector<Point> centroids =
        initializeCentroids(data);

    for (int iteration = 0;
         iteration < MAX_ITERATIONS;
         iteration++) {

        // Step 1:
        // Assign each point to nearest centroid
        assignClustersParallel(
            data,
            centroids,
            labels
        );

        // Step 2:
        // Calculate new centroids
        vector<Point> newCentroids =
            updateCentroidsParallel(
                data,
                labels,
                centroids
            );

        // Step 3:
        // Check centroid movement
        double maxMovement = 0.0;

        for (int c = 0; c < K; c++) {

            double movement =
                sqrt(
                    squaredDistance(
                        centroids[c],
                        newCentroids[c]
                    )
                );

            maxMovement =
                max(maxMovement, movement);
        }

        centroids = newCentroids;

        // Step 4:
        // Check convergence
        if (maxMovement < TOLERANCE) {
            break;
        }
    }

    // Final assignment using final centroids
    assignClustersParallel(
        data,
        centroids,
        labels
    );

    return centroids;
}

// -----------------------------
// Main
// -----------------------------
int main(int argc, char* argv[]) {

    if (argc != 2) {
        cerr << "Usage: "
             << argv[0]
             << " <csv_file>"
             << endl;

        return 1;
    }

    try {

        string filename = argv[1];

        // Load dataset
        vector<Point> data =
            loadData(filename);

        vector<int> labels(data.size());

        // -----------------------------
        // Time ONLY K-Means execution
        // -----------------------------
        auto start =
            chrono::high_resolution_clock::now();

        vector<Point> centroids =
            runKMeansParallel(
                data,
                labels
            );

        auto end =
            chrono::high_resolution_clock::now();

        chrono::duration<double> elapsed =
            end - start;

        // -----------------------------
        // Calculate cluster sizes
        // -----------------------------
        vector<int> clusterCounts(K, 0);

        for (int label : labels) {
            clusterCounts[label]++;
        }

        // -----------------------------
        // Calculate inertia
        // -----------------------------
        double inertia = 0.0;

        for (int i = 0;
             i < static_cast<int>(data.size());
             i++) {

            inertia +=
                squaredDistance(
                    data[i],
                    centroids[labels[i]]
                );
        }

        // -----------------------------
        // Output
        // -----------------------------
        cout << "Customers: "
             << data.size()
             << endl;

        cout << "K: "
             << K
             << endl;

        cout << "Threads: "
             << omp_get_max_threads()
             << endl;

        cout << "Execution time: "
             << elapsed.count()
             << " seconds"
             << endl;

        cout << "Inertia: "
             << inertia
             << endl;

        cout << "Cluster sizes:"
             << endl;

        for (int c = 0; c < K; c++) {
            cout << "Cluster "
                 << c
                 << ": "
                 << clusterCounts[c]
                 << endl;
        }

    }
    catch (const exception& e) {

        cerr << "Error: "
             << e.what()
             << endl;

        return 1;
    }

    return 0;
}
