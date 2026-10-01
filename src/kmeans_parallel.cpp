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
#include <omp.h>

using namespace std;

// Our dataset has exactly three numerical features.
constexpr int FEATURES = 3;
constexpr int K = 4;
constexpr int MAX_ITERATIONS = 100;
constexpr double TOLERANCE = 1e-4;

using Point = array<double, FEATURES>;

// --------------------------------------------------
// 1. Read the preprocessed CSV file
// --------------------------------------------------

vector<Point> loadData(const string& filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    vector<Point> data;
    string line;

    // Skip the CSV header.
    getline(file, line);

    while (getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        stringstream ss(line);
        string value;
        Point point;

        for (int j = 0; j < FEATURES; j++) {
            if (!getline(ss, value, ',')) {
                throw runtime_error("Invalid CSV row: " + line);
            }

            point[j] = stod(value);

            if (!isfinite(point[j])) {
                throw runtime_error("Non-finite feature value");
            }
        }

        data.push_back(point);
    }

    if (data.size() < K) {
        throw runtime_error("Not enough customers for K clusters");
    }

    return data;
}

// --------------------------------------------------
// 2. Calculate squared Euclidean distance
// --------------------------------------------------

double squaredDistance(const Point& a, const Point& b) {
    double distance = 0.0;

    for (int j = 0; j < FEATURES; j++) {
        double difference = a[j] - b[j];
        distance += difference * difference;
    }

    return distance;
}

// --------------------------------------------------
// 3. Initialize centroids
// --------------------------------------------------

vector<Point> initializeCentroids(const vector<Point>& data) {
    mt19937 generator(42);

    // Select K distinct row indices.
    vector<int> indices(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        indices[i] = static_cast<int>(i);
    }

    // Partial Fisher-Yates shuffle.
    for (int i = 0; i < K; i++) {
        uniform_int_distribution<int> distribution(
            i, static_cast<int>(data.size()) - 1
        );

        int selected = distribution(generator);
        swap(indices[i], indices[selected]);
    }

    vector<Point> centroids;

    for (int i = 0; i < K; i++) {
        centroids.push_back(data[indices[i]]);
    }

    return centroids;
}

// --------------------------------------------------
// 4. PARALLEL: Assign customers to nearest centroid
// --------------------------------------------------

void assignClustersParallel(
    const vector<Point>& data,
    const vector<Point>& centroids,
    vector<int>& labels
) {
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(data.size()); i++) {

        double minimumDistance =
            numeric_limits<double>::max();

        int nearestCluster = 0;

        for (int c = 0; c < K; c++) {
            double distance =
                squaredDistance(data[i], centroids[c]);

            if (distance < minimumDistance) {
                minimumDistance = distance;
                nearestCluster = c;
            }
        }

        labels[i] = nearestCluster;
    }
}

// --------------------------------------------------
// 5. PARALLEL: Calculate new centroids
// --------------------------------------------------

vector<Point> updateCentroidsParallel(
    const vector<Point>& data,
    const vector<int>& labels,
    const vector<Point>& oldCentroids
) {
    int numThreads = omp_get_max_threads();

    // Each thread gets its own sums and counts.
    vector<vector<Point>> threadSums(
        numThreads,
        vector<Point>(K, Point{})
    );

    vector<vector<int>> threadCounts(
        numThreads,
        vector<int>(K, 0)
    );

    // Each thread works on its own local sums/counts.
    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();

        #pragma omp for
        for (long long i = 0;
             i < static_cast<long long>(data.size());
             i++) {

            int cluster = labels[i];

            threadCounts[threadId][cluster]++;

            for (int j = 0; j < FEATURES; j++) {
                threadSums[threadId][cluster][j] += data[i][j];
            }
        }
    }

    // Combine results from all threads.
    vector<Point> sums(K, Point{});
    vector<int> counts(K, 0);

    for (int t = 0; t < numThreads; t++) {
        for (int c = 0; c < K; c++) {

            counts[c] += threadCounts[t][c];

            for (int j = 0; j < FEATURES; j++) {
                sums[c][j] += threadSums[t][c][j];
            }
        }
    }

    // Calculate new centroids.
    vector<Point> newCentroids = oldCentroids;

    for (int c = 0; c < K; c++) {

        if (counts[c] == 0) {
            // Keep previous centroid if cluster is empty.
            continue;
        }

        for (int j = 0; j < FEATURES; j++) {
            newCentroids[c][j] =
                sums[c][j] / counts[c];
        }
    }

    return newCentroids;
}

// --------------------------------------------------
// 6. Parallel K-Means
// --------------------------------------------------

int runKMeansParallel(
    const vector<Point>& data,
    vector<Point>& centroids,
    vector<int>& labels
) {
    int iterations = 0;

    for (int iteration = 0;
         iteration < MAX_ITERATIONS;
         iteration++) {

        // Phase 1: Parallel assignment
        assignClustersParallel(
            data,
            centroids,
            labels
        );

        // Phase 2: Parallel centroid update
        vector<Point> newCentroids =
            updateCentroidsParallel(
                data,
                labels,
                centroids
            );

        // Calculate maximum centroid movement.
        double maximumMovement = 0.0;

        for (int c = 0; c < K; c++) {

            double movement = sqrt(
                squaredDistance(
                    centroids[c],
                    newCentroids[c]
                )
            );

            if (movement > maximumMovement) {
                maximumMovement = movement;
            }
        }

        centroids = newCentroids;
        iterations++;

        if (maximumMovement < TOLERANCE) {
            break;
        }
    }

    // Ensure final labels match final centroid positions.
    assignClustersParallel(
        data,
        centroids,
        labels
    );

    return iterations;
}

// --------------------------------------------------
// 7. Main
// --------------------------------------------------

int main(int argc, char* argv[]) {

    if (argc != 2) {
        cerr << "Usage: " << argv[0]
             << " <dataset.csv>" << endl;

        return 1;
    }

    try {

        vector<Point> data =
            loadData(argv[1]);

        vector<Point> centroids =
            initializeCentroids(data);

        vector<int> labels(
            data.size(),
            -1
        );

        // Time only the clustering computation.
        auto start =
            chrono::steady_clock::now();

        int iterations =
            runKMeansParallel(
                data,
                centroids,
                labels
            );

        auto end =
            chrono::steady_clock::now();

        double elapsed =
            chrono::duration<double>(
                end - start
            ).count();

        // Calculate cluster counts and inertia.
        vector<int> counts(K, 0);
        double inertia = 0.0;

        for (size_t i = 0;
             i < data.size();
             i++) {

            int cluster = labels[i];

            counts[cluster]++;

            inertia += squaredDistance(
                data[i],
                centroids[cluster]
            );
        }

        cout << "Customers: "
             << data.size()
             << endl;

        cout << "K: "
             << K
             << endl;

        cout << "Iterations: "
             << iterations
             << endl;

        cout << "Execution time: "
             << elapsed
             << " seconds"
             << endl;

        cout << "Inertia: "
             << inertia
             << endl;

        for (int c = 0; c < K; c++) {

            cout << "\nCluster "
                 << c
                 << " ("
                 << counts[c]
                 << " customers)"
                 << endl;

            cout << "Centroid: ";

            for (int j = 0;
                 j < FEATURES;
                 j++) {

                cout << centroids[c][j]
                     << " ";
            }

            cout << endl;
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