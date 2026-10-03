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

using namespace std;

// K-Means features: lifetime orders, lifetime spend, and AOV.
constexpr int FEATURES = 3;
constexpr int K = 4;
constexpr int MAX_ITERATIONS = 100;
constexpr double TOLERANCE = 1e-4;

using Point = array<double, FEATURES>;

// --------------------------------------------------
// 1. Read the preprocessed CSV file
// --------------------------------------------------

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
        throw runtime_error("Cannot open file: " + filename);
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
        if (trim(line).empty()) {
            continue;
        }

        try {
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
                    throw runtime_error("Non-finite feature value");
                }
            }

            data.push_back(point);
        } catch (const invalid_argument&) {
            throw runtime_error("Invalid numeric value in CSV row: " + line);
        } catch (const out_of_range&) {
            throw runtime_error("Numeric value out of range in CSV row: " + line);
        }
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
// 4. Assign customers to their nearest centroids
// --------------------------------------------------

void assignClusters(
    const vector<Point>& data,
    const vector<Point>& centroids,
    vector<int>& labels
) {
    for (size_t i = 0; i < data.size(); i++) {
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
// 5. Calculate new centroids
// --------------------------------------------------

vector<Point> updateCentroids(
    const vector<Point>& data,
    const vector<int>& labels,
    const vector<Point>& oldCentroids
) {
    vector<Point> sums(K, Point{});
    vector<int> counts(K, 0);

    for (size_t i = 0; i < data.size(); i++) {
        int cluster = labels[i];

        counts[cluster]++;

        for (int j = 0; j < FEATURES; j++) {
            sums[cluster][j] += data[i][j];
        }
    }

    vector<Point> newCentroids = oldCentroids;

    for (int c = 0; c < K; c++) {
        if (counts[c] == 0) {
            // Keep the previous centroid if empty.
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
// 6. Sequential K-Means
// --------------------------------------------------

int runKMeans(
    const vector<Point>& data,
    vector<Point>& centroids,
    vector<int>& labels
) {
    int iterations = 0;

    for (int iteration = 0;
         iteration < MAX_ITERATIONS;
         iteration++) {

        // Phase 1: Assignment
        assignClusters(data, centroids, labels);

        // Phase 2: Update
        vector<Point> newCentroids =
            updateCentroids(data, labels, centroids);

        // Calculate maximum centroid movement.
        double maximumMovement = 0.0;

        for (int c = 0; c < K; c++) {
            double movement = sqrt(
                squaredDistance(
                    centroids[c], newCentroids[c]
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
    assignClusters(data, centroids, labels);

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
        vector<Point> data = loadData(argv[1]);

        vector<Point> centroids =
            initializeCentroids(data);

        vector<int> labels(data.size(), -1);

        // Time only the clustering computation.
        auto start = chrono::steady_clock::now();

        int iterations =
            runKMeans(data, centroids, labels);

        auto end = chrono::steady_clock::now();

        double elapsed =
            chrono::duration<double>(end - start).count();

        vector<int> counts(K, 0);
        double inertia = 0.0;

        for (size_t i = 0; i < data.size(); i++) {
            int cluster = labels[i];

            counts[cluster]++;
            inertia += squaredDistance(
                data[i], centroids[cluster]
            );
        }

        cout << "Customers: " << data.size() << endl;
        cout << "K: " << K << endl;
        cout << "Iterations: " << iterations << endl;
        cout << "Execution time: " << elapsed
             << " seconds" << endl;
        cout << "Inertia: " << inertia << endl;

        for (int c = 0; c < K; c++) {
            cout << "\nCluster " << c
                 << " (" << counts[c]
                 << " customers)" << endl;

            cout << "Centroid: ";

            for (int j = 0; j < FEATURES; j++) {
                cout << centroids[c][j] << " ";
            }

            cout << endl;
        }

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
