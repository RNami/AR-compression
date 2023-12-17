#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

std::vector<double> calculateACoefs(const std::vector<std::vector<double>>& pic) {
    int rows = pic.size();
    int cols = pic[0].size();
    std::vector<double> aCoefs(4);

    std::vector<double> row_sums(rows);
    for (int i = 0; i < rows; i++) {
        row_sums[i] = std::accumulate(pic[i].begin(), pic[i].end(), 0.0);
    }
    double pic_mean = std::accumulate(row_sums.begin(), row_sums.end(), 0.0) / (rows * cols);

    double R22, R33, R44, R55;
    double R12, R34, R45;
    double R13, R23, R14, R24, R15, R25, R35;

    R22 = R33 = R44 = R55 = 0.0;
    R12 = R34 = R45 = 0.0;
    R13 = R23 = R14 = R24 = R15 = R25 = R35 = 0.0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double pic_ij = pic[i][j];
            double pic_ij1 = (j < cols - 1) ? pic[i][j + 1] : 0.0;
            double pic_i1j1 = (i < rows - 1 && j < cols - 1) ? pic[i + 1][j + 1] : 0.0;
            double pic_i1j = (i < rows - 1) ? pic[i + 1][j] : 0.0;
            double pic_i2j1 = (i < rows - 2 && j < cols - 1) ? pic[i + 2][j + 1] : 0.0;
            double pic_ij2 = (i < rows - 1 && j < cols - 2) ? pic[i][j + 2] : 0.0;
            double pic_i2j2 = (i < rows - 2 && j < cols - 2) ? pic[i + 2][j + 2] : 0.0;

            R22 += pic_ij * pic_ij;
            R33 += pic_ij * pic_ij;
            R44 += pic_ij * pic_ij;
            R55 += pic_ij * pic_ij;

            R12 += pic_ij * pic_ij1;
            R34 += pic_ij * pic_ij1;
            R45 += pic_ij * pic_i1j1;

            R13 += pic_ij * pic_i1j1;
            R23 += pic_ij * pic_i1j1;
            R14 += pic_ij * pic_i1j;
            R24 += pic_ij * pic_i1j;
            R15 += pic_ij * pic_i2j1;
            R25 += pic_ij * pic_ij2;
            R35 += pic_ij * pic_i2j2;
        }
    }

    R22 /= (rows * cols);
    R33 /= (rows * cols);
    R44 /= (rows * cols);
    R55 /= (rows * cols);

    R12 /= (rows * cols);
    R34 /= (rows * cols);
    R45 /= (rows * cols);

    R13 /= (rows * cols);
    R23 /= (rows * cols);
    R14 /= (rows * cols);
    R24 /= (rows * cols);
    R15 /= (rows * cols);
    R25 /= (rows * cols);
    R35 /= (rows * cols);

    std::vector<std::vector<double>> Phi = {{R22, R23, R24, R25},
                                            {R23, R22, R34, R35},
                                            {R24, R34, R22, R45},
                                            {R25, R35, R45, R22}};

    std::vector<double> R = {R12, R13, R14, R15};

    for (int i = 0; i < 4; i++) {
        aCoefs[i] = 0.0;
        for (int j = 0; j < 4; j++) {
            aCoefs[i] += Phi[i][j] * R[j];
        }
    }

    return aCoefs;
}

int main() {
    std::vector<std::vector<double>> pic = {{1, 2, 3, 4},
                                            {5, 6, 7, 8},
                                            {9, 10, 11, 12},
                                            {13, 14, 15, 16}};

    std::vector<double> aCoefs = calculateACoefs(pic);

    std::cout << "aCoefs: ";
    for (double coef : aCoefs) {
        std::cout << coef << " ";
    }
    std::cout << std::endl;

    return 0;
}