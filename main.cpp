#include <iostream>
#include <libiomp/omp.h>
#include <cmath>



#define dx(N) 1.0 / N
#define dy(N) 1.0 / N

using namespace std;

int const N1 = 300;
int const N2 = 200;
int const N3 = 500;
int const N4 = 1000;
int const N5 = 2000;
int k;

// formula : u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) - h * h * f[i][j];

/*
 * [       ^      ]
 * [      f1      ]
 * [ < f3    f4 > ]
 * [      f2      ]
 * [      v       ]
 */

void report(int, double, const char *);
void to_file(double*, int);
void print_matr_u(double*, int);
void ode_solve_seq(double*, double*, int, double, bool);
void ode_solve_parallel(double*, double*, int, double, bool);
void init_u(double*, int);
void init_f(double*, int);
void exec_comp(int, bool);
void replot(double*, int);
void render_plot(double*);

int main() {
    omp_set_num_threads(4);
    exec_comp(N1, true);
    exec_comp(N2, true);
    exec_comp(N3, false);
    exec_comp(N4, false);
    exec_comp(N5, false);
}

/////////////////////////////////////////////////////////////////////////////////

double f1(double x) {
    return 0;//sin(x*x);
}

double f2(double x) {
    return 2;//cos(3*x);
}

double f3(double x) {
    return (2 - x) * (2 - x) / 2; //10*sin(x*x);
}

double f4(double x) {
    return 2 - x; //10*sin(6*x);
}

double func(double x, double y) {
    return (exp(-x) + exp(-y)) / 10;
}

void init_u(double* u, int N) {
    for (int j = 0; j < N; j++) u[0 + j] = f1(j * dx(N));
    for (int j = 0; j < N; j++) u[(N-1) * N + j] = f2(j * dx(N));
    for (int i = 0; i < N; i++) u[i* N + 0] = f3((N - i) * dy(N));
    for (int i = 0; i < N; i++) u[i * N + N-1] = f4((N - i) * dy(N));
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            u[i* N + j] = 0;
        }
    }
}

void init_f(double* f, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            f[i * N + j] = func(dx(N) * j, dy(N) * i);
        }
    }
}

void print_matr_u(double* u, int N) {
    cout << endl;
    for (int i = 0; i < N; i++){
        printf("%.5f", u[i * N + 0]);
        for (int j = 1; j < N; j++) {
            printf(" %.5f", u[i * N + j]);
        }
        printf("\n");
    }
}

void to_file(double* u, int N) {
    FILE * f = fopen("out.txt", "w+");
    for (int i = 0; i < N; i++){
        fprintf(f, "%.5f", u[i * N + 0]);
        for (int j = 1; j < N; j++) {
            fprintf(f, " %.5f", u[i * N + j]);
        }
        fprintf(f, "\n");
    }
    fflush(f);
    fclose(f);
}

void report(int N, double time, const char* method) {
    printf("computed %d x %d for %.5f seconds (%s)\n", N, N, time, method);
}

void ode_solve_parallel(double* u, double* f, int N, double eps, bool need_replot) {
    double deltaMax, temp, delta, deltaMaxThread = 0;
    k = 0;
    int i, j;
    do {
        deltaMax = 0;
#pragma omp parallel for private (i, temp, delta, deltaMaxThread)
        for(i = 1; i < N - 1; i++) {
            deltaMaxThread = 0;
            for (j = 1; j < N - 1; j++) {
                double a = i == 0 ? 0 : u[(i - 1) * N + j];
                double b = i == N - 1 ? 0 : u[(i + 1) * N + j];
                double c = j == 0 ? 0 : u[i * N + j - 1];
                double d = j == N - 1 ? 0 : u[i * N + j + 1];
                temp = u[i * N + j];
                u[i * N + j] = 0.25 * (a + b + c + d - dx(N) * dy(N) * f[i * N + j]);
                delta = fabs(temp - u[i * N + j]);
                if (deltaMaxThread < delta) deltaMaxThread = delta;

            }
        }
#pragma omp critical
        {
            k++;
            if (deltaMax < deltaMaxThread) deltaMax = deltaMaxThread;
            //if (need_replot) replot(u, N);
        }
    } while (deltaMax > eps);
}

void ode_solve_seq(double* u, double* f, int N, double eps, bool need_replot) {
    double deltaMax, temp, delta;
    int i, j;
    do {
        deltaMax = 0;
        for(i = 1; i < N-1; i++) {
            for (j = 1; j < N - 1; j++) {
                double a = i == 0 ? 0 : u[(i - 1) * N + j];
                double b = i == N - 1 ? 0 : u[(i + 1) * N + j];
                double c = j == 0 ? 0 : u[i * N + j - 1];
                double d = j == N - 1 ? 0 : u[i * N + j + 1];
                temp = u[i * N + j];
                u[i * N + j] = 0.25 * (a + b + c + d - dx(N) * dy(N) * f[i * N + j]);
                delta = fabs(temp - u[i * N + j]);
                if (deltaMax < delta) deltaMax = delta;
            }
        }
        //if (need_replot) replot(u, N);
    } while (deltaMax > eps);
}

void init_out_txt() {
    FILE* f = fopen("out.txt", "w+");
    fprintf(f, "0.0 1.0 2.0\n1.0 2.0 3.0\n2.0 3.0 4.0");
    fflush(f);
    fclose(f);
}


void exec_comp(int N, bool plot) {
    double* u = (double*)malloc(sizeof(double) * N * N);
    double* f = (double*)malloc(sizeof(double) * N * N);
    double start, end;
    init_out_txt();
    cout << "BEGIN COMPUTATION" << endl;
    init_u(u, N);
    init_f(f, N);
    start = omp_get_wtime();
    ode_solve_parallel(u, f, N, 0.0001, plot);
    end = omp_get_wtime();
    report(N, end - start, "parallel");
    init_u(u, N);
    init_f(f, N);
    start = omp_get_wtime();
    ode_solve_seq(u, f, N, 0.0001, plot);
    end = omp_get_wtime();
    report(N, end - start, "sequential");
    if (plot) {
        replot(u, N);
    }
    free(u);
    free(f);
    cout << endl;
}

void replot(double* u, int N) {
    to_file(u, N);
    cout << endl << k << endl;
    system("gnuplot -p -e \"set pm3d map;splot 'out.txt' matrix\"");
    getchar();
    system("pkill gnuplot");
}