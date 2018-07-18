#include <iostream>
using namespace std;
#define learning_rate .0001;

int something[10];
int my_main();

// initialize structures
void init_vector(int arg[], int length);
void init_2d_matrix(int arg[], int length);
void init_3d_matrix(int arg[], int length);

// set up supervised learning processes
void shuffle_data(int arg[]);
void build_training_set(int arg[]);
void build_test_set(int arg[]);

// initialize supervised learning algorithm
void initialize_gradient_descent(int arg[]);

// ml algorithms
void gradient_descent(int arg[]);
void gradient_descent_min_cost(int arg[]);


void runThisFunction(int arg[]);


int my_main() {
    std::cout << "Hello, World!" << std::endl;
    runThisFunction(something);
    return 0;
}

int main() {
    return my_main();
}

void runThisFunction(int arg[]) {
    cout << "this function \n";
}

void shuffle_data(int arg[]) {
    // shuffle data to random order
}

void build_training_set(int arg[]) {
    // this will be 75% of test data;
}
void build_test_set(int arg[]) {
    // this will be 25% of test data;
}


void initialize_gradient_descent(int arg[]) {
    // initialize gradient descent with starting values for each param
}

void gradient_descent(int arg[]) {
    // combined linear function (hypothesis)
    // and cost function (which tells us our margin of error)
    // moves to derivative becoming zero.
    // learning rate adjusts steps (will be slow if 'learning_rate' def is too small)

    // for each param.  ex j=0, j=1
    // Oj = Oj - learning_rate * derivative of J(O0, O1)
    // repeat until convergence
}

void gradient_descent_min_cost(int arg[]) {
    // apply gradient descent to linear regression model to minimize cost function.
    // h = hypothesis
    // J(O0, O1) = (1/2m) for each i=1 to m: (hO(x^(i))-y^(i))^2
}