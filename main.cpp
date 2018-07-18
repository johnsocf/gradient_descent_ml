#include <iostream>
using namespace std;

// determines size of steps in each iteration of gradient descent
#define learning_rate .0001;

// defines whether we are considering all
// training data or small subset.
#define batch true;



int something[10];
int my_main();

// general reference for variables
// m = number of training examples
// x's = input variables/ features
// y's = output variable/ target variable

// import data
void import_data();

// initialize structures
void init_vector(int arg[], int length);
void init_2d_matrix(int arg[], int length);
void init_3d_matrix(int arg[], int length);

// set up supervised learning processes
int scale_data(int arg[]);
void shuffle_data(int arg[]);
void build_training_set(int arg[]);
void build_test_set(int arg[]);

// matrix functions
void multiple_matrices(int arg[], int arg[]);
void add_matrices(int arg[], int arg[]);

// initialize supervised learning algorithm
void initialize_gradient_descent(int arg[]);

// ml algorithms
int hypothesis_linear_regression(int arg[]);
void gradient_descent(int arg[]);
void gradient_descent_min_cost(int arg[]);

// ml algorithms multi features (takes matrix data, not just vectors)

// additional notation:
// n = number of features
// x^(i) = input features of ith training example.  INDEX into matrix column for ith training example, not exponent
// xj^(i) = value feature j in ith training example.  INDEX into matrix row for also j attribute.  (subscript j)
// not 0 indexed.
// params = O1, O2, O3..., On
// params = O theta.  where theta is n+1 dimensional vector.  these are a VECTOR.
// cost function J gets passed a vector instead of individual params.
// J(O).
int hypothesis_linear_regression_multi_feat(int arg[]);
void gradient_descent_multi_feat(int arg[]);
void gradient_descent_min_cost_multi_feat(int arg[]);

//ToDo: plot data from matrix initially
// plot learnt algorithm
// using MATLAB.

// possible enhancements:
    // 3D structure
    // programatically adjustable learning rate.
    // feature scaling.


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

void import_data() {
    // import data via csv.
    // build matrix
    // build test sets
}

// creating testing sets
void shuffle_data(int arg[]) {
    // shuffle data to random order
}

void build_training_set(int arg[]) {
    // this will be 75% of test data;
}
void build_test_set(int arg[]) {
    // this will be 25% of test data;
}

void scale_data(int arg[]) {
    // utilize feature scaling to arrive at best contoured data for regression.
    // utilizes matrix mult.
}

// init
void initialize_gradient_descent(int arg[]) {
    // initialize gradient descent with starting values for each param
    // initialize params of O0, O1, and all j = params from i to m
    // each param gets initialized to 0
}

// algorithms
int hypothesis_linear_regression(int arg[]) {
    // return hypothesis based on linear regression formula
    // hO(x) = O0 + O1x;
    //return O0 + O1 * x;
    // return 0;
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
    // J(O0, O1) = (1/2m) * sum of for each i=1 to m: (hO(x^(i))-y^(i))^2
    // derivative in this context is: (hO(x^(i))-y^(i))^2

    // determine partial derivative for each j term.
    // repeat until convergence.
    // updating simultaneously.
    // j = 0
    // (1/m) * sum of for each i=1 to m: (hO(x^(i))-y^(i))
    // j = 0
    // (1/m) * sum of for each i=1 to m: (hO(x^(i))-y^(i)) * x^(i)

    // for h0 - utilize hypothesis linear model drawing on linear descent eq;
}

// multi dimensional
int hypothesis_linear_regression_multi_feat(int arg[]) {
    // return hypothesis
    //h0(x) = O0 + O1x1 + O2x2... etc
    // where weight = param value of attribute for item:
    //h0(x) = O0 + for each param: (weight * x)

    // each param will carry a weight which will effect the hypothesis.
    // x gets assigned 0
    // x0 = 1.  creating 0 index based vector.  [x0, x1, x2, x3, x4 etc...]
    //  theta (O) also is 0 indexed. - since the first coefficient is always 1, these matrices match up.
    // Otx - theta transpose x.
    // multivariate linear regression.
    // [O0, O1, O2, O3, O4, O5...] and vert array of:
    // [
        // x0,
        // x1,
        // x2,
        // x3,
        // x4
        // etc...
    // ]
    // Otx - Theta transpose x.  builds linear equation of multivariate linear regression.
    // return hypothesis hO(x) = Otx.
        // O0x0.  x0 = 1.
        // h0(x) = Otx = O0x0 + O1x1 + O2x2... + ... + Onxn
        // we are representing linear algebra for our predictibility learning algorithms now in
        // a useful matrix structure.
}

void gradient_descent_multi_feat(int arg[]) {
    // cost function with multiple params:
    // J(O0, O1, ...On) = (1/2m) * sum of for each i=1 to m: (( for each j=0 to j=n Ojx^(i)j )-y^(i))^2
    // basically the hypothesis for taking into account each param, before this:
    // J(O0, O1, ...On) = (1/2m) * sum of for each i=1 to m: (hO(x^(i))-y^(i))^2
    // J(O) - J of theta, where theta is a vector.
    // J(0) = (1/2m) * sum of for each i=1 to m: (( for each j=0 to j=n Ojx^(i)j )-y^(i))^2
}

void gradient_descent_min_cost_multi_feat(int arg[]) {
    // toDo: define loop more clearly with stopping point when Oj == Oj. or when Oj == 0.
    // note xj^(i) is ith term jth attr.
    // for each j where j=0, ... j = n
    // Oj = Oj - learning_rate * (1/m) sum of for each (hO(x^(i))-y^(i))xj^(i)
    // simultaneously update.

    // because x0^(i) = 1, same pattern as above with one attr, but iteration through each.
    // ToDo: define input and output and configure.
    // ToDo: given vector for an item returns to us the output - on a scale of linear regression for output y.
}