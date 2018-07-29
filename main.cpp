#include <iostream>
#include <iomanip>
#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;

int rowA = 0;
int colA = 0;

// determines size of steps in each iteration of gradient descent
//#define learning_rate .0001;

// defines whether we are considering all
// training data or small subset.
#define batch true;

int test_loops = 0;

int something[10];
int my_main();

// general reference for variables
// m = number of training examples
// x's = input variables/ features
// y's = output variable/ target variable
// capital X = matrix.  (caps designate matrix)

// import data and helpers
void import_data();
void print_matrix(vector< vector<double> > array);
void print_vector(vector<double> my_vector);
vector< vector<double> > build_matrix(string file_name);
vector<double> get_vector_slice(vector<vector <double> > array, int column);
vector< vector<double> > get_matrix_input_data(vector<vector <double> > array, int column);
void separate_vector_from_matrix_by_column(vector< vector<double> > matrix, int column);

// initialize structures
void init_vector(int arg[], int length);
void init_2d_matrix(int arg[], int length);
void init_3d_matrix(int arg[], int length);

// set up supervised learning processes
void scale_data(int arg[]);
void shuffle_data(int arg[]);
void build_training_set(int arg[]);
void build_test_set(int arg[]);

// matrix functions
int training_example_length(int y[]);
void multiple_matrices(int X[], int X2[]);
void add_matrices(int X[], int X2[]);

// initialize supervised learning algorithm
vector<double> initialize_gradient_descent(int initial_values, int param_length);

// ml algorithms
vector<double>  hypothesis_linear_regression(vector<double> coefficient_training_vector, vector< vector<double> > data_matrix);
void gradient_descent(int arg[]);
void gradient_descent_min_cost(vector< vector<double> > total_matrix, int y_column, vector<double> cost_vector, vector<double> hypothesis_vector, double learning_rate);

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
    int this_array[5] = {3, 4, 5, 6, 7};
    int m = training_example_length(this_array);

//    for (int i=0; i <=10; i++) {
//        for (int j=0; j<= 10; j++) {
//            for (int k=0; k<= 10; k++) {
//                cout << i << "," << j << "," << k << "," << (2 + 3*i + 8*j) << "\n";
//            }
//            //cout << i << "," << j << "," << (2 + 3*i + 4*j) << "\n";
//        }
//    }

    import_data();

    return 0;
}

int main() {
    return my_main();
}

void runThisFunction(int arg[]) {
    cout << "this function \n";
}

void import_data() {



    // column to extract y vector
    int y_column = 2;
    vector< vector<double> > data_matrix = build_matrix("test4.csv");
    cout << "data matrix size: " <<  data_matrix.size() << "\n";
    vector<double> coefficient_training_vector = initialize_gradient_descent(100, data_matrix[0].size());
    vector<double> cost_vector = initialize_gradient_descent(100, data_matrix[0].size());
    double learning_rate = .004;
    gradient_descent_min_cost(data_matrix, y_column, cost_vector, coefficient_training_vector, learning_rate);

    // import data via csv.
    // build matrix
    // build test sets
}

void separate_vector_from_matrix_by_column(vector< vector<double> > matrix, int column) {
    get_vector_slice(matrix, column);
    get_matrix_input_data(matrix, column);
}

vector< vector<double> > build_matrix(string fileName) {
    ifstream inFile;
    inFile.open(fileName);
    if (!inFile.is_open()) {
        cout << "open file failed \n";
    } else {
        cout << "file opened \n";
    }
    string line, val;
    vector<double> v;
    vector<vector <double> > array;


    while (getline(inFile, line)) {
        vector<double> v;
        stringstream s (line);
        while (getline (s, val, ','))
            //cout << "print val: " <<  stod(val) << "\n";
            v.push_back(stod(val));
        //cout << "pushed back \n";
        array.push_back(v);
    }

    return array;
}

vector< vector<double> > get_matrix_input_data(vector<vector <double> > array, int column) {
    // reference vector by pointer
    //vector<double> pos = array;
    //vector< vector<double> > *pointer = &array;
    //double some_num = pointer*[0][1];
    //cout << "print val: " <<  some_num << "\n";

    // first iterate through rows.
    for (int i=0; i<array.size(); i++) {
        //cout << "i: " << i << "\n";
        vector<double>::iterator it;
        it = array[i].begin();
        // add identity row to first


        // remove output row.
        array[i].erase(array[i].begin() + 2);
        array[i].insert(it, 1);
        //cout << "print val: " <<  array[i][1] << "\n";
    };

    // add identity


    //print_matrix(array);
    return array;
}

vector<double> get_vector_slice(vector<vector <double> > array, int column) {
    vector<double> result;
    // reference vector by pointer
    //print_matrix(array);

    //cout << "print val: " <<  some_num << "\n";
    for (int i=0; i<array.size(); i++) {
        //cout << "i: " << i << "\n";
        result.push_back(array[i][column]);
        array[i].erase(array[i].begin() + 2);
        //cout << "print val: " <<  array[i][2] << "\n";
    };
    //cout << "erased? : \n";
    //print_matrix(array);
    return result;
}



void print_matrix(vector< vector<double> > array) {
    for (auto& row:array) {
        for (auto& val:row)
            cout << val << " ";
        cout << "\n";
    }
}

void print_vector(vector<double> my_vector) {
    for (int i=0; i< my_vector.size(); i++)
        cout << my_vector[i] << "   ";
    cout << "\n";
}

// define variables
int training_example_length(int y[]) {
    //return length(y);
    return 0;
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
vector<double> initialize_gradient_descent(int initial_values, int param_length) {
    // initialize gradient descent with starting values for each param
    // initialize params of O0, O1, and all j = params from i to m
    // each param gets initialized to 0
    cout << "param length: " << param_length << "\n";
    vector<double> initial_cost_matrix;
    for (int i=0; i<param_length; i++) {
        initial_cost_matrix.push_back(initial_values);
        //initial_cost_matrix[i] = initial_values;
    }

    //print_vector(initial_cost_matrix);
    return initial_cost_matrix;
}

// algorithms
vector<double>  hypothesis_linear_regression(vector<double> coefficient_training_vector, vector< vector<double> > data_matrix) {
    // return hypothesis based on linear regression formula
    // hO(x) = O0 + O1x;
    //return O0 + O1 * x;
    //cout << "coefficient training in linear reg: \n";
    //print_vector(coefficient_training_vector);
    vector<double> hypothesis;

    // for each row in data vector... use eq.
    // data times param. at i.

    int m = data_matrix.size();
    int n = data_matrix[0].size();
//    if (data_matrix.size()!=0) {
//        c_s = data_matrix[0].size();
//    }

    //cout << " row length: " << m << "\n";
    //cout << " parameter length: " << c_s << "\n";
    //cout << "coefficient training: \n";
    //print_vector(coefficient_training_vector);


    for (int row=0; row < m; row++) {
         //first row.
        double hyp = 0;
        for (int column=0; column < n; column++) {
            //cout << "param row" << row << "\n params: " << coefficient_training_vector[column] << " \n";
            //cout << "data in each row row: " << row << " \ndata in row: " << data_matrix[row][column] << "\n";
            hyp += coefficient_training_vector[column] * data_matrix[row][column];
             //do stuff ...
        }
        hypothesis.push_back(hyp);
    }

    //given a set of data with an identity row at position 0 for the linear eq.
    //calculates a hypothesis vector for each row based on multiplying params times data for each column.
    // vector has a value per row.  calculations are per row.

    //print_vector(hypothesis);

//    for (int i=0; i<param_vector.size(); i++) {
//        // toDo: data_vector[0] is temp because we're passing in a matrix but accessing a simple vector untill
//        // we get to multi variate
//        hypothesis[i] += (param_vector[i] * data_matrix[0][i]);
//    }


    return hypothesis;
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

void gradient_descent_min_cost(vector< vector<double> > total_matrix, int y_column, vector<double> cost_vector, vector<double> coefficient_training_vector, double learning_rate) {
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

    vector<double> y_vector = get_vector_slice(total_matrix, y_column);
    vector< vector<double> > data_matrix = get_matrix_input_data(total_matrix, y_column);
    //cout << "rint matrix data: \n";
    //print_matrix(data_matrix);
    //cout << "coefficient training: \n";
    //sprint_vector(coefficient_training_vector);

    vector<double> hypothesis_vector = hypothesis_linear_regression(coefficient_training_vector, data_matrix);
//
    //cout << "hypothesis vector: \n";
    //print_vector(hypothesis_vector);
//
    cout << "coeff vector lr: " << learning_rate << "\n";
    print_vector(coefficient_training_vector);
//
//    // item length (# of items)
    double m = data_matrix.size();
//    // param length (# of params)
    int n = coefficient_training_vector.size();
    int tolerance = .2;
    bool this_is_inf = false;
    bool keep_going = false;
    bool is_good = true;
//
//
    //double learning_rate = .0328;
    bool rebuild_for_lower_error = false;
    //cout << "n: " << n << "\n";
//
    bool is_right = true;
    for (int j=0; j < n; j++) {
        double sum_of_derivatives_for_each_param = 0;
        for (int i=0; i < m; i++) {
            double ho = floor((hypothesis_vector[i] * 100) + .5) / 100;
            if (isinf(ho)) {ho = 0;}
            double yo = floor((y_vector[i] * 100) + .5) / 100;
            if (isinf(yo)) {yo = 0;}
            double doj = floor((data_matrix[i][j] * 100) + .5) / 100;
            if (isinf(doj)) {doj = 0;}
            sum_of_derivatives_for_each_param += (ho - yo) * doj;
//            cout << "hyp vect: " << hypothesis_vector[i] << "\n";
//            cout << "y vect: " << y_vector[i] << "\n";
//            cout << "data m: " << data_matrix[i][j] << "\n";
            if (isinf(sum_of_derivatives_for_each_param)) {
                cout << "hyp vect exp: " << floor(hypothesis_vector[i]) << "\n";
                cout << "hyp vect: " << ho << " - ";
                cout << "y vect: " << yo << " * ";

                cout << "data m: " << doj << "\n";
                cout << "hyp: " << ho-yo << "\n";
                cout << "hyp: " << ho-yo << "\n";
                this_is_inf = true;
                cout << "learning rate: " << learning_rate << "\n";
                //return;
            }
            if (ho-yo > .15) {
                rebuild_for_lower_error = true;
            }
            if (ho-yo > .2) {
                is_right = false;
            }
        }

        double coefficient_training_vector_at_j;
        double coefficient_reset_value = coefficient_training_vector[j] - learning_rate * (1/m) * sum_of_derivatives_for_each_param;

        //cout << "coef training vector at j" << coefficient_training_vector[j] << "\n";
        //cout << "sum of deriv" << sum_of_derivatives_for_each_param << "\n";
       // cout << "reset val" << coefficient_reset_value << "\n";
        if (!(abs(coefficient_training_vector[j] - coefficient_reset_value) < tolerance)) {
            //cout << "test rebuild flag: " << rebuild_for_lower_error << "\n";
            rebuild_for_lower_error = true;
        }

        //bool test_inf
        if (isinf(sum_of_derivatives_for_each_param)) {
            cout << "is inf: " << coefficient_training_vector_at_j << "\n";
            return;
        }
        coefficient_training_vector_at_j = floor((coefficient_reset_value * 100) + .5) / 100;

        if ((coefficient_training_vector_at_j - coefficient_training_vector[j]) > 100) {
            is_good = false;
        }
        //coefficient_training_vector_at_j < coefficient_training_vector[j]
//        if (coefficient_reset_value < 0 && test_loops > 50 && learning_rate > .31) {
//            learning_rate = learning_rate - .01;
//        }
//        if (coefficient_training_vector_at_j < coefficient_training_vector[j] || coefficient_training_vector[j] < 0 && learning_rate > .035 && test_loops > 50) {
//            learning_rate = learning_rate - .001;
//        } else {
//            learning_rate = learning_rate + .001;
//        }

        coefficient_training_vector[j] = coefficient_training_vector_at_j;
    }

    //if (test_loops > 5000) {learning_rate = learning_rate + .0001;}

    if (rebuild_for_lower_error) {
        cout << "loop!: " << test_loops << "\n";
        test_loops ++;
        //learning_rate = learning_rate + .0001;
        if (is_right) {test_loops = 0;}
        if (test_loops > 25) {
            learning_rate = learning_rate + .00015;
            test_loops = 0;}
        if (!is_good && (learning_rate > .0001)) {learning_rate = learning_rate - .0001;}
        cout << "learning rate iiii: " << learning_rate << "\n";
        gradient_descent_min_cost(total_matrix, y_column, cost_vector, coefficient_training_vector, learning_rate);
    }

    // adjust learning rate programatically.
    // if # loops over 7819 and check margin of error...
        // re-run with learning loop +.001
    // if there's an inf - decrease

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