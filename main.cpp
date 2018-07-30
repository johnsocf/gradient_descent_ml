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

int total_running_loop = 0;

int total_loop = 0;

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
void print_linear_eq(vector<double> coeff_vector);
double calculate_standard_error_of_estimate(vector<double> fresh_test_data_matrix_y, vector<vector<double> > data_matrix, vector<double> linear_coefficients);


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
void gradient_descent_min_cost(vector< vector<double> > total_matrix, int y_column, vector<double> cost_vector, vector<double> hypothesis_vector, double learning_rate, vector< vector<double> >testing_matrix_data, vector<double>y_vector_testing, bool set_learning_rate_manually);
vector< vector<double> > split_testing_data_from_matrix(vector< vector<double> >& data_matrix);

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

    runThisFunction(something);
    int this_array[5] = {3, 4, 5, 6, 7};
    int m = training_example_length(this_array);

//    for (int i=0; i <=10; i++) {
//        for (int j=0; j<= 10; j++) {
//            for (int k=0; k<= 10; k++) {
//                cout << i << "," << j << "," << k << "," << (2 + 3*i + 4*j + 5*k) << "\n";
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
    //vector< vector<double> > data_matrix = build_matrix("bds.csv");
    vector< vector<double> > data_matrix = build_matrix("test4.csv");

    vector< vector<double> > testing_matrix_data = split_testing_data_from_matrix(data_matrix);
    vector<double> y_vector_testing = get_vector_slice(testing_matrix_data, y_column);


    vector<double> coefficient_training_vector = initialize_gradient_descent(100, data_matrix[0].size());
    vector<double> cost_vector = initialize_gradient_descent(100, data_matrix[0].size());
    double learning_rate = .004;
    bool set_learning_rate_manuall = false;
    gradient_descent_min_cost(data_matrix, y_column, cost_vector, coefficient_training_vector, learning_rate, testing_matrix_data, y_vector_testing, set_learning_rate_manuall);

    // import data via csv.
    // build matrix
    // build test sets
}

vector< vector<double> > split_testing_data_from_matrix(vector< vector<double> >& data_matrix) {
    int m = data_matrix.size();
    int n = data_matrix[0].size();
    int set_deliminator = round(m * .75);
    vector<double> v;
    vector< vector <double> > test_set;
    vector< vector <double> > minimized_data_set;
    for (int i=set_deliminator; i<m; i++) {
        for (int column=0; column<n; column++) {
            int row = i - set_deliminator;
        }
        test_set.push_back(data_matrix[i]);
    };
    for (int i=0; i<set_deliminator; i++) {
        for (int column=0; column<n; column++) {
            int row = i - set_deliminator;
        }
        minimized_data_set.push_back(data_matrix[i]);
    };

    data_matrix = minimized_data_set;
    cout << "print data matrix: \n";
    print_matrix(data_matrix);
    cout << "print training matrix: \n";
    print_matrix(test_set);
    return test_set;
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
        cout << "please wait for linear equation for gradient descent based on input data \n";
    }
    string line, val;
    vector<double> v;
    vector<vector <double> > array;


    while (getline(inFile, line)) {
        vector<double> v;
        stringstream s (line);
        while (getline (s, val, ','))
            v.push_back(stod(val));
        array.push_back(v);
    }

    return array;
}

vector< vector<double> > get_matrix_input_data(vector<vector <double> > array, int column) {

    // first iterate through rows.
    for (int i=0; i<array.size(); i++) {
        //cout << "i: " << i << "\n";
        vector<double>::iterator it;
        it = array[i].begin();
        // add identity row to first


        // remove output row.
        array[i].erase(array[i].begin() + 2);
        array[i].insert(it, 1);
    };

    // add identity


    return array;
}

vector<double> get_vector_slice(vector<vector <double> > array, int column) {
    vector<double> result;
    // reference vector by pointer
    for (int i=0; i<array.size(); i++) {
        result.push_back(array[i][column]);
        array[i].erase(array[i].begin() + 2);

    };
    return result;
}

double calculate_standard_error_of_estimate(vector<double> fresh_test_data_matrix_y, vector<vector<double> > data_matrix, vector<double> linear_coefficients) {
    int m = data_matrix.size();
    int n = linear_coefficients.size();
    string equation = "";
    int standard_error_of_estimate = 0;
    for (int row=0; row < m; row++) {
        double linear_eq_y = 0;
        for (int column=0; column < n; column++) {
            //cout << "linear eq: " << linear_coefficients[column] * data_matrix[row][column] << "\n";
            //cout << "coeff: " << linear_coefficients[column] << "\n";

            if (column == 0) {
                //cout << "coeff: " << linear_coefficients[column] << "\n";
                linear_eq_y += linear_coefficients[column];
            } else {
                //cout << "data: " << data_matrix[row][column-1] << "\n";
//                cout << "coeff: " << linear_coefficients[column] << "\n";
//                cout << "============================== 'n";
                linear_eq_y += linear_coefficients[column] * data_matrix[row][column-1];
            }
        }
        cout << "error: " << pow((fresh_test_data_matrix_y[row] - linear_eq_y), 2) << "\n";
        standard_error_of_estimate += pow((fresh_test_data_matrix_y[row] - linear_eq_y), 2);
    }
    return standard_error_of_estimate;
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
    vector<double> initial_cost_matrix;
    for (int i=0; i<param_length; i++) {
        initial_cost_matrix.push_back(initial_values);
    }
    return initial_cost_matrix;
}

// algorithms
vector<double>  hypothesis_linear_regression(vector<double> coefficient_training_vector, vector< vector<double> > data_matrix) {
    // return hypothesis based on linear regression formula
    vector<double> hypothesis;

    int m = data_matrix.size();
    int n = data_matrix[0].size();


    for (int row=0; row < m; row++) {
         //first row.
        double hyp = 0;
        for (int column=0; column < n; column++) {
            hyp += coefficient_training_vector[column] * data_matrix[row][column];
        }
        hypothesis.push_back(hyp);
    }

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

void gradient_descent_min_cost(vector< vector<double> > total_matrix, int y_column, vector<double> cost_vector, vector<double> coefficient_training_vector, double learning_rate, vector< vector<double> > testing_matrix_data, vector<double>y_vector_testing, bool set_learning_rate_manually) {
    vector<double> y_vector = get_vector_slice(total_matrix, y_column);
    vector< vector<double> > data_matrix = get_matrix_input_data(total_matrix, y_column);
    vector<double> hypothesis_vector = hypothesis_linear_regression(coefficient_training_vector, data_matrix);
    double m = data_matrix.size();
    int n = coefficient_training_vector.size();
    int tolerance = .3;
    bool this_is_inf = false;
    bool keep_going = false;
    bool is_good = true;
    bool rebuild_for_lower_error = false;
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
            if (isinf(sum_of_derivatives_for_each_param)) {
                ///...
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
        //cout << "coef: " << coefficient_training_vector[j] << " ";
        if (!(abs(coefficient_training_vector[j] - coefficient_reset_value) < tolerance)) {
            rebuild_for_lower_error = true;
        }

        if (isinf(sum_of_derivatives_for_each_param)) {
            cout << "is inf: " << coefficient_training_vector_at_j << "\n";
            return;
        }
        coefficient_training_vector_at_j = floor((coefficient_reset_value * 100) + .5) / 100;

        if ((coefficient_training_vector_at_j - coefficient_training_vector[j]) > 100) {
            is_good = false;
        }
        coefficient_training_vector[j] = coefficient_training_vector_at_j;
    }
    //cout << "\n";


    if (rebuild_for_lower_error) {
        test_loops ++;
        if (!set_learning_rate_manually) {
            if (is_right) {test_loops = 0;}
            if (test_loops > 25) {
                learning_rate = learning_rate + .00005;
                test_loops = 0;}
            if (!is_good && (learning_rate > .0001) ) {learning_rate = learning_rate - .0001;}
        } else {
            if (test_loops > 25 && is_good) {
                learning_rate = learning_rate + .0001;
                test_loops = 0;}
            else if (!is_good && (learning_rate > .0001)) {learning_rate = learning_rate - .0001;}
        }
        total_loop ++;
        if (total_running_loop > 6000) {
            print_linear_eq(coefficient_training_vector);
            double error_calculated = calculate_standard_error_of_estimate(y_vector_testing, testing_matrix_data, coefficient_training_vector);
            cout << "standard error of estimate: " << error_calculated << "\n";
            return;
        }
        total_running_loop ++;
        gradient_descent_min_cost(total_matrix, y_column, cost_vector, coefficient_training_vector, learning_rate, testing_matrix_data, y_vector_testing, set_learning_rate_manually);

    }

}

void print_linear_eq(vector<double> coeff_vector) {
    int n = coeff_vector.size();
    string alphabet_vars[] = {"y", "x", "z", "a", "b", "c"};
    string equation = "";
    for (int row=0; row < n; row++) {

        switch(row) {
            case 0:
                equation += alphabet_vars[row] + " = " + to_string((int)round(coeff_vector[row]));
                break;
            default:
                equation += " + " + to_string((int)round(coeff_vector[row])) + "" + alphabet_vars[row];
        }

    }
    cout << "the machine learned equation is: \n" << equation << ".  happy linear predicting!\n";
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