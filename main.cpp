#include <iostream>
#include <iomanip>
#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <cmath>

using namespace std;

// global variables
int test_loops = 0;

int total_running_loop = 0;

int total_loop = 0;

void my_main();

// general reference for variables
// m = number of training examples
// x's = input variables/ features
// y's = output variable/ target variable
// capital X = matrix.  (caps designate matrix)

// get data
void import_data();

// build structure
vector< vector<double> > build_matrix(string file_name);
vector<double> get_vector_slice(vector<vector <double> > array, int column);
vector< vector<double> > get_matrix_input_data(vector<vector <double> > array, int column);
void print_linear_eq(vector<double> coeff_vector);


// initialize supervised learning algorithm
vector<double> initialize_gradient_descent(int initial_values, int param_length);

// ml algorithms
vector<double>  hypothesis_linear_regression(vector<double> coefficient_training_vector, vector< vector<double> > data_matrix);
void gradient_descent_min_cost(vector< vector<double> > total_matrix, int y_column, vector<double> cost_vector, vector<double> hypothesis_vector, double learning_rate, vector< vector<double> >testing_matrix_data, vector<double>y_vector_testing, bool set_learning_rate_manually);
vector< vector<double> > split_testing_data_from_matrix(vector< vector<double> >& data_matrix);
double calculate_standard_error_of_estimate(vector<double> fresh_test_data_matrix_y, vector<vector<double> > data_matrix, vector<double> linear_coefficients);
string print_matlab_script_queues(vector<double> coefficient_training_vector);
// dev helpers
void print_matrix(vector< vector<double> > array);
void print_vector(vector<double> my_vector);


void my_main() {
    import_data();
}

int main() {
    my_main();
}

// get data
void import_data() {

    // column to extract y vector
    int y_column = 2;
    vector< vector<double> > data_matrix = build_matrix("bds_edited.csv");
    //vector< vector<double> > data_matrix = build_matrix("test6.csv");

    vector< vector<double> > testing_matrix_data = split_testing_data_from_matrix(data_matrix);
    vector<double> y_vector_testing = get_vector_slice(testing_matrix_data, y_column);


    vector<double> coefficient_training_vector = initialize_gradient_descent(100, data_matrix[0].size());
    vector<double> cost_vector = initialize_gradient_descent(100, data_matrix[0].size());
//    double learning_rate = .06;
//    bool set_learning_rate_manually = true;

    // for bds_edited.csv:
    double learning_rate = .3;

    // for test4.csv:
    // double learning_rate = .003;
    bool set_learning_rate_manually = false;
    gradient_descent_min_cost(data_matrix, y_column, cost_vector, coefficient_training_vector, learning_rate, testing_matrix_data, y_vector_testing, set_learning_rate_manually);

}

// build structure
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
    return test_set;
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
        double hyp = 0;
        for (int column=0; column < n; column++) {
            hyp += coefficient_training_vector[column] * data_matrix[row][column];
        }
        hypothesis.push_back(hyp);
    }

    return hypothesis;
}

void gradient_descent_min_cost(vector< vector<double> > total_matrix, int y_column, vector<double> cost_vector, vector<double> coefficient_training_vector, double learning_rate, vector< vector<double> > testing_matrix_data, vector<double>y_vector_testing, bool set_learning_rate_manually) {
    vector<double> y_vector = get_vector_slice(total_matrix, y_column);
    vector< vector<double> > data_matrix = get_matrix_input_data(total_matrix, y_column);
    vector<double> hypothesis_vector = hypothesis_linear_regression(coefficient_training_vector, data_matrix);

    print_vector(coefficient_training_vector);
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
                learning_rate = learning_rate + .00015;
                test_loops = 0;}
            if (!is_good && (learning_rate > .0001) ) {learning_rate = learning_rate - .0001;}
        }
        else {
            if (test_loops > 25 && is_good) {
                learning_rate = learning_rate + .0001;
                test_loops = 0;}
            else if (!is_good && (learning_rate > .0001)) {learning_rate = learning_rate - .0001;}
        }
        total_loop ++;
        cout << "total run loop" << total_running_loop << "\n";
        if (total_running_loop > 6800) {
            print_linear_eq(coefficient_training_vector);
            double error_calculated = calculate_standard_error_of_estimate(y_vector_testing, testing_matrix_data, coefficient_training_vector);
            cout << "standard error of estimate: " << error_calculated << "\n";
            //string matlab_script = print_matlab_script_queues(coefficient_training_vector);
            cout << "the matlab commands to depict this graphically are: \n";
            //cout << matlab_script <<"\n";
            return;
        }
        total_running_loop ++;
        gradient_descent_min_cost(total_matrix, y_column, cost_vector, coefficient_training_vector, learning_rate, testing_matrix_data, y_vector_testing, set_learning_rate_manually);

    }

}

string print_matlab_script_queues(vector<double> coefficient_training_vector) {
    string string_for_matlab_graph = "";
    return string_for_matlab_graph;
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

double calculate_standard_error_of_estimate(vector<double> fresh_test_data_matrix_y, vector<vector<double> > data_matrix, vector<double> linear_coefficients) {
    int m = data_matrix.size();
    int n = linear_coefficients.size();
    double standard_error_of_estimate = 0;
    for (int row=0; row < m; row++) {
        double linear_eq_y = 0;
        for (int column=0; column < n; column++) {

            if (column == 0) {
                linear_eq_y += (int)round(linear_coefficients[column]);
            } else {
                //cout << "coeff test " << linear_coefficients[column] * data_matrix[row][column-1] << "\n";
                linear_eq_y += (int)round(linear_coefficients[column]) * data_matrix[row][column-1];
            }
        }
        //cout << "error: " << pow((fresh_test_data_matrix_y[row] - linear_eq_y), 2) << "\n";
        standard_error_of_estimate += pow((fresh_test_data_matrix_y[row] - linear_eq_y), 2);
    }
    cout << "standard error: " << standard_error_of_estimate << "\n";
    return standard_error_of_estimate;
}

// functions for dev
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
