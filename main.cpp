#include <iostream>
#include <iomanip>
#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

int rowA = 0;
int colA = 0;

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
vector<double>  hypothesis_linear_regression(vector<double> param_vector, vector< vector<double> > data_matrix);
void gradient_descent(int arg[]);
void gradient_descent_min_cost(vector< vector<double> > total_matrix, int y_column, vector<double> cost_vector);

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
    int y_column = 1;
    vector< vector<double> > simple_matrix = build_matrix("temp_x_rental_num_y_simple.csv");
    vector<double> cost_matrix = initialize_gradient_descent(0, simple_matrix.size());
    gradient_descent_min_cost(simple_matrix, y_column, cost_matrix);

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
        array[i].erase(array[i].begin() + 1);
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
    //vector<double> pos = array;
    //vector< vector<double> > *pointer = &array;
    //double some_num = pointer*[0][1];
    //cout << "print val: " <<  some_num << "\n";
    for (int i=0; i<array.size(); i++) {
        //cout << "i: " << i << "\n";
        result.push_back(array[i][column]);
        array[i].erase(array[i].begin() + 1);
        //cout << "print val: " <<  array[i][1] << "\n";
    };

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
        cout << my_vector[i] << " \n";
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
    vector<double> initial_cost_matrix;
    for (int i=0; i<param_length; i++) {
        initial_cost_matrix.push_back(initial_values);
        //initial_cost_matrix[i] = initial_values;
    }

    //print_vector(initial_cost_matrix);
    return initial_cost_matrix;
}

// algorithms
vector<double>  hypothesis_linear_regression(vector<double> param_vector, vector< vector<double> > data_matrix) {
    // return hypothesis based on linear regression formula
    // hO(x) = O0 + O1x;
    //return O0 + O1 * x;

    vector<double> hypothesis;

    // for each row in data vector... use eq.
    // data times param. at i.

    int m = data_matrix.size();
    int c_s = 0;
    if (data_matrix.size()!=0) {
        c_s = data_matrix[0].size();
    }

    //cout << " row length: " << m << "\n";
    //cout << " parameter length: " << c_s << "\n";

    double hyp = 0;
    for (int row=0; row < m; row++) {
         //first row.
        for (int column=0; column < c_s; column++) {
            cout << "input param" << row << " params: " << param_vector[row] << " ";
            cout << "each in each row:" << row << " data for params: " << data_matrix[row][column] << "\n";
            hyp += param_vector[row] * data_matrix[row][column];
             //do stuff ...
        }
        hypothesis.push_back(hyp);
    }

    //given a set of data with an identity row at position 0 for the linear eq.
    //calculates a hypothesis vector for each row based on multiplying params times data for each column.
    // vector has a value per row.  calculations are per row.

    print_vector(hypothesis);

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

void gradient_descent_min_cost(vector< vector<double> > total_matrix, int y_column, vector<double> cost_vector) {
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

    //toDo: add in next;
    vector<double> new_cost_vector;
    vector<double> hypothesis_vector = hypothesis_linear_regression(cost_vector, data_matrix);

    int m = data_matrix.size();
    int c_s = data_matrix[0].size();

    double sum = 0;


    for (int row=0; row < m; row++) {

        double xi = cost_vector[row];
        if (row==0) {
            xi = 1;
        }
        // need two loops
        //new_cost_vector[row]  += (hypothesis_vector[row] - y_vector[row]) * xi;


        //first row.
        //sum += (hypothesis_vector[row] - y_vector[row]);
//            for (int column = 0; column < c_s; column++) {
//                //new_cost_vector.push_back()
//                //do stuff ...
//            }
    }

    for (int row=0; row < m; row++) {
        double regression_eq_result;
        if (row == 0) {
            regression_eq_result = (1/m) * sum;
        } else {
            // data_matrix[row]
            // xi - not sure what this is for each row but it pertains to the row.
            // i think it's the prev cost for the row.
            //regression_eq_result = (1/m) * sum * xi;
        }

        //new_cost_vector.push_back()
    }



//    for (int row=0; row< data_matrix.size(); row++) {
//        // first row.
//
//        for (int column=0; row< data_matrix[0].size(); column++) {
//            cout << "each in each row:" << data_matrix[row][column] << "\n";
//            //cout << "colum stuff: " << col
//            // do stuff ...
//        }
//    }

//    for (int i=0; i<=data_matrix.size(); i++) {
//        sum += (hypothesis_vector[i] - y_vector[i]);
//        for (int j=0; i<=data_matrix[i].size(); i++) {
//            new_cost_vector[j] = (1/m) * sum;
//        }
//        sum = 0;
//    }
//


    // if each cost isn't at 0....
    // call recursively
    //gradient_descent_min_cost(total_matrix, y_column, new_cost_vector);

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