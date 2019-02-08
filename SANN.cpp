/*

                Modification of the Standard MLP code started in
		June 2016 to use Simulated Annealing to explore
	           		the cost surface.
                    
                        by Sarah Aguasvivas-Manzano


                     Backprop started in June 2016
                     Modification with SA April 2017

The Algorithm:

-Start with an initial guess for the weights and initial temperature
-Calculate the cost function of the random guess through forward pass. 
-Generate a random neighboring solution.
-Calculate new solution cost
 - if Cnew<Cold, accept that solution and move
 - else if exp((Cold-Cnew)/T > rand(0,1)) , accept the new solution
 - otherwise, use previous solution until minimum temperature is reached.

*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <string>
#include <ctime>
#include <ctype.h>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>

using namespace std;

int numNN = 0;
const double a = 0.9;      //optimum momentum 1.7159
const double Tol = 2.0e-1;       //Tolerance  9e-2 for MNIST
/*
//Parameters for SA:
double T= 50000.0;       //Temperature in SA algorithm
double tempReduceRate= 0.95; //rate of reduction of temperature
int tempReducePeriod= 500; //annealing schedule
double Tmin= 5e-5;           //minimum temperature to be used
double alpha= 5e-3;       //perturbation rate
*/

long double T= 30000.0;       //Temperature in SA algorithm
long double tempReduceRate= 0.95; //rate of reduction of temperature
int tempReducePeriod= 25; //annealing schedule
double Tmin= 5e-100;           //minimum temperature to be used
long double alpha= 5e-3;       //perturbation rate


double Sigmoid(double);
double SigmoidPrime(double);
void readData(double, int &, int &);
int inputHidden();
void inputStuff(int, int *);

class NeuralNet {

public:
	int  numH, layers, *numNeurons;
	void initNetwork(int);
	double error;
	double *des, *inp;
        
	NeuralNet(){
		numNN++;
	//	numNeurons = new int [layers + 2];
	}
};

class Layer : public NeuralNet {

public:

	void operator+(Layer a);
	void operator=(Layer a);
	int n, m;
long 	double *in, *out;
long 	double **w ,b,  **w_temp;
	void initWeights(int, int);
      long  double *delta;

	Layer(int n1, int m1){
    layers++;

		//cout << "Layer created... " << endl;
		n = n1;
		m = m1;
			in = new long double[m];
			out = new long double[m];
            delta= new long double [m];
        
			//Initializing input/output:
		for (int i = 0; i < m; i++) {
			in[i] = 0.0;
			out[i] = 0.0;
		}
        if (n!= 0 && m!=0) {
            
            w= new long double *[n];
            for(int i=0; i<n; i++) w[i]= new long double [m];
            
            w_temp= new long double *[n];
            for(int i=0; i<n; i++) w_temp[i]= new long double [m];
            
            initWeights(n, m);
        }
    }

	~Layer(){
    }

};

void Layer::operator+(Layer b){
	
	/* Weighted sum to pass information between current layer and layer b.
	This will be used later in the FwdPass subroutine. */

	double *v = new double [b.m];

	for (int i = 0; i<b.m; i++) v[i] = 0.0;
    

    for (int j = 0; j<b.m; j++){
        for (int i = 0; i<b.n; i++){
            v[j] += b.w[i][j] * this->out[i];
        }
        b.in[j] = v[j] + b.b;
        b.out[j] = Sigmoid(b.in[j]);
        
    }
    delete [] v;
}

void Layer::operator=(Layer b) {
    
	/*This was created in case I needed to copy an element. */

	b.n = this->n;
	b.m = this->m;
	b.numNeurons = this->numNeurons;

	for (int i = 0; i < this->n; i++) {
		for (int j = 0; j < this->m; j++) {
			b.w[i][j] = this->w[i][j];
		}
		b.b = this->b;
		b.in[i] = this->in[i];
	}

	for (int i = 0; i > this->m; i++) b.out[i] = this->out[i];

}

void NeuralNet::initNetwork(int H) {
	numH = H;          //number of hidden layers
	layers = numH + 2; //adding input and output layers
}

void Layer::initWeights(int n, int m){

    
	double n0 = n;
	double n1 = m;
	double u = 1.0 / n0, v,gamma, sum = 0.0;

	gamma = 0.7*pow(n1, u);
    
	for (int i = 0; i<n; i++){
		for (int j = 0; j<m; j++){
         
			v =0.1*(1.0*((double)rand()) / ((double)RAND_MAX) - 0.5);
			w[i][j] = v;
			sum += pow(w[i][j], 2);
			w[i][j] = gamma*w[i][j] / (pow(sum, 0.5));
		}
	}
                        b = 0.1*(1.0*((double)rand())/((double)RAND_MAX) -0.5);

}

double Sigmoid(double x){
	return a*tanh(x);
//      return 1.0/(1.0+exp(-a*x));
}

double SigmoidPrime(double x){
//	return 1-pow(tanh(x),2);
      return a*Sigmoid(x)*(1.0-Sigmoid(x));
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}


void RowCol(int & rowA, int & colA, ifstream & fileIN, string & datafile){

	string lineA;
	double x;
	bool fai = 1 ;

	do {
		//Entering the data file name:
		cout << "\nIf you want to use Iris type 'Iris.txt'" << endl;
		cout << "Enter data file name i.e. filename.txt: " << endl;
        
		cin >> datafile;

		fileIN.open(datafile);

		if (fileIN.fail()){
			cout << "**********************************" << endl;
			cerr << "\nError at opening file :( ";
			fai = 1;
		}
		else fai = 0;

	} while (fai==1);

	//Reading data:
    
    rowA= 0;
    colA= 0;
    
	while (fileIN.good()){
		while (getline(fileIN, lineA)){
			istringstream streamA(lineA);
			colA = 0;
			while (streamA >> x){
				colA++;
			}
			rowA++;
		}
	}
}

int inputHidden(){
	int an;
	bool bFail = true;
	do{
		cout << "Enter amount of hidden layers: ";
		cin >> an;
		bFail = cin.fail();
		cin.clear();
		cin.ignore(numeric_limits<streamsize>::max(), '\n');

	} while (an <= 0 || bFail == true);
	return an;
}

void inputStuff(int layers, int *array){

	bool bFail = true;

	for (int i = 0; i<layers; i++){
		if (i == 0){

			bFail = true;
			do{
				cout << "Enter # of neurons in input layer: ";
				cin >> array[0];

				bFail = cin.fail();
				cin.clear();
				cin.ignore(numeric_limits<streamsize>::max(), '\n');

			} while (array[0] <= 0 || bFail == true);
		}
		else if (i != layers - 1){

			bFail = true;

			do{
				cout << "Enter # of neurons in hidden layer #" << i << ": ";
				cin >> array[i];

				bFail = cin.fail();
				cin.clear();
				cin.ignore(numeric_limits<streamsize>::max(), '\n');

			} while (array[i] <= 0 || bFail == true);
		}
		else {
			bFail = true;
			do{
				cout << "Enter amount of neurons in output layer: ";
				cin >> array[layers - 1];

				bFail = cin.fail();
				cin.clear();
				cin.ignore(numeric_limits<streamsize>::max(), '\n');

			} while (array[layers - 1] <= 0 || bFail == true);
		}
	}
}

void initLayers(NeuralNet & NN, int *x, vector < Layer > &list, vector < Layer > &oldWeights){
	
    /*This will store the layers with their corresponding numbers of
	neurons inside the vector called 'list'. It is important to know that in
	this context, m is the amounts of neurons before the layer and m is the amount
	of neurons in the layer. This convention will ease the computations. 
	*/
	NN.numNeurons = new int [NN.layers + 2];
    
	NN.numNeurons[0] = 0;
		for (int i = 0; i < NN.layers+1; i++){
			NN.numNeurons[i+1] = x[i];
		}
		NN.numNeurons[NN.layers+1] = 0;

		for (int ii = 0; ii < NN.layers; ii++){
			Layer lay(NN.numNeurons[ii], NN.numNeurons[ii+1]);
                        oldWeights.push_back(lay);
			list.push_back(lay);
		}
}

int getLargest(vector <Layer > &list){
    
   /* Getting largest amount of neurons (m) through all the network for
    further use. */
    
    int larg1 = 0;
    
    for (int i=0; i< list.size(); i++){
        if(list[i].m > larg1) larg1 = list[i].m;
    }
    
    return larg1;
}

void FwdPass(vector < Layer > &list) {

    /* This part of the code performs a weighted sum of the values in
     the layers of the NN in a forward way */
    
    for (int i = 0; i < ((int)list.size()-1); i++) {
        list[i] + list[i + 1];
    }

}

 void rejectWeights (vector <Layer> &list, vector <Layer> &oldWeights){
    
for (int iii= 1; iii< (int)list.size(); iii++){
        for (int i = 0; i < list[iii].n; i++) {
            for (int j = 0; j < list[iii].m; j++) {
                list[iii].w[i][j]= oldWeights[iii].w[i][j];
            }
        }
             list[iii].b= oldWeights[iii].b;
    }
}

void updateWeights(vector < Layer > &list,  vector < Layer > &oldWeights, ofstream & weights, bool tf) {
    
    for (int iii= 1; iii< (int)list.size(); iii++){
        for (int i = 0; i < list[iii].n; i++) {
            for (int j = 0; j < list[iii].m; j++) {
        
        oldWeights[iii].w[i][j]= list[iii].w[i][j];
        oldWeights[iii].b= list[iii].b;
        
        list[iii].w[i][j]+= alpha*((2.0*(long double)rand()/((long double)RAND_MAX)-1.0)); 

        if(tf== 1) weights<< list[iii].w[i][j] << "   ";
            }        
}
list[iii].b+=  alpha*((2.0*(long double)rand()/((long double)RAND_MAX)-1.0)); 
//    if(tf==1) weights<< list[iii].b<< "  ";
    }
if(tf==1) weights<< endl;
}

void calcError(vector <Layer> &list, int larg, NeuralNet & NN, double *err, int randRow, int rowA){
     double sum = 0.0; 
    double *sum_temp = new double [list[list.size()-1].m];
    for (int i=0; i<list[list.size()-1].m; i++) sum_temp[i]=0.0;

    for (int i = 0; i < list[list.size()-1].m; i++){
        sum_temp[i] += pow((NN.des[i] - abs(list[list.size() - 1].out[i])), 2);
    }
    
   err[randRow]=0.0;

    for (int i=0; i<list[list.size()-1].m ; i++) err[randRow] +=  sum_temp[i] / 2.0;
    
    sum=0.0;    
    
    for (int i = 0; i < rowA; i++){
        sum += err[i];
    }
    NN.error= sum/(double)rowA;
}

void simAnnealing(NeuralNet & NN, int rowA, double ** training, vector < Layer > &list, int larg, bool tf, int a, int colA, vector < Layer> &oldWeights) {
    
    double *err = new double[rowA];
    for (int i = 0; i < rowA; i++) err[i] = 1000.0;
    ofstream weights;
    ofstream error;
    int randRow;
    double *input= new double [a];
    double *desired= new double [colA-a];
    double CostOld=1e8, CostNew=1e8; 
    error.open("error.txt");
    weights.open("weights.txt");
    int epoch=0, protoEpoch=0;
    NN.error= 70.0;
    double prob;
    NN.inp = new double[list[0].m];
    
    NN.des = new double[list[list.size() - 1].m];

    while(CostNew>Tol) { //  && T>Tmin ){   //start of the Simulated Annealing
    epoch++;
    CostOld= CostNew;
    updateWeights(list, oldWeights,weights, tf);

    for (int iii=0; iii< rowA; iii++){ //cost function
        randRow = iii;                        //before it was random, now it is not
        
        for (int i=0; i < a; i++){
            input[i]= training[randRow][i];
        }
        
        for (int j = a, k = 0; j <colA && k < (colA - a); j++, k++) {
            desired[k] = training[randRow][j];
        }
        
        for (int i = 0; i < list[0].m; i++) {
            list[0].out[i] = input[i];
            NN.inp[i] = list[0].out[i];
        }
        
        for (int i = 0; i < list[list.size()-1].m; i++) {
            NN.des[i] = desired[i];
        }
 
        FwdPass(list);
        calcError(list, larg, NN, err, randRow, rowA); 

        } // end of the cost function 
        CostNew= NN.error;
        
        tf=0; 
        
        prob= exp((CostOld-CostNew)/(T));
        
        //cout<< "      "<< prob<<endl;
//cout<< CostNew<< " "<< CostOld<<endl;        
      if(CostNew<CostOld){;
      }  
       else if(prob>((double)rand()/(double)RAND_MAX)){
	}
	else{
rejectWeights(list,oldWeights);
}

      
/*        if(CostNew>=CostOld){
         rejectWeights(list, oldWeights);
        // CostNew= CostOld;
        }
        else if(prob<((double)rand()/(double)RAND_MAX)){
        rejectWeights(list, oldWeights);
        //CostNew= CostOld;
        }
*/
         if (CostOld != CostNew) {
            cout<< CostNew<< endl;
         tf=1; 
         }
         error<< CostNew<< endl;
         if(epoch%tempReducePeriod==0) T= T*tempReduceRate;
     

} //end of the Simulated Annealing while

cout<< "Trained in " << epoch<< " iterations"<<endl;
     weights.close();
     error.close();
delete [] err;
delete [] input;
delete [] desired; 
}

double testingCV(vector <Layer> & list, ifstream & validation, int instances_val, int colA, int a){
    
    double err_val, sum_temp=0.0, *err;
    err= new double [instances_val];
    double *val= new double [a];
    double *label_val= new double [colA-a];
    string line;
    double x;
    
    
    for (int i=0; i < instances_val; i++){
        getline(validation, line);
        
        istringstream streamA(line);
        
        for (int u=0; u<a; u++){
            streamA>>x;
            val[u]=x;
        }
        for (int u=0; u<(colA-a); u++){
            streamA>>x;
            label_val[u]= x;
        }
        
        for (int j=0; j<list[0].m; j++){
            list[0].out[j]= val[j];
        }
       
        FwdPass(list);
      
        for (int j=0; j<list[list.size()-1].m; j++){
            sum_temp+= pow((label_val[j]-list[list.size()-1].out[j]),2);
        }
        
        err[i]= sum_temp/2.0;
    }
    
    sum_temp=0.0;
    
    for (int i=0; i<instances_val; i++) sum_temp+= err[i];
    
    err_val= sum_temp/(double)instances_val;
    
    delete [] err;
    delete [] val;
    delete [] label_val;
    return err_val;
}


void divideTrainTest(int instances_tr, int instances_te, int instances_val, ofstream & training, ofstream & testing, ofstream & validation, int rowA, int colA, int a, ifstream & fileIN){
    
    int randRow;
    string line;
    int *noRepeat= new int [rowA];
    int *tag= new int [rowA];
    bool rep= true;
    int ro;

    //  Training:
    
    for (int i=0; i<instances_tr; i++){
        rep=true;
        randRow= rand()%rowA;
        
        if(i>0){
            while(rep==true){
                for(int iii=0; iii<i; iii++){
                    if(noRepeat[iii]==randRow){
                        rep= true;
                        randRow= rand()%rowA;
                        break;
                    }
                    else rep= false;
                }
            }
        }
        
        noRepeat[i]= randRow;
        tag[randRow]= 1;
        
    }
    
    
    //  Testing:
    
    for (int i=0; i<instances_te; i++){
        rep=true;
        randRow= rand()%rowA;
        
        while(rep==true){
            for(int iii=0; iii<i+instances_tr; iii++){
                if(noRepeat[iii]==randRow){
                    rep= true;
                    randRow= rand()%rowA;
                    break;
                }
                else rep= false;
            }
        }
        
        noRepeat[i+instances_tr]= randRow;
        tag[randRow]= 2;
        
    }
    
    
    // Validation:
    
    for (int i=0; i<instances_val; i++){
        rep=true;
        randRow= rand()%rowA;
        
        while(rep==true){
            for(int iii=0; iii<i+instances_tr+instances_te; iii++){
                if(noRepeat[iii]==randRow){
                    rep= true;
                    randRow= rand()%rowA;
                    break;
                }
                else rep= false;
            }
        }
        
        noRepeat[i+instances_tr+instances_te]= randRow;
        tag[randRow]= 3;
        
    }
    
    //File Split:
    
    fileIN.clear();
    fileIN.seekg(0);
    
    ro=0;
    while(ro<rowA){
        getline(fileIN, line);
        istringstream streamA(line);
        
        
        if (tag[ro]==1)
        {
                training<< line<< endl;
        }
        if (tag[ro]==2)
        {
                testing << line<< endl;
        }
        if (tag[ro]==3)
        {
                validation<< line<< endl;
        }
        ro++;
    }
    
    cout<< "\n... new training/testing/validation files stored, you can re-use those files in the future...\n "<< endl;
    
    delete [] noRepeat;
    delete [] tag;
}

void createPointsTest(int rows, int colA, ifstream & testing, vector <Layer> & list, int a, double fwd){
    
    ofstream testPlot;
    string line;
    double *test=new double [a];
    double *label_te= new double [colA-a];
    double x;
    
    testPlot.open("TestPlot.txt");
    
  
    for (int i=0; i<rows; i++){
            getline(testing, line);
            
            istringstream streamA(line);
            
            for (int u=0; u<a; u++){
                streamA>>x;
                test[u]= x;
            }

            for (int u=0; u<(colA-a); u++){
                streamA>>x;
                label_te[u]= x;
            }
   
                for (int j=0; j<a; j++){
                    list[0].out[j]= test[j];
                }
                
                fwd= get_cpu_time();
                FwdPass(list);
                fwd= get_cpu_time()-fwd;
                
                for (int ii=0; ii<list[list.size()-1].m; ii++){
                    testPlot<< abs(list[list.size()-1].out[ii])<< "  ";
                }
                for (int ii=0; ii<list[list.size()-1].m; ii++){
                    testPlot<< label_te[ii]<<"  ";
                }
                
                testPlot<< endl;
    }
    
    testPlot.close();
    
    delete [] test;
    delete [] label_te;
}

void assignTraining(int instances_tr, int colA, int a, ifstream & training1, double **training){
    
    string line;
    double x;
    training1.open("Training.txt");
    training1.clear();
    training1.seekg(0);
    
    for (int ro=0;getline(training1, line);ro++){
            istringstream streamA(line);
        for (int co=0; co< colA; co++){
            streamA>>x;
            training[ro][co]=x;
        }

    }
    
    training1.close();
      
}

void DoYouWish(int instances_tr, int instances_te, int instances_val, int rowA, int colA, int a, ifstream & fileIN){
    
        ofstream training1, validation, testing;
        char yn;
    cout<< "\n If you are running a case for the first time say 'no' to the next question \n"<<endl;
    cout<< "\nDo you wish to re-use previously stored training/testing/validation data (y/n)?: ";
    cin>> yn;
    
    if (yn== 'n'){
        
        training1.open("Training.txt");
        testing.open("Testing.txt");
        validation.open("Validation.txt");
        
        divideTrainTest(instances_tr, instances_te,  instances_val, training1,  testing, validation,  rowA,  colA, a, fileIN);
        
        training1.close();
        testing.close();
        validation.close();
    } else{
        cout<<"\n...Using recycled data..."<<endl;
    }
    
}

    
int main(){
    
    
    NeuralNet NN;
    
    vector<Layer> list;                                     //"array" of layer objects
    vector<Layer> oldWeights;
   
    int an, *array, rowA=1, colA=1, larg;
    int instances_tr, instances_val, instances_te;
    double tra, tess, fwd, err_val;
    
    ofstream trainTest;
    ifstream fileIN;
    ifstream tr, te, val;
    string datafile;
    double **training;
    
    bool tf=true;
    
    srand(static_cast <unsigned int> (time(NULL)*1000));
    
    trainTest.open("trainTest.txt");

    an = inputHidden();
    NN.initNetwork(an);
    array = new int[NN.layers];
    
    inputStuff(NN.layers, array);
    RowCol(rowA, colA, fileIN, datafile);
    
    fileIN.clear();
    fileIN.seekg(0);

    instances_tr= (int)(0.7*rowA);
    instances_val= (int)(0.0*rowA);
    instances_te= (int)(0.3*rowA);
    
    training= new double * [instances_tr];
     for (int i=0; i< instances_tr; i++) training[i]= new double [colA];

    initLayers(NN, array, list, oldWeights);
    
    larg= getLargest(list);
    
    for (int i=0; i<NN.layers; i++){
        trainTest<< "Layer " << i << "'s # of neurons is"<<array[i]<< endl;
    }
    trainTest<< endl;
    
    // Cross-Validation Subroutine:
    
    DoYouWish(instances_tr, instances_te, instances_val, rowA, colA, array[0], fileIN);
    
    tr.open("Training.txt");
    te.open("Testing.txt");
    val.open("Validation.txt");
    
    assignTraining(instances_tr, colA, array[0], tr, training);
    

        //Training:
    
    for (int iii=0; iii<1; iii++){

        tra= get_cpu_time();
        
        simAnnealing(NN, instances_tr, training, list, larg, tf, array[0], colA, oldWeights);
        
        tra= get_cpu_time()-tra;
        
        
        //Validation:
        err_val= testingCV(list, val, instances_val, colA, array[0]);
        
        cout<< iii<< "\t" << setprecision(7)<< NN.error<< "\t" << err_val<< endl;
        
        trainTest<<iii<< "\t" <<setprecision(7)<<  NN.error << "\t" << err_val<< endl;
        
        tf=false;
        
    }
    
    
    tess= get_cpu_time(); // this is to test a group of data, not just forward passing
      createPointsTest(instances_te, colA, te, list, array[0], fwd);
    tess= get_cpu_time()-tess;
    
    cout<< "CPU time for training set: "<< (double)tra<< " s"<<endl;
    cout<< "CPU time for testing set: "<< (double)tess<< " s"<< endl;
    cout<< "CPU time for forward pass on a single data point: "<< fwd<< " s"<< endl;
    
    tr.close();
    te.close();
    val.close();
    trainTest.close();
    fileIN.close();
    
    return 0;
    
}

