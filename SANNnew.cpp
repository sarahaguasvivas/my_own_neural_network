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
 -Calculate 
 - if Cnew<Cold, accept that solution and move
 - else if exp((Cold-Cnew)/T > rand(0,1)) , accept the new solution
 - otherwise, use previous solution until minimum temperature is reached.
 
 */

#include <iostream>
#include <iomanip>
#include <array>
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
#include <math.h>

using namespace std;

int numNN = 0;
const double a = 1.7159;      //optimum momentum 1.7159
const double Tol = 1.9e-1;      
int numCrossVal= 1;

double  T=1.0;
double tempReduceRate= 0.95;
int   tempReducePeriod= 50;
double alpha=8e-1;
double  Tmin= 1e-300;


double Sigmoid(double);
void readData(double, int &, int &);
int inputHidden();
void inputStuff(int, int *);

class NeuralNet {
    //This contains general information about the Neural Network.
    
public:
    int  numH, layers, *numNeurons;
    void initNetwork(int);
    double error;
    double *des , *inp;
    
    //constructor
    NeuralNet(){
        numNN++;
       // numNeurons = new int [layers + 2];
    }
};

class Layer : public NeuralNet {
    
    /*The operational part of my code takes place in the layer class.
     There is no neuron class because it is more practical to store information
     as an array of neurons than neurons per se.*/
    
public:
    
    void operator+(Layer a);
    int n, m;
    double *in, *out;
    double **w , b;
    void initWeights(int, int);
    double *delta;
    
    Layer(int n1, int m1){
        
        layers++;
        //cout << "Layer created... " << endl;
        n = n1;
        m = m1;
        in = new double[m];
        out = new double[m];
        delta= new double [m];
        
        //Initializing input/output:
        for (int i = 0; i < m; i++) {
            in[i] = 0.0;
            out[i] = 0.0;
        }
        if (n!= 0 && m!=0) {
            
            w= new double *[n];
            for(int i=0; i<n; i++) w[i]= new double [m];
            
            initWeights(n, m);
        }
    }
    
    ~Layer(){
    }
    
};

void Layer::operator+(Layer b){
    
    /* Weighted sum to pass information between current layer and layer b.
     This will be used later in the FwdPass subroutine. */
    
    double *v = new double[b.m];
    
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

void NeuralNet::initNetwork(int H) {
    numH = H;          //number of hidden layers
    layers = numH + 2; //adding input and output layers
}

void Layer::initWeights(int n, int m){
    
    double n0 = n;
    double n1 = m;
    double u = 1.0 / n0, gamma, v, sum = 0.0;
    
    gamma = 0.7*pow(n1, u);
    
    for (int i = 0; i<n; i++){
        for (int j = 0; j<m; j++){
            
            v = 0.1*(1.0*((double)rand()) / ((double)RAND_MAX) - 0.5);
            w[i][j] = v;
            
        }
    }
    b = 0.1*(1.0*((double)rand()/((double)RAND_MAX))- 0.5)+0.1;
}

double Sigmoid(double x){
   return 1.0 /(1.0 + exp(-a*x));
//   return a*tanh(x);
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

void readData(double **arrayA, int rowA, int colA, ifstream & fileIN, string datafile){
    
    fileIN.open(datafile);
    string lineA;
    double x;
    bool fai = 1 ;
    int rowA1=0, colA1=0;
    
    while (fileIN.good()){
        while (getline(fileIN, lineA)){
            istringstream streamA(lineA);
            colA1 = 0;
            while (streamA >> x){
                arrayA[rowA1][colA1] = x;
                colA1++;
            }
            rowA1++;
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

void initLayers(NeuralNet & NN, int *xx, vector < Layer > &list, vector <Layer > &candidate, vector <Layer> &initial){
    
     /*
     This will store the layers with their corresponding numbers of
     neurons inside the vector called 'list'. It is important to know that in
     this context, m is the amounts of neurons before the layer and m is the amount
     of neurons in the layer. This convention will ease the computations.
     */
    
    NN.numNeurons = new int [NN.layers + 2];
    
    int *arrayy= new int [NN.layers];
    for (int i=0; i<NN.layers; i++) arrayy[i]= xx[i];
    
    NN.numNeurons[0] = 0;
    for (int i = 0; i <(int)(NN.layers); i++){

        NN.numNeurons[i+1] = xx[i];
       
    }
    
    NN.numNeurons[NN.layers+1] = 0;

    for (int ii = 0; ii <(int)NN.layers; ii++){
        Layer lay(NN.numNeurons[ii], NN.numNeurons[ii+1]);
        list.push_back(lay);
        candidate.push_back(lay);
        initial.push_back(lay);
    }
    for (int i=0;i<NN.layers;i++) {
        xx[i]= arrayy[i];
    }
}

int getLargest(vector <Layer > &list){
    
    /* Getting largest number of neurons (m) through all the network for
     further use. */
    
    int larg1 = 0;
    
    for (int i=0; i< (int)list.size(); i++){
        if(list[i].m > larg1) larg1 = list[i].m;
    }
    
    return larg1;
}

void FwdPass(vector < Layer > list) {
    
    /* This part of the code performs a weighted sum of the values in
     the layers of the NN in a forward way */
   
    for (int i = 0; i < (int)(list.size()-1); i++) {
        list[i] + list[i + 1];
    }
}


void acceptCandidate(vector <Layer> &list, vector <Layer> &candidate){

    for (int iii= 1; iii< (int)list.size(); iii++){
     
        for (int i = 0; i < (int)list[iii].n; i++) {
       
            for (int j = 0; j < (int)list[iii].m; j++) {
                list[iii].w[i][j]= candidate[iii].w[i][j];
            }
        }
       list[iii].b= candidate[iii].b;
    }

}
void getInitial(vector <Layer> &list, vector <Layer> &initial){

for (int iii= 1; iii< (int)list.size(); iii++){
         for (int i = 0; i < (int)list[iii].n; i++) {  
           for (int j = 0; j < (int)list[iii].m; j++) {
    
        initial[iii].w[i][j]= list[iii].w[i][j];
           }
         } 
       initial[iii].b= list[iii].b;
      }  

}

void newCandidateWeights(vector < Layer > &list,vector <Layer> &candidate, ofstream & weights, bool tf) {

//default_random_engine generator;
//cauchy_distribution <double> distribution(5.0, 1.0);

   
int N, count;
double *Y, *U;
double sumsum= 0.0;

for (int iii= 1; iii< (int)(list.size()); iii++){
        for (int i = 0; i < (int)list[iii].n; i++) {
          for (int j = 0; j < (int)list[iii].m; j++) {
          N++;
          }
        }
N++; //for bias
}

 Y= new double [N];
 U= new double [N];
  
    for (int i=0;i<N;i++){
     Y[i]= 2.0*(double)rand()/(double)RAND_MAX-1.0;
     // sumsum+= pow(Y[i],2.0);
    }
    for (int i=0; i<N; i++){
      U[i]= Y[i];///pow(sumsum, 0.5);
      }
    count=0;

    for (int iii= 1; iii< (int)(list.size()); iii++){ 
        for (int i = 0; i < (int)list[iii].n; i++) { 
          for (int j = 0; j < (int)list[iii].m; j++) {
      candidate[iii].w[i][j]= list[iii].w[i][j] +  alpha*U[count];
                
                if(tf == 1) weights<< list[iii].w[i][j] << "   ";
               count++; 
          }
        }
       count++;
      candidate[iii].b = list[iii].b + alpha*U[count];
        
      if(tf==1) weights<< list[iii].b<< endl;
    }
delete [] Y;
delete [] U;
}


double testingCV(vector <Layer> & list, double **testing, double **label_te, int instances_te, int colA, int a){
    double err_test, sum_temp=0.0, *err1;
    err1= new double [instances_te];
    
    for (int i=0; i <instances_te; i++){
        
        for (int j=0; j<(int)list[0].m; j++){
            list[0].out[j]= testing[i][j];
        }
        
        FwdPass(list);
        sum_temp=0.0;
        for (int j=0; j<(int)list[list.size()-1].m; j++){
            sum_temp+= pow((label_te[i][j]-list[list.size()-1].out[j]),2);
        }
        
        err1[i]= sum_temp/2.0;
    }
    
    sum_temp=0.0;
    
    for (int i=0; i<instances_te; i++) sum_temp+= err1[i];
    
    err_test= sum_temp/(double)instances_te;
    delete [] err1;
    return err_test;
}

double Cost(NeuralNet & NN,vector <Layer> & candidate, int iii,int rowA, int colA, int a, double **training, double **label_tr, double *err) {
   
    double sum_temp;
    double *input= new double [a];
    double *desired= new double [colA-a];
  
  for (iii=0; iii< rowA; iii++){ //cost function
        
        for (int i=0; i < (candidate[0].m); i++){
            input[i]= training[iii][i];   

        }
        for (int i=0; i<(candidate[candidate.size()-1].m); i++) {
            desired[i] = label_tr[iii][i];

        }
   
        for (int i = 0; i < (candidate[0].m); i++) {
            candidate[0].out[i] = input[i];
        }
        for (int i = 0; i <(candidate[candidate.size()-1].m); i++) {
            NN.des[i] = desired[i];
        }
        FwdPass(candidate);


        sum_temp=0.0;
        for (int i = 0; i < (candidate[candidate.size()-1].m); i++){
            sum_temp += pow((NN.des[i] - candidate[candidate.size() - 1].out[i]),2);
        }
        
        err[iii]= sum_temp;
    } // end of the cost function
    
    sum_temp=0.0;
    for (int i = 0; i < rowA; i++){
        sum_temp += err[i];
    }
    
    NN.error= (sum_temp/(double)rowA);

    return NN.error;
    
    delete [] input;
    delete [] desired;
}

void simAnnealing(NeuralNet & NN, int rowA, double ** training, double **label_tr, vector < Layer > &list, bool tf, int a, int colA, vector < Layer> & candidate, int & maxEpoch) {
    long double prob;
    ofstream weights;
    ofstream error;
   double *err= new double [rowA];
    double CostNew=70, CostOld=70;
    error.open("error.txt");
    weights.open("weights.txt");
    int epoch=0;
    NN.inp = new double[list[0].m];
    NN.des = new double[list[list.size() - 1].m];
    double tempe=T;
    for (int i=0; i<rowA; i++) err[i]= 1000.0;
    int jjj;
    double myRand;

while(CostNew>Tol  && tempe>Tmin ){   //start of the Simulated Annealing
//  while(epoch<Iter){     
   epoch++;
  jjj = rand() % rowA;

   CostOld= CostNew;     
        newCandidateWeights(list,candidate,weights,tf);
          CostNew= Cost(NN,candidate,jjj, rowA, colA, a, training, label_tr, err);
       
          prob= exp((CostOld-CostNew)/(tempe));
          myRand= (double)rand()/(double)RAND_MAX;
 
        if(CostNew<CostOld){
            acceptCandidate(list, candidate);
            
       } else if (prob>myRand){
            acceptCandidate(list, candidate);
          }  else{
        CostNew=CostOld;
        NN.error= CostOld;
          }
      if(CostNew!= CostOld)
 cout<<setprecision(8)<< CostNew<< endl;
    
      if(CostNew!=CostOld) error<<setprecision(8)<< CostNew<< endl;
      if(epoch%tempReducePeriod==0) {
tempe= tempe*tempReduceRate;
}

if(epoch%50000) tf=1; 
else tf=0;
       
    } //end of the Simulated Annealing while
  
    maxEpoch= epoch;
    cout<< "Trained in " << epoch<< " iterations"<<endl;
    weights.close();
    error.close();
    delete [] err;
}

void divideTrainTest(int instances_tr, int instances_te, int instances_val, double **training, double **testing, double **validation,  double **label_tr, double **label_te, double **label_val, int rowA, int colA, double **data, int a){
    
    int randRow;
    double **tr, **te, **val;
    tr= new double *[instances_tr];
    te= new double *[instances_te];
    val= new double *[instances_val];
    
    for(int i=0; i<instances_tr; i++) tr[i]= new double [colA];
    for(int i=0; i<instances_te; i++) te[i]= new double [colA];
    for(int i=0; i<instances_val; i++) val[i]= new double [colA];
    
    int *noRepeat= new int [rowA];
    bool rep= true;
    
    //cout<< "Tr: "<<endl;
    for (int i=0; i<instances_tr; i++){
        
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
        
        // cout<< randRow<< setw(6);
        
        for( int j=0; j<colA; j++){
            tr[i][j]= data[randRow][j];
            //  cout<< tr[i][j]<< setw(6);
        }
        //cout<< endl;
    }
    // cout<< endl<< "Testing Set: "<<endl;
    
    for (int i=0; i<instances_te; i++){
        
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
        noRepeat[i+instances_tr]= randRow;
        
        
        // cout<< randRow<< setw(6);
        for (int j=0; j<colA; j++){
            
            te[i][j]= data[randRow][j];
            //       cout<<te[i][j]<< setw(6);
        }
        //   cout<< endl;
    }
    // cout<< endl<< "Validation set: "<< endl;
    
    for (int i=0; i<instances_val; i++){
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
        noRepeat[i+instances_tr+instances_te]= randRow;
        
        for (int j=0; j<colA; j++){
            val[i][j]= data[randRow][j];
        }
    }
    
    //  cout<< "Training: "<<endl;
    for (int i=0; i<instances_tr; i++){
        for(int j=0; j< a; j++){
            training[i][j]= tr[i][j];
            //      cout<< training[i][j]<< " ";
        }
        //   cout<< endl;
    }
    // cout<< "\n\n Testing:"<<endl;
    
    for (int i=0; i<instances_te; i++){
        for(int j=0; j< a; j++){
            testing[i][j]= te[i][j];
      //             cout<< testing[i][j]<<"  ";
        }
        //    cout<< endl;
    }
    
    for (int i=0; i<instances_val; i++){
        for(int j=0; j< a; j++){
            validation[i][j]= val[i][j];
        }
    }
    // cout<< "Label Tr: "<<endl;
    for (int i = 0; i < instances_tr; i++) {
        for (int j = a, k = 0; j <colA && k < (colA - a); j++, k++) {
           label_tr[i][k] = tr[i][j];
           //         cout<< label_tr[i][j]<< " ";
        }
        //   cout<<"\n\n"<< endl;
    }
    // cout<< "\n\n"<<endl;
    
    for (int i = 0; i < instances_te; i++) {
        for (int j = a, k = 0; j <colA && k < (colA - a); j++, k++) {
            label_te[i][k] = te[i][j];
        //        cout<< label_te[i][k]<< "  ";
        }
        //   cout<<endl;
    }
  //   cout<< "\n\n"<<endl;
    
    for (int i = 0; i < instances_val; i++) {
        for (int j = a, k = 0; j <colA && k < (colA - a); j++, k++) {
            label_val[i][k] = val[i][j];
        }
    }
    
    delete [] noRepeat;
    delete [] tr;
    delete [] te;
    delete [] val;
    
}

void createPointsTest(NeuralNet & NN, int rows, double **testing, double **label_te, vector <Layer> & list, int a, double fwd, int colA){
    
    ofstream testPlot;
    
    testPlot.open("TestPlot.txt");
   
    for (int i=0; i < rows; i++){

        for (int j=0; j<a; j++){
            list[0].out[j]= testing[i][j];
        NN.inp[j]= list[0].out[j];
        }
     
        fwd= get_cpu_time();
        FwdPass(list);
        fwd= get_cpu_time()-fwd;
        
        for (int ii=0; ii<list[list.size()-1].m; ii++){
            testPlot<< list[list.size()-1].out[ii]<< "  ";
        }
        for (int ii=0; ii<list[list.size()-1].m; ii++){
            testPlot<< label_te[i][ii]<<"  ";
        }

        testPlot<< endl;
    }
    testPlot.close();
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
    fileIN.close();
}

int main(){
    
    NeuralNet NN;
    int *array , an, rowA =0, colA = 0;
    double **data;
    
    string datafile;
    ifstream fileIN;
    
    vector< Layer > list;                       //"array" of layer objects
    vector< Layer > candidate;
    vector <Layer > initial;
    int larg;
    double **training, **validation, **testing;
    int instances_tr, instances_val, instances_te;
    double err_val;
    ofstream trainTest;
    double **label_tr, **label_te, **label_val;
    bool tf=true;
    double tra, tess, fwd;
    ofstream cpuTimes1;
    int maxEpoch;
    ofstream settings; 
    srand(static_cast <unsigned int> (time(NULL)*100));
    trainTest.open("trainTest.txt");
    cpuTimes1.open("cpuTimes.txt");
    settings.open("settings.txt");
    an = inputHidden();
    NN.initNetwork(an);
    array = new int [NN.layers];
  // int iii=0; 
    inputStuff(NN.layers, array);
    
    RowCol(rowA, colA, fileIN, datafile);
    
    data= new double *[rowA];
    for (int i=0; i<rowA; i++) data[i]= new double [colA];
    
    readData(data, rowA, colA, fileIN, datafile);
    
    // Splitting data into training-testing-validation (in this case with a 40/30/30 scheme).
    
    instances_tr= (int)(0.7*rowA);
    instances_val= (int)(0.0*rowA);
    instances_te= (int)(0.3*rowA);
    
    training = new double *[instances_tr];
    testing= new double   *[instances_te];
    validation= new double*[instances_val];
    
    label_tr= new double *[instances_tr];
    label_te= new double *[instances_te];
    label_val= new double*[instances_val];
    
    for (int j=0; j<instances_tr; j++) training[j]= new double [colA-array[NN.layers-1]];
    for (int j=0; j<instances_te; j++) testing[j]=  new double [colA-array[NN.layers-1]];
    for (int j=0; j<instances_val; j++)validation[j]=new double [colA-array[NN.layers-1]];
    
    for (int i=0; i<instances_tr; i++) label_tr[i]= new double [colA-array[0]];
    for (int i=0; i<instances_te; i++) label_te[i]= new double [colA-array[0]];
    for (int i=0; i<instances_val; i++) label_val[i]= new double [colA-array[0]];
    
    initLayers(NN, array, list, candidate, initial);
    larg= getLargest(list);
   getInitial(list,initial); 
    
 cpuTimes1 << "Run # \t CPU time training \t Epochs Needed"<<endl;
  
 divideTrainTest(instances_tr, instances_te, instances_val, training, testing, validation, label_tr, label_te, label_val, rowA, colA, data, array[0]);

    // Cross-Validation Subroutine:
    for (int iii=0; iii<numCrossVal; iii++){
   // int iii=0;
//    for (T=5.0; T<5000000; T+=50000){
//        for (tempReduceRate=0.5; tempReduceRate<=0.85; tempReduceRate*=17){
//            for (tempReducePeriod=5; tempReducePeriod<=125; tempReducePeriod*=25){
//               for (alpha=0.01; alpha<1; alpha+= 0.03 ){
     iii++;    
     acceptCandidate(list, initial);
    
        settings<< iii << "   T= "<< T << "; tempReduceRate= "<< tempReduceRate<< "; tempReducePeriod= "<<tempReducePeriod<< "; alpha="<< alpha<<endl;
      
        tra= get_cpu_time();
        //Training:
        simAnnealing(NN, instances_tr, training, label_tr, list, tf, array[0], colA, candidate, maxEpoch);
        tra= get_cpu_time()-tra;
        
        //Testing:
        err_val= testingCV(list, validation, label_val, instances_val, colA, array[0]);
        
        cout<< iii<< "\t" << setprecision(7)<< NN.error << "\t" << err_val<< endl;
        trainTest<<iii<< "\t" <<setprecision(7)<<  NN.error << "\t" << err_val<< endl;
        
        cpuTimes1 <<iii<< "\t"<< setprecision(7)<<tra<<"\t"<< maxEpoch<<endl;
        
        tf=false;
        
               }
          //  }
  //      }
   // }
/*for (int iii=0; iii<list.size(); iii++){
   for (int i=0; i< list[iii].n ; i++){
     for (int j=0; j<list[iii].m; j++){
cout<< list[iii].w[i][j]<< "  ";

     }
   }
cout<< endl;
}
cout<< endl;
*/
    tess= get_cpu_time(); // this is to test a group of data
    createPointsTest(NN, instances_te, testing, label_te, list, array[0], fwd, colA);
    tess= get_cpu_time()-tess;
    
    cout<< "CPU time for training set: "<< (double)tra<<endl;
    cout<< "CPU time for testing set: "<< (double)tess<< endl;
    cout<< "CPU time for forward pass on a single data point: "<< fwd<<endl;
    
    
    trainTest.close();
    cpuTimes1.close();
    settings.close();
delete [] data;
delete [] array;
delete [] training;
delete [] testing;
delete [] validation;
delete [] label_tr;
delete [] label_te;
delete [] label_val;   

 return 0;
    
}
