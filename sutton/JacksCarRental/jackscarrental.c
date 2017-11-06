
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// function prototypes
double fact(double);
void initCaches();
double getQ(int, int);
void evaluatePolicy();
int improvePolicy();
void policyIteration();
void valueIteration();
void writeValuesAndPolicy(char *, char *);

// global arrays for values and policy
double values[441] = {0};
int policy[441] = {5};

// cache the powers and factorials
double powers3[21] = {0};
double powers2[21] = {0};
double powers4[21] = {0};
double factorials[21] = {0};
double expo3 = 0;
double expo2 = 0;
double expo4 = 0;

double fact(double num){

  double result = 1;
  
  for(int i = 2; i <= num; i++){
    result *= i;
  }

  return result;
}

void initCaches(){

  expo3 = exp(-3.0);
  expo2 = exp(-2.0);
  expo4 = exp(-4.0);

  for(int i = 0; i < 21; i++){

    powers3[i] = pow(3.0,i);
    powers2[i] = pow(2.0,i);
    powers4[i] = pow(4.0,i);
    factorials[i] = fact(i);
  }
}

double getQ(int s, int a){

  double q = 0;
  
  // decode the number of cars at each location from s
  double numLoc1 = floor(s/21.0);
  double numLoc2 = s % 21;

  for(int sprime = 0; sprime < 441; sprime++){

    // decode the number of cars at each location from sprime
    double numLoc1prime = floor(sprime/21.0);
    double numLoc2prime = sprime % 21;

    double sprimeProb = 0;

    for(int ret1 = 0; ret1 <= 20; ret1++){
      for(int req1 = 0; req1 <= 20; req1++){
	for(int ret2 = 0; ret2 <= 20; ret2++){
	  for(int req2 = 0; req2 <= 20; req2++){
	    
	    /*double prob = ((powf(3.0,ret1)/fact(ret1)) * expf(-3.0)) * ((powf(3.0,req1)/fact(req1)) * expf(-3.0)) *
	      ((powf(2.0,ret2)/fact(ret2)) * expf(-2.0)) * ((powf(4.0,req2)/fact(req2)) * expf(-4.0));*/

	    double prob = ((powers3[ret1]/factorials[ret1]) * expo3) * ((powers3[req1]/factorials[req1]) * expo3) *
	      ((powers2[ret2]/factorials[ret2]) * expo2) * ((powers4[req2]/factorials[req2]) * expo4);

	    // clip the new numbers of the two locations between 0 and 20
	    double newValue1 = fmin(numLoc1 + ret1, 20);
	    double newValue2 = fmin(numLoc2 + ret2, 20);

	    double minCars = 0;
	    if(a < 5){
	      minCars = fmin(numLoc2 - fmax(0, numLoc2 + (a-5)), fmin(newValue1 - (a-5), 20) - newValue1);
	      newValue1 += minCars;
	      newValue2 -= minCars;	      
	    }else if(a > 5){
	      minCars = fmin(numLoc1 - fmax(0, numLoc1 - (a-5)), fmin(newValue2 + (a-5), 20) - newValue2);
	      newValue1 -= minCars;
	      newValue2 += minCars;	      
	    }

	    double newValue1AfterReq = fmax(0, newValue1 - req1);
	    double newValue2AfterReq = fmax(0, newValue2 - req2);

	    if((int)(newValue1AfterReq) == (int)(numLoc1prime) && (int)(newValue2AfterReq) == (int)(numLoc2prime)){

	      double reward = ((newValue1 - newValue1AfterReq) + (newValue2 - newValue2AfterReq)) * 10 - minCars * 2;
	      q += prob * reward;
	      sprimeProb += prob;
	    }    
	  }	  
	}	
      }
    }

    q += sprimeProb * 0.9 * values[sprime];
  }

  return q;
}

void evaluatePolicy(){

  double delta = 100;
  double theta = 1; // was 0.01

  while(delta > theta){

    delta = 0;

    for(int s = 0; s < 441; s++){
      
      double temp = values[s];
      values[s] = getQ(s,policy[s]);
      printf("Eval state %d : %lf\n", s, values[s]);
      delta = fmaxf(delta, fabsf(temp-values[s]));
    }

    printf("Delta: %lf\n", delta);
  }  
}

int improvePolicy(){

  int policy_stable = 1;

  for(int s = 0; s < 441; s++){

    int temp = policy[s];
    int maxAction = 0;
    double maxActionValue = -1000;

    for(int a = 0; a < 11; a++){

      // get Q(s,a)
      double newValue = getQ(s,a);
      if(newValue > maxActionValue){
	maxActionValue = newValue;
	maxAction = a;
      }
    }

    printf("Improved state %d: %d\n", s, maxAction);

    policy[s] = maxAction;

    if(temp != policy[s]){
      policy_stable = 0;
    }
  }

  return policy_stable;
}

void policyIteration(){

  setbuf(stdout,NULL);
  
  int episodeCounter = 1;
  int policy_stable = 0;

  initCaches();

  while(1){

    printf("Episode %d\n", episodeCounter);

    printf("Evaluating current policy..\n");
    evaluatePolicy();

    printf("Improving current policy..\n");
    policy_stable = improvePolicy();

    if(policy_stable == 1){
      writeValuesAndPolicy("./valuesFinal.csv","./policyFinal.csv");
    }

    char * valuesFileName = (char *)malloc(20);
    char * policyFileName = (char *)malloc(20);
    sprintf(valuesFileName, "./values_ep%d.csv", episodeCounter);
    sprintf(policyFileName, "./policy_ep%d.csv", episodeCounter);
    writeValuesAndPolicy(valuesFileName, policyFileName);

    episodeCounter += 1;
  }
}

void valueIteration(){

  setbuf(stdout,NULL);
  
  double delta = 1000;
  double epsilon = 1;

  initCaches();

  int iterationCounter = 1;

  // find the value for each state
  printf("Value iteration...\n");
  while(delta > epsilon){

    printf("Iteration %d delta %lf\n", iterationCounter, delta);

    delta = 0;

    for(int s = 0; s < 441; s++){

      double temp = values[s];

      // find the action that maximizes the value of state s
      double maxActionValue = -1000;
      for(int a = 0; a < 11; a++){	
	double newValue = getQ(s,a);
	if(newValue > maxActionValue){
	  maxActionValue = newValue;
	}
      }
      
      values[s] = maxActionValue;

      delta = fmaxf(delta, fabsf(temp-values[s]));

      printf("State %d value: %lf\n", s, values[s]);
    }

    iterationCounter++;
  }

  // find the policy for each state from the values
  printf("Finding the policy...\n");
  for(int s = 0; s < 441; s++){

    int maxAction = 0;
    double maxActionValue = -1000;

    for(int a = 0; a < 11; a++){
      double newValue = getQ(s,a);
      if(newValue > maxActionValue){
	maxActionValue = newValue;
	maxAction = a;
      }
    }

    policy[s] = maxAction;
  }

  // write values and policy
  writeValuesAndPolicy("./valuesFinal.csv","./policyFinal.csv");
}

void writeValuesAndPolicy(char * valuesFileName, char * policyFileName){

  FILE *fpv;
  FILE *fpp;

  fpv = fopen(valuesFileName, "w");
  fpp = fopen(policyFileName, "w");

  for(int s = 0; s < 441; s++){
    if(s < 440){
      fprintf(fpv, "%lf,", values[s]);
      fprintf(fpp, "%d,", policy[s]);
    }
    else{
      fprintf(fpv, "%lf", values[s]);
      fprintf(fpp, "%d", policy[s]);
    }
  }

  fclose(fpv);
  fclose(fpp);
}

double main(){

  //policyIteration();
  valueIteration();
  
}
