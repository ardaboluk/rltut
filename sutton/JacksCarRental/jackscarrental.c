
#include <stdio.h>
#include <math.h>

double values[441] = {0};
int policy[441] = {5};

// cache the powers and factorials
double powers3[21] = {0};
double powers2[21] = {0};
double powers4[21] = {0};
int factorials[21] = {0};
double expo3 = 0;
double expo2 = 0;
double expo4 = 0;

int fact(int num){

  int result = 1;
  
  for(int i = 2; i <= num; i++){
    result *= i;
  }

  return result;
}

void initCaches(){

  expo3 = expf(-3.0);
  expo2 = expf(-2.0);
  expo4 = expf(-4.0);

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
  int numLoc1 = (int)floor(s/21.0);
  int numLoc2 = s % 21;

  for(int sprime = 0; sprime < 441; sprime++){

    // decode the number of cars at each location from sprime
    int numLoc1prime = (int)floor(sprime/21.0);
    int numLoc2prime = sprime % 21;

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
	    int newValue1 = fmax(0, fmin(numLoc1 + ret1, 20));
	    int newValue2 = fmax(0, fmin(numLoc2 + ret2, 20));

	    int minCars = 0;
	    if(a < 5){
	      minCars = fmin(numLoc2 - fmax(0, numLoc2 + (a-5)), fmin(newValue1 - (a-5), 20) - newValue1);
	      newValue1 += minCars;
	      newValue2 -= minCars;	      
	    }else if(a > 5){
	      minCars = fmin(numLoc1 - fmax(0, numLoc1 - (a-5)), fmin(newValue2 + (a-5), 20) - newValue2);
	      newValue1 -= minCars;
	      newValue2 += minCars;	      
	    }

	    int newValue1AfterReq = fmax(0, newValue1 - req1);
	    int newValue2AfterReq = fmax(0, newValue2 - req2);

	    if(newValue1AfterReq == numLoc1prime && newValue2AfterReq == numLoc2prime){

	      int reward = ((newValue1 - newValue1AfterReq) + (newValue2 - newValue2AfterReq)) * 10 - minCars * 2;
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
  double theta = 0.01;

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

    policy[s] = maxAction;

    if(temp != policy[s]){
      policy_stable = 0;
    }
  }

  return policy_stable;
}

void writeValuesAndPolicy(char * valuesFileName, char * policyFileName){

  FILE *fpv;
  FILE *fpp;

  fpv = fopen(valuesFileName, "w");
  fpp = fopen(policyFileName, "w");

  for(int s = 0; s < 441; s++){
    if(s < 440){
      fprintf(fpv, "%f,", values[s]);
      fprintf(fpp, "%d,", policy[s]);
    }
    else{
      fprintf(fpv, "%f", values[s]);
      fprintf(fpp, "%d", policy[s]);
    }
  }

  fclose(fpv);
  fclose(fpp);
}

int main(){

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

    char * valuesFileName;
    char * policyFileName;
    sprintf(valuesFileName, "./values_ep%d.csv", episodeCounter);
    sprintf(policyFileName, "./policy_ep%d.csv", episodeCounter);
    writeValuesAndPolicy(valuesFileName, policyFileName);

    episodeCounter += 1;
  }
}
