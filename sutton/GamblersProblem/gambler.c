
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_STATES 101
#define NUM_ACTIONS 101
// the probability of the coin coming up heads
#define P_H 0.4
// the task is an undiscounted episodic task
#define gamma 1

#define abs(x) ((x) < 0 ? (-x) : (x))

double values[NUM_STATES] = {0};
int policy[NUM_STATES] = {0};

void initPolicy();
double min(double, double);
double max(double, double);
double getQ(int, int);
void valueIteration();
void writeValuesAndPolicy(char *, char *);

void initPolicy(){
  for(int s = 0; s < NUM_STATES; s++){
    policy[s] = rand() % NUM_STATES;
  }
}

// min function, breaks ties randomly
double min(double x, double y){

  double minVal = 0;
  if(x < y){
    minVal = x;
  }else if(y < x){
    minVal = y;
  }else{
    int randVal = rand() % 2;
    if(randVal == 0)
      minVal = x;
    else
      minVal = y;
  }
  return minVal;
}

// max function, breaks ties randomly
double max(double x, double y){

  double maxVal = 0;
  if(x > y){
    maxVal = x;
  }else if(y > x){
    maxVal = y;
  }else{
    int randVal = rand() % 2;
    if(randVal == 0)
      maxVal = x;
    else
      maxVal = y;
  }
  return maxVal;
}

double getQ(int s, int a){

  double q = 0;
  
  int sprimeHeads = s + a;
  int sprimeTails = s - a;

  int rewardHeads = 0;
  int rewardTails = 0;

  if(sprimeHeads == 100){
    rewardHeads = 1;
  }

  if(sprimeTails == 100){
    rewardTails = 1;
  }
  
  q = (1-P_H) * (rewardTails + gamma * values[sprimeTails]) + (P_H) * (rewardHeads + gamma * values[sprimeHeads]);
  
  return q;
}

void valueIteration(){

  setbuf(stdout,NULL);

  double delta = 1000;
  double epsilon = 0.001;

  int iterationCounter = 1;

  // find the value for each state
  while(delta > epsilon){

    printf("Iteration %d delta %lf\n", iterationCounter, delta);

    delta = 0;

    for(int s = 0; s < NUM_STATES; s++){

      double temp = values[s];

      // find the action that maximizes the value of state s
      double maxActionValue = -1000;
      for(int a = 0; a <= (int)min(s, 100-s); a++){	
	double newValue = getQ(s,a);
	if(newValue > maxActionValue){
	  maxActionValue = newValue;
	}
      }
      
      values[s] = maxActionValue;

      delta = max(delta, abs(temp-values[s]));

      printf("State %d value: %lf\n", s, values[s]);
    }

    iterationCounter++;
  }

  // find the policy for each state from the values
  printf("Finding the policy...\n");
  for(int s = 0; s < NUM_STATES; s++){

    int maxAction = 0;
    double maxActionValue = -1000;

    for(int a = 0; a <= (int)min(s, 100 - s); a++){
      double newValue = getQ(s,a);
      if(newValue > maxActionValue){
	maxActionValue = newValue;
	maxAction = a;
      }
    }

    policy[s] = maxAction;
    printf("Policy for state %d: %d\n", s, maxAction);
  }

  // write values and policy
  writeValuesAndPolicy("./valuesFinal.csv","./policyFinal.csv");
}

void writeValuesAndPolicy(char * valuesFileName, char * policyFileName){

  FILE *fpv;
  FILE *fpp;

  fpv = fopen(valuesFileName, "w");
  fpp = fopen(policyFileName, "w");

  for(int s = 0; s < NUM_STATES; s++){
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

int main(){

  time_t t;
  srand((unsigned)time(&t));

  initPolicy();

  valueIteration();

  return 0;
}
