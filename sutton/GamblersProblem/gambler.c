
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_STATES 100
// the probability of the coin coming up heads
#define P_H 0.4
// the task is an undiscounted episodic task
#define gamma 1

double values[NUM_STATES] = {0};
int policy[NUM_STATES] = {0};

// helper array for choosing max action
double actionValues[50] = {0};

void initPolicy();
double getQ(int, int);
void valueIteration();
int chooseMaxAction(int);
void writeValuesAndPolicy(char *, char *);

void initPolicy(){
  for(int s = 1; s < NUM_STATES; s++){
    policy[s] = rand() % NUM_STATES;
  }
}

double getQ(int s, int a){

  double q = 0;
  
  int sprimeHeads = s + a;
  int sprimeTails = s - a;

  int rewardHeads = 0;

  if(sprimeHeads == 100){
    rewardHeads = 1;
  }

  q = (1-P_H) * (gamma * values[sprimeTails]) + (P_H) * (rewardHeads + gamma * values[sprimeHeads]);
  
  return q;
}

void valueIteration(){

  setbuf(stdout,NULL);

  double delta = 1000;
  double epsilon = 0.000;

  int iterationCounter = 1;

  // find the value for each state
  while(delta > epsilon){

    printf("Iteration %d delta %g\n", iterationCounter, delta);

    delta = 0;

    for(int s = 1; s < NUM_STATES; s++){

      double temp = values[s];

      // find the maximum action value
      double maxActionValue = -1000;
      for(int a = 0; a <= (int)fmin(s, 100-s); a++){	
	double newValue = getQ(s,a);
	if(newValue > maxActionValue){
	  maxActionValue = newValue;
	}
      }
      
      values[s] = maxActionValue;

      delta = fmax(delta, fabs(temp-values[s]));
    }

    iterationCounter++;
  }

  // find the policy for each state from the values
  printf("Finding the policy...\n");
  for(int s = 1; s < NUM_STATES; s++){

    policy[s] = chooseMaxAction(s);
    printf("Policy for state %d: %d\n", s, policy[s]);
  }

  // write values and policy
  writeValuesAndPolicy("./valuesFinal.csv","./policyFinal.csv");
}

int chooseMaxAction(int s){

  int maxAction = 0;

  int numActions = (int)fmin(s,100-s);
  double maxActionValue = -1000;
  // number of actions that has the maximum value
  int numMaxActions = 0;
  
  for(int a = 0; a <= numActions; a++){
    actionValues[a] = getQ(s,a);
    if(actionValues[a] > maxActionValue){
      maxActionValue = actionValues[a];
      numMaxActions = 1;
    }else if(actionValues[a] == maxActionValue){
      numMaxActions++;
    }
  }

  // randomly choose an action between the max actions
  int randMaxActionNum = rand() % numMaxActions;

  int maxActionCounter = 0;
  for(int a = 0; a <= numActions; a++){
    if(actionValues[a] == maxActionValue){
      if(maxActionCounter == randMaxActionNum){
	maxAction = a;
	break;
      }else{
	maxActionCounter++;
      }
    }
  }

  return maxAction;  
}

void writeValuesAndPolicy(char * valuesFileName, char * policyFileName){

  FILE *fpv;
  FILE *fpp;

  fpv = fopen(valuesFileName, "w");
  fpp = fopen(policyFileName, "w");

  for(int s = 1; s < NUM_STATES; s++){
      fprintf(fpv, "%lf,", values[s]);
      fprintf(fpp, "%d,", policy[s]);    
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
