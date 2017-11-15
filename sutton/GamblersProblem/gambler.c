
#include <stdio.h>

#define NUM_STATES 101

// the probability of the coin coming up heads
const double P_H = 0.4;

double values[NUM_STATES] = {0};
int policy[NUM_STATES] = {0};

double getQ(int, int);
void valueIteration();

double getQ(int s, int a){
  
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
      for(int a = 0; a < s; a++){	
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
  for(int s = 0; s < NUM_STATES; s++){

    int maxAction = 0;
    double maxActionValue = -1000;

    for(int a = 0; a < s; a++){
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

int main(){

  return 0;
}
