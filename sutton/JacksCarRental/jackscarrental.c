
#include <stdio.h>
#include <math.h>

float values[441] = 0;
int policy[441] = 5;

int getQ(int s, int a){

    
}

void evaluatePolicy(){

  float delta = 100;
  float theta = 0.001;

  while(delta > theta){

    delta = 0;

    for(int s = 0; s < 441; s++){

      float temp = values[s];
      values[s] = getQ(s,policy[s]);
      delta = fmaxf(delta, fabsf(temp-values[s]));
    }

    printf("Delta: %d", delta);
  }  
}

int improvePolicy(){

  policy_stable = 1;

  for(int s = 0; s < 441; s++){

    int temp = policy[s];
    int maxAction = 0;
    float maxActionValue = -1000;

    for(int a = 0; a < 11; a++){

      // get Q(s,a)
      float newValue = getQ(s,a);
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
      fprintf(fpp, "%f,", policy[s]);
    }
    else{
      fprintf(fpv, "%f", values[s]);
      fprintf(fpp, "%f", policy[s]);
    }
  }

  fclose(fpv);
  fclose(fpp)
}

int main(){

  int episodeCounter = 0;
  int policy_stable = 0;

  while(1){

    printf("Episode %d", episodeCounter);

    printf("Evaluating current policy..");
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
