#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>

void trainXOR(Brain *brain);
void testXOR(Brain *brain);

#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"


int main(int argc, char *argv[]) {
	
	//
	//
	//			 D
	//
	//			 D
	//	
	//			 D
	//		O
	//			 D
	//		O
	//			 D
	//		O		  X
	//			 D
	//		O		  X
	//			 D
	//	
	int depths[] = {2, 3, 5, 1};
	int numLayers = (int) (sizeof(depths) / sizeof(int));

	printf("[-] Creating net with %d layers...\n", numLayers);
	Brain *brain = brain_create(numLayers, depths, 0.01);

	printf("Training on the XOR function...\n");
	trainXOR(brain);

	printf("Testing on the XOR function...\n");
	testXOR(brain);

	printf("[+] Done.\n");
}


void trainXOR(Brain *brain) {
	float inputs[4][2] = {{1, 0}, {1, 1}, {0, 0}, {0, 1}};
	float outputs[4][1] = {{1}, {0}, {0}, {1}};

	for (int i = 0; i < 4*400; i++) {
		brain_train(brain, inputs[i%4], outputs[i%4]);
	}
}

char *itb(int value) {
	return value == 1 ? "true" : "false";
}

char *itbh(int value) {
	return value == 1 ? GRN "true" RESET : RED "false" RESET;
}

int roundValue(float value) {
	return value < 0.5f ? 0 : 1;
}

void testXOR(Brain *brain) {

	float inputs[4][2] = {{1, 0}, {1, 1}, {0, 0}, {0, 1}};
	float outputs[4][1] = {{1}, {0}, {0}, {1}};

	for (int i = 0; i < 4; i++) {
		
		Layer *output = brain_perform_inference(brain, inputs[i]);
		
		printf("Performing test #%d with input (%s, %s), expected %s\n", 
			i + 1,
			itb(inputs[i][0]), 
			itb(inputs[i][1]), 
			itb(outputs[i][0]));

		float *values = layer_get_values(output);
		float value = values[0];
		free(values);
		printf("Got value: %f", value);

		printf("Correct?: %s\n", itbh(roundValue(value) == outputs[i][0]));
	}
}


