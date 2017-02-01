#include "neuron.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

 typedef struct Neuron Neuron;
 typedef struct Layer Layer;
 typedef struct Brain Brain;

struct Neuron {
	int id;
	float value;
	float activation;
	float error;

	float *weights;  // list of weights
	int numWeights;
};

typedef struct Layer {
	int numNeurons;
	Neuron **neurons;
	Layer *next;	// implicit list.
	Layer *prev;
} Layer;

typedef struct Brain {
	Layer **layers;
	float learning_rate;
	int numLayers;
} Brain;

float sigmoid(float x) {
	// http://www.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node72.html
	float exp_value;
	float return_value;

	/*** Exponential calculation ***/
	exp_value = exp((double) -x);

	/*** Final sigmoid value ***/
	return_value = 1 / (1 + exp_value);

	return return_value;
}

float randActivation() {
	return (float)rand()/(float)(RAND_MAX);
}

float randWeight() {
	return (float)rand()/(float)(RAND_MAX);
}

Neuron *neuron_create(int idx, int value, int activation, int numWeights) {
	Neuron *neuron = calloc(1, sizeof(Neuron));
	neuron->value;
	neuron->activation;
	neuron->id = idx;
	neuron->error = 0;
	neuron->weights = numWeights > 0 ? calloc(sizeof(float), numWeights) : NULL;
	neuron->numWeights = numWeights;

	// randomize weights
	for (int i = 0; i < numWeights; i++) {
		neuron->weights[i] = randWeight();
	}

	return neuron;
}

void neuron_connect_to(Neuron *from, Neuron *to, float weight) {
	to->weights[from->id] = weight;
}

void layer_print(Layer *layer) {

	printf("<Layer count='%d'>\n", layer->numNeurons);

	for (int i = 0; i < layer->numNeurons; i++) {
		Neuron *neuron = layer->neurons[i];
		printf("\t<Neuron value='%.1f' activ='%.2f'>", neuron->value, neuron->activation);
		
		for (int j = 0; j < neuron->numWeights; j++) {
			printf("%.2f ", neuron->weights[j]);
		}
		
		printf("</Neuron>\n");
	}

	printf("</Layer>\n");
}

Layer *layer_create(int numNeurons, Layer *previous) {
	// initialize layer.
	Layer *layer = calloc(1, sizeof(Layer));
	
	// allocate a bunch of neurons.
	srand((unsigned)time(NULL));

	Neuron **neurons = calloc(sizeof(Neuron *), numNeurons + 1);
	neurons[numNeurons] = NULL; // null terminator.
	
	int prevCount = previous != NULL ? previous->numNeurons : 0;

	for (int i = 0; i < numNeurons; i++) {
		neurons[i] = neuron_create(i, 0, randActivation(), prevCount);
	}
	
	layer->neurons = neurons;
	layer->numNeurons = numNeurons;

	return layer;
}

float *layer_get_values(Layer *layer) {
	float *values = malloc(sizeof(float) * layer->numNeurons);

	int i = 0;
	Neuron **neurons = layer->neurons;
	for (Neuron *neuron = *neurons; neuron != NULL; neuron = *(++neurons)) {
		values[i++] = neuron->value;
	}

	return values;
}

float dot(float *a, float *b, int dimen) {
	float product = 0;
	for (int i = 0; i < dimen; i++) {
		product += a[i] * b[i];
	}
	return product;
}

// Sets all the neuron values to 0 in a layer.
void layer_clear_values(Layer *layer) {
	if (!layer) return;
	Neuron **neurons = layer->neurons;
	for (int i = 0; i < layer->numNeurons; i++) {
		neurons[i]->value = 0;
	}
}

// Sets all of the values to the values specified in the array.
void layer_set_values(Layer *layer, float values[]) {
	if (!layer) return;
	Neuron **neurons = layer->neurons;
	for (int i = 0; i < layer->numNeurons; i++) {
		neurons[i]->value = values[i];
	}
}

void brain_print(Brain *brain) {
	Layer **layers = brain->layers;
	
	printf("<brain>\n");
	
	for (int i = 0; i < brain->numLayers; i++) {
		layer_print(layers[i]);
	}

	printf("</brain>\n");
}

Brain *brain_create(int numLayers, int depths[], float learning_rate) {
	Brain *brain = calloc(1, sizeof(Brain));

	Layer **layers = calloc(sizeof(Layer *), numLayers);
	Layer *prev = NULL;

	for (int i = 0; i < numLayers; i++) {
		layers[i] = layer_create(depths[i], prev);
		prev = layers[i];
	}

	brain->learning_rate = learning_rate;
	brain->layers = layers;
	brain->numLayers = numLayers;
	
	return brain;
}

Layer *brain_perform_inference(Brain *brain, float values[]) {
	
	Layer **layers = brain->layers;

	// give input data to the network.
	layer_set_values(layers[0], values);

	for (int i = 0; i < brain->numLayers-1; i++) {

		Layer *current = layers[i];
		Layer *next = layers[i+1];
		
		layer_clear_values(next);

		// collect all the values from this row.
		float values[current->numNeurons];
		for (int i = 0; i < current->numNeurons; i++) {
			values[i] = current->neurons[i]->value;
		}

		// for each weight in the next row, 
		// perform a dot product of (values * weights[i]), for the i-th node.
		Neuron **neurons = next->neurons;
		for (Neuron *neuron = *neurons; neuron != NULL; neuron = *(++neurons)) {
			neuron->value = sigmoid( 
				dot(values, neuron->weights, current->numNeurons) 
					+ 
				neuron->activation);
		}
	}

	// return the outer-most layer.
	return layers[brain->numLayers - 1];
}

void brain_train(Brain *brain, float data[], float expect[]) {

	Layer *last = brain_perform_inference(brain, data);

	Layer *current = last;

	/** See how bad the error was. **/
	Neuron **neurons = current->neurons; int j = 0;
	for (Neuron *neuron = *neurons; neuron != NULL; neuron = *(++neurons)) {
		// error computation
		neuron->error = neuron->value - expect[j];
		j++;
	}

	/** For each layer, see how much each prior neuron affected it. **/
	for (int i = brain->numLayers - 1; i >= 1; i--) {
		Layer *current = brain->layers[i];
		Layer *prev = brain->layers[i-1];

		Neuron **neurons = current->neurons;
		Neuron **neuronsPrev = prev->neurons;

		for (int j = 0; j < current->numNeurons; j++) {
			Neuron *neuron = neurons[j];

			for (int k = 0; k < neuron->numWeights; k++) {
				float dw = brain->learning_rate * neuron->weights[k] * neuron->error;
				neuron->weights[k] = neuron->weights[k] + dw;
				neuronsPrev[k]->error = dw; // TODO: Is this correct?
			}
		}
	}
}
