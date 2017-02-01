
#ifndef NEURON_H
#define NEURON_H

typedef struct Neuron Neuron;
typedef struct Synapse Synapse;
typedef struct Layer Layer;
typedef struct Brain Brain;

void layer_print(Layer *layer);
float *layer_get_values(Layer *layer);

Brain *brain_create(int numLayers, int depths[], float learning_rate);
Layer *brain_perform_inference(Brain *brain, float values[]);
void brain_train(Brain *brain, float data[], float expect[]);
void brain_print(Brain *brain);

#endif


