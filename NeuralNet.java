package com.storassa.javapp;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNet {

   private Neuron[][] net;
   private int size;
   private int hiddenLayers;
   double[][] delta;
   private double min = -1, max = +1;

   /**
    * initialize the neural network
    * 
    * @param _size
    *           the size of the input data
    * @param _hiddenLayers
    *           the number of hidden layers
    * @param hiddenLayerSize
    *           the size of each hidden layer
    **/
   public NeuralNet(int _size, int _hiddenLayers, int[] hiddenLayerSize) {

      size = _size;
      hiddenLayers = _hiddenLayers;
      delta = new double[hiddenLayers + 2][];
      Synapse syn;

      // create the L layer of the net
      net = new Neuron[hiddenLayers + 2][];
      net[hiddenLayers + 1] = new Neuron[size + 1];
      for (int i = 0; i <= size; i++)
         net[hiddenLayers + 1][i] = new Neuron(hiddenLayers + 1, i);

      // create the 0 layer of the net
      net[0] = new Neuron[1];
      net[0][0] = new Neuron(0, 0);

      // create the hidden layers
      for (int i = 0; i < hiddenLayers; i++) {
         net[i + 1] = new Neuron[hiddenLayerSize[i]];
         net[i + 1][0] = new Neuron(i + 1, 0);
         for (int t = 1; t < hiddenLayerSize[i]; t++)
            net[i + 1][t] = new Neuron(i + 1, t);
      }

      // initialize the synapses of 0 layer of the net
      for (int i = 0; i < hiddenLayerSize[0]; i++) {
         syn = new Synapse(net[1][i], net[0][0]);
         net[1][i].addToSynapsis(syn);
         net[0][0].addFromSynapsis(syn);
      }

      // initialize the synapses of L layer of the net
      for (int i = 0; i <= size; i++) {
         for (int k = 1; k < hiddenLayerSize[hiddenLayers - 1]; k++) {
            syn = new Synapse(net[hiddenLayers + 1][i], net[hiddenLayers][k]);
            net[hiddenLayers + 1][i].addToSynapsis(syn);
            net[hiddenLayers][k].addFromSynapsis(syn);
         }
      }

      // initialize the synapses of last (near input) hidden layer of the net
      for (int i = 0; i < hiddenLayerSize[hiddenLayers - 1]; i++)
         for (int t = 1; t < hiddenLayerSize[hiddenLayers - 2]; t++) {
            syn = new Synapse(net[hiddenLayers][i], net[hiddenLayers - 1][t]);
            net[hiddenLayers][i].addToSynapsis(syn);
            net[hiddenLayers - 1][t].addFromSynapsis(syn);
         }

      // initialize the other hidden layers
      for (int i = hiddenLayers - 1; i > 1; i++)
         for (int t = 0; t < hiddenLayerSize[i]; t++)
            for (int k = 1; k < hiddenLayerSize[i - 1]; k++) {
               syn = new Synapse(net[i][t], net[i + 1][k]);
               net[i][t].addToSynapsis(syn);
               net[i + 1][k].addFromSynapsis(syn);
            }
   }

   /**
    * set the input data as outputs of the layer L of the neural network
    * 
    * @param input
    *           the input data
    */
   public void setInput(double[] input) {

      // check that the size of the input is correct
      if (input.length != size)
         throw new RuntimeException(
               "Input size does not match network topology");

      // set the output of the L layer neurons
      net[hiddenLayers + 1][0].output = 1;
      for (int i = 1; i < size; i++)
         net[hiddenLayers + 1][i].output = input[i - 1];
   }

   /**
    * compute all the values in the neural network (signals and outputs)
    */
   public void compute() {

      double signal = 0;

      // compute the network excluding the layer 0
      for (int i = hiddenLayers; i > 0; i--) {
         net[i][0].output = 1;
         for (int t = 1; t < net[i].length; t++) {
            signal = 0;
            for (int k = 0; k < net[i + 1].length; k++)
               signal += net[i + 1][k].output
                     * net[i][t].prev.get(t - 1).weight;
            net[i][t].output = Math.tanh(signal);
         }
      }

      // compute the layer 0
      signal = 0;
      for (int k = 0; k < net[1].length; k++)
         signal += net[1][k].output * net[0][0].prev.get(k).weight;
      net[0][0].output = Math.tanh(signal);
   }

   /**
    * update the neural network given the desired target and the eta constant
    * 
    * @param target
    *           the target given by the input data
    * @param eta
    *           the constant for the update of weights
    */
   public void update(double target, double eta) {
      for (int i = 0; i < hiddenLayers + 2; i++)
         delta[i] = new double[net[i].length];

      // compute the derivative of squared error (delta of last node)
      delta[0][0] = 2 * (net[0][0].output - target)
            * (1 - Math.pow(Math.tanh(net[0][0].signal), 2));
       System.out.println(net[0][0].output - target);

      // compute all the other layer deltas
      for (int i = 1; i < hiddenLayers + 1; i++)
         for (int t = 0; t < net[i].length; t++)
            delta[i][t] = evaluateDelta(i, t);

      // // compute the first hidden layer deltas
      // for (int i = 0; i < net[1].length; i++)
      // delta[0][0] = evaluateDelta(net[1][i].output,
      // net[1][i].next.get(0).weight, delta[0][0], 1);
      //
      // // compute all the other layer deltas
      // for (int i = 1; i < hiddenLayers + 2; i++)
      // for (int t = 1; t < net[i].length; t++)
      // for (int k = 0; k < net[i][t].next.size(); k++)
      // sum += 1;

      // compute the weights
      for (int i = 1; i < hiddenLayers + 2; i++)
         for (int t = 0; t < net[i].length; t++) {
            if (i == 1) {
               double var = eta * net[i][t].output * delta[i][t];
               net[i][t].next.get(0).weight -= eta * net[i][t].output
                     * delta[i][t];
            } else
               for (int k = 1; k < net[i - 1].length; k++) {
                  double var = eta * net[i][t].output * delta[i][t];
                  net[i][t].next.get(k - 1).weight -= eta * net[i][t].output
                        * delta[i][t];
               }

         }

      double sum;
      for (int i = 0; i < hiddenLayers + 2; i++) {
         sum = 0;
         for (int t = 0; t < net[i].length; t++) {
            for (Synapse syn : net[i][t].next)
               sum += syn.weight;
         }
//         System.out.println("Sum of layer " + i + " weights: " + sum);
      }

      // update all the output and signals
      compute();

   }

   /**
    * scale the target value in order to be used also for real value predictions
    * 
    * @param _min
    *           the minimum value of the range
    * @param _max
    *           the maximum value of the range
    */
   public void scale(double _min, double _max) {
      min = _min;
      max = _max;
   }

   /**
    * get the output in the user defined range, set with
    * NeuralNet.scale(int,int)
    * 
    * @return the scaled output of node (0, 0)
    */
   public double getScaledOutput() {
      double result = net[0][0].output;
      return (result + 1) * (max - min) / 2 + min;
   }

   /**
    * get the output in the [-1, +1] range
    * 
    * @return the real output of node (0, 0)
    */
   public double getRealOutput() {
      return net[0][0].output;
   }

   /**
    * for unit testing
    * 
    * @param args
    *           arguments
    */
   public static void main(String[] args) {
      int[] layerSize = { 10, 10 };
      double[] input = new double[10];
      Random r = new Random();
      int sum = 0;

      NeuralNet net = new NeuralNet(10, 2, layerSize);

      for (int t = 0; t < TRAINING_SIZE; t++) {
         for (int i = 0; i < 10; i++) {
            input[i] = r.nextInt(2);
            sum += input[i];
         }
         net.setInput(input);
         net.compute();
         net.update(sum % 2, 0.9);
         sum = 0;
      }

      for (int i = 0; i < 10; i++) {
         input[i] = r.nextInt(2);
         sum += input[i];
      }
      net.setInput(input);
      net.compute();
      System.out.println(sum % 2 + " -> " + net.getRealOutput());
   }

   private class Neuron {
      double signal;
      double output;
      ArrayList<Synapse> next;
      ArrayList<Synapse> prev;
      int layer;
      int index;

      public Neuron(int _layer, int _index) {
         index = _index;
         layer = _layer;
         next = new ArrayList<Synapse>();
         prev = new ArrayList<Synapse>();
      }

      public void addFromSynapsis(Synapse syn) {
         prev.add(syn);
         signal += syn.out;
         output = Math.tanh(signal);
      }

      public void addToSynapsis(Synapse syn) {
         next.add(syn);
      }

      public String toString() {
         StringBuilder result = new StringBuilder();

         result.append("Layer: " + layer + ", index: " + index);
         result.append("\nSignal = " + signal);
         result.append("\nOutput = " + output);

         return result.toString();
      }
   }

   private class Synapse {
      final Neuron from;
      final Neuron to;
      double weight;
      double out;

      public Synapse(Neuron _from, Neuron _to) {
         Random r = new Random();
         from = _from;
         to = _to;
         weight = r.nextDouble();
         out = from.output * weight;
      }

      public String toString() {
         StringBuilder result = new StringBuilder();

         result.append("[" + from.layer + "," + from.index + "]");
         result.append("-----");
         result.append(weight);
         result.append("-----> ");
         result.append("[" + to.layer + "," + to.index + "]");

         return result.toString();
      }
   }

   private double evaluateDelta(int currentLayer, int currentIndex) {
      double result = 0;

      // if layer 1 is updating, the following layer (the 0) does not have
      // the bias neuron
      if (currentLayer == 1)
         result = net[currentLayer][currentIndex].next.get(0).weight
               * delta[0][0];
      else
         for (int i = 0; i < net[currentLayer - 1].length - 1; i++) {
            double _weight = net[currentLayer][currentIndex].next.get(i).weight;
            double _delta = delta[currentLayer + 1][i];
            result += net[currentLayer][currentIndex].next.get(i).weight
                  * delta[currentLayer - 1][i];
         }

      result *= 1 - Math.pow(net[currentLayer][currentIndex].output, 2d);

      return result;
   }

   private final static int TRAINING_SIZE = 100000;
}
