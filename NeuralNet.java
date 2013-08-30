package com.storassa.javapp;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNet {

   private Neuron[][] net;
   private int size;
   private int hiddenLayers;
   private double min = -1, max = +1;

   public NeuralNet(int _size, int _hiddenLayers, int[] hiddenLayerSize) {

      size = _size;
      hiddenLayers = _hiddenLayers;
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
      for (int i = hiddenLayers -1; i > 1; i++)
         for (int t = 0; t < hiddenLayerSize[i]; t++)
            for (int k = 1; k < hiddenLayerSize[i - 1]; k++) {
               syn = new Synapse(net[i][t], net[i + 1][k]);
               net[i][t].addToSynapsis(syn);
               net[i + 1][k].addFromSynapsis(syn);
            }
            
      System.out.println();
   }
   
   public void setInput(double[] input) {
      
      // check that the size of the input is correct
      if (input.length != size)
         throw new RuntimeException("Input size does not match network topology");
      
      // set the output of the L layer neurons
      for (int i = 1; i < size; i++)
         net[hiddenLayers + 1][i].output = input[i - 1];      
   }
   
   public void compute() {
      
      double signal = 0;
      
      // compute the network excluding the layer 0
      for (int i = hiddenLayers; i > 0; i--)
         for (int t = 1; t < net[i].length; t++) {
            for (int k = 0; k < net[i + 1].length; k++)
               signal += net[i + 1][k].output * net[i][t].prev.get(k).weight;
            net[i][t].output = Math.tanh(signal);
         }
      
      // compute the layer 0
      signal = 0;
      for (int k = 0; k < net[1].length; k++)
         signal += net[1][k].output * net[0][0].prev.get(k).weight;
      net[0][0].output = Math.tanh(signal);
   }
   
   public void scale(double _min, double _max) {
      min = _min;
      max = _max;
   }
   
   public double getScaledOutput() {
      double result = net[0][0].output;
      return (result + 1) * (max - min) / 2 + min;
   }
   
   public double getRealOutput() {
      return net[0][0].output;
   }
   
   public static void main(String[] args) {
      int[] layerSize = { 3, 3 };
      double[] input = { 1d, 2d, 3d, 4d, 5d };
      NeuralNet net = new NeuralNet(5, 2, layerSize);
      net.setInput(input);
      net.compute();
      System.out.println(net.getRealOutput());
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

}
