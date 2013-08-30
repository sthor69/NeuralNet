package com.storassa.javapp;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNet {

   private Neuron[][] net;

   public NeuralNet(int inputSize, int hiddenLayers, int[] hiddenLayerSize) {

      Synapse syn;

      // create the L layer of the net
      net = new Neuron[hiddenLayers + 2][];
      net[hiddenLayers + 1] = new Neuron[inputSize + 1];
      for (int i = 0; i <= inputSize; i++)
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
      for (int i = 0; i <= inputSize; i++) {
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
            
      System.out.println();
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

   public static void main(String[] args) {
      int[] layerSize = { 3, 3 };
      NeuralNet net = new NeuralNet(5, 2, layerSize);
   }
}
