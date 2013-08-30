package com.storassa.javapp;

import java.util.LinkedList;
import java.util.Random;

public class NeuralNet {

   private Neuron[][] net;

   public NeuralNet(double[] input) {

      int size = input.length;
      Synapsis syn;

      // create the net
      net = new Neuron[size][];
      for (int i = 0; i < size; i++) {
         net[i] = new Neuron[i + 2];
         for (int t = 0; t < i + 2; t++)
            net[i][t] = new Neuron(i, t);
      }
      

      // initialize last layer of the net
      for (int i = 0; i <= size; i++) {
         
         // initialize output of neuron 0 to 1 and all other neurons to input
         if (i > 0)
            net[size - 1][i].output = input[i - 1];
         else
            net[size - 1][i].output = 1;
         
         // initialize all synapsis
         for (int k = 1; k <= size - 1; k++) {
            syn = new Synapsis(net[size - 1][i], net[size - 2][k]);
            net[size - 1][i].next.add(syn);
            net[size - 2][k].prev.add(syn);
         }
      }

      // create the rest of the net
      for (int i = size - 2; i >= 0; i--) {

         // initialize first neuron to 1
         net[i][0].output = 1;
         net[i][0].signal = 0;
         net[i][0].prev = null;

         // for layers higher than zero, updates synopsis
         if (i > 0) {
            for (int t = 1; t < i - 2; t++) {
               syn = new Synapsis(net[i][0], net[i + 1][t]);
               net[i][0].next.add(syn);
               net[i + 1][t].prev.add(syn);
            }
         }

         // initialize all the other neurons
         for (int t = 1; t < i; t++) {
            for (int k = 1; k < i; k++) {
               syn = new Synapsis(net[i][t], net[i - 1][k]);
               net[i - 1][k].prev.add(syn);
               net[i][t].next.add(syn);
            }
         }
      }

      System.out.println();
   }

   private class Neuron {
      double signal;
      double output;
      LinkedList<Synapsis> next;
      LinkedList<Synapsis> prev;
      int layer;
      int index;
      
      public Neuron(int _layer, int _index) {
         index = _index;
         layer = _layer;
         next = new LinkedList<Synapsis>();
         prev = new LinkedList<Synapsis>();
      }
      
      public String toString() {
         StringBuilder result = new StringBuilder();
         
         result.append("Layer: " + layer + ", index: " + index);
         result.append("\nSignal = " + signal);
         result.append("\nOutput = " + output);
         
         return result.toString();
      }
   }

   private class Synapsis {
      final Neuron from;
      final Neuron to;
      double weight;

      public Synapsis(Neuron _from, Neuron _to) {
         Random r = new Random();
         from = _from;
         to = _to;
         weight = r.nextDouble();
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
      double[] input = {1d, 2d, 3d, 4d, 5d};
      NeuralNet net = new NeuralNet(input);
   }
}
