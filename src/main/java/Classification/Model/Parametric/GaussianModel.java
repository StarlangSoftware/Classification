package Classification.Model.Parametric;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.HashMap;

import Classification.Model.ValidatedModel;
import Math.DiscreteDistribution;
import Math.Vector;

public abstract class GaussianModel extends ValidatedModel implements Serializable {
    protected DiscreteDistribution priorDistribution;

    /**
     * Abstract method calculateMetric takes an {@link Instance} and a String as inputs.
     *
     * @param instance {@link Instance} input.
     * @param Ci       String input.
     * @return A double value as metric.
     */
    protected abstract double calculateMetric(Instance instance, String Ci);

    /**
     * Loads the prior probability distribution from an input model file.
     * @param input Input model file.
     * @return Prior probability distribution.
     * @throws IOException If the input file can not be read, the method throws IOException.
     */
    protected int loadPriorDistribution(BufferedReader input) throws IOException {
        int size = Integer.parseInt(input.readLine());
        priorDistribution = new DiscreteDistribution();
        for (int i = 0; i < size; i++){
            String line = input.readLine();
            String[] items = line.split(" ");
            for (int j = 0; j < Integer.parseInt(items[1]); j++){
                priorDistribution.addItem(items[0]);
            }
        }
        return size;
    }

    /**
     * Loads hash map of vectors from input model file.
     * @param input Input model file.
     * @param size Number of vectors to be read from input model file.
     * @return Hash map of vectors.
     * @throws IOException If the input file can not be read, the method throws IOException.
     */
    protected HashMap<String, Vector> loadVectors(BufferedReader input, int size) throws IOException {
        HashMap<String, Vector> map = new HashMap<>();
        for (int i = 0; i < size; i++){
            String line = input.readLine();
            String[] items = line.split(" ");
            Vector vector = new Vector(Integer.parseInt(items[1]), 0);
            for (int j = 2; j < items.length; j++){
                vector.setValue(j - 2, Double.parseDouble(items[j]));
            }
            map.put(items[0], vector);
        }
        return map;
    }

    /**
     * The predict method takes an Instance as an input. First it gets the size of prior distribution and loops this size times.
     * Then it gets the possible class labels and and calculates metric value. At the end, it returns the class which has the
     * maximum value of metric.
     *
     * @param instance {@link Instance} to predict.
     * @return The class which has the maximum value of metric.
     */
    public String predict(Instance instance) {
        String predictedClass;
        String Ci;
        double metric;
        double maxMetric = -Double.MAX_VALUE;
        int size;
        if (instance instanceof CompositeInstance) {
            predictedClass = ((CompositeInstance) instance).getPossibleClassLabels().get(0);
            size = ((CompositeInstance) instance).getPossibleClassLabels().size();
        } else {
            predictedClass = priorDistribution.getMaxItem();
            size = priorDistribution.size();
        }
        for (int i = 0; i < size; i++) {
            if (instance instanceof CompositeInstance) {
                Ci = ((CompositeInstance) instance).getPossibleClassLabels().get(i);
            } else {
                Ci = priorDistribution.getItem(i);
            }
            if (priorDistribution.containsItem(Ci)) {
                metric = calculateMetric(instance, Ci);
                if (metric > maxMetric) {
                    maxMetric = metric;
                    predictedClass = Ci;
                }
            }
        }
        return predictedClass;
    }

    /**
     * Saves the prior probability distribution to an output model file.
     * @param output Output model file.
     */
    protected void savePriorDistribution(PrintWriter output){
        output.println(priorDistribution.size());
        for (int i = 0; i < priorDistribution.size(); i++){
            output.println(priorDistribution.getItem(i) + " " + priorDistribution.getValue(i));
        }
    }

    /**
     * Saves hash map of vectors to an output model file.
     * @param output Output model file.
     * @param map Hash map of vectors.
     */
    protected void saveVectors(PrintWriter output, HashMap<String, Vector> map){
        for (String c : map.keySet()){
            Vector vector = map.get(c);
            output.print(c + " " + vector.size());
            for (int i = 0; i < vector.size(); i++){
                output.print(" " + vector.getValue(i));
            }
            output.println();
        }
    }

    /**
     * Calculates the posterior probability distribution for the given instance according to Gaussian model.
     * @param instance Instance for which posterior probability distribution is calculated.
     * @return Posterior probability distribution for the given instance.
     */
    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        return null;
    }
}
