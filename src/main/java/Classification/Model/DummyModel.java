package Classification.Model;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Parameter.Parameter;
import Math.DiscreteDistribution;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

public class DummyModel extends Model implements Serializable {

    private DiscreteDistribution distribution;

    /**
     * The predict method takes an Instance as an input and returns the entry of distribution which has the maximum value.
     *
     * @param instance Instance to make prediction.
     * @return The entry of distribution which has the maximum value.
     */
    public String predict(Instance instance) {
        if ((instance instanceof CompositeInstance)) {
            ArrayList<String> possibleClassLabels = ((CompositeInstance) instance).getPossibleClassLabels();
            return distribution.getMaxItem(possibleClassLabels);
        } else {
            return distribution.getMaxItem();
        }
    }

    /**
     * Calculates the posterior probability distribution for the given instance according to dummy model.
     * @param instance Instance for which posterior probability distribution is calculated.
     * @return Posterior probability distribution for the given instance.
     */
    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        return distribution.getProbabilityDistribution();
    }

    /**
     * Saves the dummy model to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            saveDiscreteDistribution(output, distribution);
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Training algorithm for the dummy classifier. Actually dummy classifier returns the maximum occurring class in
     * the training data, there is no training. Sets the distribution using the given {@link InstanceList}.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters -
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        distribution = trainSet.classDistribution();
    }

    /**
     * Loads the dummy model from an input file.
     * @param fileName File name of the dummy model.
     */
    @Override
    public void loadModel(String fileName) {
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            distribution = loadDiscreteDistribution(input);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
