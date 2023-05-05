package Classification.Model;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Math.DiscreteDistribution;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;

public class DummyModel extends Model implements Serializable {

    private DiscreteDistribution distribution;

    /**
     * Constructor which sets the distribution using the given {@link InstanceList}.
     *
     * @param trainSet {@link InstanceList} which is used to get the class distribution.
     */
    public DummyModel(InstanceList trainSet) {
        this.distribution = trainSet.classDistribution();
    }

    public DummyModel(String fileName){
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), StandardCharsets.UTF_8));
            distribution = loadDiscreteDistribution(input);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

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

    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        return distribution.getProbabilityDistribution();
    }

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

}
