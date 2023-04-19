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
            distribution = new DiscreteDistribution();
            int size = Integer.parseInt(input.readLine());
            for (int i = 0; i < size; i++){
                String line = input.readLine();
                String[] items = line.split(" ");
                int count = Integer.parseInt(items[1]);
                for(int j = 0; j < count; j++){
                    distribution.addItem(items[0]);
                }
            }
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
            output.println(distribution.size());
            for (int i = 0; i < distribution.size(); i++){
                output.println(distribution.getItem(i) + " " + distribution.getValue(i));
            }
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

}
