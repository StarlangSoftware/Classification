package Classification.Model;

import Classification.Instance.Instance;
import Classification.Model.DecisionTree.DecisionTree;
import Math.DiscreteDistribution;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

public class TreeEnsembleModel extends Model implements Serializable {

    private ArrayList<DecisionTree> forest;

    /**
     * A constructor which sets the {@link ArrayList} of {@link DecisionTree} with given input.
     *
     * @param forest An {@link ArrayList} of {@link DecisionTree}.
     */
    public TreeEnsembleModel(ArrayList<DecisionTree> forest) {
        this.forest = forest;
    }

    /**
     * The predict method takes an {@link Instance} as an input and loops through the {@link ArrayList} of {@link DecisionTree}s.
     * Makes prediction for the items of that ArrayList and returns the maximum item of that ArrayList.
     *
     * @param instance Instance to make prediction.
     * @return The maximum prediction of a given Instance.
     */
    public String predict(Instance instance) {
        DiscreteDistribution distribution = new DiscreteDistribution();
        for (DecisionTree tree : forest) {
            distribution.addItem(tree.predict(instance));
        }
        return distribution.getMaxItem();
    }

    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        DiscreteDistribution distribution = new DiscreteDistribution();
        for (DecisionTree tree : forest) {
            distribution.addItem(tree.predict(instance));
        }
        return distribution.getProbabilityDistribution();
    }

    public void generateTestCode(String codeFileName, String methodName){
        int j = 1;
        for (DecisionTree tree : forest){
            tree.generateTestCode(codeFileName + "-" + j + ".txt", "testC45_" + j);
            j++;
        }
        try {
            PrintWriter output = new PrintWriter(codeFileName);
            output.println("public static String " + methodName + "(String[] testData){");
            output.println("\tCounterHashMap<String> counts = new CounterHashMap<>();");
            output.println("\tfor (int i = 1; i <= " + forest.size() + "; i++){");
            output.println("\t\tswitch (i){");
            for (int i = 1; i <= forest.size(); i++){
                output.println("\t\t\tcase " + i + ":");
                output.println("\t\t\t\tcounts.put(testC45_" + i +  "(testData));");
                output.println("\t\t\t\tbreak;");
            }
            output.println("\t\t}");
            output.println("\t}");
            output.println("\treturn counts.max();");
            output.println("}");
            output.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

}
