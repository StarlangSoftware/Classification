package Classification.Model.Ensemble;

import Classification.Instance.Instance;
import Classification.Model.DecisionTree.DecisionNode;
import Classification.Model.DecisionTree.DecisionTree;
import Classification.Model.Model;
import Math.DiscreteDistribution;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

public abstract class TreeEnsembleModel extends Model implements Serializable {

    protected ArrayList<DecisionTree> forest;

    @Override
    public void loadModel(String fileName) {
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            int numberOfTrees = Integer.parseInt(input.readLine());
            forest = new ArrayList<>();
            for (int i = 0; i < numberOfTrees; i++){
                forest.add(new DecisionTree(new DecisionNode(input)));
            }
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
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

    /**
     * Calculates the posterior probability distribution for the given instance according to ensemble tree model.
     * @param instance Instance for which posterior probability distribution is calculated.
     * @return Posterior probability distribution for the given instance.
     */
    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        DiscreteDistribution distribution = new DiscreteDistribution();
        for (DecisionTree tree : forest) {
            distribution.addItem(tree.predict(instance));
        }
        return distribution.getProbabilityDistribution();
    }

    /**
     * Saves the ensemble tree model such as Random Forest or Bagging to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            output.println(forest.size());
            for (DecisionTree tree : forest) {
                tree.getRoot().saveTxt(output);
            }
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
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
        } catch (FileNotFoundException ignored) {
        }
    }

}
