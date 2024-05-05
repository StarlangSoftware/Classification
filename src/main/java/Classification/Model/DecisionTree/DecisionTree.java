package Classification.Model.DecisionTree;

import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.ValidatedModel;
import Classification.Performance.ClassificationPerformance;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;

public class DecisionTree extends ValidatedModel implements Serializable {

    private final DecisionNode root;

    /**
     * Constructor that sets root node of the decision tree.
     *
     * @param root DecisionNode type input.
     */
    public DecisionTree(DecisionNode root) {
        this.root = root;
    }

    /**
     * Loads a decision tree model from an input model file.
     * @param fileName Model file name.
     */
    public DecisionTree(String fileName){
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            root = new DecisionNode(input);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The predict method  performs prediction on the root node of given instance, and if it is null, it returns the possible class labels.
     * Otherwise it returns the returned class labels.
     *
     * @param instance Instance make prediction.
     * @return Possible class labels.
     */
    public String predict(Instance instance) {
        String predictedClass = root.predict(instance);
        if ((predictedClass == null) && ((instance instanceof CompositeInstance))) {
            predictedClass = ((CompositeInstance) instance).getPossibleClassLabels().get(0);
        }
        return predictedClass;
    }

    /**
     * Calculates the posterior probability distribution for the given instance according to Decision tree model.
     * @param instance Instance for which posterior probability distribution is calculated.
     * @return Posterior probability distribution for the given instance.
     */
    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        return root.predictProbabilityDistribution(instance);
    }

    /**
     * Saves the decision tree model to an output file.
     * @param fileName Output file name.
     */
    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            root.saveTxt(output);
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Accessor for the root node of the decision tree.
     * @return Root node of the tree.
     */
    public DecisionNode getRoot(){
        return root;
    }

    /**
     * The prune method takes a {@link DecisionNode} and an {@link InstanceList} as inputs. It checks the classification performance
     * of given InstanceList before pruning, i.e making a node leaf, and after pruning. If the after performance is better than the
     * before performance it prune the given InstanceList from the tree.
     *
     * @param node     DecisionNode that will be pruned if conditions hold.
     * @param pruneSet Small subset of tree that will be removed from tree.
     */
    public void pruneNode(DecisionNode node, InstanceList pruneSet) {
        ClassificationPerformance before, after;
        if (node.leaf){
            return;
        }
        before = testClassifier(pruneSet);
        node.leaf = true;
        after = testClassifier(pruneSet);
        if (after.getAccuracy() < before.getAccuracy()) {
            node.leaf = false;
            for (DecisionNode child : node.children) {
                pruneNode(child, pruneSet);
            }
        }
    }

    public void generateTestCode(String codeFileName, String methodName){
        try {
            PrintWriter output = new PrintWriter(codeFileName);
            output.println("public static String " + methodName + "(String[] testData){");
            root.generateTestCode(output, 1);
            output.println("\treturn \"\";");
            output.println("}");
            output.close();
        } catch (FileNotFoundException ignored) {
        }
    }

    /**
     * The prune method takes an {@link InstanceList} and  performs pruning to the root node.
     *
     * @param pruneSet {@link InstanceList} to perform pruning.
     */
    public void prune(InstanceList pruneSet) {
        pruneNode(root, pruneSet);
    }
}
