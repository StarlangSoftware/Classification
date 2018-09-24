package Classification.Model.DecisionTree;

import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Attribute.DiscreteIndexedAttribute;
import Classification.Classifier.Classifier;
import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;
import Classification.Parameter.RandomForestParameter;
import Classification.Performance.ClassificationPerformance;
import Math.DiscreteDistribution;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class DecisionNode implements Serializable {

    private ArrayList<DecisionNode> children = null;
    private InstanceList data = null;
    private String classLabel = null;
    private boolean leaf = false;
    private DecisionCondition condition = null;

    /**
     * The entropyForDiscreteAttribute method takes an attributeIndex and creates an ArrayList of DiscreteDistribution.
     * Then loops through the distributions and calculates the total entropy.
     *
     * @param attributeIndex Index of the attribute.
     * @return Total entropy for the discrete attribute.
     */
    private double entropyForDiscreteAttribute(int attributeIndex) {
        double sum = 0.0;
        ArrayList<DiscreteDistribution> distributions = data.attributeClassDistribution(attributeIndex);
        for (DiscreteDistribution distribution : distributions) {
            sum += (distribution.getSum() / data.size()) * distribution.entropy();
        }
        return sum;
    }

    /**
     * The createChildrenForDiscreteIndexed method creates an ArrayList of DecisionNodes as children and a partition with respect to
     * indexed attribute.
     *
     * @param attributeIndex Index of the attribute.
     * @param attributeValue Value of the attribute.
     * @param parameter      RandomForestParameter like seed, ensembleSize, attributeSubsetSize.
     * @param isStump        Refers to decision trees with only 1 splitting rule.
     */
    private void createChildrenForDiscreteIndexed(int attributeIndex, int attributeValue, RandomForestParameter parameter, boolean isStump) {
        Partition childrenData;
        childrenData = data.divideWithRespectToIndexedAttribute(attributeIndex, attributeValue);
        children = new ArrayList<DecisionNode>();
        children.add(new DecisionNode(childrenData.get(0), new DecisionCondition(attributeIndex, new DiscreteIndexedAttribute("", attributeValue, ((DiscreteIndexedAttribute) data.get(0).getAttribute(attributeIndex)).getMaxIndex())), parameter, isStump));
        children.add(new DecisionNode(childrenData.get(1), new DecisionCondition(attributeIndex, new DiscreteIndexedAttribute("", -1, ((DiscreteIndexedAttribute) data.get(0).getAttribute(attributeIndex)).getMaxIndex())), parameter, isStump));
    }

    /**
     * The createChildrenForDiscrete method creates an ArrayList of values, a partition with respect to attributes and an ArrayList
     * of DecisionNodes as children.
     *
     * @param attributeIndex Index of the attribute.
     * @param parameter      RandomForestParameter like seed, ensembleSize, attributeSubsetSize.
     * @param isStump        Refers to decision trees with only 1 splitting rule.
     */
    private void createChildrenForDiscrete(int attributeIndex, RandomForestParameter parameter, boolean isStump) {
        Partition childrenData;
        ArrayList<String> valueList;
        valueList = data.getAttributeValueList(attributeIndex);
        childrenData = data.divideWithRespectToAttribute(attributeIndex);
        children = new ArrayList<DecisionNode>();
        for (int i = 0; i < valueList.size(); i++) {
            children.add(new DecisionNode(childrenData.get(i), new DecisionCondition(attributeIndex, new DiscreteAttribute(valueList.get(i))), parameter, isStump));
        }
    }

    /**
     * The createChildrenForContinuous method creates an ArrayList of DecisionNodes as children and a partition with respect to
     * continious attribute and the given split value.
     *
     * @param attributeIndex Index of the attribute.
     * @param parameter      RandomForestParameter like seed, ensembleSize, attributeSubsetSize.
     * @param isStump        Refers to decision trees with only 1 splitting rule.
     * @param splitValue     Split value is used for partitioning.
     */
    private void createChildrenForContinuous(int attributeIndex, double splitValue, RandomForestParameter parameter, boolean isStump) {
        Partition childrenData;
        childrenData = data.divideWithRespectToAttribute(attributeIndex, splitValue);
        children = new ArrayList<DecisionNode>();
        children.add(new DecisionNode(childrenData.get(0), new DecisionCondition(attributeIndex, '<', new ContinuousAttribute(splitValue)), parameter, isStump));
        children.add(new DecisionNode(childrenData.get(1), new DecisionCondition(attributeIndex, '>', new ContinuousAttribute(splitValue)), parameter, isStump));
    }

    /**
     * The DecisionNode method takes {@link InstanceList} data as input and then it sets the class label parameter by finding
     * the most occurred class label of given data, it then gets distinct class labels as class labels ArrayList. Later, it adds ordered
     * indices to the indexList and shuffles them randomly. Then, it gets the class distribution of given data and finds the best entropy value
     * of these class distribution.
     * <p>
     * If an attribute of given data is {@link DiscreteIndexedAttribute}, it creates a Distribution according to discrete indexed attribute class distribution
     * and finds the entropy. If it is better than the last best entropy it reassigns the best entropy, best attribute and best split value according to
     * the newly founded best entropy's index. At the end, it also add new distribution to the class distribution .
     * <p>
     * If an attribute of given data is {@link DiscreteAttribute}, it directly finds the entropy. If it is better than the last best entropy it
     * reassigns the best entropy, best attribute and best split value according to the newly founded best entropy's index.
     * <p>
     * If an attribute of given data is {@link ContinuousAttribute}, it creates two distributions; left and right according to class distribution
     * and discrete distribution respectively, and finds the entropy. If it is better than the last best entropy it reassigns the best entropy,
     * best attribute and best split value according to the newly founded best entropy's index. At the end, it also add new distribution to
     * the right distribution and removes from left distribution .
     *
     * @param data      {@link InstanceList} input.
     * @param condition {@link DecisionCondition} to check.
     * @param parameter RandomForestParameter like seed, ensembleSize, attributeSubsetSize.
     * @param isStump   Refers to decision trees with only 1 splitting rule.
     */
    public DecisionNode(InstanceList data, DecisionCondition condition, RandomForestParameter parameter, boolean isStump) {
        double bestEntropy, entropy;
        DiscreteDistribution leftDistribution, rightDistribution, classDistribution;
        int bestAttribute = -1, size;
        double bestSplitValue = 0, previousValue, splitValue;
        Instance instance;
        ArrayList<String> classLabels;
        this.condition = condition;
        this.data = data;
        classLabel = Classifier.getMaximum(data.getClassLabels());
        leaf = true;
        classLabels = data.getDistinctClassLabels();
        if (classLabels.size() == -1) {
            return;
        }
        if (isStump && condition != null) {
            return;
        }
        ArrayList<Integer> indexList = new ArrayList<Integer>();
        for (int i = 0; i < data.get(0).attributeSize(); i++) {
            indexList.add(i);
        }
        if (parameter != null) {
            Collections.shuffle(indexList, new Random(parameter.getSeed()));
            size = parameter.getAttributeSubsetSize();
        } else {
            size = data.get(0).attributeSize();
        }
        classDistribution = data.classDistribution();
        bestEntropy = data.classDistribution().entropy();
        for (int j = 0; j < size; j++) {
            int index = indexList.get(j);
            if (data.get(0).getAttribute(index) instanceof DiscreteIndexedAttribute) {
                for (int k = 0; k < ((DiscreteIndexedAttribute) data.get(0).getAttribute(index)).getMaxIndex(); k++) {
                    DiscreteDistribution distribution = data.discreteIndexedAttributeClassDistribution(index, k);
                    if (distribution.getSum() > 0) {
                        classDistribution.removeDistribution(distribution);
                        entropy = (classDistribution.entropy() * classDistribution.getSum() + distribution.entropy() * distribution.getSum()) / data.size();
                        if (entropy < bestEntropy) {
                            bestEntropy = entropy;
                            bestAttribute = index;
                            bestSplitValue = k;
                        }
                        classDistribution.addDistribution(distribution);
                    }
                }
            } else {
                if (data.get(0).getAttribute(index) instanceof DiscreteAttribute) {
                    entropy = entropyForDiscreteAttribute(index);
                    if (entropy < bestEntropy) {
                        bestEntropy = entropy;
                        bestAttribute = index;
                    }
                } else {
                    if (data.get(0).getAttribute(index) instanceof ContinuousAttribute) {
                        data.sort(index);
                        previousValue = -Double.MAX_VALUE;
                        leftDistribution = data.classDistribution();
                        rightDistribution = new DiscreteDistribution();
                        for (int k = 0; k < data.size(); k++) {
                            instance = data.get(k);
                            if (k == 0) {
                                previousValue = ((ContinuousAttribute) instance.getAttribute(index)).getValue();
                            } else {
                                if (((ContinuousAttribute) instance.getAttribute(index)).getValue() != previousValue) {
                                    splitValue = (previousValue + ((ContinuousAttribute) instance.getAttribute(index)).getValue()) / 2;
                                    previousValue = ((ContinuousAttribute) instance.getAttribute(index)).getValue();
                                    entropy = (leftDistribution.getSum() / data.size()) * leftDistribution.entropy() + (rightDistribution.getSum() / data.size()) * rightDistribution.entropy();
                                    if (entropy < bestEntropy) {
                                        bestEntropy = entropy;
                                        bestSplitValue = splitValue;
                                        bestAttribute = index;
                                    }
                                }
                            }
                            leftDistribution.removeItem(instance.getClassLabel());
                            rightDistribution.addItem(instance.getClassLabel());
                        }
                    }
                }
            }
        }
        if (bestAttribute != -1) {
            leaf = false;
            if (data.get(0).getAttribute(bestAttribute) instanceof DiscreteIndexedAttribute) {
                createChildrenForDiscreteIndexed(bestAttribute, (int) bestSplitValue, parameter, isStump);
            } else {
                if (data.get(0).getAttribute(bestAttribute) instanceof DiscreteAttribute) {
                    createChildrenForDiscrete(bestAttribute, parameter, isStump);
                } else {
                    if (data.get(0).getAttribute(bestAttribute) instanceof ContinuousAttribute) {
                        createChildrenForContinuous(bestAttribute, bestSplitValue, parameter, isStump);
                    }
                }
            }
        }
    }

    /**
     * The prune method takes a {@link DecisionTree} and an {@link InstanceList} as inputs. It checks the classification performance
     * of given InstanceList before pruning, i.e making a node leaf, and after pruning. If the after performance is better than the
     * before performance it prune the given InstanceList from the tree.
     *
     * @param tree     DecisionTree that will be pruned if conditions hold.
     * @param pruneSet Small subset of tree that will be removed from tree.
     */
    public void prune(DecisionTree tree, InstanceList pruneSet) {
        ClassificationPerformance before, after;
        if (leaf)
            return;
        before = tree.testClassifier(pruneSet);
        leaf = true;
        after = tree.testClassifier(pruneSet);
        if (after.getAccuracy() < before.getAccuracy()) {
            leaf = false;
            for (DecisionNode node : children) {
                node.prune(tree, pruneSet);
            }
        }
    }

    /**
     * The predict method takes an {@link Instance} as input and performs prediction on the DecisionNodes and returns the prediction
     * for that instance.
     *
     * @param instance Instance to make prediction.
     * @return The prediction for given instance.
     */
    public String predict(Instance instance) {
        if (instance instanceof CompositeInstance) {
            ArrayList<String> possibleClassLabels = ((CompositeInstance) instance).getPossibleClassLabels();
            DiscreteDistribution distribution = data.classDistribution();
            String predictedClass = distribution.getMaxItem(possibleClassLabels);
            if (leaf) {
                return predictedClass;
            } else {
                for (DecisionNode node : children) {
                    if (node.condition.satisfy(instance)) {
                        String childPrediction = node.predict(instance);
                        if (childPrediction != null) {
                            return childPrediction;
                        } else {
                            return predictedClass;
                        }
                    }
                }
                return predictedClass;
            }
        } else {
            if (leaf) {
                return classLabel;
            } else {
                for (DecisionNode node : children) {
                    if (node.condition.satisfy(instance)) {
                        return node.predict(instance);
                    }
                }
                return classLabel;
            }
        }
    }
}
