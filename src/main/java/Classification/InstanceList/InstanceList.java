package Classification.InstanceList;

import Classification.Attribute.*;
import Classification.DataSet.DataDefinition;
import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.Instance.InstanceClassComparator;
import Classification.Instance.InstanceComparator;
import Classification.Model.Model;
import Math.DiscreteDistribution;
import Math.Vector;
import Math.Matrix;
import Sampling.Bootstrap;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class InstanceList implements Serializable {

    protected ArrayList<Instance> list;

    /**
     * Empty constructor for an instance list. Initializes the instance list with zero instances.
     */
    public InstanceList() {
        list = new ArrayList<Instance>();
    }

    /**
     * Constructor for an instance list with a given data definition, data file and a separator character. Each instance
     * must be stored in a separate line separated with the character separator. The last item must be the class label.
     * The function reads the file line by line and for each line; depending on the data definition, that is, type of
     * the attributes, adds discrete and continuous attributes to a new instance. For example, given the data set file
     * <p>
     * red;1;0.4;true
     * green;-1;0.8;true
     * blue;3;1.3;false
     * <p>
     * where the first attribute is a discrete attribute, second and third attributes are continuous attributes, the
     * fourth item is the class label.
     *
     * @param definition Data definition of the data set.
     * @param separator  Separator character which separates the attribute values in the data file.
     * @param fileName   Name of the data set file.
     */
    public InstanceList(DataDefinition definition, String separator, String fileName) {
        Instance current;
        String line;
        list = new ArrayList<Instance>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF8"));
            line = br.readLine();
            while (line != null) {
                String[] attributeList = line.split(separator);
                if (attributeList.length == definition.attributeCount() + 1) {
                    current = new Instance(attributeList[attributeList.length - 1]);
                    for (int i = 0; i < attributeList.length - 1; i++) {
                        switch (definition.getAttributeType(i)) {
                            case DISCRETE:
                                current.addAttribute(new DiscreteAttribute(attributeList[i]));
                                break;
                            case BINARY:
                                current.addAttribute(new BinaryAttribute(Boolean.parseBoolean(attributeList[i])));
                                break;
                            case CONTINUOUS:
                                current.addAttribute(new ContinuousAttribute(Double.parseDouble(attributeList[i])));
                                break;
                        }
                    }
                    list.add(current);
                }
                line = br.readLine();
            }
        } catch (FileNotFoundException fileNotFoundException) {
            System.out.println("Dataset with fileName " + fileName + " not found");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Empty constructor for an instance list. Initializes the instance list with the given instance list.
     *
     * @param list New list for the list variable.
     */
    public InstanceList(ArrayList<Instance> list) {
        this.list = list;
    }

    /**
     * Adds instance to the instance list.
     *
     * @param instance Instance to be added.
     */
    public void add(Instance instance) {
        list.add(instance);
    }

    /**
     * Adds a list of instances to the current instance list.
     *
     * @param instanceList List of instances to be added.
     */
    public void addAll(ArrayList<Instance> instanceList) {
        list.addAll(instanceList);
    }

    /**
     * Returns size of the instance list.
     *
     * @return Size of the instance list.
     */
    public int size() {
        return list.size();
    }

    /**
     * Accessor for a single instance with the given index.
     *
     * @param index Index of the instance.
     * @return Instance with index 'index'.
     */
    public Instance get(int index) {
        return list.get(index);
    }

    /**
     * Sorts instance list according to the attribute with index 'attributeIndex'.
     *
     * @param attributeIndex index of the attribute.
     */
    public void sort(int attributeIndex) {
        InstanceComparator comparator = new InstanceComparator(attributeIndex);
        Collections.sort(list, comparator);
    }

    /**
     * Sorts instance list.
     */
    public void sort() {
        Collections.sort(list, new InstanceClassComparator());
    }

    /**
     * Shuffles the instance list.
     * @param seed Seed is used for random number generation.
     */
    public void shuffle(int seed) {
        Collections.shuffle(list, new Random(seed));
    }

    /**
     * Shuffles the instance list.
     * @param random Random function.
     */
    public void shuffle(Random random) {
        Collections.shuffle(list, random);
    }

    /**
     * Creates a bootstrap sample from the current instance list.
     *
     * @param seed To create a different bootstrap sample, we need a new seed for each sample.
     * @return Bootstrap sample.
     */
    public Bootstrap bootstrap(int seed) {
        return new Bootstrap<Instance>(list, seed);
    }

    /**
     * Extracts the class labels of each instance in the instance list and returns them in an array of {@link String}.
     *
     * @return An array list of class labels.
     */
    public ArrayList<String> getClassLabels() {
        ArrayList<String> classLabels = new ArrayList<String>();
        for (Instance instance : list) {
            classLabels.add(instance.getClassLabel());
        }
        return classLabels;
    }

    /**
     * Extracts the class labels of each instance in the instance list and returns them as a set.
     *
     * @return An {@link ArrayList} of distinct class labels.
     */
    public ArrayList<String> getDistinctClassLabels() {
        ArrayList<String> classLabels = new ArrayList<String>();
        for (Instance instance : list) {
            if (!classLabels.contains(instance.getClassLabel())) {
                classLabels.add(instance.getClassLabel());
            }
        }
        return classLabels;
    }

    /**
     * Extracts the possible class labels of each instance in the instance list and returns them as a set.
     *
     * @return An {@link ArrayList} of distinct class labels.
     */
    public ArrayList<String> getUnionOfPossibleClassLabels() {
        ArrayList<String> possibleClassLabels = new ArrayList<String>();
        for (Instance instance : list) {
            if (instance instanceof CompositeInstance) {
                CompositeInstance compositeInstance = (CompositeInstance) instance;
                for (String possibleClassLabel : compositeInstance.getPossibleClassLabels()) {
                    if (!possibleClassLabels.contains(possibleClassLabel)) {
                        possibleClassLabels.add(possibleClassLabel);
                    }
                }
            } else {
                if (!possibleClassLabels.contains(instance.getClassLabel())) {
                    possibleClassLabels.add(instance.getClassLabel());
                }
            }
        }
        return possibleClassLabels;
    }

    /**
     * Extracts distinct discrete values of a given attribute as an array of strings.
     *
     * @param attributeIndex Index of the discrete attribute.
     * @return An array of distinct values of a discrete attribute.
     */
    public ArrayList<String> getAttributeValueList(int attributeIndex) {
        ArrayList<String> valueList = new ArrayList<String>();
        for (Instance instance : list) {
            if (!valueList.contains(((DiscreteAttribute) instance.getAttribute(attributeIndex)).getValue())) {
                valueList.add(((DiscreteAttribute) instance.getAttribute(attributeIndex)).getValue());
            }
        }
        return valueList;
    }
    
    /**
     * Calculates the mean of a single attribute for this instance list (m_i). If the attribute is discrete, the maximum
     * occurring value for that attribute is returned. If the attribute is continuous, the mean value of the values of
     * all instances are returned.
     *
     * @param index Index of the attribute.
     * @return The mean value of the instances as an attribute.
     */
    private Attribute attributeAverage(int index) {
        if (list.get(0).getAttribute(index) instanceof DiscreteAttribute) {
            ArrayList<String> values = new ArrayList<String>();
            for (Instance instance : list) {
                values.add(((DiscreteAttribute) instance.getAttribute(index)).getValue());
            }
            return new DiscreteAttribute(Model.getMaximum(values));
        } else {
            if (list.get(0).getAttribute(index) instanceof ContinuousAttribute) {
                double sum = 0.0;
                for (Instance instance : list) {
                    sum += ((ContinuousAttribute) instance.getAttribute(index)).getValue();
                }
                return new ContinuousAttribute(sum / list.size());
            } else {
                return null;
            }
        }
    }

    /**
     * Calculates the mean of a single attribute for this instance list (m_i).
     *
     * @param index Index of the attribute.
     * @return The mean value of the instances as an attribute.
     */
    private ArrayList<Double> continuousAttributeAverage(int index) {
        if (list.get(0).getAttribute(index) instanceof DiscreteIndexedAttribute) {
            int maxIndexSize = ((DiscreteIndexedAttribute) list.get(0).getAttribute(index)).getMaxIndex();
            ArrayList<Double> values = new ArrayList<Double>();
            for (int i = 0; i < maxIndexSize; i++) {
                values.add(0.0);
            }
            for (Instance instance : list) {
                int valueIndex = ((DiscreteIndexedAttribute) instance.getAttribute(index)).getIndex();
                values.set(valueIndex, values.get(valueIndex) + 1);
            }
            for (int i = 0; i < values.size(); i++) {
                values.set(i, values.get(i) / list.size());
            }
            return values;
        } else {
            if (list.get(0).getAttribute(index) instanceof ContinuousAttribute) {
                double sum = 0.0;
                for (Instance instance : list) {
                    sum += ((ContinuousAttribute) instance.getAttribute(index)).getValue();
                }
                ArrayList<Double> values = new ArrayList<>();
                values.add(sum / list.size());
                return values;
            } else {
                return null;
            }
        }
    }

    /**
     * Calculates the standard deviation of a single attribute for this instance list (m_i). If the attribute is discrete,
     * null returned. If the attribute is continuous, the standard deviation  of the values all instances are returned.
     *
     * @param index Index of the attribute.
     * @return The standard deviation of the instances as an attribute.
     */
    private Attribute attributeStandardDeviation(int index) {
        if (list.get(0).getAttribute(index) instanceof ContinuousAttribute) {
            double average, sum = 0.0;
            for (Instance instance : list) {
                sum += ((ContinuousAttribute) instance.getAttribute(index)).getValue();
            }
            average = sum / list.size();
            sum = 0.0;
            for (Instance instance : list) {
                sum += Math.pow(((ContinuousAttribute) instance.getAttribute(index)).getValue() - average, 2);
            }
            return new ContinuousAttribute(Math.sqrt(sum / (list.size() - 1)));
        } else {
            return null;
        }
    }

    /**
     * Calculates the standard deviation of a single continuous attribute for this instance list (m_i).
     *
     * @param index Index of the attribute.
     * @return The standard deviation of the instances as an attribute.
     */
    private ArrayList<Double> continuousAttributeStandardDeviation(int index) {
        if (list.get(0).getAttribute(index) instanceof DiscreteIndexedAttribute) {
            int maxIndexSize = ((DiscreteIndexedAttribute) list.get(0).getAttribute(index)).getMaxIndex();
            ArrayList<Double> averages = new ArrayList<Double>();
            for (int i = 0; i < maxIndexSize; i++) {
                averages.add(0.0);
            }
            for (Instance instance : list) {
                int valueIndex = ((DiscreteIndexedAttribute) instance.getAttribute(index)).getIndex();
                averages.set(valueIndex, averages.get(valueIndex) + 1);
            }
            for (int i = 0; i < averages.size(); i++) {
                averages.set(i, averages.get(i) / list.size());
            }
            ArrayList<Double> values = new ArrayList<Double>();
            for (int i = 0; i < maxIndexSize; i++) {
                values.add(0.0);
            }
            for (Instance instance : list) {
                int valueIndex = ((DiscreteIndexedAttribute) instance.getAttribute(index)).getIndex();
                for (int i = 0; i < maxIndexSize; i++) {
                    if (i == valueIndex) {
                        values.set(i, values.get(i) + Math.pow(1 - averages.get(i), 2));
                    } else {
                        values.set(i, values.get(i) + Math.pow(averages.get(i), 2));
                    }
                }
            }
            for (int i = 0; i < values.size(); i++) {
                values.set(i, Math.sqrt(values.get(i) / (list.size() - 1)));
            }
            return values;
        } else {
            if (list.get(0).getAttribute(index) instanceof ContinuousAttribute) {
                double average, sum = 0.0;
                for (Instance instance : list) {
                    sum += ((ContinuousAttribute) instance.getAttribute(index)).getValue();
                }
                average = sum / list.size();
                sum = 0.0;
                for (Instance instance : list) {
                    sum += Math.pow(((ContinuousAttribute) instance.getAttribute(index)).getValue() - average, 2);
                }
                ArrayList<Double> result = new ArrayList<>();
                result.add(Math.sqrt(sum / (list.size() - 1)));
                return result;
            } else {
                return null;
            }
        }
    }

    /**
     * The attributeDistribution method takes an index as an input and if the attribute of the instance at given index is
     * discrete, it returns the distribution of the attributes of that instance.
     *
     * @param index Index of the attribute.
     * @return Distribution of the attribute.
     */
    public DiscreteDistribution attributeDistribution(int index) {
        DiscreteDistribution distribution = new DiscreteDistribution();
        if (list.get(0).getAttribute(index) instanceof DiscreteAttribute) {
            for (Instance instance : list) {
                distribution.addItem(((DiscreteAttribute) instance.getAttribute(index)).getValue());
            }
        }
        return distribution;
    }

    /**
     * The attributeClassDistribution method takes an attribute index as an input. It loops through the instances, gets
     * the corresponding value of given attribute index and adds the class label of that instance to the discrete distributions list.
     *
     * @param attributeIndex Index of the attribute.
     * @return Distribution of the class labels.
     */
    public ArrayList<DiscreteDistribution> attributeClassDistribution(int attributeIndex) {
        ArrayList<DiscreteDistribution> distributions = new ArrayList<DiscreteDistribution>();
        ArrayList<String> valueList = getAttributeValueList(attributeIndex);
        for (String ignored : valueList) {
            distributions.add(new DiscreteDistribution());
        }
        for (Instance instance : list) {
            distributions.get(valueList.indexOf(((DiscreteAttribute) instance.getAttribute(attributeIndex)).getValue())).addItem(instance.getClassLabel());
        }
        return distributions;
    }

    /**
     * The discreteIndexedAttributeClassDistribution method takes an attribute index and an attribute value as inputs.
     * It loops through the instances, gets the corresponding value of given attribute index and given attribute value.
     * Then, adds the class label of that instance to the discrete indexed distributions list.
     *
     * @param attributeIndex Index of the attribute.
     * @param attributeValue Value of the attribute.
     * @return Distribution of the class labels.
     */
    public DiscreteDistribution discreteIndexedAttributeClassDistribution(int attributeIndex, int attributeValue) {
        DiscreteDistribution distribution = new DiscreteDistribution();
        for (Instance instance : list) {
            if (((DiscreteIndexedAttribute) instance.getAttribute(attributeIndex)).getIndex() == attributeValue) {
                distribution.addItem(instance.getClassLabel());
            }
        }
        return distribution;
    }

    /**
     * The classDistribution method returns the distribution of all the class labels of instances.
     *
     * @return Distribution of the class labels.
     */
    public DiscreteDistribution classDistribution() {
        DiscreteDistribution distribution = new DiscreteDistribution();
        for (Instance instance : list) {
            distribution.addItem(instance.getClassLabel());
        }
        return distribution;
    }

    /**
     * The allAttributesDistribution method returns the distributions of all the attributes of instances.
     *
     * @return Distributions of all the attributes of instances.
     */
    public ArrayList<DiscreteDistribution> allAttributesDistribution() {
        ArrayList<DiscreteDistribution> distributions = new ArrayList<DiscreteDistribution>();
        for (int i = 0; i < list.get(0).attributeSize(); i++) {
            distributions.add(attributeDistribution(i));
        }
        return distributions;
    }

    /**
     * Returns the mean of all the attributes for instances in the list.
     *
     * @return Mean of all the attributes for instances in the list.
     */
    public Instance average() {
        Instance result = new Instance(list.get(0).getClassLabel());
        for (int i = 0; i < list.get(0).attributeSize(); i++) {
            result.addAttribute(attributeAverage(i));
        }
        return result;
    }

    /**
     * Calculates mean of the attributes of instances.
     *
     * @return Mean of the attributes of instances.
     */
    public ArrayList<Double> continuousAttributeAverage() {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < list.get(0).attributeSize(); i++) {
            result.addAll(continuousAttributeAverage(i));
        }
        return result;
    }

    /**
     * Returns the standard deviation of attributes for instances.
     *
     * @return Standard deviation of attributes for instances.
     */
    public Instance standardDeviation() {
        Instance result = new Instance(list.get(0).getClassLabel());
        for (int i = 0; i < list.get(0).attributeSize(); i++) {
            result.addAttribute(attributeStandardDeviation(i));
        }
        return result;
    }

    /**
     * Returns the standard deviation of continuous attributes for instances.
     *
     * @return Standard deviation of continuous attributes for instances.
     */
    public ArrayList<Double> continuousAttributeStandardDeviation() {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < list.get(0).attributeSize(); i++) {
            result.addAll(continuousAttributeStandardDeviation(i));
        }
        return result;
    }

    /**
     * Calculates a covariance {@link Matrix} by using an average {@link Vector}.
     *
     * @param average Vector input.
     * @return Covariance {@link Matrix}.
     */
    public Matrix covariance(Vector average) {
        double mi, mj, xi, xj;
        Matrix result = new Matrix(list.get(0).continuousAttributeSize(), list.get(0).continuousAttributeSize());
        for (Instance instance : list) {
            ArrayList<Double> continuousAttributes = instance.continuousAttributes();
            for (int i = 0; i < instance.continuousAttributeSize(); i++) {
                xi = continuousAttributes.get(i);
                mi = average.getValue(i);
                for (int j = 0; j < instance.continuousAttributeSize(); j++) {
                    xj = continuousAttributes.get(j);
                    mj = average.getValue(j);
                    result.addValue(i, j, (xi - mi) * (xj - mj));
                }
            }
        }
        result.divideByConstant(list.size() - 1);
        return result;
    }

    /**
     * Accessor for the instances.
     *
     * @return Instances.
     */
    public ArrayList<Instance> getInstances() {
        return list;
    }

}
