package Classification.InstanceList;

import Classification.Attribute.*;
import Classification.Classifier.Classifier;
import Classification.DataSet.DataDefinition;
import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.Instance.InstanceClassComparator;
import Classification.Instance.InstanceComparator;
import Math.DiscreteDistribution;
import Math.Vector;
import Math.Matrix;
import Sampling.Bootstrap;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class InstanceList implements Serializable{

    protected ArrayList<Instance> list;

    /**
     * Empty constructor for an instance list. Initializes the instance list with zero instances.
     */
    public InstanceList(){
        list = new ArrayList<Instance>();
    }

    /**
     * Constructor for an instance list with a given data definition, data file and a separator character. Each instance
     * must be stored in a separate line separated with the character separator. The last item must be the class label.
     * The function reads the file line by line and for each line; depending on the data definition, that is, type of
     * the attributes, adds discrete and continuous attributes to a new instance. For example, given the data set file
     *
     * red;1;0.4;true
     * green;-1;0.8;true
     * blue;3;1.3;false
     *
     * where the first attribute is a discrete attribute, second and third attributes are continuous attributes, the
     * fourth item is the class label.
     * @param definition Data definition of the data set.
     * @param separator Separator character which separates the attribute values in the data file.
     * @param fileName Name of the data set file.
     */
    public InstanceList(DataDefinition definition, String separator, String fileName){
        Instance current;
        String line;
        list = new ArrayList<Instance>();
        try{
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF8"));
            line = br.readLine();
            while (line != null){
                String[] attributeList = line.split(separator);
                if (attributeList.length == definition.attributeCount() + 1){
                    current = new Instance(attributeList[attributeList.length - 1]);
                    for (int i = 0; i < attributeList.length - 1; i++){
                        switch (definition.getAttributeType(i)){
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
        }
        catch (FileNotFoundException fileNotFoundException){
            System.out.println("Dataset with fileName " + fileName + " not found");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Mutator for the list variable.
     * @param list New list for the list variable.
     */
    public InstanceList(ArrayList<Instance> list){
        this.list = list;
    }

    /**
     * Adds instance to the instance list.
     * @param instance Instance to be added.
     */
    public void add(Instance instance){
        list.add(instance);
    }

    /**
     * Adds a list of instances to the current instance list.
     * @param instanceList List of instances to be added.
     */
    public void addAll(ArrayList<Instance> instanceList){
        list.addAll(instanceList);
    }

    /**
     * Returns size of the instance list.
     * @return Size of the instance list.
     */
    public int size(){
        return list.size();
    }

    /**
     * Accessor for a single instance with the given index.
     * @param index Index of the instance.
     * @return Instance with index 'index'.
     */
    public Instance get(int index){
        return list.get(index);
    }

    /**
     * Sorts attribute list according to the attribute with index 'attributeIndex'.
     * @param attributeIndex index of the attribute.
     */
    public void sort(int attributeIndex){
        InstanceComparator comparator = new InstanceComparator(attributeIndex);
        Collections.sort(list, comparator);
    }

    public void sort(){
        Collections.sort(list, new InstanceClassComparator());
    }

    /**
     * Shuffles the instance list.
     */
    public void shuffle(int seed){
        Collections.shuffle(list, new Random(seed));
    }

    /**
     * Creates a bootstrap sample from the current instance list.
     * @param seed To create a different bootstrap sample, we need a new seed for each sample.
     * @return Bootstrap sample
     */
    public Bootstrap bootstrap(int seed){
        return new Bootstrap<Instance>(list, seed);
    }

    /**
     * Extracts the class labels of each instance in the instance list and returns them in an array of string.
     * @return An array list of class labels.
     */
    public ArrayList<String> getClassLabels(){
        ArrayList<String> classLabels = new ArrayList<String>();
        for (Instance instance:list){
            classLabels.add(instance.getClassLabel());
        }
        return classLabels;
    }

    /**
     * Extracts the class labels of each instance in the instance list and returns them as a set.
     * @return An array list of distinct class labels.
     */
    public ArrayList<String> getDistinctClassLabels(){
        ArrayList<String> classLabels = new ArrayList<String>();
        for (Instance instance:list){
            if (!classLabels.contains(instance.getClassLabel())){
                classLabels.add(instance.getClassLabel());
            }
        }
        return classLabels;
    }

    /**
     * Extracts the possible class labels of each instance in the instance list and returns them as a set.
     * @return An array list of distinct class labels.
     */
    public ArrayList<String> getUnionOfPossibleClassLabels(){
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
                if (!possibleClassLabels.contains(instance.getClassLabel())){
                    possibleClassLabels.add(instance.getClassLabel());
                }
            }
        }
        return possibleClassLabels;
    }

    /**
     * Divides the instances in the instance list into partitions so that all instances of a class are grouped in a
     * single partition.
     * @return Groups of instances according to their class labels.
     */
    public Partition divideIntoClasses(){
        ArrayList<String> classLabels = getDistinctClassLabels();
        Partition result = new Partition();
        for (String classLabel : classLabels)
            result.add(new InstanceListOfSameClass(classLabel));
        for (Instance instance:list){
            result.get(classLabels.indexOf(instance.getClassLabel())).add(instance);
        }
        return result;
    }

    /**
     * Extracts distinct discrete values of a given attribute as an array of strings.
     * @param attributeIndex Index of the discrete attribute
     * @return An array of distinct values of a discrete attribute.
     */
    public ArrayList<String> getAttributeValueList(int attributeIndex){
        ArrayList<String> valueList = new ArrayList<String>();
        for (Instance instance:list){
            if (!valueList.contains(((DiscreteAttribute)instance.getAttribute(attributeIndex)).getValue())){
                valueList.add(((DiscreteAttribute)instance.getAttribute(attributeIndex)).getValue());
            }
        }
        return valueList;
    }

    /**
     * Creates a stratified partition of the current instance list. In a stratified partition, the percentage of each
     * class is preserved. For example, let say there are three classes in the instance list, and let the percentages of
     * these classes be %20, %30, and %50; then the percentages of these classes in the stratified partitions are the
     * same, that is, %20, %30, and %50.
     * @param ratio Ratio of the stratified partition. Ratio is between 0 and 1. If the ratio is 0.2, then 20 percent
     *              of the instances are put in the first group, 80 percent of the instances are put in the second group
     * @return 2 group stratified partition of the instances in this instance list.
     */
    public Partition stratifiedPartition(double ratio, Random random){
        int[] counts;
        DiscreteDistribution distribution;
        Partition partition = new Partition();
        partition.add(new InstanceList());
        partition.add(new InstanceList());
        distribution = classDistribution();
        counts = new int[distribution.size()];
        ArrayList<Integer> randomArray = new ArrayList<Integer>();
        for (int i = 0; i < size(); i++)
            randomArray.add(i);
        Collections.shuffle(randomArray, random);
        for (int i = 0; i < size(); i++){
            Instance instance = list.get(randomArray.get(i));
            int classIndex = distribution.getIndex(instance.getClassLabel());
            if (counts[classIndex] < size() * ratio * distribution.getProbability(instance.getClassLabel())){
                partition.get(0).add(instance);
            } else {
                partition.get(1).add(instance);
            }
            counts[classIndex]++;
        }
        return partition;
    }

    /**
     * Creates a partition of the current instance list.
     * @param ratio Ratio of the partition. Ratio is between 0 and 1. If the ratio is 0.2, then 20 percent
     *              of the instances are put in the first group, 80 percent of the instances are put in the second group
     * @return 2 group partition of the instances in this instance list.
     */
    public Partition partition(double ratio, Random random){
        Partition partition = new Partition();
        partition.add(new InstanceList());
        partition.add(new InstanceList());
        Collections.shuffle(list, random);
        for (int i = 0; i < size(); i++){
            Instance instance = list.get(i);
            if (i < size() * ratio){
                partition.get(0).add(instance);
            } else {
                partition.get(1).add(instance);
            }
        }
        return partition;
    }

    /**
     * Creates a partition depending on the distinct values of a discrete attribute. If the discrete attribute has 4
     * distinct values, the resulting partition will have 4 groups, where each group contain instance whose
     * values of that discrete attribute are the same.
     * @param attributeIndex Index of the discrete attribute.
     * @return L groups of instances, where L is the number of distinct values of the discrete attribute with index
     * attributeIndex.
     */
    public Partition divideWithRespectToAttribute(int attributeIndex){
        ArrayList<String> valueList = getAttributeValueList(attributeIndex);
        Partition result = new Partition();
        for (String value:valueList){
            result.add(new InstanceList());
        }
        for (Instance instance:list){
            result.get(valueList.indexOf(((DiscreteAttribute) instance.getAttribute(attributeIndex)).getValue())).add(instance);
        }
        return result;
    }

    public Partition divideWithRespectToIndexedAttribute(int attributeIndex, int attributeValue){
        Partition result = new Partition();
        result.add(new InstanceList());
        result.add(new InstanceList());
        for (Instance instance:list){
            if (((DiscreteIndexedAttribute)instance.getAttribute(attributeIndex)).getIndex() == attributeValue){
                result.get(0).add(instance);
            } else {
                result.get(1).add(instance);
            }
        }
        return result;
    }

    /**
     * Creates a two group partition depending on the values of a continuous attribute. If the value of the attribute is
     * less than splitValue, the instance is forwarded to the first group, else it is forwarded to the second group.
     * @param attributeIndex Index of the continuous attribute
     * @param splitValue Threshold to divide instances
     * @return Two groups of instances as a partition.
     */
    public Partition divideWithRespectToAttribute(int attributeIndex, double splitValue){
        Partition result = new Partition();
        result.add(new InstanceList());
        result.add(new InstanceList());
        for (Instance instance:list){
            if (((ContinuousAttribute)instance.getAttribute(attributeIndex)).getValue() <= splitValue){
                result.get(0).add(instance);
            } else {
                result.get(1).add(instance);
            }
        }
        return result;
    }

    /**
     * Calculates the mean of a single attribute for this instance list (m_i). If the attribute is discrete, the maximum
     * occurring value for that attribute is returned. If the attribute is continuous, the mean value of the values of
     * all instances are returned.
     * @param index Index of the attribute
     * @return The mean value of the instances as an attribute.
     */
    private Attribute attributeAverage(int index){
        if (list.get(0).getAttribute(index) instanceof DiscreteAttribute){
            ArrayList<String> values = new ArrayList<String>();
            for (Instance instance:list){
                values.add(((DiscreteAttribute)instance.getAttribute(index)).getValue());
            }
            return new DiscreteAttribute(Classifier.getMaximum(values));
        } else {
            if (list.get(0).getAttribute(index) instanceof ContinuousAttribute){
                double sum = 0.0;
                for (Instance instance:list){
                    sum += ((ContinuousAttribute)instance.getAttribute(index)).getValue();
                }
                return new ContinuousAttribute(sum / list.size());
            } else {
                return null;
            }
        }
    }

    private ArrayList<Double> continuousAttributeAverage(int index){
        if (list.get(0).getAttribute(index) instanceof DiscreteIndexedAttribute){
            int maxIndexSize = ((DiscreteIndexedAttribute) list.get(0).getAttribute(index)).getMaxIndex();
            ArrayList<Double> values = new ArrayList<Double>();
            for (int i = 0; i < maxIndexSize; i++){
                values.add(0.0);
            }
            for (Instance instance:list){
                int valueIndex = ((DiscreteIndexedAttribute) instance.getAttribute(index)).getIndex();
                values.set(valueIndex, values.get(valueIndex) + 1);
            }
            for (int i = 0; i < values.size(); i++){
                values.set(i, values.get(i) / list.size());
            }
            return values;
        } else {
            if (list.get(0).getAttribute(index) instanceof ContinuousAttribute){
                double sum = 0.0;
                for (Instance instance:list){
                    sum += ((ContinuousAttribute)instance.getAttribute(index)).getValue();
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
     * @param index Index of the attribute
     * @return The standard deviation of the instances as an attribute.
     */
    private Attribute attributeStandardDeviation(int index){
        if (list.get(0).getAttribute(index) instanceof ContinuousAttribute){
            double average, sum = 0.0;
            for (Instance instance:list){
                sum += ((ContinuousAttribute)instance.getAttribute(index)).getValue();
            }
            average = sum / list.size();
            sum = 0.0;
            for (Instance instance:list){
                sum += Math.pow(((ContinuousAttribute) instance.getAttribute(index)).getValue() - average, 2);
            }
            return new ContinuousAttribute(Math.sqrt(sum / (list.size() - 1)));
        } else {
            return null;
        }
    }

    private ArrayList<Double> continuousAttributeStandardDeviation(int index){
        if (list.get(0).getAttribute(index) instanceof DiscreteIndexedAttribute){
            int maxIndexSize = ((DiscreteIndexedAttribute) list.get(0).getAttribute(index)).getMaxIndex();
            ArrayList<Double> averages = new ArrayList<Double>();
            for (int i = 0; i < maxIndexSize; i++){
                averages.add(0.0);
            }
            for (Instance instance:list){
                int valueIndex = ((DiscreteIndexedAttribute) instance.getAttribute(index)).getIndex();
                averages.set(valueIndex, averages.get(valueIndex) + 1);
            }
            for (int i = 0; i < averages.size(); i++){
                averages.set(i, averages.get(i) / list.size());
            }
            ArrayList<Double> values = new ArrayList<Double>();
            for (int i = 0; i < maxIndexSize; i++){
                values.add(0.0);
            }
            for (Instance instance:list){
                int valueIndex = ((DiscreteIndexedAttribute) instance.getAttribute(index)).getIndex();
                for (int i = 0; i < maxIndexSize; i++){
                    if (i == valueIndex){
                        values.set(i, values.get(i) + Math.pow(1 - averages.get(i), 2));
                    } else {
                        values.set(i, values.get(i) + Math.pow(averages.get(i), 2));
                    }
                }
            }
            for (int i = 0; i < values.size(); i++){
                values.set(i, Math.sqrt(values.get(i) / (list.size() - 1)));
            }
            return values;
        } else {
            if (list.get(0).getAttribute(index) instanceof ContinuousAttribute){
                double average, sum = 0.0;
                for (Instance instance:list){
                    sum += ((ContinuousAttribute)instance.getAttribute(index)).getValue();
                }
                average = sum / list.size();
                sum = 0.0;
                for (Instance instance:list){
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

    public DiscreteDistribution attributeDistribution(int index){
        DiscreteDistribution distribution = new DiscreteDistribution();
        if (list.get(0).getAttribute(index) instanceof DiscreteAttribute){
            for (Instance instance:list){
                distribution.addItem(((DiscreteAttribute)instance.getAttribute(index)).getValue());
            }
        }
        return distribution;
    }

    public ArrayList<DiscreteDistribution> attributeClassDistribution(int attributeIndex){
        ArrayList<DiscreteDistribution> distributions = new ArrayList<DiscreteDistribution>();
        ArrayList<String> valueList = getAttributeValueList(attributeIndex);
        for (String ignored :valueList){
            distributions.add(new DiscreteDistribution());
        }
        for (Instance instance:list){
            distributions.get(valueList.indexOf(((DiscreteAttribute)instance.getAttribute(attributeIndex)).getValue())).addItem(instance.getClassLabel());
        }
        return distributions;
    }

    public DiscreteDistribution discreteIndexedAttributeClassDistribution(int attributeIndex, int attributeValue){
        DiscreteDistribution distribution = new DiscreteDistribution();
        for (Instance instance:list){
            if (((DiscreteIndexedAttribute)instance.getAttribute(attributeIndex)).getIndex() == attributeValue){
                distribution.addItem(instance.getClassLabel());
            }
        }
        return distribution;
    }

    public DiscreteDistribution classDistribution(){
        DiscreteDistribution distribution = new DiscreteDistribution();
        for (Instance instance:list){
            distribution.addItem(instance.getClassLabel());
        }
        return distribution;
    }

    public ArrayList<DiscreteDistribution> allAttributesDistribution(){
        ArrayList<DiscreteDistribution> distributions = new ArrayList<DiscreteDistribution>();
        for (int i = 0; i < list.get(0).attributeSize(); i++){
            distributions.add(attributeDistribution(i));
        }
        return distributions;
    }

    public Instance average(){
        Instance result = new Instance(list.get(0).getClassLabel());
        for (int i = 0; i < list.get(0).attributeSize(); i++){
            result.addAttribute(attributeAverage(i));
        }
        return result;
    }

    public ArrayList<Double> continuousAttributeAverage(){
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < list.get(0).attributeSize(); i++){
            result.addAll(continuousAttributeAverage(i));
        }
        return result;
    }

    public Instance standardDeviation(){
        Instance result = new Instance(list.get(0).getClassLabel());
        for (int i = 0; i < list.get(0).attributeSize(); i++){
            result.addAttribute(attributeStandardDeviation(i));
        }
        return result;
    }

    public ArrayList<Double> continuousAttributeStandardDeviation(){
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < list.get(0).attributeSize(); i++){
            result.addAll(continuousAttributeStandardDeviation(i));
        }
        return result;
    }

    public Matrix covariance(Vector average){
        double mi, mj, xi, xj;
        Matrix result = new Matrix(list.get(0).continuousAttributeSize(), list.get(0).continuousAttributeSize());
        for (Instance instance:list){
            ArrayList<Double> continuousAttributes = instance.continuousAttributes();
            for (int i = 0; i < instance.continuousAttributeSize(); i++){
                xi = continuousAttributes.get(i);
                mi = average.getValue(i);
                for (int j = 0; j < instance.continuousAttributeSize(); j++){
                    xj = continuousAttributes.get(j);
                    mj = average.getValue(j);
                    result.addValue(i, j, (xi - mi) * (xj - mj));
                }
            }
        }
        result.divideByConstant(list.size() - 1);
        return result;
    }

    public ArrayList<Instance> getInstances(){
        return list;
    }

}
