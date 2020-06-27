package Classification.InstanceList;

import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Attribute.DiscreteIndexedAttribute;
import Classification.Instance.Instance;
import Math.DiscreteDistribution;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class Partition {

    private ArrayList<InstanceList> multiList;

    /**
     * Divides the instances in the instance list into partitions so that all instances of a class are grouped in a
     * single partition.
     * @param instanceList Instance list for which partition will be created.
     */
    public Partition(InstanceList instanceList) {
        this();
        ArrayList<String> classLabels = instanceList.getDistinctClassLabels();
        for (String classLabel : classLabels)
            add(new InstanceListOfSameClass(classLabel));
        for (Instance instance : instanceList.getInstances()) {
            get(classLabels.indexOf(instance.getClassLabel())).add(instance);
        }
    }

    /**
     * Creates a stratified partition of the current instance list. In a stratified partition, the percentage of each
     * class is preserved. For example, let's say there are three classes in the instance list, and let the percentages of
     * these classes be %20, %30, and %50; then the percentages of these classes in the stratified partitions are the
     * same, that is, %20, %30, and %50.
     *
     * @param instanceList Instance list for which partition will be created.
     * @param ratio Ratio of the stratified partition. Ratio is between 0 and 1. If the ratio is 0.2, then 20 percent
     *              of the instances are put in the first group, 80 percent of the instances are put in the second group.
     * @param random random is used as a random number.
     * @param stratified If true, stratified partition is obtained.
     */
    public Partition (InstanceList instanceList, double ratio, Random random, boolean stratified) {
        this();
        add(new InstanceList());
        add(new InstanceList());
        if (stratified){
            int[] counts;
            DiscreteDistribution distribution;
            distribution = instanceList.classDistribution();
            counts = new int[distribution.size()];
            ArrayList<Integer> randomArray = new ArrayList<Integer>();
            for (int i = 0; i < instanceList.size(); i++)
                randomArray.add(i);
            Collections.shuffle(randomArray, random);
            for (int i = 0; i < instanceList.size(); i++) {
                Instance instance = instanceList.get(randomArray.get(i));
                int classIndex = distribution.getIndex(instance.getClassLabel());
                if (counts[classIndex] < instanceList.size() * ratio * distribution.getProbability(instance.getClassLabel())) {
                    get(0).add(instance);
                } else {
                    get(1).add(instance);
                }
                counts[classIndex]++;
            }
        } else {
            instanceList.shuffle(random);
            for (int i = 0; i < instanceList.size(); i++) {
                Instance instance = instanceList.get(i);
                if (i < instanceList.size() * ratio) {
                    get(0).add(instance);
                } else {
                    get(1).add(instance);
                }
            }
        }
    }

    /**
     * Creates a partition depending on the distinct values of a discrete attribute. If the discrete attribute has 4
     * distinct values, the resulting partition will have 4 groups, where each group contain instance whose
     * values of that discrete attribute are the same.
     *
     * @param instanceList Instance list for which partition will be created.
     * @param attributeIndex Index of the discrete attribute.
     */
    public Partition(InstanceList instanceList, int attributeIndex) {
        this();
        ArrayList<String> valueList = instanceList.getAttributeValueList(attributeIndex);
        for (String value : valueList) {
            add(new InstanceList());
        }
        for (Instance instance : instanceList.getInstances()) {
            get(valueList.indexOf(((DiscreteAttribute) instance.getAttribute(attributeIndex)).getValue())).add(instance);
        }
    }

    /**
     * Creates a partition depending on the distinct values of a discrete indexed attribute.
     *
     * @param instanceList Instance list for which partition will be created.
     * @param attributeIndex Index of the discrete indexed attribute.
     * @param attributeValue Value of the attribute.
     */
    public Partition(InstanceList instanceList, int attributeIndex, int attributeValue) {
        this();
        add(new InstanceList());
        add(new InstanceList());
        for (Instance instance : instanceList.getInstances()) {
            if (((DiscreteIndexedAttribute) instance.getAttribute(attributeIndex)).getIndex() == attributeValue) {
                get(0).add(instance);
            } else {
                get(1).add(instance);
            }
        }
    }

    /**
     * Creates a two group partition depending on the values of a continuous attribute. If the value of the attribute is
     * less than splitValue, the instance is forwarded to the first group, else it is forwarded to the second group.
     *
     * @param instanceList Instance list for which partition will be created.
     * @param attributeIndex Index of the continuous attribute
     * @param splitValue     Threshold to divide instances
     */
    public Partition(InstanceList instanceList, int attributeIndex, double splitValue) {
        this();
        add(new InstanceList());
        add(new InstanceList());
        for (Instance instance : instanceList.getInstances()) {
            if (((ContinuousAttribute) instance.getAttribute(attributeIndex)).getValue() <= splitValue) {
                get(0).add(instance);
            } else {
                get(1).add(instance);
            }
        }
    }

    /**
     * Constructor for generating a partition.
     */
    public Partition() {
        multiList = new ArrayList<InstanceList>();
    }

    /**
     * Adds given instance list to the list of instance lists.
     *
     * @param list Instance list to add.
     */
    public void add(InstanceList list) {
        multiList.add(list);
    }

    /**
     * Returns the size of the list of instance lists.
     *
     * @return The size of the list of instance lists.
     */
    public int size() {
        return multiList.size();
    }

    /**
     * Returns the corresponding instance list at given index of list of instance lists.
     *
     * @param index Index of the instance list.
     * @return Instance list at given index of list of instance lists.
     */
    public InstanceList get(int index) {
        return multiList.get(index);
    }

    /**
     * Returns the instances of the items at the list of instance lists.
     *
     * @return Instances of the items at the list of instance lists.
     */
    public ArrayList<Instance>[] getLists() {
        ArrayList<Instance>[] result = new ArrayList[multiList.size()];
        for (int i = 0; i < multiList.size(); i++) {
            result[i] = multiList.get(i).getInstances();
        }
        return result;
    }
}
