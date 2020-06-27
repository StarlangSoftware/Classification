package Classification.DataSet;

import Classification.Attribute.*;
import Classification.FeatureSelection.FeatureSubSet;
import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.Partition;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class DataSet {

    private InstanceList instances;
    private DataDefinition definition;

    /**
     * Constructor for generating a new {@link DataSet}.
     */
    public DataSet() {
        definition = null;
        instances = new InstanceList();
    }

    /**
     * Constructor for generating a new {@link DataSet} with given {@link DataDefinition}.
     *
     * @param definition Data definition of the data set.
     */
    public DataSet(DataDefinition definition) {
        this.definition = definition;
        instances = new InstanceList();
    }

    /**
     * Constructor for generating a new {@link DataSet} from given {@link File}.
     *
     * @param file {@link File} to generate {@link DataSet} from.
     */
    public DataSet(File file) {
        Instance instance;
        int i = 0;
        instances = new InstanceList();
        definition = new DataDefinition();
        try {
            Scanner input = new Scanner(file);
            while (input.hasNext()) {
                String instanceText = input.nextLine();
                String[] attributes = instanceText.split(",");
                if (i == 0) {
                    for (int j = 0; j < attributes.length - 1; j++) {
                        try {
                            Double.parseDouble(attributes[j]);
                            definition.addAttribute(AttributeType.CONTINUOUS);
                        } catch (NumberFormatException e) {
                            definition.addAttribute(AttributeType.DISCRETE);
                        }
                    }
                } else {
                    if (attributes.length != definition.attributeCount() + 1) {
                        continue;
                    }
                }
                if (!attributes[attributes.length - 1].contains(";")) {
                    instance = new Instance(attributes[attributes.length - 1]);
                } else {
                    String[] labels = attributes[attributes.length - 1].split(";");
                    instance = new CompositeInstance(labels);
                }
                for (int j = 0; j < attributes.length - 1; j++) {
                    switch (definition.getAttributeType(j)) {
                        case CONTINUOUS:
                            try {
                                instance.addAttribute(new ContinuousAttribute(Double.parseDouble(attributes[j])));
                            } catch (NumberFormatException e) {
                            }
                            break;
                        case DISCRETE:
                            instance.addAttribute(new DiscreteAttribute(attributes[j]));
                            break;
                    }
                }
                if (instance.attributeSize() == definition.attributeCount()) {
                    instances.add(instance);
                }
                i++;
            }
            input.close();
        } catch (FileNotFoundException e) {
            System.out.println(e.toString());
        }
    }

    /**
     * Constructor for generating a new {@link DataSet} with a {@link DataDefinition}, from a {@link File} by using a separator.
     *
     * @param definition Data definition of the data set.
     * @param separator  Separator character which separates the attribute values in the data file.
     * @param fileName   Name of the data set file.
     */

    public DataSet(DataDefinition definition, String separator, String fileName) {
        this.definition = definition;
        instances = new InstanceList(definition, separator, fileName);
    }

    /**
     * Checks the correctness of the attribute type, for instance, if the attribute of given instance is a Binary attribute,
     * and the attribute type of the corresponding item of the data definition is also a Binary attribute, it then
     * returns true, and false otherwise.
     *
     * @param instance {@link Instance} to checks the attribute type.
     * @return true if attribute types of given {@link Instance} and data definition matches.
     */
    private boolean checkDefinition(Instance instance) {
        for (int i = 0; i < instance.attributeSize(); i++) {
            if (instance.getAttribute(i) instanceof BinaryAttribute) {
                if (definition.getAttributeType(i) != AttributeType.BINARY)
                    return false;
            } else {
                if (instance.getAttribute(i) instanceof DiscreteIndexedAttribute) {
                    if (definition.getAttributeType(i) != AttributeType.DISCRETE_INDEXED)
                        return false;
                } else {
                    if (instance.getAttribute(i) instanceof DiscreteAttribute) {
                        if (definition.getAttributeType(i) != AttributeType.DISCRETE)
                            return false;
                    } else {
                        if (instance.getAttribute(i) instanceof ContinuousAttribute) {
                            if (definition.getAttributeType(i) != AttributeType.CONTINUOUS)
                                return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    /**
     * Adds the attribute types according to given {@link Instance}. For instance, if the attribute type of given {@link Instance}
     * is a Discrete type, it than adds a discrete attribute type to the list of attribute types.
     *
     * @param instance {@link Instance} input.
     */
    private void setDefinition(Instance instance) {
        ArrayList<AttributeType> attributeTypes = new ArrayList<>();
        for (int i = 0; i < instance.attributeSize(); i++) {
            if (instance.getAttribute(i) instanceof BinaryAttribute) {
                attributeTypes.add(AttributeType.BINARY);
            } else {
                if (instance.getAttribute(i) instanceof DiscreteIndexedAttribute) {
                    attributeTypes.add(AttributeType.DISCRETE_INDEXED);
                } else {
                    if (instance.getAttribute(i) instanceof DiscreteAttribute) {
                        attributeTypes.add(AttributeType.DISCRETE);
                    } else {
                        if (instance.getAttribute(i) instanceof ContinuousAttribute) {
                            attributeTypes.add(AttributeType.CONTINUOUS);
                        }
                    }
                }
            }
        }
        definition = new DataDefinition(attributeTypes);
    }

    /**
     * Returns the size of the {@link InstanceList}.
     *
     * @return Size of the {@link InstanceList}.
     */
    public int sampleSize() {
        return instances.size();
    }

    /**
     * Returns the size of the class label distribution of {@link InstanceList}.
     *
     * @return Size of the class label distribution of {@link InstanceList}.
     */
    public int classCount() {
        return instances.classDistribution().size();
    }

    /**
     * Returns the number of attribute types at {@link DataDefinition} list.
     *
     * @return The number of attribute types at {@link DataDefinition} list.
     */
    public int attributeCount() {
        return definition.attributeCount();
    }

    /**
     * Returns the number of discrete attribute types at {@link DataDefinition} list.
     *
     * @return The number of discrete attribute types at {@link DataDefinition} list.
     */
    public int discreteAttributeCount() {
        return definition.discreteAttributeCount();
    }

    /**
     * Returns the number of continuous attribute types at {@link DataDefinition} list.
     *
     * @return The number of continuous attribute types at {@link DataDefinition} list.
     */
    public int continuousAttributeCount() {
        return definition.continuousAttributeCount();
    }

    /**
     * Returns the accumulated {@link String} of class labels of the {@link InstanceList}.
     *
     * @return The accumulated {@link String} of class labels of the {@link InstanceList}.
     */
    public String getClasses() {
        String result;
        ArrayList<String> classLabels = instances.getDistinctClassLabels();
        result = classLabels.get(0);
        for (int i = 1; i < classLabels.size(); i++) {
            result = result + ";" + classLabels.get(i);
        }
        return result;
    }

    /**
     * Returns the general information about the given data set such as the number of instances, distinct class labels,
     * attributes, discrete and continuous attributes.
     *
     * @param dataSetName Data set name.
     * @return General information about the given data set.
     */
    public String info(String dataSetName) {
        String result = "DATASET: " + dataSetName + "\n";
        result = result + "Number of instances: " + sampleSize() + "\n";
        result = result + "Number of distinct class labels: " + classCount() + "\n";
        result = result + "Number of attributes: " + attributeCount() + "\n";
        result = result + "Number of discrete attributes: " + discreteAttributeCount() + "\n";
        result = result + "Number of continuous attributes: " + continuousAttributeCount() + "\n";
        result = result + "Class labels: " + getClasses();
        return result;
    }

    /**
     * Returns a formatted String of general information aboutt he data set.
     *
     * @param dataSetName Data set name.
     * @return Formatted String of general information aboutt he data set.
     */
    public String toString(String dataSetName) {
        return String.format("%20s%15d%15d%20d%15d%15d", dataSetName, sampleSize(), classCount(), attributeCount(), discreteAttributeCount(), continuousAttributeCount());
    }

    /**
     * Adds a new instance to the {@link InstanceList}.
     *
     * @param current {@link Instance} to add.
     */
    public void addInstance(Instance current) {
        if (definition == null) {
            setDefinition(current);
            instances.add(current);
        } else {
            if (checkDefinition(current)) {
                instances.add(current);
            }
        }
    }

    /**
     * Adds all the instances of given instance list to the {@link InstanceList}.
     *
     * @param instanceList {@link InstanceList} to add instances from.
     */
    public void addInstanceList(ArrayList<Instance> instanceList) {
        for (Instance instance : instanceList) {
            addInstance(instance);
        }
    }

    /**
     * Returns the instances of {@link InstanceList}.
     *
     * @return The instances of {@link InstanceList}.
     */
    public ArrayList<Instance> getInstances() {
        return instances.getInstances();
    }

    /**
     * Returns instances of the items at the list of instance lists from the partitions.
     *
     * @return Instances of the items at the list of instance lists from the partitions.
     */
    public ArrayList<Instance>[] getClassInstances() {
        return new Partition(instances).getLists();
    }

    /**
     * Accessor for the {@link InstanceList}.
     *
     * @return The {@link InstanceList}.
     */
    public InstanceList getInstanceList() {
        return instances;
    }

    /**
     * Accessor for the data definition.
     *
     * @return The data definition.
     */
    public DataDefinition getDataDefinition() {
        return definition;
    }

    /**
     * Return a subset generated via the given {@link FeatureSubSet}.
     *
     * @param featureSubSet {@link FeatureSubSet} input.
     * @return Subset generated via the given {@link FeatureSubSet}.
     */
    public DataSet getSubSetOfFeatures(FeatureSubSet featureSubSet) {
        DataSet result = new DataSet(definition.getSubSetOfFeatures(featureSubSet));
        for (int i = 0; i < instances.size(); i++) {
            result.addInstance(instances.get(i).getSubSetOfFeatures(featureSubSet));
        }
        return result;
    }

    /**
     * Print out the instances of {@link InstanceList} as a {@link String}.
     *
     * @param outFileName File name to write the output.
     */
    public void writeToFile(String outFileName) {
        PrintWriter writer;
        try {
            writer = new PrintWriter(outFileName, "UTF-8");
            for (int i = 0; i < instances.size(); i++) {
                writer.write(instances.get(i).toString() + "\n");
            }
            writer.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }

}