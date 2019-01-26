package Classification.Instance;

import Classification.Attribute.Attribute;
import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Attribute.DiscreteIndexedAttribute;
import Classification.FeatureSelection.FeatureSubSet;
import Classification.Model.Svm.NodeList;
import Math.*;

import java.io.Serializable;
import java.util.ArrayList;

public class Instance implements Serializable {

    private String classLabel;
    private ArrayList<Attribute> attributes;

    /**
     * Constructor for a single instance. Given the attributes and class label, it generates a new instance.
     *
     * @param classLabel Class label of the instance.
     * @param attributes Attributes of the instance.
     */
    public Instance(String classLabel, ArrayList<Attribute> attributes) {
        this.classLabel = classLabel;
        this.attributes = attributes;
    }

    /**
     * Constructor for a single instance. Given the class label, it generates a new instance with 0 attributes.
     * Attributes must be added later with different addAttribute methods.
     *
     * @param classLabel Class label of the instance.
     */
    public Instance(String classLabel) {
        this.classLabel = classLabel;
        this.attributes = new ArrayList<Attribute>();
    }

    /**
     * Adds a discrete attribute with the given {@link String} value.
     *
     * @param value Value of the discrete attribute.
     */
    public void addAttribute(String value) {
        attributes.add(new DiscreteAttribute(value));
    }

    /**
     * Adds a continuous attribute with the given {@link double} value.
     *
     * @param value Value of the continuous attribute.
     */
    public void addAttribute(double value) {
        attributes.add(new ContinuousAttribute(value));
    }

    /**
     * Adds a new attribute.
     *
     * @param attribute Attribute to be added.
     */
    public void addAttribute(Attribute attribute) {
        attributes.add(attribute);
    }

    /**
     * Adds a {@link Vector} of continuous attributes.
     *
     * @param vector {@link Vector} that has the continuous attributes.
     */
    public void addVectorAttribute(Vector vector) {
        for (int i = 0; i < vector.size(); i++) {
            attributes.add(new ContinuousAttribute(vector.getValue(i)));
        }
    }

    /**
     * Removes attribute with the given index from the attributes list.
     *
     * @param index Index of the attribute to be removed.
     */
    public void removeAttribute(int index) {
        attributes.remove(index);
    }

    /**
     * Removes all the attributes from the attributes list.
     */
    public void removeAllAttributes() {
        attributes.clear();
    }

    /**
     * Accessor for a single attribute.
     *
     * @param index Index of the attribute to be accessed.
     * @return Attribute with index 'index'.
     */
    public Attribute getAttribute(int index) {
        return attributes.get(index);
    }

    /**
     * Returns the number of attributes in the attributes list.
     *
     * @return Number of attributes in the attributes list.
     */
    public int attributeSize() {
        return attributes.size();
    }

    /**
     * Returns the number of continuous and discrete indexed attributes in the attributes list.
     *
     * @return Number of continuous and discrete indexed attributes in the attributes list.
     */
    public int continuousAttributeSize() {
        int size = 0;
        for (Attribute attribute : attributes) {
            size += attribute.continuousAttributeSize();
        }
        return size;
    }

    /**
     * The continuousAttributes method creates a new {@link ArrayList} result and it adds the continuous attributes of the
     * attributes list and also it adds 1 for the discrete indexed attributes
     * .
     *
     * @return result {@link ArrayList} that has continuous and discrete indexed attributes.
     */
    public ArrayList<Double> continuousAttributes() {
        ArrayList<Double> result = new ArrayList<Double>();
        for (Attribute attribute : attributes) {
            result.addAll(attribute.continuousAttributes());
        }
        return result;
    }

    /**
     * Accessor for the class label.
     *
     * @return Class label of the instance.
     */
    public String getClassLabel() {
        return classLabel;
    }

    /**
     * Converts instance to a {@link String}.
     *
     * @return A string of attributes separated with comma character.
     */
    public String toString() {
        String result = "";
        for (Attribute attribute : attributes) {
            result = result + attribute.toString() + ",";
        }
        result = result + classLabel;
        return result;
    }

    /**
     * The getSubSetOfFeatures method takes a {@link FeatureSubSet} as an input. First it creates a result {@link Instance}
     * with the class label, and adds the attributes of the given featureSubSet to it.
     *
     * @param featureSubSet {@link FeatureSubSet} an {@link ArrayList} of indices.
     * @return result Instance.
     */
    public Instance getSubSetOfFeatures(FeatureSubSet featureSubSet) {
        Instance result = new Instance(classLabel);
        for (int i = 0; i < featureSubSet.size(); i++) {
            result.addAttribute(attributes.get(featureSubSet.get(i)));
        }
        return result;
    }

    /**
     * The toVector method returns a {@link Vector} of continuous attributes and discrete indexed attributes.
     *
     * @return {@link Vector} of continuous attributes and discrete indexed attributes.
     */
    public Vector toVector() {
        ArrayList<Double> values = new ArrayList<Double>();
        for (Attribute attribute : attributes) {
            values.addAll(attribute.continuousAttributes());
        }
        return new Vector(values);
    }

    /**
     * Returns a new {@link NodeList} with the {@link ArrayList} that has continuous and discrete indexed attributes.
     *
     * @return {@link NodeList} that has continuous and discrete indexed attributes.
     */
    public NodeList toNodeList() {
        return new NodeList(continuousAttributes());
    }
}
