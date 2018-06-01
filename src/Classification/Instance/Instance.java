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

public class Instance implements Serializable{

    private String classLabel;
    private ArrayList<Attribute> attributes;

    /**
     * Constructor for a single instance. Given the attributes and class label, it generates a new instance.
     * @param classLabel Class label of the instance.
     * @param attributes Attributes of the instance.
     */
    public Instance(String classLabel, ArrayList<Attribute> attributes){
        this.classLabel = classLabel;
        this.attributes = attributes;
    }

    /**
     * Constructor for a single instance. Given the class label, it generates a new instance with 0 attributes.
     * Attributes must be added later with different addAttribute methods.
     * @param classLabel Class label of the instance.
     */
    public Instance(String classLabel){
        this.classLabel = classLabel;
        this.attributes = new ArrayList<Attribute>();
    }

    /**
     * Adds a discrete attribute with the given value.
     * @param value Value of the discrete attribute.
     */
    public void addAttribute(String value){
        attributes.add(new DiscreteAttribute(value));
    }

    /**
     * Adds a continuous attribute with the given value.
     * @param value Value of the continuous attribute.
     */
    public void addAttribute(double value){
        attributes.add(new ContinuousAttribute(value));
    }

    /**
     * Add a new attribute.
     * @param attribute Attribute to be added.
     */
    public void addAttribute(Attribute attribute){
        attributes.add(attribute);
    }

    public void addVectorAttribute(Vector vector){
        for (int i = 0; i < vector.size(); i++){
            attributes.add(new ContinuousAttribute(vector.getValue(i)));
        }
    }

    /**
     * Removes attribute with the given index from the attribute list.
     * @param index Index of the attribute to be removed.
     */
    public void removeAttribute(int index){
        attributes.remove(index);
    }

    /**
     * Removes all attributes from the attribute list.
     */
    public void removeAllAttributes(){
        attributes.clear();
    }

    /**
     * Accessor for a single attribute.
     * @param index Index of the attribute to be accessed.
     * @return Attribute with index 'index'.
     */
    public Attribute getAttribute(int index){
        return attributes.get(index);
    }

    /**
     * Returns the number of attributes in the attribute list.
     * @return Number of attributes in the attribute list.
     */
    public int attributeSize(){
        return attributes.size();
    }

    public int continuousAttributeSize(){
        int size = 0;
        for (Attribute attribute:attributes){
            if (attribute instanceof ContinuousAttribute){
                size++;
            } else {
                if (attribute instanceof DiscreteIndexedAttribute){
                    DiscreteIndexedAttribute discreteIndexedAttribute = (DiscreteIndexedAttribute)attribute;
                    size += discreteIndexedAttribute.getMaxIndex();
                }
            }
        }
        return size;
    }

    public ArrayList<Double> continuousAttributes(){
        ArrayList<Double> result = new ArrayList<Double>();
        for (Attribute attribute:attributes){
            if (attribute instanceof ContinuousAttribute){
                result.add(((ContinuousAttribute) attribute).getValue());
            } else {
                if (attribute instanceof DiscreteIndexedAttribute){
                    DiscreteIndexedAttribute discreteIndexedAttribute = (DiscreteIndexedAttribute)attribute;
                    for (int i = 0; i < discreteIndexedAttribute.getMaxIndex(); i++){
                        if (i != discreteIndexedAttribute.getIndex()){
                            result.add(0.0);
                        } else {
                            result.add(1.0);
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Accessor for the class label.
     * @return Class label of the instance.
     */
    public String getClassLabel(){
        return classLabel;
    }

    /**
     * Converts instance to a string.
     * @return A string of attributes separated with tab character.
     */
    public String toString(){
        String result = "";
        for (Attribute attribute:attributes){
            result = result + attribute.toString() + "\t";
        }
        result = result + classLabel;
        return result;
    }

    public Instance getSubSetOfFeatures(FeatureSubSet featureSubSet){
        Instance result = new Instance(classLabel);
        for (int i = 0; i < featureSubSet.size(); i++){
            result.addAttribute(attributes.get(featureSubSet.get(i)));
        }
        return result;
    }

    public Vector toVector(){
        ArrayList<Double> values = new ArrayList<Double>();
        for (int i = 0; i < attributeSize(); i++){
            if (getAttribute(i) instanceof ContinuousAttribute){
                values.add(((ContinuousAttribute) getAttribute(i)).getValue());
            } else {
                if (getAttribute(i) instanceof DiscreteIndexedAttribute){
                    DiscreteIndexedAttribute discreteIndexedAttribute = (DiscreteIndexedAttribute)getAttribute(i);
                    for (int j = 0; j < discreteIndexedAttribute.getMaxIndex(); j++){
                        if (j == discreteIndexedAttribute.getIndex()){
                            values.add(1.0);
                        } else {
                            values.add(0.0);
                        }
                    }
                }
            }
        }
        return new Vector(values);
    }

    public NodeList toNodeList(){
        return new NodeList(continuousAttributes());
    }
}
