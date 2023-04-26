package Classification.DataSet;

import Classification.Attribute.AttributeType;
import Classification.FeatureSelection.FeatureSubSet;

import java.util.ArrayList;

public class DataDefinition {

    private ArrayList<AttributeType> attributeTypes;

    private String[][] attributeValueList;

    /**
     * Constructor for creating a new {@link DataDefinition}.
     */
    public DataDefinition() {
        attributeTypes = new ArrayList<>();
    }

    /**
     * Constructor for creating a new {@link DataDefinition} with given attribute types.
     *
     * @param attributeTypes Attribute types of the data definition.
     */
    public DataDefinition(ArrayList<AttributeType> attributeTypes) {
        this.attributeTypes = attributeTypes;
    }

    /**
     * Constructor for creating a new {@link DataDefinition} with given attribute types.
     *
     * @param attributeTypes Attribute types of the data definition.
     * @param attributeValueList Array of array of strings to represent all possible values of discrete features.
     */
    public DataDefinition(ArrayList<AttributeType> attributeTypes, String[][] attributeValueList) {
        this.attributeTypes = attributeTypes;
        this.attributeValueList = attributeValueList;
    }

    public int numberOfValues(int attributeIndex){
        return attributeValueList[attributeIndex].length;
    }

    public int featureValueIndex(int attributeIndex, String value){
        for (int i = 0; i < attributeValueList[attributeIndex].length; i++){
            if (attributeValueList[attributeIndex][i].equals(value)){
                return i;
            }
        }
        return -1;
    }

    /**
     * Returns the number of attribute types.
     *
     * @return Number of attribute types.
     */
    public int attributeCount() {
        return attributeTypes.size();
    }

    /**
     * Counts the occurrences of binary and discrete type attributes.
     *
     * @return Count of binary and discrete type attributes.
     */
    public int discreteAttributeCount() {
        int count = 0;
        for (AttributeType attributeType : attributeTypes) {
            if (attributeType.equals(AttributeType.DISCRETE) || attributeType.equals(AttributeType.BINARY)) {
                count++;
            }
        }
        return count;
    }

    /**
     * Counts the occurrences of binary and continuous type attributes.
     *
     * @return Count of of binary and continuous type attributes.
     */
    public int continuousAttributeCount() {
        int count = 0;
        for (AttributeType attributeType : attributeTypes) {
            if (attributeType.equals(AttributeType.CONTINUOUS)) {
                count++;
            }
        }
        return count;
    }

    /**
     * Returns the attribute type of the corresponding item at given index.
     *
     * @param index Index of the attribute type.
     * @return Attribute type of the corresponding item at given index.
     */
    public AttributeType getAttributeType(int index) {
        return attributeTypes.get(index);
    }

    /**
     * Adds an attribute type to the list of attribute types.
     *
     * @param attributeType Attribute type to add to the list of attribute types.
     */
    public void addAttribute(AttributeType attributeType) {
        attributeTypes.add(attributeType);
    }

    /**
     * Removes the attribute type at given index from the list of attributes.
     *
     * @param index Index to remove attribute type from list.
     */
    public void removeAttribute(int index) {
        attributeTypes.remove(index);
    }

    /**
     * Clears all the attribute types from list.
     */
    public void removeAllAttributes() {
        attributeTypes.clear();
    }

    /**
     * Generates new subset of attribute types by using given feature subset.
     *
     * @param featureSubSet {@link FeatureSubSet} input.
     * @return DataDefinition with new subset of attribute types.
     */
    public DataDefinition getSubSetOfFeatures(FeatureSubSet featureSubSet) {
        ArrayList<AttributeType> newAttributeTypes = new ArrayList<>();
        for (int i = 0; i < featureSubSet.size(); i++) {
            newAttributeTypes.add(attributeTypes.get(featureSubSet.get(i)));
        }
        return new DataDefinition(newAttributeTypes);
    }

}