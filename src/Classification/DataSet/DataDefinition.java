package Classification.DataSet;

import Classification.Attribute.AttributeType;
import Classification.FeatureSelection.FeatureSubSet;

import java.util.ArrayList;

public class DataDefinition {

    private ArrayList<AttributeType> attributeTypes;

    public DataDefinition(){
        attributeTypes = new ArrayList<>();
    }

    public DataDefinition(ArrayList<AttributeType> attributeTypes){
        this.attributeTypes = attributeTypes;
    }

    public int attributeCount(){
        return attributeTypes.size();
    }

    public int discreteAttributeCount(){
        int count = 0;
        for (AttributeType attributeType : attributeTypes){
            if (attributeType.equals(AttributeType.DISCRETE) || attributeType.equals(AttributeType.BINARY)){
                count++;
            }
        }
        return count;
    }

    public int continuousAttributeCount(){
        int count = 0;
        for (AttributeType attributeType : attributeTypes){
            if (attributeType.equals(AttributeType.CONTINUOUS)){
                count++;
            }
        }
        return count;
    }

    public AttributeType getAttributeType(int index){
        return attributeTypes.get(index);
    }

    public void addAttribute(AttributeType attributeType){
        attributeTypes.add(attributeType);
    }

    public void removeAttribute(int index){
        attributeTypes.remove(index);
    }

    public void removeAllAttributes(){
        attributeTypes.clear();
    }

    public DataDefinition getSubSetOfFeatures(FeatureSubSet featureSubSet){
        ArrayList<AttributeType> newAttributeTypes = new ArrayList<>();
        for (int i = 0; i < featureSubSet.size(); i++){
            newAttributeTypes.add(attributeTypes.get(featureSubSet.get(i)));
        }
        return new DataDefinition(newAttributeTypes);
    }

}
