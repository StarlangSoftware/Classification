package Classification.Model.DecisionTree;

import Classification.Attribute.Attribute;
import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Attribute.DiscreteIndexedAttribute;
import Classification.Instance.Instance;

import java.io.Serializable;

public class DecisionCondition implements Serializable{

    private int attributeIndex = -1;
    private char comparison;
    private Attribute value;

    public DecisionCondition(int attributeIndex, Attribute value){
        this.attributeIndex = attributeIndex;
        comparison = '=';
        this.value = value;
    }

    public DecisionCondition(int attributeIndex, char comparison, Attribute value){
        this.attributeIndex = attributeIndex;
        this.comparison = comparison;
        this.value = value;
    }

    public boolean satisfy(Instance instance){
        if (value instanceof DiscreteIndexedAttribute){
            if (((DiscreteIndexedAttribute) value).getIndex() != -1){
                return ((DiscreteIndexedAttribute)instance.getAttribute(attributeIndex)).getIndex() == ((DiscreteIndexedAttribute) value).getIndex();
            } else {
                return true;
            }
        } else {
            if (value instanceof DiscreteAttribute){
                return (((DiscreteAttribute)instance.getAttribute(attributeIndex)).getValue().equalsIgnoreCase(((DiscreteAttribute) value).getValue()));
            } else {
                if (value instanceof ContinuousAttribute){
                    if (comparison == '<'){
                        return ((ContinuousAttribute)instance.getAttribute(attributeIndex)).getValue() <= ((ContinuousAttribute) value).getValue();
                    } else {
                        if (comparison == '>'){
                            return ((ContinuousAttribute)instance.getAttribute(attributeIndex)).getValue() > ((ContinuousAttribute) value).getValue();
                        }
                    }
                }
            }
        }
        return false;
    }
}
