package Classification.Model.DecisionTree;

import Classification.Attribute.Attribute;
import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Attribute.DiscreteIndexedAttribute;
import Classification.Instance.Instance;

import java.io.Serializable;

public class DecisionCondition implements Serializable {

    private int attributeIndex = -1;
    private char comparison;
    private Attribute value;

    /**
     * A constructor that sets attributeIndex and {@link Attribute} value. It also assigns equal sign to the comparison character.
     *
     * @param attributeIndex Integer number that shows attribute index.
     * @param value          The value of the {@link Attribute}.
     */
    public DecisionCondition(int attributeIndex, Attribute value) {
        this.attributeIndex = attributeIndex;
        comparison = '=';
        this.value = value;
    }

    /**
     * A constructor that sets attributeIndex, comparison and {@link Attribute} value.
     *
     * @param attributeIndex Integer number that shows attribute index.
     * @param value          The value of the {@link Attribute}.
     * @param comparison     Comparison character.
     */
    public DecisionCondition(int attributeIndex, char comparison, Attribute value) {
        this.attributeIndex = attributeIndex;
        this.comparison = comparison;
        this.value = value;
    }

    public int getAttributeIndex(){
        return attributeIndex;
    }

    public Attribute getValue(){
        return value;
    }

    public char getComparison(){
        return comparison;
    }

    /**
     * The satisfy method takes an {@link Instance} as an input.
     * <p>
     * If defined {@link Attribute} value is a {@link DiscreteIndexedAttribute} it compares the index of {@link Attribute} of instance at the
     * attributeIndex and the index of {@link Attribute} value and returns the result.
     * <p>
     * If defined {@link Attribute} value is a {@link DiscreteAttribute} it compares the value of {@link Attribute} of instance at the
     * attributeIndex and the value of {@link Attribute} value and returns the result.
     * <p>
     * If defined {@link Attribute} value is a {@link ContinuousAttribute} it compares the value of {@link Attribute} of instance at the
     * attributeIndex and the value of {@link Attribute} value and returns the result according to the comparison character whether it is
     * less than or greater than signs.
     *
     * @param instance Instance to compare.
     * @return True if gicen instance satisfies the conditions.
     */
    public boolean satisfy(Instance instance) {
        if (value instanceof DiscreteIndexedAttribute) {
            if (((DiscreteIndexedAttribute) value).getIndex() != -1) {
                return ((DiscreteIndexedAttribute) instance.getAttribute(attributeIndex)).getIndex() == ((DiscreteIndexedAttribute) value).getIndex();
            } else {
                return true;
            }
        } else {
            if (value instanceof DiscreteAttribute) {
                return (((DiscreteAttribute) instance.getAttribute(attributeIndex)).getValue().equalsIgnoreCase(((DiscreteAttribute) value).getValue()));
            } else {
                if (value instanceof ContinuousAttribute) {
                    if (comparison == '<') {
                        return ((ContinuousAttribute) instance.getAttribute(attributeIndex)).getValue() <= ((ContinuousAttribute) value).getValue();
                    } else {
                        if (comparison == '>') {
                            return ((ContinuousAttribute) instance.getAttribute(attributeIndex)).getValue() > ((ContinuousAttribute) value).getValue();
                        }
                    }
                }
            }
        }
        return false;
    }
}
