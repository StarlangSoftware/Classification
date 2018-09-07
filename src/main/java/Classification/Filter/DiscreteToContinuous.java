package Classification.Filter;

import Classification.Attribute.AttributeType;
import Classification.Attribute.ContinuousAttribute;
import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.Instance.Instance;

public class DiscreteToContinuous extends LaryFilter{

    /**
     * Constructor for discrete to continuous filter.
     * @param dataSet The dataSet whose instances whose discrete attributes will be converted to continuous attributes using
     *                     1-of-L encoding.
     */
    public DiscreteToContinuous(DataSet dataSet){
        super(dataSet);
    }

    /**
     * Converts discrete attributes of a single instance to continuous version using 1-of-L encoding. For example, if
     * an attribute has values red, green, blue; this attribute will be converted to 3 continuous attributes where
     * red will have the value 100, green will have the value 010, and blue will have the value 001.
     * @param instance The instance to be converted
     */
    protected void convertInstance(Instance instance) {
        int size = instance.attributeSize();
        for (int i = 0; i < size; i++){
            if (attributeDistributions.get(i).size() > 0){
                int index = attributeDistributions.get(i).getIndex(instance.getAttribute(i).toString());
                for (int j = 0; j < attributeDistributions.get(i).size(); j++){
                    if (j != index){
                        instance.addAttribute(new ContinuousAttribute(0));
                    } else {
                        instance.addAttribute(new ContinuousAttribute(1));
                    }
                }
            }
        }
        removeDiscreteAttributes(instance, size);
    }

    /**
     * Converts the data definition with discrete attributes, to data definition with continuous attributes. Basically,
     * for each discrete attribute with L possible values, L more continuous attributes will be added.
     */
    protected void convertDataDefinition() {
        DataDefinition dataDefinition = dataSet.getDataDefinition();
        int size = dataDefinition.attributeCount();
        for (int i = 0; i < size; i++){
            if (attributeDistributions.get(i).size() > 0){
                for (int j = 0; j < attributeDistributions.get(i).size(); j++){
                    dataDefinition.addAttribute(AttributeType.CONTINUOUS);
                }
            }
        }
        removeDiscreteAttributes(size);
    }

}
