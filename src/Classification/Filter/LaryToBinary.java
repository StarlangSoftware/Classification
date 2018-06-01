package Classification.Filter;

import Classification.Attribute.AttributeType;
import Classification.Attribute.BinaryAttribute;
import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.Instance.Instance;

public class LaryToBinary extends LaryFilter{

    /**
     * Constructor for L-ary discrete to binary discrete filter.
     * @param dataSet The instances whose L-ary discrete attributes will be converted to binary discrete attributes
     */
    public LaryToBinary(DataSet dataSet){
        super(dataSet);
    }

    /**
     * Converts discrete attributes of a single instance to binary discrete version using 1-of-L encoding. For example, if
     * an attribute has values red, green, blue; this attribute will be converted to 3 binary attributes where
     * red will have the value true false false, green will have the value false true false, and blue will have the value false false true.
     * @param instance The instance to be converted
     */
    public void convertInstance(Instance instance) {
        int size = instance.attributeSize();
        for (int i = 0; i < size; i++){
            if (attributeDistributions.get(i).size() > 0){
                int index = attributeDistributions.get(i).getIndex(instance.getAttribute(i).toString());
                for (int j = 0; j < attributeDistributions.get(i).size(); j++) {
                    if (j != index){
                        instance.addAttribute(new BinaryAttribute(false));
                    } else {
                        instance.addAttribute(new BinaryAttribute(true));
                    }
                }
            }
        }
        removeDiscreteAttributes(instance, size);
    }

    /**
     * Converts the data definition with L-ary discrete attributes, to data definition with binary discrete attributes.
     */
    public void convertDataDefinition() {
        DataDefinition dataDefinition = dataSet.getDataDefinition();
        int size = dataDefinition.attributeCount();
        for (int i = 0; i < size; i++){
            if (attributeDistributions.get(i).size() > 0){
                for (int j = 0; j < attributeDistributions.get(i).size(); j++) {
                    dataDefinition.addAttribute(AttributeType.BINARY);
                }
            }
        }
        removeDiscreteAttributes(size);
    }
}
