package Classification.Filter;

import Classification.Attribute.AttributeType;
import Classification.Attribute.DiscreteIndexedAttribute;
import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.Instance.Instance;

public class DiscreteToIndexed extends LaryFilter{

    public DiscreteToIndexed(DataSet dataSet){
        super(dataSet);
    }

    protected void convertInstance(Instance instance) {
        int size = instance.attributeSize();
        for (int i = 0; i < size; i++){
            if (attributeDistributions.get(i).size() > 0) {
                int index = attributeDistributions.get(i).getIndex(instance.getAttribute(i).toString());
                instance.addAttribute(new DiscreteIndexedAttribute(instance.getAttribute(i).toString(), index, attributeDistributions.get(i).size()));
            }
        }
        removeDiscreteAttributes(instance, size);
    }

    protected void convertDataDefinition() {
        DataDefinition dataDefinition = dataSet.getDataDefinition();
        int size = dataDefinition.attributeCount();
        for (int i = 0; i < size; i++){
            if (attributeDistributions.get(i).size() > 0) {
                dataDefinition.addAttribute(AttributeType.DISCRETE_INDEXED);
            }
        }
        removeDiscreteAttributes(size);
    }
}
