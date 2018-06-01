package Classification.Filter;

import java.util.ArrayList;

import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.Instance.Instance;
import Math.DiscreteDistribution;

public abstract class LaryFilter extends FeatureFilter{
    protected ArrayList<DiscreteDistribution> attributeDistributions;

    public LaryFilter(DataSet dataSet){
        super(dataSet);
        attributeDistributions = dataSet.getInstanceList().allAttributesDistribution();
    }

    protected void removeDiscreteAttributes(Instance instance, int size){
        int k = 0;
        for (int i = 0; i < size; i++){
            if (attributeDistributions.get(i).size() > 0){
                instance.removeAttribute(k);
            } else {
                k++;
            }
        }
    }

    protected void removeDiscreteAttributes(int size){
        DataDefinition dataDefinition = dataSet.getDataDefinition();
        int k = 0;
        for (int i = 0; i < size; i++){
            if (attributeDistributions.get(i).size() > 0){
                dataDefinition.removeAttribute(k);
            } else {
                k++;
            }
        }
    }

}
