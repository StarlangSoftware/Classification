package Classification.Parameter;

public class RandomForestParameter extends BaggingParameter{

    private int attributeSubsetSize;

    public RandomForestParameter(int seed, int ensembleSize, int attributeSubsetSize){
        super(seed, ensembleSize);
        this.attributeSubsetSize = attributeSubsetSize;
    }

    public int getAttributeSubsetSize(){
        return attributeSubsetSize;
    }

}
