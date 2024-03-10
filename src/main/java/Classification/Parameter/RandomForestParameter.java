package Classification.Parameter;

public class RandomForestParameter extends BaggingParameter {

    private final int attributeSubsetSize;

    /**
     * Parameters of the random forest classifier.
     *
     * @param seed                Seed is used for random number generation.
     * @param ensembleSize        The number of trees in the bagged forest.
     * @param attributeSubsetSize Integer value for the size of attribute subset.
     */
    public RandomForestParameter(int seed, int ensembleSize, int attributeSubsetSize) {
        super(seed, ensembleSize);
        this.attributeSubsetSize = attributeSubsetSize;
    }

    /**
     * Accessor for the attributeSubsetSize.
     *
     * @return The attributeSubsetSize.
     */
    public int getAttributeSubsetSize() {
        return attributeSubsetSize;
    }

}