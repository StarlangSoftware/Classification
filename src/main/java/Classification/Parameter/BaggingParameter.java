package Classification.Parameter;

public class BaggingParameter extends Parameter {

    protected int ensembleSize;

    /**
     * Parameters of the bagging trees algorithm.
     *
     * @param seed         Seed is used for random number generation.
     * @param ensembleSize The number of trees in the bagged forest.
     */
    public BaggingParameter(int seed, int ensembleSize) {
        super(seed);
        this.ensembleSize = ensembleSize;
    }

    /**
     * Accessor for the ensemble size.
     *
     * @return The ensemble size.
     */
    public int getEnsembleSize() {
        return ensembleSize;
    }

}
