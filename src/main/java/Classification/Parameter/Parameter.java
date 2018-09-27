package Classification.Parameter;

public class Parameter {
    private int seed;

    /**
     * Constructor of {@link Parameter} class which assigns given seed value to seed.
     *
     * @param seed Seed is used for random number generation.
     */
    public Parameter(int seed) {
        this.seed = seed;
    }

    /**
     * Accessor for the seed.
     *
     * @return The seed.
     */
    public int getSeed() {
        return seed;
    }
}