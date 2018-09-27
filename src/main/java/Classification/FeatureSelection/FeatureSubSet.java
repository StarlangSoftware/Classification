package Classification.FeatureSelection;

import java.util.ArrayList;

public class FeatureSubSet {
    private ArrayList<Integer> indexList;

    /**
     * A constructor that sets the indexList {@link ArrayList}.
     *
     * @param indexList An ArrayList consists of integer indices.
     */
    public FeatureSubSet(ArrayList<Integer> indexList) {
        this.indexList = indexList;
    }

    /**
     * A constructor that takes number of features as input and initializes indexList with these numbers.
     *
     * @param numberOfFeatures Indicates the indices of indexList.
     */
    public FeatureSubSet(int numberOfFeatures) {
        indexList = new ArrayList<>();
        for (int i = 0; i < numberOfFeatures; i++) {
            indexList.add(i);
        }
    }

    /**
     * A constructor that creates a new ArrayList for indexList.
     */
    public FeatureSubSet() {
        indexList = new ArrayList<>();
    }

    /**
     * The clone method creates a new ArrayList with the elements of indexList and returns it as a new FeatureSubSet.
     *
     * @return A new ArrayList with the elements of indexList and returns it as a new FeatureSubSet.
     */
    public FeatureSubSet clone() {
        ArrayList<Integer> newIndexList = new ArrayList<>();
        for (Integer index : indexList) {
            newIndexList.add(index);
        }
        return new FeatureSubSet(newIndexList);
    }

    /**
     * The size method returns the size of the indexList.
     *
     * @return The size of the indexList.
     */
    public int size() {
        return indexList.size();
    }

    /**
     * The get method returns the item of indexList at given index.
     *
     * @param index Index of the indexList to be accessed.
     * @return The item of indexList at given index.
     */
    public int get(int index) {
        return indexList.get(index);
    }

    /**
     * The contains method returns True, if indexList contains given input number and False otherwise.
     *
     * @param featureNo Feature number that will be checked.
     * @return True, if indexList contains given input number.
     */
    public boolean contains(int featureNo) {
        return indexList.contains(featureNo);
    }

    /**
     * The add method adds given Integer to the indexList.
     *
     * @param featureNo Integer that will be added to indexList.
     */
    public void add(Integer featureNo) {
        indexList.add(featureNo);
    }

    /**
     * The remove method removes the item of indexList at the given index.
     *
     * @param index Index of the item that will be removed.
     */
    public void remove(int index) {
        indexList.remove(index);
    }
}
