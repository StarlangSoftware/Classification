package Classification.FeatureSelection;

import java.util.ArrayList;

public class FeatureSubSet {
    private ArrayList<Integer> indexList;

    public FeatureSubSet(ArrayList<Integer> indexList){
        this.indexList = indexList;
    }

    public FeatureSubSet(int numberOfFeatures){
        indexList = new ArrayList<>();
        for (int i = 0; i < numberOfFeatures; i++){
            indexList.add(i);
        }
    }

    public FeatureSubSet(){
        indexList = new ArrayList<>();
    }

    public FeatureSubSet clone(){
        ArrayList<Integer> newIndexList = new ArrayList<>();
        for (Integer index : indexList){
            newIndexList.add(index);
        }
        return new FeatureSubSet(newIndexList);
    }

    public int size(){
        return indexList.size();
    }

    public int get(int index){
        return indexList.get(index);
    }

    public boolean contains(int featureNo){
        return indexList.contains(featureNo);
    }

    public void add(Integer featureNo){
        indexList.add(featureNo);
    }

    public void remove(int index){
        indexList.remove(index);
    }
}
