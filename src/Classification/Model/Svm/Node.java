package Classification.Model.Svm;

public class Node {
    private int index;
    private double value;

    public Node(int index, double value){
        this.index = index;
        this.value = value;
    }

    public Node clone(){
        return new Node(index, value);
    }

    public int getIndex(){
        return index;
    }

    public double getValue(){
        return value;
    }

}
