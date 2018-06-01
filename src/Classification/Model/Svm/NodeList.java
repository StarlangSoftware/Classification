package Classification.Model.Svm;

import java.util.ArrayList;

public class NodeList {
    private ArrayList<Node> nodes;

    public NodeList(ArrayList<Double> values){
        nodes = new ArrayList<>();
        for (int i = 0; i < values.size(); i++){
            if (values.get(i) != 0){
                nodes.add(new Node(i, values.get(i)));
            }
        }
    }

    public NodeList(){
    }

    public NodeList clone(){
        NodeList result = new NodeList();
        result.nodes = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i++){
            result.nodes.add(nodes.get(i).clone());
        }
        return result;
    }

    public double dot(NodeList nodeList){
        double sum = 0;
        int px = 0, py = 0;
        while (px < nodes.size() && py < nodeList.nodes.size()){
            if (nodes.get(px).getIndex() == nodeList.nodes.get(py).getIndex()){
                sum += nodes.get(px).getValue() * nodeList.nodes.get(py).getValue();
                px++;
                py++;
            } else {
                if (nodes.get(px).getIndex() > nodeList.nodes.get(py).getIndex()){
                    py++;
                } else {
                    px++;
                }
            }
        }
        return sum;
    }

    public Node get(int index){
        return nodes.get(index);
    }

    public int size(){
        return nodes.size();
    }
}
