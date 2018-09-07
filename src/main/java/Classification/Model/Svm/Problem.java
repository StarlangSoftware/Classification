package Classification.Model.Svm;

import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;

import java.util.HashMap;

public class Problem {
    private int l;
    private double[] y;
    private NodeList[] x;

    public Problem(NodeList[] x, double[] y){
        this.x = x;
        this.y = y;
        this.l = y.length;
    }

    public Problem(InstanceList instanceList){
        Instance instance;
        l = instanceList.size();
        x = new NodeList[l];
        y = new double[l];
        for (int i = 0; i < instanceList.size(); i++){
            instance = instanceList.get(i);
            x[i] = instance.toNodeList();
        }
    }

    public int getL(){
        return l;
    }

    public NodeList[] getX(){
        return x;
    }

    public double[] getY(){
        return y;
    }

}
