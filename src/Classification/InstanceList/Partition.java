package Classification.InstanceList;

import Classification.Instance.Instance;

import java.util.ArrayList;

public class Partition {

    private ArrayList<InstanceList> multiList;

    public Partition(){
        multiList = new ArrayList<InstanceList>();
    }

    public void add(InstanceList list){
        multiList.add(list);
    }

    public int size(){
        return multiList.size();
    }

    public InstanceList get(int index){
        return multiList.get(index);
    }

    public ArrayList<Instance>[] getLists(){
        ArrayList<Instance>[] result = new ArrayList[multiList.size()];
        for (int i = 0; i < multiList.size(); i++){
            result[i] = multiList.get(i).getInstances();
        }
        return result;
    }
}
