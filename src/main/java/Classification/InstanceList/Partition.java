package Classification.InstanceList;

import Classification.Instance.Instance;

import java.util.ArrayList;

public class Partition {

    private ArrayList<InstanceList> multiList;

    /**
     * Constructor for generating a partition.
     */
    public Partition() {
        multiList = new ArrayList<InstanceList>();
    }

    /**
     * Adds given instance list to the list of instance lists.
     *
     * @param list Instance list to add.
     */
    public void add(InstanceList list) {
        multiList.add(list);
    }

    /**
     * Returns the size of the list of instance lists.
     *
     * @return The size of the list of instance lists.
     */
    public int size() {
        return multiList.size();
    }

    /**
     * Returns the corresponding instance list at given index of list of instance lists.
     *
     * @param index Index of the instance list.
     * @return Instance list at given index of list of instance lists.
     */
    public InstanceList get(int index) {
        return multiList.get(index);
    }

    /**
     * Returns the instances of the items at the list of instance lists.
     *
     * @return Instances of the items at the list of instance lists.
     */
    public ArrayList<Instance>[] getLists() {
        ArrayList<Instance>[] result = new ArrayList[multiList.size()];
        for (int i = 0; i < multiList.size(); i++) {
            result[i] = multiList.get(i).getInstances();
        }
        return result;
    }
}
