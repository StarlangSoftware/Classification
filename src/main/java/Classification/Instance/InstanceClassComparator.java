package Classification.Instance;

import java.util.Comparator;

public class InstanceClassComparator implements Comparator<Instance> {

    public int compare(Instance o1, Instance o2) {
        return o1.getClassLabel().compareTo(o2.getClassLabel());
    }
}
