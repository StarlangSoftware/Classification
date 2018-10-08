package Classification.Model;

import Classification.Classifier.Classifier;
import Classification.DistanceMetric.DistanceMetric;
import Classification.Instance.CompositeInstance;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class KnnModel extends Model implements Serializable {

    private InstanceList data;
    private int k;
    private DistanceMetric distanceMetric;

    /**
     * Constructor that sets the data {@link InstanceList}, k value and the {@link DistanceMetric}.
     *
     * @param data           {@link InstanceList} input.
     * @param k              K value.
     * @param distanceMetric {@link DistanceMetric} input.
     */
    public KnnModel(InstanceList data, int k, DistanceMetric distanceMetric) {
        this.data = data;
        this.k = k;
        this.distanceMetric = distanceMetric;
    }

    /**
     * The predict method takes an {@link Instance} as an input and finds the nearest neighbors of given instance. Then
     * it returns the first possible class label as the predicted class.
     *
     * @param instance {@link Instance} to make prediction.
     * @return The first possible class label as the predicted class.
     */
    public String predict(Instance instance) {
        InstanceList nearestNeighbors = nearestNeighbors(instance);
        String predictedClass;
        if (instance instanceof CompositeInstance && nearestNeighbors.size() == 0) {
            predictedClass = ((CompositeInstance) instance).getPossibleClassLabels().get(0);
        } else {
            predictedClass = Classifier.getMaximum(nearestNeighbors.getClassLabels());
        }
        return predictedClass;
    }

    /**
     * The nearestNeighbors method takes an {@link Instance} as an input. First it gets the possible class labels, then loops
     * through the data {@link InstanceList} and creates new {@link ArrayList} of {@link KnnInstance}s and adds the corresponding data with
     * the distance between data and given instance. After sorting this newly created ArrayList, it loops k times and
     * returns the first k instances as an {@link InstanceList}.
     *
     * @param instance {@link Instance} to find nearest neighbors/
     * @return The first k instances which are nearest to the given instance as an {@link InstanceList}.
     */
    public InstanceList nearestNeighbors(Instance instance) {
        InstanceList result = new InstanceList();
        ArrayList<KnnInstance> instances = new ArrayList<KnnInstance>();
        ArrayList<String> possibleClassLabels = null;
        if (instance instanceof CompositeInstance) {
            possibleClassLabels = ((CompositeInstance) instance).getPossibleClassLabels();
        }
        for (int i = 0; i < data.size(); i++) {
            if (!(instance instanceof CompositeInstance) || possibleClassLabels.contains(data.get(i).getClassLabel())) {
                instances.add(new KnnInstance(data.get(i), distanceMetric.distance(data.get(i), instance)));
            }
        }
        Collections.sort(instances, new KnnInstanceComparator());
        for (int i = 0; i < Math.min(k, instances.size()); i++) {
            result.add(instances.get(i).instance);
        }
        return result;
    }


    private class KnnInstance {
        private double distance;
        private Instance instance;

        /**
         * The constructor that sets the instance and distance value.
         *
         * @param instance {@link Instance} input.
         * @param distance Double distance value.
         */
        private KnnInstance(Instance instance, double distance) {
            this.instance = instance;
            this.distance = distance;
        }

        /**
         * The toString method returns the concatenation of class label of the instance and the distance value.
         *
         * @return The concatenation of class label of the instance and the distance value.
         */
        public String toString() {
            String str = "";
            str += instance.getClassLabel() + " " + distance;
            return str;
        }
    }


    private class KnnInstanceComparator implements Comparator<KnnInstance> {
        /**
         * The compare method takes two {@link KnnInstance}s as inputs and returns -1 if the distance of first instance is
         * less than the distance of second instance, 1 if the distance of first instance is greater than the distance of second instance,
         * and 0 if they are equal to each other.
         *
         * @param instance1 First {@link KnnInstance} to compare.
         * @param instance2 SEcond {@link KnnInstance} to compare.
         * @return -1 if the distance of first instance is less than the distance of second instance,
         * 1 if the distance of first instance is greater than the distance of second instance,
         * 0 if they are equal to each other.
         */
        public int compare(KnnInstance instance1, KnnInstance instance2) {
            if (instance1.distance < instance2.distance) {
                return -1;
            } else {
                if (instance1.distance > instance2.distance) {
                    return 1;
                } else {
                    return 0;
                }
            }
        }
    }
}
