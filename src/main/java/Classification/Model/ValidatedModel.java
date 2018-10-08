package Classification.Model;

import Classification.InstanceList.InstanceList;
import Classification.Performance.ClassificationPerformance;

import java.io.Serializable;

public abstract class ValidatedModel extends Model implements Serializable {

    /**
     * The testClassifier method takes an {@link InstanceList} as an input and returns an accuracy value as {@link ClassificationPerformance}.
     *
     * @param data {@link InstanceList} to test.
     * @return Accuracy value as {@link ClassificationPerformance}.
     */
    public ClassificationPerformance testClassifier(InstanceList data) {
        double total = data.size();
        int count = 0;
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).getClassLabel().equalsIgnoreCase(predict(data.get(i)))) {
                count++;
            }
        }
        return new ClassificationPerformance(count / total);
    }

}
