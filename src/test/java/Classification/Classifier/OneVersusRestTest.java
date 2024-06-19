package Classification.Classifier;

import Classification.Model.*;
import Classification.Model.DecisionTree.*;
import Classification.Parameter.*;
import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.*;

public class OneVersusRestTest extends ClassifierTest {

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        int classLabelSize = iris.getInstanceList().getDistinctClassLabels().size();
        ArrayList<Model> models = new ArrayList<>();
        for (int i = 0; i < classLabelSize; i++) {
            models.add(new DecisionTree());
        }
        OneVersusRest oneVersusRest = new OneVersusRest(models);
        C45Parameter c45Parameter = new C45Parameter(1, true, 0.2);
        oneVersusRest.train(iris.getInstanceList(), c45Parameter);
        assertEquals(4.0, 100 * oneVersusRest.test(iris.getInstanceList()).getErrorRate(), 0.01);
        classLabelSize = dermatology.getInstanceList().getDistinctClassLabels().size();
        models.clear();
        for (int i = 0; i < classLabelSize; i++) {
            models.add(new DecisionStump());
        }
        oneVersusRest.train(dermatology.getInstanceList(), null);
        assertEquals(14.20, 100 * oneVersusRest.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        models.clear();
        for (int i = 0; i < classLabelSize; i++) {
            models.add(new DecisionTree());
        }
        oneVersusRest.train(dermatology.getInstanceList(), c45Parameter);
        assertEquals(2.45, 100 * oneVersusRest.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }
}
