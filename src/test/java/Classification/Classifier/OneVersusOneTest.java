package Classification.Classifier;

import Classification.Model.*;
import Classification.Model.DecisionTree.*;
import Classification.Model.Ensemble.OneVersusOne;
import Classification.Parameter.*;
import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.*;

public class OneVersusOneTest extends ClassifierTest {

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        int classLabelSize = iris.getInstanceList().getDistinctClassLabels().size();
        ArrayList<Model> models = new ArrayList<>();
        for (int i = 0; i < (classLabelSize * (classLabelSize - 1)) / 2; i++) {
            models.add(new DecisionTree());
        }
        OneVersusOne oneVersusOne = new OneVersusOne(models);
        C45Parameter c45Parameter = new C45Parameter(1, true, 0.2);
        oneVersusOne.train(iris.getInstanceList(), c45Parameter);
        assertEquals(4.67, 100 * oneVersusOne.test(iris.getInstanceList()).getErrorRate(), 0.01);
        classLabelSize = dermatology.getInstanceList().getDistinctClassLabels().size();
        models.clear();
        for (int i = 0; i < (classLabelSize * (classLabelSize - 1)) / 2; i++) {
            models.add(new DecisionStump());
        }
        oneVersusOne.train(dermatology.getInstanceList(), null);
        assertEquals(3.83, 100 * oneVersusOne.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        models.clear();
        for (int i = 0; i < (classLabelSize * (classLabelSize - 1)) / 2; i++) {
            models.add(new DecisionTree());
        }
        oneVersusOne.train(dermatology.getInstanceList(), c45Parameter);
        assertEquals(1.91, 100 * oneVersusOne.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }
}
