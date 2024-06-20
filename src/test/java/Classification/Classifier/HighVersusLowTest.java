package Classification.Classifier;

import Classification.Model.*;
import Classification.Model.DecisionTree.*;
import Classification.Model.Ensemble.HighVersusLow;
import Classification.Parameter.*;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;

import static org.junit.Assert.*;

public class HighVersusLowTest extends ClassifierTest {

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        ArrayList<String> classLabels = iris.getInstanceList().getDistinctClassLabels();
        Collections.sort(classLabels);
        int classLabelSize = classLabels.size();
        ArrayList<Model> models = new ArrayList<>();
        for (int i = 0; i < classLabelSize - 1; i++) {
            models.add(new DecisionTree());
        }
        HighVersusLow highVersusLow = new HighVersusLow(models, classLabels);
        C45Parameter c45Parameter = new C45Parameter(1, true, 0.2);
        highVersusLow.train(iris.getInstanceList(), c45Parameter);
        assertEquals(4.0, 100 * highVersusLow.test(iris.getInstanceList()).getErrorRate(), 0.01);
    }
}
