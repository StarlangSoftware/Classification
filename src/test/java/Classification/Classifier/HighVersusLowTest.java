package Classification.Classifier;

import Classification.Model.*;
import Classification.Model.DecisionTree.*;
import Classification.Model.Ensemble.HighVersusLow;
import Classification.Parameter.*;
import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.*;

public class HighVersusLowTest extends ClassifierTest {

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        ArrayList<String> classLabels = new ArrayList<>();
        classLabels.add("unacc");
        classLabels.add("acc");
        classLabels.add("good");
        classLabels.add("vgood");
        ArrayList<Model> models = new ArrayList<>();
        for (int i = 0; i < classLabels.size() - 1; i++) {
            models.add(new DecisionTree());
        }
        HighVersusLow highVersusLow = new HighVersusLow(models, classLabels);
        C45Parameter c45Parameter = new C45Parameter(1, true, 0.2);
        highVersusLow.train(car.getInstanceList(), c45Parameter);
        assertEquals(6.71, 100 * highVersusLow.test(car.getInstanceList()).getErrorRate(), 0.01);
        classLabels.clear();
        classLabels.add("low");
        classLabels.add("mid");
        classLabels.add("high");
        models.clear();
        for (int i = 0; i < classLabels.size() - 1; i++) {
            models.add(new DecisionTree());
        }
        highVersusLow.train(maternal.getInstanceList(), c45Parameter);
        assertEquals(17.06, 100 * highVersusLow.test(maternal.getInstanceList()).getErrorRate(), 0.01);
    }
}
