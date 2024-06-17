package Classification.Classifier;

import Classification.Model.BaggingModel;
import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Parameter.BaggingParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class BaggingTest extends ClassifierTest{

    @Test
    public void testTrain() throws DiscreteFeaturesNotAllowed {
        BaggingModel bagging = new BaggingModel();
        BaggingParameter baggingParameter = new BaggingParameter(1, 100);
        bagging.train(iris.getInstanceList(), baggingParameter);
        bagging.saveTxt("models/bagging-iris.txt");
        bagging.loadModel("models/bagging-iris.txt");
        assertEquals(0.0, 100 * bagging.test(iris.getInstanceList()).getErrorRate(), 0.01);
        bagging.train(bupa.getInstanceList(), baggingParameter);
        bagging.saveTxt("models/bagging-bupa.txt");
        bagging.loadModel("models/bagging-bupa.txt");
        assertEquals(0.0, 100 * bagging.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        bagging.train(dermatology.getInstanceList(), baggingParameter);
        bagging.saveTxt("models/bagging-dermatology.txt");
        bagging.loadModel("models/bagging-dermatology.txt");
        assertEquals(0.0, 100 * bagging.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        bagging.train(car.getInstanceList(), baggingParameter);
        bagging.saveTxt("models/bagging-car.txt");
        bagging.loadModel("models/bagging-car.txt");
        assertEquals(0.0, 100 * bagging.test(car.getInstanceList()).getErrorRate(), 0.01);
        bagging.train(tictactoe.getInstanceList(), baggingParameter);
        bagging.saveTxt("models/bagging-tictactoe.txt");
        bagging.loadModel("models/bagging-tictactoe.txt");
        assertEquals(0.0, 100 * bagging.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }

}