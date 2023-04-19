package Classification.Classifier;

import org.junit.Test;

import static org.junit.Assert.*;

public class DummyTest extends ClassifierTest{

    @Test
    public void testTrain() {
        Dummy dummy = new Dummy();
        dummy.train(iris.getInstanceList(), null);
        dummy.getModel().saveTxt("models/dummy-iris.txt");
        dummy.loadModel("models/dummy-iris.txt");
        assertEquals(66.67, 100 * dummy.test(iris.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(bupa.getInstanceList(), null);
        dummy.getModel().saveTxt("models/dummy-bupa.txt");
        dummy.loadModel("models/dummy-bupa.txt");
        assertEquals(42.03, 100 * dummy.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(dermatology.getInstanceList(), null);
        dummy.getModel().saveTxt("models/dummy-dermatology.txt");
        dummy.loadModel("models/dummy-dermatology.txt");
        assertEquals(69.40, 100 * dummy.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(car.getInstanceList(), null);
        dummy.getModel().saveTxt("models/dummy-car.txt");
        dummy.loadModel("models/dummy-car.txt");
        assertEquals(29.98, 100 * dummy.test(car.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(tictactoe.getInstanceList(), null);
        dummy.getModel().saveTxt("models/dummy-tictactoe.txt");
        dummy.loadModel("models/dummy-tictactoe.txt");
        assertEquals(34.66, 100 * dummy.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(nursery.getInstanceList(), null);
        dummy.getModel().saveTxt("models/dummy-nursery.txt");
        dummy.loadModel("models/dummy-nursery.txt");
        assertEquals(66.67, 100 * dummy.test(nursery.getInstanceList()).getErrorRate(), 0.01);
        dummy.train(chess.getInstanceList(), null);
        dummy.getModel().saveTxt("models/dummy-chess.txt");
        dummy.loadModel("models/dummy-chess.txt");
        assertEquals(83.77, 100 * dummy.test(chess.getInstanceList()).getErrorRate(), 0.01);
    }

}