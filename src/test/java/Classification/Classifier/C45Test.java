package Classification.Classifier;

import Classification.Parameter.C45Parameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class C45Test extends ClassifierTest{

    @Test
    public void testTrain() {
        C45 c45 = new C45();
        C45Parameter c45Parameter = new C45Parameter(1, true, 0.2);
        c45.train(iris.getInstanceList(), c45Parameter);
        c45.getModel().saveTxt("models/c45-iris.txt");
        c45.loadModel("models/c45-iris.txt");
        assertEquals(4.00, 100 * c45.test(iris.getInstanceList()).getErrorRate(), 0.01);
        c45.train(bupa.getInstanceList(), c45Parameter);
        c45.getModel().saveTxt("models/c45-bupa.txt");
        c45.loadModel("models/c45-bupa.txt");
        assertEquals(42.03, 100 * c45.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        c45.train(dermatology.getInstanceList(), c45Parameter);
        c45.getModel().saveTxt("models/c45-dermatology.txt");
        c45.loadModel("models/c45-dermatology.txt");
        assertEquals(2.19, 100 * c45.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        c45.train(car.getInstanceList(), c45Parameter);
        c45.getModel().saveTxt("models/c45-car.txt");
        c45.loadModel("models/c45-car.txt");
        assertEquals(8.16, 100 * c45.test(car.getInstanceList()).getErrorRate(), 0.01);
        c45.train(tictactoe.getInstanceList(), c45Parameter);
        c45.getModel().saveTxt("models/c45-tictactoe.txt");
        c45.loadModel("models/c45-tictactoe.txt");
        assertEquals(14.61, 100 * c45.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
    }
}