package Classification.Classifier;

import Classification.Model.DecisionTree.DecisionTree;
import Classification.Parameter.C45Parameter;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class C45Test extends ClassifierTest{

    @Test
    public void testTrain() {
        DecisionTree c45 = new DecisionTree();
        C45Parameter c45Parameter = new C45Parameter(1, true, 0.2);
        c45.train(iris.getInstanceList(), c45Parameter);
        c45.saveTxt("models/c45-iris.txt");
        c45.loadModel("models/c45-iris.txt");
        assertEquals(4.00, 100 * c45.test(iris.getInstanceList()).getErrorRate(), 0.01);
        c45.train(bupa.getInstanceList(), c45Parameter);
        c45.saveTxt("models/c45-bupa.txt");
        c45.loadModel("models/c45-bupa.txt");
        assertEquals(42.03, 100 * c45.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        c45.train(dermatology.getInstanceList(), c45Parameter);
        c45.saveTxt("models/c45-dermatology.txt");
        c45.loadModel("models/c45-dermatology.txt");
        assertEquals(2.19, 100 * c45.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        c45.train(car.getInstanceList(), c45Parameter);
        c45.saveTxt("models/c45-car.txt");
        c45.loadModel("models/c45-car.txt");
        assertEquals(8.16, 100 * c45.test(car.getInstanceList()).getErrorRate(), 0.01);
        c45.train(carIndexed.getInstanceList(), c45Parameter);
        c45.saveTxt("models/c45-carIndexed.txt");
        c45.loadModel("models/c45-carIndexed.txt");
        assertEquals(3.36, 100 * c45.test(carIndexed.getInstanceList()).getErrorRate(), 0.01);
        c45.train(tictactoe.getInstanceList(), c45Parameter);
        c45.saveTxt("models/c45-tictactoe.txt");
        c45.loadModel("models/c45-tictactoe.txt");
        assertEquals(14.61, 100 * c45.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        c45.train(tictactoeIndexed.getInstanceList(), c45Parameter);
        c45.saveTxt("models/c45-tictactoeIndexed.txt");
        c45.loadModel("models/c45-tictactoeIndexed.txt");
        assertEquals(4.49, 100 * c45.test(tictactoeIndexed.getInstanceList()).getErrorRate(), 0.01);
    }
}