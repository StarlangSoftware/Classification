package Classification.Classifier;

import Classification.Model.DecisionTree.DecisionStump;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class C45StumpTest extends ClassifierTest{

    @Test
    public void testTrain() {
        DecisionStump c45Stump = new DecisionStump();
        c45Stump.train(iris.getInstanceList(), null);
        c45Stump.saveTxt("models/c45stump-iris.txt");
        c45Stump.loadModel("models/c45stump-iris.txt");
        assertEquals(33.33, 100 * c45Stump.test(iris.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(bupa.getInstanceList(), null);
        c45Stump.saveTxt("models/c45stump-bupa.txt");
        c45Stump.loadModel("models/c45stump-bupa.txt");
        assertEquals(36.81, 100 * c45Stump.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(dermatology.getInstanceList(), null);
        c45Stump.saveTxt("models/c45stump-dermatology.txt");
        c45Stump.loadModel("models/c45stump-dermatology.txt");
        assertEquals(49.73, 100 * c45Stump.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(car.getInstanceList(), null);
        c45Stump.saveTxt("models/c45stump-car.txt");
        c45Stump.loadModel("models/c45stump-car.txt");
        assertEquals(29.98, 100 * c45Stump.test(car.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(tictactoe.getInstanceList(), null);
        c45Stump.saveTxt("models/c45stump-tictactoe.txt");
        c45Stump.loadModel("models/c45stump-tictactoe.txt");
        assertEquals(30.06, 100 * c45Stump.test(tictactoe.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(nursery.getInstanceList(), null);
        c45Stump.saveTxt("models/c45stump-nursery.txt");
        c45Stump.loadModel("models/c45stump-nursery.txt");
        assertEquals(29.03, 100 * c45Stump.test(nursery.getInstanceList()).getErrorRate(), 0.01);
        c45Stump.train(chess.getInstanceList(), null);
        c45Stump.saveTxt("models/c45stump-chess.txt");
        c45Stump.loadModel("models/c45stump-chess.txt");
        assertEquals(80.92, 100 * c45Stump.test(chess.getInstanceList()).getErrorRate(), 0.01);
    }
}