package Classification.Filter;

import Classification.Classifier.ClassifierTest;
import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Model.KnnModel;
import Classification.Model.LinearPerceptronModel;
import Classification.Parameter.KnnParameter;
import Classification.Parameter.LinearPerceptronParameter;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PcaTest extends ClassifierTest {

    @Test
    public void testLinearPerceptron() throws DiscreteFeaturesNotAllowed {
        LinearPerceptronModel linearPerceptron = new LinearPerceptronModel();
        LinearPerceptronParameter linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
        Pca pca = new Pca(iris);
        pca.convert();
        linearPerceptron.train(iris.getInstanceList(), linearPerceptronParameter);
        assertEquals(2.00, 100 * linearPerceptron.test(iris.getInstanceList()).getErrorRate(), 0.01);
        linearPerceptronParameter = new LinearPerceptronParameter(1, 0.01, 0.99, 0.2, 100);
        pca = new Pca(bupa);
        pca.convert();
        linearPerceptron.train(bupa.getInstanceList(), linearPerceptronParameter);
        assertEquals(43.19, 100 * linearPerceptron.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        pca = new Pca(dermatology);
        pca.convert();
        linearPerceptron.train(dermatology.getInstanceList(), linearPerceptronParameter);
        assertEquals(1.91, 100 * linearPerceptron.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testKnn() {
        KnnModel knn = new KnnModel();
        KnnParameter knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
        Pca pca = new Pca(iris);
        pca.convert();
        knn.train(iris.getInstanceList(), knnParameter);
        assertEquals(3.33, 100 * knn.test(iris.getInstanceList()).getErrorRate(), 0.01);
        pca = new Pca(bupa);
        pca.convert();
        knn.train(bupa.getInstanceList(), knnParameter);
        assertEquals(19.13, 100 * knn.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        pca = new Pca(dermatology);
        pca.convert();
        knn.train(dermatology.getInstanceList(), knnParameter);
        assertEquals(3.82, 100 * knn.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

}