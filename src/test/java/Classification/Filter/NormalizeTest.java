package Classification.Filter;

import Classification.Classifier.*;
import Classification.DistanceMetric.EuclidianDistance;
import Classification.Parameter.KMeansParameter;
import Classification.Parameter.KnnParameter;
import Classification.Parameter.LinearPerceptronParameter;
import Classification.Parameter.MultiLayerPerceptronParameter;
import org.junit.Test;

import static org.junit.Assert.*;

public class NormalizeTest extends ClassifierTest {

    @Test
    public void testLinearPerceptron() throws DiscreteFeaturesNotAllowed {
        LinearPerceptron linearPerceptron = new LinearPerceptron();
        LinearPerceptronParameter linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
        Normalize normalize = new Normalize(iris);
        normalize.convert();
        linearPerceptron.train(iris.getInstanceList(), linearPerceptronParameter);
        assertEquals(5.33, 100 * linearPerceptron.test(iris.getInstanceList()).getErrorRate(), 0.01);
        normalize = new Normalize(bupa);
        normalize.convert();
        linearPerceptron.train(bupa.getInstanceList(), linearPerceptronParameter);
        assertEquals(32.17, 100 * linearPerceptron.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        normalize = new Normalize(dermatology);
        normalize.convert();
        linearPerceptron.train(dermatology.getInstanceList(), linearPerceptronParameter);
        assertEquals(1.09, 100 * linearPerceptron.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testMultiLayerPerceptron() throws DiscreteFeaturesNotAllowed {
        MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron();
        MultiLayerPerceptronParameter multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 1, 0.99, 0.2, 100, 3);
        Normalize normalize = new Normalize(iris);
        normalize.convert();
        multiLayerPerceptron.train(iris.getInstanceList(), multiLayerPerceptronParameter);
        assertEquals(3.33, 100 * multiLayerPerceptron.test(iris.getInstanceList()).getErrorRate(), 0.01);
        multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.5, 0.99, 0.2, 100, 30);
        normalize = new Normalize(bupa);
        normalize.convert();
        multiLayerPerceptron.train(bupa.getInstanceList(), multiLayerPerceptronParameter);
        assertEquals(25.22, 100 * multiLayerPerceptron.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.1, 0.99, 0.2, 100, 20);
        normalize = new Normalize(dermatology);
        normalize.convert();
        multiLayerPerceptron.train(dermatology.getInstanceList(), multiLayerPerceptronParameter);
        assertEquals(0.27, 100 * multiLayerPerceptron.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testKnn() {
        Knn knn = new Knn();
        KnnParameter knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
        Normalize normalize = new Normalize(iris);
        normalize.convert();
        knn.train(iris.getInstanceList(), knnParameter);
        assertEquals(4.67, 100 * knn.test(iris.getInstanceList()).getErrorRate(), 0.01);
        normalize = new Normalize(bupa);
        normalize.convert();
        knn.train(bupa.getInstanceList(), knnParameter);
        assertEquals(16.52, 100 * knn.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        normalize = new Normalize(dermatology);
        normalize.convert();
        knn.train(dermatology.getInstanceList(), knnParameter);
        assertEquals(1.91, 100 * knn.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

    @Test
    public void testKMeans() {
        KMeans kMeans = new KMeans();
        KMeansParameter kMeansParameter = new KMeansParameter(1);
        Normalize normalize = new Normalize(iris);
        normalize.convert();
        kMeans.train(iris.getInstanceList(), kMeansParameter);
        assertEquals(14.66, 100 * kMeans.test(iris.getInstanceList()).getErrorRate(), 0.01);
        normalize = new Normalize(bupa);
        normalize.convert();
        kMeans.train(bupa.getInstanceList(), kMeansParameter);
        assertEquals(41.44, 100 * kMeans.test(bupa.getInstanceList()).getErrorRate(), 0.01);
        normalize = new Normalize(dermatology);
        normalize.convert();
        kMeans.train(dermatology.getInstanceList(), kMeansParameter);
        assertEquals(3.55, 100 * kMeans.test(dermatology.getInstanceList()).getErrorRate(), 0.01);
    }

}